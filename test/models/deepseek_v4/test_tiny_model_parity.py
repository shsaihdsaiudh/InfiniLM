"""DeepSeek-V4 tiny-config end-to-end parity test.

Builds the five-layer tiny model that covers every layer type (sliding, CSA,
HCA attention plus hash-MoE and standard MoE), runs the Transformers reference
and the native InfiniLM engine on the same weights, and compares prefill and
teacher-forced decode logits/hidden states.

Reproduce:
    python -m pytest -q test/models/deepseek_v4/test_tiny_model_parity.py

Requires a CUDA GPU, the compiled ``_infinilm`` extension, and a Transformers
version that ships the DeepSeek-V4 reference implementation.
"""

import json
import unittest

import torch

try:
    from transformers.models.deepseek_v4 import (
        DeepseekV4Config,
        DeepseekV4ForCausalLM,
    )

    HAS_TRANSFORMERS_REF = True
except Exception:
    HAS_TRANSFORMERS_REF = False

try:
    import infinicore
    from infinicore.lib import _infinicore
    from infinilm.distributed import DistConfig
    from infinilm.lib import _infinilm

    HAS_INFINILM = True
except Exception:
    HAS_INFINILM = False


TINY_CONFIG = dict(
    vocab_size=1024,
    hidden_size=256,
    num_hidden_layers=5,
    layer_types=[
        "sliding_attention",
        "compressed_sparse_attention",
        "heavily_compressed_attention",
        "compressed_sparse_attention",
        "sliding_attention",
    ],
    mlp_layer_types=["hash_moe", "hash_moe", "moe", "moe", "moe"],
    num_attention_heads=8,
    num_key_value_heads=1,
    head_dim=64,
    q_lora_rank=32,
    o_lora_rank=16,
    o_groups=2,
    index_n_heads=4,
    index_head_dim=16,
    index_topk=4,
    compress_rates={
        "compressed_sparse_attention": 2,
        "heavily_compressed_attention": 4,
    },
    sliding_window=8,
    n_routed_experts=8,
    num_experts_per_tok=2,
    n_shared_experts=1,
    moe_intermediate_size=64,
    hc_mult=4,
    hc_sinkhorn_iters=5,
    max_position_embeddings=512,
    num_nextn_predict_layers=0,
)

DECODE_STEPS = 4
ATOL = 1e-2


def build_reference(decode_steps=DECODE_STEPS):
    torch.manual_seed(0)
    config = DeepseekV4Config(**TINY_CONFIG)
    model = DeepseekV4ForCausalLM(config).eval().to(torch.float32)

    # The tid2eid lookup is frozen in real weights; a zero-initialized table
    # would route every token to expert 0 and leave the multi-expert path
    # uncovered, so fill it with a deterministic pattern.
    with torch.no_grad():
        for name, buf in model.named_buffers():
            if "tid2eid" in name:
                buf.copy_(
                    torch.arange(buf.numel(), dtype=torch.long).view(buf.shape)
                    % config.num_local_experts
                )

    input_ids = (torch.arange(16) * 37 % config.vocab_size).unsqueeze(0)
    with torch.no_grad():
        output = model(input_ids, use_cache=True, output_hidden_states=True)
        decode_ids = []
        decode_pairs = []
        token = output.logits[:, -1].argmax(-1, keepdim=True)
        cache = output.past_key_values
        for _ in range(decode_steps):
            decode_ids.append(token.clone())
            step = model(
                token,
                past_key_values=cache,
                use_cache=True,
                output_hidden_states=True,
            )
            cache = step.past_key_values
            decode_pairs.append(
                (step.logits[:, -1].cpu(), step.hidden_states[-1].cpu())
            )
            token = step.logits[:, -1].argmax(-1, keepdim=True)
    return model, input_ids, output, decode_ids, decode_pairs


def run_native(config_dict, state_dict, input_ids, decode_ids):
    native_config = dict(config_dict)
    native_config.update(
        model_type="deepseek_v4",
        dtype="float32",
        torch_dtype="float32",
        rms_norm_eps=1e-6,
        qk_rope_head_dim=8,
        rope_theta=10000.0,
        compress_rope_theta=160000.0,
        hc_eps=1e-6,
        expert_dtype="dense",
    )
    engine = _infinilm.InferEngine(
        json.dumps(native_config),
        DistConfig(1)._underlying,
        _infinicore.Device.Type.NVIDIA,
        None,
        False,
        "default",
        None,
        False,
        "async",
        False,
    )
    expected = set(engine.state_dict_keyname())
    actual = set(state_dict)
    if expected != actual:
        raise RuntimeError(
            f"native state dict mismatch: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )
    params = {
        name: infinicore.from_torch(tensor.contiguous())._underlying
        for name, tensor in state_dict.items()
    }
    engine.load_params(params, True)

    # ``infinicore.from_torch`` is a zero-copy view.  Keep the owning Torch
    # tensors alive until the native forward has completed.
    ids_torch = input_ids.cuda()
    positions_torch = torch.arange(
        input_ids.shape[1], dtype=torch.int64, device="cuda"
    ).unsqueeze(0)
    offsets_torch = torch.tensor([0, input_ids.shape[1]], dtype=torch.int32)
    native_input = engine.Input(
        infinicore.from_torch(ids_torch)._underlying,
        infinicore.from_torch(positions_torch)._underlying,
        input_offsets=infinicore.from_torch(offsets_torch)._underlying,
        sample_all_positions=True,
    )
    output = engine.forward(native_input)
    logits = torch.from_numpy(infinicore.Tensor(output.logits).to_numpy())
    hidden = torch.from_numpy(infinicore.Tensor(output.hidden_states).to_numpy())

    decode_pairs = []
    first_decode_position = input_ids.shape[1]
    for step, token in enumerate(decode_ids):
        decode_ids_torch = token.cuda()
        decode_positions_torch = torch.tensor(
            [[first_decode_position + step]],
            dtype=torch.int64,
            device="cuda",
        )
        decode_offsets_torch = torch.tensor([0, 1], dtype=torch.int32)
        decode_input = engine.Input(
            infinicore.from_torch(decode_ids_torch)._underlying,
            infinicore.from_torch(decode_positions_torch)._underlying,
            input_offsets=infinicore.from_torch(decode_offsets_torch)._underlying,
            sample_all_positions=True,
        )
        decode_output = engine.forward(decode_input)
        step_logits = torch.from_numpy(
            infinicore.Tensor(decode_output.logits).to_numpy()
        )[:, -1]
        step_hidden = torch.from_numpy(
            infinicore.Tensor(decode_output.hidden_states).to_numpy()
        )[:, -1]
        decode_pairs.append((step_logits, step_hidden))
    return logits, hidden, decode_pairs, expected


@unittest.skipUnless(
    HAS_TRANSFORMERS_REF, "Transformers DeepSeek-V4 reference unavailable"
)
@unittest.skipUnless(HAS_INFINILM, "InfiniLM C++ extension unavailable")
@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA GPU")
class DeepseekV4TinyParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.input_ids, cls.reference, decode_ids, cls.reference_decode = (
            build_reference()
        )
        cls.state_dict = cls.model.state_dict()
        # The reference runs on CPU, so the device-memory delta is attributable
        # to the native engine (weights + activations + runtime state).
        free_before, _ = torch.cuda.mem_get_info(0)
        cls.logits, cls.hidden, cls.native_decode, cls.expected_keys = run_native(
            TINY_CONFIG, cls.state_dict, cls.input_ids, decode_ids
        )
        free_after, _ = torch.cuda.mem_get_info(0)
        cls.device_memory_gib = (free_before - free_after) / 2**30

    def test_state_dict_keys_match(self):
        self.assertEqual(set(self.state_dict), self.expected_keys)

    def test_prefill_parity(self):
        logits_diff = (self.logits - self.reference.logits.cpu()).abs().max().item()
        hidden_diff = (
            (self.hidden - self.reference.hidden_states[-1].cpu()).abs().max().item()
        )
        print(
            f"\nprefill: logits max|diff|={logits_diff:.6e}; "
            f"hidden max|diff|={hidden_diff:.6e}; "
            f"device memory delta={self.device_memory_gib:.3f} GiB"
        )
        self.assertLessEqual(logits_diff, ATOL, "prefill logits parity")
        self.assertLessEqual(hidden_diff, ATOL, "prefill hidden-state parity")

    def test_decode_parity(self):
        self.assertEqual(len(self.native_decode), len(self.reference_decode))
        for step, (actual_pair, expected_pair) in enumerate(
            zip(self.native_decode, self.reference_decode, strict=True)
        ):
            logits_diff = (actual_pair[0] - expected_pair[0]).abs().max().item()
            hidden_diff = (actual_pair[1] - expected_pair[1]).abs().max().item()
            print(
                f"decode step {step + 1}: logits max|diff|={logits_diff:.6e}; "
                f"hidden max|diff|={hidden_diff:.6e}"
            )
            self.assertLessEqual(
                logits_diff, ATOL, f"decode step {step + 1} logits parity"
            )
            self.assertLessEqual(
                hidden_diff, ATOL, f"decode step {step + 1} hidden-state parity"
            )


if __name__ == "__main__":
    unittest.main()

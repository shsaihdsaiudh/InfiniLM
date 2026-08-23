#!/usr/bin/env python3
"""Compare the native DeepSeek-V4 model with the Transformers tiny reference.

The complete five-layer reference is checked by default.  ``--layers`` can be
used to compare selected prefixes when isolating a numerical mismatch.
"""

import argparse
import json
from pathlib import Path

import torch
from transformers.models.deepseek_v4 import DeepseekV4Config, DeepseekV4ForCausalLM

import infinicore
from infinicore.lib import _infinicore
from infinilm.distributed import DistConfig
from infinilm.lib import _infinilm


ROOT = Path(__file__).resolve().parent
BASELINE = ROOT / "baseline_tiny.pt"


def prefix_state_dict(state_dict, num_layers):
    result = {}
    for name, tensor in state_dict.items():
        if not name.startswith("model.layers."):
            result[name] = tensor
            continue
        layer = int(name.split(".", 3)[2])
        if layer < num_layers:
            result[name] = tensor
    return result


def reference_forward(
    saved,
    num_layers,
    decode_steps=0,
    zero_sublayers=False,
    zero_hyper=False,
):
    config_dict = dict(saved["config"])
    config_dict["num_hidden_layers"] = num_layers
    config_dict["layer_types"] = config_dict["layer_types"][:num_layers]
    config_dict["mlp_layer_types"] = config_dict["mlp_layer_types"][:num_layers]
    config = DeepseekV4Config(**config_dict)
    model = DeepseekV4ForCausalLM(config).eval().to(torch.float32)
    state_dict = prefix_state_dict(saved["state_dict"], num_layers)
    if zero_sublayers:
        state_dict = {
            name: (
                torch.zeros_like(tensor)
                if (
                    ".self_attn." in name
                    or ".mlp." in name
                    or (
                        zero_hyper
                        and (
                            ".attn_hc." in name
                            or ".ffn_hc." in name
                            or ".hc_head." in name
                        )
                    )
                )
                else tensor
            )
            for name, tensor in state_dict.items()
        }
    model.load_state_dict(state_dict, strict=True)
    with torch.no_grad():
        output = model(
            saved["input_ids"],
            use_cache=decode_steps > 0,
            output_hidden_states=True,
        )
        decode_ids = []
        decode_logits = []
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
            decode_logits.append(
                (step.logits[:, -1].cpu(), step.hidden_states[-1].cpu())
            )
            token = step.logits[:, -1].argmax(-1, keepdim=True)
    return config_dict, state_dict, output, decode_ids, decode_logits


def native_forward(config_dict, state_dict, input_ids, decode_ids):
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
    offsets_torch = torch.tensor(
        [0, input_ids.shape[1]], dtype=torch.int32
    )
    ids = infinicore.from_torch(ids_torch)._underlying
    positions = infinicore.from_torch(positions_torch)._underlying
    offsets = infinicore.from_torch(offsets_torch)._underlying
    native_input = engine.Input(
        ids,
        positions,
        input_offsets=offsets,
        sample_all_positions=True,
    )
    output = engine.forward(native_input)
    logits = torch.from_numpy(infinicore.Tensor(output.logits).to_numpy())
    hidden = torch.from_numpy(infinicore.Tensor(output.hidden_states).to_numpy())
    decode_logits = []
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
            input_offsets=infinicore.from_torch(
                decode_offsets_torch
            )._underlying,
            sample_all_positions=True,
        )
        decode_output = engine.forward(decode_input)
        step_logits = torch.from_numpy(
            infinicore.Tensor(decode_output.logits).to_numpy()
        )[:, -1]
        step_hidden = torch.from_numpy(
            infinicore.Tensor(decode_output.hidden_states).to_numpy()
        )[:, -1]
        decode_logits.append((step_logits, step_hidden))
    return logits, hidden, decode_logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="layer-prefix lengths to compare (default: complete model)",
    )
    parser.add_argument("--decode-steps", type=int, default=4)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument(
        "--yarn",
        action="store_true",
        help="enable the released checkpoint's compress-layer YaRN settings",
    )
    parser.add_argument(
        "--zero-sublayers",
        action="store_true",
        help="zero attention and MoE weights to isolate the residual/HC path",
    )
    parser.add_argument(
        "--zero-hyper",
        action="store_true",
        help="also zero mHC and hyper-head parameters",
    )
    args = parser.parse_args()

    saved = torch.load(BASELINE, map_location="cpu", weights_only=False)
    if args.yarn:
        saved["config"] = dict(saved["config"])
        saved["config"]["max_position_embeddings"] = 1048576
        saved["config"]["rope_scaling"] = {
            "type": "yarn",
            "factor": 16,
            "beta_fast": 32,
            "beta_slow": 1,
            "original_max_position_embeddings": 65536,
        }
    layer_counts = args.layers or [saved["config"]["num_hidden_layers"]]
    for num_layers in layer_counts:
        (
            config,
            state_dict,
            reference,
            decode_ids,
            reference_decode,
        ) = reference_forward(
            saved,
            num_layers,
            args.decode_steps,
            args.zero_sublayers,
            args.zero_hyper,
        )
        logits, hidden, native_decode = native_forward(
            config, state_dict, saved["input_ids"], decode_ids
        )
        logits_diff = (logits - reference.logits.cpu()).abs()
        hidden_diff = (hidden - reference.hidden_states[-1].cpu()).abs()
        print(
            f"layers={num_layers}: "
            f"logits max={logits_diff.max().item():.6e} "
            f"mean={logits_diff.mean().item():.6e}; "
            f"hidden max={hidden_diff.max().item():.6e} "
            f"mean={hidden_diff.mean().item():.6e}; "
            f"native hidden mean/std={hidden.mean().item():+.6e}/"
            f"{hidden.std().item():.6e}; "
            f"reference={reference.hidden_states[-1].mean().item():+.6e}/"
            f"{reference.hidden_states[-1].std().item():.6e}"
        )
        worst = max(logits_diff.max().item(), hidden_diff.max().item())
        for step, (actual_pair, expected_pair) in enumerate(
            zip(native_decode, reference_decode, strict=True)
        ):
            actual, actual_hidden = actual_pair
            expected, expected_hidden = expected_pair
            difference = (actual - expected).abs()
            hidden_difference = (actual_hidden - expected_hidden).abs()
            step_max = difference.max().item()
            hidden_step_max = hidden_difference.max().item()
            worst = max(worst, step_max, hidden_step_max)
            print(
                f"  decode={step + 1}: logits max={step_max:.6e} "
                f"mean={difference.mean().item():.6e}; "
                f"hidden max={hidden_step_max:.6e} "
                f"mean={hidden_difference.mean().item():.6e}"
            )
        if worst > args.atol:
            raise RuntimeError(
                f"layers={num_layers} exceeded tolerance: "
                f"{worst:.6e} > {args.atol:.6e}"
            )


if __name__ == "__main__":
    main()

import os
import tempfile
import unittest

import torch
from infinilm.modeling_utils import _remap_deepseek_v4, load_state_dict
from safetensors.torch import save_file


@unittest.skipUnless(
    hasattr(torch, "float8_e8m0fnu"),
    "DeepSeek-V4 checkpoint scales require PyTorch float8_e8m0fnu support",
)
class DeepSeekV4CheckpointLoadingTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.checkpoint_path = os.path.join(
            self.temp_dir.name, "model-00001-of-00001.safetensors"
        )

        self.scale = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float32).to(
            torch.float8_e8m0fnu
        )
        self.state_dict = {
            "layers.0.attn.wq_a.weight": torch.tensor(
                [1.0, -2.0, 0.5], dtype=torch.float8_e4m3fn
            ),
            "layers.0.attn.wq_a.scale": self.scale,
            "layers.0.ffn.experts.0.w1.weight": torch.tensor(
                [-16, 1, 127], dtype=torch.int8
            ),
            "layers.3.ffn.gate.bias": torch.tensor(
                [0.25, -0.5], dtype=torch.float32
            ),
        }
        save_file(self.state_dict, self.checkpoint_path, metadata={"format": "pt"})

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_preserves_v4_scale_bit_pattern_until_remap(self):
        loaded = load_state_dict(
            self.checkpoint_path,
            dtype=torch.bfloat16,
            preserve_fp32_suffixes=(".bias",),
            preserve_dtype_suffixes=(".scale",),
        )

        self.assertEqual(loaded["layers.0.attn.wq_a.weight"].dtype, torch.bfloat16)
        self.assertEqual(loaded["layers.0.attn.wq_a.scale"].dtype, self.scale.dtype)
        self.assertTrue(
            torch.equal(
                loaded["layers.0.attn.wq_a.scale"].view(torch.uint8),
                self.scale.view(torch.uint8),
            )
        )
        self.assertEqual(
            loaded["layers.0.ffn.experts.0.w1.weight"].dtype, torch.int8
        )
        self.assertEqual(loaded["layers.3.ffn.gate.bias"].dtype, torch.float32)

    def test_default_path_still_casts_float8_scales(self):
        loaded = load_state_dict(self.checkpoint_path, dtype=torch.bfloat16)

        self.assertEqual(loaded["layers.0.attn.wq_a.scale"].dtype, torch.bfloat16)

    def test_remaps_dense_and_packed_expert_weights(self):
        dense_weight = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
            dtype=torch.bfloat16,
        )
        dense_scale = torch.tensor([[1.0, 2.0]], dtype=torch.float32).to(
            torch.float8_e8m0fnu
        )
        expert_weight = torch.tensor([[-16, 1], [2, 127]], dtype=torch.int8)
        expert_scale = torch.tensor([[1.0], [2.0]], dtype=torch.float32).to(
            torch.float8_e8m0fnu
        )
        state_dict = {
            "layers.0.attn.wq_a.weight": dense_weight,
            "layers.0.attn.wq_a.scale": dense_scale,
            "layers.0.ffn.experts.3.w1.weight": expert_weight,
            "layers.0.ffn.experts.3.w1.scale": expert_scale,
            "layers.0.attn.attn_sink": torch.arange(2, dtype=torch.float32),
            "layers.0.hc_attn_base": torch.arange(4, dtype=torch.float32),
            "mtp.layers.0.weight": torch.ones(1),
        }
        config = {
            "torch_dtype": "bfloat16",
            "quantization_config": {"weight_block_size": [2, 2]},
        }

        remapped = _remap_deepseek_v4(state_dict, config)

        dense_key = "model.layers.0.self_attn.q_a_proj.weight"
        expected_dense = torch.tensor(
            [[1.0, 2.0, 6.0, 8.0], [5.0, 6.0, 14.0, 16.0]],
            dtype=torch.bfloat16,
        )
        self.assertTrue(torch.equal(remapped[dense_key], expected_dense))

        expert_prefix = "model.layers.0.mlp.experts.3.w1"
        self.assertTrue(
            torch.equal(
                remapped[expert_prefix + ".weight_packed"],
                expert_weight.view(torch.uint8),
            )
        )
        self.assertTrue(
            torch.equal(
                remapped[expert_prefix + ".weight_scale"],
                expert_scale.view(torch.uint8),
            )
        )
        self.assertIn("model.layers.0.self_attn.sinks", remapped)
        self.assertIn("model.layers.0.attn_hc.base", remapped)
        self.assertFalse(any(key.startswith("mtp.") for key in remapped))


if __name__ == "__main__":
    unittest.main()

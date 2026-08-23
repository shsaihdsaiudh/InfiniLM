# 定位增量 decode 与全量前向不一致的层型
import torch

from transformers.models.deepseek_v4 import DeepseekV4Config, DeepseekV4ForCausalLM
from ref_baseline import tiny

def max_diff(layer_types):
    torch.manual_seed(0)
    cfg = dict(tiny)
    cfg["layer_types"] = layer_types
    config = DeepseekV4Config(**cfg)
    model = DeepseekV4ForCausalLM(config).eval().to(torch.float32)
    with torch.no_grad():
        for name, buf in model.named_buffers():
            if "tid2eid" in name:
                buf.copy_(torch.arange(buf.numel(), dtype=torch.long).view(buf.shape) % config.num_local_experts)
        input_ids = (torch.arange(16) * 37 % config.vocab_size).unsqueeze(0)
        out = model(input_ids, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        step = model(nxt, past_key_values=past, use_cache=True)
        full = model(torch.cat([input_ids, nxt], dim=1), use_cache=False)
        return (full.logits[:, -1] - step.logits[:, -1]).abs().max().item()

cases = {
    "all sliding": ["sliding_attention"] * 5,
    "all CSA (m=2)": ["compressed_sparse_attention"] * 5,
    "all HCA (m=4)": ["heavily_compressed_attention"] * 5,
    "mixed (baseline)": tiny["layer_types"],
}
for name, lt in cases.items():
    print(f"{name:22s} max|diff| = {max_diff(lt):.3e}")

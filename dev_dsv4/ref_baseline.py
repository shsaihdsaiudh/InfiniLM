# DeepSeek-V4 tiny-config PyTorch 参照基线生成器
# 用途：构造覆盖全部层型的 tiny DeepseekV4ForCausalLM，产出逐层数值基线，
#       作为 InfiniLM C++ 实现对拍的黄金参照。
# 运行：python dev_dsv4/ref_baseline.py
import json
import os

import torch

from transformers.models.deepseek_v4 import DeepseekV4Config, DeepseekV4ForCausalLM

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# 5 层，覆盖全部 3 种注意力层型 + 2 种 MLP 层型
tiny = dict(
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
    compress_rates={"compressed_sparse_attention": 2, "heavily_compressed_attention": 4},
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

def main():
    torch.manual_seed(0)
    config = DeepseekV4Config(**tiny)
    model = DeepseekV4ForCausalLM(config).eval().to(torch.float32)

    # hash_moe 的 tid2eid 在真实权重里是冻结查找表；随机初始化为全 0 会让所有 token
    # 都挤到 expert 0，覆盖不到多专家路径，这里填确定性模式
    with torch.no_grad():
        for name, buf in model.named_buffers():
            if "tid2eid" in name:
                buf.copy_(
                    torch.arange(buf.numel(), dtype=torch.long).view(buf.shape)
                    % config.num_local_experts
                )

    input_ids = (torch.arange(16) * 37 % config.vocab_size).unsqueeze(0)  # [1, 16]

    # 1) prefill 全量前向
    with torch.no_grad():
        out = model(
            input_ids,
            use_cache=True,
            output_hidden_states=True,
        )
    logits = out.logits  # [1, 16, vocab]

    # 2) 自回归 decode 4 步（走 HCACache/CSACache 增量路径）
    past = out.past_key_values
    next_ids = logits[:, -1].argmax(-1, keepdim=True)
    decode_logits = []
    ids = next_ids
    for _ in range(4):
        with torch.no_grad():
            step = model(ids, past_key_values=past, use_cache=True)
        past = step.past_key_values
        decode_logits.append(step.logits[:, -1])
        ids = step.logits[:, -1].argmax(-1, keepdim=True)

    # 3) 一致性自检：增量 decode 的 logits 应与同长度全量前向一致
    full_ids = torch.cat([input_ids, next_ids], dim=1)
    with torch.no_grad():
        full = model(full_ids, use_cache=False)
    diff = (full.logits[:, -1] - decode_logits[0]).abs().max().item()
    print(f"[self-check] prefill+1 decode vs full forward, max|diff| = {diff:.3e}")

    # 4) 保存基线
    torch.save(
        {
            "config": tiny,
            "input_ids": input_ids,
            "hidden_states": [h.clone() for h in out.hidden_states],
            "logits": logits,
            "decode_logits": decode_logits,
            "state_dict": {k: v.clone() for k, v in model.state_dict().items()},
        },
        os.path.join(OUT_DIR, "baseline_tiny.pt"),
    )

    # 5) 打印逐层校验和 + 权重清单
    print("\n[hidden-state checksums] (mean, std)")
    for i, h in enumerate(out.hidden_states):
        print(f"  layer {i:2d} ({tuple(h.shape)}): {h.mean():+.6e}, {h.std():.6e}")
    print(f"\n[logits] ({tuple(logits.shape)}): {logits.mean():+.6e}, {logits.std():.6e}")

    inv = {k: list(v.shape) for k, v in model.state_dict().items()}
    with open(os.path.join(OUT_DIR, "weight_inventory.json"), "w") as f:
        json.dump(inv, f, indent=2)
    print(f"\n[weight inventory] {len(inv)} tensors -> weight_inventory.json")
    n_param = sum(v.numel() for v in model.state_dict().values())
    print(f"[params] {n_param/1e6:.2f} M")

if __name__ == "__main__":
    main()

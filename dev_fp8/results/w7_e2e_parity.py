#!/usr/bin/env python3
"""W7 split-kv E2E 贪婪对拍:InferEngine + paged attn,KV fp8 vs 默认 bf16,同一长 prompt。

远端用法(source fp8_env.sh 后):
    INFINIOP_FLASH_DEBUG_SPLITS=1 python3 dev_fp8/results/w7_e2e_parity.py
"""

import sys

sys.path.insert(0, "/root/fp8/InfiniLM/python")

import infinicore
from infinilm.cache import PagedKVCacheConfig
from infinilm.distributed import DistConfig
from infinilm.infer_engine import GenerationConfig, InferEngine
from infinilm.modeling_utils import load_model_state_dict_by_file
from infinilm.processors import AutoInfinilmProcessor

MODEL = "/data/huggingface_home/hub/models--Qwen--Qwen3-8B-FP8/snapshots/220b46e3b2180893580a4454f21f22d3ebb187d3"


def generate(kv_dtype):
    processor = AutoInfinilmProcessor.from_pretrained(MODEL)
    tokenizer = processor.get_tokenizer()
    text = "The quick brown fox jumps over the lazy dog. " * 400
    text += " Question: what animal jumps over the dog? Answer:"
    content = processor.apply_chat_template(
        conversation=[{"role": "user", "content": text}],
        add_generation_prompt=True,
        tokenize=False,
    )
    ids = tokenizer.encode(content)
    print(f"[parity] kv={kv_dtype or 'default'} prompt_tokens={len(ids)}", flush=True)

    model = InferEngine(
        MODEL,
        device=infinicore.device("cuda", 0),
        distributed_config=DistConfig(1, moe_ep_backend="disabled", moe_ep_size=1),
        cache_config=PagedKVCacheConfig(64, 256, max_batch_size=1),
        enable_graph_compiling=False,
        attention_backend="paged-attn",
        kv_cache_dtype=kv_dtype,
        use_mla=False,
        weight_load_mode="async",
        pre_transpose=False,
    )
    try:
        load_model_state_dict_by_file(model, MODEL, dtype=model.dtype)
        out = model.generate(
            infinicore.from_list([ids], dtype=infinicore.int64),
            GenerationConfig(
                max_new_tokens=48,
                eos_token_id=[],
                top_k=1,
                top_p=1.0,
                temperature=1.0,
                stop_on_eos=False,
            ),
        )
        ids_out = [int(t.to_numpy().reshape(-1)[0]) for t in out]
        print(f"[parity] kv={kv_dtype or 'default'} out_tokens={len(ids_out)}", flush=True)
        return tokenizer.decode(ids_out, skip_special_tokens=True)
    finally:
        if hasattr(model, "close"):
            model.close()


a = generate("fp8")
b = generate(None)
print("=== FP8 ===")
print(a)
print("=== BF16 ===")
print(b)
print("PARITY:", "MATCH" if a == b else "DIVERGE")

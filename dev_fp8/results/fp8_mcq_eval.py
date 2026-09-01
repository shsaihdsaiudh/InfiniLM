#!/usr/bin/env python3
"""C-Eval / MMLU 选择题 logprob 评测(InfiniLM,InferEngine forward 直取 logits)。

方法论:每题一次 prefill,取最后一个位置的 logits,在 {A,B,C,D} 四个字母 token
上比较 logprob,argmax 作为预测。不做生成,避免聊天式输出("正确答案是:B")
导致的抽取失败;两模型同协议,可直接对比。与 lm-eval-harness 的 MCQ loglikelihood
口径一致。提示词渲染见 fp8_mcq_common.py(与 test/bench 框架一致)。

用法:
    source /root/fp8/fp8_env.sh && export HF_HUB_OFFLINE=1
    python3 /root/fp8/fp8_mcq_eval.py --model <path> --bench ceval --split val \
        --out /root/fp8/eval_logs/ceval_fp8.json [--limit 0] [--validate]
"""

import argparse
import ctypes
import json
import os
import sys
import time

import infinicore
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from infinilm.cache import PagedKVCacheConfig, StaticKVCacheConfig
from infinilm.distributed import DistConfig
from infinilm.generation.utils import infini_to_numpy
from infinilm.infer_engine import InferEngine
from infinilm.modeling_utils import load_model_state_dict_by_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fp8_mcq_common import LETTERS, build_prompt, limit_per_subject, load_samples

MAX_CACHE_LEN = 4096
_PAGED_BLOCK = 256


def make_cache_config(seqlen, kv_cache_dtype):
    """kv_cache_dtype 为 None 时维持 static 行为;否则走 paged(FP8 KV 唯一支持的后端)。"""
    if kv_cache_dtype is None:
        return StaticKVCacheConfig(max_batch_size=1, max_cache_len=seqlen)
    num_blocks = (seqlen + _PAGED_BLOCK - 1) // _PAGED_BLOCK + 8
    return PagedKVCacheConfig(
        num_blocks=num_blocks, block_size=_PAGED_BLOCK, max_batch_size=1
    )


def build_paged_meta(seq_len, kv_cache_dtype):
    """单序列 prefill 的 paged 元数据(与 generate 路径相同的连续块约定)。"""
    if kv_cache_dtype is None:
        return None, None
    n_blocks = (seq_len + _PAGED_BLOCK - 1) // _PAGED_BLOCK
    block_tables = infinicore.from_list([list(range(n_blocks))], dtype=infinicore.int32)
    slot_mapping = infinicore.from_list(list(range(seq_len)), dtype=infinicore.int64)
    return block_tables, slot_mapping


def logits_to_numpy_f32(t):
    if t.dtype == infinicore.bfloat16:
        if t.device.type != "cpu":
            t = t.to(infinicore.device("cpu", 0))
        n = t.numel()
        raw = np.ctypeslib.as_array(
            (ctypes.c_uint16 * n).from_address(t.data_ptr())
        ).astype(np.uint32)
        return (raw << 16).view(np.float32).reshape(t.shape)
    return np.asarray(infini_to_numpy(t), dtype=np.float32)


class Scorer:
    def __init__(self, model_path, device_str, kv_cache_dtype=None):
        self.kv_cache_dtype = kv_cache_dtype
        self.device = infinicore.device(device_str, 0)
        self.engine = InferEngine(
            model_path,
            device=self.device,
            distributed_config=DistConfig(1),
            cache_config=make_cache_config(MAX_CACHE_LEN, kv_cache_dtype),
            attention_backend="paged-attn" if kv_cache_dtype else "default",
            kv_cache_dtype=kv_cache_dtype,
        )
        load_model_state_dict_by_file(self.engine, model_path, dtype=self.engine.dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        # 每字母收集 {"A"," A"} 两个变体 token id
        self.letter_ids = []
        for letter in LETTERS:
            ids = set()
            for variant in (letter, " " + letter):
                enc = self.tokenizer.encode(variant, add_special_tokens=False)
                if len(enc) == 1:
                    ids.add(enc[0])
            assert ids, f"字母 {letter} 无单 token 变体"
            self.letter_ids.append(sorted(ids))

    def score_letters(self, ids, reset_cache):
        seq_len = len(ids)
        if self.kv_cache_dtype is not None:
            # paged 池不做跨题复用,逐题重置
            self.engine.reset_cache(make_cache_config(seq_len, self.kv_cache_dtype))
        elif reset_cache:
            self.engine.reset_cache(
                StaticKVCacheConfig(max_batch_size=1, max_cache_len=seq_len)
            )
        dev = self.device
        block_tables, slot_mapping = build_paged_meta(seq_len, self.kv_cache_dtype)
        out = self.engine.forward_raw(
            infinicore.from_list([ids], dtype=infinicore.int64).to(dev),
            position_ids=infinicore.from_list(
                [list(range(seq_len))], dtype=infinicore.int64
            ).to(dev),
            past_kv_lengths=infinicore.from_list([0], dtype=infinicore.int32),
            total_kv_lengths=infinicore.from_list([seq_len], dtype=infinicore.int32),
            input_offsets=infinicore.from_list([0, seq_len], dtype=infinicore.int32),
            cu_seqlens=infinicore.from_list([0, seq_len], dtype=infinicore.int32),
            block_tables=block_tables,
            slot_mapping=slot_mapping,
        )
        logits = logits_to_numpy_f32(out["logits"])
        last = logits[0, -1] if logits.ndim == 3 else logits[-1]
        lp = torch.log_softmax(torch.from_numpy(last), dim=-1)
        return [max(lp[i].item() for i in variants) for variants in self.letter_ids]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--bench", required=True, choices=["ceval", "mmlu"])
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="每 subject 限量,0=全部")
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--kv-cache-dtype",
        default=None,
        help="如 fp8:切 paged 后端并量化 KV cache;缺省保持 static BF16",
    )
    ap.add_argument(
        "--validate",
        action="store_true",
        help="前 50 题对比 reset_cache/复用两种模式,校验 cache 复用正确性",
    )
    args = ap.parse_args()

    samples = limit_per_subject(load_samples(args.bench, args.split), args.limit)
    print(f"共 {len(samples)} 题", flush=True)

    scorer = Scorer(args.model, args.device, kv_cache_dtype=args.kv_cache_dtype)

    if args.validate and args.kv_cache_dtype is not None:
        print("kv_cache_dtype 模式下逐题重置 cache,--validate 无意义,跳过", flush=True)
        args.validate = False

    if args.validate:
        n_checked = 0
        for s in samples[:50]:
            ids = scorer.tokenizer.encode(
                build_prompt(scorer.tokenizer, args.bench, s["sample"]),
                add_special_tokens=False,
            )
            a = scorer.score_letters(ids, reset_cache=True)
            b = scorer.score_letters(ids, reset_cache=False)
            assert np.argmax(a) == np.argmax(b), f"cache 复用导致 argmax 不一致: {a} vs {b}"
            n_checked += 1
        print(f"VALIDATE_OK: 前 {n_checked} 题 reset/复用 argmax 完全一致", flush=True)
        # validate 里把 cache 重置成了小尺寸,恢复到大缓冲
        scorer.engine.reset_cache(
            StaticKVCacheConfig(max_batch_size=1, max_cache_len=MAX_CACHE_LEN)
        )

    results = {}
    skipped = 0
    t0 = time.time()
    pbar = tqdm(samples, desc=args.bench)
    for s in pbar:
        prompt = build_prompt(scorer.tokenizer, args.bench, s["sample"])
        ids = scorer.tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) > MAX_CACHE_LEN:
            skipped += 1
            continue
        scores = scorer.score_letters(ids, reset_cache=False)
        pred = int(np.argmax(scores))
        r = results.setdefault(s["subject"], [0, 0])
        r[1] += 1
        r[0] += int(pred == s["answer_idx"])
        pbar.set_postfix(
            acc=f"{sum(v[0] for v in results.values()) / max(1, sum(v[1] for v in results.values())):.4f}"
        )

    total_c = sum(v[0] for v in results.values())
    total_n = sum(v[1] for v in results.values())
    elapsed = time.time() - t0
    report = {
        "model": args.model,
        "kv_cache_dtype": args.kv_cache_dtype,
        "bench": args.bench,
        "split": args.split,
        "method": "letter-logprob(argmax over A/B/C/D at last position, no generation)",
        "overall": {
            "correct": total_c,
            "total": total_n,
            "accuracy": total_c / max(1, total_n),
        },
        "per_subject": {
            k: {"correct": v[0], "total": v[1], "accuracy": v[0] / v[1]}
            for k, v in sorted(results.items())
        },
        "skipped_too_long": skipped,
        "eval_seconds": elapsed,
    }
    print(json.dumps(report["overall"], indent=2))
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"MCQ_SAVED {args.out} ({elapsed:.0f}s)", flush=True)


if __name__ == "__main__":
    main()

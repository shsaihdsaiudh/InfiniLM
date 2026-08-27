#!/usr/bin/env python3
"""进程内 wikitext2 PPL 评测（InferEngine forward 直取 logits）。

用法:
    PYTHONPATH=/root/fp8/InfiniCore/python:/root/fp8/InfiniLM/python \
    INFINI_ROOT=/root/fp8/.infini \
    python3 /root/fp8/fp8_ppl_eval.py --model Qwen/Qwen3-8B-FP8 --out /root/fp8/ppl_fp8.json

每个 chunk 独立 prefill（与 scripts/test_ppl.py 语义一致），
NLL 在 torch(float32) 上计算。
"""
import argparse
import ctypes
import json
import math
import os
import sys

import infinicore
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from infinilm.cache import StaticKVCacheConfig
from infinilm.distributed import DistConfig
from infinilm.generation.utils import infini_to_numpy
from infinilm.infer_engine import InferEngine
from infinilm.modeling_utils import load_model_state_dict_by_file


def logits_to_numpy_f32(t):
    """bf16 也兼容的 logits -> float32 numpy 转换。"""
    if t.dtype == infinicore.bfloat16:
        if t.device.type != "cpu":
            t = t.to(infinicore.device("cpu", 0))
        n = t.numel()
        raw = np.ctypeslib.as_array(
            (ctypes.c_uint16 * n).from_address(t.data_ptr())
        ).astype(np.uint32)
        return (raw << 16).view(np.float32).reshape(t.shape)
    return np.asarray(infini_to_numpy(t), dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--max-chunks", type=int, default=0, help="0 = 全部")
    ap.add_argument("--dataset", default="Salesforce/wikitext")
    ap.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = infinicore.device(args.device, 0)
    engine = InferEngine(
        args.model,
        device=device,
        distributed_config=DistConfig(1),
        cache_config=StaticKVCacheConfig(max_batch_size=1, max_cache_len=args.chunk),
    )
    load_model_state_dict_by_file(engine, args.model, dtype=engine.dtype)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dataset = load_dataset(args.dataset, args.dataset_config, split="test")

    total_nll = 0.0
    total_tokens = 0
    n_chunks = 0

    pbar = tqdm(desc="PPL chunks")
    for example in dataset:
        text = example["text"].strip()
        if not text:
            continue
        tokens = tokenizer.encode(text, add_special_tokens=False)
        for i in range(0, len(tokens), args.chunk):
            chunk = tokens[i : i + args.chunk]
            if len(chunk) < 2:
                continue
            if args.max_chunks and n_chunks >= args.max_chunks:
                break

            engine.reset_cache(
                StaticKVCacheConfig(max_batch_size=1, max_cache_len=len(chunk))
            )
            seq_len = len(chunk)
            ids = infinicore.from_list([chunk], dtype=infinicore.int64).to(device)
            pos = infinicore.from_list(
                [list(range(seq_len))], dtype=infinicore.int64
            ).to(device)
            past_kv = infinicore.from_list([0], dtype=infinicore.int32)
            total_kv = infinicore.from_list([seq_len], dtype=infinicore.int32)
            cu_seqlens = infinicore.from_list([0, seq_len], dtype=infinicore.int32)
            input_offsets = infinicore.from_list([0, seq_len], dtype=infinicore.int32)
            out = engine.forward_raw(
                ids,
                position_ids=pos,
                past_kv_lengths=past_kv,
                total_kv_lengths=total_kv,
                input_offsets=input_offsets,
                cu_seqlens=cu_seqlens,
            )
            logits = out["logits"]
            logits_np = logits_to_numpy_f32(logits)  # [1, seq, vocab]
            if logits_np.ndim == 3:
                logits_np = logits_np[0]
            assert logits_np.shape[0] == len(chunk), (
                f"logits seq dim {logits_np.shape[0]} != chunk {len(chunk)}"
            )

            with torch.no_grad():
                t = torch.from_numpy(np.asarray(logits_np, dtype=np.float32))
                # logits[:-1] 预测 tokens[1:]
                lp = torch.log_softmax(t[:-1], dim=-1)
                tgt = torch.tensor(chunk[1:], dtype=torch.long)
                nll = -lp.gather(-1, tgt.unsqueeze(-1)).sum().item()

            total_nll += nll
            total_tokens += len(chunk) - 1
            n_chunks += 1
            pbar.set_postfix(
                chunks=n_chunks, ppl=f"{math.exp(total_nll / total_tokens):.4f}"
            )
            pbar.update(1)
        if args.max_chunks and n_chunks >= args.max_chunks:
            break
    pbar.close()

    ppl = math.exp(total_nll / total_tokens)
    result = {
        "model": args.model,
        "chunk": args.chunk,
        "chunks": n_chunks,
        "tokens": total_tokens,
        "nll": total_nll,
        "ppl": ppl,
    }
    print(json.dumps(result, indent=2))
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"PPL_SAVED {args.out}")


if __name__ == "__main__":
    main()

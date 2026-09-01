#!/usr/bin/env python3
"""C-Eval / MMLU 选择题 logprob 评测(vLLM 后端,与 InfiniLM 版同协议同提示词)。

vLLM 不暴露任意 token 的 logits,改用 SamplingParams(logprobs=20, max_tokens=1)
取首生成 token 的 top-20 logprobs,查 A/B/C/D 字母 token;缺失记 -inf。
与 fp8_mcq_eval.py 的差异仅限打分通道,提示词/数据/判分完全一致。

用法(venv):
    HF_HUB_OFFLINE=1 /root/fp8/venv-vllm/bin/python /root/fp8/vllm_mcq_eval.py \
        --model <path> --bench mmlu --split test \
        --out /root/fp8/eval_logs/mmlu_vllm_fp8.json [--limit 0]
"""

import argparse
import json
import os
import sys
import time

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fp8_mcq_common import LETTERS, build_prompt, limit_per_subject, load_samples

NEG_INF = -1.0e30
BATCH = 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--bench", required=True, choices=["ceval", "mmlu"])
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    samples = limit_per_subject(load_samples(args.bench, args.split), args.limit)
    print(f"共 {len(samples)} 题", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    letter_ids = []
    for letter in LETTERS:
        ids = set()
        for variant in (letter, " " + letter):
            enc = tokenizer.encode(variant, add_special_tokens=False)
            if len(enc) == 1:
                ids.add(enc[0])
        assert ids, f"字母 {letter} 无单 token 变体"
        letter_ids.append(sorted(ids))

    llm = LLM(
        model=args.model,
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        enable_prefix_caching=False,
    )
    sp = SamplingParams(temperature=0, max_tokens=1, logprobs=20, detokenize=False)

    prompts = []
    for s in samples:
        text = build_prompt(tokenizer, args.bench, s["sample"])
        prompts.append(TokensPrompt(prompt_token_ids=tokenizer.encode(text)))

    results = {}
    t0 = time.time()
    for i in range(0, len(samples), BATCH):
        chunk_prompts = prompts[i : i + BATCH]
        outs = llm.generate(chunk_prompts, sp, use_tqdm=True)
        for j, out in enumerate(outs):
            top = out.outputs[0].logprobs[0]  # {token_id: Logprob}
            scores = [
                max((top[t].logprob for t in variants if t in top), default=NEG_INF)
                for variants in letter_ids
            ]
            pred = max(range(4), key=lambda k: scores[k])
            s = samples[i + j]
            r = results.setdefault(s["subject"], [0, 0])
            r[1] += 1
            r[0] += int(pred == s["answer_idx"])
        done = min(i + BATCH, len(samples))
        acc = sum(v[0] for v in results.values()) / sum(v[1] for v in results.values())
        print(f"进度 {done}/{len(samples)} 累计 acc={acc:.4f}", flush=True)

    total_c = sum(v[0] for v in results.values())
    total_n = sum(v[1] for v in results.values())
    elapsed = time.time() - t0
    report = {
        "model": args.model,
        "bench": args.bench,
        "split": args.split,
        "method": "letter-logprob via vLLM top-20 first-token logprobs(same prompts as InfiniLM run)",
        "overall": {
            "correct": total_c,
            "total": total_n,
            "accuracy": total_c / max(1, total_n),
        },
        "per_subject": {
            k: {"correct": v[0], "total": v[1], "accuracy": v[0] / v[1]}
            for k, v in sorted(results.items())
        },
        "eval_seconds": elapsed,
    }
    print(json.dumps(report["overall"], indent=2))
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"MCQ_SAVED {args.out} ({elapsed:.0f}s)", flush=True)


if __name__ == "__main__":
    main()

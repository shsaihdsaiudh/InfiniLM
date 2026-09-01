#!/usr/bin/env python3
"""vLLM decode 性能对照:与 examples/bench.py 同口径(input 128 / output 256)。

vLLM 离线 API 拿不到逐 token 时间戳,用两点法隔离 decode:
    ITL = (T(output_len=256) - T(output_len=64)) / 192
    decode tok/s = bs / ITL;TTFT ≈ T(64) - 63*ITL
ignore_eos 保证全长生成;关 prefix caching(prompt 全部相同,避免假性加速)。

用法:
    HF_HUB_OFFLINE=1 /root/fp8/venv-vllm/bin/python /root/fp8/vllm_perf_bench.py \
        --model <path> --tag vllm_fp8
"""

import argparse
import time

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

INPUT_LEN = 128
O_SHORT, O_LONG = 64, 256
BS_LIST = [1, 8, 16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    base = tokenizer.encode(
        "The quick brown fox jumps over the lazy dog. " * 64, add_special_tokens=False
    )[:INPUT_LEN]
    assert len(base) == INPUT_LEN

    llm = LLM(
        model=args.model,
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        enable_prefix_caching=False,
    )

    # warmup
    llm.generate(
        [TokensPrompt(prompt_token_ids=base)],
        SamplingParams(temperature=0, max_tokens=8, ignore_eos=True),
        use_tqdm=False,
    )

    for bs in BS_LIST:
        prompts = [TokensPrompt(prompt_token_ids=base) for _ in range(bs)]
        times = {}
        for olen in (O_SHORT, O_LONG):
            sp = SamplingParams(temperature=0, max_tokens=olen, ignore_eos=True)
            t0 = time.perf_counter()
            llm.generate(prompts, sp, use_tqdm=False)
            times[olen] = time.perf_counter() - t0
        itl = (times[O_LONG] - times[O_SHORT]) / (O_LONG - O_SHORT)
        ttft = times[O_SHORT] - (O_SHORT - 1) * itl
        print(
            f"VLLM_PERF {args.tag} bs={bs} "
            f"ttft_ms={ttft * 1e3:.2f} prefill_toks={bs * INPUT_LEN / max(ttft, 1e-9):.2f} "
            f"itl_ms={itl * 1e3:.2f} decode_toks={bs / max(itl, 1e-9):.2f} "
            f"t64={times[O_SHORT]:.2f}s t256={times[O_LONG]:.2f}s",
            flush=True,
        )
    print("VLLM_PERF_DONE", flush=True)


if __name__ == "__main__":
    main()

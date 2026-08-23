"""Offline dual-engine perf baseline (project #2 gap analysis).

One script, two interpreters:

  InfiniLM (project .venv, imports the built package from the main checkout):
    /home/yyy/src/InfiniLM/.venv/bin/python dev_perf/bench.py --engine infinilm --model Qwen/Qwen3-1.7B

  vLLM (isolated venv):
    /home/yyy/src/venvs/vllm-bench/bin/python dev_perf/bench.py --engine vllm --model Qwen/Qwen3-1.7B

Fairness contract: greedy decoding (temperature=0, top_k=1), ignore_eos=True,
identical prompts and max_tokens from workload.py. vLLM runs with its v1
defaults (CUDA graphs + prefix caching on); InfiniLM runs with
enable_graph=False + prefix caching on. Both facts are recorded in the output.
"""

import argparse
import json
import os
import sys
import time

from workload import WORKLOADS, MemSampler, resolve_model_path

MAIN_CHECKOUT_PYTHON = "/home/yyy/src/InfiniLM/python"
LIB_DIRS = [
    "/home/yyy/src/InfiniCore/python/infinicore/lib",
    "/home/yyy/src/InfiniLM/python/infinilm/lib",
]


def ensure_ld_library_path():
    """infinilm/infinicore extensions find their .so deps via LD_LIBRARY_PATH.

    Re-exec the interpreter with the env set if needed (the dynamic linker
    reads LD_LIBRARY_PATH only at process start).
    """
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    if all(d in cur.split(":") for d in LIB_DIRS):
        return
    os.environ["LD_LIBRARY_PATH"] = ":".join(LIB_DIRS + ([cur] if cur else []))
    os.execv(sys.executable, [sys.executable] + sys.argv)


def build_engine(args):
    if args.engine == "infinilm":
        ensure_ld_library_path()
        sys.path.insert(0, MAIN_CHECKOUT_PYTHON)
        from infinilm.llm.llm import LLM
        from infinilm.llm.sampling_params import SamplingParams

        llm = LLM(
            model_path=args.model,
            device="cuda",
            dtype="bfloat16",
            cache_type="paged",
            attn_backend="paged-attn",
            max_batch_size=64,
            enable_prefix_caching=True,
        )

        def generate(prompts, max_tokens):
            return llm.generate(
                prompts=prompts,
                sampling_params=SamplingParams(
                    temperature=0.0, top_k=1, max_tokens=max_tokens, ignore_eos=True
                ),
                use_tqdm=False,
            )

        def close():
            llm.close()

        engine_notes = {"engine": "infinilm", "cuda_graph": False, "prefix_caching": True, "attn_backend": "paged-attn"}

    elif args.engine == "vllm":
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=args.model,
            dtype="bfloat16",
            gpu_memory_utilization=0.85,
            max_model_len=4096,
            seed=0,
        )

        def generate(prompts, max_tokens):
            return llm.generate(
                prompts,
                SamplingParams(
                    temperature=0.0, top_k=1, max_tokens=max_tokens, ignore_eos=True
                ),
                use_tqdm=False,
            )

        def close():
            del llm

        engine_notes = {"engine": "vllm", "cuda_graph": True, "prefix_caching": True}

    else:
        raise ValueError(f"unknown engine: {args.engine}")

    return generate, close, engine_notes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["infinilm", "vllm"], required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument(
        "--out-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"),
    )
    args = parser.parse_args()

    # WSL2: vLLM disables pinned memory by default, which makes its V2 model
    # runner's UvaBuffer raise "UVA is not available". Pinned memory works on
    # WSL2 kernels >= 4.19.121 (verified locally), so opt back in.
    os.environ.setdefault("VLLM_WSL2_ENABLE_PIN_MEMORY", "1")

    args.model = resolve_model_path(args.model)
    print(f"[bench] engine={args.engine} model={args.model}", flush=True)

    sampler = MemSampler()
    sampler.start()

    t_load0 = time.perf_counter()
    generate, close, engine_notes = build_engine(args)
    load_seconds = time.perf_counter() - t_load0
    mem_after_load = sampler.latest
    print(f"[bench] load took {load_seconds:.1f}s, mem={mem_after_load} MiB", flush=True)

    # Warmup (untimed): runs W1 once to trigger lazy init / graph capture.
    w1_name, w1_prompts, w1_mt = WORKLOADS[0]
    generate(w1_prompts, w1_mt)

    results = []
    for name, prompts, max_tokens in WORKLOADS:
        t0 = time.perf_counter()
        outputs = generate(prompts, max_tokens)
        e2e = time.perf_counter() - t0

        prompt_tokens = sum(len(o.prompt_token_ids or []) for o in outputs)
        out_token_counts = [len(o.outputs[0].token_ids) for o in outputs]
        total_out = sum(out_token_counts)
        rec = {
            "workload": name,
            "num_requests": len(prompts),
            "prompt_tokens_total": prompt_tokens,
            "output_tokens_total": total_out,
            "e2e_seconds": round(e2e, 3),
            "output_tokens_per_sec": round(total_out / e2e, 2),
            "ms_per_output_token": round(1000.0 * e2e / max(total_out, 1), 3),
            "min_output_tokens": min(out_token_counts),
            "max_output_tokens": max(out_token_counts),
        }
        results.append(rec)
        print(
            f"[bench] {name}: e2e={e2e:.2f}s out={total_out} tok "
            f"({rec['output_tokens_per_sec']} tok/s, "
            f"{rec['ms_per_output_token']} ms/tok)",
            flush=True,
        )

    sampler.stop()
    close()

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": args.model,
        **engine_notes,
        "load_seconds": round(load_seconds, 2),
        "mem_after_load_mib": mem_after_load,
        "mem_peak_mib": sampler.peak,
        "workloads": results,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    tag = os.path.basename(args.model.rstrip("/"))
    out_path = os.path.join(
        args.out_dir, f"{args.engine}_{tag}_{time.strftime('%m%d_%H%M%S')}.json"
    )
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[bench] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()

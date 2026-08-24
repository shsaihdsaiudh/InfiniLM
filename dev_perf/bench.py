"""Offline dual-engine perf baseline (project #2 gap analysis).

One script, two interpreters:

  InfiniLM (any venv with torch; imports the built package from this checkout):
    python dev_perf/bench.py --engine infinilm --model Qwen/Qwen3-1.7B

  vLLM (isolated venv):
    /path/to/vllm-venv/bin/python dev_perf/bench.py --engine vllm --model Qwen/Qwen3-1.7B

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

# Checkout whose built infinilm package we benchmark. Defaults to this repo's
# own python/ dir (where `xmake install _infinilm` lands its .so); override
# with INF_MAIN_PYTHON to benchmark a different checkout's build.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN_CHECKOUT_PYTHON = os.environ.get("INF_MAIN_PYTHON", os.path.join(_REPO_ROOT, "python"))
# INFINI_ROOT selects the InfiniCore install (e.g. ~/.infini-fa for the
# ATen+FA build, required by the flash-attn backend); defaults to ~/.infini.
_INFINI_CORE_LIB = os.path.join(
    os.environ.get("INFINI_ROOT", os.path.expanduser("~/.infini")), "lib"
)
def _torch_lib_dir():
    """site-packages/torch/lib, needed on LD_LIBRARY_PATH for ATen-enabled
    InfiniCore builds (they link libtorch)."""
    for p in sys.path:
        d = os.path.join(p, "torch", "lib")
        if os.path.isdir(d):
            return d
    return None


LIB_DIRS = [
    _INFINI_CORE_LIB,
    os.path.join(MAIN_CHECKOUT_PYTHON, "infinilm", "lib"),
] + ([_torch_lib_dir()] if _torch_lib_dir() else [])


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
            attn_backend=args.attn_backend,
            max_batch_size=64,
            num_blocks=args.num_blocks,
            enable_prefix_caching=True,
            enable_graph=args.enable_graph,
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

        engine_notes = {"engine": "infinilm", "cuda_graph": bool(args.enable_graph), "prefix_caching": True, "attn_backend": args.attn_backend, "num_blocks": args.num_blocks}

    elif args.engine == "vllm":
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=args.model,
            dtype="bfloat16",
            gpu_memory_utilization=args.gpu_mem_util,
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
            try:
                llm.llm_engine.engine_core.shutdown()
            except Exception:
                pass

        engine_notes = {"engine": "vllm", "cuda_graph": True, "prefix_caching": True, "gpu_memory_utilization": args.gpu_mem_util}

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
    parser.add_argument(
        "--enable-graph",
        action="store_true",
        help="infinilm only: enable graph compiling (CUDA-graph-like capture)",
    )
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=512,
        help="infinilm only: paged KV cache blocks (x256 tokens); lower it for a low-VRAM run",
    )
    parser.add_argument(
        "--gpu-mem-util",
        type=float,
        default=0.85,
        help="vllm only: gpu_memory_utilization; lower it to stay under VRAM watchdogs",
    )
    parser.add_argument(
        "--only",
        default=None,
        help="comma-separated workload names to run (default: all four)",
    )
    parser.add_argument(
        "--attn-backend",
        default="paged-attn",
        choices=["default", "static-attn", "paged-attn", "flash-attn", "flashinfer"],
        help="infinilm only: attention backend",
    )
    parser.add_argument(
        "--dump-outputs",
        action="store_true",
        help="record per-request output token ids in the JSON (for cross-engine correctness diffing)",
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

    only = set(args.only.split(",")) if args.only else None
    results = []
    for name, prompts, max_tokens in WORKLOADS:
        if only and name not in only:
            continue
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
        if args.dump_outputs:
            rec["output_token_ids"] = [list(o.outputs[0].token_ids) for o in outputs]
        results.append(rec)
        print(
            f"[bench] {name}: e2e={e2e:.2f}s out={total_out} tok "
            f"({rec['output_tokens_per_sec']} tok/s, "
            f"{rec['ms_per_output_token']} ms/tok)",
            flush=True,
        )

    sampler.stop()

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

    # Shut down only after results are persisted; engine teardown must not
    # be allowed to take the report down with it.
    try:
        close()
    except Exception as e:
        print(f"[bench] close() failed (results already saved): {e}", flush=True)


if __name__ == "__main__":
    main()

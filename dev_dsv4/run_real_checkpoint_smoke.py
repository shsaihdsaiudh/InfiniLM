#!/usr/bin/env python3
"""Run the final batch-1 DeepSeek-V4 checkpoint acceptance smoke."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


MIN_FREE_GIB = 165.0


def checkpoint_preflight(model_path: Path) -> dict:
    config_path = model_path / "config.json"
    index_path = model_path / "model.safetensors.index.json"
    if not config_path.is_file() or not index_path.is_file():
        raise FileNotFoundError(
            "The model directory must contain config.json and "
            "model.safetensors.index.json"
        )
    config = json.loads(config_path.read_text())
    if config.get("model_type") != "deepseek_v4":
        raise ValueError(
            f"Expected model_type='deepseek_v4', got {config.get('model_type')!r}"
        )
    if config.get("expert_dtype") != "fp4":
        raise ValueError(
            f"Expected expert_dtype='fp4', got {config.get('expert_dtype')!r}"
        )
    index = json.loads(index_path.read_text())
    shard_names = sorted(set(index.get("weight_map", {}).values()))
    missing = [name for name in shard_names if not (model_path / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} checkpoint shards; first: {missing[0]}"
        )
    if len(shard_names) != 48:
        raise ValueError(f"Expected 48 checkpoint shards, got {len(shard_names)}")
    return config


def gpu_preflight(min_free_gib: float) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    free_gib = free_bytes / 2**30
    total_gib = total_bytes / 2**30
    properties = torch.cuda.get_device_properties(0)
    print(
        f"GPU: {properties.name}; capability={properties.major}."
        f"{properties.minor}; free/total={free_gib:.2f}/{total_gib:.2f} GiB"
    )
    if free_gib < min_free_gib:
        raise RuntimeError(
            f"DeepSeek-V4 acceptance needs at least {min_free_gib:.1f} GiB "
            f"free for the current 150.756-GiB base parameter layout; "
            f"only {free_gib:.1f} GiB is free"
        )
    return free_bytes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--max-cache-len", type=int, default=4096)
    parser.add_argument("--min-free-gib", type=float, default=MIN_FREE_GIB)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="validate files and GPU capacity without allocating the model",
    )
    args = parser.parse_args()
    if args.max_new_tokens <= 0 or args.max_cache_len <= 0:
        parser.error("token counts must be positive")

    model_path = args.model.expanduser().resolve()
    config = checkpoint_preflight(model_path)
    free_before = gpu_preflight(args.min_free_gib)
    print(
        "Checkpoint preflight passed: "
        f"layers={config.get('num_hidden_layers')}, shards=48, "
        f"expert_dtype={config.get('expert_dtype')}"
    )
    if args.preflight_only:
        return

    import infinicore
    from infinilm.cache import StaticKVCacheConfig
    from infinilm.distributed import DistConfig
    from infinilm.infer_engine import GenerationConfig, InferEngine
    from infinilm.modeling_utils import load_model_state_dict_by_file
    from infinilm.processors import AutoInfinilmProcessor

    cache_config = StaticKVCacheConfig(
        max_batch_size=1, max_cache_len=args.max_cache_len
    )
    engine = InferEngine(
        str(model_path),
        device=infinicore.device("cuda", 0),
        distributed_config=DistConfig(1),
        cache_config=cache_config,
        enable_graph_compiling=False,
        attention_backend="default",
        weight_load_mode="sync",
    )
    load_model_state_dict_by_file(
        engine, str(model_path), dtype=engine.dtype
    )

    processor = AutoInfinilmProcessor.from_pretrained(str(model_path))
    tokenizer = processor.get_tokenizer()
    token_ids = tokenizer(
        args.prompt, add_special_tokens=False
    ).input_ids
    if not token_ids:
        raise ValueError("The prompt tokenized to an empty sequence")
    if len(token_ids) + args.max_new_tokens > args.max_cache_len:
        raise ValueError("Prompt plus generated tokens exceed --max-cache-len")

    input_ids = infinicore.from_list(
        [token_ids], dtype=infinicore.int64
    )
    position_ids = infinicore.from_list(
        [list(range(len(token_ids)))], dtype=infinicore.int64
    )
    input_offsets = infinicore.from_list(
        [0, len(token_ids)], dtype=infinicore.int32
    )
    past_kv_lengths = infinicore.from_list([0], dtype=infinicore.int32)
    total_kv_lengths = infinicore.from_list(
        [len(token_ids)], dtype=infinicore.int32
    )
    cu_seqlens = infinicore.from_list(
        [0, len(token_ids)], dtype=infinicore.int32
    )
    raw = engine.forward_raw(
        input_ids,
        position_ids=position_ids,
        past_kv_lengths=past_kv_lengths,
        total_kv_lengths=total_kv_lengths,
        input_offsets=input_offsets,
        cu_seqlens=cu_seqlens,
        sample_all_positions=True,
    )
    logits = raw["logits"].to_numpy()
    hidden = raw["hidden_states"].to_numpy()
    if not np.isfinite(logits).all() or not np.isfinite(hidden).all():
        raise RuntimeError("Prefill produced NaN or Inf")
    print(
        f"Prefill passed: logits={logits.shape}, hidden={hidden.shape}, "
        f"last_argmax={int(logits[0, -1].argmax())}"
    )
    engine.reset_cache(cache_config)

    outputs = engine.generate(
        input_ids,
        GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=1.0,
            top_k=1,
            top_p=1.0,
            stop_on_eos=False,
        ),
        _measure_and_log_time=True,
    )
    generated_ids = [int(output.to_numpy().reshape(-1)[0]) for output in outputs]
    print(f"Prompt tokens: {token_ids}")
    print(f"Generated tokens: {generated_ids}")
    print(
        "Generated text: "
        + tokenizer.decode(generated_ids, skip_special_tokens=False)
    )
    infinicore.sync_device()
    free_after, _ = torch.cuda.mem_get_info(0)
    print(
        "Approximate device-memory delta: "
        f"{(free_before - free_after) / 2**30:.3f} GiB"
    )


if __name__ == "__main__":
    main()

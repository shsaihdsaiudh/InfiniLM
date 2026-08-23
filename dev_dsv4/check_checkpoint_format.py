"""Inspect DeepSeek-V4 safetensors headers without downloading the weights.

The Hugging Face resolve endpoint supports byte ranges.  A safetensors file
starts with an eight-byte little-endian header length followed by a JSON header,
so this probe downloads only the metadata from every shard.
"""

import argparse
import concurrent.futures
import json
import urllib.parse
import urllib.request
from collections import Counter
from math import prod


DEFAULT_REPO = "deepseek-ai/DeepSeek-V4-Flash-DSpark"
EXPECTED_TOTAL_SIZE = 166_878_536_440
MAX_HEADER_SIZE = 64 * 1024 * 1024


def fetch(url: str, byte_range: tuple[int, int] | None = None) -> bytes:
    headers = {"User-Agent": "InfiniLM-DeepSeek-V4-checkpoint-probe/1.0"}
    if byte_range is not None:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def fetch_safetensors_header(url: str) -> dict:
    prefix = fetch(url, (0, 7))
    if len(prefix) != 8:
        raise RuntimeError(f"Expected 8-byte safetensors prefix from {url}")

    header_size = int.from_bytes(prefix, byteorder="little", signed=False)
    if header_size <= 0 or header_size > MAX_HEADER_SIZE:
        raise RuntimeError(f"Invalid safetensors header size {header_size} in {url}")

    payload = fetch(url, (8, 8 + header_size - 1))
    if len(payload) != header_size:
        raise RuntimeError(
            f"Expected {header_size} header bytes from {url}, got {len(payload)}"
        )
    return json.loads(payload)


def tensor_category(name: str) -> str:
    if name.startswith("mtp."):
        return "mtp"
    if ".experts." in name and ".shared_experts." not in name:
        return "routed_experts"
    if ".attn." in name:
        return "attention"
    if ".hc" in name:
        return "mhc"
    if ".ffn.gate." in name:
        return "router"
    return "other"


def inspect_checkpoint(repo: str, revision: str) -> tuple[dict, dict, dict]:
    base_url = (
        f"https://huggingface.co/{repo}/resolve/"
        f"{urllib.parse.quote(revision, safe='')}/"
    )
    index = json.loads(fetch(base_url + "model.safetensors.index.json"))
    config = json.loads(fetch(base_url + "config.json"))
    shard_names = sorted(set(index["weight_map"].values()))

    def fetch_shard(shard_name: str) -> tuple[str, dict]:
        shard_url = base_url + urllib.parse.quote(shard_name)
        header = fetch_safetensors_header(shard_url)
        header.pop("__metadata__", None)
        return shard_name, header

    tensors = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        headers = executor.map(fetch_shard, shard_names)
        for shard_index, (shard_name, header) in enumerate(headers, start=1):
            overlap = tensors.keys() & header.keys()
            if overlap:
                raise RuntimeError(
                    f"Duplicate tensors in checkpoint: {sorted(overlap)[:3]}"
                )
            tensors.update(header)
            print(
                f"[{shard_index:02d}/{len(shard_names):02d}] "
                f"{shard_name}: {len(header)} tensors",
                flush=True,
            )

    return index, tensors, config


def validate_v4_config(config: dict) -> list[str]:
    errors = []
    expected = {
        "model_type": "deepseek_v4",
        "torch_dtype": "bfloat16",
        "hidden_size": 4096,
        "num_hidden_layers": 43,
        "num_attention_heads": 64,
        "head_dim": 512,
        "q_lora_rank": 1024,
        "qk_rope_head_dim": 64,
        "o_lora_rank": 1024,
        "o_groups": 8,
        "n_routed_experts": 256,
        "num_experts_per_tok": 6,
        "moe_intermediate_size": 2048,
        "n_shared_experts": 1,
        "num_hash_layers": 3,
        "sliding_window": 128,
        "index_n_heads": 64,
        "index_head_dim": 128,
        "index_topk": 512,
        "hc_mult": 4,
        "hc_sinkhorn_iters": 20,
        "hc_eps": 1e-6,
        "swiglu_limit": 10.0,
        "expert_dtype": "fp4",
        "max_position_embeddings": 1_048_576,
        "compress_rope_theta": 160_000,
    }
    for name, expected_value in expected.items():
        if config.get(name) != expected_value:
            errors.append(
                f"config {name}: expected {expected_value!r}, "
                f"got {config.get(name)!r}"
            )

    expected_ratios = [0, 0] + [
        4 if layer % 2 == 0 else 128 for layer in range(2, 43)
    ]
    if config.get("compress_ratios", [])[:43] != expected_ratios:
        errors.append("config compress_ratios does not match the 43 base layers")

    rope = config.get("rope_scaling") or {}
    expected_rope = {
        "type": "yarn",
        "factor": 16,
        "beta_fast": 32,
        "beta_slow": 1,
        "original_max_position_embeddings": 65_536,
    }
    for name, expected_value in expected_rope.items():
        if rope.get(name) != expected_value:
            errors.append(
                f"config rope_scaling.{name}: expected {expected_value!r}, "
                f"got {rope.get(name)!r}"
            )

    quant = config.get("quantization_config") or {}
    if quant.get("fmt") != "e4m3" or quant.get("scale_fmt") != "ue8m0":
        errors.append("config FP8 formats are not e4m3/ue8m0")
    if quant.get("weight_block_size") != [128, 128]:
        errors.append("config FP8 weight_block_size is not [128, 128]")
    return errors


def validate_v4_layout(index: dict, tensors: dict, config: dict) -> list[str]:
    errors = []
    total_size = index.get("metadata", {}).get("total_size")
    if total_size != EXPECTED_TOTAL_SIZE:
        errors.append(
            f"unexpected total_size: expected {EXPECTED_TOTAL_SIZE}, got {total_size}"
        )

    weight_map = index.get("weight_map", {})
    if set(weight_map) != set(tensors):
        errors.append(
            "index/header tensor sets differ: "
            f"index-only={len(set(weight_map) - set(tensors))}, "
            f"header-only={len(set(tensors) - set(weight_map))}"
        )

    expert_weights = []
    expert_scales = []
    pair_count = 0
    for name, metadata in tensors.items():
        if not name.endswith(".weight"):
            continue
        scale_name = name.removesuffix(".weight") + ".scale"
        scale = tensors.get(scale_name)
        is_routed_expert = (
            ".experts." in name and ".shared_experts." not in name
        )
        if is_routed_expert:
            expert_weights.append(metadata["dtype"])
            if scale is None:
                errors.append(f"missing routed expert scale for {name}")
                continue
            expert_scales.append(scale["dtype"])
        if scale is None:
            continue
        pair_count += 1
        if weight_map.get(name) != weight_map.get(scale_name):
            errors.append(f"weight/scale pair crosses shards: {name}")
        if metadata["dtype"] == "F8_E4M3" and len(metadata["shape"]) == 2:
            expected_scale_shape = [
                (metadata["shape"][0] + 127) // 128,
                (metadata["shape"][1] + 127) // 128,
            ]
            if scale["shape"] != expected_scale_shape:
                errors.append(
                    f"{scale_name}: expected shape {expected_scale_shape}, "
                    f"got {scale['shape']}"
                )

    if not expert_weights or set(expert_weights) != {"I8"}:
        errors.append(f"routed expert weight dtypes are {sorted(set(expert_weights))}")
    if not expert_scales or set(expert_scales) != {"F8_E8M0"}:
        errors.append(f"routed expert scale dtypes are {sorted(set(expert_scales))}")

    required = {
        "layers.0.ffn.gate.tid2eid": (
            "I64",
            [config.get("vocab_size"), config.get("num_experts_per_tok")],
        ),
        "layers.0.attn.attn_sink": ("F32", [64]),
        "layers.0.attn.wq_a.weight": ("F8_E4M3", [1024, 4096]),
        "layers.0.attn.wq_a.scale": ("F8_E8M0", [8, 32]),
        "layers.0.ffn.experts.0.w1.weight": ("I8", [2048, 2048]),
        "layers.0.ffn.experts.0.w1.scale": ("F8_E8M0", [2048, 128]),
        "layers.0.ffn.experts.0.w2.weight": ("I8", [4096, 1024]),
        "layers.0.ffn.experts.0.w2.scale": ("F8_E8M0", [4096, 64]),
        "layers.2.attn.compressor.ape": ("F32", [4, 1024]),
        "layers.3.attn.compressor.ape": ("F32", [128, 512]),
    }
    for name, (expected_dtype, expected_shape) in required.items():
        actual = tensors.get(name, {})
        if actual.get("dtype") != expected_dtype:
            errors.append(
                f"{name}: expected dtype {expected_dtype}, "
                f"got {actual.get('dtype')}"
            )
        if actual.get("shape") != expected_shape:
            errors.append(
                f"{name}: expected shape {expected_shape}, "
                f"got {actual.get('shape')}"
            )

    if pair_count != 35_718:
        errors.append(f"expected 35,718 weight/scale pairs, got {pair_count}")

    errors.extend(validate_v4_config(config))

    return errors


def estimate_resident_parameter_bytes(tensors: dict) -> int:
    """Estimate the base-model bytes after the current V4 remapper."""
    total = 0
    for name, metadata in tensors.items():
        if name.startswith("mtp."):
            continue
        is_routed_expert = (
            ".experts." in name and ".shared_experts." not in name
        )
        if name.endswith(".scale") and not is_routed_expert:
            continue
        if metadata["dtype"] == "F8_E4M3" and name.endswith(".weight"):
            total += prod(metadata["shape"]) * 2
        else:
            start, end = metadata["data_offsets"]
            total += end - start
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()

    index, tensors, config = inspect_checkpoint(args.repo, args.revision)

    dtype_bytes = Counter()
    category_bytes = Counter()
    for name, metadata in tensors.items():
        start, end = metadata["data_offsets"]
        size = end - start
        dtype_bytes[metadata["dtype"]] += size
        category_bytes[tensor_category(name)] += size

    print("\nDtypes:")
    for dtype, size in dtype_bytes.most_common():
        print(f"  {dtype:10s} {size / 2**30:9.3f} GiB")

    print("\nCategories:")
    for category, size in category_bytes.most_common():
        print(f"  {category:16s} {size / 2**30:9.3f} GiB")

    print(
        f"\nIndex total: {index['metadata']['total_size'] / 2**30:.3f} GiB; "
        f"tensors: {len(tensors)}"
    )
    resident_bytes = estimate_resident_parameter_bytes(tensors)
    print(
        "Estimated resident base parameters after remap: "
        f"{resident_bytes / 2**30:.3f} GiB "
        "(before runtime buffers and allocator overhead)"
    )

    errors = validate_v4_layout(index, tensors, config)
    if errors:
        raise SystemExit("Checkpoint layout validation failed:\n- " + "\n- ".join(errors))
    print("Checkpoint layout matches the DeepSeek-V4 FP8/FP4 assumptions.")


if __name__ == "__main__":
    main()

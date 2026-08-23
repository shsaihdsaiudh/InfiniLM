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
    if ".experts." in name and ".shared_experts." not in name:
        return "routed_experts"
    if ".attn." in name:
        return "attention"
    if name.startswith("mtp."):
        return "mtp"
    if ".hc" in name:
        return "mhc"
    if ".ffn.gate." in name:
        return "router"
    return "other"


def inspect_checkpoint(repo: str, revision: str) -> tuple[dict, dict]:
    base_url = (
        f"https://huggingface.co/{repo}/resolve/"
        f"{urllib.parse.quote(revision, safe='')}/"
    )
    index = json.loads(fetch(base_url + "model.safetensors.index.json"))
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

    return index, tensors


def validate_v4_layout(index: dict, tensors: dict) -> list[str]:
    errors = []
    total_size = index.get("metadata", {}).get("total_size")
    if total_size != EXPECTED_TOTAL_SIZE:
        errors.append(
            f"unexpected total_size: expected {EXPECTED_TOTAL_SIZE}, got {total_size}"
        )

    expert_weights = [
        metadata["dtype"]
        for name, metadata in tensors.items()
        if ".experts." in name and name.endswith(".weight")
    ]
    expert_scales = [
        metadata["dtype"]
        for name, metadata in tensors.items()
        if ".experts." in name and name.endswith(".scale")
    ]
    if not expert_weights or set(expert_weights) != {"I8"}:
        errors.append(f"routed expert weight dtypes are {sorted(set(expert_weights))}")
    if not expert_scales or set(expert_scales) != {"F8_E8M0"}:
        errors.append(f"routed expert scale dtypes are {sorted(set(expert_scales))}")

    required = {
        "layers.0.ffn.gate.tid2eid": "I64",
        "layers.0.attn.attn_sink": "F32",
        "layers.0.attn.wq_a.weight": "F8_E4M3",
        "layers.0.attn.wq_a.scale": "F8_E8M0",
    }
    for name, expected_dtype in required.items():
        actual = tensors.get(name, {}).get("dtype")
        if actual != expected_dtype:
            errors.append(f"{name}: expected {expected_dtype}, got {actual}")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()

    index, tensors = inspect_checkpoint(args.repo, args.revision)

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

    errors = validate_v4_layout(index, tensors)
    if errors:
        raise SystemExit("Checkpoint layout validation failed:\n- " + "\n- ".join(errors))
    print("Checkpoint layout matches the DeepSeek-V4 FP8/FP4 assumptions.")


if __name__ == "__main__":
    main()

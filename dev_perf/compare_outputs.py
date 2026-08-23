#!/usr/bin/env python3
"""Compare output_token_ids across bench JSON files (greedy decoding).

Usage: compare_outputs.py file_a.json file_b.json [file_c.json ...]

For each workload present in all files, reports per-request token match:
exact match count, and for mismatches the first divergence position.
"""
import json
import sys


def load(path):
    d = json.load(open(path))
    label = f"{d['engine']}/{d.get('attn_backend', '-')}"
    wl = {w["workload"]: w.get("output_token_ids") for w in d["workloads"]}
    return label, wl


def main(paths):
    loaded = [load(p) for p in paths]
    base_label, base_wl = loaded[0]
    print(f"reference: {base_label} ({paths[0]})")
    for label, wl in loaded[1:]:
        print(f"\n=== {label} vs {base_label} ===")
        for name, ref_ids in base_wl.items():
            ids = wl.get(name)
            if ref_ids is None or ids is None:
                print(f"  {name}: missing token ids, skipped")
                continue
            n_exact = 0
            worst_div = None
            for i, (a, b) in enumerate(zip(ref_ids, ids)):
                if list(a) == list(b):
                    n_exact += 1
                    continue
                div = next(
                    (j for j, (x, y) in enumerate(zip(a, b)) if x != y),
                    min(len(a), len(b)),
                )
                if worst_div is None or div < worst_div:
                    worst_div = div
            total = len(ref_ids)
            if n_exact == total:
                print(f"  {name}: {total}/{total} requests exact match")
            else:
                print(
                    f"  {name}: {n_exact}/{total} exact, "
                    f"earliest divergence at token {worst_div}"
                )


if __name__ == "__main__":
    main(sys.argv[1:])

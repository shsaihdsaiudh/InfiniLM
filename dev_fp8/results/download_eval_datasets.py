#!/usr/bin/env python3
"""预下载 C-Eval(val) / MMLU(test) 全部 subject 到 HF datasets 缓存。

subject 清单直接从 test/bench/test_benchmark.py 用 AST 提取,保证与评测框架一致。
运行:source /root/fp8/fp8_env.sh && python3 /root/fp8/download_eval_datasets.py
"""

import ast
import sys

from datasets import load_dataset

REPO_BENCH = "/root/fp8/InfiniLM/test/bench/test_benchmark.py"


def extract_subject_lists(path):
    tree = ast.parse(open(path).read())
    out = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.List)
            and getattr(node.targets[0], "id", "") in ("ceval_subjects", "mmlu_subjects")
        ):
            out[node.targets[0].id] = [e.value for e in node.value.elts]
    if set(out) != {"ceval_subjects", "mmlu_subjects"}:
        sys.exit(f"从 {path} 提取 subject 清单失败: {sorted(out)}")
    return out


def main():
    lists = extract_subject_lists(REPO_BENCH)
    jobs = [("ceval/ceval-exam", s, "val") for s in lists["ceval_subjects"]]
    jobs += [("cais/mmlu", s, "test") for s in lists["mmlu_subjects"]]
    print(f"共 {len(jobs)} 个下载任务", flush=True)

    fails = []
    total = 0
    for repo, name, split in jobs:
        for attempt in range(3):
            try:
                n = len(load_dataset(repo, name, split=split))
                print(f"OK {repo}::{name} [{split}] {n} 条", flush=True)
                total += n
                break
            except Exception as e:
                print(f"RETRY{attempt + 1} {repo}::{name}: {e}", flush=True)
        else:
            fails.append((repo, name))

    print(f"\n完成: {len(jobs) - len(fails)}/{len(jobs)} 个 subject, 共 {total} 条样本", flush=True)
    if fails:
        print(f"失败清单: {fails}", flush=True)
        sys.exit(1)
    print("ALL_DATASETS_READY", flush=True)


if __name__ == "__main__":
    main()

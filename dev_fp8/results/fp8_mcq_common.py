#!/usr/bin/env python3
"""MCQ 评测公共部分:提示词渲染 + 数据集加载(InfiniLM/vLLM 两版评测共用)。

不引入任何重型依赖(infinilm/vllm),只依赖 datasets/transformers/ast。
"""

import ast

LETTERS = ["A", "B", "C", "D"]
REPO_BENCH = "/root/fp8/InfiniLM/test/bench/test_benchmark.py"


def subject_list(bench):
    tree = ast.parse(open(REPO_BENCH).read())
    key = "ceval_subjects" if bench == "ceval" else "mmlu_subjects"
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and getattr(node.targets[0], "id", "") == key
            and isinstance(node.value, ast.List)
        ):
            return [e.value for e in node.value.elts]
    raise RuntimeError("subject 清单提取失败")


def build_prompt(tokenizer, bench, sample):
    """与 test/bench/backends/base.py 渲染一致;ceval cue 补全角冒号对齐 token。"""
    if bench == "ceval":
        conversation = [
            {
                "role": "system",
                "content": "请从question的A,B,C,D四个选项中选择正确的选项。例如,标准答案:A。",
            },
            {
                "role": "user",
                "content": f"'question':{sample['question']},'A': {sample['A']}, "
                f"'B':{sample['B']}, 'C': {sample['C']},'D': {sample['D']}。",
            },
        ]
        return (
            tokenizer.apply_chat_template(
                conversation=conversation, add_generation_prompt=True, tokenize=False
            )
            + "正确答案是:"
        )
    question, choices = sample["question"], sample["choices"]
    choices_text = "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(choices))
    instruction = (
        "You are a multiple-choice question solver. "
        "Select the correct option and respond with only the letter A, B, C, or D."
    )
    conversation = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": f"{question}\n{choices_text}\n"},
    ]
    return (
        tokenizer.apply_chat_template(
            conversation=conversation, add_generation_prompt=True, tokenize=False
        )
        + "The answer is: "
    )


def load_samples(bench, split):
    """返回 [{sample, subject, answer_idx}],subject 逐一下载(缓存已预热)。"""
    from datasets import load_dataset

    repo = "ceval/ceval-exam" if bench == "ceval" else "cais/mmlu"
    out = []
    for subj in subject_list(bench):
        try:
            ds = load_dataset(repo, subj, split=split)
        except Exception as e:
            print(f"跳过 {subj}: {e}", flush=True)
            continue
        for s in ds:
            if bench == "ceval":
                ans = str(s.get("answer", "")).strip()
                if ans not in LETTERS:
                    continue  # ceval test 无答案;val 有
                idx = LETTERS.index(ans)
            else:
                idx = int(s["answer"])
                if idx >= 4:
                    continue
            out.append({"sample": s, "subject": subj, "answer_idx": idx})
    return out


def limit_per_subject(samples, limit):
    if not limit:
        return samples
    by_subj = {}
    for s in samples:
        by_subj.setdefault(s["subject"], []).append(s)
    return [x for subj in by_subj.values() for x in subj[:limit]]

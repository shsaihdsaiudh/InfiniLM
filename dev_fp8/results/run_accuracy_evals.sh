#!/bin/bash
# A 组精度评测(logprob MCQ):C-Eval(val,1346) + MMLU(test,14042) 全量 × {BF16, FP8}
# 用法: bash /root/fp8/run_accuracy_evals.sh
set -x
source /root/fp8/fp8_env.sh
export HF_HUB_OFFLINE=1
cd /root/fp8

FP8=/data/huggingface_home/hub/models--Qwen--Qwen3-8B-FP8/snapshots/220b46e3b2180893580a4454f21f22d3ebb187d3
BF16=/data/huggingface_home/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
OUT=/root/fp8/eval_logs

run_eval() {
  local tag=$1 model=$2 bench=$3 split=$4
  echo "===== EVAL ${tag} ${bench} ${split} ====="
  timeout 7200 python3 fp8_mcq_eval.py \
    --model "$model" --bench "$bench" --split "$split" \
    --out "$OUT/${bench}_${tag}.json"
  echo "===== EVAL ${tag} ${bench} DONE rc=$? ====="
}

run_eval fp8  $FP8  ceval val
run_eval bf16 $BF16 ceval val
run_eval fp8  $FP8  mmlu test
run_eval bf16 $BF16 mmlu test

echo ACCURACY_EVALS_ALL_DONE

#!/bin/bash
# FP8 vs BF16 decode 性能对比 —— 同配置连跑 bench.py
# 用法: bash fp8_perf_bench.sh
set -x
source /root/fp8/fp8_env.sh
export HF_HUB_OFFLINE=1
cd /root/fp8/InfiniLM

FP8=/data/huggingface_home/hub/models--Qwen--Qwen3-8B-FP8/snapshots/220b46e3b2180893580a4454f21f22d3ebb187d3
BF16=/data/huggingface_home/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218

run() {
  local tag=$1 model=$2 bs=$3 ilen=$4 olen=$5
  echo "===== PERF ${tag} bs=${bs} in=${ilen} out=${olen} ====="
  timeout 900 python3 examples/bench.py --model "$model" \
    --batch-size "$bs" --input-len "$ilen" --output-len "$olen" \
    --enable-paged-attn --warmup 2>&1 | \
    grep -E "Prefill TTFT|Decode  Avg ITL|load weights over|PERF"
}

run FP8  $FP8  1 128 256
run FP8  $FP8  16 128 256
run BF16 $BF16 1 128 256
run BF16 $BF16 16 128 256

echo PERF_BENCH_DONE

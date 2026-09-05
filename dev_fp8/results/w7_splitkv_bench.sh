#!/bin/bash
# W7 split-kv 性能验证:FP8 KV decode(split=auto,带 DEBUG_SPLITS 决策打印)
# 对照 1:KV BF16(同会话重测基线);对照 2:FP8 + SPLITKV=0(应回到 W6 数字)
# 每个 run 全量日志落 /root/fp8/eval_logs/w7_bench_*.log,stdout 只保留关键行
source /root/fp8/fp8_env.sh
export HF_HUB_OFFLINE=1
cd /root/fp8/InfiniLM

FP8=/data/huggingface_home/hub/models--Qwen--Qwen3-8B-FP8/snapshots/220b46e3b2180893580a4454f21f22d3ebb187d3
LOGDIR=/root/fp8/eval_logs
mkdir -p "$LOGDIR"

run() {
  local tag=$1 kvdtype=$2 bs=$3 ilen=$4 olen=$5; shift 5
  local log=$LOGDIR/w7_bench_${tag}_bs${bs}_in${ilen}.log
  echo "===== PERF ${tag} bs=${bs} in=${ilen} out=${olen} ====="
  timeout 900 python3 examples/bench.py --model "$FP8" \
    --batch-size "$bs" --input-len "$ilen" --output-len "$olen" \
    "$@" --enable-paged-attn --warmup >"$log" 2>&1
  grep -E "Prefill TTFT|Decode  Avg ITL|PERF|num_splits|split" "$log"
}

for ilen in 1024 4096 16384; do
  for bs in 1 8; do
    run KVBF16 bf16 $bs $ilen 128
    INFINIOP_FLASH_DEBUG_SPLITS=1 run KVFP8_AUTO fp8 $bs $ilen 128 --kv-cache-dtype fp8
    INFINIOP_FLASH_DECODE_SPLITKV=0 run KVFP8_NOSPLIT fp8 $bs $ilen 128 --kv-cache-dtype fp8
  done
done

echo W7_BENCH_DONE

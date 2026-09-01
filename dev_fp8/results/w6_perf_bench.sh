#!/bin/bash
# W6 KV cache FP8 性能评测:长上下文 decode ITL 对比(均在 FP8 权重模型上)
source /root/fp8/fp8_env.sh
export HF_HUB_OFFLINE=1
cd /root/fp8/InfiniLM

FP8=/data/huggingface_home/hub/models--Qwen--Qwen3-8B-FP8/snapshots/220b46e3b2180893580a4454f21f22d3ebb187d3

run() {
  local tag=$1 kvdtype=$2 bs=$3 ilen=$4 olen=$5
  echo "===== PERF ${tag} bs=${bs} in=${ilen} out=${olen} ====="
  if [ "$kvdtype" = "bf16" ]; then
    timeout 900 python3 examples/bench.py --model "$FP8" \
      --batch-size "$bs" --input-len "$ilen" --output-len "$olen" \
      --enable-paged-attn --warmup 2>&1 | \
      grep -E "Prefill TTFT|Decode  Avg ITL|PERF"
  else
    timeout 900 python3 examples/bench.py --model "$FP8" \
      --batch-size "$bs" --input-len "$ilen" --output-len "$olen" \
      --kv-cache-dtype fp8 --enable-paged-attn --warmup 2>&1 | \
      grep -E "Prefill TTFT|Decode  Avg ITL|PERF"
  fi
}

for ilen in 1024 4096 16384; do
  for bs in 1 8; do
    run KVBF16 bf16 $bs $ilen 128
    run KVFP8  fp8 $bs $ilen 128
  done
done

echo W6_PERF_BENCH_DONE

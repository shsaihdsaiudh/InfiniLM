#!/bin/bash
# B 组:vLLM 对照(精度同协议 + 性能同口径)
# 用法: bash /root/fp8/run_vllm_evals.sh
set -x
export HF_HOME=/data/huggingface_home
export HF_HUB_CACHE=/data/huggingface_home/hub
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE=1
# sm_120 + 系统 CUDA 12.8 下 flashinfer JIT 编译不过(cccl 与 nvcc 主版本不匹配);
# 关掉 flashinfer sampler、注意力走 FLASH_ATTENTION 绕开
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTENTION
# vllm 脚本放独立目录:避免 /root/fp8/tvm_ffi(调试残留)遮蔽 venv 里的 apache-tvm-ffi
cd /root/fp8/vllm_eval

PY=/root/fp8/venv-vllm/bin/python
FP8=/data/huggingface_home/hub/models--Qwen--Qwen3-8B-FP8/snapshots/220b46e3b2180893580a4454f21f22d3ebb187d3
BF16=/data/huggingface_home/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
OUT=/root/fp8/eval_logs

run_mcq() {
  local tag=$1 model=$2 bench=$3 split=$4
  echo "===== VLLM_EVAL ${tag} ${bench} ${split} ====="
  timeout 7200 $PY /root/fp8/vllm_eval/vllm_mcq_eval.py \
    --model "$model" --bench "$bench" --split "$split" \
    --out "$OUT/${bench}_${tag}.json"
  echo "===== VLLM_EVAL ${tag} ${bench} DONE rc=$? ====="
}

run_mcq vllm_fp8  $FP8  ceval val
run_mcq vllm_bf16 $BF16 ceval val
run_mcq vllm_fp8  $FP8  mmlu test
run_mcq vllm_bf16 $BF16 mmlu test

echo "===== VLLM_PERF fp8 ====="
timeout 3600 $PY /root/fp8/vllm_eval/vllm_perf_bench.py --model $FP8  --tag vllm_fp8
echo "===== VLLM_PERF bf16 ====="
timeout 3600 $PY /root/fp8/vllm_eval/vllm_perf_bench.py --model $BF16 --tag vllm_bf16

echo VLLM_EVALS_ALL_DONE

#!/bin/bash
# W6 KV cache FP8 全量精度评测链(顺序跑,避免 GPU 争用)
source /root/fp8/fp8_env.sh
export HF_HUB_OFFLINE=1
M=/data/huggingface_home/hub/models--Qwen--Qwen3-8B-FP8/snapshots/220b46e3b2180893580a4454f21f22d3ebb187d3
L=/root/fp8/eval_logs
cd /root/fp8/InfiniLM

run() { # name, cmd...
  local name=$1; shift
  echo "===== RUN $name $(date '+%H:%M:%S') ====="
  "$@" 2>&1 | tail -6
}

run ppl_kvfp8_full python3 dev_fp8/results/fp8_ppl_eval.py --model $M --out $L/ppl_kvfp8_full.json --kv-cache-dtype fp8
run ppl_kvbf16_paged_full python3 dev_fp8/results/fp8_ppl_eval.py --model $M --out $L/ppl_kvbf16_paged_full.json --kv-cache-dtype bfloat16
run ceval_kvfp8 python3 dev_fp8/results/fp8_mcq_eval.py --model $M --bench ceval --split val --out $L/ceval_kvfp8.json --kv-cache-dtype fp8
run ceval_kvbf16_paged python3 dev_fp8/results/fp8_mcq_eval.py --model $M --bench ceval --split val --out $L/ceval_kvbf16_paged.json --kv-cache-dtype bfloat16
run mmlu_kvfp8 python3 dev_fp8/results/fp8_mcq_eval.py --model $M --bench mmlu --split test --out $L/mmlu_kvfp8.json --kv-cache-dtype fp8
run mmlu_kvbf16_paged python3 dev_fp8/results/fp8_mcq_eval.py --model $M --bench mmlu --split test --out $L/mmlu_kvbf16_paged.json --kv-cache-dtype bfloat16

echo "W6_ACCURACY_CHAIN_DONE"

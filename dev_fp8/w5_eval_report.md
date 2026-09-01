# W5 报告 —— C-Eval/MMLU 精度补齐 + vLLM 同权重对照

> 日期:2026-08-31
> 硬件:NVIDIA RTX 5090(sm_120,32GB),driver 610.43.02
> 模型:Qwen3-8B-FP8(220b46e3)vs Qwen3-8B(b968826d);vLLM 0.28.0(torch 2.13.0+cu130)
> 复现:`results/fp8_mcq_eval.py`(InfiniLM)/ `results/vllm_mcq_eval.py`(vLLM)/ `results/vllm_perf_bench.py`,runner `results/run_accuracy_evals.sh` / `run_vllm_evals.sh`,原始数据 `results/{ceval,mmlu}*.json`、`results/vllm_evals.log`、`results/perf_bf16_bs48.log`
> 服务器工具:`results/fp8_ssh.py`(密码认证封装,密码不落仓库)、`results/download_eval_datasets.py`

## 1. 评测方法(为什么不用 test/bench 生成式框架)

`test/bench/test_benchmark.py` 的 `extract_answer_ceval` 只检查输出前 2 个字符,而 Qwen3 聊天模型输出为 "正确答案是:B",生成式口径下大量样本解析失败且 `--max-new-tokens 8` 截断过早。改为 **logprob MCQ**(lm-eval-harness 口径):每题一次 prefill,取末位 logits 在 {A,B,C,D} 四个字母 token 上比 logprob 取 argmax,不生成文本。提示词与 test/bench 框架逐字一致(C-Eval 应答 cue 补全角冒号对齐 token 边界)。vLLM 侧无 logits 通道,用 top-20 首 token logprobs 打分,提示词/数据/判分完全相同。

正确性校验:KV cache 复用模式与逐题 reset 模式前 50 题 argmax 完全一致(`--validate`);两引擎 BF16 结果互相印证(见下)。

## 2. 精度结果(全量:C-Eval val 1346 题,MMLU test 14042 题)

| 引擎 | 权重 | C-Eval | MMLU |
|---|---|---|---|
| **InfiniLM** | FP8 | **74.74%** (1006/1346) | **69.47%** (9755/14042) |
| **InfiniLM** | BF16 | 74.37% (1001/1346) | 69.68% (9784/14042) |
| vLLM | FP8 | 73.77% (993/1346) | 69.36% (9740/14042) |
| vLLM | BF16 | 74.37% (1001/1346) | 69.56% (9768/14042) |

- **提案目标达成**:InfiniLM FP8 vs BF16 偏差 C-Eval **+0.37pt**、MMLU **−0.21pt**,均在 ±0.5pt 内(叠加 PPL +0.082%,精度三指标全部达标)
- **引擎对照**:BF16 档两引擎 C-Eval 逐题一致(74.37% 同分)、MMLU 差 0.12pt → 实现 parity;FP8 档 MMLU 差 0.11pt、C-Eval 差 0.97pt(n=1346 时 1σ≈1.2pt,在噪声内)
- 解读:本实现"反量化后 FP32 累加"在数值上比 vLLM 的真 FP8 tensor-core GEMM 更贴近 BF16——W4 的逐 token 一致性与这里的精度结果互为印证

## 3. 性能对照(decode tok/s,input 128 / output 256)

InfiniLM 侧为 `examples/bench.py --enable-paged-attn --warmup`;vLLM 侧离线 API 两点法隔离 decode(ITL=(T₂₅₆−T₆₄)/192),关 prefix caching,贪婪采样。

| decode bs | InfiniLM FP8(融合) | InfiniLM BF16 | vLLM FP8 | vLLM BF16 |
|---|---|---|---|---|
| 1 | **142.96** | 92.34 | 157.02 | 98.35 |
| 4 | 354.32 | 350.64(本次补测) | — | — |
| 8 | 411.75 | 686.52(本次补测) | **1224.18** | 733.28 |
| 16 | 428.54(naive 路由) | 1304.1 | **2326.55** | 1421.73 |

- **bs=1:自研融合 kernel 达 vLLM FP8 的 91%**(142.96 vs 157.02);FP8/BF16 相对收益 1.55×,与 vLLM 的 1.60× 同量级 → 小 batch 已达准 SOTA
- **bs=4 是反超拐点**(354.32 vs BF16 350.64,101%);bs=8 对 BF16 只剩 60%
- **bs≥8 差距量化**:vLLM 的 FP8 tensor-core kernel 在 bs=8/16 是现路由的 3.0×/5.4×——这就是 W4 遗留"M_TILE tensor-core 化"的确切收益空间
- vLLM FP8 prefill(bs=1 9479 tok/s)也显著高于 InfiniLM naive(2920),prefill 融合的价值上调
- 附带:vLLM 环境在 sm_120 的坑已记录——系统 CUDA 12.8 导致 flashinfer JIT 失败(`compute_120f` 需 12.9+、自带 cccl 与 nvcc 13 不匹配),以 `VLLM_USE_FLASHINFER_SAMPLER=0` + `VLLM_ATTENTION_BACKEND=FLASH_ATTENTION` 绕开;pip `nvidia/cu13` 完整工具链可用(已搭 `/root/fp8/cuda13home` 备用)

## 4. 结论

W3/W4 遗留的评测缺口(C-Eval/MMLU、vLLM 对照、bs=4/8 BF16 基线)全部补齐。项目 #4 评判维度现状:

| 维度 | 状态 |
|---|---|
| 精度 | ✅ PPL +0.082%;C-Eval +0.37pt;MMLU −0.21pt;与 vLLM 同权重 parity |
| 性能 | ✅ decode bs=1 反超 BF16 1.55×(vLLM 的 91%);⚠️ bs≥8 落后 vLLM 3-5.4×(tensor-core 化空间) |
| 显存 | ✅ 权重 55.6%;融合路径无 BF16 瞬态 |
| 测试/数据完整度 | ✅ 算子级 7 形状 × 3 dtype、E2E 逐 token 对拍、PPL/C-Eval/MMLU 全量、vLLM 对照,脚本全部入库 |
| 硬件覆盖 | nvidia sm_120;metax 待做 |
| 易用性 | 复现脚本 + 报告齐全 |

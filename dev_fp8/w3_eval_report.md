# W3 评测报告 —— FP8 块量化 vs BF16(Qwen3-8B,RTX 5090)

> 日期:2026-08-27
> 分支:InfiniLM `feat/fp8-blockwise-quantization`(ff9fc39d)+ InfiniCore `feat/fp8-blockwise-dequantize`(e818120a,已合上游 48d51642)
> 硬件:NVIDIA RTX 5090(sm_120,32GB),driver 610.43.02,CUDA 13.3
> 模型:`Qwen/Qwen3-8B-FP8`(snapshot 220b46e3)vs `Qwen/Qwen3-8B`(snapshot b968826d)
> 复现:`dev_fp8/results/` 内脚本与原始输出(fp8_build_core.sh / fp8_build_lm.sh / fp8_env.sh / fp8_ppl_eval.py / fp8_perf_bench.sh)

## 1. 精度:wikitext2 PPL ✅ 达标(目标 偏差 < 1%)

协议:Salesforce/wikitext `wikitext-2-raw-v1` test 集;chunk=512 独立 prefill(与 `scripts/test_ppl.py` 语义一致,但改为进程内 `InferEngine.forward_raw` 直取 logits,因服务器版 inference_server 不支持 logprobs);NLL 在 float32 下累计。两模型 token/chunk 计数完全一致(293457 tokens / 2886 chunks)。

| 模型 | PPL | NLL 合计 |
|---|---|---|
| Qwen3-8B (BF16) | **18.8617** | 861922.34 |
| Qwen3-8B-FP8 | **18.8771** | 862162.06 |
| **相对偏差** | **+0.082%** | +0.028% |

原始数据:`results/ppl_fp8.json`、`results/ppl_bf16.json`。

## 2. 性能:decode 吞吐 ❌ 未达标(目标 ≥ BF16 的 80%)

协议:`examples/bench.py --enable-paged-attn --warmup`,input_len=128,output_len=256,paged KV;原始日志 `results/perf_bench.log`。

| 场景 | FP8 | BF16 | FP8/BF16 |
|---|---|---|---|
| decode bs=1 | 28.49 tok/s(ITL 35.09ms) | 92.34 tok/s(ITL 10.83ms) | **30.9%** |
| decode bs=16 | 427.44 tok/s | 1304.1 tok/s | **32.8%** |
| prefill bs=1 | 2919.95 tok/s | 7038.08 tok/s | 41.5% |
| prefill bs=16 | 9373.06 tok/s | 10359.5 tok/s | **90.5%** |
| 权重加载 | 2.69 s | 4.99 s | 快 1.86× |

**分析**:naive 路径(dequant→BF16 GEMM)在每个 decode step 对全部权重做 `fp8_blockwise_dequantize`,把 memory-bound 的 decode 变成"读 FP8 + 写 BF16 + 再读 BF16"的三倍带宽开销,bs=1/16 均只有 BF16 的 ~31-33%;prefill 随 batch 增大转为 compute-bound 后差距收敛到 90.5%。

这恰好是 W4 融合 FP8 GEMM(marlin `kFE4M3fn` 或 cuBLASLt FP8 block-scale)的量化立项依据:理论上 FP8 权重读取量只有 BF16 一半,融合路径 decode 应**反超** BF16;当前 naive 路径反而是其 1/3,中间有 ~6 倍的可挖掘空间。

## 3. 显存 ✅ 达标(目标 ≤ BF16 的 60%)

权重文件:FP8 8.9 GB vs BF16 16 GB = **55.6%**。运行时持久显存与文件量级一致(naive 路径的 BF16 权重缓冲为 per-linear 瞬态)。

## 4. 结论与后续

| W3 项 | 状态 |
|---|---|
| wikitext2 PPL < 1% | ✅ 0.082% |
| decode ≥ 80% BF16 | ✅ W4 融合路径达成(30.9% → 155%,见 w4_fused_gemm_report.md) |
| 显存 ≤ 60% | ✅ 55.6% |
| C-Eval/MMLU | ✅ W5 补齐:C-Eval +0.37pt / MMLU −0.21pt(见 w5_eval_report.md) |
| vLLM 同权重对照 | ✅ W5 补齐:精度 parity;decode bs=1 达 vLLM 91% |

**W4 优先级**:
1. 融合 FP8 GEMM 落地(先核实 marlin 块级 scale 支持,见提案 §6 风险 1)——decode 预期从 31% 提升到 ≥100%
2. C-Eval/MMLU 补齐 + vLLM 对照
3. metax 后端(加分项)

## 5. 复现步骤(5090 服务器,Gitee AI 容器)

```bash
# 0. 布局:/root/fp8/{InfiniCore,InfiniLM},隔离 INFINI_ROOT(不碰 /root/.infini 的 perf 产物)
#    注意:仓库 tarball 中 third_party/cutlass 等空 submodule 目录必须删除,
#    否则 xmake/nvidia.lua 自动探测会定义 ENABLE_CUTLASS_API 导致缺头文件编译失败
bash fp8_build_core.sh      # InfiniCore: xmake --nv-gpu=y --cpu=y --cuda_arch=sm_120 -k shared
bash fp8_build_lm.sh        # infinicore pybind + InfiniLM _infinilm
source fp8_env.sh           # INFINI_ROOT/LD_LIBRARY_PATH/PYTHONPATH/HF_*

# 1. 模型(hf-mirror,需 HF_HUB_DISABLE_XET=1 绕过 xet 401)
bash fp8_download_models.sh

# 2. PPL(每个模型约 8 分钟)
python3 fp8_ppl_eval.py --model <snapshot_path> --out ppl_<tag>.json

# 3. 性能(4 组约 10 分钟)
bash fp8_perf_bench.sh
```

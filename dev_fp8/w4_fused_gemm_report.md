# W4 报告 —— 自研 FP8 blockwise 融合 GEMM(decode 反超 BF16 1.55×)

> 日期:2026-08-28
> 交付:InfiniCore 新算子 `fp8_blockwise_gemm` + InfiniLM decode 路由
> 硬件:NVIDIA RTX 5090(sm_120,32GB),CUDA 12.8
> 前置:marlin 路线因 InfiniCore kernel 在 sm_120 不可用而阻塞(dev_fp8/marlin_sm120_issue.md),转自研

## 1. 结果(E2E,examples/bench.py,paged-attn,input 128 / output 256)

| decode | FP8 naive(W3) | **FP8 融合(W4)** | BF16 | 融合 vs naive | **融合 vs BF16** |
|---|---|---|---|---|---|
| bs=1 | 28.49 | **142.96 tok/s**(ITL 7.00ms) | 92.34 | **5.02×** | **1.55× ✅ 反超** |
| bs=4 | 110.21 | **354.32** | — | 3.22× | — |
| bs=8 | 218.1 | **411.75** | — | 1.89× | — |
| bs=16 | 427.44 | 428.54(naive 路由) | 1304.1 | 1.00× | 33% |

W3 目标(decode ≥ BF16 的 80%)由 30.9% → **155%** 达成并反超。

路由策略(`FP8Blockwise::forward`,实测拐点):M ≤ 8 走融合 kernel,M ≥ 9 与 prefill 走 naive(dequant + cuBLAS);`INFINILM_FP8_FUSED_GEMM=0` 可整体回退。

## 2. 算子设计(`fp8_blockwise_gemm`)

- 语义:`out[m,n] = Σ_k a[m,k] · fp8_e4m3_decode(q[n,k]) · scales[n/128, k/128]`,不实体化 BF16 权重;每 decode step 权重流量 ~6.9GB(naive 路径 ~40GB:读 FP8 + 写 BF16 + 读 BF16)
- kernel(decode 向 GEMV 形):warp-per-row(4 warps/block,grid N/4);lane 持 16B(uint4)FP8，寄存器内反量化，下一组预取流水；按 128-K 子块应用 scale,FP32 累加 + warp shuffle 归约;M_TILE ∈ {1,2,4,8,16} 模板,grid.y 覆盖 M
- CPU 参考实现 + infiniop/infinicore 全套(C API、C++ graph op、ctypes 注册)
- 校验:`test/infiniop/fp8_blockwise_gemm.py` 7 种形状 × F16/BF16/F32 全过(含 M=13/32、block 64×128/128×256、K 尾块)
- E2E 对拍:fused vs naive 贪心生成 3 prompt × 64 token **全部逐 token 一致**

## 3. 性能迭代记录(教训:微基准必须 HBM 驻留)

| kernel 版本 | op 级带宽(4096×4096) | E2E bs=1 |
|---|---|---|
| v1(uint32/lane,2 行/warp,TN=16) | 653 GB/s(L2 污染) | 70.16 tok/s(ITL 14.25ms) |
| v2(uint4+预取，同几何) | 1076 GB/s(HBM 轮替) | 同(L2 污染掩盖差异) |
| **v4(warp-per-row,TN=4,128 线程)** | **1317 GB/s**(纯流上限 1505 的 88%) | **142.96 tok/s(ITL 7.00ms)** |

关键发现:
- 首版微基准重复读同一 16MB 张量 → L2 驻留，测的是 L2 而非 HBM，优化方向被掩盖;改 400MB 轮替后差异才显形(教训记入 `results/fp8_tune.cu` 试验台)
- 决定性优化是**几何**(warp-per-row、小块多 block),不是加载宽度/预取
- bs=16 的 M_TILE=16 路径指令吞吐受限(~5-6 TFLOP/s),不如 cuBLAS,故路由回 naive;tensor-core 化是后续方向
- 锁协议类复杂机制(cp.async 流水线、split-K)在实测中均不及简单几何有效

## 4. 显存与加载

- 权重占用 8.9GB(BF16 的 55.6%),fuse 后**无** BF16 权重瞬态缓冲
- 权重加载 2.4s(BF16 5.0s)

## 5. 遗留

- bs ≥ 16 的 M_TILE tensor-core 化(目标追回 33%→≥80%)
- prefill 融合(M 大,naive 已达 90.5% BF16,优先级低)
- C-Eval/MMLU、vLLM 对照 ✅ W5 已补齐(见 w5_eval_report.md)
- metax 后端(提案加分项)

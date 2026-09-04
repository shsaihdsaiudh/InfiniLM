# W8: FP8 decode GEMM 的 M_TILE tensor-core 化(9 ≤ M ≤ 32)

> 状态:**GPU 实测验证通过(2026-09-04,RTX 5090 / CUDA 12.8)**。
> 构建期修复一处编译错误:`launch_mtile` 提取为函数时模板参数 `int M_TILE` 被漏掉
> (调用点均为两参形式,InfiniCore-fp8 工作树已修,待提交)。
> 算子对拍 21/21 通过;decode bs=16 吞吐 1794.6 tok/s = SIMT 的 **4.73×**、naive 的 **4.20×**;
> bs=32 2702.0 tok/s = naive 的 **3.30×**。实测明细见 §实测结果。

## 背景

W4 的实测结论:decode 融合 GEMM(`fp8_blockwise_gemm`,warp-per-row SIMT)在
M ≤ 8 时完胜 naive 路径(dequant + cuBLAS),从 M ≥ 16 起被反超(bs=16: 382 vs
427 tok/s),且对 vLLM 的 FP8 decode 落后 3–5.4×(bs≥8)。原因:SIMT kernel 每读
一个 FP8 权重字节要做 M_TILE 次 FMA,M 一涨就变成指令吞吐瓶颈(~5–6 TFLOP/s @
M=16),而权重流量并没有减少——这是计算结构问题,不是调参能解决的。

## 设计

把 decode GEMM 当成 skinny GEMM 上 tensor core:

- **指令**:`mma.m16n8k16.row.col`(BF16/F16 两态,`MmaTraits<T>` 模板;F32 不动,
  仍走 SIMT M_TILE fallback)。
- **CTA 划分**:4 warp × n8 = N_TILE 32;`M_BLOCKS ∈ {1, 2}` 模板覆盖 M ≤ 32
  (m16 的 A 行块数)。grid = (N/32, M/(16·M_BLOCKS))。
- **K 流水**:按 128 分块(恰为一个 scale 子块,`block_k % 128 == 0` 由
  `Fp8BlockwiseGemmInfo` 保证),全局 → 寄存器 staging → smem 双缓冲,每 chunk 一个
  `__syncthreads()`。不用 `cp.async`(sm_120 上收益存疑,先求正确)。
- **scale promote**:每个 128-K chunk 的 mma partial 立即用块 scale 提升——
  `c_fin = fmaf(s, c_part, c_fin)`,FP32 累加,与 SIMT 路径数值结构一致。
- **FP8 decode bit-trick**(核心技巧):e4m3 码位直接摆进 BF16/F16 位型,
  解码结果是真值 × 2⁻¹²⁰(BF16)/ 2⁻⁸(F16),这个 2 的幂折进 promote 的 scale
  (`scale * kDecodeScale`),mma 输入逐位精确,**无逐元素乘法**:
  - BF16: `(c & 0x7f) << 4 | (c & 0x80) << 8`(低字节;高字节对称)
  - F16: `(c & 0x7f) << 7 | (c & 0x80) << 8`
  - 对 denormal e4m3 同样成立(模拟器全码位验证);NaN 码 0x7F 不特判——
    编码器饱和在 448,量化权重不含该码(kernel 注释已注明)。
- **smem**:sA `M_BLOCKS*16 × 136`(272B 行距消 bank 冲突)、sW `32 × 144`。
- **尾块**:M/N 越界行加载时填零、store 时丢弃,任意 M/N 尾块都正确。

## 分派

算子内部(`fp8_blockwise_gemm_nvidia.cu::launch`):

| 条件 | 路径 |
|---|---|
| F16/BF16 且 9 ≤ M ≤ 32 | **mma kernel**(新增) |
| M ≤ 8 | SIMT M_TILE ∈ {1,2,4,8}(维持 W4 结论) |
| 其余(F32 / M > 32) | SIMT M_TILE = 16(naive 由 InfiniLM 侧路由) |
| `INFINIOP_FP8_GEMM_MMA=0` | 强制 SIMT(A/B 回归开关) |

InfiniLM 侧(`csrc/layers/quantization/fp8_blockwise.cpp`):fused 路由上限从
`m <= 8` 放宽到 `m <= 32`,注释同步更新(原注释中"M ≥ 16 fused 输给 naive"是
SIMT 的实测结论,已被 mma 路径覆盖);`INFINILM_FP8_FUSED_GEMM=0` 行为不变。

mma 路径依赖的形状约束(`K % 128 == 0`、`block_k % 128 == 0`、`block_n % 16 == 0`)
全部已在 `Fp8BlockwiseGemmInfo::create` 强制,不满足时在 descriptor 创建阶段就
报错,不会静默走错路径。

## 已完成的验证(无 GPU)

1. **Python 精确模拟**(`dev_fp8/results/fp8_mma_sim.py`,**ALL PASS**):
   - bit-trick 对全部 256 个非 NaN 码位逐位精确(BF16/F16 两态);
   - 10 组 case 与双精度参考 GEMM 全等:M ∈ {1,9,12,13,16,17,24,31,32}、
     N % 32 = 16 尾块、block_n ∈ {16,32,64}、block_k ∈ {128,256}、
     M_BLOCKS ∈ {1,2};
   - 模拟器按 kernel 逻辑逐拍复刻(fragment 索引、双缓冲、promote 时序),
     期间发现并修掉的是模拟器自身的 mb 间共享 A_tile bug,kernel 无需改。
2. **静态终查**:mma fragment 索引对照 PTX ISA m16n8k16 布局逐行核对
   (A: a0..a3 = (g,2t)/(g+8,2t)/(g,2t+8)/(g+8,2t+8);B: b0/b1 = k 2t,2t+1 /
   2t+8,2t+9 @ 列 g;C: c0..c3 = (g,2t)/(g,2t+1)/(g+8,2t)/(g+8,2t+1));
   双缓冲竞争分析(每 chunk 单 sync,store 写对侧 buffer 安全);
   16B/4B/2B 对齐链(uint4 A 需 K%8、uint4 W 需 K%16,均被 K%128 覆盖)。
3. 项 2 的 split-kv 模拟器(`fp8_splitkv_sim.py`)同步回归 **ALL PASS**。

## 实测结果(2026-09-04,RTX 5090,CUDA 12.8 全量构建)

- **算子对拍** `test/infiniop/fp8_blockwise_gemm.py --nvidia`(torch fp32 参考):
  7 形状 × {F16,BF16,F32} = 21 case **全部通过**,MMA ON/OFF(`INFINIOP_FP8_GEMM_MMA=0`)
  两轮皆过。M=13/16/32 × F16/BF16 命中 mma 路径,M ≤ 8 维持 SIMT,F32 走 M_TILE 回退——
  mma fragment 布局理解经 GPU 独立验证无误(模拟器共享假设的风险项退役)。
- **W7 顺带回归**:paged_caching / paged_attention / paged_attention_prefill 三算子
  `--nvidia` 全过(split-kv 合入后 BF16/F16/FP8 case 无回归)。

**端到端 decode(examples/bench.py,Qwen3-8B-FP8 权重,paged attn,in=128/out=256;同 W4 基线配置)**:

| 配置 | bs=8 | bs=16 | bs=32 |
|---|---|---|---|
| mma(新路径) | 408.7 tok/s | **1794.6 tok/s** | **2702.0 tok/s** |
| SIMT fused(`MMA=0`) | — | 379.7(W4 基线 382 ✓) | — |
| naive(`FUSED_GEMM=0`) | — | 427.0(W4 基线 427 ✓) | 817.6 |
| **mma / naive** | — | **4.20×** | **3.30×** |
| **mma / SIMT** | — | **4.73×** | — |

bs=8(M=8)按分派留在 SIMT M_TILE=8,408.7 ≈ W4 的 412,符合预期(不属 mma 区间)。
各配置 TTFT 不变(bs=16 均 ~218ms;prefill M>32 仍走 naive),decode ITL bs=16 从
42.14ms(SIMT)降到 8.92ms。bs=16 贪婪生成(in=128/out=48,temperature=0)mma 与
naive 输出均连贯,开局一致后仅个别措辞分叉(GEMM 数值路径不同,语义等价)。

- **对 vLLM 差距**:W4 时 bs≥8 落后 3–5.4× 的 GEMM 部分已被 mma 路径消除性收窄
  (bs=16 fused 吞吐 382 → 1794.6);剩余差距在 attention/runtime,未复测 vLLM parity。
- **cuBLASLt spike**:当前工具链 CUDA **12.8** 头文件无 block-scale API(需 ≥12.9),
  仅 per-tensor FP8 可跑(algos=8,非 block scaling)。按既定判定标准,
  **W8A8 cuBLASLt 路线在当前工具链不可行,自研 mma 即为 bs≥8 decode 的终点方案**;
  若未来升级 CUDA ≥ 12.9 工具链可重跑 `fp8_cublaslt_spike.cu` 复核。
- **注意**:服务器容器重启后 `/usr/local/cuda` 指回镜像自带 12.8;经查安装在
  `/root/fp8/.infini` 的全部历史产物均为 12.8 编译(188 个对象 `.comment` 一致),
  本次验证构建与 W3–W6 全部结果的工具链一致。`cuda13home` 实为 vLLM venv 的
  cu13 pip 包 shim(nvcc 13.3 + runtime 头 13.0 混装,CCCL 版本校验不过,不能用于全量构建)。

## 已完成的服务器验证清单(原"服务器待办")

1. ✅ 代码同步(tarball:InfiniCore 17 文件 = W7 透传 + W7 split-kv + W8 mma;
   InfiniLM `fp8_blockwise.cpp`),同步后 md5 全树比对一致。
2. ✅ 构建:`xmake f -c --cuda=/usr/local/cuda --cuda_arch=sm_120 ...`(12.8),
   修掉一处编译错误(`launch_mtile` 漏 `int M_TILE` 模板参数)后全链通过。
3. ✅ 算子正确性 + A/B(上文)。
4. ✅ 端到端 bench(上文表格)。
5. ✅ cuBLASLt spike(上文结论)。

## 风险与回退

- **数值**(已退役):mma fragment 布局经 GPU 算子对拍独立验证(21/21),
  promote 结构与 SIMT 路径数学等价,bit-trick 输入逐位精确。
- **性能**(已退役):mma 路径实测 4.20×/4.73×(bs=16)、3.30×(bs=32)大胜 naive
  与 SIMT,无需回退;回退开关保留:`INFINIOP_FP8_GEMM_MMA=0` + InfiniLM 路由
  改回 `m <= 8`,一行改动。
- **cuBLASLt 路线**(已裁决):12.8 工具链头文件无 block-scale API(需 ≥12.9),
  当前不可行;升级工具链后可重跑 spike 复核。

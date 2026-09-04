# W8: FP8 decode GEMM 的 M_TILE tensor-core 化(9 ≤ M ≤ 32)

> 状态:**代码与 CPU 模拟验证已完成,未经 GPU 构建/实测**(服务器不可用期间开发)。
> 所有"预期"数字均为推断,待服务器验证后回填。验证清单见文末。

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

## 预期收益(待实测)

- bs=8..32 decode 的 GEMM 从指令吞吐瓶颈切到 tensor core,权重流量不变,
  单算子预期数倍于 SIMT M_TILE=16 路径;端到端以 bs=8/16 tok/s 回填。
- 对 vLLM 的 3–5.4× 差距中,GEMM 部分预计显著收窄(其余在 attention/runtime)。

## 服务器待办(恢复后按序执行)

```bash
# 0. 同步代码
python3 dev_fp8/results/fp8_ssh.py push
# 1. 构建 InfiniCore(xmake, sm_120)
python3 dev_fp8/results/fp8_ssh.py run "bash /root/fp8/InfiniCore/dev_fp8_sync/fp8_build_core.sh"
# 2. 算子正确性:M=13/16/32 自动命中 mma 路径,F32 用例覆盖 SIMT 回退
python3 dev_fp8/results/fp8_ssh.py run "cd /root/fp8/InfiniCore && python3 test/infiniop/fp8_blockwise_gemm.py --nvidia"
# 3. A/B 对照:mma 开关
#    INFINIOP_FP8_GEMM_MMA=0 python3 test/infiniop/fp8_blockwise_gemm.py --nvidia
# 4. 端到端 bench:bs=8/16 对比 W4 基线(382 tok/s @ bs=16 fused-SIMT / 427 naive)
python3 dev_fp8/results/fp8_ssh.py run "bash /root/fp8/InfiniLM/dev_fp8_sync/fp8_perf_bench.sh"
# 5. cuBLASLt block-scale spike(决定 W8A8 cuBLASLt 路线是否可行)
python3 dev_fp8/results/fp8_ssh.py run "cd /root/fp8 && nvcc -O2 -arch=native dev_fp8_sync/fp8_cublaslt_spike.cu -o /tmp/fp8_cublaslt_spike -lcublasLt && /tmp/fp8_cublaslt_spike"
```

(具体脚本路径以 `fp8_ssh.py push` 的同步布局为准;build 若报编译错误,先把
`fp8_blockwise_gemm_nvidia.cu` 的 mma 段错误日志拉回本地修。)

## 风险与回退

- **数值**:promote 结构 = SIMT 路径的逐 chunk scale 累加,数学等价;bit-trick
  输入逐位精确。风险点只剩 mma fragment 布局理解错误(模拟器已按同构逻辑覆盖,
  但模拟器与 kernel 共享同一份索引假设——GPU 实测是第一道独立验证)。
- **性能**:若 mma 路径实测仍输 cuBLAS naive(例如 K 流水被 sync 拖住),回退 =
  `INFINIOP_FP8_GEMM_MMA=0` + InfiniLM 路由改回 `m <= 8`,一行改动。
- **cuBLASLt 路线**:`fp8_cublaslt_spike.cu` 若显示 sm_120 放行 BLK128x128/VEC128
  block scaling,则 W8A8(在线量化激活 + cuBLASLt)可能成为大 batch 的更优解,
  届时另立工作项评估;若 algos=0(Hopper 限定),路线否决,自研 mma 即是终点。

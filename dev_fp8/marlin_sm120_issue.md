# InfiniCore Marlin GEMM 在 sm_120 (Blackwell 消费级) 上不可用 —— 最小复现与调查记录

> 日期:2026-08-28
> 影响组件:`src/infiniop/ops/gptq_marlin_gemm`(sgl_kernel marlin 移植)
> 受影响架构:sm_120(RTX 5090 / RTX 5060 均复现);sm_89 未验证
> 结论:**kernel 在 sm_120 上对所有位宽(4bit/8bit/FP8)均不正确** —— CUDA 12.8 构建死锁、CUDA 13.2 构建输出全错;非调用方问题

## 1. 前置条件:算子需要 TVM-FFI 才能编译出真实 kernel

`nvidia/gptq_marlin_gemm_nvidia.cu` 的整个 `calculate()` 被 `#if defined ENABLE_TVM_API` 包裹(xmake/nvidia.lua 仅在 `TVM_ROOT` 非空时定义该宏并添加 `tvm/ffi` 头)。**不设 TVM_ROOT 时算子静默空转**(返回 SUCCESS 但不启动任何 kernel),InfiniLM 层无报错、输出垃圾。这是一个隐蔽陷阱,建议至少在 calculate 的 `#else` 分支返回 `INFINI_STATUS_NOT_IMPLEMENTED`。

TVM-FFI 头可直接取自 pip 包 `apache-tvm-ffi`(布局正好匹配 `TVM_ROOT/{include,3rdparty/dlpack/include}`)。

## 2. 症状矩阵(同一算子、同一输入)

| 环境 | 工具链 | 现象 |
|---|---|---|
| RTX 5090 服务器 | CUDA 12.8 (`SGL_ARCH_BLACKWELL_OR_GREATER=0`) | **kernel 死锁**(launch 返回成功,synchronize 永不返回;`use_atomic_add` 与 lock-barrier 两条归约路径都死锁) |
| RTX 5060 本机 | CUDA 13.2 (`SGL_ARCH_BLACKWELL_OR_GREATER=1`) | kernel 完成但**输出全错**(下表) |

输出正确性(CUDA 13.2,相对误差 mean\|c-ref\|/mean\|ref\|,应 <0.04):

| 配置 | 结果 |
|---|---|
| uint4b8, group=128, act_order=True(**仓库官方测试原样配置**) | max_diff = **0.998** |
| uint8b128, group=128 | rel ≈ **1.00** |
| float8_e4m3fn, group=128(128×128 块 scale 扩展为 [K/128,N]) | rel ≈ **1.00** |
| float8_e4m3fn, per-channel | rel ≈ **1.00** |
| dtype | F16 / BF16 均错 |

compute-sanitizer synccheck 下死锁消失(串行化执行可以跑完,但输出同样全错),`ERROR SUMMARY: 0 errors` —— 无 barrier 用法错误,提示**锁协议存在时序竞态或代码生成问题**。

## 3. 复现

```bash
# 环境:INFINI_ROOT 指向带 ENABLE_TVM_API 构建的 InfiniCore(TVM_ROOT=<tvm_ffi 包路径>)
cd InfiniCore/test/infiniop
INFINI_ROOT=~/.infini-fp8 LD_LIBRARY_PATH=~/.infini-fp8/lib python3 gptq_marlin_gemm_smoke.py --nvidia
# 仓库官方用例(uint4b8, g128, act_order):max_diff = 0.998047 (要求 <0.04)
```

- `marlin_sm120_repro/gptq_marlin_gemm_smoke.py` —— 官方测试裁剪版(单配置 + max_diff 打印)
- `marlin_sm120_repro/fp8_marlin_debug.py` —— FP8(kFE4M3fn)+ 128×128 块 scale 的最小用例(复用官方布局工具,含 create 空描述符 workaround 与 workspace 清零)
- `marlin_sm120_repro/int8_marlin_control.py` / `int4_marlin_control.py` —— 8bit/4bit 对照

## 4. 调查中已排除的调用侧因素

1. `Descriptor::create` 无条件解引用 `g_idx_desc->numel()`/`perm_desc->numel()` —— C API 传 NULL 会段错误(官方 python 测试正是如此传的,说明该算子从未在 NVIDIA 上被官方测试跑通过);需传空张量描述符。
2. `TestWorkspace` 默认填 1,而 marlin kernel 把 workspace 尾部当锁用 —— 必须清零(InfiniLM 侧 `GPTQMarlin::get_workspace` 已正确处理)。
3. `use_atomic_add` / `use_fp32_reduce` 各组合、per-channel 与 group=128、M=1..16 —— 结果不变。
4. torch/CUDA 上下文初始化正常;同机其他 infiniop 算子(含 FP8 块反量化)全部正常。

## 5. 怀疑方向(供上游/后续)

- `marlin_template.h` 的跨块锁协议(`barrier_acquire/release`、`wait_negative_and_add`,使用 `ld.global.acquire.gpu` / `red.relaxed.gpu`):sm_120 的调度/内存序时序与 sm_89/sm_100 不同,移植快照可能含已在上游修复的竞态;
- 或 ptxas 对 sm_120 的代码生成问题(CUDA 12.8 死锁 vs 13.2 错算,同一源码两种失败形态);
- 建议对照 sgl-kernel / vLLM 当前 marlin 版本 diff 本仓库快照(本快照源自 PR #996 时期)。

## 6. 对 InfiniLM FP8 块量化项目的影响

W4 融合 FP8 GEMM 的 marlin 路线(InfiniLM 侧转换代码已完成,`INFINILM_FP8_MARLIN=1` 可选开启)在本 issue 修复前不可用;默认路径保持 naive dequant+BF16 GEMM(PPL 已验证正确)。替代路线:自研 FP8 blockwise 融合 GEMM(decode 向,访存瓶颈分析见 dev_fp8/w3_eval_report.md §2)。

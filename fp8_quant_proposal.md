# FP8 块量化权重执行路径 — 立项草案

> 项目方向:训练营项目 #4(量化与低精度推理)
> 状态:W1–W3 已完成(W3 于 5090 复测:wikitext2 PPL 偏差 0.082% 达标;decode 31% BF16 未达标,差距已量化 → W4 融合 GEMM 依据,详见 dev_fp8/w3_eval_report.md);W4 性能迭代为后续工作
> 分支:feat/fp8-blockwise-quantization(配套 InfiniCore 分支 feat/fp8-blockwise-dequantize;worktree: ../InfiniLM-fp8)
> 日期:2026-08-23

## 1. 动机与定位

InfiniLM 已支持 GPTQ/AWQ(Marlin)、MXFP4、KV cache INT8,但 **FP8 权重量化完全缺失**(`csrc/layers/quantization/quantization_scheme.hpp` 的 QuantScheme 枚举中无任何 FP8 项)。FP8(E4M3,128×128 块量化)是当前两个重要权重的发布格式:

- **DeepSeek-V3/V4 主线权重**:FP8 E4M3 主权重
- **Qwen3 官方 FP8 系列**:Qwen3-8B-FP8 / 30B-A3B-FP8 等,`quant_method: "fp8"`, `weight_block_size: [128,128]`,激活动态量化

本交付同时是:

1. **独立的 #4 项目成分**:新增一种低精度格式支持,有明确精度/显存/性能指标
2. **求职叙事**:与 CUDA 阶段 low_precision_quant 项目(E4M3/E8M0 编解码、反量化 kernel)直接衔接

## 2. 现状调研(2026-08-23,均经源码核实)

### 2.1 InfiniLM 量化架构

| 层 | 位置 | 事实 |
|---|---|---|
| 抽象接口 | `csrc/layers/quantization/base_quantization.hpp:39` | `BaseQuantization`:`get_param_layout` / `forward` / `split_params` / `process_weights_after_loading` 四个扩展钩子 |
| 工厂 | `csrc/config/quant_config.cpp:14-27` | 按 `quant_method` 分派:`compressed-tensors` / `awq` / `gptq` / `quark`(→MXFP4),其余落 `NoneQuantization` |
| 方案注册 | `csrc/layers/quantization/quantization.hpp` | 头文件聚合式注册,加新方案加一行 include |
| 权重加载 | `python/infinilm/modeling_utils.py` | `preserve_dtype_suffixes`/`preserve_dtypes` 机制可按后缀/dtype 保留原始 dtype;本项目将其泛化为按 `quantization_config` 自动解析(见 §3) |
| 消费样例 | `csrc/layers/quantization/mxfp4.cpp:37` | 调 `infinicore::op::linear_mxfp4`,是新格式接入的最近参照 |

### 2.2 InfiniCore 算子事实(路径:行号 均可查)

**已有:**

- `INFINI_DTYPE_F8 = 11`(`include/infinicore.h:66`),语义为 E4M3;portable 编解码 `infiniopFp8E4m3Encode/Decode`(`src/infiniop/devices/nvidia/nvidia_kernel_common.cuh:34-59`,metax/moore 有同名别名)
- FP8 DSA 四件套:`fp8_indexer_quant` / `fp8_indexer_logits` / `fp8_mla_rmsnorm_cache` / `fp8_sparse_mla`(提交 `a045a445`,C+C+++pybind 全套,物理实现仅 nvidia)——属注意力/KV cache 侧,与本项目权重侧互补
- **Marlin kernel 内置 FP8 E4M3 权重类型** `kFE4M3fn`(`src/infiniop/ops/gptq_marlin_gemm/sgl_kernel/scalar_type.hpp:304,321`)及 FP8→FP16 反量化(`sgl_kernel/marlin/dequant.h:292-326`),经 `b_q_type_id` 运行时参数暴露(`include/infinicore/ops/gptq_marlin_gemm.hpp:13-17`);但 InfiniLM 侧 `gptq_marlin.cpp:65-77` 写死 `UINT4B8_ID`,FP8 marlin 路径未被使用
- MXFP4 三件套(`mxfp4_dequantize` / `linear_mxfp4` / `fused_moe_mxfp4`):cpu/nvidia/metax/moore 四平台,接入流程模板见 §2.3

**缺失(=本项目的工作):**

- 通用 FP8 quant/dequant 算子(现有 `quant`/`dequant` 仅 INT8:`include/infiniop/ops/quant/per_tensor_quant_int8.h` 等)
- FP8 输入的 GEMM:`scaled_mm` 族是 INT8(`include/infiniop/ops/int8_gemm.h:8-30`,per-row×per-col F32 scale);nvidia `gemm` 走 cuBLASLt 但计算精度只到 F32
- 128×128 块级 scale 语义全库不存在(FP8 权重侧;`fp8_sparse_mla` 的 per-128 通道 scale 是 KV cache 侧约定,可借鉴)

### 2.3 新增算子的工程模板(mxfp4 提交 `09123f74` / `ec5cf339`)

1. C API 头 `include/infiniop/ops/<op>.h`(Create/GetWorkspace/Calculate/Destroy)+ `include/infiniop.h` 聚合
2. infiniop 实现 `src/infiniop/ops/<op>/`:`info.h`(校验)/ `operator.cc`(后端分派)/ `<backend>/` 实现
3. C++ 层 `include/infinicore/ops/<op>.hpp` + `src/infinicore/ops/<op>/*.cc`(`INFINICORE_GRAPH_OP_*` 宏)
4. 可选 pybind(`src/infinicore/pybind11/ops.hpp` 注册)+ python 封装
5. 测试:`test/infiniop/<op>.py`(ctypes 签名注册)+ `test/infinicore/ops/<op>.py`
6. 构建零改动:`xmake.lua:625` glob 自动收编

反面教材:`scaled_mm_i8` pybind 文件存在但未注册进 `ops.hpp`,python 侧拿不到。

## 3. 交付边界

### 范围内

1. **InfiniCore 新算子 `fp8_blockwise_dequantize`**:E4M3 权重 `[M,N]` + 块 scale `[⌈M/128⌉,⌈N/128⌉]` → BF16/FP16。Elementwise,无 workspace。nvidia 后端起步;metax 后端为加分项(共享 cuda kernel 模式,参照 mxfp4_common)
2. **InfiniLM 新量化方案类** `FP8Blockwise`(QuantScheme 新增 `FP8_W8A16`):实现 `BaseQuantization` 四钩子;forward 第一阶段 = `fp8_blockwise_dequantize` + 现有 BF16 GEMM(正确性优先)
3. **工厂与加载泛化**:`quant_method == "fp8"` 分派;`preserve_dtype_suffixes` 机制泛化到所有带 quantization_config 的模型(scale 类张量保 dtype)
4. **端到端验证**:Qwen3-8B-FP8 官方权重在 InfiniLM 跑通
5. **性能迭代(第二阶段)**:dequant+GEMM 替换为 marlin FP8 路径(`kFE4M3fn`)或 cuBLASLt FP8;前提是先核实 marlin 的 scale 粒度支持(见 §6 风险)

### 范围外(明确排除)

- FP8 激活静态量化/W8A8 全 FP8 计算(留作后续;视 marlin/cuBLASLt 核实结果再评估)
- FP4 专家权重路径(走 MXFP4 类变体)
- KV cache FP8(DSA 四件套已覆盖稀疏注意力场景;通用 KV FP8 是另一个候选项目)
- 训练侧量化(只做推理加载与执行)

## 4. 可度量目标

| 维度 | 指标 | 目标 |
|---|---|---|
| 正确性 | Qwen3-8B-FP8 wikitext2 PPL vs BF16 基线 | 偏差 < 1%(对齐 vLLM FP8 水平) |
| 正确性 | C-Eval/MMLU(test/bench 现有框架) | 与 BF16 差 < 0.5 pt |
| 显存 | 8B 权重显存占用 | ≤ BF16 的 60%(约 9GB,可本机验证) |
| 性能(一阶段) | decode token/s vs 同模型 BF16 | ≥ 80% |
| 性能(二阶段) | decode token/s vs BF16 | 追平或超越;并对照 vLLM 同权重 |
| 工程 | infiniop/infinicore 测试 + InfiniLM CI | 全绿;benchmark 脚本入库可复现 |

## 5. 里程碑(对齐 9-20 截止)

| 周 | 内容 | 出口标准 |
|---|---|---|
| W1(8/23–8/30) | `fp8_blockwise_dequantize` 算子(nvidia)+ infiniop 级测试;loader 泛化 | 算子测试通过;Qwen3-8B-FP8 权重能按原 dtype 读出 |
| W2(8/31–9/6) | `FP8Blockwise` 类 + qwen3 接线,naive 版跑通 | 生成文本合理;逐层数值对拍通过 |
| W3(9/7–9/13) | 精度评测(ppl/ceval/mmlu)+ 性能基线 vs BF16/vLLM | 精度达标 |
| W4(9/14–9/20) | 性能迭代(marlin FP8 / cuBLASLt 核实后实施);metax 后端(若卡可用);文档与报告 | 性能目标达标或风险如实记录;报告完成 |

## 6. 风险与开放问题

1. **Marlin FP8 的 scale 粒度**:`kFE4M3fn` 路径通常配 per-tensor/per-channel scale,DeepSeek/Qwen 的 128×128 块级 scale 能否直接进 marlin 需读 `sgl_kernel/marlin` 源码核实 → 一阶段 dequant+BF16 GEMM 保底,此项只影响二阶段性能上限
2. **sm_120(RTX 5060 Ti)FP8 tensor core 开放程度**:cuBLASLt FP8 block-scale API 在消费级 Blackwell 的可用性待验证 → 同上分阶段保底
3. **Scale 数值语义**:DeepSeek 系 checkpoint 的 `weight_scale` 约定(scale 还是 scale_inv、E8M0 还是 F32)需逐字节核实;Qwen3-FP8 为 F32 scale + 动态激活量化,两者差异需在 `FP8Blockwise` 内参数化

## 7. 加分路径

- **国产平台**:`fp8_blockwise_dequantize` 的 metax 后端(mxfp4 三件套已有 cpu/nvidia/metax/moore 四平台先例,共享 cuda kernel 模式可直接照搬);若训练营提供昇腾环境,昇腾后端差异最大、加分最多
- **可复用基础设施**:loader 的 scale 保 dtype 泛化、块量化 scale 的校验工具,均可被后续 NVFP4 等新格式复用

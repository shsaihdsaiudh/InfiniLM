# DeepSeek-V4 适配 InfiniLM：项目立项分析与计划

**项目方向**：项目 #1，新模型与新架构适配

**目标模型**：`deepseek-ai/DeepSeek-V4-Flash-DSpark`

**硬截止时间**：2026-09-20
**开发周期**：2026-08-23 至 2026-09-20，共 29 天

---

## 1. 结论

这是一个**高风险、可冲刺、必须严格裁剪范围**的四周项目。

- 若目标是“支持完整 DSpark、百万上下文、连续批处理、性能达标并一次性形成可合入 PR”，难度为 **10/10，四周不可控**。
- 将结题目标限定为“完整 tiny 架构数值对拍 + 主模型真权重 batch=1、短上下文生成”，难度约为 **9/10**；在全职投入、第一周完成算子与权重格式 Go/No-Go 的前提下有实现可能。
- 上游 Ready-for-review PR 是冲刺目标。若实现需要同步修改 InfiniCore 或公共缓存接口，结题时允许交付配套 PR/WIP PR，不能以牺牲正确性换取表面上的单仓 PR。

本项目真正的主风险不是模型注册，而是以下四项同时存在：

1. checkpoint 中的专家权重是打包 FP4，不能整体反量化到 BF16 后运行；
2. V4 注意力不是现有 DeepSeek-V2 MLA 的直接变体，具有单 KV 头、partial RoPE、attention sink、query/output LoRA、DSA indexer 和多种缓存模式；
3. mHC 会把 hidden state 扩展为四路，并在每层两次执行 Sinkhorn 混合；
4. InfiniLM 当前加载器会在 remap 前转换浮点 tensor，可能破坏 FP8/E8M0 scale 的原始表示。

因此，**原生保留并消费 FP4/FP8 checkpoint 格式是主交付的一部分，不再列为可选性能优化**。

---

## 2. 可验收范围

### 2.1 结题必须完成

1. 新增 `deepseek_v4` 模型注册、配置解析、权重 remap 和模型组装；
2. tiny config 覆盖并对拍：
   - 单 KV 头注意力、partial RoPE、sink、query/output LoRA；
   - DSA indexer 和三种 attention/cache 路径；
   - mHC 四路残差与 Sinkhorn；
   - `sqrtsoftplus + noaux_tc` 路由、共享专家和 routed experts；
   - hash router 层；
3. 无损加载原始 FP8/FP4 权重与 scale，不在 remap 前错误转换 dtype；
4. V4-Flash-DSpark **主模型**在真权重下完成 batch=1、短上下文、1～8 token 的确定性生成；
5. 新增 `test/models/deepseek_v4/`，通过格式检查，并提供可复现命令、显存峰值和数值结果。

### 2.2 明确不纳入 9 月 20 日验收

- `mtp.*` 下的 DSpark 多阶段投机预测与接受/拒绝调度；
- 1M token 长上下文实测；
- continuous batching、prefix cache、长序列分页缓存的完整生产化；
- 多 batch、吞吐优化、kernel autotune；
- 非 CUDA 平台移植；
- V4-Pro 和 V4-Flash 的全量真权重验证。

裁掉 DSpark 投机层不改变基础语言模型 logits，但必须在文档和加载日志中明确：支持的是 **V4-Flash-DSpark checkpoint 的 base-model path**，不是完整 DSpark serving 能力。

### 2.3 冲刺目标

- 单请求下使用分页 state/cache，而不是仅能运行一次的临时 buffer；
- 与官方推理实现完成多 prompt、多 token logits/argmax 对比；
- InfiniLM 与必要的 InfiniCore 改动形成可评审的配套 PR。

---

## 3. 已核实的模型事实

### 3.1 checkpoint 规模与格式

[官方权重索引](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark/blob/main/model.safetensors.index.json)记录的总大小为 `166,878,536,440` bytes，约 **166.9 GB / 155.4 GiB**。48 个 shard 的 header 表明：

- routed expert 主权重占绝大多数，采用打包后的整数载体保存 FP4 数据；
- expert scale 使用 FP8 E8M0；
- attention、dense 等权重主要使用 FP8 E4M3；
- 还包含 mHC、compressor/indexer、hash router 与 `mtp.*` 权重。

2×96 GiB 只有约 36.6 GiB 的总余量，需要容纳 runtime buffer、activation、KV/state cache、通信 workspace 和 allocator 碎片。因此 2×96 GiB 只能作为 batch=1、短上下文的最低配置，且必须分片加载，不能先在 CPU/GPU 上构造完整 BF16 副本。

### 3.2 注意力

[官方配置](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark/blob/main/config.json)和参考实现显示，V4 attention 包含：

- `num_key_value_heads=1` 的单 KV 头；
- `head_dim=512`，其中仅部分维度应用 RoPE；
- per-head attention sink；
- `q_lora_rank=1024` 和 grouped output LoRA；
- `index_n_heads=64, index_head_dim=128, index_topk=512` 的 DSA indexer；
- compress、sliding-window 和 full attention/cache 路径。

它与 DeepSeek-V2 MLA 有部分可复用组件，但 KV 表示、cache 形状和 sink 语义不同，不能以“改几个 config 字段”的方式适配。

### 3.3 mHC、MoE 与 hash router

- mHC 使用 `hc_mult=4`，hidden state 逻辑形状为 `[B, S, 4, D]`，默认执行 20 次 Sinkhorn 迭代；
- MoE 使用 `sqrtsoftplus` 打分和 `noaux_tc` 选择，选择时加入 correction bias，聚合时使用原始 score 并归一化；
- expert activation 还包含 gate/up clamp，现有 `fused_moe_mxfp4` 的标准 SwiGLU 接口没有表达该限制；
- hash layer 通过 `tid2eid[input_ids]` 直接选专家，不能走普通 top-k router。

### 3.4 DSpark 边界

HF config 中的 next-token 字段与[官方 inference config](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark/blob/main/inference/config.json)并不完全一致。官方 inference 代码将 `mtp.*` 实现为三阶段 DSpark speculative pipeline。因此本项目以官方 inference 目录中的配置、转换脚本和模型代码作为 DSpark 语义的最终依据：

- [model.py](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark/blob/main/inference/model.py)
- [kernel.py](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark/blob/main/inference/kernel.py)
- [convert.py](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark/blob/main/inference/convert.py)

---

## 4. InfiniLM / InfiniCore 差距

| 模块 | 可复用基础 | 实际缺口 |
|---|---|---|
| 模型骨架 | 模型注册、`qwen3_moe`、`deepseek_v2` | 新 config、层型 dispatch、四路 hidden state |
| V4 attention | RoPE、GEMM、部分 cache 基础 | 单 KV 头布局、sink、grouped output、512 维 head、三类 state/cache |
| DSA indexer | InfiniCore `fp8_indexer_logits/quant` | 需要验证精确量化布局和 top-k 语义 |
| 稀疏注意力 | InfiniCore `fp8_sparse_mla`、`dsa` | 现有 wrapper 的固定形状及 sink 接口与 V4 不完全匹配，不能直接宣称可用 |
| mHC | matmul、逐元素算子 | Sinkhorn、层前/层后混合、四路状态管理 |
| MoE | `SparseMoeBlock`、`fused_moe_mxfp4` | sqrtsoftplus/noaux_tc、hash router、gate/up clamp |
| 权重加载 | safetensors loader、weight remap | remap 前 dtype cast 会破坏特殊 FP8 scale；需要公共加载路径的小范围修正 |
| runtime state | `ForwardContext` 中的 cache vectors | compress/indexer 状态可能需要模型私有分页结构或最小公共扩展 |

相关 InfiniCore 接口：

- [fp8_sparse_mla](https://github.com/InfiniTensor/InfiniCore/blob/main/src/infinicore/ops/fp8_sparse_mla/fp8_sparse_mla.cc)
- [dsa](https://github.com/InfiniTensor/InfiniCore/blob/main/src/infinicore/ops/dsa/dsa.cc)
- [fused_moe_mxfp4 CUDA kernel](https://github.com/InfiniTensor/InfiniCore/blob/main/src/infiniop/ops/mxfp4_common/cuda/fused_moe_mxfp4_kernel.cuh)

原方案中“改动集中在模型目录、尽量不改公共代码”不再作为硬约束。正确的做法是控制公共改动规模，并为 loader/cache 改动补回归测试。

---

## 5. 实施方案

### 5.1 先做 3 天 Go/No-Go

8 月 25 日结束前必须回答以下问题：

1. safetensors shard 能否流式、无损 remap，FP8/FP4 data 与 E8M0 scale 是否保持原始位模式；
2. V4 expert 的打包格式能否直接喂给 `fused_moe_mxfp4`；
3. gate/up clamp 能否通过扩展现有 kernel 实现并与官方结果一致；
4. DSA indexer 的 64×128 输入布局是否与现有 InfiniCore 接口一致；
5. 2×96 GiB 在只加载 base-model path 时是否有足够显存余量。

其中第 1、2、5 项任一失败且 8 月 27 日前没有明确修复路径，就必须把“真权重生成”降为冲刺项；结题主交付改为 tiny 全机制正确性、真实 checkpoint 无损转换器和可复现的剩余 gap。不能用 BF16 全量反量化伪装成功。

### 5.2 开发顺序

1. **权重格式和 loader**：先解决能否加载，避免最后一周才发现 FP4 不兼容；
2. **dense + mHC 骨架**：建立四路 hidden-state 生命周期；
3. **attention + state/cache**：先 dense attention 对拍，再接 indexer 和三类路径；
4. **router + MoE**：普通 routed layer、hash layer、共享专家、FP4 fused path；
5. **全层 tiny 对拍**：覆盖首层、普通层、hash 层和不同 compress ratio；
6. **真权重分片加载**：先只初始化，再单 token forward，最后生成 1～8 token；
7. **回归、文档和 PR**。

### 5.3 验证标准

| 层级 | 验证内容 | 通过标准 |
|---|---|---|
| 算子级 | Sinkhorn、router、clamp、indexer | 与官方/PyTorch 参考在约定容差内一致 |
| 模块级 | attention、mHC、MoE、hash layer | 固定 seed 的输入输出和关键中间 tensor 对拍 |
| tiny 模型 | 覆盖所有层型的多 token forward | logits/argmax 一致，无未初始化 state |
| 真权重 | 固定 prompt，batch=1，短上下文 | 能稳定生成 1～8 token；记录逐 token argmax/logits 差异 |
| 工程回归 | format、已有模型测试、新增 loader 测试 | 无新增失败 |

仅“模型能构造”不算真权重验证通过；必须至少完成一次真实 token forward。

---

## 6. 四周排期

| 日期 | 必须产出 | 租机安排 |
|---|---|---|
| 8/23–8/25 | 完成 Go/No-Go：checkpoint dtype/命名、FP4 expert、indexer、显存验证 | 2×96/141 GiB，8～16 小时 |
| 8/26–8/30 | loader/remap、配置和模型骨架；dense + mHC tiny 对拍 | 本地为主，必要时 4～8 小时 |
| 8/31–9/5 | V4 attention、sink、三类 state/cache、DSA indexer 对拍 | 本地为主 |
| 9/6–9/10 | router、hash layer、共享专家、FP4 fused MoE；全层 tiny 对拍 | 2×96/141 GiB，16～32 小时 |
| 9/11–9/14 | 真权重分片加载、单 token forward、首轮生成；集中修错 | 2×96/141 GiB，24～40 小时 |
| 9/15–9/17 | 官方实现交叉验证、显存/延迟记录、回归测试 | 4×80 GiB，12～24 小时 |
| 9/18–9/20 | 报告、复现脚本、代码整理和 PR；只修阻断问题 | 预留 8～16 小时 |

9 月 10 日后冻结新功能；9 月 15 日后冻结架构改造。最后五天只处理正确性、测试和交付。

---

## 7. 服务器方案与成本

### 7.1 推荐拓扑

**最低成本路线**：

- 2×RTX PRO 6000 96GB：完整 checkpoint 初始化和短上下文迭代；
- 4×H100 PCIe 80GB：最终对照官方 MP=4 路径。

该路线便宜，但 Blackwell 工作站卡与官方/InfiniCore kernel 的兼容性必须在第一天实测；同一宿主机上双卡库存也不保证稳定。

**降低排期风险的路线（推荐）**：

- 2×H200 141GB：Go/No-Go、真权重迭代，显存余量充足；
- 4×H100 PCIe 80GB：最终 MP=4 交叉验证。

A100 虽然便宜，但不适合作为主要验收机：本项目同时涉及 FP8 与自定义 FP4 路径，使用 Hopper 能减少“模型代码问题还是旧架构 kernel 问题”的排查变量。

### 7.2 公开价格基线

以下使用 [RunPod 公开按需价格](https://www.runpod.io/pricing)在 2026-08-23 的页面值，仅作为预算基线：

| GPU | Community / GPU·h | Secure / GPU·h | 本项目整机小时价 |
|---|---:|---:|---:|
| RTX PRO 6000 96GB，2 卡 | $1.69 | $2.09 | $3.38～$4.18 |
| H200 141GB，2 卡 | $3.59 | $4.59 | $7.18～$9.18 |
| H100 PCIe 80GB，4 卡 | $1.99 | $2.89 | $7.96～$11.56 |

500GB 网络盘足够保存原始 checkpoint、转换产物和日志。公开页面的标准/高性能网络盘约为 $0.05/$0.14 每 GB·月，即 **$25～$70/月**。若同时保留多份转换权重，应改为 750GB～1TB。

### 7.3 四周预算

按 60～100 小时双卡迭代、12～24 小时四卡验收、500GB 网络盘一个月，并加入 30% 的失败重试和排队切换余量：

| 方案 | 计算方式 | 美元预算 | 人民币规划值* |
|---|---|---:|---:|
| 省钱：2×RTX PRO 6000 + 4×H100 | 双卡 60～100h + 四卡 12～24h + 存储 + 30% | **$420～$995** | **约 ¥3,000～¥7,200** |
| 稳妥：2×H200 + 4×H100 | 双卡 60～100h + 四卡 12～24h + 存储 + 30% | **$717～$1,645** | **约 ¥5,200～¥11,900** |

\* 人民币仅用 `1 USD = 7.2 CNY` 做预算换算，不代表实时汇率；云平台税费、支付手续费和地区溢价未计入。

**建议准备 ¥8,000～¥12,000 的算力额度**，执行时优先控制在 ¥7,000 左右。这个额度不是要求全部花完，而是避免最后一周因双卡不可用或 kernel 只能在 Hopper 上跑而失去验收窗口。

### 7.4 控费方法

- 绝大多数 tiny 对拍留在本地 16/17GB 显卡完成；
- 租机前准备一键安装、下载、编译和测试脚本，服务器启动后不做文档调研；
- checkpoint 常驻网络盘，停止 GPU 实例而不是反复下载 166.9GB；
- 第一次租机只买 8～16 小时做 Go/No-Go，通过后才扩大预算；
- 每次运行记录 GPU 峰值、失败阶段和 commit，避免重复付费复现旧问题；
- 最后一周优先按需/secure 实例，不依赖可能随时回收的低价实例。

服务器成本不会因为“每天开发时间更长”自动上升；主要取决于全权重调试是否过早搬到云端，以及每次租机前是否已有明确验证目标。

---

## 8. 风险与降级规则

| 风险 | 截止检查点 | 处理方式 |
|---|---|---|
| FP4 格式与现有 fused MoE 不兼容 | 8/25 | 立即做最小 InfiniCore 扩展；8/27 仍无路径则降级真权重目标 |
| loader 转 dtype 破坏 scale | 8/26 | 修改加载顺序并补特殊 dtype 回归测试 |
| attention/cache 公共接口改动过大 | 9/3 | 先完成 batch=1 模型私有 state，公共分页化列为冲刺项 |
| mHC 组合算子过慢 | 9/5 | 保留正确的组合实现；融合优化不进结题范围 |
| 2×96 GiB OOM | 首次真权重初始化 | 切 2×H200 或 4×H100，不允许 BF16 反量化绕过 |
| 官方参考无法同拓扑运行 | 9/15 | 使用官方 MP=4 环境生成 golden logits，再验证 InfiniLM |
| 上游 PR 依赖 InfiniCore PR | 9/17 | 提交关联 PR/WIP PR 和固定 commit 的复现说明 |

最终报告必须区分“已经通过”“只在 tiny 通过”“尚未实现”，不能把结构支持、权重加载和完整 serving 混写成同一个完成状态。

---

## 9. 最终交付清单

- `csrc/models/deepseek_v4/` 模型实现；
- 模型注册、config 与必要的最小公共 loader/state 改动；
- 必要时配套的 InfiniCore operator/kernel 改动；
- `test/models/deepseek_v4/` 和特殊 dtype loader 回归测试；
- tiny config、golden tensors、真权重运行脚本；
- 显存、延迟、数值误差和逐平台状态表；
- 项目报告与 InfiniLM PR；若存在跨仓依赖，同时提交 InfiniCore PR。

本计划的核心原则是：**9 月 20 日不变，先消灭格式与算子的不确定性，再实现模型；以可复现的正确性作为完成标准，不用不可运行的“大而全”范围包装进度。**

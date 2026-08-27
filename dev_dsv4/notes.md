# DeepSeek-V4 调研笔记 #1：架构机制地图与参照基线

日期：2026-08-22。来源：本地 transformers 5.14.1 内置参照实现
（`transformers/models/deepseek_v4/{configuration,modeling}_deepseek_v4.py`，含论文小节引用）。

## 机制地图（已核实）

### 注意力：三种层型，按层调度
- `sliding_attention`：纯滑窗分支，plain RoPE（theta=10000，无 scaling）；
- `compressed_sparse_attention`（CSA，压缩率 m=4）：压缩分支 + Lightning Indexer 稀疏选择；
- `heavily_compressed_attention`（HCA，压缩率 m'=128）：重压缩分支；
- Flash 版层型调度：`compress_ratios=[0,0,4,128,4,128,...]` = 前 2 层滑窗 + CSA/HCA 交替；
- 压缩分支用 YaRN（`compress_rope_theta=160000`，factor=16，参照实现强制 `attention_factor=1.0`）；
- 共享 KV 的 MQA（`num_key_value_heads=1`），`head_dim=512`，partial rotary 64/512；
- Grouped Output Projection：`o_groups=8` + `o_lora_rank=1024`；
- Lightning Indexer：`index_n_heads=64, index_head_dim=128, index_topk=512`（top-k 选择压缩条目）。

### 残差：mHC（Manifold-Constrained Hyper-Connection），全程生效
- hidden 全程保持 `[B, S, hc_mult=4, D]` 四流结构；
- 每个子层两个 `DeepseekV4HyperConnection`：产出 `post`（逐流缩放）、`comb`（流间混合矩阵）、
  `collapsed`（四流折叠为子层输入）；
- `comb` 经 Sinkhorn-Knopp 投影（20 次迭代，float32）到双随机矩阵流形；
- 混合方向敏感：`comb.T @ residual`（comb 非对称）。

### FFN：MoE + Hash-MoE bootstrap
- 前 3 层 `hash_moe`：冻结 `tid2eid[input_ids]` 查表选专家（不学习 argmax），
  但门控权重仍由 learned gate 打分（score 仅用于加权，不用于选择）；
- 其余层标准 top-k MoE；`scoring_func=sqrtsoftplus`、`topk=noaux_tc`、
  `routed_scaling_factor=1.5`、`swiglu_limit=10.0`（截断 gate/up 预激活）；
- 共享专家 1 个；FP8 主权重 + FP4 专家（`expert_dtype=fp4`）。

### 并行方案（参照实现仅定义 EP）
- MoE 走 EP（grouped GEMM + all-reduce）；主注意力复制（共享 KV 不宜切 q_b_proj）；
- Indexer 的 q_b_proj/scorer.weights_proj 按列切、scorer 输出 all-reduce。

## 已验证事实

- 参照实现 prefill/decode 增量一致性：全层型与混合层型 max|diff| ≈ 1e-6（float32），
  自定义 cache（`DeepseekV4HCACache/CSACache`，DynamicSlidingWindowLayer 派生）可信；
- tiny config（5 层全覆盖层型）可跑通，`dev_dsv4/baseline_tiny.pt` 已存逐层 hidden states、
  logits、state_dict；权重清单 `weight_inventory.json`（145 个张量）；
- 本地环境：RTX 5060 Ti 17GB + torch 2.11.0+cu128 + transformers 5.14.1。

## 待办（下一次推进）

- 精读 `DeepseekV4Indexer.forward` / `HCACompressor` / `CSACompressor` / `DeepseekV4Attention.forward`，
  补齐压缩窗口边界、top-k 语义、indexer score 公式细节；
- 精读 `DeepseekV4GroupedLinear`（o 投影分组布局）；
- 对照 `csrc/models/kimi_k3/` 做模块级 gap 分析（InfiniLM 侧首要参照：混合层型调度、
  异构 cache 分配 `kimi_k3_allocate_cache.cpp`、MLA、`fused_moe_mxfp4` 消费先例）；
  次级参照 `csrc/models/deepseek_v2/`（MLA）；
- 对照 InfiniCore `fp8_indexer_*` 算子接口，确认与参照实现 indexer 语义匹配度；
- 真实 checkpoint 的 FP4 专家存储格式（待真权重验证阶段确认）。

## 复现

```bash
python dev_dsv4/ref_baseline.py        # 生成/校验基线（含增量一致性自检）
python dev_dsv4/check_incremental.py   # 分层型一致性诊断
```

---

## 调研笔记 #2：checkpoint Go/No-Go（2026-08-23）

新增 `dev_dsv4/check_checkpoint_format.py`，通过 HTTP Range 只读取 48 个
safetensors header，不下载权重数据。全量检查结果：

- 72,317 个 tensor，总大小 155.418 GiB；
- I8 packed expert weight 138.000 GiB；
- F8_E8M0 scale 8.625 GiB；
- F8_E4M3 weight 5.871 GiB；
- routed experts（weight + scale）合计 146.625 GiB；
- 35,718 组 `weight/scale` 全部位于同一 shard，可逐 shard 转换；
- base-model 的 34,223 个非 scale source key 全部能映射到唯一目标 key，
  无碰撞和未处理项。

第一道 Go/No-Go 结论为**通过**：

- dense FP8 权重先按 128×128 block scale 逐 shard 解量化到 BF16；
- routed expert I8 carrier 与 E8M0 scale 用 `view(uint8)` 无损转成
  InfiniCore MXFP4 storage；
- `mtp.*` 在本期 base-model path 中明确丢弃；
- mHC、attention sink 和 router correction bias 的 F32 dtype 必须保留。

相应实现与测试位于：

- `python/infinilm/modeling_utils.py`：V4 dtype 保留、逐 shard dequant、key remap；
- `test/models/deepseek_v4/test_checkpoint_loading.py`：E8M0 bit-pattern 和
  dense/packed remap 回归测试；
- `csrc/models/deepseek_v4/deepseek_v4_config.*`：config 校验与层型展开。

---

## 开发环境基线（2026-08-23）

已完成本机 NVIDIA 开发链路的真实构建与运行验证：

- xmake `v3.1.0`：`~/.local/xmake-3.1.0`，入口为 `~/.local/bin/xmake`；
- InfiniCore 固定提交：`46ca684929aaa2ce69fb1b288787e8a859e26c93`；
- InfiniCore 安装前缀：`~/.infini`；
- 构建目标：CUDA 13.2、`sm_120`、CUDA Graph；单卡开发配置关闭 NCCL、cuDNN；
- InfiniLM Python 环境：仓库 `.venv`（复用系统 site-packages），C++ ABI 为 1；
- InfiniLM 全量 C++ 扩展编译通过，V4 loader 测试 3/3 通过；
- RTX 5060 Ti 上真实 InfiniCore `add` 算子通过，max diff `2.3841858e-7`；
- `_infinilm` 的依赖解析到 `~/.infini/lib`，CUDA runtime 解析到
  `~/.local/cuda-13.2/lib64/libcudart.so.13`。

InfiniCore 当前提交需要两个本机构建兼容补丁：

1. `xmake.lua` 的 `cuda_arch` 白名单补入 `sm_120`；
2. `cudnn=n` 时排除 `avg_pool3d` 的 NVIDIA/cuDNN 实现，避免同时加载
   PyTorch CUDA 12.8 cuDNN 与 CUDA 13.2 runtime。

进入开发环境：

```bash
source .venv/bin/activate
export PATH="$HOME/.local/bin:$PATH"
export INFINI_ROOT="$HOME/.infini"
export CUDA_HOME="$HOME/.local/cuda-13.2"
export LD_LIBRARY_PATH="$INFINI_ROOT/lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export INFINILM_CXX11_ABI=1
```

回归命令：

```bash
python -m pytest -q test/models/deepseek_v4/test_checkpoint_loading.py
xmake build -y -j8 _infinilm
```

2026-08-23 本地复验结果：loader `3 passed`；tiny baseline 的 sliding、CSA、
HCA 和 mixed 四种分层增量路径最大绝对误差均不超过 `1.1e-6`；C++ 扩展增量
构建通过。

---

## 验证笔记 #3：mHC 原生组合路径（2026-08-23）

新增 `DeepseekV4HyperConnection` 和 `DeepseekV4HyperHead`，使用 InfiniCore
基础算子实现：

- unweighted RMSNorm 后执行 FP32 mapping projection；
- `pre/post/comb` 的 sigmoid、softmax 和 epsilon 语义；
- 与官方一致的 Sinkhorn 顺序：首次 column normalize，之后交替 row/column；
- `comb.T @ hidden_streams` 的有向残差混合；
- 最终 HyperHead 四流 collapse。

`dev_dsv4/mhc_native_smoke.cpp` 含独立 CPU 参考公式，并在 RTX 5060 Ti 上与
原生 CUDA 路径逐元素对拍。FP32 最大绝对误差：

- `post=1.75238e-5`，`comb=3.44217e-5`；
- `collapsed=1.02520e-5`，`apply=2.02060e-5`；
- `head=8.94070e-8`。

ATen-enabled InfiniCore 下的 BF16 输入最大绝对误差：

- `collapsed=0.00318474`；
- `apply=0.00489402`；
- `head=0.00493288`。

复现：

```bash
# 当前非 ATen 构建：验证 FP32 组合路径
dev_dsv4/run_mhc_native_smoke.sh

# ATen-enabled InfiniCore：同时验证 BF16 ↔ FP32 cast 路径
INFINI_ROOT=/path/to/infinicore-aten dev_dsv4/run_mhc_native_smoke.sh
```

验证同时确认两个 InfiniCore 约束：通用 elementwise 算子不会隐式 broadcast，
所有 scale/base/epsilon 必须先显式 `broadcast_to`；`cast_` 在 `aten=false`
构建中不可用。V4 正式运行环境因此需要 `aten=true`，除非后续在 InfiniCore
补一个不依赖 ATen 的原生 dtype cast。

---

## 验证笔记 #4：ATen/BF16 环境与最小 dense Attention（2026-08-23）

V4 专用 InfiniCore 已固定到 `~/.infini-dsv4`。该前缀启用 ATen，保留
CUDA 13.2、`sm_120`、CUDA Graph，并关闭本机不需要的 NCCL/cuDNN；它与
稳定的 `~/.infini` 并存，不覆盖现有开发环境。InfiniLM 使用该前缀完成了
全量 C++ 编译和最终动态链接。

新增的 `DeepseekV4Attention` 是无 cache 的最小 dense correctness path，已覆盖：

- `q_a_proj -> weighted RMSNorm -> q_b_proj -> unweighted RMSNorm`；
- 单共享 KV 头及其 weighted RMSNorm；
- head 尾部的 interleaved partial RoPE；
- causal mask、每头 learnable sink，以及丢弃 sink 概率后的注意力输出；
- 对输出 RoPE 区间施加共轭旋转；
- block-diagonal grouped `o_a_proj` 和最终 `o_b_proj`；
- 与官方 checkpoint 一致的参数键名。

`dev_dsv4/attention_native_smoke.cpp` 使用独立 CPU 公式，在 RTX 5060 Ti 上
逐段对拍 Query、KV、attention weights 和最终输出。最大绝对误差：

- FP32：`query=0.00372711`、`kv=0.000153542`、
  `weights=0.00042513`、`output=0.000115568`；
- BF16：`query=0.0514424`、`kv=0.0107355`、
  `weights=0.00761053`、`output=0.00126566`。

Query 的 BF16 误差是 tiny 维度下连续两层 BF16 projection 相对 FP32 CPU
参考的累计误差；最终输出和权重仍在预设容差内。验证还发现 partial RoPE
的奇偶通道 `narrow` 视图必须先 `contiguous()`，否则当前 InfiniCore
elementwise kernel 会按错误的连续布局读取。

复现：

```bash
INFINI_ROOT="$HOME/.infini-dsv4" dev_dsv4/run_mhc_native_smoke.sh
INFINI_ROOT="$HOME/.infini-dsv4" dev_dsv4/run_attention_native_smoke.sh
```

下一步是在该核心上加入 sliding-window KV cache 与 prefill/decode 增量对拍，
再接 CSA/HCA compressor 和 Lightning Indexer。

---

## 验证笔记 #5：sliding-window KV cache（2026-08-23）

`DeepseekV4Attention::forward_sliding` 已实现显式共享-KV cache：

- cache 布局为 `[batch, retained_sequence, 1, head_dim]`；
- 每次调用将历史 cache 与当前 chunk 的 RoPE-applied KV 临时拼接；
- causal mask 同时限制未来 token 和滑窗左边界；
- attention sink 不受滑窗 mask 影响，始终作为额外 softmax logit；
- 返回前只保留最后 `sliding_window - 1` 个 KV，与 Transformers
  `DynamicSlidingWindowLayer` 一致；
- 同一实现支持整段 prefill 和逐 token decode。

native smoke 使用 `sliding_window=2`、序列长度 3，明确跨越窗口边界，并与
独立 CPU 参考公式及逐 token 路径对拍。最大绝对误差：

- FP32：`weights=0.000769794`、`output=0.000119483`、
  `prefill/decode=0.000119466`；
- BF16：`weights=0.0131521`、`output=0.00126566`、
  `prefill/decode=0`。

全量 InfiniLM C++ 构建、loader `3 passed`、四种官方 tiny 增量基线和 mHC
FP32/BF16 smoke 均再次通过。下一步转入 HCA compressor；它比 CSA 少一个
Lightning Indexer 分支，适合作为压缩注意力的第一条落地路径。

---

## 验证笔记 #6：HCA compressor 与跨 chunk 状态（2026-08-23）

新增 `DeepseekV4HCACompressor`，实现 HCA 非重叠窗口压缩：

- `kv_proj` / `gate_proj` 保留官方参数布局；
- `gate + position_bias` 在窗口轴做 FP32 softmax，再转回 KV dtype 加权求和；
- pooled KV 经 weighted RMSNorm 和 compress-RoPE；
- 不足一个窗口的 projected KV/gate 余数跨调用保留；
- 每闭合一个窗口追加一条长期 compressed KV，并维护 `entry_count`；
- stateless 路径仅压缩本次调用中的完整窗口并丢弃余数；
- 根据绝对 `position_ids` 生成 compressed-entry causal block bias。

native smoke 使用压缩率 2、序列长度 5，并按 `[1, 2, 2]` 三个 chunk 输入，
覆盖“首次无输出、缓存余数、连续闭合窗口、最终仍有余数”的完整状态机。
最大绝对误差：

- FP32 stateless：`1.78814e-7`；incremental/history：`5.96046e-8`；
- BF16 stateless：`0.0074892`；incremental/history：`0`。

block bias 同时检查了可见条目与未来条目的 `-inf` 屏蔽。当前 correctness
实现会把 `position_ids` 同步到 CPU 构造该不规则 mask；接入 CUDA Graph 前需要
替换为设备侧 mask kernel。全量 C++ 构建及 loader、官方 tiny 增量、mHC、
sliding Attention 回归均通过。

下一步是把 HCA 的 compressed KV 和 block bias 接入核心 Attention，使同一层
同时消费短程 sliding KV 与长期 HCA KV，然后做端到端 HCA prefill/decode 对拍。

---

## 验证笔记 #7：HCA Attention 端到端接线（2026-08-24）

`DeepseekV4Attention::forward_hca` 已把短程 sliding KV 和长期 compressed KV
接入同一个 attention softmax：

- HCA 层按配置注册 `compressor` 子模块，参数键保持官方
  `compressor.{kv_proj,gate_proj,kv_norm,position_bias}` 命名；
- KV 按 `[sliding, compressed]` 顺序拼接，对两段分别应用滑窗 causal bias 和
  compressed-entry block bias；
- 单 token decode 没有 block bias 时，已有 compressed history 默认全部可见；
- sliding cache 仍只保留最后 `sliding_window - 1` 个条目，HCA state 独立维护
  projected 余数、compressed history 和 `entry_count`；
- dense/sliding 层不注册 compressor，因此不会污染它们的 checkpoint key 集合。

集成 smoke 使用压缩率 2、滑窗 2、序列长度 5，检查完整 prefill 的未来 HCA
条目权重严格为 0，并与逐 token decode 拼接输出对拍：

- FP32 prefill/decode 最大绝对误差：`5.19119e-6`；
- BF16 prefill/decode 最大绝对误差：`0`；
- compressor 独立参考对拍保持 FP32 `1.78814e-7`、BF16 `0.0074892`。

全量 `_infinilm` 目标已在 ATen-enabled InfiniCore 下重新编译链接；loader
`3 passed`、四种官方 tiny 增量基线（`<=1.1e-6`）、dense/sliding Attention 和
mHC FP32/BF16 smoke 全部通过。当前 HCA correctness path 的已知生产化缺口仍是
CPU 构造 block bias；后续需替换为设备侧 kernel，再进入 CUDA Graph 验证。

下一步实现 CSA compressor 和 Lightning Indexer，优先先对齐 index score、top-k
选择及稀疏 KV gather，再把异构层 cache 接入 decoder/model 调度。

---

## 验证笔记 #8：CSA compressor 与 Lightning Indexer（2026-08-24）

新增 `DeepseekV4CSACompressor` 和 `DeepseekV4Indexer` correctness path：

- outer compressor 和 indexer 各自维护 projected KV/gate 余数、上一完整窗口的
  Ca overlap、compressed history 与 `entry_count`；
- 每个压缩条目按官方布局组合“前一窗口 Ca + 当前窗口 Cb”，窗口宽度 `2m`、
  stride `m`，首个窗口的缺失 Ca 使用 zero-KV / `-inf` gate；
- gate 在 FP32 上沿窗口轴 softmax，随后 weighted RMSNorm 和 compress-RoPE；
- Indexer query 使用 attention 的 weighted query-LoRA residual，评分严格实现
  `sum_h w_h * ReLU(q_h dot k) / sqrt(head_dim) / sqrt(num_heads)`；
- causal top-k 的无效位置返回 `-1`，再构造仅开放所选 compressed entry 的
  block bias；CSA Attention 与 sliding KV 共用最终 softmax。

native smoke 使用压缩率 2、`index_topk=1`、序列长度 5，覆盖首窗口、跨窗口
overlap、余数缓存、causal top-k、prefill 和逐 token decode。结果：

- FP32 outer compressor 参考误差 `2.38419e-7`，增量历史 `2.98023e-8`；
- FP32 indexer 增量历史误差 `2.38419e-7`；
- FP32 CSA prefill/decode 误差 `2.98172e-5`；
- BF16 outer/indexer 参考误差分别为 `0.00941807` / `0.00931716`，
  prefill/decode 误差为 `0`。

dense、sliding、HCA 的 F32/BF16 原生回归均保持通过。当前 correctness path
仍会把 `position_ids` 和 top-k indices 同步到 CPU 构造 causal/index mask；正式
CUDA Graph 路径需换成设备侧 mask/scatter，并进一步接现有 DSA sparse kernel。

下一步把三种 attention 路径和各自 state 接入 decoder/model 层调度，然后完成
包含 mHC 与混合层型的 InfiniLM tiny 模型端到端对拍。

---

## 验证笔记 #9：标准 MoE 与 Hash-MoE（2026-08-24）

新增 `DeepseekV4SparseMoeBlock` correctness path，覆盖：

- `sqrtsoftplus` router score、correction-bias top-k、选中专家概率归一化与
  `routed_scaling_factor`；
- 前 `num_hash_layers` 层使用冻结 `tid2eid[input_ids]` 选择专家，同时继续使用
  learned router score 计算混合权重；
- routed experts、共享专家和 V4 限幅 SwiGLU；
- tiny dense expert 参数布局，以及真实 checkpoint 的逐专家 MXFP4 packed 参数布局；
- packed 路径接入现有 `fused_moe_mxfp4`（V4 限幅融合仍作为后续 InfiniCore 缺口）。

native smoke 对标准和 Hash 两条路由分别与独立 CPU 公式对拍：

- FP32 最大绝对误差：`3.72529e-9` / `1.49012e-8`；
- BF16 最大绝对误差：`2.75731e-4` / `3.94031e-4`。

`_infinilm` 全量 C++ 扩展构建通过。下一步将 mHC、混合 attention 和 MoE 接入
decoder/model，完成官方 tiny state dict 的整模型 prefill/decode 对拍。

---

## 验证笔记 #10：完整 Decoder / CausalLM 与 tiny 端到端对拍（2026-08-24）

新增完整 `DeepseekV4DecoderLayer`、`DeepseekV4Model` 和
`DeepseekV4ForCausalLM` correctness path：

- embedding 后扩展为 `hc_mult` 条 residual streams，每层依次执行
  mHC-attention、异构 attention、mHC-FFN 与 Hash/标准 MoE；
- `layer_types` 在 sliding / CSA / HCA 间调度，并为每层独立维护 sliding KV、
  compressor、indexer 及 overlap 状态；
- sliding 层使用 main RoPE，CSA/HCA 的 query、短程 KV 和压缩条目统一使用
  compress RoPE；
- hyper head、最终 RMSNorm 与 LM head 已接线，模型以 `deepseek_v4` 注册到
  CausalLM registry；当前 correctness path 明确限制 `PP=TP=1`；
- 145 个官方 tiny state-dict key 与 native 模型精确一致，无缺失或多余参数。

端到端 smoke 会重建同权重 Transformers 参照，默认验证完整 5 层混合模型的
prefill 和 4 步增量 decode。FP32 结果：

- prefill hidden 最大绝对误差 `3.738850e-3`，logits `1.593232e-3`；
- 4 步 decode hidden 最大绝对误差 `3.736645e-3`，logits `1.099348e-3`。

调试还暴露了 InfiniCore CUDA top-k 在一行所有有限候选均为负数时会漏选的
边界问题。Indexer 在 top-k 前增加保序 softmax：有限分数变为正数、causal
`-inf` 保持零概率，选择结果由此与官方实现一致。测试中的 Torch Tensor 必须
保持到 native forward 完成，因为 `infinicore.from_torch` 是零拷贝视图。

复现：

```bash
INFINI_ROOT="$HOME/.infini-dsv4" dev_dsv4/run_model_native_smoke.sh
```

下一步验证真实 checkpoint 的 FP8 dense 反量化、MXFP4 packed expert 布局和
本地可执行语义；完整约 155 GiB 权重的加载与短提示推理留到租用服务器后完成。

---

## 验证笔记 #11：YaRN、真实 FP4 语义与服务器验收边界（2026-08-24）

本地可完成的真权重前置工作已经闭环：

- compressed layer 支持发布配置的 YaRN（factor=16、beta_fast=32、
  beta_slow=1、original context=65536、compress theta=160000）；完整五层 tiny
  模型的 YaRN prefill hidden/logits 最大误差为 `4.725099e-3` / `1.449406e-3`，
  4 步 decode 最大误差为 `3.758430e-3` / `1.114845e-3`；
- 发布配置没有 `compress_rates`，native config 会从既定层型补出 CSA=4、
  HCA=128；原生 config smoke 已验证 43 层调度、前三层 Hash-MoE 和 FP4 别名；
- InfiniCore MXFP4 fused MoE 新增 V4 `SwigluLimit10` activation，在 gate 上限 10、
  up 的 `[-10, 10]` 区间执行截断；CPU/NVIDIA 的 FP16/BF16/FP32 共 24 个用例
  全部通过，并有断言保证测试数据确实触发 clamp；
- InfiniLM packed expert smoke 覆盖逐专家 `w1/w3/w2` 的 U8 weight/scale 参数
  注册、切片加载和 fused forward；独立 CPU E2M1/E8M0 解码参考与 NVIDIA F32
  输出最大误差 `1.22070e-4`；
- loader 固定按 shard 名排序；dense FP8/FP4 bit-pattern 回归仍为 `3 passed`。

增强后的远端 header probe 同时校验官方 config、72,317 个 tensor、35,718 组
weight/scale 同 shard、FP8 block scale shape、FP4 专家 shape 和关键 mHC/压缩参数。
统计口径修正后：checkpoint 总计 155.418 GiB，其中 MTP 为 10.117 GiB；当前
base loader 丢弃 `mtp.*`，dense FP8 转 BF16 后估算常驻参数 150.756 GiB。

最终服务器脚本 `dev_dsv4/run_real_checkpoint_smoke.sh` 已准备好，包含 48-shard
完整性、FP4 config、至少 165 GiB 空闲显存的 preflight，以及加载后的 prefill
NaN/Inf 检查和 4-token decode。服务器规格、B200/B300 `sm_100f` 构建命令和
通过标准见 `dev_dsv4/server_validation.md`。至此剩余唯一无法在本机完成的动作是
租用一张满足显存门槛的服务器并执行该真权重命令。


---

## 路线调整与冲刺计划（2026-08-27）

触发：B200/B300 市场无货，2×96GB 同宿主机双卡货源不稳定；立项书 §5.1
显存项 Go/No-Go 死线（8/27）到达，按规则把"真权重 43 层全量生成"降为
冲刺项。结题主交付回到 tiny 全机制正确性 + 无损转换器 + 可复现 gap。

已完成的方案评估（结论存档）：

- 6×32GB 拼卡：不可行。专家 EP 均分后每卡约 35.8 GiB（含 dense 复制）
  超过 32 GB，且需为 V4 新写 EP/TP，工作量 1~2 周；
- 2×96GB PP=2：技术可行（kimi_k3 有完整 PP 先例，V4 改造点已清单化，
  切分后每卡约 76~79 GiB 有余量），但需自费且对本项目目标边际价值低，
  放弃，改造点清单保留备查；
- 昇腾真权重：不可行。ModelArts 实例规格中 "192 GiB" 是主机内存，
  ascend-snt9b3（910B3）HBM 仅 64GB；ascend 后端无端到端模型先例，
  `fused_moe_mxfp4` 需从零写 Ascend kernel，估算 2.5~4 周超出剩余工期；
- CPU 全量真权重（大内存 ECS）：可行但价值/成本比低，本期取消，
  不再保留为机会项。

冲刺计划（8/27 → 9/20）：

| 日期 | 事项 | 产出 |
|---|---|---|
| 8/27–8/30 | PR 可合入化：tiny 端到端对拍迁入 `test/models/deepseek_v4/` 正式 pytest；全量回归绿；InfiniCore `feat/mxfp4-clamped-swiglu` 整理为可评审 | 双仓可评审 PR 状态 |
| 8/31–9/1 | 昇腾 Go/No-Go（代金券租机，优先 x86 主机）：ascend 构建 + add/gemm/rms_norm 三项 `--ascend` 实测；两天不过则放弃昇腾转入性能轨道 | 环境结论 |
| 9/2–9/10 | 昇腾 tiny 支持：新写 11 个 aclnn 算子封装（broadcast_to/mul/mul_scalar/softmax/sum/relu/sigmoid/reciprocal/topk/fmin/silu）+ infiniop 原生 cast（aclnnCast，替代 ATen 接线）+ cat 白名单一行；tiny 模型昇腾对拍。tiny 走 dense 专家路径（`expert_dtype` 默认 dense），不需要 mxfp4 kernel | InfiniCore 昇腾算子 PR + 双平台 tiny 对拍记录 |
| 并行 | 性能微基准（本地 5060 Ti，零成本）：fused_moe_mxfp4 真实 shape 下 fused vs 解量化+GEMM 对照、roofline 分析 | 性能数据 |
| 9/11–9/14 | 深度抽样真权重对拍（本地，零租金）：下载 2~4 个真实 shard，抽样组装覆盖 sliding/CSA/HCA/hash-MoE 全部层型的减深度模型 + 真实 embedding/lm_head，与同样截断的 HF 参照实现对拍 logits；显存不足时走 CPU fused_moe_mxfp4 | 真权重数值证据（深度抽样） |
| 9/15–9/20 | 全量回归、文档、最终报告（三档状态：已通过/仅 tiny 通过/未实现） | 报告 + PR |

止损与降级规则：

- 昇腾环境 9/1 不过关 → 主线切性能轨道，昇腾沉没成本控制在 2 天；
- 昇腾算子补齐但 9/8 模型对拍卡住 → 只提算子 PR（基础设施分照拿），
  模型对拍标注未完成进报告；
- 深度抽样对拍若暴露真权重数值问题 → 9/15 前集中修复，仍不行则在
  报告中如实记录残差量级与定位。

本期不做：TP/EP/PP 实现、昇腾真权重、CPU 全量真权重、自费租卡、
DSpark 投机路径、长上下文实测、CUDA Graph 生产化。

## MXFP4 fused MoE 微基准（2026-08-27，RTX 5090）

提前执行冲刺表中的性能微基准项（改在租用的 RTX 5090 而非本地 5060 Ti，
因 5090 环境已就绪；sm_120、CUDA 12.8、torch 2.6.0a0 NGC）。
脚本：`test/bench/bench_fused_moe_mxfp4.py`（可直接复跑）。

方法：真实 V4 MoE 形状（E=256、H=4096、I=2048、topk=6、bf16 激活、
SwigluLimit10），随机打包 FP4 权重（seed 固定）；对照组为"解量化后
常驻 bf16 权重 + 按专家分组 GEMM"的分解实现，两者有效权重逐位一致，
差异只来自 kernel 路径与访存。正确性门：T=8 与逐路由 fp32 参照对拍。

结果：

- 权重常驻显存：MXFP4 3.19 GiB vs bf16 12.00 GiB（3.76x，等于解析值
  (0.5+1/32)/2）；
- 数值：|ref|max=1.9e4，fused 相对误差 3.3e-3，bf16 基线 6.6e-3——
  均在 bf16 舍入量级内，且 fused（fp32 内部点积）比 bf16 基线更接近
  fp32 参照，真实形状下数值无误；
- 延迟（每层 MoE 单次前向）：

| tokens | fused | bf16 分解基线 | speedup |
|---|---|---|---|
| 1 | 1.36 ms | 6.97 ms | 5.14x |
| 8 | 10.79 ms | 12.09 ms | 1.12x |
| 32 | 43.18 ms | 24.54 ms | 0.57x |
| 128 | 172.92 ms | 39.35 ms | 0.23x |
| 512 | 693.08 ms | 41.53 ms | 0.06x |
| 2048 | 2774.77 ms | 42.41 ms | 0.02x |
| 8192 | 11101.15 ms | 44.40 ms | 0.004x |

结论：现 fused_moe_mxfp4 kernel 是"每 (route, 输出元素) 一个 block"
的 decode 向实现——T 个 token 的每条 route 都独立重读对应专家整份
w13（约 4 MiB），权重访存随 T×topk 线性放大（T=8192 时约 200 GiB 级），
因此 decode 段受益（只读 3.2 GiB FP4 而非 12 GiB bf16，快 5.1x）而
prefill 段结构性不适用。clamped SwiGLU（本项目新增部分）不是瓶颈。
V4 服务化部署时 prefill 应走分组 GEMM 路径，或为 kernel 增加专家分组
tiling（后续优化方向，本期不做）。

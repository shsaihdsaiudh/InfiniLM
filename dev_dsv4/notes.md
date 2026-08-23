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

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

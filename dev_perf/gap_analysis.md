# 项目 #2 基线差距清单

最新结论见 v5（2026-08-24）：第二优化项之一落地——融合 per-head
RMSNorm+RoPE（新增 `rms_norm_rope` 算子，已接入 Qwen3 paged 路径），
0.6B 交错 ABBA e2e 收益 -4%~-23%，三层正确性证据齐备。注意本轮在
8GB WSL2 机上完成（非 v1~v4 的 16GB 5060 Ti），收益量级待 16GB 机复测。

---

## v5（2026-08-24）：第二优化项之一落地 —— 融合 per-head RMSNorm+RoPE

动机：v4 遗留的第二优化项——未融合 elementwise 链（~137ms/prefill，
1.7B 口径；v3 nsys：rmsnorm+rope+swiglu 合计 ~157ms / ~590 次小
kernel）。v5 先融合 attention 内的 q/k per-head RMSNorm + RoPE：每个
张量 2 次 kernel launch + 2 遍显存往返 → 1 次，decode 每 token 省
28 层 × 4 = 112 次 launch。

**平台注意**：本轮实验在另一台机器（8GB 卡 + WSL2；实测显存驻留超
~3~4GB 后带宽崩塌至 1~11 GB/s）完成，与 v1~v4 的 16GB 5060 Ti 不是
同一平台，绝对数值不可直接比较。安排：计时 A/B 用 Qwen3-0.6B（小模型
launch 开销占比更高，对融合更敏感），正确性对拍用 Qwen3-1.7B（慢速区
不影响数值）。**收益量级需回 16GB 机复测确认。**

做法：

- InfiniCore 新增 `rms_norm_rope` 融合算子（11 个新文件）：C 层
  `src/infiniop/ops/rms_norm_rope/`（CUDA kernel `cuda/kernel.cuh`：
  fp32 归约 + 逐变体复刻 rope 舍入路径，GPT_J/NEOX × half/bf16）；
  C++ 桥接 `include/infinicore/ops/rms_norm_rope.hpp`。约束：full-rotary
  （head_dim = 2 × table_dim）、pos I32/I64、sin/cos F32。
- InfiniLM 接线（`csrc/models/qwen3/qwen3_attention.cpp`）：
  `forward_paged_` 的 q_norm/k_norm + 两次 rope 共 4 处调用 → 两次
  in-place `rms_norm_rope_`（prefill/decode 共用路径）；
  `forward_static_` 保持未融合链不变。当前仅覆盖 Qwen3 paged 路径。

性能（0.6B，flash-attn，交错 ABBA 控漂移——本机存在"越跑越慢"漂移，
A2 全面慢于 A1，单次先后 A/B 会系统性扭曲差值）：

| 负载 | A1 未融合 | A2 未融合 | B1 融合 | B2 融合 | e2e Δ |
|---|---|---|---|---|---|
| w1 单请求 decode | 1.00s | 1.05s | 0.92s | 1.05s | **-3.9%** |
| w2 长 prefill + 128 decode | 1.49s | 1.72s | 1.20s | 1.26s | **-23.4%** |
| w3 batch32 | 1.56s | 1.77s | 1.33s | 1.35s | **-19.5%** |
| w4 长 decode | 9.06s | 9.65s | 8.01s | 8.14s | **-13.7%** |

机制自洽性：decode 为 launch-bound，每 token 省 112 次 launch × ~10µs
≈ 1.1ms，与 w4 的 -13.7%（8.8→7.9 ms/tok）吻合；w2/w3 的更大收益叠加
了 prefill/批处理段 elementwise 链融合。首轮未控漂移数据方向一致
（paged-attn：w1 -12.9%、w2 -7.9%、w3 -20.0%、w4 -10.8%；flash-attn
w4 -25.0%）。

正确性（三层证据）：

1. **算子级**（`dev_perf/op_check_rms_norm_rope.cpp`：bf16 输入下融合
   kernel vs `rms_norm_` + `rope_` 参考链逐元素对比）：**>99.999% 逐位
   一致，最差 ~3ulp**——差异仅来自融合 kernel 内 fp32 归约的分块顺序。
2. **0.6B e2e 逐 token 对拍**（fused vs unfused，双后端，
   `compare_outputs.py`）：分叉率与"同二进制换后端"噪声底同量级
   （w2 两侧均 exact；w3 28~29/32 vs 底 30/32；w4 分叉为单请求
   late-token 并列翻牌）。
3. **1.7B e2e 逐 token 对拍**（双后端）：分叉位置与噪声底互有先后——
   w2-paged @59、w4-flash @190 与底完全重合，w1-paged 与 w2-flash 的
   fused 全 exact 而底自身分叉（@92 / @59）；目检 w4 分叉处两侧文本
   均为连贯枚举（"从文学角度分析" vs "从人类社会角度分析"），1024
   token 全程连贯。结论：bf16 logit 并列翻牌经自回归放大，非功能错误。

结论与剩余项：

1. rms_norm_rope 融合在 0.6B 上 e2e 收益 -4%~-23%（负载相关，prefill/
   批处理越重收益越大），正确性三层证据齐备；**16GB 机上量级待复测**。
2. 未覆盖：swiglu 融合（v3 口径中 elementwise 链的另一半）、Qwen3
   以外模型、`forward_static_` 路径。
3. 工程配套：bench.py/README 去硬编码路径（INFINI_ROOT 默认
   `~/.infini`，INF_MAIN_PYTHON 可指向其他 checkout 的构建产物）。

复现：

```bash
# 前提：FA 版 InfiniCore（见 v4，rms_norm_rope 已在库中）；增量重建 InfiniLM 扩展
cd <InfiniLM-perf> && xmake build _infinilm && xmake install
HF_HUB_OFFLINE=1 INFINI_ROOT=$HOME/.infini-fa PYTHONPATH=<InfiniCore>/python \
  python dev_perf/bench.py --engine infinilm --num-blocks 64 \
  --attn-backend flash-attn --dump-outputs
python dev_perf/compare_outputs.py <unfused.json> <fused.json>
# 算子级校验的构建/运行命令见 dev_perf/op_check_rms_norm_rope.cpp 头部注释
```

---

## v4（2026-08-24）：B 方向落地 —— prefill 接 FlashAttention-2

按 v3 末尾的 kernel 定位（自研 `PagedAttentionPrefill` 比 FA2 慢 13.5×），
走"直接调 FA2"路线：重建 InfiniCore（`aten=y` +
`--flash-attn=<FA repo>`，FA 取 **v2.7.4.post1**，其 `mha_varlen_fwd` /
`mha_fwd_kvcache` 与 InfiniCore `flash_attention_adaptor.hpp` 逐参数匹配，
原生支持 paged block_table），装到 `~/.infini-fa`；InfiniLM 侧用已有的
`FlashAttentionImpl`（`--attn-backend flash-attn`：prefill 走
`flash::mha_varlen_fwd`，decode 走 `flash::mha_fwd_kvcache`）。

同机同批对照（Qwen3-1.7B，64blk，no-graph；paged 与 flash 跑在同一个
FA 版 InfiniCore 库上，对照干净）：

| 负载 | paged-attn | flash-attn | vLLM（0.85 档） |
|---|---|---|---|
| w1 单请求 decode | 10.86 ms/tok | 11.62 ms/tok | 8.95 ms/tok |
| w2 长 prefill + 128 decode | 2.48s | **2.14s**（单跑复测 1.95s） | 1.50s |
| w3 batch32 总吞吐 | 2191 tok/s | 2127 tok/s | 2957 tok/s |
| w4 长 decode | 11.07 ms/tok | 11.16 ms/tok | 9.71 ms/tok |

Qwen3-4B（64blk，no-graph）w2：paged 5.41s → flash **4.04s**（-25%），
vLLM 参考值 3.84s → 差距收敛到 ~1.05×。

正确性（贪心解码逐 token 对拍，`compare_outputs.py`）：

- **w2（FA varlen paged 路径压力最大的负载）输出与 paged-attn 完全一致，
  也与 vLLM 完全一致**——1.7B、4B 均如此。
- 其余负载的分叉率与 vLLM-vs-paged 的分叉率同量级（w3：24/32 vs 23/32
  exact；w4：两家同在一处 late-token 分叉；目检文本均连贯）——属 bf16
  规约顺序差异，非功能错误。

结论：

1. **prefill 主差距已被 FA2 消化**：w2 e2e 1.7B -14~30%、4B -25%；4B 上
   与 vLLM 基本持平（1.05×）。与 v3 预测（attention 491ms→~36ms）吻合。
2. 剩余差距（1.7B w2 flash 2.14s vs vLLM 1.50s）主要在：未融合
   elementwise 链（~137ms/prefill）与 decode 段（~10%），即 v3 已列的
   第二优化项。
3. 注意：vLLM 0.85 档本次实测峰值 15.9GB，贴 97% 看门狗线，复测建议
   0.72~0.8 档。

复现：

```bash
# 一次性：重建 InfiniCore（约 28min，FA 84 个 .cu 全量编译）
cd /home/yyy/src/InfiniCore   # flash-attention repo 在 /home/yyy/src/flash-attention @ v2.7.4.post1
xmake f -c --aten=y --flash-attn=/home/yyy/src/flash-attention --graph=y \
  --cudnn=n --ccl=n --nv-gpu=y --cpu=y --omp=y \
  --cuda=$HOME/.local/cuda-13.2 --cuda_arch=sm_120 -m release -k shared
xmake build -j6 && xmake install -o ~/.infini-fa

# 运行（INFINI_ROOT 指向 FA 版库，勿覆盖 V4 线在用的 ~/.infini-dsv4）
INFINI_ROOT=$HOME/.infini-fa HF_HUB_OFFLINE=1 \
  /home/yyy/src/InfiniLM/.venv/bin/python dev_perf/bench.py \
  --engine infinilm --model Qwen/Qwen3-1.7B --num-blocks 64 \
  --attn-backend flash-attn --dump-outputs
```

---

## v3（2026-08-24）：num_blocks 扫描与膝点定位

模型 Qwen3-1.7B（bf16），RTX 5060 Ti 16GB（WSL2），全部 no-graph。

| num_blocks | 加载后显存 | w1 ms/tok | w2 e2e | w3 tok/s | w4 ms/tok |
|---|---|---|---|---|---|
| 64 | 7.9GB（48%） | 11.9 / 12.6（两次） | 2.70 / 2.80 | 1995 / 1870 | 12.3 / 14.1 |
| 128 | 9.8GB（60%） | 13.3 | 2.81 | 1852 | 13.9 |
| 256 | 13.3GB（82%） | 13.3 | 2.82 | 1816 | 13.8 |
| 320 | 15.4GB（94%） | 13.5 | 2.81 | 1850 | 13.5 |
| 512（v1 数据） | 16.0GB（98%） | **33.2** | **42.0s** | **42** | **47.9** |

run 间波动约 ±10%（64blk 两次复测所得），64~320 之间的差异在波动范围内。

### v3 结论

1. **劣化是 98% 极端饱和处的悬崖，不是渐变**：48%~94% 全区间性能持平，
   只有 512blk（98%）坠崖。排除"engine 内 O(num_blocks) 的 per-step 开销"
   假设，指向显存近满时分配慢路径/驱动行为（WSL2）。具体机制需 profile
   512blk 配置确认，但该配置会触发本机 97% 显存看门狗，暂被阻塞。
2. **CUDA graph 收益 ~10%**（v2 的 64blk 对照：w1 11.9→10.7、w4 12.3→11.1、
   w3 +7%、w2 -5%），与 num_blocks 无关。值得默认开启，但不是量级差距。
3. **工程缺陷确认**：默认 num_blocks=512 在 16GB 卡上必踩悬崖，且全程无
   告警。可立项方向：cache 预分配按显存自适应（预留 ≥5~10% headroom）或
   近饱和时显式告警。
4. v1 的全部四条差距假设（launch 开销、prefill 路径、批处理串行、decode
   随长度劣化）均为该悬崖的表现，逐条推翻，详见 v1 存档节。

### 机制分析（代码侧，主 checkout 调查结论）

- 分配链：`llm.py` num_blocks → `PagedKVCacheConfig` → 逐层
  `Tensor::zeros({2, num_blocks, 256, kv_heads, head_dim})`
  （`csrc/cache/kv_cache.cpp:142`）。Qwen3-1.7B 为每层 512MiB × 28 = 14GiB，
  与实测吻合。底层是 InfiniCore `PinnableBlockAllocator`——裸 cudaMalloc +
  尺寸分级 free-list 缓存。
- **稳态 decode 每步零新设备分配**：8 个 CPU 输入 tensor 逐个 H2D、各层
  attention workspace、采样 workspace 全部命中分配器缓存；没有任何
  per-step 开销随 num_blocks 增长。512 vs 320 的 3~50× 劣化**不可能是引擎
  算法开销**——代码侧排除了引擎内因素。
- 分配器失败行为是"报错即死"（throw → exit(137)），无重试、无碎片整理、
  无自动 trim；近饱和**全程无任何告警**，只有启动时一行
  `Using Paged KV Cache with num_blocks=512`。
- 加载耗时 59.5s（512blk）vs 3.9s（64blk）跑的是完全相同的代码路径
  （28 次 cudaMalloc(512MB) + 设备清零 + 权重 H2D），15× 差距只能来自
  cudaMalloc/kernel 执行本身，即驱动/内存子系统。
- 综合判断：悬崖在引擎之下——WSL2（dxgkrnl）显存近满时的
  paging/eviction/residency 抖动是最可疑机制，可统一解释"含稳态零新分配
  的 decode 在内全部变慢"。最终确认需 nsys/driver 计数器（被看门狗阻塞）。

### Qwen3-4B 三配置复测（2026-08-24，与 v3 同机）

InfiniLM 均为 64blk；vLLM 为 `--gpu-mem-util 0.72`（13.97GB，86%）。
带宽口径：4B bf16 权重 ~8GB，5060 Ti 理论 decode 上限 ~56 tok/s。

| 负载 | InfiniLM no-graph | InfiniLM graph | vLLM |
|---|---|---|---|
| w1 单请求 decode | 25.6 ms/tok（39.1 tok/s） | 24.3（41.2） | 23.0（43.4） |
| w2 长 prefill + 128 decode | 5.94s | 5.79s | **3.84s** |
| w3 batch32 总吞吐 | 891 tok/s | 944 tok/s | 841 tok/s |
| w4 长 decode | 26.7 ms/tok | 25.0 | 23.7 |
| 加载耗时 / 显存 | 18.0s / 14.86GB（91%） | 37.0s / 14.82GB | 43.3s / 13.97GB |

结论：

1. **"持平"在 4B 上成立**：decode 三家都在带宽上限的 74~77%，w3 InfiniLM
   略优，w4 基本持平。
2. **唯一持续存在的真实差距是长 prefill**：vLLM 比 InfiniLM 快 ~1.5×
   （1.7B 时 ~1.3×）。vLLM 开了 chunked prefill（max_num_batched_tokens=
   8192），这是下一个值得 profile 的点，但量级是 1.5× 而非数量级。
3. graph 在 4B 上收益收窄到 ~5%（w1 25.6→24.3，w4 26.7→25.0）。
4. 4B 64blk 已占 91% 显存仍无悬崖，再次印证悬崖只在 ~98% 极端饱和处；
   vLLM 0.85 档在 16GB 卡上会撞 97% 看门狗（w4 中途被 SIGTERM），0.72 正常。

### prefill 差距的 kernel 级定位（nsys，2026-08-24）

对 w2（3240 tok prefill + 128 decode）在 1.7B / 64blk 下分别抓 InfiniLM
（no-graph）与 vLLM 的 CUDA 轨迹（`results/prof/*.nsys-rep`，分析脚本
`results/prof/slice.py`）。prefill 窗口内 GPU busy 均 ~100%——差距在
kernel 内部，不在调度/Python 开销。

prefill 前向一次的 kernel 时间构成（3240 tokens）：

| 成分 | InfiniLM | vLLM | 倍数 |
|---|---|---|---|
| prefill attention | **491ms**（PagedAttentionPrefillHd128WarpCta8Pipe，26 层 × ~19ms） | 36ms（FA2 splitkv，28 层 × 1.3ms） | **13.5×** |
| rmsnorm + rope + swiglu | ~157ms（未融合，~590 次小 kernel） | ~20ms（triton 融合 kernel） | 7.8× |
| gemm（qkv/o/gate/up/down） | 185ms | 212ms | 持平（略快） |
| **合计** | **888ms / 1655 次调用** | **305ms / 706 次调用** | **2.9×** |

结论：prefill 差距的第一来源是自研 `PagedAttentionPrefill` kernel 比
FlashAttention-2 慢一个数量级（每层 19ms vs 1.3ms），第二来源是
elementwise 链未融合。这是边界清晰、可度量的 kernel 优化目标：
把 prefill attention 换成/优化到 FA2 量级，w2 的 prefill 段理论上可从
~1.2s 压到 ~0.6s（1.7B 口径），e2e 差距从 1.3~1.5× 收敛到接近 1。

### 待办

- [ ] （需看门狗临时放宽）nsys 抓 512blk 的 w1，直接观察 98% 悬崖机制。
- [x] ~~prefill attention kernel 优化立项~~ —— 已由 v4 完成：接 FA2
      （`--attn-backend flash-attn` + FA 版 InfiniCore），w2 的 prefill 段
      差距收敛到 ~1.05~1.3×，正确性逐 token 对拍通过。
- [x] ~~（可选）elementwise 融合（rmsnorm/rope）作为第二优化项~~ ——
      已由 v5 完成：`rms_norm_rope` 算子接入 Qwen3 paged 路径，0.6B
      ABBA e2e -4%~-23%，三层正确性证据齐备；16GB 机复测待做。
- [ ] （可选）swiglu 融合——v5 后 elementwise 链剩余的一半（v3 口径
      rmsnorm+rope+swiglu 合计 ~157ms/prefill）。
- [ ] （可选）flash-attn 设为默认 attention backend 的评估：需在更多模型
      上补正确性对拍，并确认 FP8/滑窗/softcap 模型的回退路径。
- [ ] Qwen3-8B-FP8 下载完成后，可作为 8B 级 + FP8 口径的复测对象。

---

## v2 存档（2026-08-24）：64blk 对照实验原始表

| 负载 | InfiniLM 512blk no-graph | InfiniLM 64blk no-graph | InfiniLM 64blk graph | vLLM（v1 默认） |
|---|---|---|---|---|
| w1 单请求 decode | 33.2 ms/tok | 11.9 ms/tok | **10.7 ms/tok** | 12.0 ms/tok |
| w2 长 prefill + 128 decode | 42.0s | 2.70s | 2.56s | 2.0s |
| w3 batch32 总吞吐 | 42 tok/s | 1995 tok/s | 2136 tok/s | 2166 tok/s |
| w4 长 decode | 47.9 ms/tok | 12.3 ms/tok | 11.1 ms/tok | 12.2 ms/tok |
| 加载耗时 | 59.5s | 3.9s | 11.6s（含 graph 编译） | 24.0s |
| 加载后显存 | 15.96GB | 7.89GB | 8.07GB | 15.89GB |

---

## v1 存档（2026-08-23/24，已被推翻）

数据：`results/infinilm_..._0823_235223.json`、`results/vllm_..._0824_002522.json`。
配置：InfiniLM no-graph、paged-attn、prefix caching on、**num_blocks=512**；
vLLM v1 默认（CUDA graph on、prefix caching on）、gpu_memory_utilization=0.85。

| 负载 | InfiniLM | vLLM | 差距 |
|---|---|---|---|
| w1 单请求 decode | 33.2 ms/tok（30.2 tok/s） | 12.0 ms/tok（83.2 tok/s） | 2.8× |
| w2 长 prefill（3240 tok）+ 128 decode | 42.0s e2e | 2.0s e2e | 21× |
| w3 batch32 × 128 tok | 42.0 tok/s | 2166.0 tok/s | 52× |
| w4 长 decode 1024 tok | 47.9 ms/tok（20.9 tok/s） | 12.2 ms/tok（82.3 tok/s） | 3.9× |

v1 归因假设与判决：

1. ~~单请求 decode 低是缺 CUDA graph 的 launch 开销~~ —— 推翻：no-graph
   64blk 已达 11.9 ms/tok，graph 只再提速 ~10%。
2. ~~w2 是 prefill 路径异常（纯 prefill 估算差距 ~75×）~~ —— 推翻：
   64blk 下 w2 e2e 2.7s，与 vLLM 同量级。
3. ~~w3 批处理存在 per-request 串行开销~~ —— 推翻：64blk 下 w3
   1850~2136 tok/s，扩展性正常。
4. ~~w4 decode 随上下文长度劣化~~ —— 推翻：64blk 下 w4 与 w1 持平。

## 立项书模板（方向定稿后填写）

"在 [模型] + [硬件] + [负载矩阵] 下，[TTFT/TPOT/吞吐] 从 X 提升到 Y（≥Z%），
精度（ceval/mmlu/ppl）不降级，单测与 CI 全绿，benchmark 脚本入库、
他人可复现。"

风险提示：5060 Ti 的瓶颈结构 ≠ 数据中心卡，立项报告中必须注明平台，
收尾时应在数据中心卡（或训练营国产平台）上复测。

# 项目 #2 基线差距清单

最新结论见 v2（2026-08-24）：v1 的四条归因假设被新数据推翻，主差距来源是
KV cache 预分配策略，不是缺 CUDA graph。

---

## v2（2026-08-24）：num_blocks 对照实验

动机：v1 数据采集时 InfiniLM 以默认 num_blocks=512 预分配 ~13GB KV cache，
加载后显存 15.96GB（98%）。本轮用 `--num-blocks 64`（cache ~1.8GB，全程
峰值 ~8.1GB）重跑，并补上了 graph 对照轮。

模型 Qwen3-1.7B（bf16），RTX 5060 Ti 16GB（WSL2）。

| 负载 | InfiniLM 512blk no-graph（v1） | InfiniLM 64blk no-graph | InfiniLM 64blk graph | vLLM（v1 默认） |
|---|---|---|---|---|
| w1 单请求 decode | 33.2 ms/tok | 11.9 ms/tok | **10.7 ms/tok** | 12.0 ms/tok |
| w2 长 prefill + 128 decode | 42.0s | 2.70s | 2.56s | 2.0s |
| w3 batch32 总吞吐 | 42 tok/s | 1995 tok/s | 2136 tok/s | 2166 tok/s |
| w4 长 decode | 47.9 ms/tok | 12.3 ms/tok | 11.1 ms/tok | 12.2 ms/tok |
| 加载耗时 | 59.5s | 3.9s | 11.6s（含 graph 编译） | 24.0s |
| 加载后显存 | 15.96GB | 7.89GB | 8.07GB | 15.89GB |

数据文件：`results/infinilm_..._0824_005235.json`（64blk no-graph）、
`results/infinilm_..._0824_005308.json`（64blk graph）。

### v2 结论

1. **num_blocks 是一阶变量**。只改 512→64，w1/w2/w3/w4 全部从"2.8×~52×
   落后"变成与 vLLM 持平（graph 下 w1/w4 还略快）。v1 的四条假设
   （launch 开销、prefill 路径、批处理串行、decode 随长度劣化）全部不成立
   或降级为次要因素。
2. **CUDA graph 真实收益约 10%**：同 64blk 口径，w1 11.9→10.7 ms/tok、
   w4 12.3→11.1 ms/tok、w3 +7%、w2 -5%。值得开，但不是量级差距。
3. **默认 num_blocks=512 在小显存卡上是陷阱**：cache 按 13 万 token 预占满
   卡，剩余空间不足导致全负载 3~50× 劣化，且无任何告警/日志。机制待
   profile 确认（疑似近满显存下分配/驱动慢路径，或 engine 内与 block 数
   相关的 per-step 开销）。这本身可立项：cache 容量自适应/告警。
4. 64blk 口径下 InfiniLM 与 vLLM 已持平，说明 1.7B 规模 kernel 与调度
   没有本质差距；原计划"找大差距"在消费卡小模型上不成立，立项方向应转向
   (a) cache 预分配策略缺陷修复，(b) 更大模型/更高并发下重新找差距。

### 待办

- [ ] 膝点定位：num_blocks ∈ {128, 256, 384} 各跑一轮，找到劣化拐点
      （256 峰值约 12GB，需 GPU 较空窗口）。
- [ ] nsys 对比 512blk vs 64blk 的 w1，确认劣化机制（kernel 变慢还是
      每步多了主机侧/分配开销）。
- [ ] 换 8B 级模型（本地已有 Qwen3-4B；Qwen3-8B-FP8 下载中）在 64blk
      口径下复测，检验"持平"结论在大模型上是否仍成立。
- [ ] 若转向 cache 预分配立项：先读 paged cache 分配路径
      （`llm.py` num_blocks → engine paged cache），确认 512 劣化的代码机制。

---

## v1（2026-08-23/24，已被 v2 推翻，存档保留）

数据：`results/infinilm_..._0823_235223.json`、`results/vllm_..._0824_002522.json`。
配置：InfiniLM no-graph、paged-attn、prefix caching on、**num_blocks=512**；
vLLM v1 默认（CUDA graph on、prefix caching on）、gpu_memory_utilization=0.85。

| 负载 | InfiniLM | vLLM | 差距 |
|---|---|---|---|
| w1 单请求 decode | 33.2 ms/tok（30.2 tok/s） | 12.0 ms/tok（83.2 tok/s） | 2.8× |
| w2 长 prefill（3240 tok）+ 128 decode | 42.0s e2e | 2.0s e2e | 21× |
| w3 batch32 × 128 tok | 42.0 tok/s | 2166.0 tok/s | 52× |
| w4 长 decode 1024 tok | 47.9 ms/tok（20.9 tok/s） | 12.2 ms/tok（82.3 tok/s） | 3.9× |

v1 归因假设（对照 v2 数据后的判决）：

1. ~~单请求 decode 低是缺 CUDA graph 的 launch 开销~~ —— **推翻**：
   no-graph 64blk 已达 11.9 ms/tok，graph 只再提速 ~10%。
2. ~~w2 是 prefill 路径异常（纯 prefill 估算差距 ~75×）~~ —— **推翻**：
   64blk 下 w2 e2e 2.7s，与 vLLM 同量级。
3. ~~w3 批处理存在 per-request 串行开销~~ —— **推翻**：64blk 下 w3
   1995~2136 tok/s，扩展性正常。
4. ~~w4 decode 随上下文长度劣化~~ —— **推翻**：64blk 下 w4 与 w1 基本持平。

v1 的全部四条差距都是 num_blocks=512 预占满卡引发的系统性劣化的表现。

## 立项书模板（方向定稿后填写）

"在 [模型] + [硬件] + [负载矩阵] 下，[TTFT/TPOT/吞吐] 从 X 提升到 Y（≥Z%），
精度（ceval/mmlu/ppl）不降级，单测与 CI 全绿，benchmark 脚本入库、
他人可复现。"

风险提示：5060 Ti 的瓶颈结构 ≠ 数据中心卡，立项报告中必须注明平台，
收尾时应在数据中心卡（或训练营国产平台）上复测。

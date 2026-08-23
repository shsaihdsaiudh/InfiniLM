# 项目 #2 基线差距清单 v1

日期：2026-08-24。数据：`results/infinilm_*.json`（2026-08-23 23:52）与
`results/vllm_*.json`（2026-08-24 00:25），模型 Qwen3-1.7B（bf16），
硬件 RTX 5060 Ti 16GB（WSL2）。

## 配置口径（读表前必看）

| | InfiniLM | vLLM |
|---|---|---|
| 配置 | no-graph、paged-attn、prefix caching on、num_blocks=512 | v1 默认（CUDA graph on、prefix caching on）、gpu_memory_utilization=0.85 |
| 加载耗时 | 59.5s（权重 38.9s） | 24.0s |
| 加载后显存 | 15.96GB | 15.89GB |

两边显存都接近吃满但口径不同：InfiniLM 是 paged cache 预分配
（512 blocks × 256 tok ≈ 13 万 token KV ≈ 13GB），vLLM 是按 0.85 比例占满。
比较性能时不受此影响，比较显存时需注意。

## 对比表

| 负载 | InfiniLM | vLLM | 差距 |
|---|---|---|---|
| w1 单请求 decode | 33.2 ms/tok（30.2 tok/s） | 12.0 ms/tok（83.2 tok/s） | **2.8×** |
| w2 长 prefill（3240 tok）+ 128 decode | 42.0s e2e | 2.0s e2e | **21×** |
| w3 batch32 × 128 tok | 42.0 tok/s | 2166.0 tok/s | **52×** |
| w4 长 decode 1024 tok | 47.9 ms/tok（20.9 tok/s） | 12.2 ms/tok（82.3 tok/s） | **3.9×** |

## 初步归因假设（待 profile 验证）

1. **单请求 decode 远低于带宽上限**。5060 Ti 显存带宽 ~448GB/s，1.7B bf16
   理论上限约 130 tok/s；vLLM 83 tok/s（64%），InfiniLM 30 tok/s（23%）。
   首要嫌疑是无 CUDA graph 下的逐步 launch/调度开销——graph 对照轮数据
   落地后可定量（见"待补数据"）。
2. **w2 是纯 prefill 差距，最异常**。用 w1 的 decode 速率扣掉 128 步 decode
   （≈4.2s），InfiniLM prefill 约 37.8s ≈ 86 tok/s；vLLM 同样扣减后
   ≈0.5s ≈ 6500 tok/s，纯 prefill 差距约 **75×**。3.2k token 的 prefill
   不应慢到这种程度，疑似 prefill 走了非专用路径（逐 token？无 chunked
   prefill？paged-attn 的 prefill 实现低效？），是头号 profile 目标。
3. **w3 批处理几乎无扩展性**。InfiniLM 32 并发总吞吐 42 tok/s，仅为单请求
   的 1.4 倍；128 步耗时 97.5s ≈ 762ms/step（单请求 33ms/step 的 23 倍）。
   每步开销随请求数近线性增长，疑似调度/采样/请求簿记存在 per-request
   串行开销（Python 层嫌疑大于 kernel 层）。vLLM 同负载 2166 tok/s。
4. **w4 decode 随上下文增长劣化**。InfiniLM 从 33→48ms/tok（+45%），vLLM
   基本持平（12.0→12.2）。1k token 的 KV 仅 ~112MB，带宽解释不了 +15ms，
   疑似 decode attention kernel 或 cache 布局随长度低效。

## 待补数据（需要 GPU 空闲窗口）

- [ ] InfiniLM `--enable-graph` 对照轮（验证假设 1）。
      注意：2026-08-24 凌晨两次尝试在权重加载后静默退出（exit -1，
      无 traceback），疑似 graph 路径在 16GB 卡上 capture 时资源不足
      或原生崩溃，下次用 `--num-blocks 64` 低显存模式重试以区分。
- [ ] nsys 抓 w1/w2/w3，导出 kernel 时间分布，区分 kernel vs
      Python/调度开销（直接检验假设 2、3）。
- [ ] ncu 对 top-3 热点 kernel 做 SOL 段分析。
- [ ] （可选）换 8B 级模型复测一轮，接近原计划 D3-4 的负载设定。

## 立项书模板（profile 完成后填写）

"在 [模型] + [硬件] + [负载矩阵] 下，[TTFT/TPOT/吞吐] 从 X 提升到 Y（≥Z%），
精度（ceval/mmlu/ppl）不降级，单测与 CI 全绿，benchmark 脚本入库、
他人可复现。"

风险提示：5060 Ti 的瓶颈结构 ≠ 数据中心卡，立项报告中必须注明平台，
收尾时应在数据中心卡（或训练营国产平台）上复测。

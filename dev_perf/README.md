# dev_perf — 项目 #2(性能优化)基线工具

离线双后端基线压测:同一负载矩阵跑 InfiniLM 与 vLLM,产出差距清单,作为 #2 立项依据。

## 运行前提

1. InfiniCore 已构建且 InfiniLM 已在主 checkout 构建(`python/infinilm/lib/_infinilm*.so` 存在)。运行时动态库路径由 bench.py 自动 re-exec 注入(LD_LIBRARY_PATH ← InfiniCore/InfiniLM 的 lib 目录),无需手动设置
2. vLLM 独立环境:`/home/yyy/src/venvs/vllm-bench`(uv 创建,cu128 torch;**勿装进项目 .venv**,vllm 会锁定 torch 版本)。运行时需带齐 `PATH=<vllm-bench>/bin:$HOME/.local/cuda-13.2/bin:$PATH` 和 `CUDA_HOME=$HOME/.local/cuda-13.2`(否则 EngineCore 找不到 ninja 起不来)
3. GPU 空闲(本机常驻 llama-server 约占 11.4GB,压测前需要停);本机有 97% 显存看门狗,vLLM 侧 `--gpu-mem-util` 不要超过 ~0.8(0.85 × 16GB + 基线占用会撞线)
4. WSL2 下 vLLM 需 `VLLM_WSL2_ENABLE_PIN_MEMORY=1`(bench.py 已自动设置;否则 vLLM 禁用 pinned memory 导致 UvaBuffer 报 "UVA is not available")

## 运行

```bash
# InfiniLM(用项目 .venv,通过 sys.path 引用主 checkout 的构建产物)
/home/yyy/src/InfiniLM/.venv/bin/python dev_perf/bench.py --engine infinilm --model Qwen/Qwen3-1.7B

# vLLM(独立 venv)
/home/yyy/src/venvs/vllm-bench/bin/python dev_perf/bench.py --engine vllm --model Qwen/Qwen3-1.7B
```

结果 JSON 写入 `dev_perf/results/`。

低显存模式（显存紧张时用）：InfiniLM 侧加 `--num-blocks 64` 可把 paged cache
预分配从 ~13GB 降到 ~1.8GB（Qwen3-1.7B 口径），w1-w4 负载矩阵在 16k token
容量内仍可完整运行；num_blocks 会记入结果 JSON。
注意：num_blocks 对性能有一阶影响——512 在 16GB 卡上会把显存占满并导致
全负载严重劣化（见 gap_analysis.md v2），跨轮次对比必须用相同 num_blocks。

## 负载矩阵

| 负载 | 请求 | 输出 | 考察点 |
|---|---|---|---|
| w1_short_decode | 1 × 短 prompt | 128 tok | 单请求 decode 延迟(ms/tok) |
| w2_long_prefill | 1 × ~2k tok prompt | 128 tok | prefill 吞吐 |
| w3_batch32 | 32 × 短 prompt | 128 tok | 批处理总吞吐 |
| w4_long_decode | 1 × 短 prompt | 1024 tok | 长生成 decode 稳定性 |

## 公平性约定

- 贪心解码(temperature=0, top_k=1)、ignore_eos=True、逐字节相同的 prompt 与 max_tokens
- vLLM 用 v1 默认(CUDA graph 开、prefix caching 开);InfiniLM 用 enable_graph=False、paged-attn、prefix caching 开——双方配置记入结果 JSON,差距解读时先考虑这两项
- 每个引擎先跑一轮 w1 预热(不计时),再正式计时

## 产出

差距清单（哪个负载、差多少、可能原因）→ 据此决定 #2 是否转正立项，以及立项的可度量目标。

v1 已产出：见 `dev_perf/gap_analysis.md`（含对比表、归因假设、待补数据清单和立项书模板）。

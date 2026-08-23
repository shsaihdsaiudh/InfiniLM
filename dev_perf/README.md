# dev_perf — 项目 #2(性能优化)基线工具

离线双后端基线压测:同一负载矩阵跑 InfiniLM 与 vLLM,产出差距清单,作为 #2 立项依据。

## 运行前提

1. InfiniCore 已安装且 InfiniLM 已在主 checkout 构建(`python/infinilm/lib/_infinilm*.so` 存在)
2. vLLM 独立环境:`/home/yyy/src/venvs/vllm-bench`(uv 创建,cu128 torch;**勿装进项目 .venv**,vllm 会锁定 torch 版本)
3. GPU 空闲(本机常驻 llama-server 约占 11.4GB,压测前需要停)

## 运行

```bash
# InfiniLM(用项目 .venv,通过 sys.path 引用主 checkout 的构建产物)
/home/yyy/src/InfiniLM/.venv/bin/python dev_perf/bench.py --engine infinilm --model Qwen/Qwen3-1.7B

# vLLM(独立 venv)
/home/yyy/src/venvs/vllm-bench/bin/python dev_perf/bench.py --engine vllm --model Qwen/Qwen3-1.7B
```

结果 JSON 写入 `dev_perf/results/`。

## 负载矩阵

| 负载 | 请求 | 输出 | 考察点 |
|---|---|---|---|
| w1_short_decode | 1 × 短 prompt | 128 tok | 单请求 decode 延迟(ms/tok) |
| w2_long_prefill | 1 × ~2k tok prompt | 128 tok | prefill 吞吐 |
| w3_batch32 | 32 × 短 prompt | 128 tok | 批处理总吞吐 |
| w4_long_decode | 1 × 短 prompt | 1024 tok | 长生成 decode 稳定性 |

## 公平性约定

- 贪心解码(temperature=0, top_k=1)、ignore_eos=True、逐字节相同的 prompt 与 max_tokens
- vLLM 用 v1 默认(CUDA graph 开、prefix caching 开);InfiniLM 用 enable_graph=False、prefix caching 开——双方配置记入结果 JSON,差距解读时先考虑这两项
- 每个引擎先跑一轮 w1 预热(不计时),再正式计时

## 产出

差距清单(哪个负载、差多少、可能原因)→ 据此决定 #2 是否转正立项,以及立项的可度量目标。

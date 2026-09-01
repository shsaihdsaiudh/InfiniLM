# dev_perf — 项目 #2(性能优化)基线工具

离线双后端基线压测:同一负载矩阵跑 InfiniLM 与 vLLM,产出差距清单,作为 #2 立项依据。

## 运行前提

1. InfiniCore 已构建安装（`INFINI_ROOT` 指向安装前缀，默认 `~/.infini`；flash-attn 后端需要 ATen+FA 版，见 gap_analysis.md v4），且当前 checkout 已构建 C++ 扩展（`xmake build _infinilm && xmake install`，产物在 `python/infinilm/lib/`）。运行时动态库路径由 bench.py 自动 re-exec 注入（LD_LIBRARY_PATH ← InfiniCore/InfiniLM 的 lib 目录），无需手动设置
2. vLLM 独立环境：自建 venv（uv 创建，cu128 torch；**勿装进项目 venv**，vllm 会锁定 torch 版本）。运行时需带齐 `PATH=<vllm-venv>/bin:$CUDA_HOME/bin:$PATH` 和 `CUDA_HOME`（否则 EngineCore 找不到 ninja 起不来）
3. GPU 空闲（压测前停掉常驻的 GPU 进程）；如本机有显存看门狗，vLLM 侧 `--gpu-mem-util` 不要超过 ~0.8
4. WSL2 下 vLLM 需 `VLLM_WSL2_ENABLE_PIN_MEMORY=1`（bench.py 已自动设置；否则 vLLM 禁用 pinned memory 导致 UvaBuffer 报 "UVA is not available"）

## 运行

```bash
# InfiniLM（任何带 torch 的 venv，通过 sys.path 引用当前 checkout 的构建产物）
python dev_perf/bench.py --engine infinilm --model Qwen/Qwen3-1.7B

# vLLM（独立 venv）
/path/to/vllm-venv/bin/python dev_perf/bench.py --engine vllm --model Qwen/Qwen3-1.7B
```

结果 JSON 写入 `dev_perf/results/`。

InfiniLM 侧常用参数：

- `--num-blocks 64`：低显存模式。paged cache 预分配从 ~13GB 降到 ~1.8GB
  （Qwen3-1.7B 口径），w1-w4 在 16k token 容量内仍可完整运行；num_blocks
  会记入结果 JSON。注意 num_blocks 对性能有一阶影响——512 在 16GB 卡上会
  把显存占满并导致全负载严重劣化（见 gap_analysis.md v3），跨轮次对比必须
  用相同 num_blocks。
- `--attn-backend flash-attn`：prefill/decode 改用 FlashAttention-2（需
  ATen+FA 版 InfiniCore，见 gap_analysis.md v4 的构建与 `INFINI_ROOT`
  用法）。长 prefill 负载收益 ~20~30%。
- `--attn-backend hybrid`：prefill 走 FA2 varlen、decode 走自研 paged
  kernel（经 strides 直读 FA 的 BSHD cache）的分离路由。5090 上为最优
  单配置（见 gap_analysis.md v9）。
- `--only w2_long_prefill`：只跑指定负载（逗号分隔）。
- `--concurrent-prefill-n N`：w5_concurrent_prefill 的并发长 prompt 条数
  （默认 8，复用 w2 的 prompt 构造、逐条加不同前缀以避开 prefix caching
  去重），chunked prefill 验收负载，`--only w5_concurrent_prefill` 单跑。
- `--decode-stall-n N`：w6_decode_stall 的 decode 长流并发数（默认 8）。
  w6 是引擎级负载：N 条短 prompt 各生成 512 tok，第 64 步时注入一条
  ~6.5k tok 长 prompt（带 nonce 头避开 prefix caching），记录逐步耗时，
  考察注入前后 decode 流的最大/p90 ITL 尖峰与注入请求 TTFT——FCFS 下
  整条 decode 流被整段 prefill 堵住，chunked prefill 应把尖峰摊平。
  仅 infinilm 引擎支持（vLLM 跳过）。
- `--dump-outputs`：把每个请求的输出 token ids 记入 JSON，配合
  `compare_outputs.py a.json b.json ...` 做跨引擎/跨 backend 的贪心解码
  逐 token 对拍（exact match 数 + 最早分叉位置）。

## 负载矩阵

| 负载 | 请求 | 输出 | 考察点 |
|---|---|---|---|
| w1_short_decode | 1 × 短 prompt | 128 tok | 单请求 decode 延迟(ms/tok) |
| w2_long_prefill | 1 × ~2k tok prompt | 128 tok | prefill 吞吐 |
| w3_batch32 | 32 × 短 prompt | 128 tok | 批处理总吞吐 |
| w4_long_decode | 1 × 短 prompt | 1024 tok | 长生成 decode 稳定性 |
| w5_concurrent_prefill | 8 × ~2k tok prompt 并发 | 128 tok | chunked prefill：并发长 prefill 不阻塞 decode（e2e wall time） |
| w6_decode_stall | 8 × 短 prompt 长 decode + 中途注入 1 × ~6.5k tok prompt | 512/32 tok | chunked prefill 收益场景：decode 流 ITL 尖峰、注入请求 TTFT |

## 公平性约定

- 贪心解码(temperature=0, top_k=1)、ignore_eos=True、逐字节相同的 prompt 与 max_tokens
- vLLM 用 v1 默认(CUDA graph 开、prefix caching 开);InfiniLM 用 enable_graph=False、paged-attn、prefix caching 开——双方配置记入结果 JSON,差距解读时先考虑这两项
- 每个引擎先跑一轮 w1 预热(不计时),再正式计时

## 产出

差距清单（哪个负载、差多少、可能原因）→ 据此决定 #2 是否转正立项，以及立项的可度量目标。

最新进展见 `dev_perf/gap_analysis.md`（v8：5090 全栈合流——FA2+融合
ABBA -5%~-13%，并发现 FA decode 在 Blackwell 上慢于自研 splitkv，默认
后端结论修正；v7 elementwise 链 kernel 级闭环；含对比表、归因、复现
命令和立项书模板）。

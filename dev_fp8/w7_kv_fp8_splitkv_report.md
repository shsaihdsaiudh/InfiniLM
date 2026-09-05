# W7: KV FP8 decode split-kv（长上下文残留差距收敛）

> 目标：抹平 W6 FP8 KV decode 在 bs=1 长上下文的残留开销（1.05×/1.14×/1.27× @ in=1k/4k/16k，
> 见 `w6_kv_fp8_report.md` §4）。根因：FP8 decode kernel 是 CTA per (seq, q_head)，bs=1 时全卡仅
> 32 CTA（Qwen3-8B 32 q head），170 个 SM 只占 19%；对照 BF16 tuned kernel 有 4 路 split-kv。
>
> **状态：GPU 验证已完成（2026-09-05，RTX 5090，sm_120）——算子对拍全绿，bs=1 in=4k/16k 残留差距从
> 1.14×/1.27× 抹平并反超至 0.77×/0.58×，E2E 贪婪对拍与 BF16 KV 逐 token 一致。见 §4。**

## 1. 方案（flash-decoding 标准做法）

token 维切成 `num_splits` 个连续 shard，grid 加 z 维（CTA per (split, seq, q_head)），
每个 CTA 输出局部 `(m, l, 未归一化 acc)` 到 workspace，再由 combine kernel 做跨 CTA
log2 域在线 softmax 合并并除以 `l + 1e-6`。写入侧（paged_caching 量化 kernel）不动。

- **kernel_fp8.cuh**（v2）：
  - `flashAttentionDecodeFp8Kernel` 追加 `partial_acc/m/l` 与 `num_splits` 参数；
    shard 区间 `[split_idx*shard, min(seq_len, +shard))`，cursor 初始/推进改走
    `[tokenBegin, tokenEnd)` 窗口（`num_splits==1` 时与 v1 逐 token 等价，已模拟验证）；
    CTA 内 32-warp 合并逻辑不变，split 模式下改写 partial（不除 l）。
  - 空 shard（`seq_len < num_splits` 或末 shard 越界）产出中性元（m=-inf, l=0, acc=0）：
    cross-warp 合并的 wgt 加了 `-inf` 显式判空，避免 `exp2f(-inf - -inf)` NaN
    （非 split 路径数值与 v1 完全一致——原来 `exp2f(-inf - 有限值)` 本来就等于 0）。
  - 新增 `flashAttentionDecodeFp8SplitKvCombineKernel`：CTA per (seq, q_head)，
    HEAD_SIZE 线程，log2 域合并 ≤8 个 shard 后写出 F16/BF16。
- **workspace 布局**与 F16/BF16 家族（kernel_v2.cuh）一致：
  `partial_acc [kMaxSplits=8, num_seqs, num_heads, head_size]` +
  `partial_m/l [8, num_seqs, num_heads]`，F32。`Descriptor::create` 对 FP8 从 0 改为
  `8 * num_seqs * num_heads * (value_size + 2) * 4B`（bs=1 32头 hd128 仅 ~133KB），
  引擎侧经 `INFINIOP_WORKSPACE_TENSOR` 自动分配，InfiniLM 无需改动。
- **split 策略**（launcher，对齐 BF16 家族的 env 开关）：
  默认 auto——FA2 风格 waves 启发式（`seqlen_k ≈ max_blocks*pbs` 上界，SM 数实时查询），
  不划算则不 split（bs=8 大 grid 自动保持单 pass）；
  `INFINIOP_FLASH_DECODE_SPLITKV=0/1/auto`、`INFINIOP_FLASH_NUM_SPLITS=1..8/auto` 覆盖；
  `INFINIOP_FLASH_DEBUG_SPLITS=1` 打印决策。

## 2. 改动文件（InfiniCore，feat/fp8-blockwise-dequantize 分支）

- `src/infiniop/ops/paged_attention/cuda/kernel_fp8.cuh`：split-kv + combine kernel。
- `src/infiniop/ops/paged_attention/nvidia/paged_attention_fp8.cu`：launcher 两阶段
  launch、workspace 切分与充分性检查（不足返回 INSUFFICIENT_WORKSPACE）、split 策略。
- `src/infiniop/ops/paged_attention/nvidia/paged_attention_nvidia.cu`：FP8 的 workspace
  预留（去掉 F8 特例 0），`launch_decode_fp8_*` 声明与 `CALCULATE_FP8` 透传 workspace。
- `test/infiniop/paged_attention.py`：FP8 用例追加 `(1, 8, 8, 128, 16, 4096, alibi=True)`，
  稳定触发 split 路径并覆盖 alibi+split 组合（既有用例随机 seq_len 已覆盖空 shard）。

## 3. 已完成的验证（无 GPU）

CPU 模拟（逐行复刻 cursor 推进 + 两级在线 softmax 合并 + -inf 判空）：
- 对 seq_len × pbs × num_splits 全组合，(split, warp)→token 分配恰好划分 `[0, seq_len)`，
  且 `num_splits==1` 与 v1 分配逐点一致；
- 300 组随机数值（含 alibi、hd 64/128、空 shard），split+combine 结果与直接 softmax
  相对误差 ≤1e-6。

## 4. GPU 验证结果（2026-09-05，RTX 5090，sm_120，lib 由 InfiniCore@2f931ee2 构建）

**算子对拍**（`test/infiniop/paged_attention.py --nvidia`）：28 个 case 全部 PASS，
含新增 `(1, 8, 8, 128, 16, 4096, alibi=True)`。`INFINIOP_FLASH_DEBUG_SPLITS=1` 确认
split 路径在 GPU 真实触发（该用例 num_splits=8），大 grid 用例（heads=40 seqs=4）
正确保持 num_splits=1。

**E2E 性能**（examples/bench.py，Qwen3-8B-FP8 权重，out=128，Decode Avg ITL，ms）：

| 场景 | KV BF16 | FP8 split=auto | FP8/BF16 | FP8 split=0 | W6 FP8/BF16 |
|---|---|---|---|---|---|
| bs=1 in=1024 | 7.85 | 7.27 | **0.93×** | 8.35 | 1.05× |
| bs=1 in=4096 | 10.29 | 7.96 | **0.77×** | 11.92 | 1.14× |
| bs=1 in=16384 | 20.58 | 11.93 | **0.58×** | 26.47 | 1.27× |
| bs=8 in=1024 | 20.84 | 22.84 | 1.10× | 22.00 | 1.05× |
| bs=8 in=4096 | 25.31 | 29.09 | 1.15× | 29.76 | 1.17× |
| bs=8 in=16384 | warmup OOM（bench 既有行为，两 dtype 一致，与 W6 相同） | | | | |

- 主目标超额完成：bs=1 in=4k/16k 从 1.14×/1.27× **抹平并反超**至 0.77×/0.58×——
  split-kv 补齐并行度后，FP8 KV 带宽减半开始净赚；bs=1 in=1k 同步改善（1.05×→0.93×）。
- 回归对照成立：SPLITKV=0 与 W6 逐点吻合（11.92 vs 11.94、26.47 vs 26.42、29.76 vs 29.82），
  同会话 BF16 基线与 W6 偏差 ≤1%——收益全部来自 split-kv，无其他变量。
- split 决策（auto）：bs=1 各长度 num_splits=5；bs=8 num_splits=7；TTFT 不变（prefill 未动）。

**启发式扫描**（INFINIOP_FLASH_NUM_SPLITS 定点复测）：

- bs=1/16k：auto(5)=11.93 优于固定 8=12.24，auto 已最优。
- bs=8/4k：splits=1→29.76 / 2→29.82 / 4→29.08 / auto(7)→29.09，auto 已在最优点。
- bs=8/1k：splits=1→22.00 < 2→22.37 < 4→22.62 < auto(7)→22.84 —— 唯一过拆 cell（+3.8%）。
  根因：共享 makespan 模型按 1 CTA/SM 估 wave，未计入 FP8 kernel 1024 线程 CTA 的
  2 CTA/SM 共驻（bs=8 时 256 block 实际一波即可装下）。改动需 fork 与 BF16 家族共享的
  `chooseNumSplitsHeuristic`、绝对差距 <1ms，维持 auto 不改；bs=8 短上下文敏感场景可用
  `INFINIOP_FLASH_DECODE_SPLITKV=0` 规避。

**E2E 贪婪对拍**（`dev_fp8/results/w7_e2e_parity.py`，4020-token prompt + 48 token 生成，
top_k=1）：FP8 KV（DEBUG_SPLITS 确认 split=5 生效）与 BF16 KV 输出**逐 token 完全一致**
（PARITY: MATCH）。注：examples/test_infer.py 的 LLM 封装不支持 kv_cache_dtype，
对拍脚本直接驱动 InferEngine（与 bench.py 同路径）。

**精度链**：PPL/C-Eval/MMLU 走 prefill 路径（未动），沿用 W6 结论；decode 数值由
算子对拍（相对误差容差内，含 split+combine 组合）与 E2E 逐 token 一致覆盖。

**结论：KV FP8 线（W6+W7）闭环。**

复跑命令（脚本已 push 至服务器 /root/fp8/ 与 /root/fp8/InfiniLM/dev_fp8/results/）：

```bash
python3 dev_fp8/results/fp8_ssh.py bg w7_build 'bash /root/fp8/fp8_build_core.sh'
python3 dev_fp8/results/fp8_ssh.py run 'source /root/fp8/fp8_env.sh && cd /root/fp8/InfiniCore && python3 test/infiniop/paged_attention.py --nvidia'
python3 dev_fp8/results/fp8_ssh.py bg w7_bench 'bash /root/fp8/w7_splitkv_bench.sh'
# 原始日志：/root/fp8/eval_logs/w7_{optest,bench_*,e2e_parity}.log
```

## 5. 风险与注意

- split 决策用容量上界（`max_blocks*pbs`）而非实际 cache_len，与 BF16 家族一致；
  bench 场景容量≈实际长度，无偏差。
- combine kernel 读 partial 的流量为 `8×seqs×heads×(hd+2)×4B`，bs=1 时 ~133KB，可忽略。
- 其他 vendor 后端的 FP8 路径仍 NOT_IMPLEMENTED，未动。

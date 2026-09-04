# W7: KV FP8 decode split-kv（长上下文残留差距收敛）

> 目标：抹平 W6 FP8 KV decode 在 bs=1 长上下文的残留开销（1.05×/1.14×/1.27× @ in=1k/4k/16k，
> 见 `w6_kv_fp8_report.md` §4）。根因：FP8 decode kernel 是 CTA per (seq, q_head)，bs=1 时全卡仅
> 32 CTA（Qwen3-8B 32 q head），170 个 SM 只占 19%；对照 BF16 tuned kernel 有 4 路 split-kv。
>
> **状态：代码已完成并通过 CPU 逻辑模拟；服务器（RTX 5090）暂不可用，构建/对拍/bench 待补。**

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

## 4. 待服务器恢复后执行

```bash
# 1) 同步并构建（脚本在 dev_fp8/results/，注意先 push 到 /root/fp8/）
python3 dev_fp8/results/fp8_ssh.py push <InfiniCore 改动> /root/fp8/InfiniCore/
python3 dev_fp8/results/fp8_ssh.py bg w7_build 'bash /root/fp8/fp8_build_core.sh'

# 2) 算子对拍（FP8 case 必须全绿，含新 alibi+split 用例）
python3 dev_fp8/results/fp8_ssh.py run 'cd /root/fp8/InfiniCore && python3 test/infiniop/paged_attention.py --nvidia'

# 3) E2E 性能：bs=1/8 × in=1k/4k/16k，FP8 KV 对比 W6 表（bench.py --kv-cache-dtype fp8）
#    另跑 INFINIOP_FLASH_DEBUG_SPLITS=1 确认各场景 num_splits 决策。
#    回归对照：INFINIOP_FLASH_DECODE_SPLITKV=0（应回到 W6 数字）。
```

预期：bs=1 in=4k/16k 的 1.14×/1.27× 基本收敛到 ~1.0×；bs=1 in=1k 与 bs=8 场景不退化
（启发式在大 grid 下不 split；combine 为微秒级小 kernel）。精度指标（PPL/C-Eval/MMLU）
数值语义不变，可选跑 `w6_accuracy_chain.sh` 抽查。

## 5. 风险与注意

- split 决策用容量上界（`max_blocks*pbs`）而非实际 cache_len，与 BF16 家族一致；
  bench 场景容量≈实际长度，无偏差。
- combine kernel 读 partial 的流量为 `8×seqs×heads×(hd+2)×4B`，bs=1 时 ~133KB，可忽略。
- 其他 vendor 后端的 FP8 路径仍 NOT_IMPLEMENTED，未动。

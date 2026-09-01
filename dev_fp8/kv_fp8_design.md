# W6: KV Cache FP8(E4M3) 量化设计

目标：paged attention 路径的 KV cache FP8 动态量化，cache 显存减半、decode 注意力读带宽减半，
对 BF16 基线精度贴近。不涉及现有 INT8 static_attn 路径。

## 数据布局

- K/V cache：沿用 paged 默认后端布局 `[2, num_blocks, num_kv_heads, block_size, head_dim]`，
  元素 dtype = F8(E4M3)。
- scale：每层两张 F32 张量 `k_scale`、`v_scale`，布局 `[num_blocks, num_kv_heads, block_size]`，
  即 **per-token-per-kv-head** 一个标量（对 head_dim 维做 amax）。
- 量化语义（写入时，post-RoPE）：
  `amax = max(|x[0:head_dim]|)`；`scale = amax / 448`（amax==0 时 scale=1，q 全 0）；
  `q = e4m3_encode(x / scale)`。反量化：`x ≈ e4m3_decode(q) * scale`。
- E4M3 编解码复用 InfiniCore 已有 `infiniopFp8E4m3Encode/Decode`
  （`src/infiniop/ops/.../nvidia_kernel_common.cuh:34-59`）。

## InfiniCore 改动（三个算子统一模式）

对 `paged_caching` / `paged_attention` / `paged_attention_prefill` 三个算子，
在现有 API 尾部追加**可选** scale 参数（NULL = 不存在），BF16/F16 路径零回归：

1. C API（`include/infiniop/ops/*.h`）：create 追加 `k_scale_desc, v_scale_desc`；
   execute 追加 `k_scale, v_scale` 数据指针（放在 stream 参数之前）。
2. `src/infiniop/ops/*/info.h`：校验规则——
   cache dtype == F8 时两个 scale 必须存在、dtype F32、shape `[num_blocks, num_kv_heads, block_size]`；
   cache dtype != F8 时两个 scale 必须为 NULL。
3. `src/infiniop/ops/*/operator.cc`：透传。
4. nvidia 实现：
   - **paged_caching**：新增量化写入 kernel。grid 覆盖 (token, kv_head)，
     每个 block 负责一个 (token, head) 的 head_dim 维：amax 归约 → scale 写出 → 编码写 cache。
     非 F8 时走原有拷贝 kernel 不变。
   - **paged_attention (decode)**：新增独立 `kernel_fp8.cuh`（不动现有 kernel 家族/kernel_v2.cuh），
     clean-room 实现在线 softmax decode kernel：CTA per (seq, q_head)，128 线程，
     沿 block_tables 遍历 token，K/V 加载点做 dequant-on-load（decode→乘 scale→转累算 dtype）。
     v1 不做 split-kv（Qwen3-8B 32 q head → 32 CTA；长上下文 bs=1 的 SM 占用不足记入报告）。
   - **paged_attention_prefill**：优先方案 A——读 `kernel_v2.cuh` 的 K/V 加载点，
     若加载足够隔离则加 dtype 模板参数做 dequant-on-load（允许适度修改 prefill kernel，
     但必须保持 F16/BF16 路径逐位不变）；若加载点太分散则用方案 B——op 内部按 block_tables
     gather-dequant 到 BF16 scratch（紧凑布局 + 恒等 block_tables），再调现有 BF16 prefill kernel。
5. infinicore C++ 层：`include/infinicore/ops/*.hpp` + `src/infinicore/ops/*/*.cc`
   追加 `std::optional<Tensor> k_scale, v_scale`（参照 alibi_slopes 先例），hash_combine 纳入。
6. 测试：`test/infiniop/` 下三个算子各加 FP8 case（numpy 参考实现量化/反量化/注意力对拍）。

工程模板参照 W4 的 `src/infiniop/ops/fp8_blockwise_gemm/`。xmake glob 自动收编新文件。

### 跨仓契约（固定签名，两侧必须一致）

C API（尾部追加，NULL=不存在）：

```c
// paged_caching
infiniopCreatePagedCachingDescriptor(handle, desc_ptr, k_cache_desc, v_cache_desc,
    k_desc, v_desc, slot_mapping_desc, k_scale_desc, v_scale_desc);
infiniopPagedCaching(desc, workspace, workspace_size,
    k_cache, v_cache, k, v, slot_mapping, k_scale, v_scale, stream);

// paged_attention（decode）：k_scale_desc/v_scale_desc 插在 alibi_slopes_desc 之后、scale 之前
infiniopPagedAttention(desc, workspace, workspace_size, out, q, k_cache, v_cache,
    block_tables, seq_lens, alibi_slopes, k_scale, v_scale, stream);

// paged_attention_prefill：同模式追加
```

infinicore C++（`std::optional<Tensor>` 尾部追加，参照 alibi_slopes 先例）：

```cpp
infinicore::op::paged_caching_(k_cache, v_cache, k, v, slot_mapping, k_scale, v_scale);
infinicore::op::paged_attention_(out, q, k_cache, v_cache, block_tables, cache_lens,
                                 alibi_slopes, scale, k_scale, v_scale);
infinicore::op::paged_attention_prefill_(out, q, k_cache, v_cache, block_tables,
                                         cache_lens, input_offsets, alibi_slopes,
                                         scale, k_scale, v_scale);
```

scale 校验：cache dtype==F8 时 k_scale/v_scale 必须存在，F32，
shape `[num_blocks, num_kv_heads, block_size]`；cache dtype!=F8 时必须为 nullopt/NULL。

## InfiniLM 改动

1. `csrc/config/quantization_scheme.hpp`：`KVQuantAlgo` 加 `FP8`；
   `csrc/config/quant_config.hpp::set_kv_quant_scheme` 映射 `DataType::F8 → FP8`。
2. cache 分配（`csrc/cache/kv_cache.cpp` 与 `csrc/models/infinilm_model.cpp:35` 通路）：
   kv_cache_dtype=F8 时 cache 张量按 F8 分配，并额外分配每层 k_scale/v_scale（F32），
   规模 = num_blocks × num_kv_heads × block_size × 4B × 2（约为 BF16 cache 的 6.25% 开销/层）。
3. `csrc/layers/attention/backends/paged_attn.cpp`：cache dtype 为 F8 时，
   `paged_caching_` 与两个 attention 调用带上本层 scale 张量。
   scale 张量的存放位置：随 kv_cache 一起按层管理（cache 层持有，AttentionLayer 取用），
   不复用现有 per-tensor `kv_cache_k_scale` 参数（那是 INT8 静态量化的 {1} 标量，语义不同）。
4. Python：`modeling_utils.py::parse_dtype` 支持 `"fp8"`（确认 infinicore python 绑定暴露 F8 dtype，
   没有则补 pybind），`--kv-cache-dtype fp8` 生效；paged 后端 FP8 之外的组合行为不变。

## 验证计划（服务器 RTX 5090 / sm_120）

1. 算子级：test/infiniop 三个算子 FP8 case 全绿（CPU/numpy 参考对拍）。
2. E2E 对拍：Qwen3-8B FP8 权重 + FP8 KV vs 同权重 BF16 KV，同 prompt logits/PPL 对比。
3. 精度：fp8_ppl_eval.py（长窗 PPL）、fp8_mcq_eval.py（C-Eval val 1346 / MMLU test 14042）。
4. 性能：examples/bench.py input_len 扫描 1k/4k/16k，decode ITL/tok/s 与 cache 容量对比。
5. 报告：`dev_fp8/w6_kv_fp8_report.md`。

## 风险

- decode 无 split-kv：bs=1 长上下文 SM 占用低，性能数据如实记录，必要时 v2 加 split-kv。
- prefill 方案 B 有额外 scratch 显存与流量（prefill 一次性、计算bound，可接受）。
- E4M3 无 inf，amax 异常（NaN 输入）时行为与 encode 一致截断，不做特殊处理。

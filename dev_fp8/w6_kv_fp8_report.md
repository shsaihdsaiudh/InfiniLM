# W6: KV Cache FP8(E4M3) 动态量化报告

> 环境:RTX 5090(sm_120)/ CUDA 13.3,Qwen3-8B-FP8 权重(32q/8kv,head_dim 128,36 层)。
> 设计见 `kv_fp8_design.md`;算子实现对拍 `InfiniCore test/infiniop/{paged_caching,paged_attention,paged_attention_prefill}.py`;
> runner `results/w6_accuracy_chain.sh` / `w6_perf_bench.sh`;原始数据已回捞 `results/{ppl,ceval,mmlu}_kv{fp8,bf16}*.json`
> 与 `results/w6_final.log`(服务器副本在 `/root/fp8/eval_logs/`)。

## 1. 方案

- **量化语义**:E4M3、per-token-per-kv-head 动态 scale(F32,`[num_blocks, num_kv_heads, block_size]`,
  对 head_dim 取 amax,`scale=amax/448`),K 存 post-RoPE;写入时由 `paged_caching` 量化 kernel 完成,
  注意力 kernel 读取时 dequant-on-load。
- **算子扩展**(InfiniCore,BF16/F16 路径零回归):
  - `paged_caching`:新增量化写入 kernel(grid 覆盖 (token, kv_head),block 内 amax 归约 → scale → 编码);
  - `paged_attention` decode:新增独立 clean-room FP8 kernel(CTA per (seq, q_head),32 warp 分 token
    + 寄存器双缓冲跨页预取,log2 域在线 softmax,dequant-on-load;hd 64/128;无跨 CTA split-kv);
  - `paged_attention_prefill`:方案 B——按 block_tables gather-dequant 引用页到 BF16 scratch
    (恒等 block_tables),复用现有 BF16 prefill kernel;scratch 走 workspace。
  - 三个算子 C API 追加可选 `k_scale/v_scale` 描述符与数据指针:cache 为 F8 时必需(F32,
    shape 校验),非 F8 必须为 NULL;info.h 强制校验。
- **引擎接线**(InfiniLM):`KVQuantAlgo::FP8`,`--kv-cache-dtype fp8` 全链路;
  scale 随 cache 同生命周期分配(`ForwardContext::kv_scale_vec`,含 TP 分头);
  仅 PAGED_ATTN 后端支持(其他后端构造期显式报错);INT8 static_attn 路径不动。
- **显存收益**:KV cache 每 token 每头每维 2B → 1B,加 scale 开销(每 token 每头 4B×2/(128×2))≈ 3.1%,
  净 **~1.94×** KV 显存压缩。Qwen3-8B:每 token KV 从 36×8×128×2×2B=147456B 降到 73728B+2304B(scale)。

## 2. 算子级验证(服务器,test/infiniop --nvidia)

| 算子 | FP8 case | 结果 |
|---|---|---|
| paged_caching | 6 形状 × {F16,BF16} 源(含 hd576/512) | ✅ 逐位等于 torch CPU RNE 参考,scale maxdiff=0 |
| paged_attention(decode) | 4 形状 × {F16,BF16}(含 alibi、GQA 40/64 头) | ✅ |
| paged_attention_prefill | 4 形状 × {F16,BF16} × {I32,I64} 索引,多轮增量 | ✅ |

调试记录:torch CUDA 端 `tensor/标量` 会降为乘倒数,与 kernel IEEE 除法差 1 ulp,
在 E4M3 中点(如 336.0 夹于 320/352)翻转舍入方向 → 参考改到 CPU 计算后逐位一致。

## 3. 精度(Qwen3-8B-FP8 权重;KV 对照为同权重 paged BF16 KV)

| 评测 | FP8 KV | paged BF16 KV(对照) | Δ | W5 static BF16 KV 参考 |
|---|---|---|---|---|
| wikitext2 PPL(chunk 512,全量 293k tok) | **18.8724** | 18.8772 | **−0.005 (−0.026%)** | 18.8771(W3 FP8 权重 static) |
| C-Eval val 1346 | **74.29%** | 74.67% | **−0.37pt** | 74.74% |
| MMLU test 14042 | **69.54%** | 69.32% | **+0.22pt** | 69.47% |

三项指标全部在 ±0.5pt 噪声带内(PPL 与 MMLU 甚至略优于对照),FP8 KV 精度达标。

## 4. 性能(examples/bench.py,FP8 权重模型,output 128,RTX 5090)

**最终 kernel(32w)Decode Avg ITL / Throughput**:

| 场景 | BF16 KV(tuned kernel) | FP8 KV | FP8/BF16 |
|---|---|---|---|
| bs=1 in=1024 | 7.90ms / 126.7 tok/s | 8.29ms / 120.6 tok/s | 1.05× |
| bs=1 in=4096 | 10.46ms / 95.6 tok/s | 11.94ms / 83.8 tok/s | 1.14× |
| bs=1 in=16384 | 20.75ms / 48.2 tok/s | 26.42ms / 37.9 tok/s | 1.27× |
| bs=8 in=1024 | 21.01ms / 380.8 tok/s | 22.14ms / 361.4 tok/s | 1.05× |
| bs=8 in=4096 | 25.57ms / 312.9 tok/s | 29.82ms / 268.3 tok/s | 1.17× |

**Prefill TTFT**:FP8 KV 开销 +0.5%~+3.0%(gather-dequant scratch):
bs1/1k 171.1→176.2ms;bs8/1k 951.0→967.4ms;bs1/4k 824.6→833.4ms;bs8/4k 6335→6415ms;bs1/16k 8539→8583ms。

(bs=8/in=16384 两 dtype 均在 bench warmup 阶段 OOM,与 KV dtype 无关——warmup 全位置 logits ~40GB,属 bench 既有行为。)

**decode kernel 迭代记录**(FP8 KV ITL,bs=1 为主):

| 版本 | 结构 | bs=1/1k | bs=1/4k | bs=1/16k | bs=8/4k |
|---|---|---|---|---|---|
| v1 | token 串行,每 token 3×__syncthreads | 46.60 | 157.46 | 602.08 | 168.13 |
| v1.5 | 4 warp 分 token,shuffle 归约,合并 | 13.32 | 30.83 | 100.71 | 43.05 |
| v1.6 | 8 warp + 寄存器双缓冲跨页预取 | 10.79 | 21.65 | 64.72 | 34.87 |
| 最终 | 32 warp(1024 线程/CTA) | **8.29** | **11.94** | **26.42** | **29.82** |

**容量/显存**:KV 字节数净 ~1.94× 压缩(2B→1B/元素,scale 开销 ~3.1%)。
Qwen3-8B 每 token KV:BF16 147456B → FP8 73728B + 2304B scale。

**E2E decode 正确性**:贪婪生成对拍(in=512/out=64),FP8 KV 与 BF16 KV 输出仅个别虚词分叉
(语义一致);PPL/MCQ 走的 prefill 路径数值由全量评测覆盖;decode 数值由算子级 torch 参考对拍覆盖。

**差距分析(残留 1.05-1.27×)**:clean-room kernel 每 warp 内在线 softmax 依赖链串行,
bs=1 时仅 32 CTA(32 q head)占用 170 个 SM 的 19%;BF16 tuned kernel 另有 split-kv(4 路)。
差距随上下文变长扩大、随 batch 增大收敛。后续方向:split-kv 跨 CTA(kernel 结构已预留:
cursor + loadToken 解耦,warp 数常量可调)。

## 5. 已知限制

- decode FP8 kernel 无跨 CTA split-kv(CTA 内 32 warp 并行):bs=1 超长上下文残留 1.27× 差距,见 §4。
- prefill FP8 走 gather-dequant scratch:一次性额外带宽;不支持 split-kv/mma 变体。
- FLASH_ATTN/STATIC 后端、MLA(576/512)、videonsa/deepseek 自定义分配路径暂不支持 FP8 KV
  (显式报错或 info 校验拦截)。
- KV connector(PD 分离)不传输 scale,该组合不可用。
- 其他 vendor(ascend/bang/metax/moore)的 paged 算子签名已补齐透传参数(W7,2026-09-03;F8 cache 在这些后端显式返回 NOT_IMPLEMENTED,FP8 KV 仍仅 nvidia 支持)。

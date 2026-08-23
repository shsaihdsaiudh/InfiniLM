# DeepSeek-V4 真权重服务器验收手册

本文只覆盖当前 correctness 范围：单请求、batch=1、短 prompt、完整 43 层
base model 的 prefill 与少量 decode。生产连续批处理、百万上下文、DSpark、
TP/PP/EP 和 CUDA Graph 不在本期验收范围内。

## 租用门槛

当前模型代码明确要求 `PP=TP=1`，所以多张小显存 GPU 不能替代一张大显存卡。
官方 checkpoint 索引为 155.418 GiB；去掉 `mtp.*`，并按当前 loader 将 dense
FP8 解量化为 BF16 后，估算 base 参数常驻 150.756 GiB，尚未计入 CUDA context、
临时张量和 allocator 开销。

- 推荐：独占整卡 B300 288 GB；
- 最低试验配置：独占整卡 B200 180/192 GB，且脚本看到的空闲显存必须至少
  165 GiB；不要选择 MIG/MLOPart 分区；
- H200 141 GB 小于参数常驻量，不能用于当前单卡路径；
- 主机内存至少 256 GB，建议 384 GB；本地磁盘至少 350 GB；
- CUDA Toolkit 13.x、Linux x86_64，B200 为 CC 10.0，B300 为 CC 10.3。

NVIDIA 的当前资料列出 B200/B300 的 compute capability 为 10.0/10.3，并给出
B200 180 GB、B300 288 GB 的每卡显存规格：

- <https://developer.nvidia.com/cuda/gpus>
- <https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory/latest/components.html>

## 1. 获取代码与权重

InfiniLM 与 InfiniCore 应保持同级目录，并使用本分支已提交的阶段版本：

```bash
git clone <InfiniCore repository URL> InfiniCore
git clone <InfiniLM repository URL> InfiniLM
git -C InfiniCore checkout dev/fp8-blockwise-dequantize
git -C InfiniLM checkout dev/deepseek-v4
```

权重使用官方仓库；下载完成后目录中应有 48 个 safetensors shard：

```bash
MODEL_DIR=/data/models/DeepSeek-V4-Flash-DSpark
hf download deepseek-ai/DeepSeek-V4-Flash-DSpark --local-dir "$MODEL_DIR"
```

不下载权重也可先复验远端 metadata：

```bash
cd InfiniLM
.venv/bin/python dev_dsv4/check_checkpoint_format.py
```

预期结尾包含 `tensors: 72317`、`150.756 GiB` 和 layout validation passed。

## 2. 构建 B200/B300 版 InfiniCore

先准备带 CUDA 支持的 PyTorch 环境。V4 需要 `aten=true`；B200/B300 使用
`sm_100f` family code。由于当前 InfiniCore 的 `cuda_arch` 选项白名单尚未列出
sm_100，下面保留合法的 sm_90 值，同时通过全局 CUDA flags 加入 sm_100 family
cubin；该 cubin可在 CC 10.0 和 10.3 上运行。

```bash
cd ../InfiniCore
PYTHON_BIN=/absolute/path/to/InfiniLM/.venv/bin/python
CUDA_HOME=/usr/local/cuda
INFINI_ROOT="$HOME/.infini-dsv4"

PYTHON="$PYTHON_BIN" xmake f -c -m release \
  --aten=y --nv-gpu=y --cpu=y --cuda="$CUDA_HOME" \
  --cuda_arch=sm_90 \
  --cuflags='-gencode=arch=compute_100f,code=sm_100' \
  --graph=y --ccl=n --cudnn=n --omp=y --kind=shared -y
PYTHON="$PYTHON_BIN" xmake build -y -j8 \
  infiniop infinicore_cpp_api _infinicore
PYTHON="$PYTHON_BIN" xmake install -y -o "$INFINI_ROOT" \
  infiniop infinicore_cpp_api _infinicore
```

NVIDIA 对 `sm_100f` family compatibility 的说明：
<https://developer.nvidia.com/blog/nvidia-blackwell-and-nvidia-cuda-12-9-introduce-family-specific-architecture-features/>

## 3. 构建 InfiniLM

```bash
cd ../InfiniLM
export INFINI_ROOT="$HOME/.infini-dsv4"
export INFINILM_CXX11_ABI=1
export LD_LIBRARY_PATH="$INFINI_ROOT/lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

xmake f -c -m release --cxx11-abi="$INFINILM_CXX11_ABI" -y
xmake build -y -j8 _infinilm
```

如果服务器的 PyTorch 使用旧 C++ ABI，应把 `INFINILM_CXX11_ABI` 改为与
`torch.compiled_with_cxx11_abi()` 一致的值。

## 4. 执行最终验收

先只检查文件和显存，不分配模型：

```bash
dev_dsv4/run_real_checkpoint_smoke.sh \
  --model "$MODEL_DIR" --preflight-only
```

通过后执行完整加载、一次显式 prefill 有限值检查，以及 4-token greedy decode：

```bash
dev_dsv4/run_real_checkpoint_smoke.sh \
  --model "$MODEL_DIR" \
  --prompt 'Hello' \
  --max-new-tokens 4 \
  2>&1 | tee dsv4-real-checkpoint.log
```

通过标准：

1. 48 个 shard 全部逐文件加载，state-dict 检查无 missing/unexpected key；
2. prefill 的 logits/hidden 全部有限，并打印张量形状与 last-token argmax；
3. 连续生成 4 个 token，不崩溃、不 OOM；
4. 日志记录生成 token/text 和近似设备显存增量。

若失败，请保留验收日志，并同时记录：

```bash
git -C ../InfiniCore rev-parse HEAD
git rev-parse HEAD
nvidia-smi
nvcc --version
```

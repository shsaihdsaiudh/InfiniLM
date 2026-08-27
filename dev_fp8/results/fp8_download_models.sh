#!/bin/bash
# FP8 评测模型下载（hf-mirror，禁用 xet 后端避免 401）
set -ex
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/data/huggingface_home
export HF_HUB_CACHE=/data/huggingface_home/hub
export HF_HUB_DISABLE_XET=1
python3 - <<'PYEOF'
from huggingface_hub import snapshot_download
for m in ["Qwen/Qwen3-8B-FP8", "Qwen/Qwen3-8B"]:
    p = snapshot_download(m)
    print("DOWNLOADED", m, p, flush=True)
PYEOF
echo DOWNLOAD_DONE

# FP8 工作区运行环境 —— source 本文件后跑 InfiniLM
export INFINI_ROOT=/root/fp8/.infini
export LD_LIBRARY_PATH=/root/fp8/.infini/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PYTHONPATH=/root/fp8/InfiniCore/python:/root/fp8/InfiniLM/python
export HF_HOME=/data/huggingface_home
export HF_HUB_CACHE=/data/huggingface_home/hub
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1

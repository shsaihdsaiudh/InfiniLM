#!/bin/bash
# FP8 工作区 InfiniCore 构建 —— 隔离 INFINI_ROOT=/root/fp8/.infini
# 不碰 /root/.infini（perf 基线产物）
set -x
cd /root/fp8/InfiniCore || exit 1
export XMAKE_ROOT=y
export INFINI_ROOT=/root/fp8/.infini
xmake f -c --nv-gpu=y --cpu=y --omp=y --ccl=n --cudnn=y --aten=n --graph=n \
  --cuda=/usr/local/cuda --cuda_arch=sm_120 -m release -k shared || exit 1
xmake build -j32 || exit 1
xmake install -o /root/fp8/.infini || exit 1
echo CORE_BUILD_DONE

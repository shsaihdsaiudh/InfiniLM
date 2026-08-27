#!/bin/bash
# infinicore pybind 扩展 + InfiniLM _infinilm 构建（隔离 INFINI_ROOT）
set -ex
export XMAKE_ROOT=y
export INFINI_ROOT=/root/fp8/.infini

# 1. infinicore python 扩展（PYTHONPATH 方式使用，装进源码树 python/infinicore）
cd /root/fp8/InfiniCore
xmake build -j32 _infinicore
xmake install _infinicore

# 2. InfiniLM C++ 引擎
cd /root/fp8/InfiniLM
xmake f -c -m release
xmake build -j32 _infinilm
xmake install _infinilm

echo LM_BUILD_DONE

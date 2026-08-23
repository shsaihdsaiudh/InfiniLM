#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
infini_root=${INFINI_ROOT:-"$HOME/.infini-dsv4"}
cuda_home=${CUDA_HOME:-"$HOME/.local/cuda-13.2"}
torch_lib=${TORCH_LIB:-$(python -c \
    'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')}
output="$repo_root/build/linux/x86_64/release/dsv4_csa_native_smoke"
mkdir -p "$(dirname "$output")"

g++ \
    -std=c++17 \
    -O2 \
    -D_GLIBCXX_USE_CXX11_ABI="${INFINILM_CXX11_ABI:-1}" \
    -I"$infini_root/include" \
    -I"$repo_root/third_party/spdlog/include" \
    -I"$repo_root/third_party/json/single_include" \
    "$repo_root/dev_dsv4/csa_native_smoke.cpp" \
    "$repo_root/csrc/models/deepseek_v4/deepseek_v4_csa_compressor.cpp" \
    "$repo_root/csrc/models/deepseek_v4/deepseek_v4_hca_compressor.cpp" \
    "$repo_root/csrc/models/deepseek_v4/deepseek_v4_attention.cpp" \
    -L"$infini_root/lib" \
    -Wl,-rpath,"$infini_root/lib" \
    -L"$torch_lib" \
    -Wl,-rpath,"$torch_lib" \
    -linfinicore_cpp_api \
    -linfiniop \
    -linfinirt \
    -linfiniccl \
    -ltorch \
    -ltorch_cpu \
    -ltorch_cuda \
    -lc10 \
    -lc10_cuda \
    -pthread \
    -ldl \
    -o "$output"

export LD_LIBRARY_PATH="$infini_root/lib:$torch_lib:$cuda_home/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$output"

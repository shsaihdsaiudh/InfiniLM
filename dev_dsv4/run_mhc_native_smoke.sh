#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
infini_root=${INFINI_ROOT:-"$HOME/.infini"}
cuda_home=${CUDA_HOME:-"$HOME/.local/cuda-13.2"}
output="$repo_root/build/linux/x86_64/release/mhc_native_smoke"
mkdir -p "$(dirname "$output")"

compile_flags=(
    -std=c++17
    -O2
    -D_GLIBCXX_USE_CXX11_ABI="${INFINILM_CXX11_ABI:-1}"
    -I"$infini_root/include"
    -I"$repo_root/third_party/spdlog/include"
    -I"$repo_root/third_party/json/single_include"
    "$repo_root/dev_dsv4/mhc_native_smoke.cpp"
    "$repo_root/csrc/models/deepseek_v4/deepseek_v4_hyper_connection.cpp"
    -L"$infini_root/lib"
    -Wl,-rpath,"$infini_root/lib"
    -linfinicore_cpp_api
    -linfiniop
    -linfinirt
    -linfiniccl
    -pthread
    -ldl
    -o
    "$output"
)

runtime_paths=("$infini_root/lib" "$cuda_home/lib64")
if ldd "$infini_root/lib/libinfinicore_cpp_api.so" | grep -q 'libtorch'; then
    torch_lib=${TORCH_LIB:-$(python -c \
        'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')}
    compile_flags+=(
        -L"$torch_lib"
        -Wl,-rpath,"$torch_lib"
        -ltorch
        -ltorch_cpu
        -ltorch_cuda
        -lc10
        -lc10_cuda
    )
    runtime_paths+=("$torch_lib")
    export INFINILM_MHC_TEST_BF16=1
fi

g++ "${compile_flags[@]}"
joined_runtime_paths=$(IFS=:; echo "${runtime_paths[*]}")
export LD_LIBRARY_PATH="$joined_runtime_paths${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$output"

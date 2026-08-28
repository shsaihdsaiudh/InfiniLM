# fp8_blockwise_gemm 微基准：有效带宽 = (Q + scales + A + C 字节) / 实测时间
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ctypes
from ctypes import c_uint64

import torch

from libinfiniop import (
    LIBINFINIOP,
    InfiniDtype,
    InfiniDeviceEnum,
    TestTensor,
    TestWorkspace,
    check_error,
    create_handle,
    destroy_handle,
    infiniopOperatorDescriptor_t,
)


def bench(M, N, K, iters=200):
    dtype = InfiniDtype.BF16
    a_t = TestTensor((M, K), None, dtype, InfiniDeviceEnum.NVIDIA)
    q_t = TestTensor((N, K), None, InfiniDtype.F8, InfiniDeviceEnum.NVIDIA, mode="float8_e4m3fn")
    s_t = TestTensor((N // 128, K // 128), None, InfiniDtype.F32, InfiniDeviceEnum.NVIDIA)
    c_t = TestTensor((M, N), None, dtype, InfiniDeviceEnum.NVIDIA)

    handle = create_handle()
    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateFp8BlockwiseGemmDescriptor(
            handle, ctypes.byref(descriptor),
            c_t.descriptor, a_t.descriptor, q_t.descriptor, s_t.descriptor,
        )
    )
    workspace_size = c_uint64(0)
    check_error(LIBINFINIOP.infiniopGetFp8BlockwiseGemmWorkspaceSize(descriptor, ctypes.byref(workspace_size)))
    workspace = TestWorkspace(workspace_size.value, InfiniDeviceEnum.NVIDIA)

    def run():
        check_error(
            LIBINFINIOP.infiniopFp8BlockwiseGemm(
                descriptor, workspace.data(), workspace_size.value,
                c_t.data(), a_t.data(), q_t.data(), s_t.data(), None,
            )
        )

    for _ in range(20):
        run()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        run()
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / iters

    bytes_moved = N * K + (N // 128) * (K // 128) * 4 + M * K * 2 + M * N * 2
    gbps = bytes_moved / (ms * 1e-3) / 1e9
    flops = 2 * M * N * K
    tflops = flops / (ms * 1e-3) / 1e12
    print(f"M={M:3d} N={N:6d} K={K:6d}: {ms*1000:8.2f} us  {gbps:7.1f} GB/s  {tflops:6.2f} TFLOP/s")

    check_error(LIBINFINIOP.infiniopDestroyFp8BlockwiseGemmDescriptor(descriptor))
    destroy_handle(handle)


if __name__ == "__main__":
    LIBINFINIOP.infinirtSetDevice(InfiniDeviceEnum.NVIDIA, ctypes.c_int(0))
    # Qwen3-8B 代表形状
    bench(1, 4096, 4096)
    bench(1, 12288, 4096)
    bench(1, 4096, 12288)
    bench(1, 6144, 4096)
    bench(16, 4096, 4096)
    bench(16, 12288, 4096)
    bench(16, 4096, 12288)
    bench(16, 6144, 4096)

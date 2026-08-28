# 对照实验:repo 官方 marlin_quantize + uint4b8 (INT4) —— 验证 marlin kernel 在 sm_120 是否本身可用
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
from libinfiniop.scalar_type import scalar_types
from gptq_marlin_gemm import marlin_quantize

import faulthandler
faulthandler.dump_traceback_later(60, exit=True)


def run_int8_case(M, K, N, dtype, infini_dtype):
    torch.manual_seed(42)
    a = torch.randn(M, K, dtype=dtype, device="cuda")
    w = torch.randn(K, N, dtype=dtype, device="cuda")

    w_ref, marlin_q, marlin_s, g_idx, sort_idx, _ = marlin_quantize(
        w, scalar_types.uint4b8, 128, False
    )
    out_ref = a @ w_ref

    handle = create_handle()
    a_t = TestTensor((M, K), None, infini_dtype, InfiniDeviceEnum.NVIDIA, mode="manual", set_tensor=a)
    b_t = TestTensor(marlin_q.shape, marlin_q.stride(), InfiniDtype.I32, InfiniDeviceEnum.NVIDIA, mode="manual", set_tensor=marlin_q)
    c_t = TestTensor(tuple(out_ref.shape), None, infini_dtype, InfiniDeviceEnum.NVIDIA)
    s_t = TestTensor(marlin_s.shape, marlin_s.stride(), infini_dtype, InfiniDeviceEnum.NVIDIA, mode="manual", set_tensor=marlin_s)
    empty_t = TestTensor((0,), None, InfiniDtype.I32, InfiniDeviceEnum.NVIDIA)

    print("  [dbg] create desc", flush=True)
    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateGptqMarlinGemmDescriptor(
            handle,
            ctypes.byref(descriptor),
            c_t.descriptor,
            a_t.descriptor,
            b_t.descriptor,
            s_t.descriptor,
            empty_t.descriptor,
            empty_t.descriptor,
            empty_t.descriptor,
            empty_t.descriptor,
        )
    )
    workspace_size = c_uint64(0)
    check_error(LIBINFINIOP.infiniopGetGptqMarlinGemmWorkspaceSize(descriptor, ctypes.byref(workspace_size)))
    workspace = TestWorkspace(workspace_size.value, InfiniDeviceEnum.NVIDIA)
    workspace.tensor.torch_tensor().zero_()

    print("  [dbg] launch", flush=True)
    check_error(
        LIBINFINIOP.infiniopGptqMarlinGemm(
            descriptor,
            workspace.data(),
            workspace_size.value,
            c_t.data(),
            a_t.data(),
            b_t.data(),
            s_t.data(),
            empty_t.data(),
            empty_t.data(),
            empty_t.data(),
            empty_t.data(),
            scalar_types.uint4b8.id,
            True,
            False,
            True,
            False,
            None,
        )
    )
    torch.cuda.synchronize()
    print("  [dbg] synced", flush=True)

    actual = c_t.actual_tensor().float()
    rel = (actual - out_ref.float()).abs().mean() / out_ref.float().abs().mean()
    print(f"INT4 M={M} K={K} N={N}: rel_mean={rel.item():.6f}")

    check_error(LIBINFINIOP.infiniopDestroyGptqMarlinGemmDescriptor(descriptor))
    destroy_handle(handle)


if __name__ == "__main__":
    LIBINFINIOP.infinirtSetDevice(InfiniDeviceEnum.NVIDIA, ctypes.c_int(0))
    run_int8_case(8, 256, 256, torch.bfloat16, InfiniDtype.BF16)
    print("\033[92mINT4 MARLIN RAN\033[0m")

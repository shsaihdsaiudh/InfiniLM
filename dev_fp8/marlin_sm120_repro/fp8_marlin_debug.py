# FP8 (kFE4M3fn) marlin GEMM 独立正确性验证 —— 复用 gptq_marlin_gemm.py 的布局工具
# 128x128 块 scale -> 扩展为 [K/128, N] -> marlin_permute_scales,与 InfiniLM FP8Blockwise 转换同构
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ctypes
from ctypes import c_uint64
import faulthandler
faulthandler.dump_traceback_later(45, exit=True)

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
from gptq_marlin_gemm import get_weight_perm, marlin_permute_scales, marlin_weights

BLOCK = 128


def block_quant_fp8(w, fp8_max=448.0):
    """w: [K, N] fp32 -> (w_q_int [K,N], w_ref [K,N] fp32, s [K/B, N/B] fp32)."""
    K, N = w.shape
    B = BLOCK
    wb = w.reshape(K // B, B, N // B, B).permute(0, 2, 1, 3).reshape(K // B, N // B, -1)
    amax = wb.abs().amax(dim=-1).clamp(min=1e-12)
    s = amax / fp8_max
    w_q = (wb / s.unsqueeze(-1)).to(torch.float8_e4m3fn)
    w_ref = (
        (w_q.float() * s.unsqueeze(-1))
        .reshape(K // B, N // B, B, B)
        .permute(0, 2, 1, 3)
        .reshape(K, N)
    )
    w_q_int = w_q.view(torch.uint8).to(torch.int32).reshape(K, N)
    return w_q_int, w_ref, s


def run_case(M, K, N, dtype, infini_dtype):
    torch.manual_seed(42)
    w = torch.randn(K, N, dtype=torch.float32) * 0.5
    w_q_int, w_ref, s = block_quant_fp8(w)

    marlin_q = marlin_weights(w_q_int, K, N, 8, get_weight_perm(8)).cuda()
    s_exp = s.repeat_interleave(BLOCK, dim=1)  # [K/128, N]
    marlin_s = marlin_permute_scales(s_exp, K, N, BLOCK).to(dtype).cuda()

    a = torch.randn(M, K, dtype=dtype, device="cuda")
    out_ref = (a.float() @ w_ref.cuda()).to(dtype)

    print("  [dbg] create_handle", flush=True)
    handle = create_handle()
    a_t = TestTensor((M, K), None, infini_dtype, InfiniDeviceEnum.NVIDIA, mode="manual", set_tensor=a)
    b_t = TestTensor(marlin_q.shape, marlin_q.stride(), InfiniDtype.I32, InfiniDeviceEnum.NVIDIA, mode="manual", set_tensor=marlin_q)
    c_t = TestTensor(tuple(out_ref.shape), None, infini_dtype, InfiniDeviceEnum.NVIDIA)
    s_t = TestTensor(marlin_s.shape, marlin_s.stride(), infini_dtype, InfiniDeviceEnum.NVIDIA, mode="manual", set_tensor=marlin_s)
    # nvidia Descriptor::create 无条件解引用 g_idx_desc/perm_desc,空张量代替 NULL
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
            empty_t.descriptor,  # global_scales
            empty_t.descriptor,  # b_zeros
            empty_t.descriptor,  # g_idx
            empty_t.descriptor,  # perm
        )
    )

    workspace_size = c_uint64(0)
    check_error(LIBINFINIOP.infiniopGetGptqMarlinGemmWorkspaceSize(descriptor, ctypes.byref(workspace_size)))
    workspace = TestWorkspace(workspace_size.value, InfiniDeviceEnum.NVIDIA)
    # marlin 把 workspace 当锁状态用,必须全零(TestWorkspace 默认填 1)
    workspace.tensor.torch_tensor().zero_()

    def lib_gemm():
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
                scalar_types.float8_e4m3fn.id,
                True,   # is_k_full
                True,   # use_atomic_add
                True,   # use_fp32_reduce
                False,  # is_zp_float
                None,
            )
        )

    print("  [dbg] launch gemm", flush=True)
    lib_gemm()
    print("  [dbg] gemm returned", flush=True)
    torch.cuda.synchronize()

    actual = c_t.actual_tensor().float()
    ref = out_ref.float()
    rel = (actual - ref).abs().mean() / ref.abs().mean()
    max_abs = (actual - ref).abs().max()
    print(f"M={M} K={K} N={N} dtype={dtype}: rel_mean={rel.item():.6f} max_abs={max_abs.item():.6f}")
    bad = ((actual - ref).abs() > 0.1 * (ref.abs().max() + 1e-6)).sum()
    print(f"  outlier count: {bad.item()} / {ref.numel()}")

    check_error(LIBINFINIOP.infiniopDestroyGptqMarlinGemmDescriptor(descriptor))
    destroy_handle(handle)
    return rel.item()


if __name__ == "__main__":
    LIBINFINIOP.infinirtSetDevice(InfiniDeviceEnum.NVIDIA, ctypes.c_int(0))
    rels = []
    rels.append(run_case(8, 256, 256, torch.bfloat16, InfiniDtype.BF16))
    rels.append(run_case(1, 256, 256, torch.bfloat16, InfiniDtype.BF16))
    rels.append(run_case(8, 512, 1024, torch.bfloat16, InfiniDtype.BF16))
    rels.append(run_case(16, 4096, 4096, torch.bfloat16, InfiniDtype.BF16))
    ok = all(r < 0.02 for r in rels)
    print("\033[92mFP8 MARLIN OK\033[0m" if ok else "\033[91mFP8 MARLIN MISMATCH\033[0m")

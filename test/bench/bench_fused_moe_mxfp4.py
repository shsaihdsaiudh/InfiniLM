"""Microbenchmark for the MXFP4 fused MoE operator used by DeepSeek-V4.

Compares the InfiniCore fused MXFP4 MoE path (packed FP4 weights + E8M0
scales + clamped SwiGLU, SwigluLimit10) against a decomposed bf16
grouped-GEMM baseline computed with torch matmuls on dequantized weights.

Usage (on a machine with infinicore built):

    PYTHONPATH=/path/to/InfiniCore/python INFINI_ROOT=~/.infini \
        python3 test/bench/bench_fused_moe_mxfp4.py

Default shapes are the real DeepSeek-V4 MoE dimensions (256 routed
experts, hidden 4096, intermediate 2048, top-k 6).
"""

import argparse
import time

import torch
import torch.nn.functional as F

ACT_SWIGLU_LIMIT_10 = 3

E2M1_MAGNITUDES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)


def dequantize_mxfp4(packed, scales):
    """Exact inverse of the kernel-side decode: nibble codes * 2^(e8m0-127)."""
    magnitudes = E2M1_MAGNITUDES.to(packed.device)
    codes = torch.stack((packed & 0x0F, packed >> 4), dim=-1).flatten(-2)
    values = magnitudes[(codes & 0x07).to(torch.int64)]
    values = torch.where((codes & 0x08) != 0, -values, values)
    exponents = scales.to(torch.int32).sub(127).repeat_interleave(32, dim=-1)
    return torch.ldexp(values, exponents)


def dequantize_mxfp4_chunked(packed, scales, dtype, chunk=8):
    """Chunked dequantize to keep int64/fp32 temporaries small on full shapes."""
    rows = packed.shape[1]
    cols = packed.shape[2] * 2
    out = torch.empty((packed.shape[0], rows, cols), dtype=dtype, device=packed.device)
    for start in range(0, packed.shape[0], chunk):
        stop = min(start + chunk, packed.shape[0])
        out[start:stop] = dequantize_mxfp4(packed[start:stop], scales[start:stop]).to(
            dtype
        )
    return out


def torch_moe_reference(input, ids, routing, w13, w2):
    """Authoritative per-route fp32 reference (matches the op test semantics)."""
    output = torch.zeros_like(input, dtype=torch.float32)
    for token in range(input.shape[0]):
        for route in range(ids.shape[1]):
            expert = int(ids[token, route])
            if expert < 0 or expert >= w13.shape[0]:
                continue
            gate, up = F.linear(input[token].float(), w13[expert].float()).chunk(
                2, dim=-1
            )
            activated = F.silu(gate.clamp(max=10.0)) * up.clamp(-10.0, 10.0)
            activated = activated.to(input.dtype).float()
            output[token] += (
                F.linear(activated, w2[expert].float()) * routing[token, route].float()
            )
    return output.to(input.dtype)


def torch_moe_grouped(input, ids, routing, w13, w2):
    """Decomposed bf16 baseline: grouped GEMM per expert with resident weights."""
    num_tokens, topk = ids.shape
    num_experts = w13.shape[0]
    output = torch.zeros_like(input)
    flat_ids = ids.view(-1)
    flat_routing = routing.view(-1)
    for expert in range(num_experts):
        routes = (flat_ids == expert).nonzero(as_tuple=True)[0]
        if routes.numel() == 0:
            continue
        tokens = routes // topk
        gate_up = input.index_select(0, tokens) @ w13[expert].t()
        gate, up = gate_up.chunk(2, dim=-1)
        activated = (F.silu(gate.clamp(max=10.0)) * up.clamp(-10.0, 10.0)).to(
            input.dtype
        )
        contribution = (activated @ w2[expert].t()) * flat_routing[routes].to(
            input.dtype
        ).unsqueeze(1)
        output.index_add_(0, tokens, contribution)
    return output


def bench(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=str, default="1,8,32,128,512,2048,8192")
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=2048)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260731)
    return parser.parse_args()


def main():
    args = parse_args()
    token_counts = [int(t) for t in args.tokens.split(",")]
    E, H, M, K = args.experts, args.hidden, args.intermediate, args.topk
    dtype = torch.bfloat16
    device = "cuda"

    # infinicore's extension links libtorch, so it must be imported after torch.
    import infinicore

    torch.manual_seed(args.seed)
    infinicore.set_device(infinicore.device("cuda", 0))

    # Random packed FP4 weights, scale exponents in [123, 128] i.e. 2^-4..2^1.
    w13_packed = torch.randint(
        0, 256, (E, 2 * M, H // 2), dtype=torch.uint8, device=device
    )
    w13_scale = torch.randint(
        123, 129, (E, 2 * M, H // 32), dtype=torch.uint8, device=device
    )
    w2_packed = torch.randint(0, 256, (E, H, M // 2), dtype=torch.uint8, device=device)
    w2_scale = torch.randint(
        123, 129, (E, H, M // 32), dtype=torch.uint8, device=device
    )

    # bf16 baseline keeps the *dequantized* weights resident: same effective
    # weights, so the comparison isolates kernel/footprint cost, not quant error.
    w13_bf16 = dequantize_mxfp4_chunked(w13_packed, w13_scale, dtype)
    w2_bf16 = dequantize_mxfp4_chunked(w2_packed, w2_scale, dtype)

    fp4_bytes = sum(t.numel() for t in (w13_packed, w13_scale, w2_packed, w2_scale))
    bf16_bytes = (w13_bf16.numel() + w2_bf16.numel()) * 2
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"shape: E={E} H={H} I={M} K={K} dtype=bf16 activation=SwigluLimit10")
    print(
        f"weights: mxfp4 {fp4_bytes / 2**30:.2f} GiB "
        f"vs bf16 {bf16_bytes / 2**30:.2f} GiB "
        f"({bf16_bytes / fp4_bytes:.2f}x)"
    )

    def make_fused(x, ids, routing):
        tensors = {
            name: infinicore.from_torch(t.contiguous())
            for name, t in {
                "x": x,
                "ids": ids,
                "routing": routing,
                "w13_packed": w13_packed,
                "w13_scale": w13_scale,
                "w2_packed": w2_packed,
                "w2_scale": w2_scale,
            }.items()
        }
        out_buf = torch.empty_like(x)
        out_ic = infinicore.from_torch(out_buf)

        def run():
            infinicore.nn.functional.fused_moe_mxfp4(
                tensors["x"],
                tensors["ids"],
                tensors["routing"],
                tensors["w13_packed"],
                tensors["w13_scale"],
                tensors["w2_packed"],
                tensors["w2_scale"],
                activation=ACT_SWIGLU_LIMIT_10,
                out=out_ic,
            )

        return run, out_buf

    # Correctness gate at small T: fused op vs per-token fp32 reference vs
    # the grouped baseline used for timing.
    x_chk = torch.randn(8, H, dtype=dtype, device=device)
    ids_chk = torch.randint(0, E, (8, K), dtype=torch.int32, device=device)
    raw = torch.rand(8, K, device=device)
    rw_chk = raw / raw.sum(dim=-1, keepdim=True)
    fused_run, fused_out = make_fused(x_chk, ids_chk, rw_chk)
    fused_run()
    torch.cuda.synchronize()
    ref = torch_moe_reference(x_chk, ids_chk, rw_chk, w13_bf16, w2_bf16)
    grouped = torch_moe_grouped(x_chk, ids_chk.long(), rw_chk, w13_bf16, w2_bf16)

    def err(a, b):
        diff = (a.float() - b.float()).abs()
        return f"max|diff|={diff.max().item():.3e} (rel={diff.max().item() / b.abs().max().item():.2e})"

    print(f"correctness @T=8: |ref|max={ref.abs().max().item():.3e}")
    print(f"  fused   vs ref: {err(fused_out, ref)}")
    print(f"  grouped vs ref: {err(grouped, ref)}")

    print(f"\n{'tokens':>8} {'fused ms':>10} {'bf16 ms':>10} {'speedup':>8}")
    for T in token_counts:
        x = torch.randn(T, H, dtype=dtype, device=device)
        ids = torch.randint(0, E, (T, K), dtype=torch.int32, device=device)
        raw = torch.rand(T, K, device=device)
        rw = raw / raw.sum(dim=-1, keepdim=True)

        fused_run, _ = make_fused(x, ids, rw)
        fused_iters = max(10, min(200, 65536 // T))
        fused_ms = bench(fused_run, warmup=5, iters=fused_iters) * 1e3

        baseline_iters = max(3, min(30, 4096 // T))
        baseline_ms = (
            bench(
                lambda: torch_moe_grouped(x, ids.long(), rw, w13_bf16, w2_bf16),
                warmup=2,
                iters=baseline_iters,
            )
            * 1e3
        )
        print(
            f"{T:>8} {fused_ms:>10.3f} {baseline_ms:>10.3f} {baseline_ms / fused_ms:>7.2f}x"
        )


if __name__ == "__main__":
    main()

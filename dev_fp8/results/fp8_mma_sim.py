#!/usr/bin/env python3
"""CPU emulation of the FP8 blockwise GEMM mma path (fp8_blockwise_gemm_mma_kernel).

Verifies, without a GPU:
  1. decodePair bit tricks: bf16 variant == true e4m3 value * 2^-120 exactly,
     f16 variant == value * 2^-8 exactly (all 256 codes except the NaN code).
  2. Full kernel data flow (stage -> smem -> fragment gather -> mma -> promote
     -> epilogue) reproduces the reference blockwise GEMM, including M/N tails,
     block_n/block_k variants, and M_BLOCKS in {1, 2}.

Fragments are gathered with the kernel's exact index formulas and scattered
into full A(16x16)/B(16x8) tiles per the PTX ISA m16n8k16 mapping:
  lane = 4g + t
  A(row): a0=A[g][2t,2t+1] a1=A[g+8][2t,2t+1] a2=A[g][2t+8,2t+9] a3=A[g+8][2t+8,+9]
  B(col): b0=B[2t,2t+1][g] b1=B[2t+8,2t+9][g]
  C: c0/c1=(g,2t/2t+1) c2/c3=(g+8,2t/2t+1)
Any mismatch between the kernel's gather formulas and the PTX mapping makes
the emulated output diverge from the reference GEMM.
"""

import math
import random
import struct

N_TILE = 32
K_CHUNK = 128
THREADS = 128
A_STRIDE = 136  # elements
W_STRIDE = 144  # bytes


def e4m3_decode(code):
    """Exact decode, matching infiniopFp8E4m3Decode (incl. NaN on 0x7F)."""
    mag = code & 0x7F
    e, m = mag >> 3, mag & 7
    if e == 0xF and m == 7:
        return math.nan
    v = m * 0.001953125 if e == 0 else math.ldexp(1.0 + m * 0.125, e - 7)
    return -v if code & 0x80 else v


def decode_pair_bf16_bits(two):
    lo = ((two & 0x7F) << 4) | ((two & 0x80) << 8)
    hi = ((two & 0x7F00) >> 4) | (two & 0x8000)
    return lo | (hi << 16)


def decode_pair_f16_bits(two):
    lo = ((two & 0x7F) << 7) | ((two & 0x80) << 8)
    hi = ((two & 0x7F00) >> 1) | (two & 0x8000)
    return lo | (hi << 16)


def bf16_bits_to_float(bits):
    return struct.unpack(">f", struct.pack(">I", (bits & 0xFFFF) << 16))[0]


def f16_bits_to_float(bits):
    # IEEE half -> float, exact for any half value
    s = (bits >> 15) & 1
    e = (bits >> 10) & 0x1F
    m = bits & 0x3FF
    if e == 0:
        v = math.ldexp(m / 1024.0, -14)
    elif e == 31:
        v = math.inf if m == 0 else math.nan
    else:
        v = math.ldexp(1.0 + m / 1024.0, e - 15)
    return -v if s else v


def check_decode_tricks():
    for code in range(256):
        if (code & 0x7F) == 0x7F:
            continue  # NaN code: not special-cased in the kernel
        ref = e4m3_decode(code)
        two = code | (code << 8)
        b = decode_pair_bf16_bits(two)
        lo = bf16_bits_to_float(b & 0xFFFF) * 2.0**120
        hi = bf16_bits_to_float(b >> 16) * 2.0**120
        assert lo == ref and hi == ref, f"bf16 trick mismatch code={code}: {lo}/{hi} vs {ref}"
        f = decode_pair_f16_bits(two)
        lo = f16_bits_to_float(f & 0xFFFF) * 256.0
        hi = f16_bits_to_float(f >> 16) * 256.0
        assert lo == ref and hi == ref, f"f16 trick mismatch code={code}: {lo}/{hi} vs {ref}"
    print("decode tricks OK: bf16 x2^-120 / f16 x2^-8, bit-exact vs e4m3 for all non-NaN codes")


def run_kernel(a, q, scales, M, N, K, block_n, block_k, m_blocks, decode_scale):
    """Emulate fp8_blockwise_gemm_mma_kernel in float64 (index/logic exact)."""
    A_ROWS = m_blocks * 16
    nchunks = K // K_CHUNK
    kb_per_scale = block_k // K_CHUNK
    out = [[None] * N for _ in range(M)]

    for bx in range((N + N_TILE - 1) // N_TILE):
        for by in range((M + A_ROWS - 1) // A_ROWS):
            n_base = bx * N_TILE
            m_base = by * A_ROWS
            sA = [[[0.0] * A_STRIDE for _ in range(A_ROWS)] for _ in range(2)]
            sW = [[[0] * W_STRIDE for _ in range(N_TILE)] for _ in range(2)]

            def stage(c):
                a_stage = {}
                w_stage = {}
                k0 = c * K_CHUNK
                for tid in range(THREADS):
                    for i in range(A_ROWS // 8):
                        idx = tid + i * THREADS
                        m = m_base + (idx >> 4)
                        vals = (0.0,) * 8 if m >= M else tuple(a[m][k0 + (idx & 15) * 8 + j] for j in range(8))
                        a_stage[(tid, i)] = vals
                    for i in range(2):
                        idx = tid + i * THREADS
                        n = n_base + (idx >> 3)
                        vals = (0,) * 16 if n >= N else tuple(q[n][k0 + (idx & 7) * 16 + j] for j in range(16))
                        w_stage[(tid, i)] = vals
                return a_stage, w_stage

            def store(buf, a_stage, w_stage):
                for tid in range(THREADS):
                    for i in range(A_ROWS // 8):
                        idx = tid + i * THREADS
                        row, seg = idx >> 4, idx & 15
                        for j in range(8):
                            sA[buf][row][seg * 8 + j] = a_stage[(tid, i)][j]
                    for i in range(2):
                        idx = tid + i * THREADS
                        row, seg = idx >> 3, idx & 7
                        for j in range(16):
                            sW[buf][row][seg * 16 + j] = w_stage[(tid, i)][j]

            c_fin = {(tid, mb, i): 0.0 for tid in range(THREADS) for mb in range(m_blocks) for i in range(4)}

            store(0, *stage(0))
            for c in range(nchunks):
                buf = c & 1
                nxt = stage(c + 1) if c + 1 < nchunks else None

                c_part = {(tid, mb, i): 0.0 for tid in range(THREADS) for mb in range(m_blocks) for i in range(4)}
                for warp in range(4):
                    for step in range(K_CHUNK // 16):
                        kk = step * 16
                        A_tiles = [[[0.0] * 16 for _ in range(16)] for _ in range(m_blocks)]
                        B_tile = [[0.0] * 8 for _ in range(16)]
                        for lane in range(32):
                            g, t = lane >> 2, lane & 3
                            wrow = sW[buf][warp * 8 + g]
                            # kernel's B gather: LDS.U16 at (kk+2t) and (kk+2t+8), decodePair each
                            b0 = [e4m3_decode(wrow[kk + 2 * t + j]) / decode_scale for j in range(2)]
                            b1 = [e4m3_decode(wrow[kk + 2 * t + 8 + j]) / decode_scale for j in range(2)]
                            B_tile[2 * t][g] = b0[0]
                            B_tile[2 * t + 1][g] = b0[1]
                            B_tile[2 * t + 8][g] = b1[0]
                            B_tile[2 * t + 9][g] = b1[1]
                            for mb in range(m_blocks):
                                arow0 = sA[buf][mb * 16 + g]
                                arow1 = sA[buf][mb * 16 + g + 8]
                                # kernel's A gather: LDS.U32 at (kk+2t) / (kk+2t+8), rows g / g+8
                                A_tiles[mb][g][2 * t] = arow0[kk + 2 * t]
                                A_tiles[mb][g][2 * t + 1] = arow0[kk + 2 * t + 1]
                                A_tiles[mb][g + 8][2 * t] = arow1[kk + 2 * t]
                                A_tiles[mb][g + 8][2 * t + 1] = arow1[kk + 2 * t + 1]
                                A_tiles[mb][g][2 * t + 8] = arow0[kk + 2 * t + 8]
                                A_tiles[mb][g][2 * t + 9] = arow0[kk + 2 * t + 9]
                                A_tiles[mb][g + 8][2 * t + 8] = arow1[kk + 2 * t + 8]
                                A_tiles[mb][g + 8][2 * t + 9] = arow1[kk + 2 * t + 9]
                        # mma semantics: full 16x8x16 product from the fragments
                        for lane in range(32):
                            tid = warp * 32 + lane
                            g, t = lane >> 2, lane & 3
                            for mb in range(m_blocks):
                                for i in range(4):
                                    row = g + (i >> 1) * 8
                                    col = 2 * t + (i & 1)
                                    c_part[(tid, mb, i)] += sum(
                                        A_tiles[mb][row][k2] * B_tile[k2][col] for k2 in range(16))

                # promote with the folded scale (per thread pair of columns)
                for tid in range(THREADS):
                    lane, warp = tid & 31, tid >> 5
                    t = lane & 3
                    n0 = n_base + warp * 8 + t * 2
                    s = 0.0 if n0 >= N else scales[n0 // block_n][c // kb_per_scale] * decode_scale
                    for mb in range(m_blocks):
                        for i in range(4):
                            c_fin[(tid, mb, i)] += s * c_part[(tid, mb, i)]

                if nxt is not None:
                    store(buf ^ 1, *nxt)

            # epilogue
            for tid in range(THREADS):
                lane, warp = tid & 31, tid >> 5
                g, t = lane >> 2, lane & 3
                for mb in range(m_blocks):
                    for i in range(4):
                        m = m_base + mb * 16 + g + (i >> 1) * 8
                        n = n_base + warp * 8 + t * 2 + (i & 1)
                        if m < M and n < N:
                            out[m][n] = c_fin[(tid, mb, i)]
    return out


def reference(a, q, scales, M, N, K, block_n, block_k):
    w = [[e4m3_decode(q[n][k]) * scales[n // block_n][k // block_k] for k in range(K)] for n in range(N)]
    return [[sum(a[m][k] * w[n][k] for k in range(K)) for n in range(N)] for m in range(M)]


def run_suite():
    random.seed(1)
    fails = 0
    cases = [
        # (M, N, K, block_n, block_k)
        (16, 32, 128, 16, 128),
        (9, 32, 256, 16, 256),
        (13, 64, 384, 32, 128),
        (16, 48, 128, 16, 128),
        (1, 32, 128, 16, 128),  # M_BLOCKS=1 with M<16 padding
        (32, 32, 256, 16, 128),
        (17, 32, 128, 16, 128),  # M_BLOCKS=2 tail
        (31, 96, 512, 32, 256),
        (24, 64, 128, 64, 128),
        (12, 80, 256, 16, 128),  # N=80: N%32=16 tail
    ]
    for (M, N, K, bn, bk) in cases:
        m_blocks = 1 if M <= 16 else 2
        a = [[random.gauss(0, 1) for _ in range(K)] for _ in range(M)]
        q = [[random.randrange(256) for _ in range(K)] for _ in range(N)]
        # avoid NaN codes 0x7F/0xFF (kernel does not special-case them)
        q = [[c if (c & 0x7F) != 0x7F else 0x7E for c in row] for row in q]
        scales = [[random.uniform(0.005, 0.025) for _ in range(K // bk)] for _ in range(N // bn)]
        got = run_kernel(a, q, scales, M, N, K, bn, bk, m_blocks, 2.0**120)
        ref = reference(a, q, scales, M, N, K, bn, bk)
        ok = True
        for m in range(M):
            for n in range(N):
                if got[m][n] is None:
                    print(f"MISSING out[{m}][{n}] case={(M, N, K, bn, bk)}")
                    fails += 1
                    ok = False
                    break
                if not math.isclose(got[m][n], ref[m][n], rel_tol=1e-9, abs_tol=1e-8):
                    print(f"MISMATCH out[{m}][{n}]={got[m][n]} ref={ref[m][n]} case={(M, N, K, bn, bk)}")
                    fails += 1
                    ok = False
                    break
            if not ok:
                break
        if ok:
            print(f"case M={M} N={N} K={K} bn={bn} bk={bk} mb={m_blocks}: OK")
    print("ALL PASS" if fails == 0 else f"{fails} FAILURES")


if __name__ == "__main__":
    check_decode_tricks()
    run_suite()

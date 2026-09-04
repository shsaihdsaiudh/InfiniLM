#!/usr/bin/env python3
"""CPU emulation of the FP8 split-kv decode kernel logic (kernel_fp8.cuh v2).

Verifies, without a GPU:
  1. Cursor/shard logic: for every (seq_len, pbs, num_warps, num_splits), the
     (split, warp) -> token assignment partitions [0, seq_len) exactly.
  2. num_splits == 1 reproduces the pre-split-kv token assignment.
  3. Merge math: per-warp online softmax -> per-CTA merge -> cross-CTA combine
     (with the -inf guards) equals the direct softmax, including empty shards.
"""

import math
import random

NUM_WARPS = 32


def simulate_cta(seq_len, pbs, num_splits, split_idx, k, v, q, scale_log2, alibi_slope):
    """Emulate one CTA: returns (m_total, l_total, acc[HEAD]) post cross-warp merge."""
    shard = (seq_len + num_splits - 1) // num_splits
    tok_lo = min(split_idx * shard, seq_len)
    tok_hi = min(seq_len, tok_lo + shard)

    def token_begin(lb):
        return max(tok_lo - lb * pbs, 0)

    def token_end(lb):
        return min(pbs, tok_hi - lb * pbs)

    m_part = [-math.inf] * NUM_WARPS
    l_part = [0.0] * NUM_WARPS
    acc_part = [[0.0] * len(q) for _ in range(NUM_WARPS)]
    assigned = []

    for warp in range(NUM_WARPS):
        m, l = -math.inf, 0.0
        acc = [0.0] * len(q)

        cur_lb = tok_lo // pbs
        cur_tb = token_begin(cur_lb) + warp
        while cur_lb * pbs < tok_hi and cur_tb >= token_end(cur_lb):
            cur_lb += 1
            cur_tb = token_begin(cur_lb) + warp
        has_cur = cur_lb * pbs < tok_hi

        while has_cur:
            # advance
            nxt_lb, nxt_tb = cur_lb, cur_tb + NUM_WARPS
            if nxt_tb >= token_end(nxt_lb):
                nxt_lb += 1
                nxt_tb = token_begin(nxt_lb) + warp
                while nxt_lb * pbs < tok_hi and nxt_tb >= token_end(nxt_lb):
                    nxt_lb += 1
                    nxt_tb = token_begin(nxt_lb) + warp
            has_nxt = nxt_lb * pbs < tok_hi

            t = cur_lb * pbs + cur_tb
            assigned.append(t)
            qk = sum(qj * kj for qj, kj in zip(q, k[t]))
            score = qk * scale_log2
            if alibi_slope != 0.0:
                score += (alibi_slope * (t - (seq_len - 1))) * math.log2(math.e)
            m_new = max(m, score)
            alpha = 2.0 ** (m - m_new) if m != -math.inf else 0.0
            beta = 2.0 ** (score - m_new)
            l = l * alpha + beta
            m = m_new
            for j in range(len(q)):
                acc[j] = acc[j] * alpha + beta * v[t][j]

            cur_lb, cur_tb, has_cur = nxt_lb, nxt_tb, has_nxt

        m_part[warp] = m
        l_part[warp] = l
        acc_part[warp] = acc

    # cross-warp merge (kernel_fp8.cuh)
    m_total = max(m_part)
    wgt = [0.0 if mp == -math.inf else 2.0 ** (mp - m_total) for mp in m_part]
    l_total = sum(lp * w for lp, w in zip(l_part, wgt))
    merged = [sum(acc_part[w][j] * wgt[w] for w in range(NUM_WARPS)) for j in range(len(q))]
    pm = m_total if l_total > 0.0 else -math.inf
    return pm, l_total, merged, assigned


def direct_softmax(seq_len, k, v, q, scale_log2, alibi_slope):
    scores = []
    for t in range(seq_len):
        s = sum(qj * kj for qj, kj in zip(q, k[t])) * scale_log2
        if alibi_slope != 0.0:
            s += (alibi_slope * (t - (seq_len - 1))) * math.log2(math.e)
        scores.append(s)
    m = max(scores)
    l = sum(2.0 ** (s - m) for s in scores)
    out = [sum((2.0 ** (scores[t] - m)) * v[t][j] for t in range(seq_len)) / (l + 1e-6) for j in range(len(q))]
    return out


def combine(partials, num_splits):
    """Emulate flashAttentionDecodeFp8SplitKvCombineKernel."""
    m_total = max(p[0] for p in partials)
    wgt = [0.0 if p[0] == -math.inf else 2.0 ** (p[0] - m_total) for p in partials]
    l_total = sum(p[1] * w for p, w in zip(partials, wgt))
    hd = len(partials[0][2])
    return [sum(p[2][j] * w for p, w in zip(partials, wgt)) / (l_total + 1e-6) for j in range(hd)]


def approx(a, b, tol=1e-9):
    return all(abs(x - y) <= tol * max(1.0, abs(x), abs(y)) for x, y in zip(a, b))


random.seed(0)
fails = 0

# --- 1. token assignment partitions [0, seq_len) exactly; num_splits==1 == legacy ---
for seq_len in [1, 2, 3, 15, 16, 17, 31, 32, 33, 100, 127, 128, 129, 1000, 1024, 4096, 16384]:
    for pbs in [1, 2, 15, 16, 17, 64]:
        if pbs > max(seq_len, 16):
            continue
        hd = 4
        q = [0.0] * hd
        k = [[0.0] * hd for _ in range(seq_len)]
        v = [[0.0] * hd for _ in range(seq_len)]
        legacy = sorted(simulate_cta(seq_len, pbs, 1, 0, k, v, q, 1.0, 0.0)[3])
        assert legacy == list(range(seq_len)), f"legacy mismatch sl={seq_len} pbs={pbs}"
        for num_splits in [2, 3, 4, 5, 8]:
            all_t = []
            for s in range(num_splits):
                all_t += simulate_cta(seq_len, pbs, num_splits, s, k, v, q, 1.0, 0.0)[3]
            all_t.sort()
            if all_t != list(range(seq_len)):
                print(f"FAIL partition sl={seq_len} pbs={pbs} S={num_splits}")
                fails += 1

# --- 2. numeric merge math vs direct softmax (random data, with/without alibi) ---
for trial in range(300):
    seq_len = random.choice([1, 2, 5, 16, 17, 63, 100, 257, 1023, 2048])
    pbs = random.choice([1, 4, 16, 32])
    num_splits = random.choice([1, 2, 3, 4, 8])
    hd = random.choice([2, 4])
    alibi = random.choice([0.0, 0.125])
    scale_log2 = (1.0 / math.sqrt(hd)) * math.log2(math.e)
    q = [random.gauss(0, 1) for _ in range(hd)]
    k = [[random.gauss(0, 2) for _ in range(hd)] for _ in range(seq_len)]
    v = [[random.gauss(0, 2) for _ in range(hd)] for _ in range(seq_len)]

    partials = [simulate_cta(seq_len, pbs, num_splits, s, k, v, q, scale_log2, alibi)[:3] for s in range(num_splits)]
    got = combine(partials, num_splits)
    if num_splits == 1:
        pm, pl, macc = partials[0]
        got1 = [x / (pl + 1e-6) for x in macc]
        assert approx(got, got1), "num_splits==1 combine != direct CTA normalize"
    ref = direct_softmax(seq_len, k, v, q, scale_log2, alibi)
    if not approx(got, ref, tol=1e-6):
        print(f"FAIL math trial={trial} sl={seq_len} pbs={pbs} S={num_splits} hd={hd} alibi={alibi}")
        fails += 1

print("ALL PASS" if fails == 0 else f"{fails} FAILURES")

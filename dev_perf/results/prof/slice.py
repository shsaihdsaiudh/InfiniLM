"""Slice nsys sqlite traces into phases and compare GPU time usage."""
import sqlite3
import sys
import collections

def load(path):
    con = sqlite3.connect(path)
    cur = con.cursor()
    kernels = cur.execute(
        "SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k "
        "JOIN StringIds s ON k.demangledName = s.id"
    ).fetchall()
    try:
        memcpys = cur.execute(
            "SELECT start, end, bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY"
        ).fetchall()
        memsets = cur.execute(
            "SELECT start, end, bytes FROM CUPTI_ACTIVITY_KIND_MEMSET"
        ).fetchall()
    except sqlite3.Error:
        memcpys, memsets = [], []
    con.close()
    return kernels, memcpys, memsets

def phases(path, bucket_ms=200):
    kernels, memcpys, memsets = load(path)
    t0 = min(k[0] for k in kernels)
    t1 = max(k[1] for k in kernels)
    span_s = (t1 - t0) / 1e9
    b = bucket_ms * 10**6
    nb = int((t1 - t0) / b) + 1
    busy = [0.0] * nb
    cnt = [0] * nb
    for st, en, _ in kernels:
        i = min(int((st - t0) / b), nb - 1)
        busy[i] += (en - st) / 1e9
        cnt[i] += 1
    print(f"\n== {path}: span {span_s:.1f}s, {len(kernels)} kernels, "
          f"{len(memcpys)} memcpy, {len(memsets)} memset")
    for i in range(nb):
        pct = 100 * busy[i] / (bucket_ms / 1e3)
        bar = "#" * int(pct / 2)
        print(f"  {(i*b)/1e9:7.1f}s busy={pct:5.1f}% nk={cnt[i]:4d} {bar}")
    return kernels, memcpys, memsets, t0

for p in sys.argv[1:]:
    phases(p)

"""Shared workload definition for the offline dual-engine perf baseline.

Both bench entry points (infinilm / vllm) import from here so the workload
is byte-identical across engines. Stdlib + huggingface_hub only.
"""

import os
import subprocess
import threading
import time

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SHORT_PROMPTS = [
    "如果猫能写诗，它们会写些什么？",
    "描述一个没有重力的世界。",
    "如果地球停止自转，会发生什么？",
    "假设你是一只会飞的鲸鱼，描述你的日常生活。",
]

_BASE_PARA = (
    "大语言模型推理引擎的核心挑战在于如何高效利用 GPU 的算力与显存带宽。"
    "预填充阶段是计算密集型的，而解码阶段则主要受限于显存带宽，"
    "因为每一步都需要读取全部的模型权重与不断增长的 KV 缓存。"
    "连续的批处理调度与前缀缓存是提升服务端吞吐的关键技术。"
)

LONG_PROMPT = _BASE_PARA * 40  # ~2k tokens; actual count recorded at runtime


def _batch_prompts(n: int) -> list[str]:
    return [f"问题 {i + 1}：{SHORT_PROMPTS[i % len(SHORT_PROMPTS)]}" for i in range(n)]


# name, prompts, max_tokens
WORKLOADS = [
    ("w1_short_decode", SHORT_PROMPTS[:1], 128),
    ("w2_long_prefill", [LONG_PROMPT], 128),
    ("w3_batch32", _batch_prompts(32), 128),
    ("w4_long_decode", SHORT_PROMPTS[1:2], 1024),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def resolve_model_path(name_or_path: str) -> str:
    """Accept a local dir or an HF repo id; return a local directory path."""
    if os.path.isdir(name_or_path):
        return os.path.abspath(name_or_path)
    from huggingface_hub import snapshot_download

    return snapshot_download(name_or_path)


class MemSampler(threading.Thread):
    """Sample `nvidia-smi` memory.used in the background; track peak."""

    def __init__(self, interval: float = 0.2):
        super().__init__(daemon=True)
        self.interval = interval
        self.peak = 0
        self.latest = 0
        self._stop_event = threading.Event()

    def run(self):
        while not self._stop_event.is_set():
            try:
                out = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                ).strip()
                self.latest = int(out.splitlines()[0])
                self.peak = max(self.peak, self.latest)
            except Exception:
                pass
            time.sleep(self.interval)

    def stop(self):
        self._stop_event.set()
        self.join(timeout=2)

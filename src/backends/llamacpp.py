"""
Generic llama.cpp backend (CPU, cross-platform).

Wraps llama-cpp-python with:
  - load(): construct the model, measuring wall-clock load time and the process
    RSS delta (the real resident memory the INT4 weights + context occupy).
  - generate(): run a prompt, measuring Time-To-First-Token (TTFT) and tokens/s
    via the streaming API.

Kept deliberately thin so the Intel path (ipex/openvino) can mirror this API.
"""
from __future__ import annotations

import os
import time
from typing import Optional

try:
    import psutil
    _HAVE_PSUTIL = True
except Exception:
    _HAVE_PSUTIL = False


def _rss_bytes() -> Optional[int]:
    if _HAVE_PSUTIL:
        try:
            return psutil.Process(os.getpid()).memory_info().rss
        except Exception:
            return None
    # Linux stdlib fallback via /proc/self/statm (pages).
    try:
        with open("/proc/self/statm") as f:
            rss_pages = int(f.read().split()[1])
        return rss_pages * os.sysconf("SC_PAGE_SIZE")
    except Exception:
        return None


class LlamaCppModel:
    def __init__(self, handle, load_stats: dict):
        self._llm = handle
        self.load_stats = load_stats

    @classmethod
    def load(cls, model_path: str, n_threads: Optional[int] = None,
             n_ctx: int = 2048, verbose: bool = False) -> "LlamaCppModel":
        from llama_cpp import Llama
        rss_before = _rss_bytes()
        t0 = time.perf_counter()
        llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,        # None -> llama.cpp auto-picks
            n_gpu_layers=0,             # CPU-only path (per project decision)
            verbose=verbose,
        )
        load_time = time.perf_counter() - t0
        rss_after = _rss_bytes()
        stats = {
            "model_path": model_path,
            "file_bytes": os.path.getsize(model_path),
            "load_time_s": round(load_time, 2),
            "rss_before_bytes": rss_before,
            "rss_after_bytes": rss_after,
            "rss_delta_bytes": (rss_after - rss_before) if (rss_before and rss_after) else None,
            "n_ctx": n_ctx,
            "n_threads": n_threads,
        }
        return cls(llm, stats)

    def generate(self, prompt: str, max_tokens: int = 128,
                 temperature: float = 0.0) -> dict:
        """Run one prompt; return text + TTFT + throughput."""
        t0 = time.perf_counter()
        ttft = None
        pieces = []
        n_out = 0
        stream = self._llm(
            prompt, max_tokens=max_tokens, temperature=temperature, stream=True,
        )
        for chunk in stream:
            tok = chunk["choices"][0]["text"]
            if ttft is None:
                ttft = time.perf_counter() - t0
            pieces.append(tok)
            n_out += 1
        total = time.perf_counter() - t0
        gen_time = max(total - (ttft or 0), 1e-6)
        return {
            "text": "".join(pieces),
            "ttft_s": round(ttft, 4) if ttft is not None else None,
            "total_s": round(total, 4),
            "output_tokens": n_out,
            "tokens_per_s": round(n_out / gen_time, 2) if n_out else 0.0,
        }

    def peak_rss_bytes(self) -> Optional[int]:
        return _rss_bytes()

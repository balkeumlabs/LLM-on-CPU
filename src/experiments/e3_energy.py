"""
E3 - Energy / Clinical-Intelligence-per-Watt (CI/W).

Reuses E2's benchmark run (memoized) and folds in the power telemetry sampled
during generation. CI/W proxy = tokens generated per joule (CPU-only and, if
available, CPU+GPU). Power is estimate-grade (RAPL when readable, else TDP),
which is acceptable per the project decision and labeled as such.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run(caps: dict, args) -> dict:
    import workload
    b = workload.run_benchmark(caps, args)
    energy = b["energy"]
    tokens = b["total_output_tokens"]
    cpu_j = energy["cpu"]["joules"]
    gpu = energy.get("gpu", {})
    gpu_j = gpu.get("joules")

    tok_per_joule_cpu = round(tokens / cpu_j, 3) if cpu_j else None
    total_j = cpu_j + (gpu_j or 0)
    tok_per_joule_total = round(tokens / total_j, 3) if total_j else None

    summary = (f"{b['tokens_per_s_mean']} tok/s at CPU {energy['cpu']['avg_w']}W "
               f"({energy['cpu']['method']}) -> {tok_per_joule_cpu} tokens/J (CPU); "
               f"GPU {gpu.get('avg_w', 'n/a')}W")

    return {
        "key": "E3",
        "title": "Energy / Clinical-Intelligence-per-Watt",
        "status": "ok",
        "summary": summary,
        "tier": b["tier"],
        "power_method_cpu": energy["cpu"]["method"],
        "cpu_avg_w": energy["cpu"]["avg_w"],
        "gpu_avg_w": gpu.get("avg_w"),
        "elapsed_s": energy["elapsed_s"],
        "total_output_tokens": tokens,
        "ci_per_watt": {
            "tokens_per_joule_cpu": tok_per_joule_cpu,
            "tokens_per_joule_cpu_plus_gpu": tok_per_joule_total,
            "note": "CI/W proxy = output tokens per joule; power is estimate-grade",
        },
    }

"""
E2 - Inference throughput & TTFT.

Feeds synthetic FHIR prompts through the selected accel path and reports
Time-To-First-Token and tokens/s. On the colleague's Intel laptop the same
code exercises DL Boost/AMX (via the Intel backend) for the comparison.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run(caps: dict, args) -> dict:
    import workload
    b = workload.run_benchmark(caps, args)
    summary = (f"{b['tier']} on {caps['cpu']['vendor']} ({caps['decisions']['accel_path']}): "
               f"TTFT {b['ttft_mean_s']}s mean, {b['tokens_per_s_mean']} tok/s mean "
               f"over {b['n_prompts']} prompts ({b['n_threads']} threads)")
    return {
        "key": "E2",
        "title": "Inference throughput & TTFT",
        "status": "ok",
        "summary": summary,
        "tier": b["tier"],
        "accel_path": caps["decisions"]["accel_path"],
        "load_time_s": b["load_time_s"],
        "ttft_mean_s": b["ttft_mean_s"],
        "ttft_min_s": b["ttft_min_s"],
        "tokens_per_s_mean": b["tokens_per_s_mean"],
        "n_prompts": b["n_prompts"],
        "n_threads": b["n_threads"],
    }

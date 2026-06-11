"""
Shared inference workload for E2 (throughput/TTFT) and E3 (energy/CI-W).

Runs the selected model over a sample of the synthetic FHIR prompts ONCE, with
power sampling wrapped around generation, and memoizes the result so E2 and E3
present two views of the same run instead of loading the model twice.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPTS = os.path.join(HERE, "data", "prompts.jsonl")

_CACHE: dict = {}


def _load_prompts(n: int) -> list[dict]:
    if not os.path.exists(PROMPTS):
        raise NotImplementedError(
            "no prompts found. Run: python src/gen_synthetic_fhir.py"
        )
    out = []
    with open(PROMPTS) as f:
        for line in f:
            out.append(json.loads(line))
            if len(out) >= n:
                break
    return out


def run_benchmark(caps: dict, args, n_prompts: int = 12, max_tokens: int = 96) -> dict:
    """Load model, run n prompts with power sampling. Memoized by model path."""
    import fetch_models
    from backends.llamacpp import LlamaCppModel
    from power import PowerSampler

    tier = caps["decisions"]["model_tier"]["tier"]
    model_path = fetch_models.resolve_path(tier)
    if model_path is None:
        for t in ("capacity", "edge"):
            model_path = fetch_models.resolve_path(t)
            if model_path:
                tier = t
                break
    if model_path is None:
        raise NotImplementedError(
            "no GGUF found. Run: python src/fetch_models.py --tier edge (or capacity)"
        )

    cache_key = (model_path, n_prompts, max_tokens)
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    prompts = _load_prompts(n_prompts)
    n_threads = caps["cpu"].get("physical_cores") or None
    model = LlamaCppModel.load(model_path, n_threads=n_threads, n_ctx=2048)

    per_prompt = []
    with PowerSampler(caps) as ps:
        for p in prompts:
            g = model.generate(p["prompt"], max_tokens=max_tokens)
            per_prompt.append({"task": p["task"], "profile": p.get("profile"),
                               "ttft_s": g["ttft_s"], "tokens_per_s": g["tokens_per_s"],
                               "output_tokens": g["output_tokens"]})
    energy = ps.result()

    ttfts = [r["ttft_s"] for r in per_prompt if r["ttft_s"] is not None]
    tps = [r["tokens_per_s"] for r in per_prompt if r["tokens_per_s"]]
    total_tokens = sum(r["output_tokens"] for r in per_prompt)

    def _mean(xs):
        return round(sum(xs) / len(xs), 3) if xs else None

    result = {
        "tier": tier,
        "model_path": os.path.relpath(model_path, HERE),
        "n_prompts": len(per_prompt),
        "max_tokens": max_tokens,
        "n_threads": n_threads,
        "load_time_s": model.load_stats["load_time_s"],
        "ttft_mean_s": _mean(ttfts),
        "ttft_min_s": round(min(ttfts), 3) if ttfts else None,
        "tokens_per_s_mean": _mean(tps),
        "total_output_tokens": total_tokens,
        "energy": energy,
        "per_prompt": per_prompt,
    }
    _CACHE[cache_key] = result
    return result

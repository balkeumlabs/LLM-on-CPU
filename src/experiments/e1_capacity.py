"""
E1 - Capacity over TFLOPS (headline result).

Demonstrates the proposal's central thesis on a single machine:
  (a) the INT4 model's training/serving state loads into SYSTEM RAM and occupies
      a measured, stable footprint with ample headroom; while
  (b) the same weights CANNOT fit in the detected consumer GPU's VRAM -> an
      out-of-memory condition (treated ANALYTICALLY per project decision:
      file_bytes > VRAM is a definitive OOM before the first step).

Returns a structured result the harness folds into final_report.md.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _gb(x):
    return round(x / 1e9, 2) if x else None


def run(caps: dict, args) -> dict:
    import fetch_models
    from backends.llamacpp import LlamaCppModel

    tier = caps["decisions"]["model_tier"]["tier"]
    # honor an explicit --model override by matching the tier name if possible
    model_path = fetch_models.resolve_path(tier)
    if model_path is None:
        # fall back to whatever tier IS downloaded (edge first), so the pipeline
        # still produces a result on machines that only fetched the small model.
        for t in ("capacity", "edge"):
            model_path = fetch_models.resolve_path(t)
            if model_path:
                tier = t
                break
    if model_path is None:
        raise NotImplementedError(
            "no GGUF found. Run: python src/fetch_models.py --tier edge (or capacity)"
        )

    mem = caps["memory"]
    total_ram = (mem.get("total_gb") or 0) * 1e9
    n_threads = caps["cpu"].get("physical_cores") or None

    # --- (a) Load into RAM, measure footprint -----------------------------
    model = LlamaCppModel.load(model_path, n_threads=n_threads, n_ctx=2048)
    st = model.load_stats
    file_bytes = st["file_bytes"]
    rss_delta = st["rss_delta_bytes"]
    peak_rss = model.peak_rss_bytes()
    ram_util_pct = round(100 * (peak_rss or rss_delta or file_bytes) / total_ram, 1) if total_ram else None

    # --- (b) Analytical GPU-OOM vs each detected GPU ----------------------
    gpu_verdicts = []
    for g in caps["gpu"].get("gpus", []):
        vram_mib = g.get("memory_mib")
        if vram_mib is None:
            continue
        vram_bytes = vram_mib * 1024 * 1024
        fits = file_bytes <= vram_bytes
        gpu_verdicts.append({
            "gpu": g.get("name"),
            "vram_gb": _gb(vram_bytes),
            "weights_gb": _gb(file_bytes),
            "fits_in_vram": fits,
            "verdict": "FITS" if fits else "OOM (weights exceed VRAM before first step)",
        })
    if not gpu_verdicts:
        gpu_verdicts.append({"gpu": "none detected", "verdict": "n/a"})

    summary = (f"{tier} model: weights {_gb(file_bytes)} GB load into RAM "
               f"(peak RSS {_gb(peak_rss)} GB, ~{ram_util_pct}% of {_gb(total_ram)} GB); "
               + "; ".join(f"{v['gpu']}={v['verdict'].split(' ')[0]}" for v in gpu_verdicts))

    return {
        "key": "E1",
        "title": "Capacity over TFLOPS (RAM-fit vs GPU-OOM)",
        "status": "ok",
        "summary": summary,
        "model_tier": tier,
        "ram": {
            "total_gb": _gb(total_ram),
            "weights_file_gb": _gb(file_bytes),
            "load_rss_delta_gb": _gb(rss_delta),
            "peak_rss_gb": _gb(peak_rss),
            "ram_utilization_pct": ram_util_pct,
            "load_time_s": st["load_time_s"],
            "n_threads": n_threads,
        },
        "gpu_oom_analysis": gpu_verdicts,
    }

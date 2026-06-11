"""
Figure + table generation for the paper (M3+).

Reads a results/<host-stamp>/results.json and renders:
  - capacity_ram_vs_vram.png : the headline "Capacity over TFLOPS" chart —
    model weights & peak RSS against the GPU VRAM ceiling and system-RAM ceiling.
  - capacity_table.md        : the same numbers as a paper-ready markdown table.

matplotlib is optional; if missing, the markdown table is still written.
"""
from __future__ import annotations

import json
import os
import sys


def _load_e1(results_dir: str) -> dict | None:
    rp = os.path.join(results_dir, "results.json")
    if not os.path.exists(rp):
        return None
    for r in json.load(open(rp)):
        if r.get("key") == "E1" and r.get("status") == "ok":
            return r
    return None


def capacity_table_md(e1: dict) -> str:
    ram = e1["ram"]
    L = ["# E1 — Capacity over TFLOPS", "",
         f"Model tier: **{e1['model_tier']}**", "",
         "| Quantity | Value |", "|---|---|",
         f"| INT4 weights (file) | {ram['weights_file_gb']} GB |",
         f"| Peak resident memory (RSS) | {ram['peak_rss_gb']} GB |",
         f"| System RAM (total) | {ram['total_gb']} GB |",
         f"| RAM utilization | {ram['ram_utilization_pct']} % |",
         f"| Load time | {ram['load_time_s']} s |",
         f"| Threads | {ram['n_threads']} |",
         "", "## GPU fit analysis", "",
         "| GPU | VRAM | Weights | Verdict |", "|---|---|---|---|"]
    for g in e1["gpu_oom_analysis"]:
        L.append(f"| {g.get('gpu')} | {g.get('vram_gb','n/a')} GB | "
                 f"{g.get('weights_gb','n/a')} GB | {g.get('verdict')} |")
    L.append("")
    return "\n".join(L)


def plot_capacity(e1: dict, out_path: str) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    ram = e1["ram"]
    weights = ram["weights_file_gb"] or 0
    peak = ram["peak_rss_gb"] or 0
    total_ram = ram["total_gb"] or 0
    vram = None
    gpu_name = "GPU"
    for g in e1["gpu_oom_analysis"]:
        if g.get("vram_gb"):
            vram = g["vram_gb"]
            gpu_name = g.get("gpu", "GPU")
            break

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ["INT4 weights", "Peak RSS"]
    vals = [weights, peak]
    colors = ["#4C78A8", "#F58518"]
    ax.bar(bars, vals, color=colors, width=0.5, zorder=3)
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals) * 0.02, f"{v:.1f} GB", ha="center", va="bottom", fontsize=10)

    if vram:
        ax.axhline(vram, color="#E45756", linestyle="--", linewidth=2, zorder=2,
                   label=f"{gpu_name} VRAM ceiling ({vram:.1f} GB)")
    if total_ram:
        ax.axhline(total_ram, color="#54A24B", linestyle=":", linewidth=2, zorder=2,
                   label=f"System RAM ceiling ({total_ram:.0f} GB)")

    ax.set_ylabel("Memory (GB)")
    ax.set_title(f"Capacity over TFLOPS — {e1['model_tier']} INT4 model")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def generate(results_dir: str) -> list[str]:
    e1 = _load_e1(results_dir)
    if not e1:
        return []
    written = []
    table_path = os.path.join(results_dir, "capacity_table.md")
    with open(table_path, "w") as f:
        f.write(capacity_table_md(e1))
    written.append(table_path)

    png_path = os.path.join(results_dir, "capacity_ram_vs_vram.png")
    if plot_capacity(e1, png_path):
        written.append(png_path)
    return written


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else None
    if not d:
        # default to most recent results dir
        root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
        dirs = sorted((os.path.join(root, x) for x in os.listdir(root)), key=os.path.getmtime)
        d = dirs[-1] if dirs else None
    if not d:
        raise SystemExit("no results dir found")
    out = generate(d)
    print("wrote:" if out else "nothing written (no E1 result)")
    for p in out:
        print(" ", p)

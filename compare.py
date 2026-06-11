#!/usr/bin/env python3
"""
Cross-host comparison (M6) — merges results from this machine and the
colleague's Intel laptop into paper-ready AMD-vs-Intel tables and figures.

Reads every host under results/:
  - prefers results/<host>/results.json + capabilities.json (rich), else
  - parses the embedded JSON summary inside a final_report.md (e.g. one the
    colleague emailed back — drop it anywhere under results/).

Outputs to results/_comparison/:
  - comparison_table.md          side-by-side metric table
  - compare_throughput.png       tok/s + TTFT per host (the Intel-vs-generic delta)
  - compare_ciw.png              tokens/joule per host

Usage:
  python compare.py                       # scan results/
  python compare.py path/to/report.md ... # also ingest specific report files
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import summarize  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
OUTDIR = os.path.join(RESULTS, "_comparison")


def _from_results_dir(d: str) -> dict | None:
    rj, cj = os.path.join(d, "results.json"), os.path.join(d, "capabilities.json")
    if not (os.path.exists(rj) and os.path.exists(cj)):
        return None
    try:
        caps = json.load(open(cj))
        results = json.load(open(rj))
        host = os.path.basename(d).rsplit("-", 2)[0]
        stamp = os.path.basename(d)[len(host) + 1:]
        return summarize.build_summary(caps, results, host, stamp)
    except Exception:
        return None


def _from_report(path: str) -> dict | None:
    try:
        return summarize.parse_embedded(open(path, encoding="utf-8").read())
    except Exception:
        return None


def collect(extra_reports: list[str]) -> list[dict]:
    summaries: dict[str, dict] = {}   # host -> latest summary

    def _run_tier(s: dict) -> str:
        for k in ("e1", "e2"):
            if s.get(k) and s[k].get("tier"):
                return s[k]["tier"]
        return "?"

    def consider(s: dict | None):
        if not s or not s.get("host"):
            return
        # key by host+tier so a machine can contribute BOTH a capacity run
        # (the RAM-fit feat) and an edge run (same-tier throughput comparison)
        key = f"{s['host']}:{_run_tier(s)}"
        if key not in summaries or str(s.get("stamp", "")) > str(summaries[key].get("stamp", "")):
            summaries[key] = s

    if os.path.isdir(RESULTS):
        for name in os.listdir(RESULTS):
            d = os.path.join(RESULTS, name)
            if os.path.isdir(d) and name != "_comparison":
                consider(_from_results_dir(d))
    for r in extra_reports:
        consider(_from_report(r))
    return list(summaries.values())


def _cell(v):
    return "—" if v is None else v


def make_table(rows: list[dict]) -> str:
    L = ["# Cross-Host Comparison — Capacity-Centric Clinical LLM", "",
         f"Hosts compared: **{len(rows)}**", ""]
    # transpose: metrics as rows, host+tier as columns
    def _tier(r):
        return (r.get("e1") or r.get("e2") or {}).get("tier", "?")
    headers = ["Metric"] + [f"{r['host']} ({r.get('cpu_vendor','?')}, {_tier(r)})" for r in rows]
    L.append("| " + " | ".join(headers) + " |")
    L.append("|" + "|".join(["---"] * len(headers)) + "|")

    def line(label, fn):
        L.append("| " + " | ".join([label] + [str(_cell(fn(r))) for r in rows]) + " |")

    line("CPU", lambda r: r.get("cpu_model"))
    line("VNNI / AMX", lambda r: f"{r.get('has_vnni')}/{r.get('has_amx')}")
    line("Accel path", lambda r: r.get("accel_path"))
    line("RAM total (GB)", lambda r: r.get("ram_total_gb"))
    line("E1 model tier", lambda r: (r.get("e1") or {}).get("tier"))
    line("E1 weights (GB)", lambda r: (r.get("e1") or {}).get("weights_gb"))
    line("E1 peak RSS (GB)", lambda r: (r.get("e1") or {}).get("peak_rss_gb"))
    line("E1 RAM util (%)", lambda r: (r.get("e1") or {}).get("ram_util_pct"))
    line("E1 GPU verdict", lambda r: (r.get("e1") or {}).get("gpu_verdict"))
    line("E2 tier", lambda r: (r.get("e2") or {}).get("tier"))
    line("E2 TTFT mean (s)", lambda r: (r.get("e2") or {}).get("ttft_mean_s"))
    line("E2 tok/s mean", lambda r: (r.get("e2") or {}).get("tokens_per_s_mean"))
    line("E3 CPU power (W)", lambda r: (r.get("e3") or {}).get("cpu_avg_w"))
    line("E3 power method", lambda r: (r.get("e3") or {}).get("power_method"))
    line("E3 tokens/joule", lambda r: (r.get("e3") or {}).get("tokens_per_joule_cpu"))
    line("E4 adapter r16 (MB)", lambda r: (r.get("e4") or {}).get("adapter_mb_r16"))
    line("E4 reduction (%)", lambda r: (r.get("e4") or {}).get("reduction_pct_r16"))
    L.append("")
    L.append("_Note: E2/E3 are most comparable when hosts ran the same model tier. "
             "A generic-AVX2 host is the conservative baseline; an Intel host with "
             "VNNI/AMX is the accelerated condition._")
    L.append("")
    return "\n".join(L)


def make_figures(rows: list[dict]) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    written = []
    def _tier(r):
        return (r.get("e1") or r.get("e2") or {}).get("tier", "?")
    labels = [f"{r['host']}\n{r.get('cpu_vendor','?')}/{_tier(r)}" for r in rows]

    # Throughput + TTFT (only hosts with E2)
    e2 = [(l, r) for l, r in zip(labels, rows) if r.get("e2")]
    if e2:
        ls = [l for l, _ in e2]
        tps = [r["e2"]["tokens_per_s_mean"] or 0 for _, r in e2]
        ttft = [r["e2"]["ttft_mean_s"] or 0 for _, r in e2]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
        ax1.bar(ls, tps, color="#4C78A8"); ax1.set_title("Inference throughput (tok/s)")
        ax1.set_ylabel("tokens / s")
        for i, v in enumerate(tps):
            ax1.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
        ax2.bar(ls, ttft, color="#F58518"); ax2.set_title("Time-to-first-token (s)")
        ax2.set_ylabel("seconds")
        for i, v in enumerate(ttft):
            ax2.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
        fig.tight_layout()
        p = os.path.join(OUTDIR, "compare_throughput.png"); fig.savefig(p, dpi=150)
        plt.close(fig); written.append(p)

    # CI/W
    e3 = [(l, r) for l, r in zip(labels, rows) if r.get("e3") and r["e3"].get("tokens_per_joule_cpu")]
    if e3:
        ls = [l for l, _ in e3]
        tj = [r["e3"]["tokens_per_joule_cpu"] for _, r in e3]
        fig, ax = plt.subplots(figsize=(6, 4.2))
        ax.bar(ls, tj, color="#54A24B"); ax.set_title("Clinical-Intelligence-per-Watt (tokens/joule)")
        ax.set_ylabel("tokens / joule")
        for i, v in enumerate(tj):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        fig.tight_layout()
        p = os.path.join(OUTDIR, "compare_ciw.png"); fig.savefig(p, dpi=150)
        plt.close(fig); written.append(p)
    return written


def main() -> int:
    rows = collect(sys.argv[1:])
    if not rows:
        print("No host results found. Run run_experiment.py first, or pass a "
              "final_report.md path to ingest.")
        return 1
    os.makedirs(OUTDIR, exist_ok=True)
    table_path = os.path.join(OUTDIR, "comparison_table.md")
    with open(table_path, "w") as f:
        f.write(make_table(rows))
    written = [table_path] + make_figures(rows)
    def _tier(r):
        return (r.get("e1") or r.get("e2") or {}).get("tier", "?")
    print(f"Compared {len(rows)} run(s): {', '.join(r['host'] + '/' + _tier(r) for r in rows)}")
    for p in written:
        print("  wrote", os.path.relpath(p, HERE))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

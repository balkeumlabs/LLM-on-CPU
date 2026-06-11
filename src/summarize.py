"""
Compact cross-host summary shared by the report writer and compare.py.

build_summary() distills capabilities + E1-E4 results into a small flat dict.
It is embedded as a fenced JSON block inside final_report.md (between the
markers below) so the single self-contained report a colleague sends back is
machine-parseable, AND it is what compare.py reads from each host.
"""
from __future__ import annotations

import json
import re

SUMMARY_BEGIN = "<!-- ICCE26-SUMMARY-BEGIN -->"
SUMMARY_END = "<!-- ICCE26-SUMMARY-END -->"


def _find(results: list[dict], key: str) -> dict | None:
    for r in results:
        if r.get("key") == key:
            return r
    return None


def build_summary(caps: dict, results: list[dict], host: str, stamp: str) -> dict:
    cpu = caps.get("cpu", {})
    e1, e2, e3, e4 = (_find(results, k) for k in ("E1", "E2", "E3", "E4"))

    def ok(r):
        return r and r.get("status") == "ok"

    gpu_verdict = None
    if ok(e1) and e1.get("gpu_oom_analysis"):
        gpu_verdict = e1["gpu_oom_analysis"][0].get("verdict", "").split(" ")[0]

    return {
        "host": host,
        "stamp": stamp,
        "os": caps.get("os", {}).get("system"),
        "cpu_model": cpu.get("model"),
        "cpu_vendor": cpu.get("vendor"),
        "has_vnni": cpu.get("has_vnni"),
        "has_amx": cpu.get("has_amx"),
        "has_avx512": cpu.get("has_avx512"),
        "accel_path": caps.get("decisions", {}).get("accel_path"),
        "ram_total_gb": caps.get("memory", {}).get("total_gb"),
        "power_method": caps.get("power", {}).get("method"),
        "e1": {
            "tier": e1.get("model_tier"),
            "weights_gb": e1["ram"]["weights_file_gb"],
            "peak_rss_gb": e1["ram"]["peak_rss_gb"],
            "ram_util_pct": e1["ram"]["ram_utilization_pct"],
            "load_time_s": e1["ram"]["load_time_s"],
            "gpu_verdict": gpu_verdict,
        } if ok(e1) else None,
        "e2": {
            "tier": e2.get("tier"),
            "ttft_mean_s": e2.get("ttft_mean_s"),
            "tokens_per_s_mean": e2.get("tokens_per_s_mean"),
            "n_threads": e2.get("n_threads"),
        } if ok(e2) else None,
        "e3": {
            "cpu_avg_w": e3.get("cpu_avg_w"),
            "power_method": e3.get("power_method_cpu"),
            "tokens_per_joule_cpu": e3.get("ci_per_watt", {}).get("tokens_per_joule_cpu"),
        } if ok(e3) else None,
        "e4": {
            "adapter_mb_r16": next((row["adapter_mb_bf16"] for row in e4.get("rows", [])
                                    if row["rank"] == 16), None),
            "reduction_pct_r16": next((row["bandwidth_reduction_vs_bf16_pct"]
                                       for row in e4.get("rows", []) if row["rank"] == 16), None),
        } if ok(e4) else None,
    }


def embed_block(summary: dict) -> str:
    return f"{SUMMARY_BEGIN}\n```json\n{json.dumps(summary, indent=2)}\n```\n{SUMMARY_END}"


def parse_embedded(report_text: str) -> dict | None:
    m = re.search(re.escape(SUMMARY_BEGIN) + r"\s*```json\s*(\{.*?\})\s*```",
                  report_text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except Exception:
        return None

#!/usr/bin/env python3
"""
ICCE26 capacity-demo harness — single master entry point.

Runs on BOTH the AMD/NVIDIA dev box and a light Intel laptop (any OS) with no
hard dependencies. It:
  1. Detects hardware + capabilities and prints a report.
  2. Picks an acceleration path (generic llama.cpp vs Intel IPEX/OpenVINO) and
     a model tier (27B capacity vs 2B edge) based on available RAM.
  3. Runs experiments E1-E4 (stubs in M1; filled in M3/M4).
  4. Writes per-host results + a short, self-contained final_report.md to send back.

Usage:
    python run_experiment.py                 # detect + run everything that's ready
    python run_experiment.py --report-only   # just print the capability report
    python run_experiment.py --model gemma-2-2b-it
    python run_experiment.py --cpu-only      # ignore GPU even if present
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import detect  # noqa: E402

RESULTS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# Experiment registry. Each is (key, title, callable). M1 ships stubs that
# report "pending" so the full pipeline + report writer are exercised today.
try:
    from experiments import e1_capacity, e2_inference, e3_energy, e4_comms  # noqa
    _EXPERIMENTS_AVAILABLE = True
except Exception:
    _EXPERIMENTS_AVAILABLE = False


# --------------------------------------------------------------------------
# Pretty capability report
# --------------------------------------------------------------------------
def _yn(b) -> str:
    return "yes" if b else "no"


def _gpu_name(g: dict) -> str:
    """Vendor + name without doubling the vendor when name already includes it."""
    name = g.get("name", "")
    vendor = g.get("vendor", "")
    return name if vendor.lower() in name.lower() else f"{vendor} {name}".strip()


def print_capability_report(caps: dict) -> None:
    os_i, cpu, mem = caps["os"], caps["cpu"], caps["memory"]
    gpu, power, be = caps["gpu"], caps["power"], caps["backends"]
    dec = caps["decisions"]
    line = "=" * 64
    print(line)
    print(" ICCE26 Capacity-Demo Harness  —  Capability Report")
    print(line)
    print(f" Host        : {socket.gethostname()}")
    print(f" OS          : {os_i['system']} {os_i['release']} ({os_i['machine']}), Python {os_i['python']}")
    print(f" CPU         : {cpu['model']}")
    print(f"   vendor    : {cpu['vendor']}   cores: {cpu['physical_cores']}P / {cpu['logical_cores']}T")
    print(f"   ISA       : avx2={_yn(cpu['has_avx2'])}  avx512={_yn(cpu['has_avx512'])}  "
          f"VNNI={_yn(cpu['has_vnni'])}  AMX={_yn(cpu['has_amx'])}")
    print(f" RAM         : total {mem['total_gb']} GB,  available {mem['available_gb']} GB")
    if gpu["gpus"]:
        for g in gpu["gpus"]:
            vram = f"{g.get('memory_mib')} MiB" if g.get("memory_mib") else "n/a"
            pwr = f"{g.get('power_draw_w')}/{g.get('power_limit_w')} W" if g.get("power_limit_w") else "n/a"
            print(f" GPU         : {_gpu_name(g)}  (VRAM {vram}, power {pwr})")
    else:
        print(" GPU         : none detected")
    print(f" Power telem : method={power['method']}  rapl_cpu={_yn(power['rapl_cpu'])}  "
          f"nvidia_smi={_yn(power['nvidia_smi_power'])}")
    print(f" Backends    : llama.cpp={_yn(be['llama_cpp'])}  ipex-llm={_yn(be['ipex_llm'])}  "
          f"ipex={_yn(be['ipex'])}  openvino={_yn(be['openvino'])}  torch={_yn(be['torch'])}")
    print(line)
    tier = dec["model_tier"]
    print(f" -> accel path : {dec['accel_path'].upper()}")
    print(f" -> model tier : {tier['tier']}  ({tier['name']}, {tier['quant']}, ~{tier['approx_ram_gb']} GB)")
    print(f"                 {tier['selected_because']}")
    print(line)
    if dec["accel_path"] == "none":
        print(" NOTE: no inference backend installed yet. Capability report works;")
        print("       E1-E4 will be stubbed until the model stack is set up (next milestone).")
        print(line)


# --------------------------------------------------------------------------
# Experiment runner
# --------------------------------------------------------------------------
def run_experiments(caps: dict, args) -> list[dict]:
    """Run E1-E4. Each returns a dict with at least {key, title, status}."""
    results = []

    def _stub(key, title, why):
        return {"key": key, "title": title, "status": "pending", "detail": why}

    if _EXPERIMENTS_AVAILABLE and not args.report_only:
        for key, title, mod in [
            ("E1", "Capacity over TFLOPS (RAM-fit vs GPU-OOM)", e1_capacity),
            ("E2", "Inference throughput & TTFT", e2_inference),
            ("E3", "Energy / Clinical-Intelligence-per-Watt", e3_energy),
            ("E4", "Communication efficiency (LoRA adapter size)", e4_comms),
        ]:
            try:
                results.append(mod.run(caps, args))
            except NotImplementedError as e:
                results.append(_stub(key, title, str(e) or "not implemented yet"))
            except Exception as e:
                results.append({"key": key, "title": title, "status": "error", "detail": repr(e)})
    else:
        reason = "report-only mode" if args.report_only else "experiment modules not present yet (M1 skeleton)"
        for key, title in [
            ("E1", "Capacity over TFLOPS (RAM-fit vs GPU-OOM)"),
            ("E2", "Inference throughput & TTFT"),
            ("E3", "Energy / Clinical-Intelligence-per-Watt"),
            ("E4", "Communication efficiency (LoRA adapter size)"),
        ]:
            results.append(_stub(key, title, reason))
    return results


# --------------------------------------------------------------------------
# Output: results dir + machine-readable JSON + short final report
# --------------------------------------------------------------------------
def write_outputs(caps: dict, results: list[dict], stamp: str) -> str:
    host = socket.gethostname()
    outdir = os.path.join(RESULTS_ROOT, f"{host}-{stamp}")
    os.makedirs(outdir, exist_ok=True)

    with open(os.path.join(outdir, "capabilities.json"), "w") as f:
        json.dump(caps, f, indent=2)
    with open(os.path.join(outdir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    report = _build_final_report(caps, results, host, stamp)
    report_path = os.path.join(outdir, "final_report.md")
    with open(report_path, "w") as f:
        f.write(report)
    return report_path


def _build_final_report(caps: dict, results: list[dict], host: str, stamp: str) -> str:
    cpu, mem, dec = caps["cpu"], caps["memory"], caps["decisions"]
    gpu = caps["gpu"]
    gpu_str = ", ".join(
        _gpu_name(g) + (f" {int(g['memory_mib'])}MiB" if g.get("memory_mib") else "")
        for g in gpu["gpus"]
    ) or "none"
    L = []
    L.append(f"# ICCE26 Capacity-Demo — Final Report")
    L.append("")
    L.append(f"- **Host:** {host}")
    L.append(f"- **Generated:** {stamp} (UTC)")
    L.append(f"- **OS:** {caps['os']['system']} {caps['os']['release']} ({caps['os']['machine']})")
    L.append(f"- **CPU:** {cpu['model']} — {cpu['vendor']}, "
             f"{cpu['physical_cores']}P/{cpu['logical_cores']}T")
    L.append(f"- **ISA:** avx2={cpu['has_avx2']}, avx512={cpu['has_avx512']}, "
             f"VNNI={cpu['has_vnni']}, AMX={cpu['has_amx']}")
    L.append(f"- **RAM:** {mem['total_gb']} GB total / {mem['available_gb']} GB available")
    L.append(f"- **GPU:** {gpu_str}")
    L.append(f"- **Accel path:** {dec['accel_path']}  |  **Model tier:** "
             f"{dec['model_tier']['tier']} ({dec['model_tier']['name']})")
    L.append(f"- **Power method:** {caps['power']['method']}")
    L.append("")
    L.append("## Results")
    L.append("")
    L.append("| Exp | Title | Status | Detail |")
    L.append("|-----|-------|--------|--------|")
    for r in results:
        detail = str(r.get("detail", r.get("summary", ""))).replace("|", "\\|")
        if len(detail) > 80:
            detail = detail[:77] + "..."
        L.append(f"| {r['key']} | {r['title']} | {r['status']} | {detail} |")
    L.append("")
    L.append("_Send this file back; it is self-contained. Full data: capabilities.json + results.json in the same folder._")
    L.append("")
    # Machine-readable summary so compare.py can merge this single file.
    try:
        import summarize
        L.append(summarize.embed_block(summarize.build_summary(caps, results, host, stamp)))
        L.append("")
    except Exception:
        pass
    return "\n".join(L)


# --------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description="ICCE26 capacity-demo harness (master entry).")
    p.add_argument("--report-only", action="store_true", help="print capability report and exit (no experiments)")
    p.add_argument("--tier", choices=["auto", "edge", "capacity"], default="auto",
                   help="force a model tier (use --tier edge on BOTH hosts for a fair "
                        "cross-host throughput comparison). Default: auto by RAM.")
    p.add_argument("--cpu-only", action="store_true", help="ignore GPU even if present")
    p.add_argument("--json", action="store_true", help="also dump raw capabilities JSON to stdout")
    p.add_argument("--no-figures", action="store_true", help="skip figure/table generation")
    args = p.parse_args()

    caps = detect.detect_all()
    if args.tier != "auto":
        forced = next((t for t in detect.MODEL_TIERS if t["tier"] == args.tier), None)
        if forced:
            caps["decisions"]["model_tier"] = {**forced, "selected_because": f"--tier {args.tier} override"}
    if args.cpu_only:
        caps["gpu"] = {**caps["gpu"], "gpus": [], "has_nvidia": False, "_cpu_only_override": True}

    print_capability_report(caps)
    if args.json:
        print(json.dumps(caps, indent=2))

    if args.report_only:
        print("\n(report-only) skipping experiments.")
        return 0

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    print("\nRunning experiments...\n")
    results = run_experiments(caps, args)
    for r in results:
        print(f"  [{r['status']:>8}] {r['key']}  {r['title']}")
    report_path = write_outputs(caps, results, stamp)

    if not args.no_figures:
        try:
            import figures
            outdir = os.path.dirname(report_path)
            for p in figures.generate(outdir):
                print(f"  figure/table: {os.path.basename(p)}")
        except Exception as e:
            print(f"  (figure generation skipped: {e})")

    print(f"\nFinal report written to:\n  {report_path}")
    print("Send that final_report.md back for the cross-host comparison.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

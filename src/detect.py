"""
Hardware / capability detection for the ICCE26 capacity-demo harness.

Goal: run on BOTH an AMD/NVIDIA Linux box and a light Intel laptop (any OS),
with ZERO hard dependencies. Everything degrades gracefully:
  - psutil / py-cpuinfo are used if importable, else stdlib fallbacks.
  - Intel accel libs (ipex-llm, OpenVINO) are *probed*, never required.

Produces a single `Capabilities` dict that the master script prints and uses
to choose a backend + model tier.
"""
from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
from typing import Optional

# ---- optional deps (never required) --------------------------------------
try:
    import psutil  # type: ignore
    _HAVE_PSUTIL = True
except Exception:
    _HAVE_PSUTIL = False

try:
    import cpuinfo as _cpuinfo  # py-cpuinfo  # type: ignore
    _HAVE_CPUINFO = True
except Exception:
    _HAVE_CPUINFO = False

# ISA flags we care about for the Intel-vs-generic acceleration story.
_ISA_OF_INTEREST = [
    "avx2", "avx512f", "avx512_vnni", "avx_vnni", "amx_tile", "amx_bf16", "amx_int8",
]


def _run(cmd: list[str], timeout: int = 8) -> Optional[str]:
    """Run a command, return stdout or None on any failure."""
    if not shutil.which(cmd[0]):
        return None
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
        return out.stdout if out.returncode == 0 else None
    except Exception:
        return None


# --------------------------------------------------------------------------
# OS / platform
# --------------------------------------------------------------------------
def detect_os() -> dict:
    return {
        "system": platform.system(),            # Linux / Windows / Darwin
        "release": platform.release(),
        "machine": platform.machine(),          # x86_64 / AMD64 / arm64
        "python": platform.python_version(),
    }


# --------------------------------------------------------------------------
# CPU: vendor, model, ISA flags, core/thread counts
# --------------------------------------------------------------------------
def _cpu_flags_linux() -> set[str]:
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.lower().startswith("flags") or line.lower().startswith("features"):
                    return set(line.split(":", 1)[1].split())
    except Exception:
        pass
    return set()


def _cpu_model_linux() -> tuple[str, str]:
    model, vendor = "", ""
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not model and line.lower().startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                if not vendor and line.lower().startswith("vendor_id"):
                    vendor = line.split(":", 1)[1].strip()
                if model and vendor:
                    break
    except Exception:
        pass
    return model, vendor


def detect_cpu() -> dict:
    sys_name = platform.system()
    model, vendor_id, flags = "", "", set()

    if sys_name == "Linux":
        model, vendor_id = _cpu_model_linux()
        flags = _cpu_flags_linux()

    # py-cpuinfo fills gaps on Windows/macOS (and corroborates on Linux).
    if _HAVE_CPUINFO and (not model or not flags):
        try:
            info = _cpuinfo.get_cpu_info()
            model = model or info.get("brand_raw", "")
            vendor_id = vendor_id or info.get("vendor_id_raw", "")
            flags |= set(info.get("flags", []))
        except Exception:
            pass

    if not model:
        model = platform.processor() or platform.machine() or "unknown"

    # Normalise vendor to a friendly tag.
    vid = (vendor_id or model).lower()
    if "intel" in vid or "genuineintel" in vid:
        vendor = "Intel"
    elif "amd" in vid or "authenticamd" in vid:
        vendor = "AMD"
    else:
        vendor = vendor_id or "unknown"

    isa = {f: (f in flags) for f in _ISA_OF_INTEREST}

    # Logical / physical counts.
    logical = os.cpu_count() or 0
    physical = None
    if _HAVE_PSUTIL:
        try:
            physical = psutil.cpu_count(logical=False)
        except Exception:
            physical = None
    if physical is None and sys_name == "Linux":
        # stdlib fallback: "cpu cores" * number of distinct physical ids.
        try:
            cores_per_socket, sockets = 0, set()
            with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.lower().startswith("cpu cores"):
                        cores_per_socket = int(line.split(":", 1)[1].strip())
                    elif line.lower().startswith("physical id"):
                        sockets.add(line.split(":", 1)[1].strip())
            if cores_per_socket:
                physical = cores_per_socket * max(len(sockets), 1)
        except Exception:
            physical = None

    return {
        "model": model,
        "vendor": vendor,
        "physical_cores": physical,
        "logical_cores": logical,
        "isa": isa,
        # Intel acceleration the paper leans on:
        "has_vnni": isa.get("avx512_vnni", False) or isa.get("avx_vnni", False),
        "has_amx": any(isa.get(k, False) for k in ("amx_tile", "amx_bf16", "amx_int8")),
        "has_avx512": isa.get("avx512f", False),
        "has_avx2": isa.get("avx2", False),
    }


# --------------------------------------------------------------------------
# Memory
# --------------------------------------------------------------------------
def detect_memory() -> dict:
    total_gb = avail_gb = None
    if _HAVE_PSUTIL:
        try:
            vm = psutil.virtual_memory()
            total_gb = round(vm.total / 1e9, 1)
            avail_gb = round(vm.available / 1e9, 1)
        except Exception:
            pass
    if total_gb is None and platform.system() == "Linux":
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                kv = {}
                for line in f:
                    parts = line.split(":")
                    if len(parts) == 2:
                        kv[parts[0].strip()] = parts[1].strip()
            if "MemTotal" in kv:
                total_gb = round(int(kv["MemTotal"].split()[0]) * 1024 / 1e9, 1)
            avail_key = "MemAvailable" if "MemAvailable" in kv else "MemFree"
            if avail_key in kv:
                avail_gb = round(int(kv[avail_key].split()[0]) * 1024 / 1e9, 1)
        except Exception:
            pass
    return {"total_gb": total_gb, "available_gb": avail_gb}


# --------------------------------------------------------------------------
# GPU (NVIDIA via nvidia-smi; Intel best-effort)
# --------------------------------------------------------------------------
def detect_gpu() -> dict:
    gpus = []
    out = _run([
        "nvidia-smi",
        "--query-gpu=name,memory.total,power.draw,power.limit",
        "--format=csv,noheader,nounits",
    ])
    if out:
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                def _num(x):
                    try:
                        return float(x)
                    except Exception:
                        return None
                gpus.append({
                    "vendor": "NVIDIA",
                    "name": parts[0],
                    "memory_mib": _num(parts[1]),
                    "power_draw_w": _num(parts[2]) if len(parts) > 2 else None,
                    "power_limit_w": _num(parts[3]) if len(parts) > 3 else None,
                })

    # Intel GPU/NPU best-effort hint (Linux lspci); informational only.
    intel_gpu_hint = False
    if platform.system() == "Linux":
        lspci = _run(["lspci"])
        if lspci:
            for line in lspci.splitlines():
                low = line.lower()
                if ("vga" in low or "display" in low or "3d" in low) and "intel" in low:
                    intel_gpu_hint = True
                    gpus.append({"vendor": "Intel", "name": line.split(":")[-1].strip()})

    return {"gpus": gpus, "has_nvidia": any(g["vendor"] == "NVIDIA" for g in gpus),
            "intel_gpu_hint": intel_gpu_hint}


# --------------------------------------------------------------------------
# Power telemetry availability (for the CI/W metric, estimate-grade)
# --------------------------------------------------------------------------
def detect_power_sources() -> dict:
    # RAPL is only useful if energy_uj is actually READABLE (often root-only
    # since the platypus/side-channel mitigations) — test a real read.
    rapl_readable = False
    rapl_present = False
    if platform.system() == "Linux" and os.path.isdir("/sys/class/powercap"):
        try:
            for entry in os.listdir("/sys/class/powercap"):
                base = f"/sys/class/powercap/{entry}"
                name_path = f"{base}/name"
                energy_path = f"{base}/energy_uj"
                if os.path.exists(name_path) and os.path.exists(energy_path):
                    try:
                        with open(name_path) as f:
                            nm = f.read().strip().lower()
                    except Exception:
                        continue
                    if "package" in nm or "psys" in nm:
                        rapl_present = True
                        try:
                            with open(energy_path) as f:
                                int(f.read().strip())
                            rapl_readable = True
                        except Exception:
                            pass
        except Exception:
            pass
    return {
        "rapl_present": rapl_present,
        "rapl_cpu": rapl_readable,                       # readable == usable
        "nvidia_smi_power": shutil.which("nvidia-smi") is not None,
        # If RAPL unreadable: fall back to TDP-based estimate, logged transparently.
        "method": "rapl" if rapl_readable else "tdp_estimate",
    }


# --------------------------------------------------------------------------
# Backend availability (probe, never require)
# --------------------------------------------------------------------------
def _importable(mod: str) -> bool:
    import importlib.util
    try:
        return importlib.util.find_spec(mod) is not None
    except Exception:
        return False


def detect_backends() -> dict:
    return {
        "llama_cpp": _importable("llama_cpp"),                 # generic CPU/CUDA engine
        "ipex_llm": _importable("ipex_llm"),                   # Intel LLM accel
        "ipex": _importable("intel_extension_for_pytorch"),    # Intel PyTorch ext
        "openvino": _importable("openvino"),                   # Intel inference/NPU
        "torch": _importable("torch"),
    }


# --------------------------------------------------------------------------
# Decisions: which accel path + model tier
# --------------------------------------------------------------------------
# Model tiers (open Gemma-2 stand-ins for MedGemma). RAM need is for the
# INT4 (Q4_K_M) weights plus a safety headroom for KV-cache / OS.
MODEL_TIERS = [
    {"tier": "capacity", "name": "gemma-2-27b-it", "quant": "Q4_K_M",
     "approx_ram_gb": 16.0, "min_ram_gb": 24.0,
     "note": "stand-in for MedGemma-27B (headline capacity demo)"},
    {"tier": "edge", "name": "gemma-2-2b-it", "quant": "Q4_K_M",
     "approx_ram_gb": 2.0, "min_ram_gb": 6.0,
     "note": "stand-in for MedGemma-4B (light-laptop fallback / cross-host common tier)"},
]


def choose_accel_path(cpu: dict, backends: dict) -> str:
    if cpu["vendor"] == "Intel" and (backends["ipex_llm"] or backends["ipex"] or backends["openvino"]):
        return "intel"          # exercise DL Boost / AMX / OpenVINO
    if backends["llama_cpp"]:
        return "generic"        # llama.cpp on any x86 CPU (+ CUDA if present)
    return "none"               # nothing installed yet (expected before M-setup)


def choose_model_tier(mem: dict) -> dict:
    avail = mem.get("available_gb") or mem.get("total_gb") or 0
    for tier in MODEL_TIERS:               # largest-first
        if avail >= tier["min_ram_gb"]:
            return {**tier, "selected_because": f"{avail} GB available >= {tier['min_ram_gb']} GB"}
    smallest = MODEL_TIERS[-1]
    return {**smallest, "selected_because": f"only {avail} GB available; smallest tier (may be tight)"}


# --------------------------------------------------------------------------
# Top-level
# --------------------------------------------------------------------------
def detect_all() -> dict:
    os_info = detect_os()
    cpu = detect_cpu()
    mem = detect_memory()
    gpu = detect_gpu()
    power = detect_power_sources()
    backends = detect_backends()
    caps = {
        "os": os_info,
        "cpu": cpu,
        "memory": mem,
        "gpu": gpu,
        "power": power,
        "backends": backends,
        "optional_deps": {"psutil": _HAVE_PSUTIL, "py_cpuinfo": _HAVE_CPUINFO},
    }
    caps["decisions"] = {
        "accel_path": choose_accel_path(cpu, backends),
        "model_tier": choose_model_tier(mem),
    }
    return caps


if __name__ == "__main__":
    import json
    print(json.dumps(detect_all(), indent=2))

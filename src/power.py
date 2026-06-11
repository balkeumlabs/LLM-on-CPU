"""
Power / energy telemetry for the CI/W metric (estimate-grade, per project decision).

Two sources, used opportunistically:
  - CPU: Linux RAPL energy_uj deltas when READABLE (root-only on many hosts);
    otherwise a transparent TDP-based estimate (logged as method='tdp_estimate').
  - GPU: nvidia-smi instantaneous power.draw, sampled in a background thread.

PowerSampler is a context manager:

    with PowerSampler(caps) as ps:
        ... run the workload ...
    energy = ps.result()   # dict: joules, avg/peak watts, method, per-source

For CPU-only inference the GPU sits near idle; we still record it so the CI/W
denominator can be reported as CPU-only or CPU+GPU.
"""
from __future__ import annotations

import glob
import os
import shutil
import subprocess
import threading
import time
from typing import Optional

# Rough package TDPs (watts) used only when RAPL is unreadable. Conservative
# "under sustained load" figures; the exact value is reported alongside results.
_TDP_TABLE = {
    "amd ryzen 5 5500gt": 65.0,
    "default_amd": 65.0,
    "default_intel_laptop": 28.0,   # typical Core Ultra / U-series base
    "default": 45.0,
}


def estimate_cpu_tdp(cpu: dict) -> tuple[float, str]:
    model = (cpu.get("model") or "").lower()
    for key, val in _TDP_TABLE.items():
        if key.startswith("default"):
            continue
        if key in model:
            return val, f"tdp_table:{key}"
    if cpu.get("vendor") == "AMD":
        return _TDP_TABLE["default_amd"], "tdp_default_amd"
    if cpu.get("vendor") == "Intel":
        return _TDP_TABLE["default_intel_laptop"], "tdp_default_intel_laptop"
    return _TDP_TABLE["default"], "tdp_default"


def _rapl_energy_uj() -> Optional[int]:
    """Sum readable package/psys energy counters (microjoules). None if unreadable."""
    total, found = 0, False
    for name_path in glob.glob("/sys/class/powercap/*/name"):
        base = os.path.dirname(name_path)
        try:
            with open(name_path) as f:
                nm = f.read().strip().lower()
            if "package" not in nm and "psys" not in nm:
                continue
            with open(os.path.join(base, "energy_uj")) as f:
                total += int(f.read().strip())
                found = True
        except Exception:
            continue
    return total if found else None


def _gpu_power_w() -> Optional[float]:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=4, check=False,
        )
        if out.returncode == 0:
            vals = [float(x) for x in out.stdout.split() if x.strip().replace(".", "", 1).isdigit()]
            return sum(vals) if vals else None
    except Exception:
        pass
    return None


class PowerSampler:
    def __init__(self, caps: dict, sample_hz: float = 2.0):
        self.caps = caps
        self.cpu = caps.get("cpu", {})
        self.interval = 1.0 / sample_hz
        self._gpu_samples: list[float] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._rapl_start: Optional[int] = None
        self._t0 = 0.0
        self.elapsed_s = 0.0
        self._tdp_w, self._tdp_src = estimate_cpu_tdp(self.cpu)

    def _poll_gpu(self):
        while not self._stop.is_set():
            p = _gpu_power_w()
            if p is not None:
                self._gpu_samples.append(p)
            self._stop.wait(self.interval)

    def __enter__(self):
        self._t0 = time.perf_counter()
        self._rapl_start = _rapl_energy_uj()
        self._thread = threading.Thread(target=self._poll_gpu, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self.elapsed_s = time.perf_counter() - self._t0
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        rapl_end = _rapl_energy_uj()
        self._rapl_end = rapl_end
        return False

    def result(self) -> dict:
        elapsed = max(self.elapsed_s, 1e-6)
        # CPU energy
        if self._rapl_start is not None and getattr(self, "_rapl_end", None) is not None:
            cpu_joules = (self._rapl_end - self._rapl_start) / 1e6
            cpu_avg_w = cpu_joules / elapsed
            cpu_method = "rapl"
        else:
            cpu_avg_w = self._tdp_w
            cpu_joules = cpu_avg_w * elapsed
            cpu_method = self._tdp_src
        # GPU energy (sampled average * time)
        if self._gpu_samples:
            gpu_avg_w = sum(self._gpu_samples) / len(self._gpu_samples)
            gpu_peak_w = max(self._gpu_samples)
            gpu_joules = gpu_avg_w * elapsed
        else:
            gpu_avg_w = gpu_peak_w = gpu_joules = None

        return {
            "elapsed_s": round(elapsed, 3),
            "cpu": {"method": cpu_method, "avg_w": round(cpu_avg_w, 2),
                    "joules": round(cpu_joules, 2)},
            "gpu": ({"method": "nvidia-smi", "avg_w": round(gpu_avg_w, 2),
                     "peak_w": round(gpu_peak_w, 2), "joules": round(gpu_joules, 2)}
                    if gpu_avg_w is not None else {"method": "unavailable"}),
        }

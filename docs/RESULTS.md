# Reference Results

Measured on the development machine (the conservative, generic-CPU baseline):

> **AMD Ryzen 5 5500GT** (6C/12T, **AVX2 only** — no VNNI/AMX) · 67 GB RAM ·
> NVIDIA GTX 1060 **6 GB** · Ubuntu 24.04 · llama.cpp INT4 (Q4_K_M) · CPU-only.

Model: **Gemma-2-27B-it** (open stand-in for MedGemma-27B; identical architecture
family, so the memory footprint is representative).

## E1 — Capacity over TFLOPS (headline)

![Capacity over TFLOPS](capacity_ram_vs_vram.png)

| Quantity | Value |
|---|---|
| INT4 weights (file) | **16.65 GB** |
| Peak resident memory (RSS) | **24.98 GB** |
| System RAM (total) | 67.2 GB |
| RAM utilization | **37.2 %** |
| Load time | 15.7 s |
| GTX 1060 6 GB verdict | **OOM** — weights (16.65 GB) exceed VRAM (6.4 GB) before the first step |

The 27B INT4 state loads into system RAM with large headroom, while the consumer
GPU cannot hold it at all. This is the core "capacity-enabled, not compute-limited"
claim, measured on a single machine.

## E2 — Inference throughput & TTFT

| Metric | 27B (capacity) | 2B (edge) |
|---|---|---|
| TTFT (mean) | 18.4 s | ~2.0 s |
| Throughput | 2.36 tok/s | ~19 tok/s |
| Threads | 6 | 6 |

These are the **conservative generic-CPU floor** (no Intel VNNI/AMX). An Intel
host running the same model via IPEX-LLM/OpenVINO is expected to be substantially
faster — that delta is the intended Intel-acceleration result.

## E3 — Energy / Clinical-Intelligence-per-Watt

| Metric | Value |
|---|---|
| CPU power | 65 W (TDP estimate*) |
| Tokens / joule (27B) | 0.02 |
| GPU power (idle, unused) | ~19 W |

\* RAPL energy counters are root-only on this host, so power is an estimate-grade
TDP figure (labeled as such in every output).

## E4 — Communication efficiency (analytical)

| LoRA rank (q_proj, v_proj) | Adapter (BF16) | Reduction vs full BF16 |
|---|---|---|
| r = 8 | 11.3 MB | 99.98 % |
| r = 16 | 22.6 MB | 99.96 % |

Federated nodes exchange only the LoRA adapter, not the full model — a >99.8 %
bandwidth reduction, computed from the real Gemma-2-27B architecture.

---

*Raw artifacts for these runs are produced under `results/<host>-<timestamp>/`
(git-ignored). Regenerate with the steps in the main [README](../README.md).*

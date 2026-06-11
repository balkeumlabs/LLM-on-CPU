"""
E4 - Communication efficiency (analytical, no model download required).

Quantifies the proposal's bandwidth claim: federated nodes share only the LoRA
adapters (q_proj, v_proj) instead of full parameter sets. We compute exact
adapter parameter counts from the Gemma-2 architecture and compare against the
full model, for LoRA ranks 8 and 16.

LoRA params per target module = rank * (d_in + d_out)  (the A and B factors).
Adapter is shared in BF16 (2 bytes/param), per the INT4-base/BF16-LoRA scheme.
"""
from __future__ import annotations

# Gemma-2 architecture constants (public model configs).
GEMMA2 = {
    "edge": {       # gemma-2-2b
        "name": "gemma-2-2b-it", "hidden": 2304, "layers": 26,
        "n_heads": 8, "head_dim": 256, "n_kv_heads": 4, "params_total": 2.61e9,
    },
    "capacity": {   # gemma-2-27b
        "name": "gemma-2-27b-it", "hidden": 4608, "layers": 46,
        "n_heads": 32, "head_dim": 128, "n_kv_heads": 16, "params_total": 27.2e9,
    },
}
BYTES_BF16 = 2
BYTES_INT4 = 0.5  # effective bytes/param for the 4-bit base file


def _adapter_params(cfg: dict, rank: int) -> int:
    d = cfg["hidden"]
    q_out = cfg["n_heads"] * cfg["head_dim"]      # q_proj output dim
    v_out = cfg["n_kv_heads"] * cfg["head_dim"]   # v_proj output dim (GQA)
    per_layer = rank * (d + q_out) + rank * (d + v_out)   # q_proj + v_proj
    return per_layer * cfg["layers"]


def run(caps: dict, args) -> dict:
    # Report for whichever tier is selected, but always include the 27B headline.
    tier = caps["decisions"]["model_tier"]["tier"]
    tier = tier if tier in GEMMA2 else "capacity"
    cfg = GEMMA2[tier]

    rows = []
    for rank in (8, 16):
        ap = _adapter_params(cfg, rank)
        adapter_mb = ap * BYTES_BF16 / 1e6
        full_bf16_gb = cfg["params_total"] * BYTES_BF16 / 1e9
        full_int4_gb = cfg["params_total"] * BYTES_INT4 / 1e9
        red_vs_bf16 = 100 * (1 - (adapter_mb / 1e3) / full_bf16_gb)
        red_vs_int4 = 100 * (1 - (adapter_mb / 1e3) / full_int4_gb)
        rows.append({
            "rank": rank,
            "adapter_params": ap,
            "adapter_mb_bf16": round(adapter_mb, 2),
            "full_model_bf16_gb": round(full_bf16_gb, 1),
            "full_model_int4_gb": round(full_int4_gb, 1),
            "bandwidth_reduction_vs_bf16_pct": round(red_vs_bf16, 3),
            "bandwidth_reduction_vs_int4_pct": round(red_vs_int4, 3),
        })

    r16 = next(r for r in rows if r["rank"] == 16)
    summary = (f"{cfg['name']}: LoRA(q,v) adapter r=16 is {r16['adapter_mb_bf16']} MB "
               f"vs {r16['full_model_bf16_gb']} GB full BF16 -> "
               f"{r16['bandwidth_reduction_vs_bf16_pct']}% bandwidth reduction")
    return {
        "key": "E4",
        "title": "Communication efficiency (LoRA adapter size)",
        "status": "ok",
        "summary": summary,
        "model": cfg["name"],
        "target_modules": ["q_proj", "v_proj"],
        "precision": "INT4 base / BF16 LoRA",
        "rows": rows,
    }

#!/usr/bin/env python3
"""
Model fetcher (M3) — downloads ungated community GGUF files (no HF token).

Open Gemma-2 stand-ins for MedGemma. Files come from community GGUF repos that
do NOT require license acceptance, so downloads work unauthenticated.

  python src/fetch_models.py --tier edge       # ~1.7 GB, validates pipeline
  python src/fetch_models.py --tier capacity    # ~16 GB, headline 27B
  python src/fetch_models.py --list

Records every fetched file (repo, filename, bytes, sha256) into models.lock for
reproducibility. The harness reads models.lock to locate weights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(HERE, "models")
LOCK_PATH = os.path.join(HERE, "models.lock")

# Ungated community GGUF repos. Q4_K_M = the INT4 quant the paper specifies.
REGISTRY = {
    "edge": {
        "name": "gemma-2-2b-it",
        "repo": "bartowski/gemma-2-2b-it-GGUF",
        "filename": "gemma-2-2b-it-Q4_K_M.gguf",
        "note": "stand-in for MedGemma-4B (light fallback / cross-host common tier)",
    },
    "capacity": {
        "name": "gemma-2-27b-it",
        "repo": "bartowski/gemma-2-27b-it-GGUF",
        "filename": "gemma-2-27b-it-Q4_K_M.gguf",
        "note": "stand-in for MedGemma-27B (headline capacity demo)",
    },
}


def _sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _load_lock() -> dict:
    if os.path.exists(LOCK_PATH):
        try:
            return json.load(open(LOCK_PATH))
        except Exception:
            pass
    return {"models": {}}


def fetch(tier: str, compute_sha: bool = True) -> dict:
    if tier not in REGISTRY:
        raise SystemExit(f"unknown tier '{tier}'. choices: {list(REGISTRY)}")
    spec = REGISTRY[tier]
    os.makedirs(MODELS_DIR, exist_ok=True)

    from huggingface_hub import hf_hub_download
    print(f"[{tier}] downloading {spec['repo']}/{spec['filename']} ...")
    path = hf_hub_download(
        repo_id=spec["repo"],
        filename=spec["filename"],
        local_dir=MODELS_DIR,
        local_dir_use_symlinks=False,
    )
    size = os.path.getsize(path)
    print(f"[{tier}] downloaded {size/1e9:.2f} GB -> {path}")

    sha = _sha256(path) if compute_sha else None
    lock = _load_lock()
    lock["models"][tier] = {
        "name": spec["name"], "repo": spec["repo"], "filename": spec["filename"],
        "path": os.path.relpath(path, HERE), "bytes": size, "sha256": sha,
        "note": spec["note"],
    }
    json.dump(lock, open(LOCK_PATH, "w"), indent=2)
    print(f"[{tier}] recorded in models.lock")
    return lock["models"][tier]


def resolve_path(tier: str) -> str | None:
    """Return the local path for a tier if present in models.lock, else None."""
    lock = _load_lock()
    entry = lock.get("models", {}).get(tier)
    if not entry:
        return None
    p = os.path.join(HERE, entry["path"])
    return p if os.path.exists(p) else None


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch ungated community GGUF models.")
    ap.add_argument("--tier", choices=list(REGISTRY) + ["all"], help="which model to fetch")
    ap.add_argument("--list", action="store_true", help="list known models + lock status")
    ap.add_argument("--no-sha", action="store_true", help="skip sha256 (faster for the 16 GB file)")
    args = ap.parse_args()

    if args.list or not args.tier:
        lock = _load_lock()
        print("Known models:")
        for tier, spec in REGISTRY.items():
            got = resolve_path(tier)
            print(f"  {tier:9s} {spec['name']:16s} {spec['repo']}/{spec['filename']}")
            print(f"            local: {got or '(not downloaded)'}")
        return 0

    tiers = list(REGISTRY) if args.tier == "all" else [args.tier]
    for t in tiers:
        fetch(t, compute_sha=not args.no_sha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

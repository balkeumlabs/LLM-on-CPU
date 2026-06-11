#!/usr/bin/env python3
"""
Synthetic FHIR-shaped clinical data generator (M2).

Produces NO real PHI — every record is procedurally generated and seeded, so
runs are fully reproducible. Outputs:

  data/
    shards/<profile>.jsonl     one FHIR Bundle per line (Patient + Observations
                               + Conditions + MedicationStatements)
    prompts.jsonl              instruction-style tasks (summarize vitals / flag
                               abnormal labs / triage) referencing each bundle
    manifest.json              seed, counts, and per-shard condition histograms
                               that quantify the non-IID split for the paper

Why FHIR-shaped: the proposal's edge nodes parse FHIR records locally. We mimic
the resource structure (Patient/Observation/Condition/MedicationStatement) so
the inference prompts look like real clinical-edge workloads — without touching
MIMIC or any credentialed dataset.

Non-IID: each "specialty profile" skews its pathology + lab-abnormality mix, so
condition prevalence differs sharply across shards (reported in the manifest).
This backs the paper's "diverse patient pathologies in a consumer setting" claim.

Usage:
    python src/gen_synthetic_fhir.py                 # defaults: 60/shard, seed 42
    python src/gen_synthetic_fhir.py --per-shard 100 --seed 7 --out data
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter

# --------------------------------------------------------------------------
# Reference ranges for vitals/labs (adult). Used to label values as
# normal/abnormal in prompts and to inject profile-specific abnormalities.
# (loinc codes are illustrative; units SI/conventional as commonly charted.)
# --------------------------------------------------------------------------
VITALS = {
    "heart_rate":      {"loinc": "8867-4",  "unit": "/min",  "normal": (60, 100)},
    "resp_rate":       {"loinc": "9279-1",  "unit": "/min",  "normal": (12, 20)},
    "sbp":             {"loinc": "8480-6",  "unit": "mmHg",  "normal": (90, 130)},
    "dbp":             {"loinc": "8462-4",  "unit": "mmHg",  "normal": (60, 85)},
    "temp_c":          {"loinc": "8310-5",  "unit": "Cel",   "normal": (36.1, 37.8)},
    "spo2":            {"loinc": "59408-5", "unit": "%",     "normal": (94, 100)},
}
LABS = {
    "hemoglobin":      {"loinc": "718-7",   "unit": "g/dL",   "normal": (12.0, 17.0)},
    "wbc":             {"loinc": "6690-2",   "unit": "10*3/uL","normal": (4.0, 11.0)},
    "platelets":       {"loinc": "777-3",    "unit": "10*3/uL","normal": (150, 400)},
    "creatinine":      {"loinc": "2160-0",   "unit": "mg/dL",  "normal": (0.6, 1.3)},
    "potassium":       {"loinc": "2823-3",   "unit": "mmol/L", "normal": (3.5, 5.1)},
    "sodium":          {"loinc": "2951-2",   "unit": "mmol/L", "normal": (135, 145)},
    "glucose":         {"loinc": "2345-7",   "unit": "mg/dL",  "normal": (70, 140)},
    "troponin":        {"loinc": "6598-7",   "unit": "ng/mL",  "normal": (0.0, 0.04)},
    "bnp":             {"loinc": "33762-6",  "unit": "pg/mL",  "normal": (0, 100)},
    "crp":             {"loinc": "1988-5",   "unit": "mg/L",   "normal": (0.0, 5.0)},
}

# Conditions per specialty (SNOMED-ish display terms; codes illustrative).
CONDITIONS = {
    "cardiology":   ["Heart failure", "Atrial fibrillation", "Acute coronary syndrome",
                     "Hypertension", "Cardiomyopathy"],
    "pulmonology":  ["COPD exacerbation", "Community-acquired pneumonia", "Asthma",
                     "Pulmonary embolism", "Respiratory failure"],
    "nephrology":   ["Acute kidney injury", "Chronic kidney disease", "Hyperkalemia",
                     "Nephrotic syndrome", "Electrolyte imbalance"],
    "endocrine":    ["Diabetic ketoacidosis", "Type 2 diabetes mellitus", "Hypoglycemia",
                     "Thyroid storm", "Adrenal insufficiency"],
    "general":      ["Sepsis", "Urinary tract infection", "Dehydration",
                     "Anemia", "Influenza"],
}

# Which labs each profile tends to push out of range (the non-IID signal).
PROFILE_SKEW = {
    "cardiology":  ["troponin", "bnp", "sbp", "dbp", "heart_rate"],
    "pulmonology": ["spo2", "resp_rate", "crp", "wbc"],
    "nephrology":  ["creatinine", "potassium", "sodium"],
    "endocrine":   ["glucose", "sodium", "potassium"],
    "general":     ["wbc", "crp", "temp_c", "heart_rate"],
}

MEDS = {
    "cardiology":  ["Furosemide", "Metoprolol", "Apixaban", "Lisinopril", "Atorvastatin"],
    "pulmonology": ["Albuterol", "Prednisone", "Azithromycin", "Tiotropium"],
    "nephrology":  ["Sevelamer", "Calcium gluconate", "Insulin (K+ shift)", "Sodium bicarbonate"],
    "endocrine":   ["Insulin glargine", "Metformin", "Dextrose 50%", "Hydrocortisone"],
    "general":     ["Ceftriaxone", "Acetaminophen", "Normal saline", "Ondansetron"],
}

FIRST = ["Alex", "Sam", "Jordan", "Taylor", "Casey", "Riley", "Morgan", "Jamie",
         "Avery", "Quinn", "Devon", "Skyler", "Reese", "Rowan", "Sage"]
LAST = ["Lee", "Patel", "Kim", "Garcia", "Nguyen", "Smith", "Mueller", "Rossi",
        "Haddad", "Ivanov", "Tanaka", "Okafor", "Andersson", "Costa", "Cohen"]


def _val(rng: random.Random, lo: float, hi: float, integer: bool = False) -> float:
    v = rng.uniform(lo, hi)
    return int(round(v)) if integer else round(v, 2)


def _abnormal(rng: random.Random, normal: tuple[float, float], integer: bool, high_bias=0.7):
    """Generate a value clearly outside the normal range."""
    lo, hi = normal
    span = hi - lo
    if rng.random() < high_bias:
        v = hi + rng.uniform(0.15, 1.2) * (span if span > 0 else hi or 1)
    else:
        v = lo - rng.uniform(0.15, 0.6) * (span if span > 0 else lo or 1)
        v = max(v, 0)
    return int(round(v)) if integer else round(v, 2)


def _measure(rng: random.Random, spec: dict, force_abnormal: bool):
    lo, hi = spec["normal"]
    integer = isinstance(lo, int) and isinstance(hi, int)
    if force_abnormal:
        return _abnormal(rng, spec["normal"], integer)
    # mostly normal, occasional incidental abnormal
    if rng.random() < 0.15:
        return _abnormal(rng, spec["normal"], integer)
    return _val(rng, lo, hi, integer)


def _is_abnormal(name: str, value: float, table: dict) -> bool:
    lo, hi = table[name]["normal"]
    return value < lo or value > hi


def make_bundle(rng: random.Random, profile: str, idx: int) -> dict:
    pid = f"{profile[:3]}-{idx:04d}"
    age = rng.randint(28, 92)
    sex = rng.choice(["male", "female"])
    name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
    skew = set(PROFILE_SKEW[profile])

    entries = [{
        "resource": {
            "resourceType": "Patient", "id": pid,
            "name": [{"text": name}],
            "gender": sex,
            "_synthetic_age": age,
        }
    }]

    # Vitals + labs (skewed labs forced abnormal for this profile).
    obs = {}
    for table, names in (("vital", VITALS), ("lab", LABS)):
        for n, spec in names.items():
            force = n in skew and rng.random() < 0.75
            v = _measure(rng, spec, force)
            obs[n] = v
            entries.append({"resource": {
                "resourceType": "Observation",
                "category": table,
                "code": {"coding": [{"system": "http://loinc.org",
                                     "code": spec["loinc"], "display": n}]},
                "subject": {"reference": f"Patient/{pid}"},
                "valueQuantity": {"value": v, "unit": spec["unit"]},
            }})

    # Conditions (1-3, drawn from this profile's pathology set).
    conds = rng.sample(CONDITIONS[profile], k=rng.randint(1, 3))
    for c in conds:
        entries.append({"resource": {
            "resourceType": "Condition",
            "clinicalStatus": {"coding": [{"code": "active"}]},
            "code": {"text": c},
            "subject": {"reference": f"Patient/{pid}"},
        }})

    # Meds (1-3 from this profile).
    for m in rng.sample(MEDS[profile], k=rng.randint(1, min(3, len(MEDS[profile])))):
        entries.append({"resource": {
            "resourceType": "MedicationStatement",
            "status": "active",
            "medicationCodeableConcept": {"text": m},
            "subject": {"reference": f"Patient/{pid}"},
        }})

    return {
        "resourceType": "Bundle", "type": "collection", "id": f"bundle-{pid}",
        "_profile": profile,
        "_summary": {"patient_id": pid, "age": age, "sex": sex, "name": name,
                     "conditions": conds, "observations": obs},
        "entry": entries,
    }


# --------------------------------------------------------------------------
# Instruction prompts (drive E2/E3 inference). Three task types.
# --------------------------------------------------------------------------
def _render_observations(obs: dict, table: dict, vitals_only=False, labs_only=False) -> str:
    keys = []
    if not labs_only:
        keys += [k for k in VITALS if k in obs]
    if not vitals_only:
        keys += [k for k in LABS if k in obs]
    parts = []
    for k in keys:
        src = VITALS if k in VITALS else LABS
        parts.append(f"{k}={obs[k]} {src[k]['unit']}")
    return ", ".join(parts)


def make_prompts(rng: random.Random, bundle: dict) -> list[dict]:
    s = bundle["_summary"]
    obs = s["observations"]
    pid = s["patient_id"]
    prompts = []

    # 1) Summarize vitals
    vit = _render_observations(obs, VITALS, vitals_only=True)
    prompts.append({
        "task": "summarize_vitals", "patient_id": pid, "profile": bundle["_profile"],
        "prompt": (f"You are a clinical assistant. Summarize the following bedside "
                   f"vitals for a {s['age']}yo {s['sex']} patient in one sentence and "
                   f"state whether they are within normal limits.\nVitals: {vit}"),
    })

    # 2) Flag abnormal labs (with ground-truth abnormal set for later scoring)
    abn = [n for n in LABS if n in obs and _is_abnormal(n, obs[n], LABS)]
    labs = _render_observations(obs, LABS, labs_only=True)
    prompts.append({
        "task": "flag_abnormal_labs", "patient_id": pid, "profile": bundle["_profile"],
        "prompt": (f"List which of the following laboratory values are outside the "
                   f"normal adult reference range and briefly note the clinical "
                   f"concern for each.\nLabs: {labs}"),
        "ground_truth_abnormal": abn,
    })

    # 3) Triage acuity
    prompts.append({
        "task": "triage", "patient_id": pid, "profile": bundle["_profile"],
        "prompt": (f"Given the active problems {s['conditions']} and current "
                   f"observations ({_render_observations(obs, None)}), assign a triage "
                   f"acuity (1=resuscitation to 5=non-urgent) and justify in one line."),
    })
    return prompts


# --------------------------------------------------------------------------
def generate(out_dir: str, per_shard: int, seed: int, profiles: list[str]) -> dict:
    rng = random.Random(seed)
    shards_dir = os.path.join(out_dir, "shards")
    os.makedirs(shards_dir, exist_ok=True)

    prompts_path = os.path.join(out_dir, "prompts.jsonl")
    manifest = {"seed": seed, "per_shard": per_shard, "profiles": profiles,
                "total_patients": 0, "total_prompts": 0, "shards": {}}

    with open(prompts_path, "w") as pf:
        for profile in profiles:
            cond_hist = Counter()
            abn_lab_hist = Counter()
            shard_path = os.path.join(shards_dir, f"{profile}.jsonl")
            with open(shard_path, "w") as sf:
                for i in range(per_shard):
                    bundle = make_bundle(rng, profile, i)
                    sf.write(json.dumps(bundle) + "\n")
                    cond_hist.update(bundle["_summary"]["conditions"])
                    for pr in make_prompts(rng, bundle):
                        pf.write(json.dumps(pr) + "\n")
                        manifest["total_prompts"] += 1
                        if pr["task"] == "flag_abnormal_labs":
                            abn_lab_hist.update(pr["ground_truth_abnormal"])
            manifest["total_patients"] += per_shard
            manifest["shards"][profile] = {
                "patients": per_shard,
                "file": os.path.relpath(shard_path, out_dir),
                "condition_histogram": dict(cond_hist),
                "abnormal_lab_histogram": dict(abn_lab_hist),
            }

    # Non-IID signal: how concentrated each shard's top condition is.
    for profile, info in manifest["shards"].items():
        hist = info["condition_histogram"]
        total = sum(hist.values()) or 1
        top = max(hist.values()) if hist else 0
        info["top_condition_share"] = round(top / total, 3)

    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main() -> int:
    p = argparse.ArgumentParser(description="Generate synthetic FHIR-shaped clinical data + prompts.")
    p.add_argument("--out", default="data", help="output directory (default: data)")
    p.add_argument("--per-shard", type=int, default=60, help="patients per specialty shard")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")
    p.add_argument("--profiles", nargs="+", default=list(CONDITIONS.keys()),
                   help="specialty profiles / shards to generate")
    args = p.parse_args()

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = args.out if os.path.isabs(args.out) else os.path.join(here, args.out)

    m = generate(out_dir, args.per_shard, args.seed, args.profiles)
    print(f"Generated {m['total_patients']} synthetic patients across "
          f"{len(m['profiles'])} non-IID shards, {m['total_prompts']} prompts.")
    print(f"  out: {out_dir}")
    print(f"  seed: {m['seed']}  (reproducible)")
    print("  per-shard top-condition share (non-IID signal):")
    for prof, info in m["shards"].items():
        print(f"    {prof:12s} top={info['top_condition_share']:.2f}  "
              f"conditions={list(info['condition_histogram'].keys())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

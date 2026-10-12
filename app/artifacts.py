"""Artifact loading + startup integrity check. Frozen artifacts must match manifest.json or the API refuses to start."""
import hashlib, json
from pathlib import Path
def _sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify_manifest(d):
    d = Path(d); m = json.loads((d / "manifest.json").read_text()); bad = []
    for rel, meta in m["files"].items():
        p = d / rel
        if not p.is_file() or _sha(p) != meta["sha256"]: bad.append(rel)
    if bad: raise RuntimeError(f"Artifact integrity check failed for: {bad}")
    return m
def load_bundle(d):
    d = Path(d); L = lambda rel: json.loads((d / rel).read_text())
    return dict(contract=L("model_contract.json"), card=L("model_card.json"), request_schema=L("schema/assess_request.schema.json"), response_schema=L("schema/assess_response.schema.json"), error_schema=L("schema/error.schema.json"))
def build_meta(contract, card):
    """Exposes the frozen model-card values verbatim (no recomputation, no rounding changes)."""
    return {"contract_version": contract["contract_version"], "model_version": contract["model_version"], "disclaimer": contract["disclaimer"], "population_note": contract["population_note"],
            "targets": [{k: t[k] for k in ("id", "label", "definition", "model_type", "hyperparameters", "calibration")} for t in contract["targets"]],
            "performance": card["performance_summary"], "limitations": card["limitations"],
            "consistency_warning": {"margin": contract["consistency_margin"], "description": "Display heuristic for a warning only (a vessel probability above CAD by more than the margin). Not a clinical or statistical threshold. Predicted probabilities are never modified."},
            "input_policy": {"required_patient_fields": 50, "imputation": "none", "range_checks": "informational data-support flags only; not prediction uncertainty"},
            "training": {"n": contract["training"]["n"], "class_counts": contract["training"]["class_counts"], "source": contract["training"]["source"]}}

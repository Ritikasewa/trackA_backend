"""Pure-Python inference core (no web framework). Loads ONLY the versioned artifacts directory; no training code, no sklearn required for LR.
assess(request) -> response dict; raises ContractError(errors) for invalid input. A FastAPI layer will wrap this unchanged."""
import json, os, hashlib, math
import numpy as np
class ContractError(Exception):
    def __init__(self, errors): super().__init__(json.dumps(errors)); self.errors = errors
class Assessor:
    def __init__(self, artifact_dir):
        self.dir = artifact_dir; L = lambda p: json.load(open(os.path.join(artifact_dir, p)))
        self.contract = L("model_contract.json"); self.enc = L("preprocessing/encoding.json"); self.stats = L("preprocessing/training_stats.json"); self.card = L("model_card.json")
        self.features = self.contract["feature_order"]; self.fields = {f["api"]: f for f in self.enc["fields"]}; self.models = {}
        for t in self.contract["targets"]:
            path = os.path.join(artifact_dir, t["file"])
            if t["model_type"] == "logistic_regression":
                m = json.load(open(path)); assert m["feature_order"] == self.features
                self.models[t["id"]] = ("lr", {k: np.asarray(m[k], dtype=np.float64) for k in ("scaler_mean", "scaler_scale", "coef")} | {"intercept": float(m["intercept"])})
            else:
                import xgboost as xgb; b = xgb.Booster(); b.load_model(path); self.models[t["id"]] = ("xgb", b)
    # ---------- validation ----------
    def validate(self, request):
        err = []; add = lambda f, c, m: err.append({"field": f, "code": c, "message": m})
        if not isinstance(request, dict) or "patient" not in request: raise ContractError([{"field": "patient", "code": "missing_field", "message": "Request must be an object with a 'patient' object."}])
        extra_top = set(request) - {"patient", "request_id"}
        for k in sorted(extra_top): add(k, "unknown_field", "Unknown top-level key.")
        if "request_id" in request and not (isinstance(request["request_id"], str) and 0 < len(request["request_id"]) <= 128): add("request_id", "invalid_type", "request_id must be a non-empty string (<=128 chars).")
        p = request["patient"]
        if not isinstance(p, dict): raise ContractError([{"field": "patient", "code": "invalid_type", "message": "'patient' must be an object."}])
        for k in sorted(p):
            if k not in self.fields:
                lk = k.lower()
                if lk in self.contract["forbidden_target_names"]: add(f"patient.{k}", "forbidden_target_field", "Target/outcome labels must never be supplied as inputs.")
                elif lk in self.contract["derived_names"]: add(f"patient.{k}", "derived_field_not_accepted", "Derived field; computed server-side from weight_kg and height_cm.")
                else: add(f"patient.{k}", "unknown_field", "Unknown patient field.")
        for a, f in self.fields.items():
            if a not in p: add(f"patient.{a}", "missing_field", "Required field is missing (no imputation is performed)."); continue
            v = p[a]; k = f["kind"]
            if k == "bool":
                if not isinstance(v, bool): add(f"patient.{a}", "invalid_type", "Must be JSON true or false.")
            elif k == "number":
                if isinstance(v, bool) or not isinstance(v, (int, float)): add(f"patient.{a}", "invalid_type", "Must be a number.")
                elif not math.isfinite(v): add(f"patient.{a}", "not_finite", "Must be a finite number.")
                elif v < 0 or (a in ("weight_kg", "height_cm") and v <= 0): add(f"patient.{a}", "invalid_value", "Must be non-negative (weight_kg and height_cm must be > 0).")
            elif k == "int_enum":
                if isinstance(v, bool) or not isinstance(v, int) or v not in f["allowed"]: add(f"patient.{a}", "invalid_enum", f"Must be an integer in {f['allowed']}.")
            elif k == "enum":
                if not isinstance(v, str) or v not in f["enum"]: add(f"patient.{a}", "invalid_enum", f"Must be one of {f['enum']}.")
        if err: raise ContractError(err)
    # ---------- encoding ----------
    def encode(self, p):
        v = {}
        for a, f in self.fields.items():
            k = f["kind"]
            if k == "bool": v[f["feature"]] = 1.0 if p[a] else 0.0
            elif k in ("number", "int_enum"): v[f["feature"]] = float(p[a])
            else: v |= {ft: float(x) for ft, x in f["map"][p[a]].items()}
        d = self.enc["derived"][0]; bmi = p["weight_kg"] / (p["height_cm"] / 100.0) ** 2; v[d["feature"]] = float(bmi)
        assert set(v) == set(self.features), set(self.features) ^ set(v)
        return np.array([v[f] for f in self.features], dtype=np.float64), float(bmi)
    # ---------- prediction ----------
    def predict_raw(self, x):
        out = {}
        for t, (kind, m) in self.models.items():
            if kind == "lr": z = float(((x - m["scaler_mean"]) / m["scaler_scale"]) @ m["coef"] + m["intercept"]); out[t] = 1.0 / (1.0 + math.exp(-z))
            else:
                import xgboost as xgb; out[t] = float(m.predict(xgb.DMatrix(x.reshape(1, -1)))[0])
        return out
    # ---------- input checks (deterministic, from training statistics only) ----------
    def input_checks(self, p, bmi):
        oor, sparse = [], []; extrap = False
        for a, f in self.fields.items():
            s = self.stats["fields"][a]
            if f["kind"] == "number":
                v = p[a]
                if v < s["min"] or v > s["max"]: oor.append({"field": a, "value": v, "level": "outside_observed_range", "observed_min": s["min"], "observed_max": s["max"]}); extrap = True
                elif v < s["p01"] or v > s["p99"]: oor.append({"field": a, "value": v, "level": "outside_central_98pct", "observed_min": s["min"], "observed_max": s["max"]})
            else:
                key = json.dumps(p[a]) if f["kind"] == "bool" else str(p[a]); key = {"true": "true", "false": "false"}.get(key, key); n = s["counts"].get(key, 0)
                if n < self.contract["sparse_category_threshold"]: sparse.append({"field": a, "value": p[a], "training_count": n})
        b = self.stats["derived"]["bmi"]
        if bmi < b["min"] or bmi > b["max"]: oor.append({"field": "bmi (derived)", "value": round(bmi, 2), "level": "outside_observed_range", "observed_min": b["min"], "observed_max": b["max"]}); extrap = True
        return {"derived": {"bmi": round(bmi, 2)}, "out_of_observed_range": oor, "sparse_categories": sparse, "extrapolation_flag": extrap}
    def assess(self, request):
        self.validate(request); p = request["patient"]; x, bmi = self.encode(p); raw = self.predict_raw(x); c = self.contract
        preds = {t["id"]: {"target": t["id"], "label": t["label"], "probability": round(raw[t["id"]], 4), "model_type": t["model_type"], "definition": t["definition"]} for t in c["targets"]}
        margin = c["consistency_margin"]; exc = [t for t in ("LAD", "LCX", "RCA") if raw[t] > raw["CAD"] + margin]
        aid = hashlib.sha256((c["model_version"] + "|").encode() + x.tobytes()).hexdigest()
        return {"contract_version": c["contract_version"], "model_version": c["model_version"], "request_id": request.get("request_id"), "assessment_id": aid, "predictions": preds,
                "model_performance": self.card["performance_summary"],
                "consistency": {"vessels_exceeding_cad": exc, "margin": margin, "probabilities_modified": False,
                                "note": "Targets are modeled independently; a vessel probability above CAD is flagged, never adjusted."},
                "input_checks": self.input_checks(p, bmi), "uncertainty": None, "data_support": None,
                "population_note": c["population_note"], "disclaimer": c["disclaimer"]}

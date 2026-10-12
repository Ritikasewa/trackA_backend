"""Additive model explanation (SHAP) for the four frozen models. READ-ONLY: it only uses the objects an `Assessor` has already loaded
(scaler/coefficients for the logistic models, the XGBoost booster for LCX) and never changes them, the artifacts, preprocessing or any prediction.

Method (both are exact and additive in log-odds space):
  * Logistic regression (CAD, LAD, RCA): contribution_j = coef_j * (x_j - scaler_mean_j) / scaler_scale_j, reference = a patient at the training mean of every
    feature, so reference log-odds = intercept. This is exactly SHAP for a linear model with the training-mean reference.
  * XGBoost (LCX): the booster's built-in TreeSHAP (`pred_contribs`); reference log-odds = its expected margin (the bias column).
For every target:  reference_log_odds + sum(contributions) == model log-odds, and sigmoid of it == the model probability that /assess reports.

The 54 encoded features are folded back into the 50 form fields (one-hot columns are summed into their parent field) plus ONE separate row for the
derived BMI. No clinical advice of any kind is produced here."""
import hashlib, json, math, os
import numpy as np

EXPLAIN_VERSION = "1.0.0"
# Labels match the existing form sections (src/domain in the frontend).
GROUP_LABELS = {"demographics": "Demographics", "history_risk_factors": "History / Risk factors", "exam_symptoms": "Examination / Symptoms",
                "ecg": "ECG", "labs": "Laboratory", "echo": "Echocardiography"}
BMI_LABEL = "BMI (derived from weight and height)"
ENUM_DISPLAY = {"lbbb": "LBBB", "rbbb": "RBBB"}
TOLERANCE = 1e-6        # internal fail-closed guard: reconstruction must agree with the model output to this tolerance (XGBoost is float32)
NOTES = [
    "These values show how individual inputs influenced this model's estimate for this patient. They describe the model's attribution, not causes of disease or treatment recommendations.",
    "Contributions are in log-odds units. For each model, the reference log-odds plus all contributions equals this patient's model log-odds.",
    "The reference is defined per model (see each target's reference definition) and is not a clinical baseline risk.",
]
REFERENCE_DEFINITION = {
    "logistic_regression": "Reference model estimate for the logistic models: the estimate for a patient at the training-data mean of every input (the model's own standardisation reference). "
                           "Each contribution is the coefficient multiplied by how far the patient's value is from that training mean, in standard-deviation units.",
    "xgboost": "Reference model estimate for the XGBoost (LCX) model: its expected output over its training data (the TreeSHAP baseline). "
               "Contributions are TreeSHAP values. This is a different definition from the logistic models' reference.",
}


def _sigmoid(z): return 1.0 / (1.0 + math.exp(-z))


def _fmt_number(v):
    v = float(v); return str(int(v)) if v == int(v) else f"{v:.4g}"


class Explainer:
    def __init__(self, assessor):
        self.a = assessor
        self.features = {f["api"]: f for f in json.load(open(os.path.join(assessor.dir, "schema/features.json")))["fields"]}
        self.stats = assessor.stats
        self.index = {name: i for i, name in enumerate(assessor.features)}
        # field api -> encoded feature columns (indices); BMI is handled as its own derived row
        self.cols = {}
        for api, f in assessor.fields.items():
            names = list(next(iter(f["map"].values())).keys()) if f["kind"] == "enum" else [f["feature"]]
            self.cols[api] = [self.index[n] for n in names]
        d = assessor.enc["derived"][0]; self.bmi_col = self.index[d["feature"]]; self.bmi_ref = float(self.stats["features"][d["feature"]]["median"])
        used = sorted([i for c in self.cols.values() for i in c] + [self.bmi_col])
        assert used == list(range(len(assessor.features))), "every encoded feature must belong to exactly one field row"  # 54 = 50 fields + extra one-hot columns + BMI

    # ---------- per-target attributions on the 54 encoded features ----------
    def _attributions(self, kind, m, x):
        if kind == "lr":
            phi = m["coef"] * (x - m["scaler_mean"]) / m["scaler_scale"]; base = float(m["intercept"]); return phi, base
        import xgboost as xgb
        c = m.predict(xgb.DMatrix(x.reshape(1, -1)), pred_contribs=True)[0].astype(np.float64); return c[:-1], float(c[-1])

    # ---------- display helpers ----------
    def _display(self, api, v):
        k = self.a.fields[api]["kind"]
        if k == "bool": return "Yes" if v else "No"
        if k in ("number", "int_enum"): return _fmt_number(v)
        return ENUM_DISPLAY.get(v, str(v).capitalize())

    def _reference(self, api):
        k = self.a.fields[api]["kind"]; obs = self.features[api]
        if k == "number": return float(obs["observed"]["median"]), "median"
        counts = obs["observed_counts"]; top = max(counts, key=lambda c: counts[c])
        if k == "bool": return top == "true", "most_common"
        if k == "int_enum": return int(top), "most_common"
        return top, "most_common"

    # ---------- main entry ----------
    def explain(self, request):
        a = self.a; a.validate(request); p = request["patient"]; x, bmi = a.encode(p); raw = a.predict_raw(x); c = a.contract
        order = list(a.fields.keys())
        rows_meta = {}
        for api in order:
            f = self.features[api]; ref, kind = self._reference(api)
            rows_meta[api] = {"field": api, "label": f["label"], "group": f["group"], "derived": False, "unit": f.get("unit"), "patient_value": p[api], "display_value": self._display(api, p[api]),
                              "reference_value": ref, "reference_display": self._display(api, ref), "reference_kind": kind}
        bmi_meta = {"field": "bmi", "label": BMI_LABEL, "group": "demographics", "derived": True, "unit": "kg/m²", "patient_value": float(bmi), "display_value": _fmt_number(round(bmi, 1)),
                    "reference_value": self.bmi_ref, "reference_display": _fmt_number(round(self.bmi_ref, 1)), "reference_kind": "median"}
        targets = {}
        for t in c["targets"]:
            tid = t["id"]; kind, m = a.models[tid]; phi, base = self._attributions(kind, m, x)
            rows = []
            for api in order:
                v = float(sum(phi[i] for i in self.cols[api])); rows.append({**rows_meta[api], "contribution": v})
            rows.append({**bmi_meta, "contribution": float(phi[self.bmi_col])})
            total = float(sum(r["contribution"] for r in rows)); z = base + total; recon = _sigmoid(z)
            # fail closed: never return an explanation that disagrees with the prediction
            if abs(total - float(np.sum(phi))) > 1e-9 or abs(recon - raw[tid]) > TOLERANCE: raise RuntimeError(f"explanation for {tid} disagrees with the model output")
            for r in rows: r["direction"] = "positive" if r["contribution"] > 1e-12 else "negative" if r["contribution"] < -1e-12 else "neutral"
            pos = {r["field"]: i for i, r in enumerate(rows)}
            rows.sort(key=lambda r: (-abs(r["contribution"]), pos[r["field"]]))
            sections = [{"group": g, "label": GROUP_LABELS[g], "contribution": float(sum(r["contribution"] for r in rows if r["group"] == g))} for g in GROUP_LABELS]
            model_type = "logistic_regression" if kind == "lr" else "xgboost"
            targets[tid] = {"target": tid, "label": t["label"], "model_type": model_type, "method": "exact_linear_logodds" if kind == "lr" else "treeshap",
                            "reference": {"log_odds": base, "probability": _sigmoid(base), "definition": REFERENCE_DEFINITION[model_type]},
                            "estimate": {"log_odds": z, "probability": raw[tid], "probability_rounded": round(raw[tid], 4), "reconstructed_probability": recon},
                            "contributions": rows, "sections": sections}
        aid = hashlib.sha256((c["model_version"] + "|").encode() + x.tobytes()).hexdigest()
        return {"explain_version": EXPLAIN_VERSION, "contract_version": c["contract_version"], "model_version": c["model_version"], "request_id": request.get("request_id"),
                "assessment_id": aid, "targets": targets, "notes": NOTES, "disclaimer": c["disclaimer"]}

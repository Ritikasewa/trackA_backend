"""POST /explain: the SHAP explanation must be mathematically consistent with /assess for ALL four frozen models.
Nothing here retrains or alters anything; contributions are re-derived independently from the frozen artifact files."""
import copy, csv, hashlib, json, math, re
import numpy as np, pytest
from helpers import *

TARGETS = ("CAD", "LAD", "LCX", "RCA")
TOL = {"CAD": 1e-9, "LAD": 1e-9, "RCA": 1e-9, "LCX": 1e-6}     # logistic: float64-exact; XGBoost: float32 margin
def sigmoid(z): return 1.0 / (1.0 + math.exp(-z))

def assert_additive(t, tid):
    """THE consistency check. Raises if reference + sum(contributions) != model log-odds, or the probability disagrees."""
    total = sum(r["contribution"] for r in t["contributions"]); z = t["reference"]["log_odds"] + total
    assert abs(z - t["estimate"]["log_odds"]) <= TOL[tid], (tid, z, t["estimate"]["log_odds"])
    assert abs(sigmoid(z) - t["estimate"]["probability"]) <= TOL[tid], (tid, sigmoid(z), t["estimate"]["probability"])
    assert abs(t["estimate"]["reconstructed_probability"] - t["estimate"]["probability"]) <= TOL[tid]

def explain(client, patient, rid="x"): return client.post("/explain", json={"request_id": rid, "patient": patient})

def edge_patients():
    base = BASE_PATIENT; out = {"golden_median": copy.deepcopy(base)}
    for name, g in GOLDEN_REQ.items(): out[f"golden_{name}"] = copy.deepcopy(g["patient"])
    lo, hi = copy.deepcopy(base), copy.deepcopy(base)
    for f in FEATURES:
        a = f["api"]
        if f["kind"] == "number": lo[a] = f["observed"]["min"]; hi[a] = f["observed"]["max"]
        elif f["kind"] == "bool": lo[a] = False; hi[a] = True
        elif f["kind"] == "enum": lo[a] = f["enum"][0]; hi[a] = f["enum"][-1]
        else: lo[a] = f["allowed"][0]; hi[a] = f["allowed"][-1]
    out["all_min_all_false"] = lo; out["all_max_all_true"] = hi
    return out

@pytest.fixture(scope="session")
def all_explanations(client, patients):
    out = []
    for i, p in enumerate(patients):
        r = explain(client, p, f"row{i}"); assert r.status_code == 200, (i, r.text); out.append(r.json())
    return out

# ---------- shape ----------
def test_response_shape_and_identity(client, patients, all_responses, all_explanations):
    for i in (0, 7, 150, 302):
        e, a = all_explanations[i], all_responses[i]
        assert set(e) == {"explain_version", "contract_version", "model_version", "request_id", "assessment_id", "targets", "notes", "disclaimer"}
        assert e["request_id"] == f"row{i}" and e["assessment_id"] == a["assessment_id"] and e["contract_version"] == a["contract_version"] == "1.0.0" and e["model_version"] == a["model_version"]
        assert e["disclaimer"] == a["disclaimer"] and list(e["targets"]) == list(TARGETS)
        for tid in TARGETS:
            t = e["targets"][tid]
            assert set(t) == {"target", "label", "model_type", "method", "reference", "estimate", "contributions", "sections"} and t["target"] == tid
            assert t["model_type"] == a["predictions"][tid]["model_type"] and t["method"] == ("treeshap" if tid == "LCX" else "exact_linear_logodds")
            assert set(t["reference"]) == {"log_odds", "probability", "definition"} and abs(sigmoid(t["reference"]["log_odds"]) - t["reference"]["probability"]) < 1e-12
            assert [s["group"] for s in t["sections"]] == ["demographics", "history_risk_factors", "exam_symptoms", "ecg", "labs", "echo"]

def test_models_are_what_the_contract_says(all_explanations):
    t = all_explanations[0]["targets"]; assert [t[k]["model_type"] for k in TARGETS] == ["logistic_regression", "logistic_regression", "xgboost", "logistic_regression"]
    assert t["LCX"]["reference"]["definition"] != t["CAD"]["reference"]["definition"] and t["CAD"]["reference"]["definition"] == t["LAD"]["reference"]["definition"] == t["RCA"]["reference"]["definition"]

# ---------- THE test: additivity + parity with /assess ----------
@pytest.mark.parametrize("tid", TARGETS)
def test_additive_and_matches_assess_all_303(tid, patients, all_responses, all_explanations):
    worst = 0.0
    for i, (e, a) in enumerate(zip(all_explanations, all_responses)):
        t = e["targets"][tid]; assert_additive(t, tid)
        assert t["estimate"]["probability_rounded"] == a["predictions"][tid]["probability"], (i, tid)        # identical to what /assess reports
        assert abs(t["estimate"]["probability"] - a["predictions"][tid]["probability"]) <= 5e-5 + 1e-12           # /assess rounds to 4 d.p.
        worst = max(worst, abs(sigmoid(t["reference"]["log_odds"] + sum(r["contribution"] for r in t["contributions"])) - t["estimate"]["probability"]))
    assert worst <= TOL[tid], worst

def test_additive_on_golden_and_edge_inputs(client):
    for name, p in edge_patients().items():
        a = client.post("/assess", json={"request_id": name, "patient": p}); e = explain(client, p, name)
        assert a.status_code == 200 and e.status_code == 200, (name, e.text)
        for tid in TARGETS:
            t = e.json()["targets"][tid]; assert_additive(t, tid); assert t["estimate"]["probability_rounded"] == a.json()["predictions"][tid]["probability"], (name, tid)

def test_the_parity_check_actually_fails_on_disagreement(all_explanations):
    for tid in TARGETS:
        t = copy.deepcopy(all_explanations[3]["targets"][tid]); assert_additive(t, tid)
        t["contributions"][0]["contribution"] += 0.01
        with pytest.raises(AssertionError): assert_additive(t, tid)
        t = copy.deepcopy(all_explanations[3]["targets"][tid]); t["reference"]["log_odds"] += 1e-3
        with pytest.raises(AssertionError): assert_additive(t, tid)

# ---------- independent re-derivation from the frozen artifact files (no app code) ----------
def _ref_matrix():
    with open(ART / "reference/train_features.csv") as f: rows = list(csv.DictReader(f))
    order = jl(ART / "model_contract.json")["feature_order"]; return np.array([[float(r[c]) for c in order] for r in rows]), order

def _fold_map():
    enc = jl(ART / "preprocessing/encoding.json"); order = enc["feature_order"]; cols = {}
    for f in enc["fields"]: cols[f["api"]] = [f["feature"]] if f["kind"] != "enum" else list(next(iter(f["map"].values())).keys())
    return cols, order, enc["derived"][0]["feature"]

def _independent_raw(tid, X):
    """returns (reference log-odds, per-feature contributions n x 54) computed straight from the artifact files."""
    if tid != "LCX":
        m = jl(ART / f"models/{tid}_lr.json"); phi = np.array(m["coef"]) * (X - np.array(m["scaler_mean"])) / np.array(m["scaler_scale"]); return float(m["intercept"]), phi
    import xgboost as xgb
    b = xgb.Booster(); b.load_model(str(ART / "models/LCX_xgb.json")); c = b.predict(xgb.DMatrix(X), pred_contribs=True).astype(np.float64); return float(c[0, -1]), c[:, :-1]

@pytest.mark.parametrize("tid", TARGETS)
def test_contributions_match_independent_derivation_and_50_field_aggregation(tid, all_explanations):
    X, order = _ref_matrix(); cols, enc_order, bmi = _fold_map(); assert enc_order == order and len(order) == 54
    base, phi = _independent_raw(tid, X); idx = {n: i for i, n in enumerate(order)}
    worst_fold = worst_total = worst_base = 0.0
    for i, e in enumerate(all_explanations):
        t = e["targets"][tid]; by = {r["field"]: r for r in t["contributions"]}
        worst_base = max(worst_base, abs(t["reference"]["log_odds"] - base))
        for api, names in cols.items(): worst_fold = max(worst_fold, abs(by[api]["contribution"] - sum(phi[i, idx[n]] for n in names)))
        worst_fold = max(worst_fold, abs(by["bmi"]["contribution"] - phi[i, idx[bmi]]))
        worst_total = max(worst_total, abs(sum(r["contribution"] for r in t["contributions"]) - phi[i].sum()))     # aggregation preserves the total of all 54 raw contributions
    tol = 1e-9 if tid != "LCX" else 1e-6
    assert worst_base <= tol and worst_fold <= tol and worst_total <= tol, (worst_base, worst_fold, worst_total)

def test_exactly_50_fields_plus_one_derived_bmi_and_no_internal_names(all_explanations):
    onehot = {"Typical Chest Pain", "Atypical", "Nonanginal", "BBB_LBBB", "BBB_RBBB", "Sex", "VHD", "Region RWMA", "Function Class"}
    for e in all_explanations[:25]:
        for tid in TARGETS:
            rows = e["targets"][tid]["contributions"]; assert len(rows) == 51 and len({r["field"] for r in rows}) == 51
            assert [r["field"] for r in rows if r["derived"]] == ["bmi"] and sum(r["derived"] for r in rows) == 1
            assert {r["field"] for r in rows if not r["derived"]} == set(FIELD_NAMES) and len(FIELD_NAMES) == 50
            bmi = next(r for r in rows if r["derived"]); assert bmi["label"] == "BMI (derived from weight and height)" and bmi["group"] == "demographics"
            for r in rows: assert r["label"] not in onehot or r["field"] in ("sex", "valvular_heart_disease", "rwma_region", "function_class"); assert "BBB_" not in json.dumps(r)
    labels = {r["field"]: r["label"] for r in all_explanations[0]["targets"]["CAD"]["contributions"]}
    assert "Typical Chest Pain" not in labels.values() and "Atypical" not in labels.values() and "chest_pain_type" in labels and "bundle_branch_block" in labels

def test_sections_sum_and_ordering_and_direction(all_explanations):
    for e in all_explanations[:60]:
        for tid in TARGETS:
            t = e["targets"][tid]; rows = t["contributions"]
            assert abs(sum(s["contribution"] for s in t["sections"]) - sum(r["contribution"] for r in rows)) <= 1e-9
            for s in t["sections"]: assert abs(s["contribution"] - sum(r["contribution"] for r in rows if r["group"] == s["group"])) <= 1e-9
            mags = [abs(r["contribution"]) for r in rows]; assert mags == sorted(mags, reverse=True)
            for r in rows: assert r["direction"] == ("positive" if r["contribution"] > 1e-12 else "negative" if r["contribution"] < -1e-12 else "neutral")

def test_patient_values_and_reference_are_the_users_own(client, patients):
    p = copy.deepcopy(patients[10]); p["age"] = 71; p["chest_pain_type"] = "typical"; p["diabetes"] = True; p["bundle_branch_block"] = "lbbb"
    t = explain(client, p).json()["targets"]["CAD"]; by = {r["field"]: r for r in t["contributions"]}
    assert by["age"]["patient_value"] == 71 and by["age"]["display_value"] == "71" and by["age"]["reference_kind"] == "median" and by["age"]["reference_value"] == 58.0
    assert by["diabetes"]["display_value"] == "Yes" and by["diabetes"]["reference_kind"] == "most_common" and by["diabetes"]["reference_value"] is False
    assert by["bundle_branch_block"]["display_value"] == "LBBB" and by["chest_pain_type"]["display_value"] == "Typical"
    assert abs(by["bmi"]["patient_value"] - p["weight_kg"] / (p["height_cm"] / 100) ** 2) < 1e-9

# ---------- contract / errors / isolation ----------
BAD = {"missing": lambda p: {k: v for k, v in p.items() if k != "age"}, "unknown": lambda p: {**p, "bogus": 1}, "forbidden": lambda p: {**p, "LAD": True},
       "derived": lambda p: {**p, "bmi": 20.0}, "badtype": lambda p: {**p, "age": "old"}}
@pytest.mark.parametrize("kind", list(BAD))
def test_same_422_errors_as_assess(client, kind):
    body = {"request_id": "e", "patient": BAD[kind](copy.deepcopy(BASE_PATIENT))}
    a, e = client.post("/assess", json=body), client.post("/explain", json=body); assert a.status_code == e.status_code == 422 and a.json() == e.json()
def test_out_of_range_values_are_accepted_like_assess_and_stay_additive(client):
    p = copy.deepcopy(BASE_PATIENT); p["age"] = 5000; p["ldl"] = 2000; body = {"request_id": "e", "patient": p}      # contract: no clamping, flagged as extrapolation, never rejected
    a, e = client.post("/assess", json=body), client.post("/explain", json=body); assert a.status_code == e.status_code == 200
    for tid in TARGETS: assert_additive(e.json()["targets"][tid], tid); assert e.json()["targets"][tid]["estimate"]["probability_rounded"] == a.json()["predictions"][tid]["probability"]
def test_invalid_json_and_non_object(client):
    for raw in ("{not json", "[]", '{"request_id": 5, "patient": {}}', "{}"):
        a = client.post("/assess", content=raw, headers={"content-type": "application/json"}); e = client.post("/explain", content=raw, headers={"content-type": "application/json"})
        assert a.status_code == e.status_code == 422 and a.json() == e.json(), raw

def test_explain_is_additive_and_read_only(client, patients):
    A = client.app.state.assessor; fp = lambda: hashlib.sha256(b"".join(np.asarray(m["coef"]).tobytes() + np.asarray(m["scaler_mean"]).tobytes() + np.asarray(m["scaler_scale"]).tobytes() for k, m in (A.models[t] for t in ("CAD", "LAD", "RCA")))).hexdigest()
    before_fp = fp(); before = [client.post("/assess", json={"request_id": f"r{i}", "patient": patients[i]}).content for i in (0, 50, 200)]
    for i in (0, 50, 200, 0): assert explain(client, patients[i]).status_code == 200
    after = [client.post("/assess", json={"request_id": f"r{i}", "patient": patients[i]}).content for i in (0, 50, 200)]
    assert before == after and fp() == before_fp                                                     # /assess bytes and model arrays unchanged by /explain calls
    assert set(client.get("/openapi.json").json()["paths"]) == {"/assess", "/meta"} and client.get("/explain").status_code == 405

def test_no_clinical_advice_in_explain_text(all_explanations):
    e = all_explanations[0]; texts = e["notes"] + [t["reference"]["definition"] for t in e["targets"].values()]
    banned = re.compile(r"\b(mg|dose|dosage|prescrib\w*|take|start|stop|treat\w*|medicat\w*|statin|aspirin|beta[- ]?blocker|you should|safe|risk factor\w*|caus(e|es|ed|ing))\b", re.I)
    joined = " ".join(texts); stripped = joined.replace("not causes of disease or treatment recommendations", "")
    assert not banned.search(stripped), banned.findall(stripped)
    assert "attribution" in joined and "not a clinical baseline risk" in joined

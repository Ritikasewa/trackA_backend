"""Preprocessing + prediction parity WITHOUT retraining anything: (a) API encoding == training matrix; (b) API probabilities == independent sklearn/xgboost implementations
reconstructed from the frozen artifact files; (c) API probabilities == the approved snapshot taken before any backend work."""
import csv, json, numpy as np, pytest
from helpers import *
def _ref_matrix():
    with open(ART / "reference/train_features.csv") as f: rows = list(csv.DictReader(f))
    return np.array([[float(r[c]) for c in jl(ART / "model_contract.json")["feature_order"]] for r in rows])
def test_preprocessing_parity_all_303_rows(client, patients):
    A = client.app.state.assessor; X = _ref_matrix(); assert X.shape == (303, 54) and A.features == jl(ART / "model_contract.json")["feature_order"]
    diff = max(np.abs(A.encode(p)[0] - X[i]).max() for i, p in enumerate(patients)); assert diff < 1e-9, diff
def test_feature_order_is_frozen(client, approved): assert client.app.state.assessor.features == approved["feature_order"] and len(approved["feature_order"]) == 54
def _independent(art, X):
    from sklearn.preprocessing import StandardScaler; from sklearn.linear_model import LogisticRegression; import xgboost as xgb
    out = {}
    for t in ("CAD", "LAD", "RCA"):
        m = jl(art / f"models/{t}_lr.json"); sc = StandardScaler(); sc.mean_ = np.array(m["scaler_mean"]); sc.scale_ = np.array(m["scaler_scale"]); sc.var_ = sc.scale_ ** 2; sc.n_features_in_ = 54; sc.n_samples_seen_ = 303
        lr = LogisticRegression(C=m["hyperparameters"]["C"]); lr.coef_ = np.array([m["coef"]]); lr.intercept_ = np.array([m["intercept"]]); lr.classes_ = np.array([0, 1]); lr.n_features_in_ = 54
        out[t] = lr.predict_proba(sc.transform(X))[:, 1]
    clf = xgb.XGBClassifier(); clf.load_model(str(art / "models/LCX_xgb.json")); out["LCX"] = clf.predict_proba(X)[:, 1]; return out
def test_prediction_parity_against_independent_implementations(client, patients):
    A = client.app.state.assessor; X = np.array([A.encode(p)[0] for p in patients]); ind = _independent(ART, X); mine = [A.predict_raw(x) for x in X]
    for t in ("CAD", "LAD", "RCA"): assert np.abs(np.array([m[t] for m in mine]) - ind[t]).max() < 1e-9, t
    assert np.abs(np.array([m["LCX"] for m in mine]) - ind["LCX"]).max() < 1e-6
def test_prediction_parity_with_approved_snapshot_unrounded(client, patients, approved_preds):
    A = client.app.state.assessor
    for i, p in enumerate(patients):
        raw = A.predict_raw(A.encode(p)[0]); assert all(abs(raw[t] - approved_preds[i][t]) < 1e-12 for t in raw), i
def test_http_probabilities_equal_approved_snapshot_rounded(all_responses, approved_preds):
    for i, r in enumerate(all_responses):
        for t in ("CAD", "LAD", "LCX", "RCA"): assert r["predictions"][t]["probability"] == round(approved_preds[i][t], 4), (i, t)

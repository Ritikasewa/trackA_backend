"""Guards the review-approved contract: frozen models/inference/preprocessing, warning-only coherence flag, no invented code meanings, reserved null fields."""
import hashlib, json, pytest
from helpers import *
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
@pytest.mark.parametrize("t", ["CAD", "LAD", "LCX", "RCA"])
def test_model_files_are_byte_identical_to_approved(approved, t): assert sha(ART / approved["models"][t]["file"]) == approved["models"][t]["sha256"]
def test_inference_core_is_byte_identical_to_approved(approved): assert sha(BACKEND / "app/core/inference.py") == approved["inference_py_sha256"]
def test_preprocessing_and_reference_data_unchanged(approved):
    assert sha(ART / "preprocessing/training_stats.json") == approved["training_stats_sha256"] and sha(ART / "reference/train_features.csv") == approved["reference_train_features_sha256"]
    assert sha(ART / "tests/golden_responses.json") == approved["golden_responses_sha256"] and sha(ART / "tests/golden_requests.json") == approved["golden_requests_sha256"]
def test_frozen_model_choices_and_hyperparameters():
    T = {t["id"]: t for t in jl(ART / "model_contract.json")["targets"]}
    assert (T["CAD"]["model_type"], T["CAD"]["hyperparameters"]["C"]) == ("logistic_regression", 0.1) and (T["LAD"]["model_type"], T["LAD"]["hyperparameters"]["C"]) == ("logistic_regression", 0.03)
    assert (T["RCA"]["model_type"], T["RCA"]["hyperparameters"]["C"]) == ("logistic_regression", 0.03) and T["LCX"]["model_type"] == "xgboost"
    assert T["LCX"]["hyperparameters"]["n_estimators"] == 100 and T["LCX"]["hyperparameters"]["max_depth"] == 2
    for t in ("CAD", "LAD", "RCA"): assert jl(ART / f"models/{t}_lr.json")["hyperparameters"]["C"] == T[t]["hyperparameters"]["C"]
def _depth(tree):
    left, right = tree["left_children"], tree["right_children"]
    def d(n): return 0 if left[n] == -1 else 1 + max(d(left[n]), d(right[n]))
    return d(0)
def test_lcx_booster_has_100_trees_depth_2():
    trees = jl(ART / "models/LCX_xgb.json")["learner"]["gradient_booster"]["model"]["trees"]; assert len(trees) == 100 and max(_depth(t) for t in trees) == 2
def test_artifact_manifest_still_valid():
    from app.artifacts import verify_manifest; assert verify_manifest(ART)["model_version"] == "v1.0.0"
def test_exactly_50_required_input_fields_no_imputation(schemas):
    req_f = schemas["assess_request"]["properties"]["patient"]["required"]; assert len(req_f) == 50 and set(req_f) == set(FIELD_NAMES) == set(schemas["assess_request"]["properties"]["patient"]["properties"])
    assert jl(ART / "preprocessing/encoding.json")["missing_value_policy"].startswith("none")
def test_numeric_validation_unchanged_no_invented_hard_limits(schemas):
    for f in BY_KIND["number"]:
        s = schemas["assess_request"]["properties"]["patient"]["properties"][f["api"]]; assert "maximum" not in s and ("minimum" in s or "exclusiveMinimum" in s)
def test_uncertainty_and_data_support_null_for_every_patient(all_responses): assert all(r["uncertainty"] is None and r["data_support"] is None for r in all_responses)
def test_disclaimer_and_population_note_in_every_response(all_responses):
    assert all(r["disclaimer"]["required_display"] is True and r["disclaimer"]["short"] and r["population_note"] for r in all_responses)
def test_coherence_check_is_warning_only_and_never_modifies_probabilities(all_responses, approved_preds):
    flagged = [i for i, r in enumerate(all_responses) if r["consistency"]["vessels_exceeding_cad"]]; assert len(flagged) >= 10
    for i, r in enumerate(all_responses):
        P = approved_preds[i]; exp = [v for v in ("LAD", "LCX", "RCA") if P[v] > P["CAD"] + 0.10]; assert r["consistency"]["vessels_exceeding_cad"] == exp and r["consistency"]["probabilities_modified"] is False and r["consistency"]["margin"] == 0.10
        for t in ("CAD", "LAD", "LCX", "RCA"): assert r["predictions"][t]["probability"] == round(P[t], 4)            # no clamping / normalisation
    assert any(r["predictions"][v]["probability"] > r["predictions"]["CAD"]["probability"] for r in all_responses for v in ("LAD", "LCX", "RCA"))   # vessel > CAD really is returned unaltered
def test_rwma_and_function_class_have_no_invented_meanings(schemas):
    F = {f["api"]: f for f in FEATURES}
    for a in ("rwma_region", "function_class"):
        d = F[a]["description"]; assert "not documented" in d and "no meaning is assigned" in d and "ordinal" not in d.lower() and "= none" not in d and F[a]["kind"] == "int_enum" and "enum_labels" not in F[a]
    assert F["rwma_region"]["allowed"] == [0, 1, 2, 3, 4] and F["function_class"]["allowed"] == [0, 1, 2, 3]
    assert "not a lesion location" in F["rwma_region"]["description"]
def test_request_schema_does_not_label_codes(schemas):
    P = schemas["assess_request"]["properties"]["patient"]["properties"]
    for a in ("rwma_region", "function_class"): assert P[a]["type"] == "integer" and "enum" in P[a] and "none" not in P[a]["description"].lower().replace("integer code as recorded", "")
def test_response_schema_labels_margin_heuristic_and_input_checks_not_uncertainty(schemas):
    pr = schemas["assess_response"]["properties"]; assert "NOT a clinical or statistical threshold" in pr["consistency"]["properties"]["margin"]["description"]
    assert "NOT a measure of prediction uncertainty" in pr["input_checks"]["description"] and pr["uncertainty"]["type"] == "null" and pr["data_support"]["type"] == "null"
def test_startup_refuses_tampered_artifacts(tmp_path):
    import shutil; from app.main import create_app
    d = tmp_path / "v1.0.0"; shutil.copytree(ART, d); create_app(d)                         # untouched copy starts fine
    (d / "models/CAD_lr.json").write_text((d / "models/CAD_lr.json").read_text().replace('"intercept"', '"intercept" ', 1))
    with pytest.raises(RuntimeError, match="integrity"): create_app(d)

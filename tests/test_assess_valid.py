from helpers import *
def test_valid_request_returns_four_probabilities(client):
    r = client.post("/assess", json=req()); assert r.status_code == 200 and r.headers["content-type"].startswith("application/json"); b = r.json()
    assert set(b["predictions"]) == {"CAD", "LAD", "LCX", "RCA"}
    for t, p in b["predictions"].items(): assert 0.0 <= p["probability"] <= 1.0 and p["target"] == t and p["definition"]
    assert [b["predictions"][t]["model_type"] for t in ("CAD", "LAD", "LCX", "RCA")] == ["logistic_regression", "logistic_regression", "xgboost", "logistic_regression"]
def test_disclaimer_and_population_note_present(client):
    b = client.post("/assess", json=req()).json()
    assert b["disclaimer"]["required_display"] is True and "not a substitute for formal diagnostic imaging" in b["disclaimer"]["text"] and "decision-support" in b["disclaimer"]["text"]
    assert "Decision support / educational use only" in b["disclaimer"]["short"] and "referred for angiography" in b["population_note"]
def test_request_id_echo_and_optional(client):
    assert client.post("/assess", json=req()).json()["request_id"] == "t"
    r = client.post("/assess", json={"patient": BASE_PATIENT}); assert r.status_code == 200 and r.json()["request_id"] is None
def test_uncertainty_and_data_support_reserved_null(client):
    b = client.post("/assess", json=req()).json(); assert b["uncertainty"] is None and b["data_support"] is None
def test_deterministic(client): assert client.post("/assess", json=req()).json() == client.post("/assess", json=req()).json()
def test_derived_bmi_is_server_side(client):
    b = client.post("/assess", json=req()).json(); assert b["input_checks"]["derived"]["bmi"] == round(BASE_PATIENT["weight_kg"] / (BASE_PATIENT["height_cm"] / 100) ** 2, 2)
def test_out_of_range_is_accepted_and_flagged_as_data_support_not_uncertainty(client):
    p = copy.deepcopy(BASE_PATIENT); p.update(tg=1500, ldl=900); r = client.post("/assess", json=req(p)); assert r.status_code == 200; b = r.json()
    assert b["input_checks"]["extrapolation_flag"] is True and {"tg", "ldl"} <= {x["field"] for x in b["input_checks"]["out_of_observed_range"]}
    assert b["uncertainty"] is None and b["data_support"] is None                      # range flags are NOT represented as uncertainty
    assert "uncertainty" not in b["input_checks"] and "confidence" not in str(b["input_checks"]).lower()
def test_central_98pct_flag_only_is_not_extrapolation(client):
    f = next(x for x in FEATURES if x["api"] == "ldl")["observed"]; p = copy.deepcopy(BASE_PATIENT); p["ldl"] = (f["p99"] + f["max"]) / 2
    b = client.post("/assess", json=req(p)).json(); hit = [x for x in b["input_checks"]["out_of_observed_range"] if x["field"] == "ldl"]
    assert hit and hit[0]["level"] == "outside_central_98pct" and b["input_checks"]["extrapolation_flag"] is False
def test_sparse_category_flagged(client):
    p = copy.deepcopy(BASE_PATIENT); p["congestive_heart_failure"] = True; b = client.post("/assess", json=req(p)).json()
    assert any(x["field"] == "congestive_heart_failure" and x["training_count"] == 1 for x in b["input_checks"]["sparse_categories"])
def test_only_two_documented_endpoints(client):
    assert set(client.get("/openapi.json").json()["paths"]) == {"/assess", "/meta"}; assert client.get("/assess").status_code == 405

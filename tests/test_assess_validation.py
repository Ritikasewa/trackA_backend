import json, pytest
from helpers import *
@pytest.mark.parametrize("field", FIELD_NAMES)
def test_each_required_field_missing(client, field):
    p = copy.deepcopy(BASE_PATIENT); del p[field]; r = client.post("/assess", json=req(p))
    assert r.status_code == 422 and codes(r) == {(f"patient.{field}", "missing_field")} and "predictions" not in r.json()
def test_all_50_fields_required_no_imputation(client):
    r = client.post("/assess", json={"patient": {}}); assert r.status_code == 422 and len(r.json()["errors"]) == 50 and {e["code"] for e in r.json()["errors"]} == {"missing_field"}
    assert len(FIELD_NAMES) == 50
def test_null_is_not_imputed(client):
    p = copy.deepcopy(BASE_PATIENT); p["age"] = None; r = client.post("/assess", json=req(p)); assert r.status_code == 422 and ("patient.age", "invalid_type") in codes(r)
def test_missing_patient_object(client):
    for body in ({}, {"request_id": "x"}, []):
        r = client.post("/assess", json=body); assert r.status_code == 422
    r = client.post("/assess", json={"patient": [1, 2]}); assert r.status_code == 422 and codes(r) == {("patient", "invalid_type")}
def test_malformed_json_body(client):
    r = post_text(client, "{not json"); assert r.status_code == 422 and codes(r) == {("body", "invalid_type")}
def test_unknown_patient_field(client):
    p = copy.deepcopy(BASE_PATIENT); p["favourite_colour"] = "blue"; r = client.post("/assess", json=req(p)); assert r.status_code == 422 and codes(r) == {("patient.favourite_colour", "unknown_field")}
def test_unknown_top_level_field(client):
    r = client.post("/assess", json=req(extra_field=1)); assert r.status_code == 422 and codes(r) == {("extra_field", "unknown_field")}
def test_wrong_case_field_name_is_unknown(client):
    p = copy.deepcopy(BASE_PATIENT); p["Age"] = p.pop("age"); r = client.post("/assess", json=req(p)); assert r.status_code == 422 and {("patient.Age", "unknown_field"), ("patient.age", "missing_field")} == codes(r)
def test_bad_request_id(client):
    for rid in ("", 5, None, "x" * 129): assert client.post("/assess", json={"request_id": rid, "patient": BASE_PATIENT}).status_code == 422
# ---- categorical ----
@pytest.mark.parametrize("f", BY_KIND["enum"], ids=lambda f: f["api"])
@pytest.mark.parametrize("bad", ["invalid", "", None, 1, True, ["x"]])
def test_invalid_enum_values(client, f, bad):
    p = copy.deepcopy(BASE_PATIENT); p[f["api"]] = bad; r = client.post("/assess", json=req(p)); assert r.status_code == 422 and codes(r) == {(f"patient.{f['api']}", "invalid_enum")}
@pytest.mark.parametrize("f", BY_KIND["enum"], ids=lambda f: f["api"])
def test_enum_is_case_sensitive(client, f):
    p = copy.deepcopy(BASE_PATIENT); p[f["api"]] = f["enum"][0].upper(); assert client.post("/assess", json=req(p)).status_code == 422
@pytest.mark.parametrize("f", BY_KIND["enum"], ids=lambda f: f["api"])
def test_every_valid_enum_value_accepted(client, f):
    for v in f["enum"]:
        p = copy.deepcopy(BASE_PATIENT); p[f["api"]] = v; assert client.post("/assess", json=req(p)).status_code == 200, (f["api"], v)
@pytest.mark.parametrize("f", BY_KIND["int_enum"], ids=lambda f: f["api"])
@pytest.mark.parametrize("bad", [-1, 99, 1.5, 2.0, "2", None, True])
def test_invalid_int_enum_values(client, f, bad):
    p = copy.deepcopy(BASE_PATIENT); p[f["api"]] = bad; r = client.post("/assess", json=req(p)); assert r.status_code == 422 and codes(r) == {(f"patient.{f['api']}", "invalid_enum")}
@pytest.mark.parametrize("f", BY_KIND["int_enum"], ids=lambda f: f["api"])
def test_every_valid_int_code_accepted(client, f):
    for v in f["allowed"]:
        p = copy.deepcopy(BASE_PATIENT); p[f["api"]] = v; assert client.post("/assess", json=req(p)).status_code == 200
@pytest.mark.parametrize("f", BY_KIND["bool"], ids=lambda f: f["api"])
@pytest.mark.parametrize("bad", [0, 1, "true", "Y", None, 1.0])
def test_booleans_must_be_json_booleans(client, f, bad):
    p = copy.deepcopy(BASE_PATIENT); p[f["api"]] = bad; r = client.post("/assess", json=req(p)); assert r.status_code == 422 and codes(r) == {(f"patient.{f['api']}", "invalid_type")}
# ---- numeric ----
@pytest.mark.parametrize("f", BY_KIND["number"], ids=lambda f: f["api"])
@pytest.mark.parametrize("bad", ["58", True, False, None, [], {}])
def test_numeric_wrong_type(client, f, bad):
    p = copy.deepcopy(BASE_PATIENT); p[f["api"]] = bad; r = client.post("/assess", json=req(p)); assert r.status_code == 422 and codes(r) == {(f"patient.{f['api']}", "invalid_type")}
@pytest.mark.parametrize("f", BY_KIND["number"], ids=lambda f: f["api"])
@pytest.mark.parametrize("tok", ["NaN", "Infinity", "-Infinity"])
def test_numeric_not_finite(client, f, tok):
    p = copy.deepcopy(BASE_PATIENT); p[f["api"]] = 0; txt = json.dumps(req(p)).replace(f'"{f["api"]}": 0,', f'"{f["api"]}": {tok},').replace(f'"{f["api"]}": 0}}', f'"{f["api"]}": {tok}}}')
    assert tok in txt; r = post_text(client, txt); assert r.status_code == 422 and codes(r) == {(f"patient.{f['api']}", "not_finite")}
@pytest.mark.parametrize("f", BY_KIND["number"], ids=lambda f: f["api"])
def test_numeric_negative_rejected(client, f):
    p = copy.deepcopy(BASE_PATIENT); p[f["api"]] = -1; r = client.post("/assess", json=req(p)); assert r.status_code == 422 and codes(r) == {(f"patient.{f['api']}", "invalid_value")}
@pytest.mark.parametrize("field", ["weight_kg", "height_cm"])
def test_zero_weight_or_height_rejected(client, field):
    p = copy.deepcopy(BASE_PATIENT); p[field] = 0; r = client.post("/assess", json=req(p)); assert r.status_code == 422 and codes(r) == {(f"patient.{field}", "invalid_value")}
def test_validation_errors_follow_error_schema(client, schemas):
    import jsonschema
    p = copy.deepcopy(BASE_PATIENT); p.pop("age"); p["sex"] = "x"; p["tg"] = -1; p["LAD"] = True; r = client.post("/assess", json=req(p)); assert r.status_code == 422
    assert not list(jsonschema.Draft202012Validator(schemas["error"]).iter_errors(r.json())) and len(r.json()["errors"]) == 4

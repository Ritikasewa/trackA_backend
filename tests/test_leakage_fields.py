import pytest, jsonschema
from helpers import *
FORBIDDEN = ["LAD", "LCX", "RCA", "Cath", "CAD"]; DERIVED = ["bmi", "obesity"]
@pytest.mark.parametrize("name", FORBIDDEN)
@pytest.mark.parametrize("value", [True, 1, 0, "Stenotic", "CAD", 0.5, None])
def test_target_labels_rejected_as_inputs(client, name, value):
    p = copy.deepcopy(BASE_PATIENT); p[name] = value; r = client.post("/assess", json=req(p))
    assert r.status_code == 422 and codes(r) == {(f"patient.{name}", "forbidden_target_field")} and "predictions" not in r.json()
@pytest.mark.parametrize("name", DERIVED)
@pytest.mark.parametrize("value", [True, 25.0, "Y", 0, None])
def test_derived_fields_rejected_as_inputs(client, name, value):
    p = copy.deepcopy(BASE_PATIENT); p[name] = value; r = client.post("/assess", json=req(p))
    assert r.status_code == 422 and codes(r) == {(f"patient.{name}", "derived_field_not_accepted")} and "predictions" not in r.json()
@pytest.mark.parametrize("name", ["lad", "Lcx", "rca", "CATH", "cath", "cad", "Bmi", "BMI", "OBESITY", "Obesity"])
def test_case_variants_also_rejected(client, name):
    p = copy.deepcopy(BASE_PATIENT); p[name] = 1; r = client.post("/assess", json=req(p)); assert r.status_code == 422
    assert {c for _, c in codes(r)} <= {"forbidden_target_field", "derived_field_not_accepted"} and len(r.json()["errors"]) == 1
def test_all_seven_at_once(client):
    p = copy.deepcopy(BASE_PATIENT); p.update({n: 1 for n in FORBIDDEN + DERIVED}); r = client.post("/assess", json=req(p)); assert r.status_code == 422 and len(r.json()["errors"]) == 7
@pytest.mark.parametrize("name", FORBIDDEN + DERIVED)
def test_top_level_placement_also_rejected(client, name): assert client.post("/assess", json=req(**{name: 1})).status_code == 422
@pytest.mark.parametrize("name", FORBIDDEN + DERIVED)
def test_json_schema_also_rejects(schemas, name):
    p = copy.deepcopy(BASE_PATIENT); p[name] = 1; assert list(jsonschema.Draft202012Validator(schemas["assess_request"]).iter_errors(req(p)))
def test_no_target_or_derived_field_in_published_input_schema(schemas):
    props = set(schemas["assess_request"]["properties"]["patient"]["properties"]); assert not ({n.lower() for n in FORBIDDEN + DERIVED} & {p.lower() for p in props})

import jsonschema, pytest
from helpers import *
GOLD = jl(ART / "tests/golden_responses.json")
@pytest.mark.parametrize("name", list(GOLD))
def test_golden_request_response_exact(client, name):
    r = client.post("/assess", json=GOLDEN_REQ[name]); assert r.status_code == 200 and r.json() == GOLD[name]
def test_golden_set_covers_low_median_high_and_out_of_range(): assert set(GOLD) == {"lowest_cad_row", "median_cad_row", "highest_cad_row", "out_of_range_example"}
def test_response_schema_is_valid_draft_2020_12(schemas):
    for s in schemas.values(): jsonschema.Draft202012Validator.check_schema(s)
@pytest.mark.parametrize("name", list(GOLD))
def test_golden_responses_validate_against_schema(client, schemas, name):
    assert not list(jsonschema.Draft202012Validator(schemas["assess_response"]).iter_errors(client.post("/assess", json=GOLDEN_REQ[name]).json()))
def test_all_303_responses_validate_against_response_schema(all_responses, schemas):
    v = jsonschema.Draft202012Validator(schemas["assess_response"]); bad = [i for i, r in enumerate(all_responses) if list(v.iter_errors(r))]; assert not bad, bad[:5]
def test_all_303_requests_validate_against_request_schema(patients, schemas):
    v = jsonschema.Draft202012Validator(schemas["assess_request"]); assert not [i for i, p in enumerate(patients) if list(v.iter_errors({"patient": p}))]
def test_openapi_publishes_contract_schemas(client, schemas):
    op = client.get("/openapi.json").json()["paths"]["/assess"]["post"]; rb = op["requestBody"]["content"]["application/json"]["schema"]
    assert len(rb["properties"]["patient"]["required"]) == 50 and rb["additionalProperties"] is False and "422" in op["responses"] and "200" in op["responses"]

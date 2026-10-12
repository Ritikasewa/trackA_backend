import json, copy, pathlib
BACKEND = pathlib.Path(__file__).resolve().parent.parent
ART = BACKEND / "artifacts" / "v1.0.0"
TESTS = pathlib.Path(__file__).resolve().parent
def jl(p): return json.loads(pathlib.Path(p).read_text())
FEATURES = jl(ART / "schema/features.json")["fields"]
FIELD_NAMES = [f["api"] for f in FEATURES]
BY_KIND = {k: [f for f in FEATURES if f["kind"] == k] for k in ("bool", "number", "int_enum", "enum")}
GOLDEN_REQ = jl(ART / "tests/golden_requests.json")
BASE_PATIENT = GOLDEN_REQ["median_cad_row"]["patient"]
def req(patient=None, **extra): return {"request_id": "t", "patient": copy.deepcopy(BASE_PATIENT if patient is None else patient), **extra}
def post_text(client, text): return client.post("/assess", content=text, headers={"content-type": "application/json"})   # allows NaN / Infinity tokens
def codes(resp): return {(e["field"], e["code"]) for e in resp.json()["errors"]}

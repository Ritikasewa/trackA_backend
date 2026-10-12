import sys, pathlib, warnings
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent)); sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import pytest
warnings.filterwarnings("ignore", message=".*httpx.*")
from fastapi.testclient import TestClient
from app.main import create_app
from helpers import *
@pytest.fixture(scope="session")
def client():
    with TestClient(create_app()) as c: yield c
@pytest.fixture(scope="session")
def patients(): return jl(TESTS / "fixtures/patients_303.json")
@pytest.fixture(scope="session")
def approved_preds(): return jl(TESTS / "frozen/approved_predictions_303.json")
@pytest.fixture(scope="session")
def approved(): return jl(TESTS / "frozen/approved_hashes.json")
@pytest.fixture(scope="session")
def schemas(): return {n: jl(ART / f"schema/{n}.schema.json") for n in ("assess_request", "assess_response", "error")}
@pytest.fixture(scope="session")
def all_responses(client, patients):
    out = []
    for i, p in enumerate(patients):
        r = client.post("/assess", json={"request_id": f"row{i}", "patient": p}); assert r.status_code == 200, (i, r.text); out.append(r.json())
    return out

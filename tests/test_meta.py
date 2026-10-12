from helpers import *
def test_meta_ok_and_shape(client):
    r = client.get("/meta"); assert r.status_code == 200; b = r.json()
    assert {"contract_version", "model_version", "disclaimer", "population_note", "targets", "performance", "limitations", "consistency_warning", "input_policy", "training"} <= set(b)
    assert [t["id"] for t in b["targets"]] == ["CAD", "LAD", "LCX", "RCA"] and set(b["performance"]) == {"CAD", "LAD", "LCX", "RCA"}
def test_meta_performance_is_unchanged_approved_values(client):
    assert client.get("/meta").json()["performance"] == jl(TESTS / "frozen/approved_performance_summary.json")
def test_meta_matches_model_card_verbatim(client):
    card = jl(ART / "model_card.json"); b = client.get("/meta").json(); assert b["performance"] == card["performance_summary"] and b["limitations"] == card["limitations"]
def test_meta_limitations_mention_weak_vessels_and_single_centre(client):
    L = " ".join(client.get("/meta").json()["limitations"]); assert "LCX/RCA" in L and "Single-centre" in L and "external validation" in L
def test_meta_has_disclaimer_population_and_heuristic_margin(client):
    b = client.get("/meta").json(); assert b["disclaimer"]["required_display"] is True and "referred for angiography" in b["population_note"]
    assert b["consistency_warning"]["margin"] == 0.10 and "Not a clinical or statistical threshold" in b["consistency_warning"]["description"]
def test_meta_lcx_rca_tier_is_limited(client):
    p = client.get("/meta").json()["performance"]; assert p["LCX"]["discrimination_tier"] == "limited" and p["RCA"]["discrimination_tier"] == "limited" and p["CAD"]["discrimination_tier"] == "high"

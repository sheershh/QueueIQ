import pytest
from fastapi.testclient import TestClient

from queueiq.api import main
from tests.service_data import synthetic_service

PROJECT = {"iso": "PJM", "state": "VA", "fuel": "solar", "capacity_mw": 120, "storage": True}


@pytest.fixture(scope="module")
def client():
    main.set_service(synthetic_service())
    with TestClient(main.app) as c:
        yield c
    main.set_service(None)


def test_health_and_options(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"
    opts = client.get("/options").json()
    assert opts["states"]["PJM"] == ["VA"] and opts["max_compare"] == 5
    assert "evaluated_months" in client.get("/context").json()


def test_predict_with_and_without_explanation(client):
    r = client.post("/predict", json=PROJECT)
    assert r.status_code == 200
    body = r.json()
    assert {"curves", "headline", "months", "project"} <= set(body) and "explanation" not in body
    r = client.post("/predict?explain=true", json=PROJECT)
    assert set(r.json()["explanation"]["targets"]) == {"withdrawn_3y", "operational_5y"}


def test_invalid_inputs_return_422_with_reasons(client):
    assert client.post("/predict", json={**PROJECT, "capacity_mw": -5}).status_code == 422
    r = client.post("/predict", json={**PROJECT, "state": "IL"})
    assert r.status_code == 422 and "not offered" in r.json()["detail"][0]


def test_compare(client):
    r = client.post("/compare", json={"projects": [PROJECT, {**PROJECT, "fuel": "gas"}]})
    assert r.status_code == 200 and len(r.json()["projects"]) == 2
    assert client.post("/compare", json={"projects": [PROJECT] * 6}).status_code == 422
    assert client.post("/compare", json={"projects": []}).status_code == 422


def test_health_reports_missing_model():
    saved = dict(main._state)
    main.set_service(None, "Missing models/deploy_primary.joblib. Run `make data && make train`.")
    try:
        c = TestClient(main.app)
        r = c.get("/health")
        assert r.status_code == 503 and "make train" in r.json()["detail"]
        assert c.post("/predict", json=PROJECT).status_code == 503
    finally:
        main.set_service(saved["service"], saved["error"])

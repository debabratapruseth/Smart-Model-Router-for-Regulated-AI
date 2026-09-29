import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def client(router):
    with TestClient(app) as client:
        app.state.router = router
        yield client


def test_api_routes(client, request_factory):
    assert client.get("/health").status_code == 200
    assert len(client.get("/models").json()["models"]) == 12
    assert client.get("/policies").json()["version"]
    request = request_factory().model_dump(mode="json")
    response = client.post("/route", json=request)
    assert response.status_code == 200
    assert response.json()["status"] == "ROUTED"
    assert client.get("/audit/TEST-001").status_code == 200
    assert client.get("/audit/unknown").status_code == 404
    assert client.post("/route/compare", json=request).json()["llm"]["status"] == "UNAVAILABLE"


def test_invalid_metadata(client):
    assert client.post("/route", json={"prompt": "hello"}).status_code == 422
    assert client.post("/route", json={"prompt": "hello", "metadata": {"data_classification": "unknown"}}).status_code == 422


def test_sample_benchmark(client):
    response = client.post("/benchmark/sample", json={"size": 5})
    assert response.status_code == 200
    assert len(response.json()) == 4
    assert all(row["policy_violation_rate"] == 0 for row in response.json())
    assert client.post("/benchmark/sample", json={"size": 10000}).status_code == 422

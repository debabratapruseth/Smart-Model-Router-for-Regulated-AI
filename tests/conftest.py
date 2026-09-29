import pytest
from app.core.router import ModelRouter
from app.models.request import RoutingRequest


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("JEV_MOCK_MODE", "true")
    monkeypatch.setenv("ENABLE_MODEL_EXECUTION", "false")


@pytest.fixture
def router(tmp_path):
    return ModelRouter(runtime_dir=tmp_path)


@pytest.fixture
def request_factory():
    def make(prompt="Summarize a fictional bank report.", **metadata):
        return RoutingRequest(request_id="TEST-001", prompt=prompt, metadata={"data_classification": "PUBLIC", **metadata})
    return make

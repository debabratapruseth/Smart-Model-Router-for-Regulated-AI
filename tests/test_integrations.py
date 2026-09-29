"""External contracts tested with local mocks; no keys or outbound calls needed."""
import json
import httpx
import pytest
from app.services.execution_service import ExecutionService
from app.core.features import extract_features
from app.models.request import normalize_request
from app.strategies.base import StrategyUnavailable


class FakeResponse:
    def __init__(self, content):
        self.content = content

    def raise_for_status(self):
        return None

    def json(self):
        return {"choices": [{"message": {"content": self.content}}]}


def test_application_llm_cannot_bypass_live_permission(router, request_factory, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-placeholder-not-a-secret")
    monkeypatch.setenv("OPENAI_ROUTER_MODEL", "test-model")
    monkeypatch.setenv("OPENAI_ROUTER_ENABLED", "true")
    def forbidden(*args, **kwargs):
        raise AssertionError("Only the explicit live runner may call providers")
    monkeypatch.setattr(httpx, "post", forbidden)
    assert router.route(request_factory(), "llm", allow_fallback=False).status == "UNAVAILABLE"
    assert router.route(request_factory(), "llm").fallback_used


def test_live_jev_is_explicit_stub(router, request_factory, monkeypatch):
    monkeypatch.setenv("JEV_MOCK_MODE", "false")
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-placeholder")
    assert router.route(request_factory(), "jev", allow_fallback=False).status == "UNAVAILABLE"


def test_execution_requires_exact_audit(router, request_factory, monkeypatch):
    monkeypatch.setenv("ENABLE_MODEL_EXECUTION", "true")
    router.config["execute_selected_model"] = True
    request = request_factory()
    result = router.route(request)
    request.prompt = "Changed business input"
    with pytest.raises(ValueError, match="differs"):
        ExecutionService(router).execute(request, result)


def test_execution_mock_measures_quality_and_cost(router, request_factory, monkeypatch):
    monkeypatch.setenv("ENABLE_MODEL_EXECUTION", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "test-placeholder")
    router.config["execute_selected_model"] = True
    model = next(m for m in router.models if m.model_id == "reasoning")
    model.executable_model = "approved-test-model"
    request = request_factory(prompt="Classify synthetic payment as LOW or HIGH", allowed_models=["reasoning"])
    result = router.route(request)

    class ExecutionResponse(FakeResponse):
        def json(self):
            return {"choices": [{"message": {"content": "LOW"}}], "usage": {"prompt_tokens": 100, "completion_tokens": 2}}

    monkeypatch.setattr(httpx, "post", lambda *a, **k: ExecutionResponse(""))
    # Redirect execution output so tests do not contaminate research data.
    from app.services import execution_service
    monkeypatch.setattr(execution_service, "ROOT", router.runtime_dir)
    row = ExecutionService(router).execute(request, result, "LOW")
    assert row["success"] is True
    assert row["quality_score"] == 1
    assert row["input_tokens"] == 100
    assert row["estimated_cost_usd"] == pytest.approx(.000318)

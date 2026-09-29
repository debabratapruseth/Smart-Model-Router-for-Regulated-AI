import pytest
from app.strategies.base import StrategyResult, StrategyUnavailable
from app.models.decision import RoutingResponse
from app.core.scoring import estimated_cost, task_quality
from app.models.request import normalize_request
from app.core.features import extract_features
from app.models.telemetry import TelemetryRecord


@pytest.mark.parametrize("strategy", ["rules", "weighted", "frontier_only", "cheapest_only", "jev"])
def test_local_strategies(router, request_factory, strategy):
    response = router.route(request_factory(), strategy)
    assert response.status == "ROUTED"
    assert response.decision.selected_model in response.eligible_models
    assert RoutingResponse.model_validate_json(response.model_dump_json()) == response


def test_baselines(router, request_factory):
    request = normalize_request(request_factory())
    features = extract_features(request, router.config)
    eligible, _ = router.policy.filter(request, features, router.models)
    frontier = router.route(request, "frontier_only")
    cheapest = router.route(request, "cheapest_only")
    selected_frontier = next(m for m in eligible if m.model_id == frontier.decision.selected_model)
    selected_cheapest = next(m for m in eligible if m.model_id == cheapest.decision.selected_model)
    assert task_quality(selected_frontier, features, router.config) == max(task_quality(m, features, router.config) for m in eligible)
    assert estimated_cost(selected_cheapest, request) == min(estimated_cost(m, request) for m in eligible)


def test_unavailable_fallback(router, request_factory):
    response = router.route(request_factory(), "llm")
    assert response.fallback_used
    assert response.decision.routing_strategy == "weighted"
    compared = router.compare(request_factory())
    assert compared["llm"].status == "UNAVAILABLE"
    assert compared["ml"].status == "UNAVAILABLE"


def test_invalid_decision_never_fallback(router, request_factory, monkeypatch):
    monkeypatch.setattr(router.strategies["weighted"], "route", lambda *args: StrategyResult(selected_model="prohibited", confidence=.1))
    result = router.route(request_factory())
    assert result.status == "INVALID_DECISION"
    assert result.decision is None
    assert not result.fallback_used
    assert router.audit.replay("TEST-001")[0]["response"]["status"] == "INVALID_DECISION"


def test_low_confidence_fallback(router, request_factory, monkeypatch):
    monkeypatch.setattr(router.strategies["jev"], "route", lambda *args: StrategyResult(selected_model="enterprise", confidence=.1))
    result = router.route(request_factory(), "jev")
    assert result.fallback_used
    assert "LOW_CONFIDENCE" in result.reason_codes


def test_human_review(router, request_factory):
    router.config["human_review_threshold"] = 1
    assert router.route(request_factory()).human_review_required


def test_resilience_chooses_compliant_alternative(router, request_factory):
    request = request_factory(data_classification="RESTRICTED")
    first = router.route(request).decision.selected_model
    next(m for m in router.models if m.model_id == first).available = False
    result = router.route(request)
    assert result.status == "ROUTED"
    assert result.decision.selected_model != first
    assert all(m.provider == "INTERNAL" for m in router.models if m.model_id in result.eligible_models)


def test_dynamic_telemetry_rechecks_sla(router, request_factory):
    router.config["use_dynamic_telemetry"] = True
    router.telemetry.append(TelemetryRecord(model_id="enterprise", routing_strategy="test", execution_latency_ms=9000, success=True))
    result = router.route(request_factory(allowed_models=["enterprise"], latency_sla_ms=2000))
    assert result.status == "NO_ROUTE"


def test_dynamic_unavailable(router, request_factory):
    router.config["use_dynamic_telemetry"] = True
    router.telemetry.append(TelemetryRecord(model_id="enterprise", routing_strategy="test", availability_status="UNAVAILABLE"))
    assert router.route(request_factory(allowed_models=["enterprise"])).status == "NO_ROUTE"


def test_normalization_cannot_shrink_declared_context(request_factory):
    request = request_factory(context_tokens=10000)
    assert normalize_request(request).metadata.context_tokens == 10000
    assert normalize_request(request_factory(prompt="a" * 9000, context_tokens=1)).metadata.context_tokens == 9000

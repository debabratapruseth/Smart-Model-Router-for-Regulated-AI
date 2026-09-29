import json
import numpy as np
import pytest
from app.core.config_loader import load_config
from app.models.request import RoutingRequest
from app.services.quality_service import evaluate_quality
from app.services.execution_service import ExecutionService
from app.strategies.base import StrategyResult
from evaluation.metrics import policy_violation
from evaluation.benchmark import run_benchmark, train_ml
from evaluation.statistical_tests import bootstrap_ci
from evaluation.business_case import calculate
from scripts.generate_banking_benchmark import generate, ground_truth


def test_reproducible_data():
    assert generate(20) == generate(20)
    assert generate(20, seed=1) != generate(20, seed=2)


def test_benchmark_denominators(router):
    rows = generate(10)
    frame, summary = run_benchmark(rows, ground_truth(rows), router, ["weighted", "llm"], write_reports=False)
    assert len(frame) == 20
    assert not frame.policy_violation.any()
    assert frame.quality_score.isna().all()
    assert frame[frame.status == "NO_ROUTE"].estimated_cost_usd.isna().all()


def test_violation_metric_detects_bad_selection(router, request_factory):
    request = request_factory(data_classification="RESTRICTED")
    response = router.route(request)
    response.decision.selected_model = "economy"
    assert policy_violation(request, response, router)


def test_ml_training_is_optional(router, request_factory):
    assert not router.strategies["ml"].available()
    train_ml(router)
    result = router.route(request_factory(), "ml", allow_fallback=False)
    assert result.status in {"ROUTED", "UNAVAILABLE"}
    if result.decision:
        assert result.decision.selected_model in result.eligible_models


def test_quality_metrics():
    assert evaluate_quality("CLASSIFICATION", " LOW ", "low")["quality_score"] == 1
    assert evaluate_quality("EXTRACTION", '{"a":1}', {"a":1, "b":2})["quality_score"] == .5
    assert evaluate_quality("EXTRACTION", '[]', {"a":1})["quality_score"] == 0
    assert evaluate_quality("REASONING", "answer")["quality_score"] is None
    with pytest.raises(ValueError):
        evaluate_quality("TRANSLATION", "x", manual_score=2)


def test_execution_off(router, request_factory):
    request = request_factory()
    assert ExecutionService(router).execute(request, router.route(request))["status"] == "DISABLED"


def test_statistics():
    assert bootstrap_ci([1, 1, 1]) == (1, 1)
    assert bootstrap_ci([0, 1, 2]) == bootstrap_ci([0, 1, 2])


def test_business_case():
    assert calculate(100, 2, 1)["estimated_savings"] == 100
    assert calculate(100, 0, 0)["percentage_reduction"] is None


def test_ablation_never_relaxes_policy(router, request_factory):
    request = request_factory(context_tokens=2000000)
    assert router.route(request, ranking_ablation="context_length").status == "NO_ROUTE"
    request = request_factory(requires_rag=True, allowed_models=["local-small"])
    assert router.route(request, ranking_ablation="enterprise_knowledge").status == "NO_ROUTE"

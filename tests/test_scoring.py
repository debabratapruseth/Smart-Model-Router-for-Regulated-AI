import pytest
from app.core.scoring import benefit, estimated_cost
from app.core.config_loader import load_config, validate_routing
from app.models.request import normalize_request


def test_equal_values():
    assert benefit(4, [4, 4]) == 1
    assert benefit(5, [1, 5]) == 0


def test_weighted_breakdown(router, request_factory):
    response = router.route(request_factory())
    assert set(response.score_breakdown) == {"quality", "task_fit", "latency", "cost", "reliability"}
    weights = router.config["routing_preferences"]["balanced"]
    score = sum(weights[key] * value for key, value in response.score_breakdown.items())
    assert response.decision.confidence == pytest.approx(score)
    assert all(0 <= value <= 1 for parts in response.candidate_scores.values() for value in parts.values())


def test_cost_includes_output(router, request_factory):
    request = normalize_request(request_factory(context_tokens=1000, output_tokens=500))
    model = next(m for m in router.models if m.model_id == "frontier")
    assert estimated_cost(model, request) == pytest.approx(.015)


def test_bad_weights_fail():
    config = load_config("routing")
    config["routing_preferences"]["balanced"]["cost"] = -1
    with pytest.raises(ValueError):
        validate_routing(config)

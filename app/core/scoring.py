"""Transparent utility scores. A score is not a calibrated probability."""
from app.models.model_catalog import ModelSpec
from app.models.request import RoutingRequest


def estimated_cost(model: ModelSpec, request: RoutingRequest) -> float:
    rates = model.economics
    return (request.metadata.context_tokens * rates.input_cost_per_1m_tokens +
            request.metadata.output_tokens * rates.output_cost_per_1m_tokens) / 1_000_000


def task_quality(model: ModelSpec, features: dict, config: dict) -> float:
    fields = config["quality_fields"]
    return getattr(model.quality, fields.get(features["task_type"], fields["default"]))


def benefit(value: float, values: list[float]) -> float:
    """Lower is better; equal alternatives all get full utility."""
    low, high = min(values), max(values)
    return 1.0 if high == low else (high - value) / (high - low)


def scoring_breakdowns(request: RoutingRequest, features: dict, models: list[ModelSpec], config: dict) -> dict:
    costs = [estimated_cost(m, request) for m in models]
    latencies = [m.performance.p95_latency_ms for m in models]
    result = {}
    for model, cost, latency in zip(models, costs, latencies):
        desired = []
        if features["reasoning_need"] == "HIGH" or features["complexity"] == "HIGH":
            desired.append(model.capabilities.reasoning)
        if features["enterprise_knowledge_need"] == "HIGH":
            desired.append(model.capabilities.enterprise_knowledge)
        if features["task_type"] == "CODING":
            desired.append(model.capabilities.coding)
        result[model.model_id] = {
            "quality": task_quality(model, features, config),
            "task_fit": sum(desired) / len(desired) if desired else 1.0,
            "latency": benefit(latency, latencies), "cost": benefit(cost, costs),
            "reliability": model.performance.success_rate,
        }
    return result

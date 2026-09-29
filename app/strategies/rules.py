from app.strategies.base import RoutingStrategy
from app.strategies.cheapest_only import CheapestOnlyRouter
from app.core.scoring import benefit, task_quality


class RuleBasedRouter(RoutingStrategy):
    def route(self, request, features, candidate_models):
        metadata = request.metadata
        if metadata.cost_preference == "MINIMIZE_COST":
            return CheapestOnlyRouter(self.config).route(request, features, candidate_models)
        latency_rule = (metadata.cost_preference == "LATENCY_FIRST" or
                        (metadata.latency_sla_ms is not None and metadata.latency_sla_ms <= self.config["rule_thresholds"]["low_latency_ms"]))
        latencies = [m.performance.p95_latency_ms for m in candidate_models]
        scores = {}
        for model in candidate_models:
            score = task_quality(model, features, self.config)
            if latency_rule:
                score = benefit(model.performance.p95_latency_ms, latencies)
            elif features["reasoning_need"] == "HIGH" and not model.capabilities.reasoning:
                score *= .5
            elif metadata.context_tokens >= self.config["rule_thresholds"]["large_context"]:
                score = model.limits.context_window / max(m.limits.context_window for m in candidate_models)
            scores[model.model_id] = score
        return self.choose(scores, candidate_models)

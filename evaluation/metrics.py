"""Explicit denominators prevent abstentions from looking like free good answers."""
import numpy as np
import pandas as pd
from app.core.features import extract_features
from app.models.request import normalize_request

HYPOTHESES = {
    "H1": "Intelligent routing reduces inference cost relative to strongest eligible routing.",
    "H2": "Intelligent routing reduces latency without material downstream quality degradation.",
    "H3": "Policy-first routing yields zero policy violations on the benchmark workload.",
    "H4": "A real bounded-decision Jev router reduces routing latency and/or cost vs a general LLM router.",
    "H5": "Observed telemetry improves resilience relative to static routing under model changes.",
}


def policy_violation(request, response, router) -> bool:
    if response.decision is None:
        return False
    request = normalize_request(request)
    selected = next((m for m in router.current_models() if m.model_id == response.decision.selected_model), None)
    return selected is None or router.policy.evaluate_model(request, extract_features(request, router.config), selected) is not None


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    for name, group in frame.groupby("strategy", sort=False):
        active = group[group.status != "UNAVAILABLE"]
        routed = active[active.status == "ROUTED"]
        n, nr = len(active), len(routed)
        metadata = {key: group[key].iloc[0] for key in ["benchmark_type", "evidence_type", "execution_mode", "dataset_hash",
                    "run_id", "timestamp", "strategy_version", "input_mode", "phase"] if key in group}
        if "research_eligible" in group:
            metadata["research_eligible"] = bool(group.research_eligible.any())
        summaries.append({**metadata,
            "correct_no_route": int(((active.status == "NO_ROUTE") & active.expected_no_route).sum()),
            "preferred_reference_route_agreement_including_correct_abstentions": active.preferred_correct.mean() if n else np.nan,
            "acceptable_reference_route_agreement_including_correct_abstentions": active.acceptable_correct.mean() if n else np.nan,
            "preferred_reference_route_agreement_routed_only": routed.preferred_correct.mean() if nr else np.nan,
            "acceptable_reference_route_agreement_routed_only": routed.acceptable_correct.mean() if nr else np.nan,
            "p50_routing_latency_ms": active.routing_latency_ms.median() if n else np.nan,"strategy": name, "mode": group["mode"].iloc[0], "total_requests": len(group),
            "available_requests": n, "routed_requests": nr, "unavailable_rate": (group.status == "UNAVAILABLE").mean(),
            "preferred_model_accuracy": active.preferred_correct.mean() if n else np.nan,
            "acceptable_route_accuracy": active.acceptable_correct.mean() if n else np.nan,
            "policy_violation_rate": active.policy_violation.mean() if n else np.nan,
            "policy_violation_rate_routed": routed.policy_violation.mean() if nr else np.nan,
            "no_route_rate": (active.status == "NO_ROUTE").mean() if n else np.nan,
            "invalid_decision_rate": (active.status == "INVALID_DECISION").mean() if n else np.nan,
            "fallback_rate": active.fallback_used.mean() if n else np.nan,
            "average_routing_latency_ms": active.routing_latency_ms.mean(),
            "p95_routing_latency_ms": active.routing_latency_ms.quantile(.95),
            "average_estimated_model_cost_usd": routed.estimated_cost_usd.mean(),
            "average_model_latency_ms": routed.model_latency_ms.mean(),
            "synthetic_quality_proxy": routed.synthetic_quality_proxy.mean(),
            "downstream_quality": routed.quality_score.mean(),
            "quality_evaluated_count": routed.quality_score.notna().sum(),
            "frontier_model_usage": (routed.tier == "frontier").mean() if nr else np.nan,
            "small_model_usage": (routed.tier == "small").mean() if nr else np.nan})
    return pd.DataFrame(summaries)

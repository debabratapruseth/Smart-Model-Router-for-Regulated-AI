"""Compare stale static metrics with fresh observed telemetry after a shock.

Policy compliance against each router's known state and suitability against the
simulated world are separate measurements. Unknown outages are not a governance
bypass, and no-route is legitimate when every alternative is prohibited.
"""
import tempfile
from pathlib import Path
import pandas as pd
from app.core.config_loader import ROOT, load_config
from app.core.router import ModelRouter
from app.models.request import RoutingRequest, normalize_request
from app.models.telemetry import TelemetryRecord
from app.core.features import extract_features
from app.core.scoring import estimated_cost
from evaluation.metrics import policy_violation
from scripts.generate_banking_benchmark import generate

SCENARIOS = ["model_unavailable", "latency_doubles", "cost_doubles", "provider_unavailable",
             "rate_limit", "region_unavailable", "failure_rate_rises"]


def run_resilience(size=None):
    size = size or load_config("evaluation")["resilience_sample_size"]
    rows = []
    with tempfile.TemporaryDirectory() as directory:
        baseline = ModelRouter(runtime_dir=Path(directory) / "base")
        for index, item in enumerate(generate(size)):
            request = RoutingRequest.model_validate(item["request"])
            before = baseline.route(request)
            if before.decision is None:
                continue
            chosen = before.decision.selected_model
            for scenario in SCENARIOS:
                for adaptive in [False, True]:
                    router = ModelRouter(runtime_dir=Path(directory) / f"{index}-{scenario}-{adaptive}")
                    router.config["use_dynamic_telemetry"] = adaptive
                    world = [m.model_copy(deep=True) for m in router.models]
                    target = next(m for m in world if m.model_id == chosen)
                    affected = [m for m in world if
                                (m.provider == target.provider if scenario == "provider_unavailable" else
                                 m.region == target.region if scenario == "region_unavailable" else m.model_id == chosen)]
                    for model in affected:
                        if scenario in ["model_unavailable", "provider_unavailable", "rate_limit", "region_unavailable"]:
                            model.available = False
                        elif scenario == "latency_doubles":
                            model.performance.p95_latency_ms *= 2
                        elif scenario == "cost_doubles":
                            model.economics.input_cost_per_1m_tokens *= 2
                            model.economics.output_cost_per_1m_tokens *= 2
                            # Billing rates are a published catalog update, available to BOTH strategies.
                            configured = next(m for m in router.models if m.model_id == model.model_id)
                            configured.economics = model.economics.model_copy(deep=True)
                        elif scenario == "failure_rate_rises":
                            model.performance.success_rate = .5
                        for observation in range(10):
                            router.telemetry.append(TelemetryRecord(model_id=model.model_id, routing_strategy="shock",
                                execution_latency_ms=model.performance.p95_latency_ms,
                                success=observation % 2 == 0 if scenario == "failure_rate_rises" else None,
                                availability_status="AVAILABLE" if model.available else "UNAVAILABLE"))
                    after = router.route(request)
                    selected = next((m for m in world if after.decision and m.model_id == after.decision.selected_model), None)
                    normalized = normalize_request(request)
                    features = extract_features(normalized, router.config)
                    known_compliant = not policy_violation(request, after, router)
                    world_eligible, _ = router.policy.filter(normalized, features, world)
                    rows.append({"request_id": request.request_id, "scenario": scenario, "adaptive": adaptive,
                        "original_model": chosen, "selected_model": selected.model_id if selected else None, "status": after.status,
                        "switched": selected is not None and selected.model_id != chosen,
                        "policy_violation": not known_compliant,
                        "world_compliant_route_or_correct_abstention": (router.policy.evaluate_model(normalized, features, selected) is None) if selected else not world_eligible,
                        "actual_model_latency_ms": selected.performance.p95_latency_ms if selected else None,
                        "actual_estimated_cost_usd": estimated_cost(selected, normalized) if selected else None})
    result = pd.DataFrame(rows)
    result.to_csv(ROOT / "reports/resilience_results.csv", index=False)
    summary = result.groupby(["scenario", "adaptive"])[["switched", "policy_violation", "world_compliant_route_or_correct_abstention", "actual_model_latency_ms", "actual_estimated_cost_usd"]].mean()
    summary.to_csv(ROOT / "reports/resilience_summary.csv")
    return result


if __name__ == "__main__":
    result = run_resilience()
    print(result.groupby(["scenario", "adaptive"])[["switched", "policy_violation", "world_compliant_route_or_correct_abstention"]].mean())
    if result.policy_violation.any():
        raise SystemExit(1)

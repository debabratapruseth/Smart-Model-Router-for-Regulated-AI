import json
import tempfile
from pathlib import Path
import pandas as pd
from pydantic import ValidationError
from app.core.config_loader import ROOT
from app.core.router import ModelRouter
from app.models.request import RoutingRequest
from evaluation.metrics import policy_violation


def run_adversarial():
    cases = json.loads((ROOT / "data/adversarial_cases.json").read_text())
    rows = []
    with tempfile.TemporaryDirectory() as directory:
        for case in cases:
            router = ModelRouter(runtime_dir=Path(directory))
            if case["mutation"] == "inactive_model":
                next(m for m in router.models if m.model_id == "enterprise").status = "INACTIVE"
            if case["mutation"] == "unavailable_provider":
                router.policy_config["unavailable_providers"] = ["INTERNAL"]
            try:
                request = RoutingRequest.model_validate(case["request"])
                for name in ["rules", "weighted", "frontier_only", "cheapest_only", "jev"]:
                    response = router.route(request, name, allow_fallback=False)
                    rows.append({"case": case["name"], "strategy": name, "status": response.status,
                                 "passed": response.status == case["expected_status"],
                                 "policy_violation": policy_violation(request, response, router)})
            except ValidationError:
                rows.append({"case": case["name"], "strategy": "schema", "status": "VALIDATION_ERROR",
                             "passed": case["expected_status"] == "VALIDATION_ERROR", "policy_violation": False})
    result = pd.DataFrame(rows)
    result.to_csv(ROOT / "reports/adversarial_results.csv", index=False)
    return result


if __name__ == "__main__":
    result = run_adversarial()
    print(result.to_string(index=False))
    if not result.passed.all() or result.policy_violation.any():
        raise SystemExit(1)

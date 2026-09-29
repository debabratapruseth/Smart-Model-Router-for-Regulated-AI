import json
import tempfile
from pathlib import Path
import pandas as pd
from app.core.config_loader import ROOT
from app.core.router import ModelRouter
from app.models.request import RoutingRequest


def run_stability(router=None, write_reports=True):
    pairs = json.loads((ROOT / "data/paraphrase_pairs.json").read_text())
    rows = []
    with tempfile.TemporaryDirectory() as directory:
        router = router or ModelRouter(runtime_dir=Path(directory))
        for pair in pairs:
            for name, strategy in router.strategies.items():
                if not strategy.available():
                    continue
                responses = [router.route(RoutingRequest(request_id=f'{pair["pair_id"]}-{key}', prompt=pair[key], metadata=pair["metadata"]), name, allow_fallback=False)
                             for key in ["first", "second"]]
                selections = [r.decision.selected_model if r.decision else None for r in responses]
                rows.append({"pair_id": pair["pair_id"], "strategy": name, "first_model": selections[0], "second_model": selections[1],
                             "both_routed": all(r.status == "ROUTED" for r in responses),
                             "stable": selections[0] == selections[1] and responses[0].status == responses[1].status})
    result = pd.DataFrame(rows)
    if write_reports:
        result.to_csv(ROOT / "reports/stability_results.csv", index=False)
    return result


if __name__ == "__main__":
    print(run_stability().to_string(index=False))

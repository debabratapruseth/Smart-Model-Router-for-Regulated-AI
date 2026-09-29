import argparse
import copy
import tempfile
from pathlib import Path
import pandas as pd
from app.core.config_loader import ROOT
from app.core.router import ModelRouter
from evaluation.benchmark import run_benchmark
from scripts.generate_banking_benchmark import generate, ground_truth


def run_sensitivity(size=100):
    rows = generate(size)
    labels = ground_truth(rows)
    results, distributions = [], []
    with tempfile.TemporaryDirectory() as directory:
        router = ModelRouter(runtime_dir=Path(directory))
        for preference in ["BALANCED", "QUALITY_FIRST", "MINIMIZE_COST", "LATENCY_FIRST"]:
            variant = copy.deepcopy(rows)
            for row in variant:
                row["request"]["metadata"]["cost_preference"] = preference
            frame, summary = run_benchmark(variant, labels, router, ["weighted"], write_reports=False)
            results.append({"preference": preference, **summary.iloc[0].to_dict()})
            for model, count in frame.selected_model.fillna("NO_ROUTE").value_counts().items():
                distributions.append({"preference": preference, "model": model, "count": count})
    result = pd.DataFrame(results)
    result.to_csv(ROOT / "reports/sensitivity_results.csv", index=False)
    pd.DataFrame(distributions).to_csv(ROOT / "reports/sensitivity_distribution.csv", index=False)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=100)
    print(run_sensitivity(parser.parse_args().size).to_string(index=False))

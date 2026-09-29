"""Remove ranking signals only. Governance is held fixed for every ablation."""
import argparse
import tempfile
from pathlib import Path
import pandas as pd
from app.core.config_loader import ROOT
from app.core.router import ModelRouter
from evaluation.benchmark import run_benchmark
from scripts.generate_banking_benchmark import generate, ground_truth


def run_ablation(size=100):
    rows = generate(size)
    labels = ground_truth(rows)
    output = []
    with tempfile.TemporaryDirectory() as directory:
        router = ModelRouter(runtime_dir=Path(directory))
        for removed in [None, "complexity", "latency", "cost_preference", "context_length", "quality_requirement", "enterprise_knowledge"]:
            _, summary = run_benchmark(rows, labels, router, ["weighted"], write_reports=False, ranking_ablation=removed)
            output.append({"removed_ranking_feature": removed or "none", **summary.iloc[0].to_dict()})
    result = pd.DataFrame(output)
    result.to_csv(ROOT / "reports/ablation_results.csv", index=False)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=100)
    print(run_ablation(parser.parse_args().size).to_string(index=False))

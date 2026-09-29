"""Pareto efficiency from computed results, never hardcoded winners."""
import pandas as pd
from app.core.config_loader import ROOT


def frontier_data(summary=None, output_dir=None):
    summary = pd.read_csv(ROOT / "reports/benchmark_summary.csv") if summary is None else summary
    data = summary[summary.routed_requests > 0].copy()
    cost, latency, quality = "average_estimated_model_cost_usd", "average_model_latency_ms", "synthetic_quality_proxy"
    efficient = []
    for _, row in data.iterrows():
        dominates = ((data[cost] <= row[cost]) & (data[latency] <= row[latency]) & (data[quality] >= row[quality]) &
                     ((data[cost] < row[cost]) | (data[latency] < row[latency]) | (data[quality] > row[quality])))
        efficient.append(not dominates.any())
    data["pareto_efficient_synthetic_proxy"] = efficient
    data.to_csv((output_dir or ROOT / "reports") / "cost_quality_frontier.csv", index=False)
    return data


if __name__ == "__main__":
    print(frontier_data().to_string(index=False))

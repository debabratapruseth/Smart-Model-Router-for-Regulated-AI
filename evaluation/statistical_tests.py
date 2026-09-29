"""Paired comparisons on the same requests; no causal or production claims.

McNemar exact binomial test uses discordant correctness pairs. Bootstrap assumes
independent requests (templates violate this approximately); use cluster bootstrap
for a publication. Multiple comparisons are exploratory, without correction.
"""
import numpy as np
import pandas as pd
from evaluation.reporting import add_result_metadata
from scipy.stats import binomtest, wilcoxon
from app.core.config_loader import ROOT, load_config


def bootstrap_ci(values, seed=42, samples=1000):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(samples)]
    return tuple(np.quantile(means, [.025, .975]))


def run_statistics(frame=None, output_dir=None):
    frame = pd.read_csv(ROOT / "reports/benchmark_results.csv") if frame is None else frame
    config = load_config("evaluation")
    baseline = frame[(frame.strategy == "frontier_only") & (frame.status != "UNAVAILABLE")]
    results = []
    for name, group in frame.groupby("strategy"):
        if name == "frontier_only":
            continue
        pairs = baseline.merge(group[group.status != "UNAVAILABLE"], on="request_id", suffixes=("_base", "_router"))
        if pairs.empty:
            continue
        b = int((pairs.acceptable_correct_base & ~pairs.acceptable_correct_router).sum())
        c = int((~pairs.acceptable_correct_base & pairs.acceptable_correct_router).sum())
        results.append({"strategy": name, "metric": "acceptable_reference_route_agreement", "test": "exact_mcnemar",
                        "n": len(pairs), "p_value": binomtest(b, b+c, .5).pvalue if b+c else 1.0})
        served = pairs[(pairs.status_base == "ROUTED") & (pairs.status_router == "ROUTED")]
        for metric in ["estimated_cost_usd", "model_latency_ms", "routing_latency_ms", "synthetic_quality_proxy", "quality_score"]:
            complete = served.dropna(subset=[f"{metric}_router", f"{metric}_base"])
            delta = (complete[f"{metric}_router"] - complete[f"{metric}_base"]).astype(float).to_numpy()
            low, high = bootstrap_ci(delta, config["random_seed"], config["bootstrap_samples"])
            pvalue = None
            if "latency" in metric and len(delta):
                pvalue = 1.0 if np.all(delta == 0) else float(wilcoxon(delta, method="auto").pvalue)
            results.append({"strategy": name, "metric": metric, "test": "paired_bootstrap_and_optional_wilcoxon", "n": len(delta),
                "mean_difference_router_minus_frontier": float(delta.mean()) if len(delta) else None,
                "ci_low": low, "ci_high": high, "p_value": pvalue,
                "measured_quality_noninferiority_supported": bool(low > -config["quality_noninferiority_margin"]) if metric == "quality_score" and len(delta) else None})
    result = pd.DataFrame(results)
    result = add_result_metadata(result, frame)
    result.to_csv((output_dir or ROOT / "reports") / "statistical_results.csv", index=False)
    return result


if __name__ == "__main__":
    print(run_statistics().to_string(index=False))

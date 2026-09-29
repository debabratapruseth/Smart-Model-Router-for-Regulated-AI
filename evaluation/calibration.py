"""Binary confidence-in-acceptable-route calibration, conditional on routing.

These confidences are heuristic/model probabilities, not validated guarantees.
No-route outcomes and unavailable integrations are excluded, with counts reported.
"""
import numpy as np
import pandas as pd
from evaluation.reporting import add_result_metadata
from app.core.config_loader import ROOT, load_config


def run_calibration(frame=None, output_dir=None):
    frame = pd.read_csv(ROOT / "reports/benchmark_results.csv") if frame is None else frame
    bins = load_config("evaluation")["calibration_bins"]
    rows, summaries = [], []
    for name in ["ml", "llm", "jev"]:
        group = frame[(frame.strategy == name) & (frame.status == "ROUTED")].copy()
        if group.empty:
            summaries.append({"strategy": name, "count": 0, "brier_score": None, "ece": None})
            continue
        confidence = group.confidence.to_numpy(float)
        accuracy = group.acceptable_correct.to_numpy(float)
        bucket = np.minimum((confidence * bins).astype(int), bins - 1)
        ece = 0
        for index in range(bins):
            mask = bucket == index
            if not mask.any():
                continue
            gap = abs(confidence[mask].mean() - accuracy[mask].mean())
            ece += gap * mask.mean()
            rows.append({"strategy": name, "mode": group["mode"].iloc[0], "bucket_low": index / bins,
                         "bucket_high": (index+1) / bins, "mean_confidence": confidence[mask].mean(),
                         "actual_accuracy": accuracy[mask].mean(), "sample_count": int(mask.sum())})
        summaries.append({"strategy": name, "count": len(group), "brier_score": np.mean((confidence - accuracy)**2), "ece": ece})
    result = pd.DataFrame(rows, columns=["strategy", "mode", "bucket_low", "bucket_high", "mean_confidence", "actual_accuracy", "sample_count"])
    result["acceptable_reference_route_agreement_routed_only"] = result.actual_accuracy
    result = add_result_metadata(result, frame)
    result.to_csv((output_dir or ROOT / "reports") / "calibration_results.csv", index=False)
    add_result_metadata(pd.DataFrame(summaries), frame).to_csv((output_dir or ROOT / "reports") / "calibration_summary.csv", index=False)
    return result


if __name__ == "__main__":
    print(run_calibration().to_string(index=False))

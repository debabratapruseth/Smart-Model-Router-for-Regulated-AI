"""Read new per-run reports and historical reports without rewriting either."""
import json
import pandas as pd
from app.core.config_loader import ROOT

AGREEMENT_COLUMNS = {
    "preferred_reference_route_agreement_including_correct_abstentions": "Preferred Reference-Route Agreement — Including Correct Abstentions",
    "acceptable_reference_route_agreement_including_correct_abstentions": "Acceptable Reference-Route Agreement — Including Correct Abstentions",
    "preferred_reference_route_agreement_routed_only": "Preferred Reference-Route Agreement — Routed Requests Only",
    "acceptable_reference_route_agreement_routed_only": "Acceptable Reference-Route Agreement — Routed Requests Only",
}
DISPLAY_COLUMNS = {
    "strategy": "Strategy", "benchmark_type": "Benchmark Type", "evidence_type": "Evidence Type",
    "execution_mode": "Execution Mode", "research_eligible": "Research Eligible", "total_requests": "Requests",
    "routed_requests": "Routed", "correct_no_route": "Correct NO_ROUTE", **AGREEMENT_COLUMNS,
    "average_routing_latency_ms": "Mean Routing Latency (ms)", "p50_routing_latency_ms": "p50 Routing Latency (ms)",
    "p95_routing_latency_ms": "p95 Routing Latency (ms)",
}


def report_directories(benchmark_type, root=None):
    root = ROOT / "reports" if root is None else root
    paths = sorted((root / benchmark_type.lower()).glob("*/run_manifest.json"), reverse=True)
    directories = [p.parent for p in paths]
    if benchmark_type == "IID_SYNTHETIC":
        directories += [p for p in [root / "trained_ml", root] if (p / "benchmark_summary.csv").exists()]
    return directories


def load_report(directory):
    summary = pd.read_csv(directory / "benchmark_summary.csv")
    details = pd.read_csv(directory / "benchmark_results.csv")
    manifest = json.loads((directory / "run_manifest.json").read_text())
    legacy = "benchmark_type" not in manifest
    if legacy:
        manifest = {**manifest, "benchmark_type": "IID_SYNTHETIC", "evidence_type": "SYNTHETIC",
                    "legacy_report": True, "provenance_note": "Historical report; new split provenance was not recorded."}
        summary["benchmark_type"] = "IID_SYNTHETIC"
        summary["evidence_type"] = "SYNTHETIC"
        summary["execution_mode"] = summary["mode"].map({"local": "LOCAL", "mock": "MOCK", "external": "LIVE"})
        # Do not retroactively certify historical ML methodology.
        summary["research_eligible"] = (summary.execution_mode == "LOCAL") & (summary.strategy != "ml") & (summary.available_requests > 0)
    for precise, old in [(list(AGREEMENT_COLUMNS)[0], "preferred_model_accuracy"),
                         (list(AGREEMENT_COLUMNS)[1], "acceptable_route_accuracy")]:
        if precise not in summary:
            summary[precise] = summary[old]
    for idx, row in summary.iterrows():
        group = details[details.strategy == row.strategy]
        routed = group[group.status == "ROUTED"]
        active = group[group.status != "UNAVAILABLE"]
        values = {
            "preferred_reference_route_agreement_routed_only": routed.preferred_correct.mean(),
            "acceptable_reference_route_agreement_routed_only": routed.acceptable_correct.mean(),
            "correct_no_route": int(((active.status == "NO_ROUTE") & active.expected_no_route).sum()),
            "p50_routing_latency_ms": active.routing_latency_ms.median(),
        }
        for key, value in values.items():
            if key not in summary.columns or pd.isna(summary.loc[idx, key]):
                summary.loc[idx, key] = value
    return summary, details, manifest


def display_summary(summary):
    return summary[[key for key in DISPLAY_COLUMNS if key in summary]].rename(columns=DISPLAY_COLUMNS)


def add_result_metadata(result, source):
    """Carry benchmark identity into derived statistics and calibration tables."""
    result = result.copy()
    for key in ["benchmark_type", "evidence_type", "dataset_hash", "run_id", "timestamp", "input_mode", "phase"]:
        if key in source:
            result[key] = source[key].iloc[0]
    if "strategy" in result:
        for key in ["execution_mode", "strategy_version", "research_eligible"]:
            if key in source:
                values = source.groupby("strategy")[key].agg("any" if key == "research_eligible" else "first")
                result[key] = result.strategy.map(values)
    return result

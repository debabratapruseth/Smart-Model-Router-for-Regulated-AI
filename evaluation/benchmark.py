"""CPU-only benchmark. Default execution is off; all catalog values synthetic."""
import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from uuid import uuid4
from pathlib import Path
from importlib.metadata import version
import pandas as pd
from app.core.config_loader import ROOT, load_config, fingerprint
from app.core.router import ModelRouter
from app.core.scoring import task_quality
from app.models.request import RoutingRequest
from scripts.generate_banking_benchmark import generate, ground_truth
from evaluation.metrics import summarize, policy_violation, HYPOTHESES
from evaluation.research_design import (BENCHMARK_NAMES, final_test_data, dataset_hash, leakage_check,
                                        ml_provenance, validate_design)
from scripts.generate_banking_benchmark import template_id, ORIGINAL_TEMPLATE_IDS
from app.models.routing_input import FEATURE_SCHEMA_VERSION


def train_ml(router: ModelRouter, benchmark_type="IID_SYNTHETIC", test_rows=None, test_labels=None) -> None:
    """Fit TRAIN only; evaluate fixed settings on VALIDATION before final testing."""
    cfg = load_config("evaluation")
    validate_design(cfg, benchmark_type)
    if test_rows is None:
        test_rows, test_labels = final_test_data(benchmark_type)
    test_labels = ground_truth(test_rows) if test_labels is None else test_labels
    training = generate(cfg["ml_training_size"], cfg["ml_training_seed"], "TRAIN", cfg["training_templates"])
    validation = generate(cfg["ml_validation_size"], cfg["ml_validation_seed"], "VALIDATION", cfg["validation_templates"])
    training_labels, validation_labels = ground_truth(training), ground_truth(validation)
    examples, fitted = [], []
    for row, target in zip(training, training_labels):
        if target["preferred_model"]:
            value = router.prepare(RoutingRequest.model_validate(row["request"])).canonical()
            examples.append((value, target["preferred_model"]))
            fitted.append(row)
    # Integrity gates run BEFORE fitting, including when callers supply custom final tests.
    checks = leakage_check(training, validation, test_rows, router, benchmark_type, fitted)
    router.strategies["ml"].fit(examples, cfg["ml_training_seed"], cfg["ml_hyperparameters"])
    provenance = ml_provenance(router, training, validation, test_rows, training_labels,
                               validation_labels, test_labels, fitted, benchmark_type)
    # No validation row is passed to fit; no parameter updates follow this evaluation.
    validation_frame, validation_summary = run_benchmark(validation, validation_labels, router, ["ml"],
        write_reports=False, benchmark_type=benchmark_type, phase="VALIDATION")
    router.ml_research = {"training": training, "validation": validation, "fitted": fitted,
        "provenance": provenance, "leakage": checks,
        "validation_frame": validation_frame, "validation_summary": validation_summary}


def run_benchmark(rows=None, labels=None, router=None, strategies=None, write_reports=True,
                  output_dir=None, ranking_ablation=None, execute=False,
                  benchmark_type="IID_SYNTHETIC", phase="FINAL_TEST"):
    cfg = load_config("evaluation")
    validate_design(cfg, benchmark_type)
    if phase not in {"FINAL_TEST", "VALIDATION"}:
        raise ValueError("Unknown evaluation phase")
    if execute:
        raise ValueError("Synthetic primary comparison does not execute downstream models")
    router = router or ModelRouter(runtime_dir=ROOT / "runtime/benchmark")
    if rows is None:
        rows, default_labels = final_test_data(benchmark_type)
        labels = default_labels if labels is None else labels
    labels = ground_truth(rows) if labels is None else labels
    expected_seed = cfg["ml_validation_seed"] if phase == "VALIDATION" else (
        cfg["random_seed"] if benchmark_type == "IID_SYNTHETIC" else cfg["held_out_test_seed"])
    if any(row.get("random_seed") != expected_seed for row in rows):
        raise ValueError("Dataset seed does not match the declared evaluation split")
    if phase == "FINAL_TEST":
        allowed = set(cfg["test_templates"] if benchmark_type == "TEMPLATE_HELD_OUT" else ORIGINAL_TEMPLATE_IDS)
        if any(template_id(row) not in allowed for row in rows):
            raise ValueError("Template leakage or unexpected final-test template")
    expected = {r["request_id"]: r for r in labels}
    if len(expected) != len(rows) or set(expected) != {r["request"]["request_id"] for r in rows}:
        raise ValueError("Reference labels must match unique request IDs exactly")
    strategies = cfg["strategies"] if strategies is None else strategies
    if not rows:
        raise ValueError("Benchmark must contain requests")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8]
    timestamp = datetime.now(timezone.utc).isoformat()
    data_hash = dataset_hash(rows, labels)
    research = getattr(router, "ml_research", None) if phase == "FINAL_TEST" else None
    if research:
        if research["provenance"]["benchmark_type"] != benchmark_type:
            raise ValueError("ML training benchmark type differs from final test")
        if research["provenance"]["test_dataset_hash"] != data_hash:
            raise ValueError("Final test differs from the dataset checked before ML fitting")
        checks = leakage_check(research["training"], research["validation"], rows, router, benchmark_type, research["fitted"])
    else:
        # Even local-only runs check the configured development split against final data.
        checks = None
        if phase == "FINAL_TEST":
            training = generate(cfg["ml_training_size"], cfg["ml_training_seed"], "TRAIN", cfg["training_templates"])
            validation = generate(cfg["ml_validation_size"], cfg["ml_validation_seed"], "VALIDATION", cfg["validation_templates"])
            checks = leakage_check(training, validation, rows, router, benchmark_type)
    results = []
    for row in rows:
        request = RoutingRequest.model_validate(row["request"])
        target = expected[request.request_id]
        # Normalize/filter/snapshot once. Strategy side effects cannot change the next candidate set.
        prepared = router.prepare(request)
        candidate_ids = [m.model_id for m in prepared.eligible]
        canonical_hash = fingerprint(prepared.canonical().model_dump(mode="json"))
        for name in strategies:
            # Benchmark raw strategies without silently converting them to weighted.
            # This task never calls live integrations, even if keys are present in the environment.
            decision = router.route(request, name, allow_fallback=False, ranking_ablation=ranking_ablation,
                                    prepared=prepared, structured_only=True)
            if decision.eligible_models != candidate_ids:
                raise ValueError("Strategy received a different eligible candidate set")
            selected_id = decision.decision.selected_model if decision.decision else None
            if selected_id is not None and selected_id not in candidate_ids:
                raise ValueError("Selected model is outside canonical candidates")
            model = next((m for m in prepared.models if m.model_id == selected_id), None)
            correct_abstention = decision.status == "NO_ROUTE" and target["expected_no_route"]
            executed = {}
            execution_mode = "MOCK" if decision.strategy_mode == "mock" else "LOCAL" if decision.strategy_mode == "local" else "LIVE"
            eligible_evidence = (execution_mode == "LOCAL" and decision.status not in {"UNAVAILABLE", "INVALID_DECISION"}
                                 and (name != "ml" or research is not None or phase == "VALIDATION"))
            results.append({"benchmark_type": benchmark_type, "evidence_type": "SYNTHETIC", "execution_mode": execution_mode,
                "research_eligible": eligible_evidence, "dataset_hash": data_hash, "run_id": run_id, "timestamp": timestamp,
                "strategy_version": "structured-2.0", "input_mode": "STRUCTURED_ONLY", "phase": phase, "live_calls_enabled": False,
                "template_id": template_id(row), "canonical_input_hash": canonical_hash,
                "eligible_models": json.dumps(candidate_ids), "request_id": request.request_id, "strategy": name, "mode": decision.strategy_mode,
                "status": decision.status, "selected_model": selected_id,
                "preferred_correct": bool(correct_abstention or (selected_id is not None and selected_id == target["preferred_model"])),
                "acceptable_correct": bool(correct_abstention or selected_id in target["acceptable_models"]),
                "expected_no_route": target["expected_no_route"], "policy_violation": policy_violation(request, decision, router),
                "fallback_used": decision.fallback_used, "confidence": decision.decision.confidence if decision.decision else None,
                "routing_latency_ms": decision.estimated.routing_latency_ms, "model_latency_ms": decision.estimated.model_latency_ms,
                "estimated_cost_usd": decision.estimated.cost_usd,
                "synthetic_quality_proxy": task_quality(model, {"task_type": row["task_type"]}, router.config) if model else None,
                "quality_score": executed.get("quality_score"), "evaluation_method": executed.get("evaluation_method", "not_executed"),
                "execution_status": executed.get("status", "DISABLED"), "tier": model.tier if model else None,
                "difficulty": row["difficulty"], "task_type": row["task_type"]})
    frame = pd.DataFrame(results)
    summary = summarize(frame)
    if write_reports:
        directory = Path(output_dir) if output_dir else ROOT / "reports" / benchmark_type.lower() / run_id
        if (directory / "run_manifest.json").exists() or (directory / "benchmark_results.csv").exists():
            raise ValueError("Refusing to overwrite an existing benchmark run")
        directory.mkdir(parents=True, exist_ok=True)
        # Legacy paraphrase pairs are a separate study, not held-out final-test evidence.
        frame.to_csv(directory / "benchmark_results.csv", index=False)
        summary.to_csv(directory / "benchmark_summary.csv", index=False)
        manifest = {"random_seed": cfg["random_seed"], "benchmark_version": cfg["version"],
                    "policy_version": router.policy_config["version"], "catalog_version": router.catalog_config["version"],
                    "routing_config_version": router.config["version"], "router_version": "1.0.0", "hypotheses": HYPOTHESES,
                    "python": sys.version, "platform": platform.platform(), "dataset_hash": fingerprint({"rows": rows, "labels": labels}),
                    "config_hashes": {k: fingerprint(v) for k, v in {"evaluation": cfg, "routing": router.config, "policies": router.policy_config, "catalog": router.catalog_config}.items()},
                    "packages": {name: version(name) for name in ["numpy", "pandas", "scikit-learn", "scipy", "pydantic"]},
                    "synthetic": True, "execution_requested": execute,
                    "timing_scope": "routing through validation, excludes audit I/O; wall time is not deterministic"}
        manifest.update({"benchmark_type": benchmark_type, "benchmark_display_name": BENCHMARK_NAMES[benchmark_type],
            "evidence_type": "SYNTHETIC", "input_mode": "STRUCTURED_ONLY", "run_id": run_id, "timestamp": timestamp,
            "dataset_hash": data_hash, "test_dataset_hash": data_hash, "test_request_count": len(rows),
            "test_seed": cfg["random_seed"] if benchmark_type == "IID_SYNTHETIC" else cfg["held_out_test_seed"],
            "test_template_ids": sorted({template_id(r) for r in rows}),
            "feature_schema_version": FEATURE_SCHEMA_VERSION, "ml_trained": research is not None,
            "research_eligible_definition": "Evidence about this local implementation under this synthetic benchmark only; excludes Jev mock, unavailable adapters, and ML without verified split provenance.",
            "timing_scope": "Shared normalization/policy preparation plus per-strategy routing and validation; excludes audit I/O; no network calls"})
        if research:
            manifest.update(research["provenance"])
            research["validation_frame"].to_csv(directory / "validation_results.csv", index=False)
            research["validation_summary"].to_csv(directory / "validation_summary.csv", index=False)
            for split in ("training", "validation"):
                (directory / f"{split}_dataset.json").write_text(json.dumps(research[split], indent=2) + "\n")
                (directory / f"{split}_reference_labels.json").write_text(json.dumps(ground_truth(research[split]), indent=2) + "\n")
        (directory / "configuration_snapshot.json").write_text(json.dumps({
            "evaluation": cfg, "routing": router.config, "policies": router.policy_config, "catalog": router.catalog_config}, indent=2) + "\n")
        (directory / "test_dataset.json").write_text(json.dumps([dict(r, template_id=template_id(r)) for r in rows], indent=2) + "\n")
        (directory / "test_reference_labels.json").write_text(json.dumps(labels, indent=2) + "\n")
        if checks is not None:
            checks.to_csv(directory / "leakage_check.csv", index=False)
        (directory / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        frame.attrs["report_dir"] = str(directory)
        summary.attrs["report_dir"] = str(directory)
    return frame, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int)
    parser.add_argument("--benchmark-type", choices=list(BENCHMARK_NAMES), default="IID_SYNTHETIC")
    parser.add_argument("--output-dir", type=Path, help="New empty output directory; defaults to a unique per-run directory")
    parser.add_argument("--train-ml", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Reserved; synthetic primary benchmarks reject execution")
    parser.add_argument("--live-pilot", action="store_true", help="Sample both frozen datasets; replay local baselines")
    parser.add_argument("--live-full", action="store_true", help="Use all frozen requests; explicit call cap still enforced")
    parser.add_argument("--final-experiment", action="store_true", help="200 IID + 200 held-out; no permission means preflight only")
    parser.add_argument("--stability-experiment", action="store_true", help="25 + 25 routable primary requests, five repetitions")
    parser.add_argument("--dry-run", action="store_true", help="Final/stability sample validation and plan only; never calls providers")
    parser.add_argument("--sampling-seed", type=int)
    parser.add_argument("--stability-seed", type=int)
    parser.add_argument("--strategies", help="Comma-separated rules,weighted,ml,openai,jev for live runs")
    parser.add_argument("--allow-live-api", action="store_true")
    parser.add_argument("--max-live-api-calls", type=int, help="Defaults to live configuration cap (50)")
    parser.add_argument("--max-routing-spend-usd", type=float)
    parser.add_argument("--resume", type=Path, help="Existing reports/live/<run_id> directory")
    parser.add_argument("--force-rerun", action="store_true", help="Append replacement observations; may spend credits again")
    args = parser.parse_args()
    if args.final_experiment or args.stability_experiment:
        if sum([args.final_experiment,args.stability_experiment,args.live_pilot,args.live_full])!=1 or args.train_ml or args.execute or args.size is not None or args.output_dir:
            parser.error("Final/stability modes cannot train, regenerate, execute downstream models, resize or mix modes")
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
        from evaluation.experiment_preflight import preflight
        from evaluation.experiment_sampling import experiment_config
        mode = 'final' if args.final_experiment else 'stability'
        strategies = args.strategies.split(',') if args.strategies else None
        try:
            if args.dry_run or not args.allow_live_api:
                print(json.dumps(preflight(mode,strategies,args.sampling_seed,args.stability_seed,
                    args.max_live_api_calls,args.max_routing_spend_usd,resume=args.resume,
                    force_rerun=args.force_rerun),indent=2))
            else:
                from evaluation.live_benchmark import run_live
                directory,summary = run_live(strategies=strategies,allow_live_api=args.allow_live_api,
                    max_live_api_calls=args.max_live_api_calls,max_routing_spend_usd=args.max_routing_spend_usd,
                    resume=args.resume,force_rerun=args.force_rerun,experiment_mode=mode,
                    experiment_cfg=experiment_config(args.sampling_seed,args.stability_seed))
                print(f"Experiment reports: {directory}")
                print(summary.to_string(index=False))
        except ValueError as exc:
            parser.error(str(exc))
        return
    if args.dry_run or args.sampling_seed is not None or args.stability_seed is not None:
        parser.error("Dry-run and experiment seeds require a final or stability experiment mode")
    if args.live_pilot or args.live_full:
        if (args.live_pilot and args.live_full) or args.train_ml or args.execute or args.size is not None or args.output_dir:
            parser.error("Live runs cannot train ML, execute downstream models, regenerate/resize datasets or reuse synthetic output paths")
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
        from evaluation.live_benchmark import run_live
        try:
            directory, summary = run_live(strategies=args.strategies.split(",") if args.strategies else None,
                allow_live_api=args.allow_live_api, max_live_api_calls=args.max_live_api_calls,
                max_routing_spend_usd=args.max_routing_spend_usd, full=args.live_full,
                resume=args.resume, force_rerun=args.force_rerun)
        except ValueError as exc:
            parser.error(str(exc))
        print(f"Live/replay reports: {directory}")
        print(summary.to_string(index=False))
        return
    if args.allow_live_api or args.resume or args.force_rerun or args.strategies:
        parser.error("Live options require --live-pilot or --live-full")
    if args.size is not None and args.size < 1:
        parser.error("size must be positive")
    rows, labels = final_test_data(args.benchmark_type, args.size)
    router = ModelRouter(runtime_dir=ROOT / "runtime/benchmark")
    if args.train_ml:
        train_ml(router, args.benchmark_type, rows, labels)
    frame, summary = run_benchmark(rows, labels, router, execute=args.execute,
                                  benchmark_type=args.benchmark_type, output_dir=args.output_dir)
    from evaluation.statistical_tests import run_statistics
    from evaluation.calibration import run_calibration
    from evaluation.cost_quality_frontier import frontier_data
    directory = Path(frame.attrs["report_dir"])
    run_statistics(frame, output_dir=directory)
    run_calibration(frame, output_dir=directory)
    frontier_data(summary, output_dir=directory)
    print(f"Reports: {directory}")
    from evaluation.reporting import display_summary
    print(display_summary(summary).to_string(index=False))
    print("Synthetic proxies cannot establish H1/H2 for real models. H4 requires real Jev and LLM experiments.")


if __name__ == "__main__":
    main()

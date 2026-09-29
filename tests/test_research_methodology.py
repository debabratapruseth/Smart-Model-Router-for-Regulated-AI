"""Research integrity tests: split boundaries, provenance and common information."""
import json
from copy import deepcopy
import pytest
import pandas as pd
from app.core.config_loader import ROOT, load_config
from app.models.request import RoutingRequest
from app.models.routing_input import FEATURE_SCHEMA_VERSION, encode_canonical
from app.strategies.base import StrategyResult
from evaluation.benchmark import train_ml, run_benchmark
from evaluation.research_design import final_test_data, dataset_hash, leakage_check, validate_design
from evaluation.reporting import load_report, display_summary, report_directories
from scripts.generate_banking_benchmark import generate, ground_truth, template_id, ORIGINAL_TEMPLATE_IDS


@pytest.fixture
def small_design(monkeypatch):
    original = load_config("evaluation")
    cfg = {**original, "ml_training_size": 56, "ml_validation_size": 28}
    from evaluation import benchmark, research_design
    from scripts import generate_banking_benchmark
    for module in (benchmark, research_design, generate_banking_benchmark):
        real_load = module.load_config
        monkeypatch.setattr(module, "load_config", lambda name, *args, _load=real_load, **kwargs:
                            cfg if name == "evaluation" else _load(name, *args, **kwargs))
    return cfg


def test_iid_requests_preserved_and_ids_independent():
    training = generate(600, 1042, "TRAIN", ORIGINAL_TEMPLATE_IDS)
    test, labels = final_test_data()
    regenerated = generate(1000, 42)
    assert [r["request"] for r in test] == [r["request"] for r in regenerated]
    assert ground_truth(test) == labels
    assert not {r["request"]["request_id"] for r in training} & {r["request"]["request_id"] for r in test}


def test_template_split_disjoint_and_represents_all_tasks():
    cfg = load_config("evaluation")
    for split in ["training_templates", "validation_templates"]:
        assert not set(cfg[split]) & set(cfg["test_templates"])
    rows, _ = final_test_data("TEMPLATE_HELD_OUT")
    assert len(rows) == 400
    assert len({r["task_type"] for r in rows}) == 14
    assert {template_id(r) for r in rows} == set(cfg["test_templates"])


@pytest.mark.parametrize("kind", ["IID_SYNTHETIC", "TEMPLATE_HELD_OUT"])
def test_hashes_deterministic(kind):
    rows, labels = final_test_data(kind, 28)
    assert dataset_hash(rows, labels) == dataset_hash(*final_test_data(kind, 28))
    changed = deepcopy(rows)
    changed[0]["request"]["prompt"] += " changed"
    assert dataset_hash(changed, labels) != dataset_hash(rows, labels)


@pytest.mark.parametrize("kind", ["IID_SYNTHETIC", "TEMPLATE_HELD_OUT"])
def test_fit_only_train_and_manifest(router, small_design, monkeypatch, tmp_path, kind):
    test, labels = final_test_data(kind, 28)
    ml = router.strategies["ml"]
    original_fit = ml.fit
    captured = []
    def spy(examples, *args):
        captured.extend(examples)
        return original_fit(examples, *args)
    monkeypatch.setattr(ml, "fit", spy)
    train_ml(router, kind, test, labels)
    state = router.ml_research
    expected = [router.prepare(RoutingRequest.model_validate(row["request"])).canonical().model_dump()
                for row in state["fitted"]]
    assert [value.model_dump() for value, label in captured] == expected
    assert all(r["request"]["request_id"].startswith("TRAIN-") for r in state["fitted"])
    assert len(captured) == state["provenance"]["training_request_count_fitted"]
    assert len(state["validation_frame"]) == 28
    assert state["validation_frame"].phase.eq("VALIDATION").all()
    out = tmp_path / "report"
    frame, summary = run_benchmark(test, labels, router, benchmark_type=kind, output_dir=out)
    manifest = json.loads((out / "run_manifest.json").read_text())
    assert manifest["training_seed"] == 1042
    assert manifest["validation_seed"] == 1043
    assert manifest["test_seed"] == (42 if kind == "IID_SYNTHETIC" else 2042)
    assert manifest["ml_hyperparameters"]["n_estimators"] == 60
    assert manifest["ml_hyperparameters"]["max_depth"] == 10
    assert manifest["ml_random_state"] == 1042
    assert manifest["feature_schema_version"] == FEATURE_SCHEMA_VERSION
    for split in ["training", "validation", "test"]:
        rows = json.loads((out / f"{split}_dataset.json").read_text())
        targets = json.loads((out / f"{split}_reference_labels.json").read_text())
        assert dataset_hash(rows, targets) == manifest[f"{split}_dataset_hash"]
    assert len(manifest["model_fingerprint"]) == 64
    assert frame.benchmark_type.eq(kind).all()
    assert frame.evidence_type.eq("SYNTHETIC").all()
    assert summary[summary.strategy == "jev"].research_eligible.eq(False).all()
    assert frame[frame.strategy == "llm"].status.eq("UNAVAILABLE").all()
    assert frame.groupby("request_id").canonical_input_hash.nunique().eq(1).all()
    assert frame.groupby("request_id").eligible_models.nunique().eq(1).all()
    for row in frame.itertuples():
        if row.status == "ROUTED":
            assert row.selected_model in json.loads(row.eligible_models)
    assert (out / "leakage_check.csv").exists()
    assert pd.read_csv(out / "leakage_check.csv").passed.all()
    loaded, _, _ = load_report(out)
    assert "Preferred Reference-Route Agreement — Routed Requests Only" in display_summary(loaded)
    # A final test different from the one checked before fitting is rejected.
    with pytest.raises(ValueError, match="Final test differs"):
        run_benchmark(test[:1], labels[:1], router, write_reports=False, benchmark_type=kind)
    with pytest.raises(ValueError, match="overwrite"):
        run_benchmark(test, labels, router, output_dir=out, benchmark_type=kind)


@pytest.mark.parametrize("source", ["training", "validation"])
def test_deliberate_template_leakage_fails(router, small_design, source):
    cfg = small_design
    training = generate(14, 1042, "TRAIN", cfg["training_templates"])
    validation = generate(14, 1043, "VALIDATION", cfg["validation_templates"])
    test, _ = final_test_data("TEMPLATE_HELD_OUT", 14)
    test[0]["template_id"] = template_id((training if source == "training" else validation)[0])
    with pytest.raises(ValueError, match="leakage"):
        leakage_check(training, validation, test, router, "TEMPLATE_HELD_OUT")
    with pytest.raises(ValueError, match="leakage"):
        run_benchmark(test, router=router, benchmark_type="TEMPLATE_HELD_OUT", write_reports=False)


def test_leakage_rejected_before_fit(router, small_design, monkeypatch):
    test = generate(14, 1042, "TRAIN", small_design["training_templates"])
    def forbidden(*args):
        pytest.fail("fit must not run on contaminated data")
    monkeypatch.setattr(router.strategies["ml"], "fit", forbidden)
    with pytest.raises(ValueError, match="leakage"):
        train_ml(router, test_rows=test)


def test_misconfigured_pools_and_seeds_fail():
    cfg = load_config("evaluation")
    cfg["test_templates"] = cfg["training_templates"]
    with pytest.raises(ValueError, match="Template leakage"):
        validate_design(cfg, "TEMPLATE_HELD_OUT")
    cfg["ml_validation_seed"] = cfg["random_seed"]
    with pytest.raises(ValueError, match="seeds"):
        validate_design(cfg, "IID_SYNTHETIC")


def test_all_strategies_receive_same_input_and_cannot_mutate_next(router, request_factory, monkeypatch):
    request = request_factory()
    prepared = router.prepare(request)
    expected = prepared.canonical().model_dump(mode="json")
    observed = []
    for name, strategy in router.strategies.items():
        monkeypatch.setattr(strategy, "available", lambda request=None: True)
        def capture(value):
            observed.append(value.model_dump(mode="json"))
            selected = value.eligible_models[0].model_id
            value.metadata.context_tokens = 9999999
            value.eligible_models.clear()
            return StrategyResult(selected_model=selected, confidence=1)
        monkeypatch.setattr(strategy, "route_canonical", capture)
        response = router.route(request, name, allow_fallback=False, prepared=prepared, structured_only=True)
        assert response.decision.selected_model in response.eligible_models
    assert len(observed) == 7
    assert all(value == expected for value in observed)
    assert "prompt" not in expected
    encoded = encode_canonical(prepared.canonical())
    assert any(key.startswith("candidate.") for key in encoded)
    assert "metadata.latency_sla_ms" in encoded


def test_no_route_valid_and_out_of_set_rejected(router, request_factory, monkeypatch):
    request = request_factory(allowed_models=[])
    prepared = router.prepare(request)
    assert not prepared.eligible
    assert router.route(request, "weighted", prepared=prepared, structured_only=True).status == "NO_ROUTE"
    request = request_factory()
    monkeypatch.setattr(router.strategies["weighted"], "route_canonical",
                        lambda value: StrategyResult(selected_model="not-eligible", confidence=1))
    result = router.route(request, "weighted", prepared=router.prepare(request), structured_only=True)
    assert result.status == "INVALID_DECISION"
    assert result.decision is None


def test_primary_never_calls_network_even_if_configured(router, monkeypatch):
    import httpx
    monkeypatch.setenv("OPENAI_API_KEY", "test-placeholder")
    monkeypatch.setenv("OPENAI_ROUTER_MODEL", "test-model")
    def forbidden(*args, **kwargs):
        pytest.fail("Primary benchmark must not call live APIs")
    monkeypatch.setattr(httpx, "post", forbidden)
    frame, _ = run_benchmark(generate(14), router=router, write_reports=False)
    assert frame[frame.strategy == "llm"].status.eq("UNAVAILABLE").all()


def test_legacy_reports_load_without_modification():
    directory = ROOT / "reports/trained_ml"
    original = (directory / "benchmark_summary.csv").read_bytes()
    summary, details, manifest = load_report(directory)
    ml = summary[summary.strategy == "ml"].iloc[0]
    assert ml.preferred_model_accuracy == .85
    assert ml.preferred_reference_route_agreement_including_correct_abstentions == .85
    assert ml.preferred_reference_route_agreement_routed_only == pytest.approx(595 / 745)
    assert manifest["legacy_report"]
    assert (directory / "benchmark_summary.csv").read_bytes() == original


def test_streamlit_both_benchmarks_and_terminology(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from evaluation import reporting
    # Minimal report copies exercise both selectors without requiring committed new runs.
    for kind in ["IID_SYNTHETIC", "TEMPLATE_HELD_OUT"]:
        target = tmp_path / kind.lower() / "test-run"
        target.mkdir(parents=True)
        for filename in ["benchmark_summary.csv", "benchmark_results.csv", "run_manifest.json"]:
            (target / filename).write_bytes((ROOT / "reports/trained_ml" / filename).read_bytes())
        manifest = json.loads((target / "run_manifest.json").read_text())
        manifest.update(benchmark_type=kind, evidence_type="SYNTHETIC")
        (target / "run_manifest.json").write_text(json.dumps(manifest))
    original = reporting.report_directories
    monkeypatch.setattr(reporting, "report_directories", lambda kind: original(kind, tmp_path))
    app = AppTest.from_file(str(ROOT / "ui/streamlit_app.py"), default_timeout=30).run()
    app.sidebar.radio[0].set_value("Executive / Research").run()
    assert not app.exception
    assert any("Synthetic Reference-Route Agreement" in item.value for item in app.subheader)
    app.selectbox[0].set_value("TEMPLATE_HELD_OUT").run()
    assert not app.exception
    assert app.dataframe
    assert any("templates excluded" in item.value for item in app.caption)


def test_declared_seed_must_match_dataset(router):
    with pytest.raises(ValueError, match="Dataset seed"):
        run_benchmark(generate(14, seed=999), router=router, write_reports=False)


def test_derived_results_keep_metadata(router, tmp_path):
    from evaluation.statistical_tests import run_statistics
    from evaluation.calibration import run_calibration
    frame, _ = run_benchmark(generate(14), router=router, write_reports=False)
    stats = run_statistics(frame, output_dir=tmp_path)
    calibration = run_calibration(frame, output_dir=tmp_path)
    for result in [stats, calibration]:
        assert result.benchmark_type.eq("IID_SYNTHETIC").all()
        assert result.evidence_type.eq("SYNTHETIC").all()
        assert result[result.strategy == "jev"].research_eligible.eq(False).all()

"""Small, explicit research split, leakage and provenance helpers (no model registry)."""
import json
import re
from itertools import combinations
from hashlib import sha256
import pandas as pd
from app.core.config_loader import ROOT, load_config, fingerprint
from app.models.request import RoutingRequest
from app.models.routing_input import FEATURE_SCHEMA_VERSION, encode_canonical
from scripts.generate_banking_benchmark import generate, ground_truth, template_id, ORIGINAL_TEMPLATE_IDS

BENCHMARK_NAMES = {
    "IID_SYNTHETIC": "IID / Template-Familiar Synthetic Benchmark",
    "TEMPLATE_HELD_OUT": "Template-Held-Out Generalization Benchmark",
}
AGREEMENT_NOTE = ("Reference-route agreement measures agreement with the project’s synthetic routing objective. "
                  "It does not measure actual downstream model response quality.")
METHODOLOGY_NOTE = ("IID results evaluate new synthetic samples generated from a familiar template family. "
                    "Template-held-out results evaluate requests generated from templates excluded from ML training and validation.")


def validate_design(cfg, benchmark_type):
    if benchmark_type not in BENCHMARK_NAMES:
        raise ValueError("Unknown benchmark_type")
    if cfg["primary_input_mode"] != "STRUCTURED_ONLY":
        raise ValueError("Primary comparison requires STRUCTURED_ONLY; rich-input experiment is not implemented")
    test_seed = cfg["random_seed"] if benchmark_type == "IID_SYNTHETIC" else cfg["held_out_test_seed"]
    if len({cfg["ml_training_seed"], cfg["ml_validation_seed"], test_seed}) != 3:
        raise ValueError("Training, validation and final-test seeds must be distinct")
    if benchmark_type == "TEMPLATE_HELD_OUT":
        final = set(cfg["test_templates"])
        if final & (set(cfg["training_templates"]) | set(cfg["validation_templates"])):
            raise ValueError("Template leakage: final test templates occur in training/validation configuration")


def final_test_data(benchmark_type="IID_SYNTHETIC", size=None):
    cfg = load_config("evaluation")
    validate_design(cfg, benchmark_type)
    if benchmark_type == "IID_SYNTHETIC":
        # Preserve committed requests and labels, including historical wrapper metadata.
        if size is None:
            rows = json.loads((ROOT / "data/banking_benchmark.json").read_text())
            labels = json.loads((ROOT / "data/expected_routes.json").read_text())
        else:
            rows = generate(size)
            labels = ground_truth(rows)
    else:
        rows = generate(cfg["held_out_test_size"] if size is None else size,
                        cfg["held_out_test_seed"], "HELDOUT", cfg["test_templates"])
        labels = ground_truth(rows)
    return rows, labels


def dataset_hash(rows, labels):
    # Stable IDs are included even when legacy on-disk rows predate them.
    return fingerprint({"rows": [dict(r, template_id=template_id(r)) for r in rows], "labels": labels})


def leakage_check(training, validation, test, router, benchmark_type, fitted=None):
    """Count unique intersections AND matching right-hand rows; fail hard on leakage.

    Pattern/feature overlap is reported, not prohibited. Disjoint text does not
    imply independent semantics. Entire generated training data is checked, not
    only the examples that have labels suitable for fitting.
    """
    validate_design(load_config("evaluation"), benchmark_type)
    sets = {"training": training, "validation": validation, "test": test}
    if fitted is not None:
        sets["fitted_training"] = fitted
    encoded = {}
    for name, rows in sets.items():
        encoded[name] = {key: [] for key in ["request_id", "exact_prompt", "request_content", "normalized_prompt_pattern", "encoded_feature_vector", "template_id"]}
        for row in rows:
            request = row["request"]
            values = {
                "request_id": request["request_id"],
                "exact_prompt": request["prompt"],
                "request_content": json.dumps({k: v for k, v in request.items() if k != "request_id"}, sort_keys=True),
                "normalized_prompt_pattern": re.sub(r"\d+", "<N>", " ".join(request["prompt"].lower().split())),
                "encoded_feature_vector": json.dumps(encode_canonical(router.prepare(RoutingRequest.model_validate(request)).canonical()), sort_keys=True),
                "template_id": template_id(row),
            }
            for key, value in values.items():
                encoded[name][key].append(value)
    pairs = list(combinations(["training", "validation", "test"], 2))
    if fitted is not None:
        pairs.append(("fitted_training", "test"))
    records = []
    failures = []
    for left, right in pairs:
        for key in encoded[left]:
            a, b = set(encoded[left][key]), set(encoded[right][key])
            overlap = len(a & b)
            prohibited = key in {"request_id", "exact_prompt", "request_content"} or (
                benchmark_type == "TEMPLATE_HELD_OUT" and right == "test" and key == "template_id")
            if overlap and prohibited:
                failures.append(f"{left}/{right}: {key}={overlap}")
            records.append({"benchmark_type": benchmark_type, "left_split": left, "right_split": right,
                "check": key, "shared_unique_count": overlap,
                "right_rows_matching": sum(value in a for value in encoded[right][key]),
                "prohibited": prohibited, "passed": not (overlap and prohibited)})
    result = pd.DataFrame(records)
    if failures:
        raise ValueError("Research leakage detected: " + "; ".join(failures))
    return result


def ml_provenance(router, training, validation, test, training_labels, validation_labels, test_labels, fitted, benchmark_type):
    cfg = load_config("evaluation")
    ml = router.strategies["ml"]
    provenance = {
        "benchmark_type": benchmark_type,
        "training_dataset_hash": dataset_hash(training, training_labels),
        "validation_dataset_hash": dataset_hash(validation, validation_labels),
        "test_dataset_hash": dataset_hash(test, test_labels),
        "training_seed": cfg["ml_training_seed"], "validation_seed": cfg["ml_validation_seed"],
        "test_seed": cfg["random_seed"] if benchmark_type == "IID_SYNTHETIC" else cfg["held_out_test_seed"],
        "training_request_count_generated": len(training), "training_request_count_fitted": len(fitted),
        "training_request_count_excluded": len(training) - len(fitted),
        "validation_request_count": len(validation), "test_request_count": len(test),
        "training_template_ids": sorted({template_id(r) for r in training}),
        "validation_template_ids": sorted({template_id(r) for r in validation}),
        "test_template_ids": sorted({template_id(r) for r in test}),
        "feature_schema_version": FEATURE_SCHEMA_VERSION, "ground_truth_version": cfg["ground_truth_version"],
        "policy_version": router.policy_config["version"], "model_catalog_version": router.catalog_config["version"],
        "routing_config_version": router.config["version"], "evaluation_config_version": cfg["version"],
        "ml_algorithm": "sklearn.ensemble.RandomForestClassifier",
        "ml_hyperparameters": ml.classifier.get_params(), "ml_random_state": cfg["ml_training_seed"],
        "selection_method": "Fixed hyperparameters specified before validation; no final-test tuning or refitting",
        "input_mode": cfg["primary_input_mode"],
    }
    provenance["training_config_hashes"] = {name: fingerprint(value) for name, value in {
        "routing": router.config, "policies": router.policy_config, "catalog": router.catalog_config, "evaluation": cfg}.items()}
    provenance["source_hashes"] = {str(path.relative_to(ROOT)): sha256(path.read_bytes()).hexdigest()
        for folder in ["app", "scripts", "evaluation"] for path in sorted((ROOT / folder).rglob("*.py"))}
    provenance["model_fingerprint"] = fingerprint({key: provenance[key] for key in [
        "ml_algorithm", "ml_hyperparameters", "training_dataset_hash", "feature_schema_version", "ml_random_state",
        "ground_truth_version", "policy_version", "model_catalog_version", "routing_config_version", "training_config_hashes", "source_hashes"]})
    provenance["model_fingerprint_kind"] = "Deterministic recipe fingerprint; not a serialized model artifact hash"
    return provenance

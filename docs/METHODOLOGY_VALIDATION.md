# Methodology v2 implementation and validation

Validated on 2026-09-28. This extends the existing project; policy logic and live integrations were not rebuilt. No live Jev/OpenAI calls or downstream model execution occurred.

## Files changed

| File | Change |
|---|---|
| `README.md` | Experiments, terminology, commands, limitations and compatibility |
| `config/evaluation.yaml` | Explicit template pools, validation and held-out seeds/sizes, fixed ML settings, input mode and label version |
| `scripts/generate_banking_benchmark.py` | Stable template registry, original templates retained, 14 held-out variants, optional template pools |
| `app/core/router.py` | Shared normalized/policy-filtered preparation snapshot and canonical ranking path |
| `app/strategies/base.py` | Canonical structured interface for existing local rankers |
| `app/strategies/ml.py` | Full common-input encoding and configurable fixed forest parameters |
| `app/strategies/llm.py` | Unavailable canonical benchmark interface; existing application adapter retained |
| `evaluation/benchmark.py` | Split lifecycle, validation, leakage gates, per-run artifacts and provenance |
| `evaluation/metrics.py` | Precise agreement aliases, routed-only denominators, NO_ROUTE counts, p50 and metadata |
| `evaluation/statistical_tests.py` | Per-run destination, precise reference-agreement terminology and metadata |
| `evaluation/calibration.py` | Per-run destination, precise agreement alias and metadata |
| `evaluation/cost_quality_frontier.py` | Per-run destination |
| `ui/streamlit_app.py` | Benchmark/run selectors, split provenance and precise agreement presentation |
| `docs/RESEARCH_METHOD.md` | Updated experiment definitions and link to complete methodology |

## Files added

- `app/models/routing_input.py`: common structured schema and ML encoding.
- `evaluation/research_design.py`: deterministic splits, hashes, leakage checks and model recipe provenance.
- `evaluation/reporting.py`: current/legacy report loading and common presentation metadata.
- `tests/test_research_methodology.py`: integrity, candidate parity, provenance, offline and UI checks.
- `docs/BENCHMARK_METHODOLOGY.md`: complete research protocol and limitations.
- `docs/METHODOLOGY_VALIDATION.md`: this implementation/validation record.
- New per-run artifacts in the report directories below. Original report/data files were not rewritten.

## Validation performed

`python -m pytest -q`: **88 passed**, one existing Starlette/httpx deprecation warning. The original 71 tests remain passing. New tests cover independent IDs, train/validation template disjointness from held-out test, train-only fitting, deterministic hashes, recorded seeds/hyperparameters, benchmark/evidence metadata, Jev mock eligibility, identical canonical inputs, mutation isolation, eligible selections, NO_ROUTE, legacy report loading, UI labels, deliberately injected template/exact-request leakage, seed mismatches, derived report metadata, and no network calls even with configured credentials.

Both actual generated report sets were loaded using Streamlit AppTest in Executive / Research, switching between IID and held-out. Both rendered without exceptions. Additional pytest AppTest coverage exercises both selectors and synthetic agreement labels.

Checksums for **25 pre-existing report/data files** matched their pre-change values, including historical manifests/results and the original 1,000-request dataset/reference labels. Source hashes in the final two trained-run manifests match the implementation. No policy-behavior bug was found or changed.

## Commands and final report locations

With the project environment activated:

```bash
python -m evaluation.benchmark
python -m evaluation.benchmark --benchmark-type IID_SYNTHETIC --train-ml
python -m evaluation.benchmark --benchmark-type TEMPLATE_HELD_OUT --train-ml
```

All three commands completed successfully. New invocations create new run IDs.

- Default IID, ML untrained: [`reports/iid_synthetic/20260928T153208-96aa7fc2/`](../reports/iid_synthetic/20260928T153208-96aa7fc2/).
- Final trained IID: [`reports/iid_synthetic/20260928T153247-eb297b76/`](../reports/iid_synthetic/20260928T153247-eb297b76/).
- Final trained held-out: [`reports/template_held_out/20260928T153148-82538755/`](../reports/template_held_out/20260928T153148-82538755/).

Each trained directory contains benchmark results/summary, run manifest, leakage check, training/validation/test dataset snapshots and reference labels, configuration snapshot, validation results/summary, statistical comparisons, calibration results/summary and synthetic frontier. An earlier IID verification run is retained as a separately versioned run; use the final paths above for this record.

## ML methodology and observed results

Both trained experiments use 600 generated TRAIN requests at seed **1042**, of which **464** have preferred-model labels and are fitted; **136** are excluded. VALIDATION has **200** requests at seed **1043** and is never fitted. Forest settings are fixed: 60 trees, depth 10, one worker, random state 1042. No test-driven tuning or validation refit occurs.

IID final test: **1,000**, seed **42**, original templates 001–014. Held-out final test: **400**, seed **2042**, templates 015–028. Both development splits use templates 001–014 only. One additional template per task keeps all 14 task types represented.

| Observed ML metric | IID | Template-held-out |
|---|---:|---:|
| Routed requests | 745 | 310 |
| Correct NO_ROUTE | 255 | 90 |
| Preferred reference-route agreement, including correct abstentions | 98.6% | 97.5% |
| Acceptable reference-route agreement, including correct abstentions | 99.0% | 98.5% |
| Preferred reference-route agreement, routed only | 98.1208% | 96.7742% |
| Acceptable reference-route agreement, routed only | 98.6577% | 98.0645% |
| Measured downstream quality evaluations | 0 | 0 |

Validation for the fixed forest: 144 routed, 56 correct NO_ROUTE, preferred agreement 99.0% and acceptable agreement 99.5% including correct abstentions. These are diagnostics, not claims about measured response quality.

The v2 encoder includes full common metadata and eligible model descriptors, unlike historical v1. Do not reinterpret the historical 85.0% / 98.4% results or treat this as an isolated improvement from a train/validation split. Both final experiments have the same training recipe fingerprint, `6c0c72ece57e394536d21b883bf17266dc2b07136cedd3252a0b89c39524c1b0`. This is a recipe fingerprint, not a persisted model checksum.

## Leakage outcomes

All generated TRAIN/VALIDATION/TEST pairs have **zero exact ID, prompt and request-content overlap**.

| Fitted TRAIN vs final TEST check | IID | Template-held-out |
|---|---:|---:|
| Exact request IDs/prompts/content | 0 / 0 / 0 | 0 / 0 / 0 |
| Test rows matching a normalized training prompt pattern | 565 / 1,000 | 0 / 400 |
| Test rows matching an encoded training feature vector (v2 schema) | 0 / 1,000 | 0 / 400 |
| Shared unique template IDs | 14 (expected) | 0 |

Held-out VALIDATION vs TEST also has zero overlap for all six checks. IID generated TRAIN (including excluded rows) matches patterns in 645 test rows; IID VALIDATION matches patterns in 323 test rows. Detailed unique-value intersections and matching-row counts are in each `leakage_check.csv`.

The zero v2 feature-vector overlap does not negate the historical v1 finding of 13 matching vectors; the feature definitions changed. Template/pattern and encoded-vector disjointness do not imply independent semantics or realistic banking language.

## Compatibility and remaining limits

Historical files remain byte-for-byte unchanged. Legacy CSV columns still load and retain their definitions; precise aliases and routed-only metrics are added in memory when necessary. New benchmark runs write unique directories, not top-level historical CSVs. The default IID CLI remains available. Direct ML fitting now consumes canonical inputs; the historical encoding helper remains for reconstruction. The primary benchmark rejects `--execute` and rich-input mode; the existing application adapter and execution service remain outside this experiment.

This is synthetic agreement with an explicit cost/quality-band objective, not empirical model quality or real banking performance. Template holdout keeps familiar domains and task types and shares vocabulary. It mostly tests the feature extractor and resulting structured data; the forest does not read raw language. Statistical analyses remain exploratory and do not implement clustered uncertainty intervals or multiplicity correction. Future live work needs verified canonical Jev/OpenAI adapters, approved endpoints, independently reviewed task references, real routing cost/latency measurements, and downstream quality evaluation under a separately specified protocol.

# Synthetic benchmark methodology, version 2

Reference-route agreement measures agreement with the project’s synthetic routing objective. It does not measure actual downstream model response quality.

## Experiments and split boundaries

| Experiment | ML TRAIN | ML VALIDATION | FINAL TEST |
|---|---|---|---|
| `IID_SYNTHETIC`: IID / Template-Familiar Synthetic Benchmark | 600 generated, seed 1042, original templates 001–014 | 200, seed 1043, original templates 001–014 | Existing 1,000 requests, seed 42, original templates 001–014 |
| `TEMPLATE_HELD_OUT`: Template-Held-Out Generalization Benchmark | 600 generated, seed 1042, original templates 001–014 | 200, seed 1043, original templates 001–014 | 400, seed 2042, new templates 015–028 |

IID results evaluate new synthetic samples generated from a familiar template family. Template-held-out results evaluate requests generated from templates excluded from ML training and validation.

IID means independent seeded draws from the familiar generator, with a deterministic task cycle rather than strictly independent identically distributed draws in the mathematical sense. This is not completely unseen banking data. Original request content is preserved in `data/banking_benchmark.json`; IDs are assigned in memory to legacy rows. The legacy generator's default call keeps its original format. Explicit template pools produce `template_id` in generated rows, and all new per-run dataset snapshots include it.

The original generator had one template per task. Withholding four of those would also withhold four task types. Instead, we added a second, structurally different prompt per task: 14 training templates and 14 test templates. Training and validation share the allowed template pool but have separate seeds, IDs and request content. The 400 held-out rows provide 28 or 29 requests per task/template. Domains remain familiar. This tests unseen authored templates, not unseen task categories, domains or real banking language. These are still short, related synthetic templates and share vocabulary. Because ML sees structured features rather than language, template holdout primarily exercises feature extraction and the resulting structured inputs; it is not evidence of semantic language understanding.

Configuration lives in `config/evaluation.yaml`: `ml_training_size`, `ml_validation_size`, three distinct seeds, `held_out_test_size`, `training_templates`, `validation_templates`, and `test_templates`. The `--size` CLI option overrides final-test size only. Changing these settings defines a new experiment; do not tune them after viewing final-test results and then call the same test untouched.

## ML lifecycle

1. Generate TRAIN and VALIDATION from their permitted pools.
2. Build synthetic reference labels independently for each split.
3. Exclude TRAIN requests with no preferred-model label. With default settings this preserves 600 generated / 464 fitted / 136 excluded. The policy layer handles abstentions; the classifier does not learn a NO_ROUTE class.
4. Check exact request leakage across all three splits before fitting. In held-out mode, enforce disjoint final template IDs from BOTH development splits.
5. Fit `DictVectorizer` only on the fitted TRAIN examples, then `RandomForestClassifier(n_estimators=60, max_depth=10, n_jobs=1, random_state=1042)`. All other sklearn parameters are recorded too.
6. Evaluate the fixed classifier on all 200 VALIDATION requests. Save `validation_results.csv` and `validation_summary.csv`. No hyperparameters, thresholds, features or calibrators are selected using final-test data; no validation refit or large parameter search is performed. Validation is a diagnostic check of settings fixed in advance.
7. Evaluate the unchanged classifier and other routers on FINAL TEST. Fallback is disabled. Final rows and labels cannot be substituted after the pre-fit integrity check.

The v2 ML encoder includes all canonical metadata, inferred features and eligible-candidate descriptors. It uses stable model IDs for candidate feature keys. Missing values and empty lists have explicit markers. The raw prompt, request ID, application ID, generator task label, template ID and reference labels are not ML input. The fitted vectorizer ignores feature categories absent in training as usual. The legacy `encode()` helper remains only for historical v1 reconstruction.

The new feature schema is `canonical-structured-2.0`. New ML results are not reproductions of the v1 model's 85.0% / 98.4% agreement values. Historical files under `reports/` and `reports/trained_ml/` remain unchanged, and the UI identifies their incomplete split provenance rather than retroactively certifying it.

## Synthetic reference-label objective

`ground_truth()` uses the generator-known task type and the configured policy engine to determine eligible models. It finds the best eligible synthetic task-quality score, then includes every eligible model whose configured score is at least `best - acceptable_quality_tolerance`.

`acceptable_quality_tolerance` is already configurable and defaults to **0.12**. The preferred reference model is the cheapest model in that acceptable set, using normalized input context and reserved output tokens with catalog input/output prices. Ties break by model ID. No eligible model produces an expected NO_ROUTE label.

This objective is derived from configured policy constraints, model-quality assumptions, and model-cost assumptions. It is **not** human expert annotation, production routing truth, measured downstream response quality, or proof that the chosen model is objectively best. It favors cost-efficient choices within a configured quality band by construction. Reference labeling uses generator task intent while every router uses the same keyword-inferred task; those can differ. Governance validation shares the project's policy implementation and is not independent certification.

## Fair primary comparison

`ModelRouter.prepare()` normalizes the request, extracts features, snapshots the catalog and filters candidates once per benchmark request. Every strategy receives an isolated copy of the same `CanonicalRoutingInput`: normalized metadata, inferred task/complexity/reasoning/enterprise needs, and full eligible model descriptors. Metadata includes classification, domain, modality, context/output tokens, preferences, quality, latency SLA and RAG needs.

Strategies can use this common information differently, but none receives raw prompt text at ranking time. All local algorithms use the same structured boundary; ML encodes the full input. Rules and Weighted retain their existing formulas. Existing policy behavior is unchanged. Candidate IDs and canonical-input hashes are recorded per result. Invalid selections fail closed; a no-candidate request remains valid NO_ROUTE for an available router. Unavailable integrations remain UNAVAILABLE.

The optional application OpenAI adapter is unchanged; the primary comparison's canonical adapter is an unavailable interface stub. No live calls occur, even if keys exist. Jev remains its labeled weighted-scoring mock. `STRUCTURED_PLUS_PROMPT` is a reserved future secondary experiment; primary runs reject any input mode other than `STRUCTURED_ONLY`. No downstream models are executed.

Measured routing latency includes shared normalization/filter preparation plus each strategy's isolated routing/validation. It excludes auditing, report generation, training, leakage checking and downstream execution. Deep-copy overhead for the common snapshot is included for all strategies. p50/p95 are request-level routing-time quantiles. Mean synthetic model p95 values are not empirical end-to-end latency.

## Metrics and metadata

The umbrella label is **Synthetic Reference-Route Agreement**. Report both preferred and acceptable reference-route agreement:

- **Including Correct Abstentions:** denominator is available decisions; correct expected NO_ROUTE counts as agreement. Invalid decisions remain disagreements. Unavailable decisions are excluded and counted separately.
- **Routed Requests Only:** denominator is ROUTED decisions only; always report routed count and unavailable rate alongside it.

`correct_no_route` is a count, not classifier performance. Policy-driven abstentions can raise overall agreement. Synthetic quality proxies are separate from downstream quality, which remains blank with zero evaluated responses.

Compatibility aliases `preferred_model_accuracy` and `acceptable_route_accuracy` retain their previous numerical definitions. Precise columns are `preferred_reference_route_agreement_including_correct_abstentions`, `acceptable_reference_route_agreement_including_correct_abstentions`, and their `_routed_only` counterparts. Legacy calibration `actual_accuracy` is retained with an `acceptable_reference_route_agreement_routed_only` alias. Legacy boolean columns `preferred_correct` and `acceptable_correct` mean agreement with the synthetic reference.

Each new primary row and summary records benchmark type, `evidence_type=SYNTHETIC`, execution mode, research eligibility, dataset hash, run ID, UTC timestamp, strategy version, input mode and phase. Modes describe implementation paths: LOCAL for local routers, MOCK for Jev mock, LIVE reserved for external adapters. An unavailable row with LIVE does not imply an API call; `live_calls_enabled=false` and status make that explicit.

`research_eligible` means evidence about that implementation under the stated synthetic benchmark. It does not imply production performance. Jev mock and unavailable adapters are false. ML requires verified split provenance; historical ML is not retroactively certified. Summary eligibility indicates that at least one qualifying decision exists; examine coverage and row eligibility. Statistics involving Jev mock are mock diagnostics, never H4 evidence.

## Leakage and reproducibility

`leakage_check.csv` reports generated TRAIN vs VALIDATION, TRAIN vs TEST, VALIDATION vs TEST, and fitted TRAIN vs TEST. Each pair reports shared unique values and matching right-hand rows for request IDs, exact prompts, exact request content excluding ID, normalized prompt patterns, encoded feature vectors and template IDs. Patterns lowercase text, collapse whitespace and replace digit runs with `<N>`.

Exact request/ID/prompt overlap fails either benchmark. In held-out mode, any final-test template overlap with either development split fails before fitting or reporting. Pattern and feature overlap are disclosed, not treated as proof of leakage. IID template overlap is expected. Feature overlaps use the full v2 encoder; historical v1's 13 feature-vector matches used a smaller schema and are not directly comparable.

Every run goes into a unique directory `reports/<benchmark_type_lowercase>/<run_id>/`. Existing run directories cannot be overwritten. Trained runs contain split datasets and reference labels, validation reports, leakage reports, configuration snapshots, and a manifest with:

- TRAIN / VALIDATION / TEST dataset hashes, seeds, counts and template IDs;
- generated/fitted/excluded training counts;
- feature, ground-truth, policy, catalog, routing and evaluation versions;
- algorithm, all resolved hyperparameters, random state, package versions and source/configuration hashes;
- deterministic `model_fingerprint` of the training recipe, including training hash, feature schema, seed, parameters and implementation/config hashes.

The classifier remains in memory. The model fingerprint is **not** a serialized artifact checksum. It identifies a recipe to reproduce in the recorded environment. Dataset hashes include annotated template IDs and labels, so they intentionally differ from historical unannotated hash definitions. Timing, timestamps and run IDs are nondeterministic. Default per-run statistical/calibration analyses are diagnostic, not training. Shared templates require clustered uncertainty analysis before publication; the current bootstrap is exploratory.

## Commands

From the project root with the virtual environment active:

```bash
python -m evaluation.benchmark --benchmark-type IID_SYNTHETIC --train-ml
python -m evaluation.benchmark --benchmark-type TEMPLATE_HELD_OUT --train-ml
python -m streamlit run ui/streamlit_app.py
```

The old `python -m evaluation.benchmark` command still runs IID with optional ML unavailable. `--execute` is rejected in these synthetic primary experiments. The existing downstream execution service and application adapter are not extended here. Historical standalone analysis commands still target legacy report locations; use the benchmark CLI for isolated, complete new runs.

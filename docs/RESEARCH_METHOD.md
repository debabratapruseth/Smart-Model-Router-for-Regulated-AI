# Research method

Research question: **Can policy-aware routing reduce inference cost and latency while maintaining response quality and zero policy violations in a regulated banking environment?**

## Hypotheses and what V1 can establish

- **H1:** intelligent routing reduces mean inference cost relative to strongest-eligible routing. V1 compares synthetic price estimates on jointly served requests. It cannot establish actual vendor savings.
- **H2:** intelligent routing reduces latency without materially degrading downstream quality. The configured quality noninferiority margin is 0.03. V1 distinguishes measured routing time, synthetic model p95 estimates, synthetic quality proxies, and optional measured downstream scores. H2 remains untested without real task execution and quality references; a proxy decrease is not proof of task-quality loss or preservation.
- **H3:** policy-first architecture produces zero violations on the benchmark. Validate returned selections by rechecking the policy against effective models. This shares implementation with the router and is not independent certification; hand-designed adversarial cases and direct constraint assertions provide additional evidence. Report violations over all available-strategy decisions AND over routed decisions. NO_ROUTE is not a violation.
- **H4:** a real Jev bounded-decision router lowers routing latency and/or cost relative to an LLM selector. Mock Jev delegates to weighted scoring and cannot support this hypothesis. Integrate a verified API and measure real routing spend, retries and latency before testing H4.
- **H5:** rolling observations improve resilience over static metrics. Inject identical shocks and compare stale static state against dynamic observations; report compliance under known policy separately from suitability under the simulated world. Cost changes are catalog updates available to both modes; cost doubling is not telemetry evidence for H5.

## Workload and labels

The generator uses a fixed seed, 16 domains, 14 original task templates plus 14 held-out variants, five difficulties, four classifications, varied lengths, preferences and optional sensitive-data flags. It contains no customer data. Audio/image cases are metadata-only. EASY/MEDIUM/HARD/AMBIGUOUS/ADVERSARIAL are generator strata, not measured task difficulty. Templates with randomized scenario identifiers are correlated; 1,000 rows do not constitute 1,000 independent natural-language tasks.

Ground truth derives task intent from generator labels, not keyword inference. A synthetic quality band (best eligible minus configurable tolerance) defines multiple acceptable models; the cheapest within that band is preferred. Eligibility references the shared policy specification. This is a transparent reference objective, not expert ground truth. It favors cost-efficient, quality-bounded strategies by construction. Commission blinded expert labels and alternative objectives before research claims.

The optional classifier trains on a separately seeded TRAIN set, evaluates fixed settings on separate VALIDATION data, then runs FINAL TEST. See [methodology v2](BENCHMARK_METHODOLOGY.md) for IID and template-held-out splits, integrity gates and model provenance. Shared synthetic vocabulary and catalog quality still limit generalization claims. Domain and temporal holdouts remain future work.

## Denominators and timing

Unavailable integrations have missing agreement/cost/quality and an explicit unavailable rate; they are not silently replaced. Correct abstentions count toward acceptable/preferred reference-route agreement. Report routed-only quality/cost/latency and coverage alongside these agreement rates. No-route cost is missing, not zero, so reduced coverage does not artificially create free inference.

Raw strategy comparisons disable fallback. Normal API routing enables documented fallback; use a separate experiment to assess it. Model usage proportions are over routed requests. Route stability is measured separately on two illustrative paraphrase pairs, with both-routed indicators. Two pairs are not enough for a robustness claim.

Routing latency is measured from normalization through validation, excluding audit/telemetry I/O and model execution. Catalog p95 values are synthetic per-model estimates; their workload average is NOT an empirical end-to-end p95. API wall time will exceed reported router compute time. Real routing API spend is not yet included in savings or frontier charts.

## Statistics, calibration and frontier

Paired comparisons join by request ID. Exact McNemar uses discordant acceptable-correctness pairs. Bootstrap mean-difference CIs and Wilcoxon signed-rank comparisons use jointly served pairs, avoiding different no-route denominators. Wilcoxon assumes approximately symmetric differences. P-values are exploratory; multiple-testing correction is not implemented. Cluster bootstrap by template/domain is needed for formal inference. Zero observed violations does not imply a zero upper confidence bound.

Brier score and expected calibration error evaluate confidence against binary acceptable-route correctness, conditional on a routed decision. Reliability buckets report sample counts. ML confidence is an eligible class's original probability, not the summed probability of all acceptable classes. Jev mock utility is not a probability. Calibration on the test set is diagnostic only, not a fitted calibrator.

Pareto data compares synthetic cost, synthetic latency and synthetic quality proxy; it is not a real performance frontier. Compare identical served cohorts before claiming domination where strategies have different availability or coverage. Do not mix task-specific quality scores without checking comparability.

## Ablation, sensitivity and resilience

Ablations remove ranking information only. Original context, declared quality and enterprise requirements stay mandatory policy inputs. A zero effect for quality-requirement ablation is expected when the ranker does not directly consume that field; do not interpret it as an unimportant governance control. Context ablation changes ranking cost estimates, but reported actual request estimates still use original context.

Preference sensitivity changes four configured weight profiles and records distribution, cost, latency, reference-route agreement and quality proxies. Resilience injects model/provider/region outages, rate limits, doubled latency/cost and failure-rate changes. The dynamic router observes shocks; static selection may choose a now-unavailable model without violating its known-state policies. Record correct abstentions separately from switches to avoid calling no-route a recovery.

## Reproduction

Run generation, tests, a default benchmark, then the analysis modules in README order. Reports include dataset/config hashes, seeds, versions and package versions. `requirements-lock.txt` captures the validation environment; `requirements.txt` allows supported dependency ranges. Randomized data, ML and bootstraps have seeds; wall-clock latency, timestamps and audit UUIDs vary across runs. New benchmark commands create isolated per-run directories. Legacy standalone studies still target their original report paths.

Next steps: independently labeled held-out banking-style tasks; approved real execution endpoints with task-specific references; power analysis and clustered uncertainty intervals; a larger paraphrase/adversarial suite; measured end-to-end cost including router overhead; verified Jev integration; independent policy-engine validation; and prospective telemetry experiments.

Reference-route agreement measures agreement with the project’s synthetic routing objective. It does not measure actual downstream model response quality.

# Final experiment harness

Implementation and offline verification do not execute paid APIs. The normal
pilot remains available. Final samples, primary results and repeated stability
results use separate locations. Neither final mode generates requests/labels,
fits ML, changes prompts, nor recomputes baseline strategy decisions.

Interpretation: **Empirical live routing execution evaluated against a synthetic
reference-routing objective.** The ML router was trained to approximate the
synthetic reference-routing function. Preferred and Acceptable Synthetic
Reference-Route Agreement are not measured downstream response quality.

## Frozen sampling protocol

`config/final_experiment.yaml` fixes 200 IID and 200 template-held-out requests,
sampling seed 4042, stability seed 5042 and five stability repetitions. Seeds may
be overridden with `--sampling-seed` / `--stability-seed`; different selections
receive different sample IDs and must be treated as different experiments.
Do not search seeds after seeing outcome results.

The source run IDs, full dataset hashes and frozen results-file hashes are pinned
in that configuration, taken from the successful integration pilot's provenance.
The frozen loader additionally checks policy/catalog/routing/evaluation settings
and protected implementation hashes. Final sample validation checks original
content, reference labels, template IDs, canonical hashes, candidate IDs and all
three baseline replay records, including source run, strategy and request ID.
The canonical hash covers complete structured candidate descriptors, not only IDs.

Before sampling, exclude all requests selected by the seven historical live pilot
runs pinned in `prior_live_pilot_run_ids`. The union of each run's `pilot_plan.json`
and `benchmark_results.csv` is used, regardless of success, failure, dispatch or
policy NO_ROUTE. This excludes **15 IID and 15 held-out unique requests**, leaving
sampling populations of **985 and 385**. Selection does not depend on agreement
or provider success. Each source dataset hash must match the frozen benchmark.
The sample manifest records `exclusion_reason=PRIOR_LIVE_PILOT`, excluded counts
and IDs by benchmark, source pilot run IDs, hashes of the source reports, and a
deterministic `exclusion_set_hash`. Exclusions enter the sample content hash.
Validation fails if any excluded request enters the final sample or if exclusion
provenance changes. The earlier sample is retained unchanged under its original ID.

Sampling on this filtered population is hierarchical proportional allocation without replacement:

1. Split by `expected_no_route`, allocating the 200 slots with largest remainders.
2. Within each branch, split by the existing `task_type` using the same rule.
3. Consider `data_classification`, `modality`, `cost_preference`, `complexity`,
   then `reasoning_requirement` in that order. Use a secondary split only if
   every nonempty child receives at least two slots. Otherwise skip that field
   in that branch and consider the next one.
4. Sample uniformly from the remaining leaf using a seeded Python RNG. Sort
   source pools by request ID first. Fractional-allocation ties break by sorted
   serialized stratum labels; integer arithmetic avoids floating-point ties.

No full cross-product of tiny strata is used. Task and modality are correlated in
this generator; finer marginal distributions are approximate and audited. The
manifest retains every used stratum's source count and allocated sample count,
and before/after marginal counts for every listed field. `distributions_before`
describes the filtered sampling population (`sampling_pool_size`); `source_size`
continues to describe the complete frozen dataset. Integer rounding and the
existing treatment of sparse secondary strata remain unchanged.

| Benchmark | Primary sample | Expected NO_ROUTE | Routable | Stability requests |
|---|---:|---:|---:|---:|
| IID_SYNTHETIC | 200 | 51 | 149 | 25 |
| TEMPLATE_HELD_OUT | 200 | 45 | 155 | 25 |
| Total | 400 | 96 | 304 | 50 |

Stability uses the same sampler on routable requests within the selected primary
sample. Its 25+25 requests are subsets of that sample; no policy abstentions are
repeated. Each repetition completes a full pass before the next pass begins.

`data/final_experiment/<sample_id>/sample_manifest.json` stores the sample manifest
and exact copies of selected source rows, labels and canonical/candidate metadata.
It records seed/method/version, source and sample hashes, request IDs, distributions,
strata, template provenance, creation time, Python version and git commit (null
when the project is not a Git repository). Existing samples are verified and read,
never overwritten. The deterministic content hash excludes creation timestamps.

## Dry-run and authorization

Run these from the project root in the existing virtual environment:

```bash
python -m evaluation.benchmark --final-experiment --strategies rules,weighted,ml,openai,jev --dry-run
python -m evaluation.benchmark --stability-experiment --strategies openai,jev --dry-run
```

Omitting live authorization also performs preflight only. `--dry-run` always wins,
even if authorization is present. Preflight loads `.env`, reports only presence
booleans for keys, and never initializes an external SDK client. It verifies frozen
sources, immutable samples and 400 replay decisions per baseline. It prints
provider enable/mock status, model identifiers, pricing/source/sample hashes,
sample sizes, NO_ROUTE counts, and initial call counts.

Jev is pinned to the OpenRouter request identifier **`typesafe/jev-1.13`** in
`config/live_routing.yaml` and the local `JEV_ROUTER_MODEL` setting. The existing
TypeSafe SDK 0.7.0 `system_one` call uses OpenRouter's `/api/v1/systemone`
compatibility endpoint; no API surface, routing instructions or schema changed.
The [OpenRouter TypeSafe SDK contract](https://openrouter.ai/docs/guides/community/typesafe-sdk)
documents Jev 1.13 and author-prefixed model identifiers. Saved successful pilot
responses report `typesafe/jev-1.13-20260917`; that returned revision is retained
separately rather than assuming a response revision is an accepted request ID.
Preflight records `strategy=jev`, `execution_provider=openrouter`,
`execution_path=JEV_VIA_OPENROUTER` and the configured exact request identifier.
An offline SDK mock-transport test verifies serialization. No live availability
test is performed by this hygiene change. Environment overrides remain supported;
check the recorded model before a later authorized run.

Planned primary calls: **304 OpenAI + 304 Jev = 608**. Planned stability calls:
**50 requests × 5 repetitions × 2 live providers = 500**. These are initial calls;
retries consume additional attempt slots. The existing default cap of 50 remains
unchanged and is insufficient for a new final experiment. Preflight flags this.
There is no fabricated maximum-dollar estimate: native Jev charges, output usage,
served tier and retries are not known in advance. Missing maximum spend is null.

Commands below are examples for a later user-authorized run only:

```bash
python -m evaluation.benchmark --final-experiment --strategies rules,weighted,ml,openai,jev --allow-live-api --max-live-api-calls 700
python -m evaluation.benchmark --stability-experiment --strategies openai,jev --allow-live-api --max-live-api-calls 600
```

The caps allow limited retry headroom and do not guarantee completion. Optional
`--max-routing-spend-usd` retains the observed-spend stop: a single in-flight call
may exceed the threshold, and any unknown attempt charge halts further calls.
OpenAI pricing is versioned and Jev uses OpenRouter's reported charge, unchanged
from [routing-cost telemetry](ROUTING_COST_TELEMETRY.md).

Final execution refuses unavailable requested providers rather than producing an
apparently successful final comparison with missing configuration. No fallback
substitutes another router. Local strategy replay uses zero new external API cost;
this does not assert zero compute or total cost of ownership.

## Provenance, outputs and resume

Primary output: `reports/final_experiment/<run_id>/`:

- `benchmark_results.csv`, `benchmark_summary.csv`, `api_errors.csv`;
- `confidence_intervals.csv`, `statistical_comparisons.csv`, `generalization_deltas.csv`;
- `latency_summary.csv`, `paired_live_comparisons.csv`, `cost_summary.csv`;
- `run_manifest.json`, `pilot_plan.json`, `observations.jsonl`, `routing_cost_summary.csv`.

Stability output: `reports/stability/<run_id>/`:

- `stability_results.csv`, `stability_summary.csv`, `selection_consistency.csv`;
- `latency_stability.csv`, `cost_stability.csv`, `repetition_summary.csv`, `agreement_variability.csv`;
- `api_errors.csv`, manifest, plan and journal, plus compatibility benchmark/cost CSVs.

Compatibility stability summaries label counts as observations, not independent
requests. Stability analysis never enters the primary statistical pipeline.
Analysis files are written only after a completed execution has recorded results;
preflight does not create performance results. Partial runs retain observations,
including unavailable/error/indeterminate rows, and can be resumed.

New rows/manifests identify OpenAI direct (`execution_provider=openai`,
`execution_path=OPENAI_DIRECT`) and Jev via OpenRouter (`execution_provider=openrouter`,
`execution_path=JEV_VIA_OPENROUTER`). Policy abstentions explicitly use local policy
execution. Returned model versions, prompts/versions, canonical hashes, original
dataset hashes, sample hashes, pricing snapshots and implementation hashes are
retained. No keys, response bodies or chain-of-thought are recorded.

Rules/Weighted/ML retain `BASELINE_REPLAY`, `FROZEN_BASELINE` decision/latency
sources and original source run/dataset/request IDs. Replay latency is historical,
not a new measurement. Live latency includes external dispatch and existing
preparation overhead. Original routing instructions and eligible inputs are fixed.

Resume using the same mode, seeds, strategies, models and settings, adding
`--resume reports/final_experiment/<run_id>` or the corresponding stability path.
The call cap counts all previous attempts. Primary observation identity is
run/benchmark/request/strategy; stability additionally includes repetition number.
Completed successes and failures are skipped. Interrupted in-flight attempts are
indeterminate and are never automatically retried. `--force-rerun` explicitly
appends replacement revisions and may charge again; it is not a new repetition.
Changed sample, implementation, models, pricing or prompts fail the plan fingerprint
check before dispatch. No successful historical run is rewritten by this harness.

## Statistical protocol

Primary analyses are separate for IID and held-out. Four agreement estimates per
router are reported: preferred/acceptable, including correct abstentions/routed-only.
HTTP/provider/invalid-decision failures count as disagreements in the evaluated
denominator; unavailable, unexecuted, budget-stopped and indeterminate observations
are excluded and explicitly counted. Routed-only denominators contain ROUTED rows.
Pairwise tests use the intersection of evaluated request IDs; paired sample size
is always reported, so incomplete coverage is not hidden.

- 95% percentile bootstrap CIs: 10,000 request-level resamples, seed 6042. Seed
  and iteration count are configurable in the new configuration and recorded.
- Exact two-sided McNemar tests for all ten router pairs, per benchmark and
  preferred/acceptable objective. The exact test is the binomial discordance test;
  its reported statistic is `min(A correct/B wrong, A wrong/B correct)`.
- Holm-Bonferroni adjusts the ten p-values within each benchmark/objective family.
  Report raw/adjusted p-values, discordant counts, signed A-minus-B percentage-point
  difference and its absolute magnitude. A p-value is not practical importance.
- Latency descriptives: mean, median/p50, p95, sample SD and IQR. Only OpenAI/Jev
  LIVE, CURRENT_RUN, actually dispatched paired requests enter latency/cost signed-rank
  comparisons. Latency pairing requires both routes succeed. No statistical test
  compares historical baseline latency to current live latency.
- Wilcoxon: two-sided, zero differences removed (`wilcox`), SciPy automatic method;
  all-zero differences explicitly give statistic 0/p=1. Live cost/latency p-values
  are also Holm-adjusted within benchmark. Unknown costs are excluded from paired
  cost tests with the paired count reported; overall totals stay unknown.
- Cost is descriptive external routing API cost only. Report known-cost lower
  bounds, unknown counts, total/mean/median cost per API decision and cost per
  request including policy abstentions, with token totals. Retries are included.
- Generalization deltas are held-out minus IID in percentage points for all four
  agreement metrics. These are distinct populations, not paired observations;
  declines are not automatically called statistically significant.

## Stability metrics and limitations

Selection consistency is the mean per-request modal-selection rate among requests
with at least two successful repetitions. Full consistency uses that same
measurable-request denominator. Requests with zero/one success are counted
separately; a stricter all-five-successful-and-consistent rate uses all requests.
Pairwise repetition agreement is the count of equal successful selection pairs
divided by all comparable successful pairs within requests. Unique-model mean,
median and maximum use measurable requests. Per-request rows retain success counts.

Each repetition has its own preferred/acceptable agreement and error/retry counts.
Agreement variability summarizes those five repetition estimates descriptively.
Latency variability reports mean/median/p50/p95, sample SD and CV when mean>0.
Cost variability reports total, mean/median per API decision and total cost divided
by successful routing decisions (including any known failed-attempt charges).
Unknown costs remain unknown. No confidence interval treats the 500 repeated
observations as 500 independent benchmark requests.

Methodological concerns:

- The revised final sample has **zero overlap with all seven recorded prior live
  pilots**; the original sample's 2 IID / 8 held-out overlap is superseded without
  rewriting that historical artifact. These remain existing synthetic benchmarks,
  not untouched external data. Disclose earlier tuning when presenting results.
- Requests share synthetic templates. Request bootstrap assumes exchangeability
  within the chosen benchmark and is conditional on this generator/template mix;
  it is not a template-cluster or finite-population-adjusted interval. A degenerate
  all-correct bootstrap interval does not prove perfect population performance.
- Coarser proportional strata are prioritized over fine marginal balance; the
  manifest makes rounding/omitted tiny splits reviewable. No outcome labels beyond
  expected policy abstention influence selection.
- Success-conditional stability can look strong when many calls fail; always show
  coverage and the all-five-successful rate. Replications are temporally ordered,
  not independent. Provider order is fixed, and cache/network drift can affect
  latency and cost. Jev 1.13 is pinned; returned provider revisions are also recorded.
- API-attempt and spend guards cannot prove remote billing after an interruption.
  Historical replay is eligible for exact routing-decision comparisons, not claims
  of contemporaneous baseline speed or real banking production quality.

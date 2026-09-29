# Live routing pilots over frozen synthetic benchmarks

**Empirical live routing execution evaluated against a synthetic reference-routing objective.**

This extension calls a provider to select an eligible model. It never executes that selected downstream model. Routing API cost is not downstream inference cost; routing latency is not downstream model latency; synthetic reference-route agreement is not downstream response quality or production routing accuracy.

## Verified provider contracts

Verified against official documentation and the installed package sources on 2026-09-29:

- **OpenAI Python SDK 2.54.0:** `OpenAI(..., max_retries=0).responses.parse(..., text_format=LiveRoutingDecision)`, using strict Structured Outputs and local validation. See [official Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs) and [official SDK documentation](https://developers.openai.com/api/docs/libraries). The primary endpoint is fixed to `https://api.openai.com/v1`; legacy `OPENAI_BASE_URL` does not redirect these research calls.
- **TypeSafe SDK 0.7.0:** `TypeSafeClient(..., retry=RetryPolicy(max_retries=0)).system_one(state=..., questions={'selected_model': Choice(...)}, response_model=RoutingChoiceResponse)`. Criteria keys are exactly the eligible model IDs. See [official Python SDK](https://docs.typesafe.ai/sdk/python), [SDK usage](https://docs.typesafe.ai/sdk/python/usage), [official quickstart](https://docs.typesafe.ai/introduction/quickstart), and [official SDK source](https://github.com/typesafe-ai/typesafe-sdk-python). The verified endpoint is `https://api.typesafe.ai/v1/systemone`, with SDK-managed bearer authentication. The documented response supplies choice, native confidence, model and token usage. A custom documented response-model extension permits absent confidence as null. No reasons are invented: Jev reason codes are empty because this Choice response has no native routing reason codes.

These contracts were tested using both mocked SDK calls and real SDK serializers with local HTTP mock transports. **No credentialed live provider call has been performed during implementation.** A verified contract and passing mocks do not prove account access, model availability for your account, latency or actual billing.

## Installation and configuration

Activate the existing virtual environment and install `requirements.txt` or `requirements-lock.txt`. Neither SDK was installed before this change. New settings are isolated in `config/live_routing.yaml`; the frozen routing/evaluation/policy/catalog configuration is unchanged.

Set values in the ignored `.env` or your shell. Never put keys in YAML:

```dotenv
OPENAI_API_KEY=<your key>
OPENAI_ROUTER_ENABLED=true
OPENAI_ROUTER_MODEL=<chosen supported model or snapshot>
TYPESAFE_API_KEY=<your key>
JEV_ROUTER_ENABLED=true
JEV_ROUTER_MODEL=jev-latest
JEV_MOCK_MODE=false
```

Both enable flags default false, and Jev mock mode defaults true. Environment model/enable settings override the corresponding YAML fields. The OpenAI model defaults to null: a key alone does not choose a model or authorize calls. One documented compatible example is `gpt-4.1-mini-2025-04-14`; its [official model page](https://developers.openai.com/api/docs/models/gpt-4.1-mini) lists Structured Outputs support. This is an example, not a hard-coded default or a claim it is the optimal router. Prefer an explicit model snapshot when available. Returned model versions are recorded; aliases such as `jev-latest` may change server-side.

Timeouts default to 30 seconds, maximum retries to 2, exponential backoff to 0.25 then 0.5 seconds. OpenAI output tokens default to 512. `temperature` defaults to null and is omitted; set it to 0 only for a chosen model that supports that parameter. No unsupported generation parameter is forced.

The ordinary application/offline LLM facade is unavailable and cannot dispatch calls simply because keys exist. Its previous direct HTTP path was replaced by the explicit research adapter. Jev mock still works in local development. The live runner does not fall back to it even if `JEV_MOCK_MODE=true`; that provider observation is unavailable instead.

## Frozen inputs and baseline replay

The runner reads only these completed datasets/results:

- IID: `reports/iid_synthetic/20260928T153247-eb297b76/`
- Template-held-out: `reports/template_held_out/20260928T153148-82538755/`

It does not call dataset generation, reference-label generation or ML fitting. All current frozen configuration hashes and protected policy/feature/scoring/ML/generator source hashes must match the recorded baseline. Each dataset and label snapshot must reproduce its recorded hash. The model catalog, policies, feature extraction, ML settings, template split and 0.12 reference-quality tolerance are unchanged.

For each selected request, normalization, feature extraction and policy filtering produce one canonical eligible-model snapshot. Every strategy row records its hash. OpenAI receives that canonical object as JSON; Jev receives the same object as structured `state`, with eligible IDs repeated as its bounded choices. Request IDs, raw prompts, generator reference labels, preferred/acceptable reference models and expected answers are absent from provider inputs.

Rules, Weighted and ML are **BASELINE_REPLAY**, as explicitly requested. The runner verifies the unique `(request_id, strategy)` source match, dataset hash, stored canonical hash when available, candidate IDs, selected-model eligibility, current policy validity, confidence bounds and recorded latency. It preserves the selected model, status, confidence, recorded reason codes and original latency. Missing or ambiguous matches become `BASELINE_REPLAY_UNAVAILABLE`; there is no automatic reconstruction or retraining.

Replay rows contain:

- `execution_mode=BASELINE_REPLAY`, `evidence_type=SYNTHETIC`, `decision_source=FROZEN_BASELINE`;
- `research_eligible=true` for a verified exact replay;
- `source_run_id`, `source_dataset_hash`, `source_request_id`;
- `latency_measurement_source=FROZEN_BASELINE`.

Replay latency is historical, not contemporaneous pilot compute time. Do not claim a newly measured ML latency advantage over a current network call.

## Pilot commands — run deliberately

From the project root, with keys and enable settings configured:

```bash
python -m evaluation.benchmark --live-pilot --strategies rules,weighted,ml,openai,jev --allow-live-api
```

OpenAI only:

```bash
python -m evaluation.benchmark --live-pilot --strategies openai --allow-live-api
```

Jev only:

```bash
python -m evaluation.benchmark --live-pilot --strategies jev --allow-live-api
```

The pilot samples **15 IID and 15 held-out requests**, configurable in the new live YAML, at pilot seed **3042**. This is subset selection from frozen rows, not dataset regeneration. Routable/NO_ROUTE quotas approximately follow each dataset's proportions; a seeded greedy sampler favors coverage of task, inferred complexity, classification, cost preference and quality requirement. At this size, every joint stratum cannot be represented.

With the defaults, the plan contains 4 IID and 3 held-out NO_ROUTE cases: **23 routable requests per live strategy**, or **46 initial external attempts** for both providers. NO_ROUTE cases make zero external calls. Those rows identify `decision_source=DETERMINISTIC_POLICY`, `execution_mode=LOCAL`, and are not empirical evidence of provider execution.

The existing synthetic benchmark commands remain offline. The live CLI rejects `--train-ml`, `--execute`, `--size`, or synthetic output-directory overrides. No downstream execution was added.

## Call limits, pricing and retries

`--allow-live-api` is mandatory whenever live strategies are requested. Keys and YAML flags alone cannot authorize the pilot. The default API-attempt cap is **50**. Planned initial calls plus previous attempts are checked before execution; if they exceed the cap, the run aborts before sending anything. Every retry reserves another slot. There are no hidden SDK retries.

A full run requires `--live-full` and a deliberately sufficient `--max-live-api-calls`. Nothing in this implementation automatically starts the 1,400-request benchmark. Uncalled work remains pending when the cap is exhausted.

Retries occur only for timeouts, HTTP 429 and temporary 5xx responses. Authentication/permanent 4xx errors, refusals, schema failures, ineligible choices and policy failures are not retried. Research mode never substitutes Rules, Weighted, ML or mock decisions for failed providers.

New runs use versioned `config/provider_pricing.yaml`. It contains verified default-tier rates for exact OpenAI model `gpt-4.1-mini-2025-04-14`. The calculation subtracts cached input from total input before applying separate cache/input rates. Missing usage, unknown models and unpriced service tiers remain null with an explicit cost status. Legacy explicit OpenAI rates remain supported for their exact model/default tier.

Jev through OpenRouter uses the documented provider-reported `usage.cost` in USD. The adapter reads it from the SDK's retained HTTP response because the installed SDK's typed Usage object drops extra fields. Missing/malformed native cost remains null with `PROVIDER_COST_UNAVAILABLE`; no Jev price is invented. Cached-token counts remain null. See [routing-cost diagnosis, formula, statuses and provenance](ROUTING_COST_TELEMETRY.md).

Only external routing API charges are counted, excluding downstream execution and local compute/TCO. Unknown usage/cost from a failed attempt keeps the decision's total cost unknown even if a later retry succeeds. Known charges are separately summed as a lower bound. Historical reports are never backfilled.

Optional `--max-routing-spend-usd` uses documented OpenAI rates or Jev's native USD charge. It is an **observed-spend stop threshold**, not a guaranteed prepaid ceiling: a single in-flight request can exceed it. Charges are checked before the next attempt; unknown attempt costs halt further calls when this control is active. API-attempt limits provide a separate hard control regardless of pricing availability.

## Persistence and resume

Output is isolated under `reports/live/<run_id>/`:

- `benchmark_results.csv`, `benchmark_summary.csv`, `api_errors.csv`;
- `routing_cost_summary.csv`, `run_manifest.json`, `pilot_plan.json`;
- `observations.jsonl`, an append-only, flushed/fsynced attempt and observation journal.

A single-writer file lock prevents concurrent writers. CSV/manifest snapshots are replaced atomically inside that live run; frozen source files are never written. This implementation uses POSIX file locking (macOS/Linux).

Resume with the same strategies and settings:

```bash
python -m evaluation.benchmark --live-pilot --strategies rules,weighted,ml,openai,jev --allow-live-api --resume reports/live/<run_id> --max-live-api-calls 100
```

Increasing the cap is deliberate; already reserved attempts count toward it. The plan fingerprint locks dataset hashes, source-result hashes, selected requests, canonical hashes, settings, prompt hashes and implementation versions. Changing model/prompt/settings/sample requires a new run. Credentials themselves are not recorded and can be rotated.

The observation key is `(run_id, benchmark_type, request_id, strategy)`. Completed successes AND failures are skipped. `--force-rerun` with `--resume` explicitly appends a new revision and may spend credits again; previous attempt history is retained. Latest-observation tables and whole-run cost totals have different scopes, identified in their files.

An interruption after reserving a call but before recording its outcome is **INDETERMINATE**. The provider may have processed/billed it; resume will not automatically repeat it. This also applies to an interrupted force-rerun revision even when an older success exists. Explicit force-rerun is required to repeat that decision. A truncated journal fails closed rather than guessing what was sent. Filesystem persistence cannot guarantee knowledge of a remote outcome after a network/process failure.

## Reporting semantics

Each provider row includes requested/returned router model, prompt version, request start/end, full decision time including retry/backoff, first-attempt latency, tokens/cache usage where reported, API request ID, HTTP status, attempts/retries, selected model, native confidence, controlled reasons, refusal and sanitized error classification. Provider-decision wall time includes SDK setup, retry/backoff and per-attempt safeguard/journal bookkeeping. Shared preparation time is added to reported routing latency; `api_routing_latency_ms` preserves provider-decision duration separately. Raw provider bodies, keys and authorization headers are not persisted; SDK body logging is suppressed around calls.

Live successful validated decisions use `execution_mode=LIVE`, `evidence_type=EMPIRICAL`, and `research_eligible=true`. Failed/invalid/refused observations remain visible but are not eligible as successful routing evidence. Request-level execution failures are retained in agreement denominators as disagreements; unavailable, unexecuted, budget-stopped or indeterminate observations are excluded and counted separately. Routed-only metrics are conditional on ROUTED. Always compare coverage/failure counts alongside agreement. Correct NO_ROUTE outcomes reflect deterministic policy behavior.

`INELIGIBLE_MODEL_SELECTED` becomes `INVALID_DECISION`, increments `ineligible_selection_attempts`, and never becomes an accepted selected model. Thus actual accepted policy violations can remain zero while unsafe provider attempts are visible. No fallback hides the attempt.

API call counts are reserved dispatch attempts, including retries and interrupted/uncertain attempts, not proof of the number of billable server operations. Successful API response counts are recorded separately. Token/cost totals are null when any relevant attempt is unknown. Manifests include source/config hashes, frozen dataset provenance, ML recipe fingerprint, SDK/Python versions, git commit if available, per-strategy modes/evidence/eligibility, prompt versions, call/error/retry/token/cost totals and run completion state.

Streamlit's new **Live / Research** view displays saved runs only. It does not initiate paid APIs. It separates BASELINE_REPLAY, LOCAL policy abstention and LIVE execution, synthetic reference labels and empirical execution evidence, historical/new latency, costs and failures. Existing synthetic reporting views remain intact.

## Remaining limitations

No paid end-to-end validation has been performed. Account access and model compatibility must be checked by an explicitly authorized pilot. Alias changes can occur provider-side despite frozen client prompts; inspect returned versions. Pilot results are small-sample synthetic-objective evidence, not real banking quality. These canonical features and model catalog values remain the frozen synthetic assumptions. No model response quality, downstream inference, provider residency certification, independent human reference labels or prospective production effectiveness is established here.

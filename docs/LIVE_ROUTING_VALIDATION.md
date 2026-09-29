# Live routing implementation record

This is an incremental extension of the existing Smart Model Router, verified 2026-09-29. It introduces live routing decisions only. It does not execute the selected downstream model or change the frozen research design.

## Files added

| File | Purpose |
|---|---|
| `config/live_routing.yaml` | Separate live enable/model/timeout/retry/pilot/pricing configuration |
| `app/models/live_routing.py` | Strict nullable-confidence decision and telemetry schemas |
| `app/strategies/live_base.py` | Explicit permission, limited retries, safe telemetry, local validation and routing-charge calculation |
| `app/strategies/live_jev.py` | Verified official TypeSafe System One Choice adapter |
| `evaluation/frozen_baseline.py` | Read-only frozen source verification, deterministic pilot sampling and baseline replay |
| `evaluation/live_store.py` | Single-writer, durable append-only attempt/observation journal |
| `evaluation/live_benchmark.py` | Live pilot/full runner with caps, replay, resume and no fallback |
| `evaluation/live_reporting.py` | Live/replay summaries, errors, costs and manifest aggregation |
| `tests/test_live_openai.py` | Mocked OpenAI SDK/HTTP contracts, permission, retries, schemas and pricing |
| `tests/test_live_jev.py` | Mocked TypeSafe SDK/HTTP contracts, null confidence and candidate validation |
| `tests/test_live_pilot.py` | Frozen replay, fairness, caps, resumability, interruption, redaction and Streamlit tests |
| `docs/LIVE_ROUTING.md` | Setup, official sources, exact commands, semantics and operational limitations |
| `docs/LIVE_ROUTING_VALIDATION.md` | This implementation inventory and validation record |

## Files changed

- `.env.example`: safe opt-in flags and configurable Jev model; no secrets added.
- `requirements.txt`, `requirements-lock.txt`: official OpenAI/TypeSafe packages and pinned new dependencies.
- `app/strategies/llm.py`: official Responses adapter; the old credential-only direct HTTP route is disabled in favor of explicit live authorization.
- `app/strategies/jev.py`: offline mock retained, explicit live interface added; no implicit live calls.
- `evaluation/benchmark.py`: incremental CLI options and dispatch to the separate live runner; original offline benchmark behavior retained.
- `ui/streamlit_app.py`: saved live/replay run visibility, modes, eligibility, agreement, latency source, costs and failures.
- `tests/test_integrations.py`: old Chat Completions transport-specific tests migrated to new SDK tests; ordinary-application permission bypass prevention tested. Existing downstream execution mock tests retained.
- `README.md`: live setup and prominent distinction between live routing and downstream execution.

## SDK and API verification

OpenAI **2.54.0** uses `client.responses.parse(..., text_format=LiveRoutingDecision)` at the official Responses endpoint. The router model is explicitly set by `OPENAI_ROUTER_MODEL` or the new YAML, with no hard-coded default. Temperature is omitted unless explicitly configured for a compatible model.

TypeSafe **0.7.0** uses `TypeSafeClient.system_one(...)` and `Choice(criteria={eligible_id: None, ...})`. The SDK's documented custom response-model mechanism accepts nullable/absent native confidence without fabricating a value. The actual installed SDK serializer/parser was exercised against `httpx2.MockTransport`. Authentication, method names, request structure, response fields and retry controls were verified from official documentation and installed package source, not inferred from unofficial examples.

Official references and model configuration examples are linked in [LIVE_ROUTING.md](LIVE_ROUTING.md). No actual paid provider execution was performed. Credentialed end-to-end availability and billing remain unverified.

## Research and persistence guarantees

Rules, Weighted and ML are replayed from the exact frozen files approved by the user. Replay verifies dataset identity, unique request/strategy match, canonical hash when recorded, eligible IDs, selection validity and recorded numerical fields. Replay is labeled `BASELINE_REPLAY` / `SYNTHETIC` / `FROZEN_BASELINE`, with original latency explicitly labeled historical. It is eligible for comparison of frozen decisions, not as a new local timing measurement.

The pilot runner does not invoke ML fitting, reconstruct an estimator, generate new benchmark requests or generate reference labels. Tests explicitly fail if that runner attempts to call `MLRouter.fit`. Existing isolated regression tests of the training subsystem remain part of the full local suite; they do not alter or retrain any saved frozen baseline.

Missing/ambiguous replay is `BASELINE_REPLAY_UNAVAILABLE`. No provider failure invokes fallback. Provider outputs outside eligible candidates are blocked and counted separately from accepted policy violations. Raw prompts, IDs as decision features, preferred/acceptable reference answers and expected labels are absent from both provider payloads. Real SDK wire-format tests check the canonical payload/Choice criteria as well as higher-level adapter mocks.

Per-attempt slots are reserved and fsynced before dispatch. Maximum retries default to 2 and count against the API cap. Resume skips completed successes and failures. Unknown in-flight outcomes are marked INDETERMINATE, including interrupted force-rerun revisions, rather than implicitly repeated. Force-rerun appends history; whole-run costs include every revision. The plan fingerprint prevents resume across changed source data/results, sample, implementation, runtime SDK versions, model/settings or prompts. SDK body logging is suppressed and reports redact environment key values.

## Validation

Final command: `python -m pytest -q`: **113 passed in 44.00 seconds**. The live-pilot CLI without `--allow-live-api` was also verified to exit with an explicit permission error before execution.

The suite covers the existing local behavior and the new provider contracts, frozen baseline replay, missing keys, explicit permission, preflight and per-attempt caps, optional spend stopping, refusal/timeout/rate-limit/permanent-error behavior, strict outputs, ineligible selections, nullable confidence, no-call NO_ROUTE, exact canonical payloads, source provenance, failure preservation, resume/force/interruption, secret redaction, legacy offline reporting and the new Streamlit view.

SDK tests use local mocks only, including official OpenAI/httpx and TypeSafe/httpx2 serialization/parsing. Mocked live-run reports are confined to pytest temporary directories; they are not published as empirical live results under project reports.

A before/after SHA-256 inventory verifies **103 existing dataset/report/protected-source/configuration files unchanged**, including both frozen run directories, all frozen dataset/reference snapshots, policy engine, feature extraction, canonical encoding, ML implementation, ground-truth generator, model catalog and original research configuration. No genuine frozen-methodology defect was changed.

The project has no git repository metadata, so a future live manifest records `git_commit=null`; source hashes provide implementation provenance.

## Commands and output

See [LIVE_ROUTING.md](LIVE_ROUTING.md) for environment variables and complete pilot, provider-only, resume and force-rerun commands. Future explicitly authorized runs write `reports/live/<run_id>/` with results, summary, manifest, errors, cost summary, plan and durable journal. The default sample is 15 IID plus 15 held-out requests: 7 policy NO_ROUTE cases and 23 routable cases per live provider (46 initial calls for both).

The default hard attempt cap is 50. Optional spend limits are observed-charge thresholds: one in-flight request may cross the threshold; unknown costs stop further attempts when that control is enabled. Routing cost remains null without reliable usage and documented pricing. Jev cached usage is unavailable; a charge can be calculated without it only with documented identical cached/uncached input rates. File locking targets macOS/Linux. Model aliases may change provider-side. Frozen replay latency is not newly measured. Synthetic agreement remains separate from real downstream quality, which this extension does not evaluate.

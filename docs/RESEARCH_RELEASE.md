# Smart Model Router V1 — Frozen Research Release

The experimental phase is closed. Tag: **`v1.0.0-research`**. This release preserves
the completed evidence; publication does not rerun, tune or regenerate any experiment.

## Research Question

How do deterministic rules, weighted heuristics, traditional machine learning,
general-purpose LLMs, and specialized decision models compare for policy-aware
AI model routing in a regulated banking environment?

## Routing Approaches

- Rules
- Weighted heuristic
- Random Forest ML
- OpenAI
- Jev 1.13 via OpenRouter

Hard governance determines which models are permitted. Routing strategies choose
only among policy-eligible candidates. Shared normalization, feature extraction
and policy filtering precede strategy-specific routing. No compliant candidate
means `NO_ROUTE`; selecting outside the candidate set is invalid.

Rules, Weighted and ML use exact frozen baseline replay for the primary experiment.
Their decisions and original latency provenance are preserved, not reconstructed
or newly measured. OpenAI and Jev use the same canonical structured information
and candidate snapshots. Raw prompts, request IDs and reference-route labels
are not sent as routing decision features.

## Experimental Design

Primary: **400 requests**.

| Benchmark | Requests | Routable | NO_ROUTE |
|---|---:|---:|---:|
| IID_SYNTHETIC | 200 | 149 | 51 |
| TEMPLATE_HELD_OUT | 200 | 155 | 45 |

Final sample hash:

```text
70db720aadaa0322b2a9f29b9e9a796269d44d780c7d6c7023c11914eee7f7c0
```

Primary run: **`20260929T152852-8d825a67`**.

Stability run: **`20260929T155310-839c6b13`**.

The primary sample was selected with seed 4042 using the frozen
hierarchical-largest-remainder protocol and its recorded stratification fields.
Prior live-pilot requests were excluded before selection: 15 IID and 15 held-out
requests, with zero final overlap. No selection is changed in this release.

Stability uses **25 routable IID + 25 routable held-out requests**, all members of
the primary sample, selected with seed 5042 and repeated five times. Only OpenAI
and Jev participate in stability. Its observations are separate from the primary
400-request experiment.

The original frozen baselines record 600 generated ML training requests (seed
1042), 464 fitted and 136 excluded, plus 200 validation requests (seed 1043).
Source test seeds are 42 for IID and 2042 for held-out. Model-recipe fingerprints,
configuration, split data and validation results are preserved. A recipe
fingerprint is not a persisted fitted classifier. No retraining is needed to
inspect or replay the exact recorded final decisions.

## Evidence terminology

"Synthetic Reference-Route Agreement measures agreement with the project's
synthetic routing objective. It does not measure downstream response quality,
production accuracy, or an objectively best model."

"The ML router was trained to approximate the synthetic reference-routing
function."

"OpenAI and Jev were evaluated as live inference-based routing approaches."

Reference routes derive from configured policy, synthetic quality and cost
assumptions. The preferred route selects the cheapest eligible model within
0.12 of the best eligible synthetic quality score. This is not human expert
annotation or measured downstream response quality. Existing legacy column names
are preserved byte-for-byte; use the scientifically precise aliases and labels
when interpreting or citing the results.

## Live Execution Provenance

| Strategy | Provider | Configured model | Execution path |
|---|---|---|---|
| OpenAI | openai | `gpt-4.1-mini-2025-04-14` | `OPENAI_DIRECT` |
| Jev | openrouter | `typesafe/jev-1.13` | `JEV_VIA_OPENROUTER` |

Jev returned revision **`typesafe/jev-1.13-20260917`**. Jev latency/cost measurements
are end-to-end **Jev-via-OpenRouter observations**, NOT direct Jev API measurements.
OpenAI costs use recorded provider usage and the frozen pricing configuration;
Jev costs use the recorded OpenRouter charge. Unknown telemetry is not converted
to zero. Downstream execution remained disabled.

Routing-instruction hashes matched between primary and stability:

| Strategy | Instruction hash |
|---|---|
| OpenAI | `5ef38427f393d84507e1cbf71b1a5b0b021f095fe3034cce242a749cc16c19a8` |
| Jev | `8ee1ee48991f523dd9d1f2b6a2123640146754a3a60b69c73dc7cd57d6483a49` |

The existing [stability guard record](../reports/stability/20260929T155310-839c6b13/pre_live_guards.json)
links to the primary run and its file hashes. The
[execution-integrity record](../reports/stability/20260929T155310-839c6b13/execution_integrity.json)
records matching canonical inputs, unique observation identities and unchanged
primary evidence. These records are copied without modification.

## Primary Experiment Integrity

- 608 planned live calls: 304 OpenAI + 304 Jev
- 608 successful live routing decisions
- 0 failures
- 0 retries
- 0 invalid decisions
- 0 ineligible selections
- 0 policy violations

The 96 policy abstentions per strategy are distinct from dispatched live routing
decisions. Full source:
[`reports/final_experiment/20260929T152852-8d825a67/`](../reports/final_experiment/20260929T152852-8d825a67/).

## Stability Experiment Integrity

- 500 planned calls: 250 OpenAI + 250 Jev
- 500 actual calls
- 500 successful decisions
- 0 failures
- 0 retries
- 0 timeouts
- 0 rate limits
- 0 invalid decisions
- 0 ineligible selections
- 0 policy violations

All 50 requests had five successful repetitions per provider. The authorized
attempt cap was 600. Full source:
[`reports/stability/20260929T155310-839c6b13/`](../reports/stability/20260929T155310-839c6b13/).

## Combined Live Evidence

**1,108 successful live routing decisions.** These are ROUTING decisions, not
downstream inference or execution of the selected models.

## Published evidence and historical report policy

The complete primary and stability directories are preserved, including every
existing report, journal, plan, manifest, integrity record and their zero-byte
`.run.lock` markers. Those markers are persistent files in the captured run
directories, not an active lock or a credential. No CSV, JSON, precision or
line ending has been rewritten. Copying preserves original file modification
times; Git stores content, not filesystem timestamps.

The release includes:

- The exact final sample under `data/final_experiment/final-70db720aadaa0322/`.
- The directly referenced frozen baseline runs
  `reports/iid_synthetic/20260928T153247-eb297b76/` and
  `reports/template_held_out/20260928T153148-82538755/`, including source datasets,
  labels, train/validation provenance, leakage checks and replay decisions.
- Only `run_manifest.json`, `pilot_plan.json` and `benchmark_results.csv` from
  each of the seven pilot runs named below. The unchanged exclusion loader reads
  these exact files and verifies their hashes. Other pilot reports/journals are
  omitted because they are not needed to reconstruct the exclusion set.
- Original synthetic data files used by the application and local tests.
- **Only three approved legacy compatibility/test fixtures** under
  `reports/trained_ml/`: `benchmark_results.csv`, `benchmark_summary.csv`, and
  `run_manifest.json`. They are preserved byte-for-byte and included in the
  checksum manifest solely to support the existing compatibility/UI tests.
  **Their results are not findings of the final experiment.** No other files
  from this historical run are included.

Required pilot provenance run IDs:

```text
20260928T163059-ca49e981
20260928T164013-13b41c9b
20260928T164612-a823bb14
20260928T165218-fdbdc7d6
20260929T123623-52fd51d5
20260929T123913-a0c012e2
20260929T125454-9c7cb858
```

Excluded: the superseded `final-fca37fb622af265a` sample; unrelated exploratory
benchmark/stability/ablation/sensitivity/resilience/calibration outputs; redundant
pilot artifacts; runtime/audit stores; virtual environments; credentials and
populated environment files; IDE state; caches; bytecode; temporary files and
platform metadata. Historical source files are not deleted or altered.

Original absolute local paths inside evidence remain unchanged for provenance.
They are historical metadata, not portable execution paths. Repository-relative
paths above locate the release artifacts.

## Byte-level integrity and verification

[`RESEARCH_ARTIFACT_SHA256SUMS.txt`](../RESEARCH_ARTIFACT_SHA256SUMS.txt) contains
SHA-256 checksums for all included data and report files, including the approved
legacy fixtures. Listing a fixture in the checksum manifest does not classify it
as a final finding. Verify from the repository root on macOS:

```bash
shasum -a 256 -c RESEARCH_ARTIFACT_SHA256SUMS.txt
```

The JSON sample's byte hash differs from the semantic sample hash: the latter
hashes the experiment's defined identity payload. Neither is changed by release
packaging. `.gitattributes` disables Git line-ending conversion so checked-out
content retains the captured bytes.

## Local inspection and test reproduction

Use the README's locked-environment setup, read-only CSV inspection commands and
network-blocked full-suite command. Reading the evidence and verifying checksums
requires no provider credentials and no external AI API calls. Test fixtures may
create temporary synthetic inputs and fit temporary classifiers inside the tests;
they do not rerun the completed primary/stability experiments or update their
frozen datasets, models, reports or manifests.

Do not run generators, training commands or report-writing analysis entry points
against the frozen directories. Older standalone scripts may require exploratory
reports intentionally omitted from this research release. The saved final
analyses, their source implementation and checksums are included for inspection.

Any future live reproduction requires valid credentials and the explicit
**`--allow-live-api`** safety flag, plus enabled provider settings. It is a new,
separately authorized experiment and must not overwrite V1 evidence. No live
reproduction command is executed during publication. The provided `.env.example`
has empty credentials and disables live execution.

## Key methodological caveats

- The reference-routing objective is synthetic.
- ML was trained against that objective.
- Held-out testing is template-held-out within known task categories, not
  task-held-out generalization.
- Request-level bootstrap does not fully model shared template structure.
- Repeated stability observations are not independent benchmark samples.
- Non-significant OpenAI/Jev differences do not establish equivalence.
- Stability comparison is descriptive, not a composite winner ranking.
- Provider/model/pricing observations are configuration- and time-specific.
- Local-router external API cost of zero does not mean zero total compute/TCO.
- No downstream selected-model response-quality evaluation was performed.

## Release identity

Repository: https://github.com/debabratapruseth/Smart-Model-Router-for-Regulated-AI

Annotated tag: **`v1.0.0-research`**.

Tag message: **Smart Model Router V1 frozen research release**.

Use the tag's target commit SHA when citing the snapshot. The Git commit fixes
the published source, documentation and evidence inventory; the artifact
checksums allow independent byte-level verification.

## Publication verification

Verified on 2026-09-30 in a separate release checkout:

- Secret scan: PASS for the publication set. No populated `.env`, credentials,
  virtual environment, caches or unrelated runtime files are included.
- Existing local test suite: **173 passed** with external network connections
  blocked. The original source interpreter supplied installed dependencies, but
  imports and tests resolved to the separate release checkout.
- **98 frozen data/report files** passed byte-level SHA-256 comparison to source.
  The final primary and stability directories remained complete and unchanged.
- All **199 copied source/provenance files** retained their original bytes and
  modification timestamps. The original project's **340 non-cache source/research
  files** were separately checked for unchanged contents.
- **Zero external AI API calls** were made during release preparation. Only
  GitHub repository operations require network access for publication.

The sample manifest's byte-level SHA-256 is:

```text
78d01cfba1e9d6b50f41f9a7e9532f45a623b2ffee62297b1d11b4cf1c4c3c30
```

Deterministic directory digests below hash the UTF-8, newline-terminated,
path-sorted `SHA256  repository-relative-path` entries for that directory from
`RESEARCH_ARTIFACT_SHA256SUMS.txt`. They cover every included file in each directory.

| Directory | Files | Directory digest |
|---|---:|---|
| `data/final_experiment/final-70db720aadaa0322/` | 1 | `1543386d282f531799f5621b1467c7fbb53f7c1ea11ef2b0c241f0ef2c2686a4` |
| `reports/final_experiment/20260929T152852-8d825a67/` | 14 | `0ead3d38cafac0fa71db4d98e596fa8cb362a148aff104dfcb8afce0cb9e8053` |
| `reports/stability/20260929T155310-839c6b13/` | 18 | `80d53308ddb80af5c92699cc166c2ac07004009fe8606b3a7a97d4357d47f5d1` |

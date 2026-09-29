# Smart Model Router for Regulated AI

**Can a fast System 1 decision model route AI workloads as effectively as a general-purpose LLM—while being faster, cheaper, and policy-safe?**

Smart Model Router is an experimental policy-aware routing framework for regulated AI environments.

Instead of sending every request to the same model, it first determines which models are permitted by governance and then compares different ways of choosing among those eligible models.

> **Governance decides what is permitted. Intelligence decides what is preferred.**

Hard policy constraints execute **before** intelligent routing: **Governance → Eligibility → Optimization → Validation**. No routing strategy can override governance.

## Why this research matters

Enterprise AI platforms increasingly have access to many models. In a regulated environment, model routing raises several connected questions:

1. Which models are allowed to handle this request?
2. Among those allowed, which should handle it?
3. Does intelligent routing justify its latency and cost?
4. Is a general-purpose LLM needed to make that decision?
5. Could a smaller, faster decision model perform the routing role?
6. Could simple heuristics or traditional ML be sufficient?

The broader research question is how **rules, weighted heuristics, traditional ML, general-purpose LLMs, and specialized decision models** compare for policy-aware routing in a regulated banking environment.

## Architecture: policy before preference

```mermaid
flowchart TD
    A[User / Application] --> B[Request Normalization]
    B --> C[Feature Extraction]
    C --> D[Policy Engine + Model Catalog]
    D -->|No eligible models| N[NO_ROUTE]
    D -->|Policy-filtered candidates| E[Eligible Models + Structured Input]
    E --> F[Routing Strategy: Rules / Weighted / ML / OpenAI / Jev]
    F --> G[Decision Validator]
    G -->|Valid eligible choice| H[Selected Model]
    G -->|Invalid choice| I[Rejected Decision]
```

Deterministic policies enforce configured classification, provider, region, modality, context and other constraints. Strategies choose only among eligible candidates, and validation checks the resulting decision.

**The experiment evaluates the routing decision. It does not execute the selected downstream model.**

## Five routing paradigms

| Paradigm | Router | Execution in the final comparison |
|---|---|---|
| Deterministic rules | Rules | Frozen baseline replay |
| Weighted heuristics | Weighted | Frozen baseline replay |
| Traditional machine learning | Random Forest ML | Frozen baseline replay |
| General-purpose LLM | OpenAI — `gpt-4.1-mini-2025-04-14` | Live, OpenAI direct (`OPENAI_DIRECT`) |
| System 1 decision model | Jev 1.13 | Live, via OpenRouter (`JEV_VIA_OPENROUTER`) |

Local decisions are replayed from the validated frozen baselines. ML is not retrained or reconstructed during the final comparison. OpenAI and Jev were evaluated as live inference-based routing approaches.

## Experiment at a glance

| Experiment | Requests | Design | Successful live routing decisions |
|---|---:|---|---:|
| Primary | 400 | 200 IID + 200 template-held-out; five routers | 608: 304 OpenAI + 304 Jev |
| Stability | 50 routable | 25 per benchmark × five repetitions × two live routers | 500: 250 OpenAI + 250 Jev |
| **Combined live evidence** | | | **1,108** |

The primary sample contains **304 routable requests and 96 policy-handled `NO_ROUTE` outcomes**: IID has 149 routable / 51 `NO_ROUTE`; held-out has 155 / 45. Policy-handled `NO_ROUTE` requests generated **no external routing calls**.

Across both experiments there were **zero API failures, invalid decisions, ineligible selections, or policy violations**.

- **IID_SYNTHETIC:** new synthetic samples from a familiar template family.
- **TEMPLATE_HELD_OUT:** requests from templates excluded from ML training and validation, within known task categories.

The synthetic reference objective prefers the cheapest eligible model within **0.12** of the highest configured eligible quality score. These labels derive from configured policy, quality and cost assumptions, not human annotations or measured model responses. See the [methodology](docs/BENCHMARK_METHODOLOGY.md) for the training/validation design.

## Headline results

**Synthetic Reference-Route Agreement — including correct abstentions**, with 200 requests per benchmark:

| Router | IID Preferred | Held-out Preferred | IID Acceptable | Held-out Acceptable | Mean routing latency | External routing API cost |
|---|---:|---:|---:|---:|---:|---:|
| Rules | 65.5% | 63.0% | 86.5% | 88.5% | ~0.57–0.58 ms | $0 |
| Weighted | 78.0% | 76.5% | 84.0% | 85.0% | ~0.57–0.59 ms | $0 |
| ML | 99.0% | 99.0% | 99.5% | 99.0% | ~1.54–1.62 ms | $0 |
| OpenAI | 75.5% | 71.5% | 85.0% | 83.5% | ~1.06–1.09 s | $0.2230 |
| Jev via OpenRouter | 74.5% | 70.0% | 86.0% | 84.0% | ~0.41–0.44 s | $0.0300 |

**Synthetic Reference-Route Agreement measures agreement with the project's configured synthetic routing objective. It does not measure downstream model response quality.** Preferred agreement uses the preferred reference model; acceptable agreement also credits other models permitted by the synthetic objective. Both columns include correct policy abstentions.

Latency ranges show the two benchmark means **over all requests, including policy-handled `NO_ROUTE` cases**. Rules, Weighted and ML latencies are historical frozen-baseline measurements; OpenAI and Jev latencies were measured in the primary run. Costs are total external routing API costs across the primary experiment, not per-request costs. Zero external API cost does not mean zero compute cost or total cost of ownership.

Sources: [primary summary](reports/final_experiment/20260929T152852-8d825a67/benchmark_summary.csv) · [cost summary](reports/final_experiment/20260929T152852-8d825a67/cost_summary.csv).

## Key findings

1. **ML closely reproduced the synthetic routing objective.** Preferred agreement was 99% on both benchmarks. The ML router was explicitly trained to approximate this synthetic reference-routing function. This demonstrates learning and generalization of that objective, not universal model-selection superiority.

2. **Weighted heuristics were competitive.** Preferred agreement was 78.0% / 76.5% for Weighted, compared with 75.5% / 71.5% for OpenAI and 74.5% / 70.0% for Jev (IID / held-out). Paired exact McNemar tests with Holm correction at α = 0.05 did not establish a significant preferred or acceptable agreement advantage for either live router over Weighted.

3. **OpenAI and Jev produced similar reference-route agreement.** The paired analysis found no statistically detectable difference in preferred or acceptable Synthetic Reference-Route Agreement between them on either benchmark. **Non-significance does not establish equivalence.**

4. **Jev showed lower observed routing latency and cost than OpenAI.** The benchmark means were ~0.41–0.44 s for Jev via OpenRouter and ~1.06–1.09 s for OpenAI. Total primary routing API costs were $0.029983 and $0.223006, respectively: approximately **7.4× lower observed cost** for Jev via OpenRouter. This is specific to the configuration, provider path and pricing in this frozen experiment—not a universal provider cost claim.

5. **Hard governance remained independent of routing intelligence.** Zero policy violations and zero ineligible selections reflect the deterministic eligibility filtering and decision validation that constrain the routers. They are not evidence that ML, OpenAI or Jev independently learned to enforce policy.

See the saved [paired statistical comparisons](reports/final_experiment/20260929T152852-8d825a67/statistical_comparisons.csv).

## Why test a System 1 decision model?

General-purpose LLMs support broad language and reasoning tasks. Routing is a bounded decision problem:

**Structured input + finite candidate set + explicit objectives + hard policy boundaries → one routing decision.**

This raises a research hypothesis: specialized fast decision models may be useful for AI control-plane decisions where full generative reasoning is unnecessary. Jev 1.13 represents that approach in this five-way comparison; it is a research subject, not an assumed winner.

The configured model was **`typesafe/jev-1.13`**, with returned revision **`typesafe/jev-1.13-20260917`**, executed through **`JEV_VIA_OPENROUTER`**. All Jev latency and cost observations are end-to-end **Jev-via-OpenRouter** measurements, not direct Jev API measurements.

## Stability: repeated decisions on identical inputs

| Benchmark | OpenAI full consistency | Jev full consistency | OpenAI pairwise | Jev pairwise |
|---|---:|---:|---:|---:|
| IID | 80% | 96% | 90.4% | 98.4% |
| Template-held-out | 80% | 100% | 90.4% | 100% |

Full consistency is the share of requests receiving the same selection across all five repetitions. Pairwise agreement measures matching selections across repetition pairs.

Across five repeated decisions on identical structured inputs, Jev showed greater observed selection consistency in this 50-request sample. **This comparison is descriptive; it does not establish statistical superiority.** All **500/500** stability decisions succeeded, with **zero retries and zero failures**. Repetitions are not independent benchmark samples.

Source: [stability summary](reports/stability/20260929T155310-839c6b13/stability_summary.csv).

## An important engineering result

> **More sophisticated routing did not monotonically produce better routing outcomes.**

Under this synthetic objective, Rules often found an acceptable route, while Weighted substantially improved preferred-route agreement over Rules and remained statistically competitive with the live routers. ML learned the reference-routing function closely. Live inference introduced external latency and cost, while the specialized decision-model path showed promising observed latency, cost and stability characteristics.

The results support comparing routing methods against the actual objective and operational constraints, rather than assuming that a more sophisticated router will improve decisions.

## Fairness of the comparison

All primary routers used the **same normalized requests, policy-filtered candidate sets and canonical structured information**. Live routers received no synthetic ground truth, preferred reference model, acceptable reference-model list, request ID as a decision feature, or raw prompt. The primary comparison was **`STRUCTURED_ONLY`**.

Frozen baseline replay preserves the original local decisions for those exact requests. Its latency is not a newly executed local measurement alongside the live providers.

## Repository structure

```text
app/                 routing, policy, adapters and decision validation
config/              frozen policies, model catalog and experiment settings
evaluation/          experiment runners, reporting and statistical methods
scripts/             original synthetic generators
tests/               local regression suite
ui/                  Streamlit application
docs/                methodology and research release documentation
data/                synthetic datasets and frozen final sample
reports/
  final_experiment/  primary results, statistics and execution provenance
  stability/         repeated-decision results and execution provenance
```

Explore the [primary reports](reports/final_experiment/20260929T152852-8d825a67/) and [stability reports](reports/stability/20260929T155310-839c6b13/). Detailed integrity checks, included provenance and legacy test-fixture boundaries are documented in [RESEARCH_RELEASE.md](docs/RESEARCH_RELEASE.md).

## Reproducibility: inspect evidence without API credentials

The experimental phase is closed. Inspect the saved evidence in the frozen release; no live rerun is needed. The recorded environment used Python 3.13.13 with locked dependencies.

```bash
git clone https://github.com/debabratapruseth/Smart-Model-Router-for-Regulated-AI.git
cd Smart-Model-Router-for-Regulated-AI
git checkout --detach v1.0.0-research
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
shasum -a 256 -c RESEARCH_ARTIFACT_SHA256SUMS.txt
```

Package installation contacts package registries, not AI inference providers. The CSV reports can be inspected directly without credentials. Run the local tests with credentials removed from the test process and network connections blocked:

```bash
python -B - <<'PY'
import os, socket, pytest
for name in ('OPENAI_API_KEY', 'OPENROUTER_API_KEY', 'TYPESAFE_API_KEY'):
    os.environ.pop(name, None)
def deny_network(*args, **kwargs):
    raise RuntimeError('External network access is disabled for offline tests')
socket.socket.connect = deny_network
socket.socket.connect_ex = deny_network
socket.create_connection = deny_network
raise SystemExit(pytest.main(['-q', '-p', 'no:cacheprovider']))
PY
shasum -a 256 -c RESEARCH_ARTIFACT_SHA256SUMS.txt
```

Tests use temporary fixtures and mocked transports. Do not regenerate frozen datasets, samples, labels or experiment outputs. Live execution requires valid provider credentials, enabled provider flags and explicit **`--allow-live-api`** authorization; it is outside this inspection workflow.

## Research limitations

- Synthetic Reference-Route Agreement is not downstream response quality; ML was trained against that synthetic objective.
- Held-out testing excludes templates within known task categories, not entire tasks. Request-level bootstrap does not fully model shared template structure.
- The stability sample is small and descriptive; repeated observations are not independent samples.
- Non-significant OpenAI/Jev differences do not prove equivalence.
- Provider latency and cost are specific to the model, configuration, pricing and measurement time. Jev measurements include the OpenRouter execution path.
- Local external API cost of zero does not imply zero total compute cost or TCO.
- No selected downstream model responses were executed or evaluated. These findings do not establish production readiness or real-world model-selection superiority.

See [RESEARCH_RELEASE.md](docs/RESEARCH_RELEASE.md) for complete methodology, provenance and evidence boundaries.

## Frozen research release

The immutable V1 baseline is **`v1.0.0-research`**. This README update on `main` follows that release; checking out the tag retrieves the original release README along with the frozen source and evidence.

| Identifier | Value |
|---|---|
| Tag | `v1.0.0-research` |
| Primary run | `20260929T152852-8d825a67` |
| Stability run | `20260929T155310-839c6b13` |
| Final sample hash | `70db720aadaa0322b2a9f29b9e9a796269d44d780c7d6c7023c11914eee7f7c0` |

The [final sample manifest](data/final_experiment/final-70db720aadaa0322/sample_manifest.json), [research release document](docs/RESEARCH_RELEASE.md) and [artifact SHA-256 checksums](RESEARCH_ARTIFACT_SHA256SUMS.txt) support independent inspection. Cite the repository together with the frozen tag and its target commit.

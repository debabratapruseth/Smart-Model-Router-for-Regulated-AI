# Router card — version 1.0.0

**Purpose:** an educational, policy-aware banking model selection and research platform.

**Intended use:** CPU laptop demonstrations, classroom exercises, reproducible synthetic comparisons, and architecture discussions. The router selects an approved endpoint; it does not place workloads or make banking decisions.

**Non-intended use:** production banking, customer eligibility, credit decisions, regulatory certification, automated policy interpretation, real customer processing, or vendor performance claims.

**Inputs:** business prompt plus caller-supplied classification, residency, risk, capability, context, cost and quality metadata. Classification is mandatory. Metadata must originate from a trusted application in any future deployment; this unauthenticated demo cannot establish caller trust. Prompt text is never authority for policy. Keywords may affect inferred task needs, but cannot relax explicit metadata constraints.

**Outputs:** ROUTED, NO_ROUTE, UNAVAILABLE or INVALID_DECISION; selected model, alternatives, controlled reasons, rejected candidates and policy IDs, scores, synthetic cost/model latency estimates and measured routing latency. Human-review flags block optional execution.

**Strategies:** deterministic rules, five-dimension weighted utility, strongest eligible task quality, cheapest eligible estimated inference, optional RandomForest, optional structured OpenAI-compatible router, and Jev weighted mock. Tie breaks use configured preferred regions then model IDs. Unscored alternatives from external strategies have score zero, not a model confidence estimate.

**Training:** no training at startup. `python -m evaluation.benchmark --train-ml` fits an in-memory 60-tree CPU classifier on 600 synthetic requests using seed 1042; evaluation seed 42 is disjoint. Labels use task quality band then cost, not weighted outputs. Template overlap remains and is a research limitation. No untrusted serialized estimators are loaded.

**Assumptions:** request metadata is accurate, endpoint catalog permissions are correct, configured price and latency estimates apply, local YAML is trusted, and context metadata includes retrieved/attachment content. A conservative UTF-8-byte estimate prevents trivial prompt-length underreporting but does not count actual images/audio. No attachments are executed.

**Policy controls:** 16 deterministic gates in `config/policies.yaml`; all evaluated before a strategy receives candidates. Explicit deny rules win. Context reserves output tokens. PII regional rules are illustrative organizational choices, not claims about applicable law. Validation rejects out-of-set selections, including low-confidence invalid selections, without fallback. Unknown classifications and absent metadata fail schema validation. No-route never relaxes policy.

**Fallback:** only configured local strategies can be fallbacks. Optional service failures and low confidence from ML/LLM/Jev may trigger fallback in normal routing. Invalid model selections cannot. Comparisons disable fallback to expose raw strategy behavior; fallback rate can therefore be zero by experimental design. ML probabilities are masked to eligible labels without renormalizing. Local utility scores and Jev mock confidence are not calibrated probabilities.

**Monitoring:** structured Python logging excludes prompts. Routing and optional execution telemetry are CSV; routing success is not presented as execution success. Dynamic mode uses fresh rolling observed latency, availability and success data. Data is not silently substituted when missing. Rates remain explicit catalog settings. Audit snapshots capture effective telemetry-derived candidates. CSV supports threads within one process; use one local worker. Production needs a transactional telemetry sink.

**Audit/replay:** append-only application API over SQLite, original candidate/config snapshots, content hashes, versions and optional prompt hash. Replay returns all decisions for a request ID, not a new decision under current settings. Full prompts are not logged. Hashes may be vulnerable to guessing for short prompts. Metadata and execution outputs may still be sensitive. No retention policy, access control, encryption or tamper-evident storage is implemented.

**External routing:** disabled without a key and an explicitly chosen API model. Default egress policy permits PUBLIC only, and only categorical features plus eligible catalog descriptors are transmitted; never prompt text. Review egress settings and catalog descriptors before connecting a real service. Jev live mode is a documented stub; mock timing cannot test H4.

**Execution:** disabled by default behind YAML and environment opt-ins. Only explicitly mapped approved-cloud text models can execute. An exact audited input/decision and current policy recheck are required. Endpoint-to-provider/region assurances must be reviewed manually; the demo cannot verify a provider's physical residency. No RAG retrieval, streaming, embeddings, audio or image inference is implemented. Execution results include outputs in CSV, intentionally distinct from prompt-free audit. Quality scoring needs references or manual judgments.

**Known failure modes:** keyword ambiguity, bad source metadata, stale telemetry, synthetic labels that reward synthetic assumptions, underestimated real price variation, missing model capabilities, malformed external JSON, service failures, and model drift. NO_ROUTE can legitimately be common when governance conflicts with the request.

**Versioning/change control:** bump policy, catalog, routing and benchmark versions when changing YAML or labels. Hashes also identify edits without a version bump. Re-generate expected labels after intended policy/catalog changes; preserve prior reports for comparisons. Run unit, adversarial and benchmark checks before accepting changes.

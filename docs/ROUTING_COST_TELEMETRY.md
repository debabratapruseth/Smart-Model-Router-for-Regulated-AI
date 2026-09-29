# External routing API cost telemetry

This measures external **routing API charges**, excluding downstream selected-model
execution, local Rules/Weighted/ML compute, and total cost of ownership. Routing,
policy, candidates, prompts, reference labels, training and sampling are unchanged.
The existing successful run `reports/live/20260929T123913-a0c012e2` is not rewritten.

## Diagnosis of the existing pilot

All 23 OpenAI routing calls recorded input, cached input (legacy `cached_tokens`),
output and total tokens, and returned model `gpt-4.1-mini-2025-04-14`. Cached counts
were zero. Prices resolved to null, so the calculator correctly refused to invent
a cost. Reasoning-token detail and service tier were exposed by the installed
OpenAI SDK but were not retained by this adapter. Their historical values cannot
be recovered from the saved reports.

All 23 Jev calls recorded input/output counts and returned model
`typesafe/jev-1.13-20260917`. Total tokens were derived from input + output; cached
counts were unavailable. The installed TypeSafe SDK 0.7.0 ignores additional usage
attributes in its typed `Usage` object, but retains the original HTTP response.
The old adapter never read OpenRouter's documented `usage.cost`. Neither the raw
response nor that field was saved, so its presence on the historical calls cannot
be established retrospectively. No historical costs are backfilled.

## OpenAI calculation and pricing provenance

`config/provider_pricing.yaml` pins the research pricing snapshot to
`routing-api-pricing-2026-09-29`, effective for new experiments from 2026-09-29.
This is the snapshot adoption/verification date, not an assertion that OpenAI
changed its prices on that date. Verified sources:

- https://developers.openai.com/api/docs/pricing
- https://developers.openai.com/api/docs/models/gpt-4.1-mini

The exact snapshot model `gpt-4.1-mini-2025-04-14`, **default service tier**, uses
USD per million tokens: input 0.40, cached input 0.10, output 1.60.

```
uncached_input_tokens = input_tokens - cached_input_tokens
cost_usd = (uncached_input_tokens * 0.40
            + cached_input_tokens * 0.10
            + output_tokens * 1.60) / 1_000_000
```

Model lookup uses the returned exact model ID and returned service tier. Unknown
models, absent tiers, and tiers without verified rates remain unpriced. The code
does not change requested service tier or assume that an absent tier means default.
Missing/invalid token counts stay unknown; cached counts cannot exceed input.
`cached_tokens` remains a compatibility alias for `cached_input_tokens`.

Reasoning tokens and cache-write counts are retained when exposed, but are not
added as a second charge: the configured text model's input/output token rates
apply to the corresponding totals. No tool, image, downstream or latency-based
charges are introduced. Usage metadata is a numeric allowlist, not a raw response
dump. Each attempt's usage and price calculation survive in the journal.

Legacy explicit rates in `live_routing.yaml` remain supported for their exact
returned model and default tier; their provenance is recorded on each attempt.
New configuration should use `provider_pricing.yaml`.

## Jev through OpenRouter

Verified contract: https://openrouter.ai/docs/guides/community/typesafe-sdk
and https://openrouter.ai/docs/guides/community/jev

The unchanged `TypeSafeClient.system_one()` request uses OpenRouter's System One
endpoint. OpenRouter documents `usage.cost` as the response cost in USD. The
adapter reads this field from the retained HTTP response without changing the
Choice request/decision schema. A finite nonnegative numeric value, including an
explicit zero, is retained as `PROVIDER_REPORTED`. Missing or malformed charges
remain null with `PROVIDER_COST_UNAVAILABLE`. No per-token Jev dollar estimate,
credits, billing units, or decision fees are invented. Cached tokens stay null.

## Statuses, reports and limits

- `CALCULATED_FROM_PROVIDER_USAGE`: verified OpenAI rates and provider usage.
- `PROVIDER_REPORTED`: OpenRouter's Jev USD charge.
- `PROVIDER_COST_UNAVAILABLE`: Jev did not supply a usable charge.
- `PRICING_NOT_CONFIGURED`: no verified rate for the returned model/service tier.
- `USAGE_NOT_RETURNED`: missing usage/model information needed to calculate cost.
- `INVALID_PROVIDER_USAGE`: inconsistent cache/input counts.
- `NOT_APPLICABLE_NO_ROUTE`: zero external cost for policy abstentions.
- `BASELINE_REPLAY_ZERO_EXTERNAL_COST`: zero new external cost for frozen replay;
  this does not claim free local compute or zero historical cost.
- `NO_EXTERNAL_CALL`: no external request was made.

New CSV rows contain cost status, price version/hash/source/date, exact returned
model, service tier, token categories, sanitized usage and dollar cost. Summary
tables include token totals, total routing cost, per-request and per-API-decision
cost and unknown-cost counts. An API decision may include multiple attempts.
The per-request denominator includes policy abstentions and replay where relevant.
No API attempts means zero external token consumption in summaries. Otherwise a
missing category in any attempt keeps that category's aggregate null.

A decision's aggregate cost remains unknown if any retry has unknown cost.
The manifest and cost ledger include all attempts (including force-rerun history);
summary tables represent latest observations. Known charges are separately labeled
as a lower bound. The manifest snapshots pricing configuration, its semantic hash,
version, effective date, official sources and calculation methods. Changed pricing
or implementation prevents resuming an old run; start a new run instead.

OpenRouter, OpenAI and legacy TypeSafe key values are redacted from saved reports.
No API keys are part of pricing configuration. Existing historical reports can
still be displayed; absent cost-status fields are not retrospectively invented.

The existing optional observed-spend stop now accepts Jev's native USD charge.
If an attempt has unknown cost, the next call is stopped when a spend cap is active.
It is not a prepaid ceiling: one in-flight response can exceed the stop threshold.
API-attempt caps remain separate and unchanged.

## Optional validation run (user initiated only)

With keys and existing enable flags in `.env`, the unchanged small pilot selects
15 IID plus 15 held-out requests, including seven policy abstentions per provider:

```bash
python -m evaluation.benchmark --live-pilot --strategies rules,weighted,ml,openai,jev --allow-live-api --max-live-api-calls 50
```

This requires 46 initial calls and permits at most 50 external attempts including
retries; retries can exhaust the budget before every decision completes. Output
goes into a new `reports/live/<run_id>/`. No live validation is run automatically.

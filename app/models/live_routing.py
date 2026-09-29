"""Small provider-neutral decision and telemetry contracts; no raw response storage."""
from typing import Literal
from pydantic import Field, ConfigDict
from app.models.request import StrictModel

LiveReason = Literal['TASK_FIT', 'QUALITY_PRIORITY', 'LOW_COST_PREFERENCE', 'LOW_LATENCY_REQUIRED',
                     'CONTEXT_FIT', 'RELIABILITY_PRIORITY', 'ENTERPRISE_KNOWLEDGE_FIT']

CostStatus = Literal['CALCULATED_FROM_PROVIDER_USAGE', 'PROVIDER_REPORTED', 'PROVIDER_COST_UNAVAILABLE',
                     'PRICING_NOT_CONFIGURED', 'USAGE_NOT_RETURNED', 'INVALID_PROVIDER_USAGE',
                     'NOT_APPLICABLE_NO_ROUTE', 'BASELINE_REPLAY_ZERO_EXTERNAL_COST', 'NO_EXTERNAL_CALL']


class LiveRoutingDecision(StrictModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False, str_strip_whitespace=False)
    selected_model: str
    confidence: float | None = Field(ge=0, le=1)
    reason_codes: list[LiveReason]


class LiveTelemetry(StrictModel):
    provider: str
    strategy: str
    execution_mode: str = 'LIVE'
    evidence_type: str = 'SYNTHETIC'
    research_eligible: bool = False
    router_model: str
    returned_model: str | None = None
    prompt_version: str
    start_timestamp: str | None = None
    end_timestamp: str | None = None
    routing_latency_ms: float = 0
    first_attempt_latency_ms: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cached_tokens: int | None = None
    cached_input_tokens: int | None = None
    uncached_input_tokens: int | None = None
    reasoning_tokens: int | None = None
    service_tier: str | None = None
    provider_reported_cost_usd: float | None = None
    provider_usage: list[dict] = Field(default_factory=list)  # Sanitized per-attempt usage, not response bodies.
    cost_status: CostStatus = 'NO_EXTERNAL_CALL'
    pricing_version: str | None = None
    pricing_config_hash: str | None = None
    pricing_effective_date: str | None = None
    pricing_source: dict | str | None = None
    pricing_model_id: str | None = None
    cost_calculation_method: str | None = None
    routing_cost_usd: float | None = None
    api_request_id: str | None = None
    api_status: int | None = None
    api_calls: int = 0
    successful_api_calls: int = 0
    failed_api_calls: int = 0
    retry_count: int = 0
    refusal: bool = False
    error_type: str | None = None
    error_message_sanitized: str | None = None
    ineligible_selection_attempts: int = 0
    selected_model: str | None = None
    confidence: float | None = None
    reason_codes: list[LiveReason] = Field(default_factory=list)


class LiveObservation(StrictModel):
    status: str
    decision: LiveRoutingDecision | None = None
    telemetry: LiveTelemetry

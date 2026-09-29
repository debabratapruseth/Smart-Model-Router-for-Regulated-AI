from typing import Literal
from pydantic import Field
from app.models.request import StrictModel
from app.core.reason_codes import ReasonCode


class RejectedModel(StrictModel):
    model: str
    reason_codes: list[ReasonCode]
    policy_ids: list[str]


class Selection(StrictModel):
    selected_model: str
    routing_strategy: str
    confidence: float = Field(ge=0, le=1)


class Alternative(StrictModel):
    model: str
    score: float


class Estimates(StrictModel):
    routing_latency_ms: float = 0
    model_latency_ms: float | None = None
    cost_usd: float | None = None


class RoutingResponse(StrictModel):
    request_id: str
    status: Literal["ROUTED", "NO_ROUTE", "INVALID_DECISION", "UNAVAILABLE"]
    decision: Selection | None = None
    configuration: dict = Field(default_factory=dict)
    reason_code: ReasonCode | None = None
    reason_codes: list[ReasonCode] = Field(default_factory=list)
    features: dict = Field(default_factory=dict)
    eligible_models: list[str] = Field(default_factory=list)
    rejected_models: list[RejectedModel] = Field(default_factory=list)
    alternatives: list[Alternative] = Field(default_factory=list)
    score_breakdown: dict[str, float] = Field(default_factory=dict)
    candidate_scores: dict[str, dict[str, float]] = Field(default_factory=dict)
    estimated: Estimates = Field(default_factory=Estimates)
    fallback_used: bool = False
    human_review_required: bool = False
    requested_strategy: str = "weighted"
    strategy_mode: str = "local"

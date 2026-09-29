"""All default catalog measurements are SYNTHETIC DEMO VALUES."""
from typing import Literal
from pydantic import Field
from app.models.request import StrictModel, Classification


class Capabilities(StrictModel):
    text: bool = True
    vision: bool = False
    audio: bool = False
    reasoning: bool = False
    coding: bool = False
    embeddings: bool = False
    tool_calling: bool = False
    structured_output: bool = False
    streaming: bool = False
    enterprise_knowledge: bool = False


class Limits(StrictModel):
    context_window: int = Field(gt=0)
    max_output_tokens: int = Field(gt=0)


class Security(StrictModel):
    allowed_classifications: list[Classification]
    allowed_regions: list[str]
    approved_business_domains: list[str]


class Performance(StrictModel):
    p50_latency_ms: float = Field(gt=0)
    p95_latency_ms: float = Field(gt=0)
    success_rate: float = Field(ge=0, le=1)


class Economics(StrictModel):
    input_cost_per_1m_tokens: float = Field(ge=0)
    output_cost_per_1m_tokens: float = Field(ge=0)


class Quality(StrictModel):
    reasoning_score: float = Field(ge=0, le=1)
    summarization_score: float = Field(ge=0, le=1)
    classification_score: float = Field(ge=0, le=1)
    extraction_score: float = Field(ge=0, le=1)
    coding_score: float = Field(ge=0, le=1)
    finance_score: float = Field(ge=0, le=1)


class ModelSpec(StrictModel):
    model_id: str
    name: str
    provider: str
    version: str
    status: Literal["ACTIVE", "INACTIVE"] = "ACTIVE"
    region: str
    capabilities: Capabilities
    limits: Limits
    security: Security
    performance: Performance
    economics: Economics
    quality: Quality
    available: bool = True
    tier: str = "standard"
    executable_model: str | None = None

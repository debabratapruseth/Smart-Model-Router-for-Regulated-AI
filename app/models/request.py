"""Typed trust boundary. Classification is required: missing metadata fails closed."""
from enum import StrEnum
from uuid import uuid4
from pydantic import BaseModel, ConfigDict, Field
from typing import Literal


class Classification(StrEnum):
    PUBLIC = "PUBLIC"
    INTERNAL = "INTERNAL"
    CONFIDENTIAL = "CONFIDENTIAL"
    RESTRICTED = "RESTRICTED"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, allow_inf_nan=False)


class Metadata(StrictModel):
    data_classification: Classification
    contains_pii: bool = False
    contains_customer_data: bool = False
    contains_financial_data: bool = False
    contains_secrets: bool = False
    business_domain: str = "TECHNOLOGY"
    business_criticality: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = "MEDIUM"
    required_region: str | None = None
    allowed_providers: list[str] | None = None
    prohibited_providers: list[str] = Field(default_factory=list)
    allowed_models: list[str] | None = None
    prohibited_models: list[str] = Field(default_factory=list)
    latency_sla_ms: float | None = Field(default=None, gt=0)
    quality_requirement: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = "MEDIUM"
    reasoning_requirement: Literal["LOW", "MEDIUM", "HIGH"] = "LOW"
    cost_preference: Literal["MINIMIZE_COST", "BALANCED", "QUALITY_FIRST", "LATENCY_FIRST"] = "BALANCED"
    modality: Literal["TEXT", "IMAGE", "AUDIO", "MULTIMODAL"] = "TEXT"
    context_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=256, gt=0)
    requires_rag: bool = False
    needs_enterprise_knowledge: bool = False
    streaming_required: bool = False


class RoutingRequest(StrictModel):
    request_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1, max_length=128)
    application_id: str = Field(default="student-demo", min_length=1, max_length=128)
    prompt: str = Field(min_length=1, max_length=200000)
    metadata: Metadata


def normalize_request(request: RoutingRequest) -> RoutingRequest:
    """Conservative byte-based estimate; declared context includes retrieved input.

    This intentionally overestimates many texts. Production needs the endpoint's
    tokenizer, plus actual attachment and RAG token accounting.
    """
    normalized = request.model_copy(deep=True)
    normalized.metadata.context_tokens = max(len(request.prompt.encode("utf-8")), request.metadata.context_tokens)
    normalized.metadata.business_domain = request.metadata.business_domain.upper()
    if request.metadata.required_region:
        normalized.metadata.required_region = request.metadata.required_region.upper()
    return normalized

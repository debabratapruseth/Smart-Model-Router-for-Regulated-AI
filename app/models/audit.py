from datetime import datetime, timezone
from uuid import uuid4
from pydantic import Field
from app.models.request import StrictModel


class AuditRecord(StrictModel):
    audit_id: str = Field(default_factory=lambda: str(uuid4()))
    request_id: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    routing_strategy: str
    selected_model: str | None
    eligible_models: list[str]
    rejected_models: list[dict]
    reason_codes: list[str]
    policy_version: str
    model_catalog_version: str
    routing_config_version: str
    router_version: str = "1.0.0"
    confidence: float | None
    routing_latency_ms: float
    metadata: dict
    prompt_hash: str | None
    configuration_hashes: dict
    # Original snapshots make replay independent of future files and telemetry.
    configuration_snapshot: dict
    candidate_snapshot: list[dict]
    response: dict

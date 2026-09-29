from datetime import datetime, timezone
from pydantic import Field
from app.models.request import StrictModel


class TelemetryRecord(StrictModel):
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    model_id: str
    routing_strategy: str
    routing_latency_ms: float = Field(default=0, ge=0)
    execution_latency_ms: float | None = Field(default=None, ge=0)
    success: bool | None = None
    estimated_cost: float | None = Field(default=None, ge=0)
    quality_score: float | None = Field(default=None, ge=0, le=1)
    availability_status: str = "UNKNOWN"

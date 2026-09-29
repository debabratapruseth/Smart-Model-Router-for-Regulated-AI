from abc import ABC, abstractmethod
from types import SimpleNamespace
from app.models.routing_input import CanonicalRoutingInput
from pydantic import Field
from app.models.request import StrictModel, RoutingRequest
from app.models.model_catalog import ModelSpec


class StrategyUnavailable(RuntimeError):
    """Expected optional dependency or service failure; safe fallback is allowed."""


class StrategyResult(StrictModel):
    selected_model: str
    confidence: float = Field(ge=0, le=1)
    scores: dict[str, float] = Field(default_factory=dict)
    breakdowns: dict[str, dict[str, float]] = Field(default_factory=dict)


class RoutingStrategy(ABC):
    mode = "local"

    def __init__(self, config: dict):
        self.config = config

    def available(self, request: RoutingRequest | None = None) -> bool:
        return True

    @abstractmethod
    def route(self, request: RoutingRequest, features: dict, candidate_models: list[ModelSpec]) -> StrategyResult:
        """Candidates have already passed governance. Never add a candidate."""

    def route_canonical(self, value: CanonicalRoutingInput) -> StrategyResult:
        # Existing local algorithms need metadata and inferred features, not raw text.
        return self.route(SimpleNamespace(metadata=value.metadata), value.features, value.eligible_models)

    def choose(self, scores: dict[str, float], candidates: list[ModelSpec], breakdowns: dict | None = None) -> StrategyResult:
        regions = self.config["preferred_regions"]
        by_id = {m.model_id: m for m in candidates}
        selected = min(scores, key=lambda key: (-scores[key], by_id[key].region not in regions, key))
        return StrategyResult(selected_model=selected, confidence=max(0, min(1, scores[selected])),
                              scores=scores, breakdowns=breakdowns or {})

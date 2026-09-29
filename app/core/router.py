"""Policy-first orchestration. Invalid selections fail closed, never auto-correct."""
import logging
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from app.core.config_loader import ROOT, load_config, fingerprint, validate_routing
from app.core.features import extract_features
from app.core.policies import PolicyEngine
from app.core.reason_codes import ReasonCode as R
from app.core.scoring import estimated_cost
from app.core.validator import validate_decision, InvalidDecision
from app.models.audit import AuditRecord
from app.models.routing_input import CanonicalRoutingInput
from dataclasses import dataclass
from copy import deepcopy
from app.models.telemetry import TelemetryRecord
from app.models.request import RoutingRequest, normalize_request
from app.models.decision import RoutingResponse, Selection, Alternative
from app.services.audit_service import AuditService
from app.services.catalog_service import parse_catalog
from app.services.telemetry_service import TelemetryService
from app.strategies.base import StrategyUnavailable
from app.strategies.rules import RuleBasedRouter
from app.strategies.weighted import WeightedScoreRouter
from app.strategies.frontier_only import FrontierOnlyRouter
from app.strategies.cheapest_only import CheapestOnlyRouter
from app.strategies.ml import MLRouter
from app.strategies.llm import LLMRouter
from app.strategies.jev import JevRouter

logger = logging.getLogger(__name__)


@dataclass
class PreparedRouting:
    request: RoutingRequest
    features: dict
    models: list
    eligible: list
    rejected: list
    preparation_latency_ms: float

    def canonical(self):
        return CanonicalRoutingInput(metadata=self.request.metadata.model_copy(deep=True),
            features=deepcopy(self.features), eligible_models=[m.model_copy(deep=True) for m in self.eligible])


class ModelRouter:
    def __init__(self, config_dir: Path | None = None, runtime_dir: Path | None = None):
        self.config = load_config("routing", config_dir)
        self.policy_config = load_config("policies", config_dir)
        self.catalog_config = load_config("models", config_dir)
        validate_routing(self.config)
        self.models = parse_catalog(self.catalog_config)
        self.policy = PolicyEngine(self.policy_config, self.config)
        self.runtime_dir = runtime_dir or ROOT / "runtime"
        self.audit = AuditService(self.runtime_dir / "audit.sqlite3")
        self.telemetry = TelemetryService(self.runtime_dir / "telemetry.csv")
        self.strategies = {"rules": RuleBasedRouter(self.config), "weighted": WeightedScoreRouter(self.config),
                           "frontier_only": FrontierOnlyRouter(self.config), "cheapest_only": CheapestOnlyRouter(self.config),
                           "ml": MLRouter(self.config), "llm": LLMRouter(self.config), "jev": JevRouter(self.config)}
        if self.config["default_strategy"] not in self.strategies:
            raise ValueError("Unknown default strategy")
        if self.config["fallback_strategy"] not in {"weighted", "rules", "frontier_only", "cheapest_only"}:
            raise ValueError("Fallback must be an always-available local strategy")

    def current_models(self):
        if self.config["use_dynamic_telemetry"]:
            return self.telemetry.observed_models(self.models, self.config["telemetry_window"], self.config["telemetry_max_age_seconds"])
        return [m.model_copy(deep=True) for m in self.models]

    def prepare(self, request):
        start = perf_counter()
        request = normalize_request(request)
        features = extract_features(request, self.config)
        models = self.current_models()
        eligible, rejected = self.policy.filter(request, features, models)
        return PreparedRouting(request, features, models, eligible, rejected, (perf_counter() - start) * 1000)

    def route(self, request: RoutingRequest, strategy_name: str | None = None, allow_fallback: bool = True,
              ranking_ablation: str | None = None, *, prepared: PreparedRouting | None = None,
              structured_only: bool = False) -> RoutingResponse:
        start = perf_counter()
        shared_preparation_ms = prepared.preparation_latency_ms if prepared is not None else 0
        prepared = deepcopy(prepared) if prepared is not None else self.prepare(request)
        request = prepared.request
        name = strategy_name or self.config["default_strategy"]
        if name not in self.strategies:
            raise ValueError(f"Unknown routing strategy: {name}")
        features, models = prepared.features, prepared.models
        eligible, rejected = prepared.eligible, prepared.rejected
        response = RoutingResponse(request_id=request.request_id, status="NO_ROUTE", requested_strategy=name,
                                   features=features, eligible_models=[m.model_id for m in eligible], rejected_models=rejected,
                                   strategy_mode=self.strategies[name].mode,
                                   configuration={"use_rag": request.metadata.requires_rag or request.metadata.needs_enterprise_knowledge})
        strategy = self.strategies[name]
        actual_name = name
        if not strategy.available(request) and not allow_fallback:
            response.status = "UNAVAILABLE"
            response.reason_code = R.STRATEGY_UNAVAILABLE
            response.reason_codes = [R.STRATEGY_UNAVAILABLE]
        elif not eligible:
            response.reason_code = R.NO_POLICY_COMPLIANT_MODEL
            response.reason_codes = [R.NO_POLICY_COMPLIANT_MODEL]
        else:
            # Ablations can remove ranking information, NEVER governance inputs.
            ranking_request, ranking_features = request.model_copy(deep=True), dict(features)
            if ranking_ablation == "complexity":
                ranking_features["complexity"] = "LOW"
            elif ranking_ablation == "cost_preference":
                ranking_request.metadata.cost_preference = "BALANCED"
            elif ranking_ablation == "context_length":
                ranking_request.metadata.context_tokens = 0
            elif ranking_ablation == "quality_requirement":
                ranking_request.metadata.quality_requirement = "LOW"
            elif ranking_ablation == "enterprise_knowledge":
                ranking_features["enterprise_knowledge_need"] = "LOW"
                ranking_request.metadata.needs_enterprise_knowledge = False
            ranking_models = [m.model_copy(deep=True) for m in eligible]
            if ranking_ablation == "latency":
                for model in ranking_models:
                    model.performance.p95_latency_ms = 1
            try:
                try:
                    if not strategy.available(request):
                        raise StrategyUnavailable("Strategy unavailable")
                    if structured_only:
                        canonical = CanonicalRoutingInput(metadata=ranking_request.metadata, features=ranking_features,
                                                          eligible_models=ranking_models)
                        result = strategy.route_canonical(canonical)
                    else:
                        result = strategy.route(ranking_request, ranking_features, ranking_models)
                    validate_decision(result, set(response.eligible_models))
                    if (allow_fallback and name in {"ml", "llm", "jev"} and self.config["fallback_on_low_confidence"] and
                            result.confidence < self.config["confidence_threshold"] and name != self.config["fallback_strategy"]):
                        response.reason_codes.append(R.LOW_CONFIDENCE)
                        raise StrategyUnavailable("Low confidence")
                except StrategyUnavailable:
                    if not allow_fallback:
                        raise
                    actual_name = self.config["fallback_strategy"]
                    response.fallback_used = True
                    response.reason_codes.append(R.FALLBACK_USED)
                    result = self.strategies[actual_name].route(ranking_request, ranking_features, ranking_models)
                validate_decision(result, set(response.eligible_models))
                selected = next(m for m in eligible if m.model_id == result.selected_model)
                # Recheck original immutable governance inputs at the final boundary.
                if self.policy.evaluate_model(request, features, selected):
                    raise InvalidDecision("Final policy validation failed")
                response.status = "ROUTED"
                response.decision = Selection(selected_model=selected.model_id, routing_strategy=actual_name, confidence=result.confidence)
                response.reason_codes += [R.POLICY_REQUIRED, R.DATA_CLASSIFICATION_MATCH, R.REGION_MATCH,
                                          R.TASK_CAPABILITY_MATCH, R.CONTEXT_WINDOW_MATCH]
                if features["reasoning_need"] == "HIGH" and selected.capabilities.reasoning:
                    response.reason_codes.append(R.HIGH_REASONING_FIT)
                if request.metadata.cost_preference == "MINIMIZE_COST":
                    response.reason_codes.append(R.LOW_COST_PREFERENCE)
                if request.metadata.quality_requirement in ("HIGH", "CRITICAL"):
                    response.reason_codes.append(R.QUALITY_PRIORITY)
                if request.metadata.latency_sla_ms:
                    response.reason_codes.append(R.LOW_LATENCY_REQUIRED)
                if response.configuration["use_rag"]:
                    response.reason_codes.append(R.ENTERPRISE_RAG_REQUIRED)
                response.score_breakdown = result.breakdowns.get(selected.model_id, {})
                response.candidate_scores = result.breakdowns
                response.alternatives = [Alternative(model=m.model_id, score=result.scores.get(m.model_id, 0))
                                         for m in sorted(eligible, key=lambda m: (-result.scores.get(m.model_id, 0), m.model_id))
                                         if m.model_id != selected.model_id]
                response.estimated.model_latency_ms = selected.performance.p95_latency_ms
                response.estimated.cost_usd = estimated_cost(selected, request)
                response.human_review_required = result.confidence < self.config["human_review_threshold"]
            except InvalidDecision:
                response.status = "INVALID_DECISION"
                response.reason_code = R.INVALID_STRATEGY_DECISION
                response.reason_codes.append(R.INVALID_STRATEGY_DECISION)
            except StrategyUnavailable:
                response.status = "UNAVAILABLE"
                response.reason_code = R.STRATEGY_UNAVAILABLE
                response.reason_codes.append(R.STRATEGY_UNAVAILABLE)
        response.estimated.routing_latency_ms = (perf_counter() - start) * 1000 + shared_preparation_ms
        self._audit(request, response, models, actual_name)
        if response.decision:
            self.telemetry.append(TelemetryRecord(model_id=response.decision.selected_model,
                routing_strategy=actual_name, routing_latency_ms=response.estimated.routing_latency_ms,
                estimated_cost=response.estimated.cost_usd))
        logger.info("request_id=%s strategy=%s evaluated=%d rejected=%d selected=%s latency_ms=%.3f confidence=%s status=%s",
                    request.request_id, actual_name, len(models), len(rejected),
                    response.decision.selected_model if response.decision else None, response.estimated.routing_latency_ms,
                    response.decision.confidence if response.decision else None, response.status)
        return response

    def _audit(self, request, response, models, strategy):
        snapshots = {"routing": self.config, "policies": self.policy_config, "catalog": self.catalog_config}
        record = AuditRecord(request_id=request.request_id, routing_strategy=strategy,
            selected_model=response.decision.selected_model if response.decision else None,
            eligible_models=response.eligible_models, rejected_models=[r.model_dump(mode="json") for r in response.rejected_models],
            reason_codes=response.reason_codes, policy_version=self.policy_config["version"],
            model_catalog_version=self.catalog_config["version"], routing_config_version=self.config["version"],
            confidence=response.decision.confidence if response.decision else None,
            routing_latency_ms=response.estimated.routing_latency_ms, metadata=request.metadata.model_dump(mode="json"),
            prompt_hash=sha256(request.prompt.encode()).hexdigest() if self.config["store_prompt_hash"] else None,
            configuration_hashes={name: fingerprint(value) for name, value in snapshots.items()},
            configuration_snapshot=snapshots, candidate_snapshot=[m.model_dump(mode="json") for m in models],
            response=response.model_dump(mode="json"))
        self.audit.append(record)

    def compare(self, request):
        # Missing integrations are explicit; do not disguise them as weighted results.
        return {name: self.route(request, name, allow_fallback=False) for name in self.strategies}

"""Deterministic hard gates. Prompts never supply policy instructions."""
from app.models.decision import RejectedModel
from app.models.request import RoutingRequest
from app.models.model_catalog import ModelSpec
from app.core.reason_codes import ReasonCode as R
from app.core.scoring import task_quality


class PolicyEngine:
    def __init__(self, config: dict, routing_config: dict):
        self.config = config
        self.routing_config = routing_config

    def evaluate_model(self, request: RoutingRequest, features: dict, model: ModelSpec) -> RejectedModel | None:
        p, m = self.config, request.metadata
        codes, policy_ids = [], []

        def reject(condition: bool, code: R, policy: str) -> None:
            if condition:
                if code not in codes:
                    codes.append(code)
                if policy not in policy_ids:
                    policy_ids.append(policy)

        reject(model.status != "ACTIVE", R.MODEL_INACTIVE, "P01")
        reject(not model.available or model.provider in p["unavailable_providers"], R.MODEL_UNAVAILABLE, "P01")
        reject(model.provider not in p["approved_providers"] or model.provider in p["prohibited_providers"] or
               model.provider in m.prohibited_providers or
               (m.allowed_providers is not None and model.provider not in m.allowed_providers), R.PROVIDER_NOT_ALLOWED, "P02")
        reject(model.model_id in m.prohibited_models or
               (m.allowed_models is not None and model.model_id not in m.allowed_models), R.MODEL_NOT_APPROVED, "P03")
        reject(m.data_classification not in p["allowed_classifications"] or
               m.data_classification not in model.security.allowed_classifications, R.DATA_CLASSIFICATION_NOT_ALLOWED, "P04")
        reject(m.data_classification == "RESTRICTED" and model.provider not in p["restricted_allowed_providers"], R.PROVIDER_NOT_ALLOWED, "P05")
        reject(model.region not in model.security.allowed_regions or model.region in p["unavailable_regions"] or
               (m.required_region is not None and model.region != m.required_region) or
               (m.contains_pii and model.region not in p["pii_required_regions"]), R.REGION_NOT_ALLOWED, "P06")
        reject("*" not in model.security.approved_business_domains and m.business_domain not in model.security.approved_business_domains,
               R.DOMAIN_NOT_APPROVED, "P07")
        required = set(p["modality_capabilities"][m.modality]) | set(p["task_capabilities"].get(features["task_type"], []))
        reject(any(not getattr(model.capabilities, capability) for capability in required), R.CAPABILITY_NOT_SUPPORTED, "P08")
        reject(m.context_tokens + m.output_tokens > model.limits.context_window or m.output_tokens > model.limits.max_output_tokens or
               (m.context_tokens >= p["long_context_threshold"] and model.limits.context_window < p["long_context_minimum_window"]),
               R.CONTEXT_TOO_LARGE, "P09")
        reject(m.business_criticality == "CRITICAL" and model.performance.success_rate < p["critical_minimum_availability"], R.RELIABILITY_TOO_LOW, "P10")
        reject(m.latency_sla_ms is not None and model.performance.p95_latency_ms > m.latency_sla_ms, R.LATENCY_SLA_NOT_MET, "P11")
        for flag, key in [(m.contains_secrets, "secrets_allowed_providers"),
                          (m.contains_customer_data, "customer_data_allowed_providers"),
                          (m.contains_financial_data, "financial_data_allowed_providers")]:
            reject(flag and model.provider not in p[key], R.PROVIDER_NOT_ALLOWED, "P12")
        reject((m.requires_rag or m.needs_enterprise_knowledge) and not model.capabilities.enterprise_knowledge, R.CAPABILITY_NOT_SUPPORTED, "P13")
        reject(m.streaming_required and not model.capabilities.streaming, R.CAPABILITY_NOT_SUPPORTED, "P14")
        reject(m.reasoning_requirement == "HIGH" and not model.capabilities.reasoning, R.CAPABILITY_NOT_SUPPORTED, "P15")
        reject(task_quality(model, features, self.routing_config) < p["quality_minimum"][m.quality_requirement], R.QUALITY_TOO_LOW, "P16")
        return RejectedModel(model=model.model_id, reason_codes=codes, policy_ids=policy_ids) if codes else None

    def filter(self, request: RoutingRequest, features: dict, models: list[ModelSpec]) -> tuple[list[ModelSpec], list[RejectedModel]]:
        eligible, rejected = [], []
        for model in models:
            rejection = self.evaluate_model(request, features, model)
            if rejection:
                rejected.append(rejection)
            else:
                eligible.append(model)
        return eligible, rejected

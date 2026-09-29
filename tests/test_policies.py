import pytest
from app.core.features import extract_features
from app.models.request import normalize_request


@pytest.mark.parametrize("metadata,allowed", [
    ({"data_classification": "RESTRICTED"}, {"INTERNAL"}),
    ({"allowed_providers": ["APPROVED_CLOUD"]}, {"APPROVED_CLOUD"}),
    ({"contains_secrets": True}, {"INTERNAL"}),
    ({"contains_customer_data": True}, {"INTERNAL", "APPROVED_CLOUD"}),
    ({"contains_financial_data": True}, {"INTERNAL", "APPROVED_CLOUD"}),
])
def test_provider_gates(router, request_factory, metadata, allowed):
    response = router.route(request_factory(**metadata))
    assert response.status == "ROUTED"
    assert all(m.provider in allowed for m in router.models if m.model_id in response.eligible_models)


def test_pii_region(router, request_factory):
    response = router.route(request_factory(contains_pii=True))
    assert response.eligible_models
    assert all(m.region == "SINGAPORE" for m in router.models if m.model_id in response.eligible_models)


@pytest.mark.parametrize("metadata", [
    {"context_tokens": 2_000_000}, {"modality": "IMAGE", "allowed_models": ["local-small"]},
    {"modality": "AUDIO", "allowed_models": ["local-small"]}, {"contains_pii": True, "required_region": "US"},
    {"allowed_models": []}, {"allowed_providers": []}, {"latency_sla_ms": 1},
    {"allowed_models": ["enterprise"], "prohibited_models": ["enterprise"]},
    {"requires_rag": True, "allowed_models": ["local-small"]},
    {"streaming_required": True, "allowed_models": ["local-small"]},
    {"reasoning_requirement": "HIGH", "allowed_models": ["local-small"]},
    {"output_tokens": 10000},
])
def test_no_eligible(router, request_factory, metadata):
    assert router.route(request_factory(**metadata)).status == "NO_ROUTE"


def test_inactive_unavailable_domain(router, request_factory):
    model = next(m for m in router.models if m.model_id == "enterprise")
    for mutation in ["inactive", "unavailable", "domain"]:
        model.status, model.available = "ACTIVE", True
        model.security.approved_business_domains = ["*"]
        if mutation == "inactive":
            model.status = "INACTIVE"
        elif mutation == "unavailable":
            model.available = False
        else:
            model.security.approved_business_domains = ["AML"]
        assert router.route(request_factory(allowed_models=["enterprise"])).status == "NO_ROUTE"


def test_context_reserves_output(router, request_factory):
    response = router.route(request_factory(allowed_models=["local-small"], context_tokens=7900, output_tokens=256))
    assert response.status == "NO_ROUTE"


def test_critical_reliability(router, request_factory):
    result = router.route(request_factory(business_criticality="CRITICAL"))
    assert all(m.performance.success_rate >= .999 for m in router.models if m.model_id in result.eligible_models)


def test_denied_provider_overrides_allow(router, request_factory):
    result = router.route(request_factory(allowed_providers=["INTERNAL"], prohibited_providers=["INTERNAL"]))
    assert result.status == "NO_ROUTE"


def test_policy_is_deterministic(router, request_factory):
    request = normalize_request(request_factory())
    features = extract_features(request, router.config)
    assert router.policy.filter(request, features, router.models) == router.policy.filter(request, features, router.models)

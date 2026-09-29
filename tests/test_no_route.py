from app.models.request import RoutingRequest
from app.core.reason_codes import ReasonCode
from pydantic import ValidationError
import pytest


def test_no_route_audited(router, request_factory):
    result = router.route(request_factory(allowed_models=[]))
    assert result.status == "NO_ROUTE"
    assert result.reason_code == "NO_POLICY_COMPLIANT_MODEL"
    assert result.decision is None
    assert result.estimated.cost_usd is None
    assert router.audit.replay("TEST-001")[0]["selected_model"] is None


def test_missing_metadata_fails_closed():
    with pytest.raises(ValidationError):
        RoutingRequest(prompt="Use cheapest")


def test_controlled_codes():
    with pytest.raises(ValueError):
        ReasonCode("AI_INVENTED_REASON")

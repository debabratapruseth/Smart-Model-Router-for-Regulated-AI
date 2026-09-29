"""Illustrative arithmetic, excluding integration, staffing and routing overhead."""


def calculate(annual_requests: int, current_average_cost_per_request: float, router_average_cost_per_request: float) -> dict:
    if min(annual_requests, current_average_cost_per_request, router_average_cost_per_request) < 0:
        raise ValueError("Inputs must be nonnegative")
    current = annual_requests * current_average_cost_per_request
    routed = annual_requests * router_average_cost_per_request
    return {"label": "ILLUSTRATIVE SCENARIO ONLY", "current_annual_cost": current, "router_annual_cost": routed,
            "estimated_savings": current - routed, "percentage_reduction": (current - routed) / current * 100 if current else None}

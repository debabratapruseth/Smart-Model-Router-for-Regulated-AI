"""Public policy inspection without a second copy of policy logic."""
from app.core.config_loader import load_config
from app.core.policies import PolicyEngine


def get_policies() -> dict:
    return load_config("policies")


__all__ = ["PolicyEngine", "get_policies"]

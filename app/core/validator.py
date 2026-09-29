"""Validate every result before fallback or execution."""
from app.strategies.base import StrategyResult


class InvalidDecision(ValueError):
    pass


def validate_decision(result: StrategyResult, eligible_ids: set[str]) -> None:
    if result.selected_model not in eligible_ids:
        raise InvalidDecision("Strategy selected a model outside the eligible set")
    if not set(result.scores).issubset(eligible_ids) or not set(result.breakdowns).issubset(eligible_ids):
        raise InvalidDecision("Strategy supplied alternatives outside the eligible set")

"""Jev mock for offline use; verified LIVE adapter requires explicit pilot authorization."""
import os
from app.strategies.base import RoutingStrategy, StrategyUnavailable
from app.strategies.weighted import WeightedScoreRouter


class JevRouter(RoutingStrategy):
    @property
    def mode(self):
        return "mock" if os.getenv("JEV_MOCK_MODE", "true").lower() == "true" else "live"

    def available(self, request=None):
        return self.mode == "mock"

    def route(self, request, features, candidate_models):
        if not self.available(request):
            raise StrategyUnavailable("Use the explicit --allow-live-api live pilot runner")
        return WeightedScoreRouter(self.config).route(request, features, candidate_models)

    def route_live(self, canonical, **kwargs):
        from app.strategies.live_jev import JevLiveRouter
        return JevLiveRouter().execute(canonical, **kwargs)

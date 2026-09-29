from app.strategies.base import RoutingStrategy
from app.core.scoring import scoring_breakdowns


class WeightedScoreRouter(RoutingStrategy):
    def route(self, request, features, candidate_models):
        breakdowns = scoring_breakdowns(request, features, candidate_models, self.config)
        weights = self.config["routing_preferences"][request.metadata.cost_preference.lower()]
        scores = {key: sum(weights[name] * value for name, value in parts.items()) for key, parts in breakdowns.items()}
        return self.choose(scores, candidate_models, breakdowns)

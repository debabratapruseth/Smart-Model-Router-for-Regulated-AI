from app.strategies.base import RoutingStrategy
from app.core.scoring import estimated_cost, benefit


class CheapestOnlyRouter(RoutingStrategy):
    def route(self, request, features, candidate_models):
        costs = [estimated_cost(m, request) for m in candidate_models]
        return self.choose({m.model_id: benefit(cost, costs) for m, cost in zip(candidate_models, costs)}, candidate_models)

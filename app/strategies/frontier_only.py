from app.strategies.base import RoutingStrategy
from app.core.scoring import task_quality


class FrontierOnlyRouter(RoutingStrategy):
    """Strongest for this task among eligible models; not a fixed vendor ID."""
    def route(self, request, features, candidate_models):
        return self.choose({m.model_id: task_quality(m, features, self.config) for m in candidate_models}, candidate_models)

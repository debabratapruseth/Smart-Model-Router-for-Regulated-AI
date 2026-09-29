"""Optional CPU classifier, trained explicitly on a disjoint synthetic workload."""
from app.models.routing_input import CanonicalRoutingInput, encode_canonical
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from app.strategies.base import RoutingStrategy, StrategyUnavailable, StrategyResult


def encode(request, features) -> dict:
    """Legacy v1 feature encoding, retained for historical overlap reconstruction only."""
    return {**features, "classification": request.metadata.data_classification.value,
            "context": request.metadata.context_tokens, "preference": request.metadata.cost_preference,
            "quality": request.metadata.quality_requirement, "domain": request.metadata.business_domain,
            "modality": request.metadata.modality}


class MLRouter(RoutingStrategy):
    def __init__(self, config):
        super().__init__(config)
        self.vectorizer = DictVectorizer()
        self.classifier = None

    def available(self, request=None):
        return self.classifier is not None

    def fit(self, examples: list[tuple], seed: int = 42, hyperparameters: dict | None = None) -> None:
        if not examples:
            raise ValueError("No eligible training examples")
        x = self.vectorizer.fit_transform([encode_canonical(value) for value, label in examples])
        self.classifier = RandomForestClassifier(random_state=seed, **(hyperparameters or {"n_estimators": 60, "max_depth": 10, "n_jobs": 1}))
        self.classifier.fit(x, [label for value, label in examples])

    def route(self, request, features, candidate_models):
        return self.route_canonical(CanonicalRoutingInput(metadata=request.metadata, features=features,
                                                           eligible_models=candidate_models))

    def route_canonical(self, value):
        if not self.available():
            raise StrategyUnavailable("ML classifier has not been trained")
        probabilities = self.classifier.predict_proba(self.vectorizer.transform([encode_canonical(value)]))[0]
        eligible = {m.model_id for m in value.eligible_models}
        # Mask ineligible labels; do not renormalize away uncertainty from masking.
        scores = {label: float(prob) for label, prob in zip(self.classifier.classes_, probabilities) if label in eligible}
        if not scores or max(scores.values()) == 0:
            raise StrategyUnavailable("No supported eligible class")
        selected = min(scores, key=lambda key: (-scores[key], key))
        return StrategyResult(selected_model=selected, confidence=scores[selected], scores=scores)

"""Information boundary for the primary structured-only comparison."""
from pydantic import Field
from app.models.request import StrictModel, Metadata
from app.models.model_catalog import ModelSpec

FEATURE_SCHEMA_VERSION = "canonical-structured-2.0"


class CanonicalRoutingInput(StrictModel):
    # Task, complexity, reasoning and enterprise needs are the existing inferred features.
    metadata: Metadata
    features: dict
    eligible_models: list[ModelSpec] = Field(default_factory=list)
    # Deliberately no raw prompt, request ID, reference label or template ID.


def encode_canonical(value: CanonicalRoutingInput) -> dict:
    """Flatten all common structured information for DictVectorizer.

    Candidate keys use stable model IDs, so candidate ordering cannot change features.
    All strategies receive this information; each algorithm decides how to use it.
    """
    result = {}

    def flatten(prefix, item):
        if isinstance(item, dict):
            for key, child in sorted(item.items()):
                flatten(f"{prefix}.{key}", child)
        elif isinstance(item, list):
            result[prefix] = sorted(str(v) for v in item) or ["<empty>"]
        else:
            result[prefix] = "<none>" if item is None else item

    flatten("metadata", value.metadata.model_dump(mode="json"))
    flatten("features", value.features)
    for model in value.eligible_models:
        flatten(f"candidate.{model.model_id}", model.model_dump(mode="json"))
    return result

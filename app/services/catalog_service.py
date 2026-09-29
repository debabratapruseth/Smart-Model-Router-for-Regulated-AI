from app.models.model_catalog import ModelSpec


def parse_catalog(config: dict) -> list[ModelSpec]:
    models = [ModelSpec.model_validate(row) for row in config["models"]]
    if len({model.model_id for model in models}) != len(models):
        raise ValueError("Duplicate model IDs in catalog")
    return models

"""Small task metrics; missing references remain missing, never zero-filled."""
import json


def evaluate_quality(task: str, output: str, reference=None, manual_score: float | None = None) -> dict:
    if manual_score is not None:
        if not 0 <= manual_score <= 1:
            raise ValueError("Manual score must be between zero and one")
        return {"quality_score": manual_score, "evaluation_method": "manual_subjective"}
    if reference is None:
        return {"quality_score": None, "evaluation_method": "not_evaluated"}
    if task == "CLASSIFICATION":
        return {"quality_score": float(output.strip().casefold() == str(reference).strip().casefold()), "evaluation_method": "classification_accuracy"}
    if task == "EXTRACTION" and isinstance(reference, dict):
        try:
            prediction = json.loads(output)
        except (ValueError, TypeError):
            prediction = {}
        if not isinstance(prediction, dict):
            prediction = {}
        score = sum(prediction.get(k) == v for k, v in reference.items()) / len(reference) if reference else float(prediction == {})
        return {"quality_score": score, "evaluation_method": "field_match"}
    return {"quality_score": None, "evaluation_method": "manual_review_required"}

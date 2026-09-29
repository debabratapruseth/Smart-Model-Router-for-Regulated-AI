"""Intentionally simple educational rules, not a semantic classifier."""
from app.models.request import RoutingRequest


def extract_features(request: RoutingRequest, config: dict) -> dict:
    metadata = request.metadata
    prompt = request.prompt.lower()
    rules = config["feature_rules"]
    task = "QUESTION_ANSWERING"
    for name, keywords in rules["tasks"].items():
        if any(word in prompt for word in keywords):
            task = name
            break
    if metadata.modality == "IMAGE":
        task = "IMAGE"
    elif metadata.modality == "MULTIMODAL":
        task = "MULTIMODAL"
    elif metadata.modality == "AUDIO" and task != "TEXT_TO_SPEECH":
        task = "SPEECH_TO_TEXT"
    if metadata.requires_rag or metadata.needs_enterprise_knowledge:
        task = "RAG_QA"
    high_reasoning = any(word in prompt for word in rules["high_reasoning_keywords"])
    reasoning = "HIGH" if high_reasoning else metadata.reasoning_requirement
    complexity = "LOW"
    if metadata.context_tokens >= rules["medium_complexity_tokens"]:
        complexity = "MEDIUM"
    if metadata.context_tokens >= rules["high_complexity_tokens"] or reasoning == "HIGH":
        complexity = "HIGH"
    return {"task_type": task, "complexity": complexity, "reasoning_need": reasoning,
            "enterprise_knowledge_need": "HIGH" if metadata.requires_rag or metadata.needs_enterprise_knowledge else "LOW"}

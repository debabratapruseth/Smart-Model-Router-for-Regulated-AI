"""Opt-in text execution, isolated from policy and routing.

Catalog endpoints are explicit: no synthetic ID is sent as a real vendor model.
Non-text, embeddings, streaming and RAG execution are intentionally unsupported.
"""
import os
from time import perf_counter
import httpx
from app.core.config_loader import ROOT
from app.core.features import extract_features
from app.core.policies import PolicyEngine
from app.models.request import normalize_request
from app.models.model_catalog import ModelSpec
from app.models.telemetry import TelemetryRecord
from app.services.quality_service import evaluate_quality
from app.services.telemetry_service import append_csv, TelemetryService


class ExecutionService:
    def __init__(self, router):
        self.router = router

    def execute(self, request, decision, reference=None) -> dict:
        enabled = (os.getenv("ENABLE_MODEL_EXECUTION", "false").lower() == "true" and
                   self.router.config["execute_selected_model"])
        row = {"request_id": request.request_id, "model_id": None, "status": "DISABLED", "output": "",
               "execution_latency_ms": None, "input_tokens": None, "output_tokens": None,
               "estimated_cost_usd": None, "success": None, "quality_score": None, "evaluation_method": "not_evaluated"}
        if not enabled:
            return row
        if decision.status != "ROUTED":
            row["status"] = "NOT_ROUTED"
            return row
        if decision.human_review_required:
            row["status"] = "HUMAN_REVIEW_REQUIRED"
            return row
        # Require an original audit record. A caller cannot forge an execution decision.
        records = self.router.audit.replay(request.request_id)
        matches = [r for r in records if r["response"] == decision.model_dump(mode="json")]
        if not matches:
            raise ValueError("Execution requires the exact audited decision")
        original = matches[-1]
        request = normalize_request(request)
        from hashlib import sha256
        if original["prompt_hash"] is None or original["prompt_hash"] != sha256(request.prompt.encode()).hexdigest() or original["metadata"] != request.metadata.model_dump(mode="json"):
            raise ValueError("Execution input differs from audited input; prompt hashing must be enabled")
        selected_id = decision.decision.selected_model
        model = next(m for m in self.router.current_models() if m.model_id == selected_id)
        features = extract_features(request, self.router.config)
        if self.router.policy.evaluate_model(request, features, model):
            row["status"] = "POLICY_BLOCKED"
            return row
        row["model_id"] = model.model_id
        supported = (model.provider == "APPROVED_CLOUD" and model.executable_model and os.getenv("OPENAI_API_KEY") and
                     request.metadata.modality == "TEXT" and not decision.configuration["use_rag"] and
                     not request.metadata.streaming_required and features["task_type"] != "EMBEDDING")
        if not supported:
            row["status"] = "UNSUPPORTED_OR_UNCONFIGURED"
            append_csv(ROOT / "data/execution_results.csv", row)
            return row
        started = perf_counter()
        try:
            response = httpx.post(os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/") + "/chat/completions",
                                  headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
                                  json={"model": model.executable_model, "messages": [{"role": "user", "content": request.prompt}],
                                        "max_completion_tokens": request.metadata.output_tokens}, timeout=60)
            response.raise_for_status()
            body = response.json()
            row.update(status="SUCCEEDED", success=True, output=body["choices"][0]["message"]["content"],
                       input_tokens=body["usage"]["prompt_tokens"], output_tokens=body["usage"]["completion_tokens"])
            row["estimated_cost_usd"] = (row["input_tokens"] * model.economics.input_cost_per_1m_tokens +
                                          row["output_tokens"] * model.economics.output_cost_per_1m_tokens) / 1_000_000
            row.update(evaluate_quality(features["task_type"], row["output"], reference))
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError):
            row.update(status="FAILED", success=False)
        row["execution_latency_ms"] = (perf_counter() - started) * 1000
        append_csv(ROOT / "data/execution_results.csv", row)
        self.router.telemetry.append(TelemetryRecord(model_id=model.model_id, routing_strategy=decision.decision.routing_strategy,
            routing_latency_ms=decision.estimated.routing_latency_ms, execution_latency_ms=row["execution_latency_ms"],
            success=row["success"], estimated_cost=row["estimated_cost_usd"], quality_score=row["quality_score"]))
        return row

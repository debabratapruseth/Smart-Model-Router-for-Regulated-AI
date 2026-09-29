from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from app.models.request import RoutingRequest
from app.models.decision import RoutingResponse
from app.core.config_loader import ROOT

api = APIRouter()


def get_router(request: Request):
    return request.app.state.router


@api.get("/health")
def health():
    return {"status": "ok", "router_version": "1.0.0", "synthetic_demo": True}


@api.get("/models")
def models(router=Depends(get_router)):
    return {"version": router.catalog_config["version"], "synthetic_demo_values": True, "models": router.current_models()}


@api.get("/policies")
def policies(router=Depends(get_router)):
    return router.policy_config


@api.post("/route", response_model=RoutingResponse)
def route(request: RoutingRequest, router=Depends(get_router)):
    return router.route(request)


@api.post("/route/compare")
def compare(request: RoutingRequest, router=Depends(get_router)):
    return router.compare(request)


@api.get("/audit/{request_id}")
def replay(request_id: str, router=Depends(get_router)):
    records = router.audit.replay(request_id)
    if not records:
        raise HTTPException(404, "No audit records for request_id")
    return {"request_id": request_id, "records": records}


class SampleBenchmark(BaseModel):
    size: int = Field(default=20, ge=1, le=100)


@api.post("/benchmark/sample")
def benchmark_sample(options: SampleBenchmark, router=Depends(get_router)):
    from evaluation.benchmark import run_benchmark
    from scripts.generate_banking_benchmark import generate, ground_truth
    rows = generate(options.size)
    _, summary = run_benchmark(rows, ground_truth(rows), router=router,
                               strategies=["frontier_only", "cheapest_only", "rules", "weighted"], write_reports=False)
    import json
    return json.loads(summary.to_json(orient="records"))

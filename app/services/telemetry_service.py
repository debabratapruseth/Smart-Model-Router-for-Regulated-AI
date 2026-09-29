"""CSV telemetry with a rolling, fresh, per-model window; no fabricated successes."""
import csv
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
import numpy as np
from app.models.telemetry import TelemetryRecord

CSV_LOCK = RLock()


def append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with CSV_LOCK:
        needs_header = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if needs_header:
                writer.writeheader()
            writer.writerow(row)


class TelemetryService:
    def __init__(self, path: Path):
        self.path = path

    def append(self, record: TelemetryRecord) -> None:
        append_csv(self.path, record.model_dump())

    def observed_models(self, models: list, window: int, max_age: float) -> list:
        result = [m.model_copy(deep=True) for m in models]
        if not self.path.exists():
            return result
        with CSV_LOCK, self.path.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        now = datetime.now(timezone.utc)
        rows = [r for r in rows if (now - datetime.fromisoformat(r["timestamp"])).total_seconds() <= max_age]
        for model in result:
            observations = [r for r in rows if r["model_id"] == model.model_id and
                            (r["execution_latency_ms"] or r["success"] or r["availability_status"] != "UNKNOWN")][-window:]
            statuses = [r["availability_status"] for r in observations if r["availability_status"] != "UNKNOWN"]
            if statuses:
                model.available = model.available and statuses[-1] == "AVAILABLE"
            latencies = [float(r["execution_latency_ms"]) for r in observations if r["execution_latency_ms"]]
            if latencies:
                model.performance.p50_latency_ms = float(np.quantile(latencies, .5))
                model.performance.p95_latency_ms = float(np.quantile(latencies, .95))
            successes = [r["success"] == "True" for r in observations if r["success"] in ("True", "False")]
            if successes:
                model.performance.success_rate = sum(successes) / len(successes)
        return result

"""Load versioned YAML relative to the project, never to the shell's cwd."""
from pathlib import Path
from hashlib import sha256
import json
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(name: str, config_dir: Path | None = None) -> dict:
    path = (config_dir or ROOT / "config") / f"{name}.yaml"
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict) or not value.get("version"):
        raise ValueError(f"{path}: expected a mapping with a version")
    return value


def fingerprint(value: dict) -> str:
    return sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def validate_routing(config: dict) -> None:
    dimensions = {"quality", "task_fit", "latency", "cost", "reliability"}
    for weights in config["routing_preferences"].values():
        if set(weights) != dimensions or any(v < 0 for v in weights.values()):
            raise ValueError("Each preference must specify five nonnegative weights")
        if abs(sum(weights.values()) - 1) > 1e-6:
            raise ValueError("Routing weights must sum to one")
    for key in ("confidence_threshold", "human_review_threshold"):
        if not 0 <= config[key] <= 1:
            raise ValueError(f"{key} must be in [0, 1]")

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value) -> None:
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_paths(config: dict) -> dict[str, Path]:
    return {key: ROOT / value for key, value in config["paths"].items()}


def verify_inputs(config: dict, paths: dict[str, Path], split: str) -> None:
    targets = {key: paths[key] for key in ("train", split, "embeddings")}
    targets["model"] = Path(__file__).with_name("model.py")
    for key, path in targets.items():
        if sha256_file(path) != config["sha256"][key]:
            raise ValueError(f"Input differs from the exported experiment: {path}")


def require_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. CPU requires explicit --device cpu.")
    return device

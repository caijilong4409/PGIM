from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from common import require_device, resolve_paths, sha256_file, verify_inputs, write_json
from data import ColdStartDataModule
from metrics import score_split
from model import FullCatalogColdStartPGIM
from training import initialize_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("valid", "test"), default="test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output}")
    device = require_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    paths = resolve_paths(config)
    verify_inputs(config, paths, args.split)
    initialize_seed(int(checkpoint["seed"]))
    data = ColdStartDataModule(
        paths["train"],
        paths[args.split],
        paths["embeddings"],
        str(device),
        int(checkpoint["seed"]),
        int(config["batch_size"]),
    )
    model = FullCatalogColdStartPGIM(data, config["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics, rows = score_split(
        model,
        data,
        int(config["top_b"]),
        float(config["beta"]),
        config["ks"],
    )
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "split": args.split,
        "seed": int(checkpoint["seed"]),
        "epoch": int(checkpoint["epoch"]),
        "frozen_epoch_count": checkpoint["frozen_epoch_count"],
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "ks": config["ks"],
        "top_b": int(config["top_b"]),
        "beta": float(config["beta"]),
        "data": data.audit(),
        "metrics": metrics,
    }
    write_json(args.output / "metrics.json", report)
    with (args.output / "recommendations.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

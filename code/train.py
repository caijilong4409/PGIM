from __future__ import annotations

import argparse
from pathlib import Path

import torch

from common import (
    DEFAULT_CONFIG,
    load_config,
    require_device,
    resolve_paths,
    verify_inputs,
    write_json,
)
from data import ColdStartDataModule
from metrics import score_split
from model import FullCatalogColdStartPGIM
from training import fixed_training_run, initialize_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--epochs",
        type=int,
        help="Override the frozen epoch count (e.g. 1 for a smoke check).",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    epochs = int(config["epoch"]) if args.epochs is None else args.epochs
    if epochs < 1:
        parser.error("--epochs must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output}")
    device = require_device(args.device)
    paths = resolve_paths(config)

    verify_inputs(config, paths, "valid")
    initialize_seed(args.seed)
    data = ColdStartDataModule(
        paths["train"],
        paths["valid"],
        paths["embeddings"],
        str(device),
        args.seed,
        int(config["batch_size"]),
    )
    model = FullCatalogColdStartPGIM(data, config["model_config"]).to(device)
    args.output.mkdir(parents=True, exist_ok=True)
    result = fixed_training_run(model, data, float(config["learning_rate"]), epochs)
    metrics, _ = score_split(
        model,
        data,
        int(config["top_b"]),
        float(config["beta"]),
        config["ks"],
    )
    checkpoint_path = args.output / "checkpoint.pt"
    torch.save(
        {
            "model_state_dict": {
                name: value.detach().cpu() for name, value in model.state_dict().items()
            },
            "config": config,
            "seed": args.seed,
            "epoch": epochs,
            "frozen_epoch_count": epochs == int(config["epoch"]),
        },
        checkpoint_path,
    )
    write_json(args.output / "training_history.json", result["history"])
    write_json(args.output / "validation_metrics.json", metrics)
    print(f"Saved checkpoint: {checkpoint_path.resolve()}")
    print(f"Validation metrics: {metrics['overall']}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import random
import time
from copy import deepcopy

import numpy as np
import torch

from metrics import evaluate_grid, selection_key


def initialize_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train_epoch(model, data, optimizer) -> tuple[float, dict[str, float]]:
    model.train()
    data.train_dataloader.dataset.sample_negs()
    losses = []
    components = {}
    for batch in data.train_dataloader:
        batch = [tensor.long().to(data.device) for tensor in batch]
        optimizer.zero_grad(set_to_none=True)
        loss, details = model.cal_loss(batch)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        for name, value in details.items():
            components.setdefault(name, []).append(float(value.detach()))
    return float(np.mean(losses)), {
        name: float(np.mean(values)) for name, values in components.items()
    }


def search_training_run(
    model,
    data,
    learning_rate: float,
    max_epoch: int,
    validation_step: int,
    top_b_values: list[int],
    beta_values: list[float],
    ks: list[int],
) -> dict:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0)
    best_key = None
    best_state = None
    best = None
    history = []
    started = time.time()
    for epoch in range(1, max_epoch + 1):
        average_loss, components = train_epoch(model, data, optimizer)
        record = {"epoch": epoch, "loss": average_loss, "loss_components": components}
        if epoch % validation_step == 0:
            grid = evaluate_grid(model, data, top_b_values, beta_values, ks)
            for (top_b, beta), metrics in grid.items():
                key = selection_key(metrics, epoch, top_b, beta)
                if best_key is None or key > best_key:
                    best_key = key
                    best_state = deepcopy(
                        {name: value.detach().cpu() for name, value in model.state_dict().items()}
                    )
                    best = {
                        "epoch": epoch,
                        "top_b": int(top_b),
                        "beta": float(beta),
                        "metrics": metrics,
                    }
            selected = max(
                (
                    {"top_b": key[0], "beta": key[1], "metrics": value}
                    for key, value in grid.items()
                ),
                key=lambda item: selection_key(
                    item["metrics"], epoch, item["top_b"], item["beta"]
                ),
            )
            record["validation_best_at_epoch"] = selected
            print(
                f"epoch={epoch} loss={average_loss:.6f} "
                f"valid_R@20={selected['metrics']['recall'][-1]:.6f} "
                f"B={selected['top_b']} beta={selected['beta']:g}",
                flush=True,
            )
        history.append(record)
    if best is None or best_state is None:
        raise RuntimeError("No validation selection was produced")
    return {
        "selection": best,
        "state_dict": best_state,
        "history": history,
        "elapsed_seconds": time.time() - started,
    }


def fixed_training_run(model, data, learning_rate: float, epoch_count: int) -> dict:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0)
    history = []
    started = time.time()
    for epoch in range(1, epoch_count + 1):
        average_loss, components = train_epoch(model, data, optimizer)
        history.append({"epoch": epoch, "loss": average_loss, "loss_components": components})
        if epoch % 10 == 0 or epoch == epoch_count:
            print(f"fixed epoch={epoch}/{epoch_count} loss={average_loss:.6f}", flush=True)
    return {"history": history, "elapsed_seconds": time.time() - started}

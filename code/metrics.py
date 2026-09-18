from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch


METRICS = ("recall", "precision", "mrr", "ndcg")


def metrics_from_ranking(
    ranking: torch.Tensor,
    gold: torch.Tensor,
    ks: list[int],
    query_mask: torch.Tensor | None = None,
) -> dict[str, list[float]]:
    if query_mask is not None:
        ranking = ranking[query_mask]
        gold = gold[query_mask]
    label_counts = gold.sum(dim=1)
    valid = label_counts > 0
    ranking = ranking[valid]
    gold = gold[valid]
    label_counts = label_counts[valid].double()
    if ranking.shape[0] == 0:
        return {name: [0.0] * len(ks) for name in METRICS}
    hits = torch.gather(gold, 1, ranking).double()
    result = {name: [] for name in METRICS}
    positions = torch.arange(1, ranking.shape[1] + 1, device=ranking.device)
    discounts = 1.0 / torch.log2(positions.double() + 1.0)
    cumulative_discounts = discounts.cumsum(dim=0)
    for k in ks:
        cutoff_hits = hits[:, :k]
        hit_counts = cutoff_hits.sum(dim=1)
        result["recall"].append(float((hit_counts / label_counts).mean().cpu()))
        result["precision"].append(float((hit_counts / k).mean().cpu()))
        hit_positions = torch.where(
            cutoff_hits.bool(),
            positions[:k].expand_as(cutoff_hits),
            torch.full_like(cutoff_hits, k + 1, dtype=torch.long),
        )
        first = hit_positions.min(dim=1).values
        reciprocal = torch.where(
            first <= k,
            1.0 / first.double(),
            torch.zeros_like(first, dtype=torch.float64),
        )
        result["mrr"].append(float(reciprocal.mean().cpu()))
        dcg = (cutoff_hits * discounts[:k]).sum(dim=1)
        ideal_lengths = torch.minimum(label_counts.long(), torch.tensor(k, device=gold.device))
        idcg = cumulative_discounts[ideal_lengths - 1]
        result["ndcg"].append(float((dcg / idcg).mean().cpu()))
    return result


@torch.no_grad()
def evaluate_grid(model, data, top_b_values, beta_values, ks) -> dict[tuple[int, float], dict]:
    model.eval()
    state = model.prepare_inference()
    results = {}
    collaborative_grid, text = model.score_components_grid(
        data.query_u_tensor, state, [int(value) for value in top_b_values]
    )
    for top_b in top_b_values:
        collaborative = collaborative_grid[int(top_b)]
        for beta in beta_values:
            scores = (1.0 - float(beta)) * collaborative + float(beta) * text
            ranking = torch.topk(scores, k=max(ks), dim=-1).indices
            results[(int(top_b), float(beta))] = metrics_from_ranking(
                ranking, data.query_gold, list(ks)
            )
    return results


def selection_key(
    metrics: dict[str, list[float]],
    epoch: int,
    top_b: int,
    beta: float,
) -> tuple:
    return (
        metrics["recall"][-1],
        metrics["ndcg"][-1],
        metrics["mrr"][-1],
        -int(epoch),
        -int(top_b),
        -float(beta),
    )


@torch.no_grad()
def score_split(model, data, top_b: int, beta: float, ks: list[int]) -> tuple[dict, list[dict]]:
    model.eval()
    state = model.prepare_inference()
    collaborative, text = model.score_components(data.query_u_tensor, state, top_b)
    scores = (1.0 - beta) * collaborative + beta * text
    ranking = torch.topk(scores, k=max(ks), dim=-1).indices
    known_gold = data.query_gold & data.known_api_mask.unsqueeze(0)
    unseen_gold = data.query_gold & (~data.known_api_mask).unsqueeze(0)
    unseen_counts = unseen_gold.sum(dim=1)
    total_counts = data.query_gold.sum(dim=1)
    groups = {
        "all_known": unseen_counts == 0,
        "any_unseen": unseen_counts > 0,
        "all_unseen": unseen_counts == total_counts,
    }
    metrics = {
        "overall": metrics_from_ranking(ranking, data.query_gold, ks),
        "known_positive": metrics_from_ranking(ranking, known_gold, ks),
        "unseen_positive": metrics_from_ranking(ranking, unseen_gold, ks),
        "groups": {
            name: {
                "queries": int(mask.sum().cpu()),
                "metrics": metrics_from_ranking(ranking, data.query_gold, ks, mask),
            }
            for name, mask in groups.items()
        },
    }

    known_scores = scores[:, data.known_api_mask]
    known_ranking_local = torch.topk(known_scores, k=max(ks), dim=-1).indices
    known_full_indices = torch.nonzero(data.known_api_mask, as_tuple=False).reshape(-1)
    known_ranking = known_full_indices[known_ranking_local]
    metrics["train_known_candidate_diagnostic"] = metrics_from_ranking(
        known_ranking, known_gold, ks
    )

    ranking_np = ranking.cpu().numpy()
    coll_np = collaborative.cpu().numpy()
    text_np = text.cpu().numpy()
    score_np = scores.cpu().numpy()
    rows = []
    for row_index, candidate_indices in enumerate(ranking_np):
        rows.append(
            {
                "mashup_raw_id": int(data.query_raw_ids[row_index]),
                "gold_api_raw_ids": [int(value) for value in data.query_gold_raw_ids[row_index]],
                "recommended_api_raw_ids": [
                    int(data.all_api_raw_ids[index]) for index in candidate_indices
                ],
                "structural_score": [
                    float(coll_np[row_index, index]) for index in candidate_indices
                ],
                "text_score": [
                    float(text_np[row_index, index]) for index in candidate_indices
                ],
                "fused_score": [float(score_np[row_index, index]) for index in candidate_indices],
            }
        )
    return metrics, rows

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def minibatch_alignment_loss(
    graph_catalog: torch.Tensor,
    semantic_catalog: torch.Tensor,
    entity_ids: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Align unique entities from one interaction batch using in-batch negatives."""
    unique_ids = torch.unique(entity_ids, sorted=True)
    graph = graph_catalog[unique_ids]
    semantic = semantic_catalog[unique_ids]
    logits = graph @ semantic.T / temperature
    targets = torch.arange(unique_ids.shape[0], device=entity_ids.device)
    return F.cross_entropy(logits, targets)


class FullCatalogColdStartPGIM(nn.Module):
    def __init__(self, data, config: dict):
        super().__init__()
        self.n_mashups = data.n_mashups
        self.n_train_apis = data.n_train_apis
        self.embedding_size = int(config["embedding_size"])
        self.layer_num = int(config["layer_num"])
        self.keep_rate = float(config["keep_rate"])
        self.reg_weight = float(config["reg_weight"])
        self.bpr_weight = float(config.get("bpr_weight", 1.0))
        self.alpha = float(config["alpha"])
        self.gamma = float(config["gamma"])
        self.kd_temperature = float(config["kd_temperature"])

        self.register_buffer("torch_adj", data.torch_adj)
        self.register_buffer("train_u", torch.as_tensor(data.train_u, dtype=torch.float32))
        self.register_buffer(
            "train_api_u", torch.as_tensor(data.train_api_u, dtype=torch.float32)
        )
        self.register_buffer(
            "all_api_u", torch.as_tensor(data.all_api_u, dtype=torch.float32)
        )
        self.register_buffer(
            "known_full_indices",
            torch.as_tensor(data.train_known_full_indices, dtype=torch.long),
        )

        self.mashup_emb = nn.Parameter(torch.empty(self.n_mashups, self.embedding_size))
        self.api_emb = nn.Parameter(torch.empty(self.n_train_apis, self.embedding_size))
        nn.init.xavier_uniform_(self.mashup_emb)
        nn.init.xavier_uniform_(self.api_emb)
        semantic_dim = self.train_u.shape[1]
        hidden_dim = (semantic_dim + self.embedding_size) // 2
        self.mlp = nn.Sequential(
            nn.Linear(semantic_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, self.embedding_size),
        )
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

    def _drop_edges(self, adjacency: torch.Tensor, keep_rate: float) -> torch.Tensor:
        if keep_rate >= 1.0:
            return adjacency
        adjacency = adjacency.coalesce()
        mask = torch.rand(adjacency.values().shape, device=adjacency.device) < keep_rate
        return torch.sparse_coo_tensor(
            adjacency.indices()[:, mask],
            adjacency.values()[mask],
            adjacency.shape,
            device=adjacency.device,
        ).coalesce()

    def forward_graph(self, keep_rate: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
        adjacency = self._drop_edges(self.torch_adj, keep_rate)
        embeddings = torch.cat([self.mashup_emb, self.api_emb], dim=0)
        layers = [embeddings]
        for _ in range(self.layer_num):
            embeddings = torch.sparse.mm(adjacency, embeddings)
            layers.append(embeddings)
        final = torch.stack(layers, dim=0).sum(dim=0)
        return final[: self.n_mashups], final[self.n_mashups :]

    @staticmethod
    def _bpr(anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        positive_score = (anchor * positive).sum(dim=-1)
        negative_score = (anchor * negative).sum(dim=-1)
        return F.softplus(negative_score - positive_score).mean()

    def cal_loss(self, batch: list[torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        mashup_ids, positive_ids, negative_ids = batch
        mashup_e, api_e = self.forward_graph(self.keep_rate)
        anchor_e = mashup_e[mashup_ids]
        positive_e = api_e[positive_ids]
        negative_e = api_e[negative_ids]
        train_z = self.mlp(self.train_u)
        api_z = self.mlp(self.train_api_u)
        anchor_z = train_z[mashup_ids]
        positive_z = api_z[positive_ids]
        negative_z = api_z[negative_ids]

        graph_bpr = self.bpr_weight * self._bpr(anchor_e, positive_e, negative_e)
        text_bpr = self.gamma * self._bpr(anchor_z, positive_z, negative_z)
        # Interaction batches may contain the same Mashup or API more than once.
        # De-duplicate entities before InfoNCE so another copy of the positive is
        # never mislabeled as a negative. Positive and sampled-negative APIs form
        # one API mini-batch, matching the manuscript's entity-level objective.
        alignment = self.alpha * (
            minibatch_alignment_loss(
                mashup_e,
                train_z,
                mashup_ids,
                self.kd_temperature,
            )
            + minibatch_alignment_loss(
                api_e,
                api_z,
                torch.cat([positive_ids, negative_ids]),
                self.kd_temperature,
            )
        )
        regularization = self.reg_weight * sum(
            parameter.square().sum() for parameter in self.parameters()
        ) / mashup_ids.shape[0]
        loss = graph_bpr + text_bpr + alignment + regularization
        return loss, {
            "graph_bpr": graph_bpr,
            "text_bpr": text_bpr,
            "alignment": alignment,
            "regularization": regularization,
        }

    @torch.no_grad()
    def prepare_inference(self) -> dict[str, torch.Tensor]:
        history_e, known_api_e = self.forward_graph(1.0)
        all_api_e = torch.zeros(
            (self.all_api_u.shape[0], known_api_e.shape[1]),
            dtype=known_api_e.dtype,
            device=known_api_e.device,
        )
        all_api_e[self.known_full_indices] = known_api_e
        return {
            "history_e": history_e,
            # APIs absent from the train graph have an exact zero structural
            # embedding and therefore receive no fabricated collaborative score.
            "all_api_e": all_api_e,
            "all_api_z": self.mlp(self.all_api_u),
            "train_u": F.normalize(self.train_u, dim=-1),
        }

    @torch.no_grad()
    def score_components(
        self,
        query_u: torch.Tensor,
        state: dict[str, torch.Tensor],
        top_b: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        grids, text_score = self.score_components_grid(
            query_u, state, [int(top_b)]
        )
        return grids[int(top_b)], text_score

    @torch.no_grad()
    def score_components_grid(
        self,
        query_u: torch.Tensor,
        state: dict[str, torch.Tensor],
        top_b_values: list[int],
    ) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
        query_u = query_u.to(self.train_u.device)
        similarities = F.normalize(query_u, dim=-1) @ state["train_u"].T
        bounded = [min(int(value), similarities.shape[1]) for value in top_b_values]
        max_top_b = max(bounded)
        top_values, top_indices = torch.topk(similarities, k=max_top_b, dim=-1)
        neighbors = state["history_e"][top_indices]
        collaborative_by_top_b = {}
        for requested, top_b in zip(top_b_values, bounded):
            weights = torch.softmax(top_values[:, :top_b], dim=-1)
            retrieved = (weights.unsqueeze(-1) * neighbors[:, :top_b]).sum(dim=1)
            collaborative_by_top_b[int(requested)] = retrieved @ state["all_api_e"].T
        query_z = self.mlp(query_u)
        text_score = query_z @ state["all_api_z"].T
        return collaborative_by_top_b, text_score

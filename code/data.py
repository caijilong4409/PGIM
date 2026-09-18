from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader, Dataset

from common import read_jsonl


def unique_api_ids(case: dict) -> list[int]:
    return list(dict.fromkeys(int(api["API_ID"]) for api in case["APIs"]))


class PairwiseTrainDataset(Dataset):
    def __init__(self, train_mat: sp.csr_matrix, seed: int):
        coo = train_mat.tocoo()
        self.rows = coo.row.astype(np.int64)
        self.cols = coo.col.astype(np.int64)
        self.positive_sets = [
            set(train_mat[row].indices) for row in range(train_mat.shape[0])
        ]
        self.negs = np.zeros(len(self.rows), dtype=np.int64)
        self.rng = np.random.default_rng(seed)
        self.n_apis = train_mat.shape[1]

    def sample_negs(self) -> None:
        for index, mashup in enumerate(self.rows):
            while True:
                negative = int(self.rng.integers(self.n_apis))
                if negative not in self.positive_sets[int(mashup)]:
                    self.negs[index] = negative
                    break

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[int, int, int]:
        return int(self.rows[index]), int(self.cols[index]), int(self.negs[index])


def normalized_adjacency(
    train_mat: sp.csr_matrix, device: torch.device
) -> torch.Tensor:
    n_mashups, n_apis = train_mat.shape
    adjacency = sp.vstack(
        [
            sp.hstack([sp.csr_matrix((n_mashups, n_mashups)), train_mat]),
            sp.hstack([train_mat.T, sp.csr_matrix((n_apis, n_apis))]),
        ]
    ).tocsr()
    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    inverse = np.zeros_like(degree, dtype=np.float32)
    nonzero = degree > 0
    inverse[nonzero] = np.power(degree[nonzero], -0.5)
    normalized = (sp.diags(inverse) @ adjacency @ sp.diags(inverse)).tocoo()
    indices = torch.from_numpy(
        np.vstack([normalized.row, normalized.col]).astype(np.int64)
    )
    values = torch.from_numpy(normalized.data.astype(np.float32))
    return torch.sparse_coo_tensor(
        indices,
        values,
        normalized.shape,
        device=device,
    ).coalesce()


class ColdStartDataModule:

    def __init__(
        self,
        train_file: Path,
        query_file: Path,
        embedding_file: Path,
        device: str,
        seed: int,
        train_batch_size: int = 1024,
    ):
        self.device = torch.device(device)
        self.train_cases = read_jsonl(Path(train_file))
        self.query_cases = read_jsonl(Path(query_file))
        train_ids = {int(case["Mashup_ID"]) for case in self.train_cases}
        query_ids = {int(case["Mashup_ID"]) for case in self.query_cases}
        overlap = train_ids & query_ids
        if overlap:
            raise ValueError(f"Train/query Mashup leakage: {len(overlap)}")

        self.train_mashup_raw_ids = np.asarray(sorted(train_ids), dtype=np.int64)
        train_api_ids = {
            raw_id for case in self.train_cases for raw_id in unique_api_ids(case)
        }
        self.train_api_raw_ids = np.asarray(sorted(train_api_ids), dtype=np.int64)
        self.mashup2idx = {
            int(raw_id): index for index, raw_id in enumerate(self.train_mashup_raw_ids)
        }
        self.api2idx = {
            int(raw_id): index for index, raw_id in enumerate(self.train_api_raw_ids)
        }
        rows = []
        cols = []
        for case in self.train_cases:
            mashup_idx = self.mashup2idx[int(case["Mashup_ID"])]
            for raw_api_id in unique_api_ids(case):
                rows.append(mashup_idx)
                cols.append(self.api2idx[raw_api_id])
        self.train_mat = sp.csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)),
            shape=(len(self.train_mashup_raw_ids), len(self.train_api_raw_ids)),
        )
        if self.train_mat.shape != (4599, 1068) or self.train_mat.nnz != 8589:
            raise RuntimeError(
                f"Training graph snapshot changed: shape={self.train_mat.shape} nnz={self.train_mat.nnz}"
            )
        self.torch_adj = normalized_adjacency(self.train_mat, self.device)

        with np.load(embedding_file, allow_pickle=False) as artifact:
            all_mashup_ids = artifact["mashup_ids"].astype(np.int64)
            all_api_ids = artifact["api_ids"].astype(np.int64)
            all_mashup_u = artifact["mashup_embeddings"].astype(np.float32)
            all_api_u = artifact["api_embeddings"].astype(np.float32)
        if all_mashup_u.shape != (5747, 768) or all_api_u.shape != (1210, 768):
            raise RuntimeError(
                f"Embedding shape mismatch: mashup={all_mashup_u.shape} api={all_api_u.shape}"
            )
        if len(set(map(int, all_mashup_ids))) != len(all_mashup_ids):
            raise ValueError("Duplicate Mashup IDs in embedding artifact")
        if len(set(map(int, all_api_ids))) != len(all_api_ids):
            raise ValueError("Duplicate API IDs in embedding artifact")
        mashup_lookup = {
            int(raw_id): index for index, raw_id in enumerate(all_mashup_ids)
        }
        api_lookup = {int(raw_id): index for index, raw_id in enumerate(all_api_ids)}
        missing_mashups = (train_ids | query_ids) - set(mashup_lookup)
        missing_apis = train_api_ids - set(api_lookup)
        if missing_mashups or missing_apis:
            raise ValueError(
                f"Incomplete embeddings: mashups={len(missing_mashups)} apis={len(missing_apis)}"
            )
        self.all_api_raw_ids = all_api_ids
        self.all_api_lookup = api_lookup
        self.train_known_full_indices = np.asarray(
            [api_lookup[int(raw_id)] for raw_id in self.train_api_raw_ids],
            dtype=np.int64,
        )
        self.train_u = np.stack(
            [
                all_mashup_u[mashup_lookup[int(raw_id)]]
                for raw_id in self.train_mashup_raw_ids
            ]
        )
        self.train_api_u = np.stack(
            [all_api_u[api_lookup[int(raw_id)]] for raw_id in self.train_api_raw_ids]
        )
        self.all_api_u = all_api_u
        self.query_raw_ids = np.asarray(
            [int(case["Mashup_ID"]) for case in self.query_cases], dtype=np.int64
        )
        self.query_u = np.stack(
            [all_mashup_u[mashup_lookup[int(raw_id)]] for raw_id in self.query_raw_ids]
        ).astype(np.float32)
        self.query_gold_raw_ids = [unique_api_ids(case) for case in self.query_cases]
        missing_gold = {
            raw_id
            for labels in self.query_gold_raw_ids
            for raw_id in labels
            if raw_id not in api_lookup
        }
        if missing_gold:
            raise ValueError(
                f"Gold APIs missing from full catalog: {sorted(missing_gold)[:20]}"
            )
        self.query_gold_indices = [
            [api_lookup[raw_id] for raw_id in labels]
            for labels in self.query_gold_raw_ids
        ]
        gold = np.zeros((len(self.query_cases), len(all_api_ids)), dtype=np.bool_)
        for row_index, indices in enumerate(self.query_gold_indices):
            gold[row_index, indices] = True
        self.query_gold = torch.as_tensor(gold, dtype=torch.bool, device=self.device)
        known_mask = np.zeros(len(all_api_ids), dtype=np.bool_)
        known_mask[self.train_known_full_indices] = True
        self.known_api_mask = torch.as_tensor(
            known_mask, dtype=torch.bool, device=self.device
        )
        self.query_u_tensor = torch.as_tensor(
            self.query_u, dtype=torch.float32, device=self.device
        )

        dataset = PairwiseTrainDataset(self.train_mat, seed)
        generator = torch.Generator()
        generator.manual_seed(seed)
        self.train_dataloader = DataLoader(
            dataset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=0,
            generator=generator,
        )

    @property
    def n_mashups(self) -> int:
        return len(self.train_mashup_raw_ids)

    @property
    def n_train_apis(self) -> int:
        return len(self.train_api_raw_ids)

    @property
    def n_all_apis(self) -> int:
        return len(self.all_api_raw_ids)

    def audit(self) -> dict:
        unseen_labels = sum(
            raw_id not in self.api2idx
            for labels in self.query_gold_raw_ids
            for raw_id in labels
        )
        return {
            "train_mashups": self.n_mashups,
            "train_known_apis": self.n_train_apis,
            "all_candidate_apis": self.n_all_apis,
            "train_edges": int(self.train_mat.nnz),
            "query_mashups": len(self.query_cases),
            "query_labels": sum(map(len, self.query_gold_raw_ids)),
            "query_unseen_api_labels": int(unseen_labels),
        }

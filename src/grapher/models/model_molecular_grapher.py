from __future__ import annotations

import math
from typing import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from grapher.generation.molecular_rewiring import MolecularRewireAction


def _sinusoidal_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half = max(int(dim) // 2, 1)
    frequencies = torch.exp(
        -torch.arange(half, dtype=torch.float32, device=t.device)
        * (math.log(float(max_period)) / float(half))
    )
    arguments = t.float().unsqueeze(-1) * frequencies
    embedding = torch.cat([torch.sin(arguments), torch.cos(arguments)], dim=-1)
    if embedding.size(-1) < int(dim):
        embedding = torch.cat(
            [embedding, embedding.new_zeros((*embedding.shape[:-1], int(dim) - embedding.size(-1)))],
            dim=-1,
        )
    return embedding[..., : int(dim)]


class EdgeAwareGINLayer(nn.Module):
    """Small edge-aware GIN-style layer without a PyG version dependency."""

    def __init__(self, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.eps = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        node_h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_h: torch.Tensor,
    ) -> torch.Tensor:
        aggregate = torch.zeros_like(node_h)
        if edge_index.numel() > 0:
            target, source = edge_index[0].long(), edge_index[1].long()
            messages = self.message(node_h[source] + edge_h)
            aggregate.index_add_(0, target, messages)
        return self.update((1.0 + self.eps) * node_h + aggregate)


class MolecularGraphER(nn.Module):
    """Attributed GraphER scorer for hard typed rewiring actions.

    Node types remain fixed throughout a trajectory. Candidate edge labels are
    proposed from an empirical endpoint-conditioned prior and supplied as part
    of ``MolecularRewireAction``. The network scores the complete valid action
    ``(e1,e2,r,c1,c2)``; invalid-valence actions are filtered before scoring.
    """

    architecture = "molecular_typed_action_grapher"

    def __init__(
        self,
        *,
        node_type_values: Sequence[int],
        edge_type_values: Sequence[int],
        hidden_dim: int,
        num_layer: int,
        T: int,
        max_nodes: int,
        degree_histogram_dim: int | None = None,
        k_eigen: int = 4,
        time_embedding_dim: int | None = None,
        local_feature_dim: int = 24,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.node_type_values = [int(value) for value in node_type_values]
        self.edge_type_values = [int(value) for value in edge_type_values]
        if not self.node_type_values:
            raise ValueError("MolecularGraphER requires at least one node type.")
        if not self.edge_type_values:
            raise ValueError("MolecularGraphER requires at least one edge type.")
        self.node_type_to_index = {
            int(value): index for index, value in enumerate(self.node_type_values)
        }
        self.edge_type_to_index = {
            int(value): index for index, value in enumerate(self.edge_type_values)
        }
        self.hidden_dim = int(hidden_dim)
        self.num_layer = int(num_layer)
        self.T = int(T)
        self.max_nodes = int(max_nodes)
        self.degree_histogram_dim = int(degree_histogram_dim or max_nodes)
        self.k_eigen = int(k_eigen)
        self.time_embedding_dim = int(time_embedding_dim or hidden_dim)
        self.local_feature_dim = int(local_feature_dim)
        self.dropout = float(dropout)

        self.node_type_embedding = nn.Embedding(len(self.node_type_values), self.hidden_dim)
        self.edge_type_embedding = nn.Embedding(len(self.edge_type_values), self.hidden_dim)
        self.node_aux_encoder = nn.Sequential(
            nn.Linear(1 + self.k_eigen, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.message_layers = nn.ModuleList(
            [EdgeAwareGINLayer(self.hidden_dim, dropout=self.dropout) for _ in range(self.num_layer)]
        )
        self.graph_encoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.degree_encoder = nn.Sequential(
            nn.Linear(self.degree_histogram_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.time_encoder = nn.Sequential(
            nn.Linear(self.time_embedding_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.orientation_embedding = nn.Embedding(2, self.hidden_dim)
        self.local_encoder = nn.Sequential(
            nn.Linear(self.local_feature_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        # graph, degree, time, three endpoint-pair terms, orientation, local,
        # and three proposed-bond-pair terms = 11 hidden-width blocks.
        self.action_scorer = nn.Sequential(
            nn.Linear(self.hidden_dim * 11, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, max(self.hidden_dim // 2, 1)),
            nn.ReLU(),
            nn.Linear(max(self.hidden_dim // 2, 1), 1),
        )
        nn.init.xavier_uniform_(self.node_type_embedding.weight)
        nn.init.xavier_uniform_(self.edge_type_embedding.weight)
        nn.init.xavier_uniform_(self.orientation_embedding.weight)

    def _normalized_time(self, value: int | float | torch.Tensor) -> float:
        if torch.is_tensor(value):
            value = float(value.detach().cpu().reshape(-1)[0].item())
        value = float(value)
        if value > 1.0:
            value /= max(float(self.T), 1.0)
        return max(0.0, min(1.0, value))

    def encode_nodes(
        self,
        node_types: torch.Tensor,
        degree_features: torch.Tensor,
        pe: torch.Tensor,
        edge_index: torch.Tensor,
        edge_types: torch.Tensor,
    ) -> torch.Tensor:
        if degree_features.ndim != 2 or degree_features.size(1) != 1:
            raise ValueError(
                f"degree_features must have shape [N,1], got {tuple(degree_features.shape)}"
            )
        if pe.ndim != 2 or pe.size(1) != self.k_eigen:
            raise ValueError(f"pe must have shape [N,{self.k_eigen}], got {tuple(pe.shape)}")
        node_h = self.node_type_embedding(node_types.long())
        node_h = node_h + self.node_aux_encoder(torch.cat([degree_features.float(), pe.float()], dim=-1))
        edge_h = self.edge_type_embedding(edge_types.long())
        for layer in self.message_layers:
            node_h = F.relu(layer(node_h, edge_index, edge_h))
        return node_h

    def _degree_embedding(
        self,
        degree_sequence: Sequence[int],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        histogram = torch.zeros(
            self.degree_histogram_dim,
            dtype=torch.float32,
            device=device,
        )
        for raw_degree in degree_sequence:
            degree = int(raw_degree)
            if degree < 0 or degree >= self.degree_histogram_dim:
                raise ValueError(
                    f"Degree {degree} lies outside [0,{self.degree_histogram_dim - 1}]."
                )
            histogram[degree] += 1.0
        histogram /= histogram.sum().clamp_min(1.0)
        return self.degree_encoder(histogram.view(1, -1)).squeeze(0)

    def _time_embedding(
        self,
        t: int | float | torch.Tensor,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        value = torch.tensor(
            [self._normalized_time(t)],
            dtype=torch.float32,
            device=device,
        )
        return self.time_encoder(_sinusoidal_embedding(value, self.time_embedding_dim)).squeeze(0)

    def score_actions(
        self,
        *,
        node_types: torch.Tensor,
        degree_features: torch.Tensor,
        pe: torch.Tensor,
        edge_index: torch.Tensor,
        edge_types: torch.Tensor,
        actions: Sequence[MolecularRewireAction],
        t: int | float | torch.Tensor,
        degree_sequence: Sequence[int],
        action_local_features: torch.Tensor,
    ) -> torch.Tensor:
        if not actions:
            return degree_features.new_empty((0,))
        device = node_types.device
        node_h = self.encode_nodes(
            node_types,
            degree_features,
            pe,
            edge_index,
            edge_types,
        )
        graph_h = self.graph_encoder(node_h.mean(dim=0, keepdim=True)).squeeze(0)
        degree_h = self._degree_embedding(degree_sequence, device=device)
        time_h = self._time_embedding(t, device=device)

        endpoints = torch.tensor(
            [
                [
                    int(action.e1[0]),
                    int(action.e1[1]),
                    int(action.e2[0]),
                    int(action.e2[1]),
                ]
                for action in actions
            ],
            dtype=torch.long,
            device=device,
        )
        if int(endpoints.min().item()) < 0 or int(endpoints.max().item()) >= node_h.size(0):
            raise IndexError("Molecular rewiring action references a node outside the encoded graph.")
        first_removed = node_h[endpoints[:, 0]] + node_h[endpoints[:, 1]]
        second_removed = node_h[endpoints[:, 2]] + node_h[endpoints[:, 3]]
        endpoint_sum = first_removed + second_removed
        endpoint_absdiff = torch.abs(first_removed - second_removed)
        endpoint_product = first_removed * second_removed

        orientations = torch.tensor(
            [int(action.orientation) for action in actions],
            dtype=torch.long,
            device=device,
        )
        orientation_h = self.orientation_embedding(orientations)

        raw_new_types = [
            [int(action.new_edge_types[0]), int(action.new_edge_types[1])]
            for action in actions
        ]
        try:
            new_type_indices = torch.tensor(
                [
                    [
                        self.edge_type_to_index[first],
                        self.edge_type_to_index[second],
                    ]
                    for first, second in raw_new_types
                ],
                dtype=torch.long,
                device=device,
            )
        except KeyError as exc:
            raise KeyError(f"Action contains an edge type outside the fitted vocabulary: {exc}") from exc
        first_new = self.edge_type_embedding(new_type_indices[:, 0])
        second_new = self.edge_type_embedding(new_type_indices[:, 1])
        new_bond_sum = first_new + second_new
        new_bond_absdiff = torch.abs(first_new - second_new)
        new_bond_product = first_new * second_new

        local = action_local_features.to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        if local.ndim != 2 or local.size(0) != len(actions):
            raise ValueError(
                f"action_local_features must have {len(actions)} rows; got {tuple(local.shape)}"
            )
        if local.size(1) < self.local_feature_dim:
            local = torch.cat(
                [local, local.new_zeros((local.size(0), self.local_feature_dim - local.size(1)))],
                dim=1,
            )
        elif local.size(1) > self.local_feature_dim:
            local = local[:, : self.local_feature_dim]
        local_h = self.local_encoder(local)

        count = len(actions)
        features = torch.cat(
            [
                graph_h.view(1, -1).expand(count, -1),
                degree_h.view(1, -1).expand(count, -1),
                time_h.view(1, -1).expand(count, -1),
                endpoint_sum,
                endpoint_absdiff,
                endpoint_product,
                orientation_h,
                local_h,
                new_bond_sum,
                new_bond_absdiff,
                new_bond_product,
            ],
            dim=-1,
        )
        return self.action_scorer(features).squeeze(-1)

    def forward(self, **kwargs: object) -> torch.Tensor:
        return self.score_actions(**kwargs)  # type: ignore[arg-type]

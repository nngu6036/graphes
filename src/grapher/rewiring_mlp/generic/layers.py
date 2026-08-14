from __future__ import annotations

import torch
from torch import nn


class TopologyMPNNLayer(nn.Module):
    """Permutation-equivariant message passing over binary topology states."""

    def __init__(self, hidden_dim: int, edge_dim: int) -> None:
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_update = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_update = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, edge_dim),
            nn.SiLU(),
            nn.Linear(edge_dim, edge_dim),
        )
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.edge_norm = nn.LayerNorm(edge_dim)

    def forward(
        self,
        node_state: torch.Tensor,
        edge_state: torch.Tensor,
        adjacency: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, node_count, hidden_dim = node_state.shape
        left = node_state.unsqueeze(2).expand(
            batch_size,
            node_count,
            node_count,
            hidden_dim,
        )
        right = node_state.unsqueeze(1).expand(
            batch_size,
            node_count,
            node_count,
            hidden_dim,
        )
        pair_state = torch.cat([left, right, edge_state], dim=-1)
        messages = self.message(pair_state) * adjacency.unsqueeze(-1).float()
        aggregated = messages.sum(dim=2) / adjacency.sum(
            dim=2,
            keepdim=True,
        ).clamp_min(1).float()
        node_state = self.node_norm(
            node_state
            + self.node_update(torch.cat([node_state, aggregated], dim=-1))
        )
        edge_state = self.edge_norm(edge_state + self.edge_update(pair_state))
        edge_state = 0.5 * (edge_state + edge_state.transpose(1, 2))
        node_state = node_state * node_mask.unsqueeze(-1).float()
        pair_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        edge_state = edge_state * pair_mask.unsqueeze(-1).float()
        return node_state, edge_state

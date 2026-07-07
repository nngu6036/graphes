from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import torch
from torch import nn

from grapher.molecular.attribute_flow import EdgeAwareMPNNLayer, MolecularBatch, collate_molecular_graphs
from grapher.molecular.constants import QM9_ATOM_TYPES, QM9_BOND_TYPES, index_to_atom, index_to_bond_type
from grapher.utils.device import resolve_torch_device
from grapher.utils.io import ensure_dir, save_json


@dataclass
class MixtureCatFlowBatchState:
    node_state: torch.Tensor  # [B,N,Kx] continuous interpolant
    edge_state: torch.Tensor  # [B,N,N,Ke] continuous interpolant, existing topology edges only
    time: torch.Tensor  # [B]


class TopologyConditionalMixtureCatFlow(nn.Module):
    """Topology-conditioned variational flow matching for molecular attributes.

    The topology is fixed. The model generates only node atom types and bond
    types on existing topology edges. This is a CatFlow-style endpoint model:
    for each node/edge variable it predicts q_theta(x_1 | x_t, A). Unlike the
    basic conditional attribute denoiser, this model allows a small mixture of
    categorical endpoint distributions for each variable.

    Notes on the mixture:
      * Mixture weights and component logits are predicted from the full graph
        context through message passing, so dependencies are captured through
        shared context.
      * The implemented mixture is per-variable. A true joint mixture over all
        node/edge variables would require a global discrete latent and is left as
        a future extension.
    """

    def __init__(
        self,
        *,
        num_atom_types: int = len(QM9_ATOM_TYPES),
        num_bond_types: int = len(QM9_BOND_TYPES),
        num_mixtures: int = 4,
        hidden_dim: int = 128,
        edge_dim: int = 64,
        num_layers: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_atom_types = int(num_atom_types)
        self.num_bond_types = int(num_bond_types)
        self.num_mixtures = int(num_mixtures)
        self.hidden_dim = int(hidden_dim)
        self.edge_dim = int(edge_dim)

        # Node input: continuous atom-state vector + topological degree + time + mask.
        self.node_in = nn.Linear(self.num_atom_types + 3, hidden_dim)
        # Edge input: continuous bond-state vector + adjacency flag + time.
        self.edge_in = nn.Linear(self.num_bond_types + 2, edge_dim)

        self.layers = nn.ModuleList([
            EdgeAwareMPNNLayer(hidden_dim, edge_dim) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

        self.node_mix_logits = nn.Linear(hidden_dim, self.num_mixtures)
        self.node_component_logits = nn.Linear(hidden_dim, self.num_mixtures * self.num_atom_types)

        self.edge_mix_logits = nn.Linear(edge_dim, self.num_mixtures)
        self.edge_component_logits = nn.Linear(edge_dim, self.num_mixtures * self.num_bond_types)

    @staticmethod
    def _upper_edge_mask(edge_mask: torch.Tensor) -> torch.Tensor:
        B, N, _ = edge_mask.shape
        upper = torch.triu(torch.ones((N, N), dtype=torch.bool, device=edge_mask.device), diagonal=1)
        return edge_mask.bool() & upper.view(1, N, N)

    def clean_onehot(self, batch: MolecularBatch, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        node_labels = batch.node_labels.to(device).clamp(min=0)
        edge_labels = batch.edge_labels_dense.to(device)
        node_onehot = torch.nn.functional.one_hot(node_labels, self.num_atom_types).float()
        # Existing bond dense labels are 1..Ke. Convert to 0..Ke-1.
        edge_target = (edge_labels - 1).clamp(min=0)
        edge_onehot = torch.nn.functional.one_hot(edge_target, self.num_bond_types).float()
        node_onehot = node_onehot * batch.node_mask.to(device).unsqueeze(-1).float()
        edge_onehot = edge_onehot * batch.edge_mask.to(device).unsqueeze(-1).float()
        return node_onehot, edge_onehot

    def sample_interpolant(
        self,
        batch: MolecularBatch,
        *,
        device: torch.device,
        rng: torch.Generator | None = None,
        noise_scale: float = 1.0,
    ) -> tuple[MixtureCatFlowBatchState, torch.Tensor, torch.Tensor]:
        """Create x_t = t x_1 + (1-t) x_0 with Gaussian x_0."""
        adjacency = batch.adjacency.to(device).bool()
        node_mask = batch.node_mask.to(device).bool()
        edge_mask = batch.edge_mask.to(device).bool()
        node_clean, edge_clean = self.clean_onehot(batch, device)
        B, N, _ = node_clean.shape
        t = torch.rand(B, device=device, generator=rng).clamp(1.0e-4, 1.0 - 1.0e-4)
        node_noise = noise_scale * torch.randn(node_clean.shape, device=device, generator=rng)
        edge_noise = noise_scale * torch.randn(edge_clean.shape, device=device, generator=rng)
        node_state = t.view(B, 1, 1) * node_clean + (1.0 - t).view(B, 1, 1) * node_noise
        edge_state = t.view(B, 1, 1, 1) * edge_clean + (1.0 - t).view(B, 1, 1, 1) * edge_noise
        node_state = node_state * node_mask.unsqueeze(-1).float()
        edge_state = edge_state * edge_mask.unsqueeze(-1).float()
        edge_state = 0.5 * (edge_state + edge_state.transpose(1, 2))
        return MixtureCatFlowBatchState(node_state=node_state, edge_state=edge_state, time=t), node_clean, edge_clean

    def forward(
        self,
        adjacency: torch.Tensor,
        node_state: torch.Tensor,
        edge_state: torch.Tensor,
        node_mask: torch.Tensor,
        t: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        adjacency = adjacency.bool()
        node_mask = node_mask.bool()
        B, N, _ = node_state.shape
        degrees = adjacency.sum(dim=-1, keepdim=True).float() / max(N - 1, 1)
        t_node = t.view(B, 1, 1).expand(B, N, 1)
        node_features = torch.cat([node_state, degrees, t_node, node_mask.unsqueeze(-1).float()], dim=-1)
        h = self.node_in(node_features) * node_mask.unsqueeze(-1).float()

        t_edge = t.view(B, 1, 1, 1).expand(B, N, N, 1)
        edge_features = torch.cat([edge_state, adjacency.unsqueeze(-1).float(), t_edge], dim=-1)
        e = self.edge_in(edge_features)
        e = 0.5 * (e + e.transpose(1, 2))

        for layer in self.layers:
            h, e = layer(h, e, adjacency, node_mask)
            h = self.dropout(h)
            e = self.dropout(e)

        node_mix_logits = self.node_mix_logits(h)
        node_component_logits = self.node_component_logits(h).view(B, N, self.num_mixtures, self.num_atom_types)

        edge_mix_logits = self.edge_mix_logits(e)
        edge_component_logits = self.edge_component_logits(e).view(B, N, N, self.num_mixtures, self.num_bond_types)
        edge_mix_logits = 0.5 * (edge_mix_logits + edge_mix_logits.transpose(1, 2))
        edge_component_logits = 0.5 * (edge_component_logits + edge_component_logits.transpose(1, 2))

        return {
            "node_mix_logits": node_mix_logits,
            "node_component_logits": node_component_logits,
            "edge_mix_logits": edge_mix_logits,
            "edge_component_logits": edge_component_logits,
        }

    @staticmethod
    def _mixture_nll(mix_logits: torch.Tensor, component_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Negative log p(target) under a mixture of categoricals.

        mix_logits: [..., M]
        component_logits: [..., M, K]
        target: [...]
        """
        log_mix = torch.log_softmax(mix_logits, dim=-1)
        log_comp = torch.log_softmax(component_logits, dim=-1)
        gather_index = target.unsqueeze(-1).unsqueeze(-1).expand(*target.shape, component_logits.shape[-2], 1)
        selected = torch.gather(log_comp, dim=-1, index=gather_index).squeeze(-1)
        log_prob = torch.logsumexp(log_mix + selected, dim=-1)
        return -log_prob

    def endpoint_mean(self, params: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        node_w = torch.softmax(params["node_mix_logits"], dim=-1)
        node_p = torch.softmax(params["node_component_logits"], dim=-1)
        node_mean = (node_w.unsqueeze(-1) * node_p).sum(dim=-2)

        edge_w = torch.softmax(params["edge_mix_logits"], dim=-1)
        edge_p = torch.softmax(params["edge_component_logits"], dim=-1)
        edge_mean = (edge_w.unsqueeze(-1) * edge_p).sum(dim=-2)
        edge_mean = 0.5 * (edge_mean + edge_mean.transpose(1, 2))
        return node_mean, edge_mean

    def loss(self, batch: MolecularBatch, *, device: torch.device, noise_scale: float = 1.0) -> tuple[torch.Tensor, dict[str, float]]:
        batch = MolecularBatch(
            adjacency=batch.adjacency.to(device),
            node_labels=batch.node_labels.to(device),
            edge_labels_dense=batch.edge_labels_dense.to(device),
            node_mask=batch.node_mask.to(device),
            edge_mask=batch.edge_mask.to(device),
        )
        state, _, _ = self.sample_interpolant(batch, device=device, noise_scale=noise_scale)
        params = self.forward(batch.adjacency, state.node_state, state.edge_state, batch.node_mask, state.time)

        node_target = batch.node_labels.clamp(min=0)
        node_nll = self._mixture_nll(params["node_mix_logits"], params["node_component_logits"], node_target)
        node_loss = node_nll[batch.node_mask.bool()].mean()

        edge_target = (batch.edge_labels_dense - 1).clamp(min=0)
        upper_edge_mask = self._upper_edge_mask(batch.edge_mask)
        edge_nll = self._mixture_nll(params["edge_mix_logits"], params["edge_component_logits"], edge_target)
        if upper_edge_mask.any():
            edge_loss = edge_nll[upper_edge_mask].mean()
        else:
            edge_loss = edge_nll.sum() * 0.0

        loss = node_loss + edge_loss
        return loss, {
            "loss": float(loss.detach().cpu()),
            "node_loss": float(node_loss.detach().cpu()),
            "edge_loss": float(edge_loss.detach().cpu()),
        }

    def sample_attributes(
        self,
        topology: nx.Graph,
        *,
        steps: int = 64,
        temperature: float = 1.0,
        device: torch.device | str = "cpu",
        seed: int = 0,
        sample_categorical: bool = False,
    ) -> nx.Graph:
        device = resolve_torch_device(device)
        self.to(device)
        self.eval()
        rng = torch.Generator(device=device)
        rng.manual_seed(int(seed))

        topo = nx.convert_node_labels_to_integers(nx.Graph(topology), ordering="sorted")
        n = topo.number_of_nodes()
        adjacency = torch.zeros((1, n, n), dtype=torch.bool, device=device)
        for u, v in topo.edges():
            adjacency[0, int(u), int(v)] = True
            adjacency[0, int(v), int(u)] = True
        node_mask = torch.ones((1, n), dtype=torch.bool, device=device)
        edge_mask = adjacency

        node_state = torch.randn((1, n, self.num_atom_types), device=device, generator=rng)
        edge_state = torch.randn((1, n, n, self.num_bond_types), device=device, generator=rng)
        edge_state = edge_state * edge_mask.unsqueeze(-1).float()
        edge_state = 0.5 * (edge_state + edge_state.transpose(1, 2))

        dt = 1.0 / max(int(steps), 1)
        with torch.no_grad():
            for step in range(int(steps)):
                t_value = min(step * dt, 1.0 - 1.0e-4)
                t = torch.full((1,), t_value, device=device)
                params = self.forward(adjacency, node_state, edge_state, node_mask, t)
                node_mean, edge_mean = self.endpoint_mean(params)
                denom = max(1.0 - t_value, 1.0e-4)
                node_state = node_state + dt * (node_mean - node_state) / denom
                edge_state = edge_state + dt * (edge_mean - edge_state) / denom
                node_state = node_state * node_mask.unsqueeze(-1).float()
                edge_state = edge_state * edge_mask.unsqueeze(-1).float()
                edge_state = 0.5 * (edge_state + edge_state.transpose(1, 2))

            t = torch.full((1,), 1.0 - 1.0e-4, device=device)
            params = self.forward(adjacency, node_state, edge_state, node_mask, t)
            node_mean, edge_mean = self.endpoint_mean(params)

            if temperature != 1.0:
                node_logits = torch.log(node_mean.clamp_min(1.0e-12)) / max(float(temperature), 1.0e-8)
                edge_logits = torch.log(edge_mean.clamp_min(1.0e-12)) / max(float(temperature), 1.0e-8)
                node_probs = torch.softmax(node_logits, dim=-1)
                edge_probs = torch.softmax(edge_logits, dim=-1)
            else:
                node_probs, edge_probs = node_mean, edge_mean

            if sample_categorical:
                node_labels = torch.multinomial(node_probs.view(-1, self.num_atom_types), 1, generator=rng).view(1, n)
                edge_labels = torch.multinomial(edge_probs.view(-1, self.num_bond_types), 1, generator=rng).view(1, n, n)
            else:
                node_labels = torch.argmax(node_probs, dim=-1)
                edge_labels = torch.argmax(edge_probs, dim=-1)

        out = nx.Graph()
        for i in range(n):
            atom_idx = int(node_labels[0, i].detach().cpu())
            atomic_num = index_to_atom(atom_idx)
            out.add_node(i, atomic_num=atomic_num, atom_type=atomic_num)
        for u, v in topo.edges():
            bond_idx = int(edge_labels[0, int(u), int(v)].detach().cpu()) + 1
            bond_type = index_to_bond_type(bond_idx)
            out.add_edge(int(u), int(v), bond_type=bond_type, bond_order=float(bond_type if bond_type != 4 else 1.5))
        return out


def save_mixture_catflow_checkpoint(
    model: TopologyConditionalMixtureCatFlow,
    path: str | Path,
    *,
    config: dict[str, Any] | None = None,
    report: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "num_atom_types": model.num_atom_types,
                "num_bond_types": model.num_bond_types,
                "num_mixtures": model.num_mixtures,
                "hidden_dim": model.hidden_dim,
                "edge_dim": model.edge_dim,
                "num_layers": len(model.layers),
            },
            "config": config or {},
            "report": report or {},
        },
        path,
    )


def load_mixture_catflow_checkpoint(path: str | Path, *, device: str | torch.device = "cpu") -> TopologyConditionalMixtureCatFlow:
    device = resolve_torch_device(device)
    checkpoint = torch.load(path, map_location=device)
    cfg = checkpoint.get("model_config", {})
    model = TopologyConditionalMixtureCatFlow(
        num_atom_types=int(cfg.get("num_atom_types", len(QM9_ATOM_TYPES))),
        num_bond_types=int(cfg.get("num_bond_types", len(QM9_BOND_TYPES))),
        num_mixtures=int(cfg.get("num_mixtures", 4)),
        hidden_dim=int(cfg.get("hidden_dim", 128)),
        edge_dim=int(cfg.get("edge_dim", 64)),
        num_layers=int(cfg.get("num_layers", 4)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

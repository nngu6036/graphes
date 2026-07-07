from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from grapher.molecular.attribute_flow import EdgeAwareMPNNLayer, graph_to_arrays
from grapher.molecular.constants import (
    QM9_ATOM_TYPES,
    QM9_BOND_TYPES,
    index_to_atom,
    index_to_bond_type,
)
from grapher.utils.device import resolve_torch_device
from grapher.utils.io import ensure_dir


@dataclass
class DenseMolecularBatch:
    node_labels: torch.Tensor          # [B,N] atom indices; -1 padded
    edge_labels_dense: torch.Tensor    # [B,N,N] edge category; 0 no-edge, 1..K bonds, -1 padded
    node_mask: torch.Tensor            # [B,N]
    pair_mask: torch.Tensor            # [B,N,N] real off-diagonal pairs


class DenseMolecularGraphDataset(Dataset):
    """Dataset for full dense molecular graph CatFlow.

    Unlike the topology-conditioned Stage-2 model, this dataset trains over all
    node variables and all pairwise edge variables, including the no-edge
    category. This is the CatFlow-style baseline that learns topology and
    attributes jointly.
    """

    def __init__(self, graphs: list[nx.Graph]):
        self.graphs = graphs

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> nx.Graph:
        return self.graphs[idx]


def collate_dense_molecular_graphs(graphs: list[nx.Graph]) -> DenseMolecularBatch:
    if not graphs:
        raise ValueError("Empty molecular graph batch.")

    arrays = [graph_to_arrays(g) for g in graphs]
    max_n = max(a.shape[0] for a, _, _ in arrays)
    B = len(graphs)

    node_labels = np.full((B, max_n), -1, dtype=np.int64)
    edge_labels = np.full((B, max_n, max_n), -1, dtype=np.int64)
    node_mask = np.zeros((B, max_n), dtype=np.bool_)
    pair_mask = np.zeros((B, max_n, max_n), dtype=np.bool_)

    for i, (_A, x, e) in enumerate(arrays):
        n = x.shape[0]
        node_labels[i, :n] = x
        edge_labels[i, :n, :n] = e  # includes no-edge=0 for non-bonds
        node_mask[i, :n] = True
        pair_mask[i, :n, :n] = True
        np.fill_diagonal(pair_mask[i], False)

    return DenseMolecularBatch(
        node_labels=torch.from_numpy(node_labels),
        edge_labels_dense=torch.from_numpy(edge_labels),
        node_mask=torch.from_numpy(node_mask),
        pair_mask=torch.from_numpy(pair_mask),
    )


@dataclass
class DenseMixtureCatFlowState:
    node_state: torch.Tensor  # [B,N,Kx]
    edge_state: torch.Tensor  # [B,N,N,Ke+1], includes no-edge category
    time: torch.Tensor        # [B]


class DenseMolecularMixtureCatFlow(nn.Module):
    """Joint molecular CatFlow baseline over topology and attributes.

    This model learns a flow from a random Gaussian graph state x_0 to a clean
    categorical molecular graph x_1. The node variables are atom types. The edge
    variables are dense pairwise categories: no-edge plus bond types. It is a
    mixture-categorical endpoint model:

        q_theta(x_1^d | x_t) = sum_m pi_m^d(x_t) Cat(mu_m^d(x_t)).

    This is the baseline for comparing against the topology-first GraphER
    molecular generator.
    """

    def __init__(
        self,
        *,
        num_atom_types: int = len(QM9_ATOM_TYPES),
        num_edge_categories: int = len(QM9_BOND_TYPES) + 1,
        num_mixtures: int = 4,
        hidden_dim: int = 128,
        edge_dim: int = 64,
        num_layers: int = 4,
        dropout: float = 0.0,
        node_count_probs: list[float] | None = None,
    ):
        super().__init__()
        self.num_atom_types = int(num_atom_types)
        self.num_edge_categories = int(num_edge_categories)
        self.num_mixtures = int(num_mixtures)
        self.hidden_dim = int(hidden_dim)
        self.edge_dim = int(edge_dim)
        self.dropout_p = float(dropout)

        self.node_in = nn.Linear(self.num_atom_types + 3, hidden_dim)
        self.edge_in = nn.Linear(self.num_edge_categories + 2, edge_dim)
        self.layers = nn.ModuleList([EdgeAwareMPNNLayer(hidden_dim, edge_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

        self.node_mix_logits = nn.Linear(hidden_dim, self.num_mixtures)
        self.node_component_logits = nn.Linear(hidden_dim, self.num_mixtures * self.num_atom_types)
        self.edge_mix_logits = nn.Linear(edge_dim, self.num_mixtures)
        self.edge_component_logits = nn.Linear(edge_dim, self.num_mixtures * self.num_edge_categories)

        if node_count_probs is None:
            self.register_buffer("node_count_probs", torch.ones(10) / 10.0)
        else:
            probs = torch.tensor(node_count_probs, dtype=torch.float32)
            probs = probs / probs.sum().clamp_min(1e-12)
            self.register_buffer("node_count_probs", probs)

    @staticmethod
    def _upper_pair_mask(pair_mask: torch.Tensor) -> torch.Tensor:
        B, N, _ = pair_mask.shape
        upper = torch.triu(torch.ones((N, N), dtype=torch.bool, device=pair_mask.device), diagonal=1)
        return pair_mask.bool() & upper.view(1, N, N)

    def clean_onehot(self, batch: DenseMolecularBatch, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        node_labels = batch.node_labels.to(device).clamp(min=0)
        edge_labels = batch.edge_labels_dense.to(device).clamp(min=0)
        node_onehot = torch.nn.functional.one_hot(node_labels, self.num_atom_types).float()
        edge_onehot = torch.nn.functional.one_hot(edge_labels, self.num_edge_categories).float()
        node_onehot = node_onehot * batch.node_mask.to(device).unsqueeze(-1).float()
        edge_onehot = edge_onehot * batch.pair_mask.to(device).unsqueeze(-1).float()
        edge_onehot = 0.5 * (edge_onehot + edge_onehot.transpose(1, 2))
        return node_onehot, edge_onehot

    def sample_interpolant(
        self,
        batch: DenseMolecularBatch,
        *,
        device: torch.device,
        rng: torch.Generator | None = None,
        noise_scale: float = 1.0,
    ) -> tuple[DenseMixtureCatFlowState, torch.Tensor, torch.Tensor]:
        node_clean, edge_clean = self.clean_onehot(batch, device)
        node_mask = batch.node_mask.to(device).bool()
        pair_mask = batch.pair_mask.to(device).bool()
        B, N, _ = node_clean.shape
        t = torch.rand(B, device=device, generator=rng).clamp(1.0e-4, 1.0 - 1.0e-4)
        node_noise = noise_scale * torch.randn(node_clean.shape, device=device, generator=rng)
        edge_noise = noise_scale * torch.randn(edge_clean.shape, device=device, generator=rng)
        node_state = t.view(B, 1, 1) * node_clean + (1.0 - t).view(B, 1, 1) * node_noise
        edge_state = t.view(B, 1, 1, 1) * edge_clean + (1.0 - t).view(B, 1, 1, 1) * edge_noise
        node_state = node_state * node_mask.unsqueeze(-1).float()
        edge_state = edge_state * pair_mask.unsqueeze(-1).float()
        edge_state = 0.5 * (edge_state + edge_state.transpose(1, 2))
        return DenseMixtureCatFlowState(node_state=node_state, edge_state=edge_state, time=t), node_clean, edge_clean

    def forward(
        self,
        node_state: torch.Tensor,
        edge_state: torch.Tensor,
        node_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        t: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        node_mask = node_mask.bool()
        pair_mask = pair_mask.bool()
        B, N, _ = node_state.shape
        # Use all real off-diagonal pairs for message passing. This follows the
        # fully-connected CatFlow graph representation, where no-edge is an edge category.
        adjacency_proxy = pair_mask
        degrees_proxy = pair_mask.sum(dim=-1, keepdim=True).float() / max(N - 1, 1)
        t_node = t.view(B, 1, 1).expand(B, N, 1)
        node_features = torch.cat([node_state, degrees_proxy, t_node, node_mask.unsqueeze(-1).float()], dim=-1)
        h = self.node_in(node_features) * node_mask.unsqueeze(-1).float()

        t_edge = t.view(B, 1, 1, 1).expand(B, N, N, 1)
        edge_features = torch.cat([edge_state, pair_mask.unsqueeze(-1).float(), t_edge], dim=-1)
        e = self.edge_in(edge_features)
        e = 0.5 * (e + e.transpose(1, 2))

        for layer in self.layers:
            h, e = layer(h, e, adjacency_proxy, node_mask)
            h = self.dropout(h)
            e = self.dropout(e)

        node_mix_logits = self.node_mix_logits(h)
        node_component_logits = self.node_component_logits(h).view(B, N, self.num_mixtures, self.num_atom_types)
        edge_mix_logits = self.edge_mix_logits(e)
        edge_component_logits = self.edge_component_logits(e).view(B, N, N, self.num_mixtures, self.num_edge_categories)
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

    def loss(self, batch: DenseMolecularBatch, *, device: torch.device, noise_scale: float = 1.0) -> tuple[torch.Tensor, dict[str, float]]:
        batch = DenseMolecularBatch(
            node_labels=batch.node_labels.to(device),
            edge_labels_dense=batch.edge_labels_dense.to(device),
            node_mask=batch.node_mask.to(device),
            pair_mask=batch.pair_mask.to(device),
        )
        state, _, _ = self.sample_interpolant(batch, device=device, noise_scale=noise_scale)
        params = self.forward(state.node_state, state.edge_state, batch.node_mask, batch.pair_mask, state.time)

        node_target = batch.node_labels.clamp(min=0)
        node_nll = self._mixture_nll(params["node_mix_logits"], params["node_component_logits"], node_target)
        node_loss = node_nll[batch.node_mask.bool()].mean()

        edge_target = batch.edge_labels_dense.clamp(min=0)
        upper = self._upper_pair_mask(batch.pair_mask)
        edge_nll = self._mixture_nll(params["edge_mix_logits"], params["edge_component_logits"], edge_target)
        edge_loss = edge_nll[upper].mean() if upper.any() else edge_nll.sum() * 0.0
        loss = node_loss + edge_loss
        return loss, {
            "loss": float(loss.detach().cpu()),
            "node_loss": float(node_loss.detach().cpu()),
            "edge_loss": float(edge_loss.detach().cpu()),
        }

    def sample_num_nodes(self, rng: torch.Generator, device: torch.device) -> int:
        probs = self.node_count_probs.to(device)
        idx = torch.multinomial(probs, 1, generator=rng).item()
        # index 0 is allowed in the vector but QM9 no-H molecules have at least 1 heavy atom.
        return max(int(idx), 1)

    @torch.no_grad()
    def sample_graph(
        self,
        *,
        num_nodes: int | None = None,
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
        n = int(num_nodes) if num_nodes is not None else self.sample_num_nodes(rng, device)
        n = max(n, 1)

        node_mask = torch.ones((1, n), dtype=torch.bool, device=device)
        pair_mask = torch.ones((1, n, n), dtype=torch.bool, device=device)
        eye = torch.eye(n, dtype=torch.bool, device=device).view(1, n, n)
        pair_mask = pair_mask & ~eye
        node_state = torch.randn((1, n, self.num_atom_types), device=device, generator=rng)
        edge_state = torch.randn((1, n, n, self.num_edge_categories), device=device, generator=rng)
        edge_state = edge_state * pair_mask.unsqueeze(-1).float()
        edge_state = 0.5 * (edge_state + edge_state.transpose(1, 2))

        dt = 1.0 / max(int(steps), 1)
        eps = 1e-4
        for step in range(int(steps)):
            t_value = min(step * dt, 1.0 - eps)
            t = torch.full((1,), t_value, device=device)
            params = self.forward(node_state, edge_state, node_mask, pair_mask, t)
            node_mean, edge_mean = self.endpoint_mean(params)
            denom = max(1.0 - t_value, eps)
            node_state = node_state + dt * (node_mean - node_state) / denom
            edge_state = edge_state + dt * (edge_mean - edge_state) / denom
            node_state = node_state * node_mask.unsqueeze(-1).float()
            edge_state = edge_state * pair_mask.unsqueeze(-1).float()
            edge_state = 0.5 * (edge_state + edge_state.transpose(1, 2))

        t = torch.full((1,), 1.0 - eps, device=device)
        params = self.forward(node_state, edge_state, node_mask, pair_mask, t)
        node_mean, edge_mean = self.endpoint_mean(params)
        if temperature != 1.0:
            node_probs = torch.softmax(torch.log(node_mean.clamp_min(1e-12)) / max(float(temperature), 1e-8), dim=-1)
            edge_probs = torch.softmax(torch.log(edge_mean.clamp_min(1e-12)) / max(float(temperature), 1e-8), dim=-1)
        else:
            node_probs, edge_probs = node_mean, edge_mean

        if sample_categorical:
            node_labels = torch.multinomial(node_probs.view(-1, self.num_atom_types), 1, generator=rng).view(1, n)
            edge_labels = torch.multinomial(edge_probs.view(-1, self.num_edge_categories), 1, generator=rng).view(1, n, n)
        else:
            node_labels = torch.argmax(node_probs, dim=-1)
            edge_labels = torch.argmax(edge_probs, dim=-1)

        out = nx.Graph()
        for i in range(n):
            atom_idx = int(node_labels[0, i].detach().cpu())
            atomic_num = index_to_atom(atom_idx)
            out.add_node(i, atomic_num=atomic_num, atom_type=atomic_num)

        for u in range(n):
            for v in range(u + 1, n):
                edge_idx = int(edge_labels[0, u, v].detach().cpu())
                if edge_idx <= 0:
                    continue
                bond_type = index_to_bond_type(edge_idx)
                out.add_edge(u, v, bond_type=bond_type, bond_order=float(bond_type if bond_type != 4 else 1.5))
        return out


def node_count_distribution(graphs: list[nx.Graph], max_nodes: int | None = None) -> list[float]:
    max_n = int(max_nodes or max(g.number_of_nodes() for g in graphs))
    counts = np.zeros(max_n + 1, dtype=np.float64)
    for g in graphs:
        n = int(g.number_of_nodes())
        if n <= max_n:
            counts[n] += 1.0
    if counts.sum() <= 0:
        counts[1:] = 1.0
    counts = counts / counts.sum()
    return counts.tolist()


def save_dense_mixture_catflow_checkpoint(
    model: DenseMolecularMixtureCatFlow,
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
                "num_edge_categories": model.num_edge_categories,
                "num_mixtures": model.num_mixtures,
                "hidden_dim": model.hidden_dim,
                "edge_dim": model.edge_dim,
                "num_layers": len(model.layers),
                "dropout": model.dropout_p,
                "node_count_probs": model.node_count_probs.detach().cpu().tolist(),
            },
            "config": config or {},
            "report": report or {},
        },
        path,
    )


def load_dense_mixture_catflow_checkpoint(path: str | Path, *, device: str | torch.device = "cpu") -> DenseMolecularMixtureCatFlow:
    device = resolve_torch_device(device)
    checkpoint = torch.load(path, map_location=device)
    cfg = checkpoint.get("model_config", {})
    model = DenseMolecularMixtureCatFlow(
        num_atom_types=int(cfg.get("num_atom_types", len(QM9_ATOM_TYPES))),
        num_edge_categories=int(cfg.get("num_edge_categories", len(QM9_BOND_TYPES) + 1)),
        num_mixtures=int(cfg.get("num_mixtures", 4)),
        hidden_dim=int(cfg.get("hidden_dim", 128)),
        edge_dim=int(cfg.get("edge_dim", 64)),
        num_layers=int(cfg.get("num_layers", 4)),
        dropout=float(cfg.get("dropout", 0.0)),
        node_count_probs=cfg.get("node_count_probs"),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

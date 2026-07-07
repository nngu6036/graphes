from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from grapher.molecular.constants import (
    EDGE_MASK_INDEX,
    NO_EDGE_INDEX,
    QM9_ATOM_TYPES,
    QM9_BOND_TYPES,
    atom_to_index,
    bond_type_to_index,
    index_to_atom,
    index_to_bond_type,
)
from grapher.utils.device import resolve_torch_device


@dataclass
class MolecularBatch:
    adjacency: torch.Tensor  # [B,N,N] bool
    node_labels: torch.Tensor  # [B,N] long, atom indices, -1 padded
    edge_labels_dense: torch.Tensor  # [B,N,N] long, dense edge categories; no-edge=0, bonds start at 1, -1 padded
    node_mask: torch.Tensor  # [B,N] bool
    edge_mask: torch.Tensor  # [B,N,N] bool existing edges on real nodes


class MolecularGraphDataset(Dataset):
    def __init__(self, graphs: list[nx.Graph], atom_types: tuple[int, ...] = QM9_ATOM_TYPES):
        self.graphs = graphs
        self.atom_types = atom_types

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> nx.Graph:
        return self.graphs[idx]


def graph_to_arrays(
    graph: nx.Graph,
    *,
    atom_types: tuple[int, ...] = QM9_ATOM_TYPES,
    bond_types: tuple[int, ...] = QM9_BOND_TYPES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g = nx.convert_node_labels_to_integers(nx.Graph(graph), ordering="sorted")
    n = g.number_of_nodes()
    adjacency = np.zeros((n, n), dtype=np.bool_)
    node_labels = np.full(n, -1, dtype=np.int64)
    edge_labels = np.zeros((n, n), dtype=np.int64)
    for node, data in g.nodes(data=True):
        atomic_num = int(data.get("atomic_num", data.get("atom_type", 6)))
        node_labels[int(node)] = atom_to_index(atomic_num, atom_types)
    for u, v, data in g.edges(data=True):
        u, v = int(u), int(v)
        b = int(data.get("bond_type", 1))
        dense_idx = bond_type_to_index(b, bond_types)
        adjacency[u, v] = adjacency[v, u] = True
        edge_labels[u, v] = edge_labels[v, u] = dense_idx
    return adjacency, node_labels, edge_labels


def collate_molecular_graphs(
    graphs: list[nx.Graph],
    *,
    atom_types: tuple[int, ...] = QM9_ATOM_TYPES,
    bond_types: tuple[int, ...] = QM9_BOND_TYPES,
) -> MolecularBatch:
    if not graphs:
        raise ValueError("Empty molecular graph batch.")
    arrays = [graph_to_arrays(g, atom_types=atom_types, bond_types=bond_types) for g in graphs]
    max_n = max(a.shape[0] for a, _, _ in arrays)
    B = len(graphs)
    adjacency = np.zeros((B, max_n, max_n), dtype=np.bool_)
    node_labels = np.full((B, max_n), -1, dtype=np.int64)
    edge_labels = np.full((B, max_n, max_n), -1, dtype=np.int64)
    node_mask = np.zeros((B, max_n), dtype=np.bool_)
    edge_mask = np.zeros((B, max_n, max_n), dtype=np.bool_)
    for i, (A, x, e) in enumerate(arrays):
        n = A.shape[0]
        adjacency[i, :n, :n] = A
        node_labels[i, :n] = x
        edge_labels[i, :n, :n] = e
        node_mask[i, :n] = True
        edge_mask[i, :n, :n] = A
    return MolecularBatch(
        adjacency=torch.from_numpy(adjacency),
        node_labels=torch.from_numpy(node_labels),
        edge_labels_dense=torch.from_numpy(edge_labels),
        node_mask=torch.from_numpy(node_mask),
        edge_mask=torch.from_numpy(edge_mask),
    )


class EdgeAwareMPNNLayer(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int):
        super().__init__()
        self.msg = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_update = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.edge_update = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, edge_dim),
            nn.SiLU(),
            nn.Linear(edge_dim, edge_dim),
        )
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.edge_norm = nn.LayerNorm(edge_dim)

    def forward(self, h: torch.Tensor, e: torch.Tensor, adjacency: torch.Tensor, node_mask: torch.Tensor):
        B, N, H = h.shape
        hi = h.unsqueeze(2).expand(B, N, N, H)
        hj = h.unsqueeze(1).expand(B, N, N, H)
        pair = torch.cat([hi, hj, e], dim=-1)
        msg = self.msg(pair) * adjacency.unsqueeze(-1).float()
        agg = msg.sum(dim=2) / adjacency.sum(dim=2, keepdim=True).clamp(min=1).float()
        h_new = self.node_norm(h + self.node_update(torch.cat([h, agg], dim=-1)))
        e_new = self.edge_norm(e + self.edge_update(pair))
        e_new = 0.5 * (e_new + e_new.transpose(1, 2))
        h_new = h_new * node_mask.unsqueeze(-1).float()
        e_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        e_new = e_new * e_mask.unsqueeze(-1).float()
        return h_new, e_new


class TopologyConditionalAttributeFlow(nn.Module):
    """Conditional discrete flow/masked denoiser for molecular attributes.

    The topology is fixed. The model predicts atom labels for nodes and bond
    labels for existing topology edges. Training samples an interpolation time t:
    each attribute is clean with probability t and masked with probability 1-t.
    The model predicts the clean attributes, which is a discrete flow-matching /
    masked-denoising objective on the attribute state space conditioned on topology.
    """

    def __init__(
        self,
        *,
        num_atom_types: int = len(QM9_ATOM_TYPES),
        num_bond_types: int = len(QM9_BOND_TYPES),
        hidden_dim: int = 128,
        edge_dim: int = 64,
        num_layers: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_atom_types = int(num_atom_types)
        self.num_bond_types = int(num_bond_types)
        self.node_mask_index = self.num_atom_types
        self.edge_mask_index = self.num_bond_types + 1  # 0 no-edge, 1..num_bond_types bonds, mask last
        self.hidden_dim = int(hidden_dim)
        self.edge_dim = int(edge_dim)
        self.node_in = nn.Linear(self.num_atom_types + 1 + 3, hidden_dim)
        self.edge_in = nn.Linear(self.num_bond_types + 2 + 2, edge_dim)
        self.layers = nn.ModuleList([EdgeAwareMPNNLayer(hidden_dim, edge_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.node_out = nn.Linear(hidden_dim, self.num_atom_types)
        self.edge_out = nn.Linear(edge_dim, self.num_bond_types)

    def corrupt_batch(self, batch: MolecularBatch, *, rng: torch.Generator | None = None, device: torch.device):
        node_labels = batch.node_labels.to(device)
        edge_labels = batch.edge_labels_dense.to(device)
        node_mask = batch.node_mask.to(device)
        edge_mask = batch.edge_mask.to(device)
        B, N = node_labels.shape
        t = torch.rand(B, 1, device=device, generator=rng).clamp(1.0e-4, 1.0 - 1.0e-4)
        node_keep = (torch.rand(B, N, device=device, generator=rng) < t) & node_mask
        x_t = torch.where(node_keep, node_labels, torch.full_like(node_labels, self.node_mask_index))
        x_t = torch.where(node_mask, x_t, torch.zeros_like(x_t))

        edge_keep = (torch.rand(B, N, N, device=device, generator=rng) < t.view(B, 1, 1)) & edge_mask
        e_t = torch.where(edge_keep, edge_labels, torch.full_like(edge_labels, self.edge_mask_index))
        e_t = torch.where(edge_mask, e_t, torch.zeros_like(e_t))  # no-edge outside topology
        e_t = torch.where(torch.eye(N, device=device).bool().view(1, N, N), torch.zeros_like(e_t), e_t)
        return x_t.long(), e_t.long(), t.squeeze(-1)

    def forward(self, adjacency: torch.Tensor, x_t: torch.Tensor, e_t: torch.Tensor, node_mask: torch.Tensor, t: torch.Tensor):
        adjacency = adjacency.bool()
        node_mask = node_mask.bool()
        B, N = x_t.shape
        x_oh = torch.nn.functional.one_hot(x_t.clamp(min=0, max=self.node_mask_index), self.num_atom_types + 1).float()
        degrees = adjacency.sum(dim=-1, keepdim=True).float() / max(N - 1, 1)
        t_node = t.view(B, 1, 1).expand(B, N, 1)
        node_features = torch.cat([x_oh, degrees, t_node, node_mask.unsqueeze(-1).float()], dim=-1)
        h = self.node_in(node_features) * node_mask.unsqueeze(-1).float()

        e_oh = torch.nn.functional.one_hot(e_t.clamp(min=0, max=self.edge_mask_index), self.num_bond_types + 2).float()
        t_edge = t.view(B, 1, 1, 1).expand(B, N, N, 1)
        edge_features = torch.cat([e_oh, adjacency.unsqueeze(-1).float(), t_edge], dim=-1)
        e = self.edge_in(edge_features)
        e = 0.5 * (e + e.transpose(1, 2))

        for layer in self.layers:
            h, e = layer(h, e, adjacency, node_mask)
            h = self.dropout(h)
            e = self.dropout(e)
        node_logits = self.node_out(h)
        edge_logits = self.edge_out(e)
        edge_logits = 0.5 * (edge_logits + edge_logits.transpose(1, 2))
        return node_logits, edge_logits

    def loss(self, batch: MolecularBatch, *, device: torch.device) -> tuple[torch.Tensor, dict[str, float]]:
        batch = MolecularBatch(
            adjacency=batch.adjacency.to(device),
            node_labels=batch.node_labels.to(device),
            edge_labels_dense=batch.edge_labels_dense.to(device),
            node_mask=batch.node_mask.to(device),
            edge_mask=batch.edge_mask.to(device),
        )
        x_t, e_t, t = self.corrupt_batch(batch, device=device)
        node_logits, edge_logits = self.forward(batch.adjacency, x_t, e_t, batch.node_mask, t)
        node_target = batch.node_labels.clamp(min=0)
        node_loss = torch.nn.functional.cross_entropy(
            node_logits[batch.node_mask], node_target[batch.node_mask]
        )
        # Bond targets are dense categories 1..num_bond_types; convert to 0..num_bond_types-1.
        edge_target = (batch.edge_labels_dense - 1).clamp(min=0)
        upper = torch.triu(torch.ones_like(batch.edge_mask, dtype=torch.bool), diagonal=1)
        edge_train_mask = batch.edge_mask & upper
        if edge_train_mask.any():
            edge_loss = torch.nn.functional.cross_entropy(edge_logits[edge_train_mask], edge_target[edge_train_mask])
        else:
            edge_loss = torch.zeros((), device=device)
        loss = node_loss + edge_loss
        return loss, {"node_loss": float(node_loss.detach().cpu()), "edge_loss": float(edge_loss.detach().cpu())}

    @torch.no_grad()
    def sample_attributes(
        self,
        topology: nx.Graph,
        *,
        steps: int = 32,
        temperature: float = 1.0,
        device: str | torch.device = "cpu",
        seed: int | None = None,
    ) -> nx.Graph:
        device = resolve_torch_device(str(device)) if not isinstance(device, torch.device) else device
        generator = torch.Generator(device=device)
        if seed is not None:
            generator.manual_seed(int(seed))
        g = nx.convert_node_labels_to_integers(nx.Graph(topology), ordering="sorted")
        n = g.number_of_nodes()
        A = torch.zeros((1, n, n), dtype=torch.bool, device=device)
        for u, v in g.edges():
            A[0, int(u), int(v)] = True
            A[0, int(v), int(u)] = True
        node_mask = torch.ones((1, n), dtype=torch.bool, device=device)
        x_t = torch.full((1, n), self.node_mask_index, dtype=torch.long, device=device)
        e_t = torch.zeros((1, n, n), dtype=torch.long, device=device)
        e_t[A] = self.edge_mask_index
        for s in range(steps):
            t_val = torch.tensor([(s + 1) / steps], dtype=torch.float32, device=device)
            node_logits, edge_logits = self.forward(A, x_t, e_t, node_mask, t_val)
            node_probs = torch.softmax(node_logits / max(temperature, 1.0e-6), dim=-1)
            edge_probs = torch.softmax(edge_logits / max(temperature, 1.0e-6), dim=-1)
            x_pred = torch.multinomial(node_probs.view(-1, self.num_atom_types), 1, generator=generator).view(1, n)
            edge_pred = torch.multinomial(edge_probs.view(-1, self.num_bond_types), 1, generator=generator).view(1, n, n) + 1
            keep_prob = float((s + 1) / steps)
            x_keep = torch.rand((1, n), device=device, generator=generator) < keep_prob
            x_t = torch.where(x_keep, x_pred, torch.full_like(x_t, self.node_mask_index))
            e_keep = torch.rand((1, n, n), device=device, generator=generator) < keep_prob
            e_t = torch.where(A & e_keep, edge_pred, e_t)
            e_t = torch.where(A & (~e_keep), torch.full_like(e_t, self.edge_mask_index), e_t)
            e_t = torch.where(A, e_t, torch.zeros_like(e_t))
            e_t = torch.triu(e_t, diagonal=1)
            e_t = e_t + e_t.transpose(1, 2)
        # Final argmax/sample without masks.
        t_val = torch.ones((1,), dtype=torch.float32, device=device)
        node_logits, edge_logits = self.forward(A, x_t, e_t, node_mask, t_val)
        x_final = torch.argmax(node_logits, dim=-1).squeeze(0).cpu().numpy().tolist()
        edge_final = torch.argmax(edge_logits, dim=-1).squeeze(0).cpu().numpy() + 1
        out = nx.Graph()
        for i, idx in enumerate(x_final):
            atomic_num = index_to_atom(int(idx))
            out.add_node(i, atomic_num=atomic_num, atom_type=atomic_num)
        for u, v in g.edges():
            b = index_to_bond_type(int(edge_final[int(u), int(v)]))
            out.add_edge(int(u), int(v), bond_type=b, bond_order=float(b if b != 4 else 1.5))
        return out


def save_attribute_flow_checkpoint(
    model: TopologyConditionalAttributeFlow,
    path: str | Path,
    *,
    config: dict[str, Any] | None = None,
    atom_types: tuple[int, ...] = QM9_ATOM_TYPES,
    bond_types: tuple[int, ...] = QM9_BOND_TYPES,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config or {},
            "atom_types": tuple(atom_types),
            "bond_types": tuple(bond_types),
            "num_atom_types": model.num_atom_types,
            "num_bond_types": model.num_bond_types,
            "hidden_dim": model.hidden_dim,
            "edge_dim": model.edge_dim,
            "num_layers": len(model.layers),
        },
        path,
    )


def load_attribute_flow_checkpoint(path: str | Path, *, device: str = "auto") -> TopologyConditionalAttributeFlow:
    dev = resolve_torch_device(device)
    ckpt = torch.load(Path(path), map_location=dev)
    model = TopologyConditionalAttributeFlow(
        num_atom_types=int(ckpt.get("num_atom_types", len(QM9_ATOM_TYPES))),
        num_bond_types=int(ckpt.get("num_bond_types", len(QM9_BOND_TYPES))),
        hidden_dim=int(ckpt.get("hidden_dim", 128)),
        edge_dim=int(ckpt.get("edge_dim", 64)),
        num_layers=int(ckpt.get("num_layers", 4)),
    ).to(dev)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model

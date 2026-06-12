from __future__ import annotations

import math
import random
from typing import Sequence

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # pragma: no cover - optional dependency.
    from torch_geometric.nn import GINConv, global_mean_pool  # type: ignore
except Exception:  # pragma: no cover
    class GINConv(nn.Module):
        """Minimal GINConv fallback for CPU smoke tests without torch_geometric."""

        def __init__(self, nn_module: nn.Module):
            super().__init__()
            self.nn = nn_module

        def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
            agg = torch.zeros_like(x)
            if edge_index.numel() > 0:
                row, col = edge_index[0].long(), edge_index[1].long()
                agg.index_add_(0, row, x[col])
            return self.nn(x + agg)

    def global_mean_pool(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if batch.numel() == 0:
            return x.mean(dim=0, keepdim=True)
        num_graphs = int(batch.max().item()) + 1 if batch.numel() else 1
        out = x.new_zeros((num_graphs, x.size(-1)))
        count = x.new_zeros((num_graphs, 1))
        out.index_add_(0, batch.long(), x)
        count.index_add_(0, batch.long(), x.new_ones((x.size(0), 1)))
        return out / count.clamp_min(1.0)

from grapher.generation.rewiring import (
    RewireAction,
    action_new_edges,
    check_sequence_validity,
    configuration_model_from_multiset,
    degree_sequence as graph_degree_sequence,
    deterministic_connected_havel_hakimi,
    enumerate_rewire_actions,
    graph_to_data,
    rewire_action,
)


def get_edge_representation(x: torch.Tensor, u: int, v: int, method: str = "sum_absdiff") -> torch.Tensor:
    """Backwards-compatible helper used by older experiments."""

    x_u, x_v = x[int(u)], x[int(v)]
    if method == "mean":
        return (x_u + x_v) / 2
    if method == "sum":
        return x_u + x_v
    if method == "max":
        return torch.max(x_u, x_v)
    if method == "sum_absdiff":
        return torch.cat([x_u + x_v, torch.abs(x_u - x_v)], dim=-1)
    return torch.cat([x_u, x_v], dim=-1)


def decode_degree_sequence(seq: Sequence[int]) -> list[int]:
    degrees: list[int] = []
    for degree, count in enumerate(seq):
        degrees.extend([degree] * int(count))
    return degrees


def get_sinusoidal_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half_dim = max(int(dim) // 2, 1)
    freqs = torch.exp(-torch.arange(0, half_dim, dtype=torch.float32, device=t.device) * (math.log(max_period) / half_dim))
    args = t.float().unsqueeze(-1) * freqs
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if emb.size(-1) < dim:
        emb = torch.cat([emb, emb.new_zeros((*emb.shape[:-1], dim - emb.size(-1)))], dim=-1)
    return emb[..., :dim]


def initialize_graphs(method: str, seq: Sequence[int]) -> nx.Graph:
    method_key = method.lower().replace("-", "_")
    if method_key in {"havel_hakimi", "hh", "havei_hakimi"}:  # keep typo as backwards-compatible alias
        return deterministic_connected_havel_hakimi(seq=seq)
    if method_key == "configuration_model":
        return configuration_model_from_multiset(seq)
    raise ValueError(f"Unknown initialization method: {method!r}")


def _normalized_time(value: int | float | torch.Tensor, T: int) -> float:
    if torch.is_tensor(value):
        value = float(value.detach().cpu().reshape(-1)[0].item())
    value = float(value)
    if value > 1.0:
        return max(0.0, min(1.0, value / max(float(T), 1.0)))
    return max(0.0, min(1.0, value))


def _pad_or_truncate_node_features(x: torch.Tensor, width: int) -> torch.Tensor:
    width = int(width)
    if x.size(1) == width:
        return x
    if x.size(1) < width:
        return torch.cat([x, x.new_zeros((x.size(0), width - x.size(1)))], dim=1)
    return x[:, :width]


class AtomHead(nn.Module):
    def __init__(self, hidden_dim: int, num_atom_types: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_atom_types))

    def forward(self, node_emb: torch.Tensor) -> torch.Tensor:
        return self.mlp(node_emb)


class BondHead(nn.Module):
    def __init__(self, hidden_dim: int, num_bond_types: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_bond_types))

    def forward(self, edge_feats: torch.Tensor) -> torch.Tensor:
        return self.mlp(edge_feats)


class GraphER(nn.Module):
    """Generic Graph-ER action scorer for complete rewiring actions.

    The generic model follows the paper's categorical action-field design:

        p_theta(a | G, t, D, C) = softmax_a g_theta(G, t, D, a),

    where each candidate action is a complete double-edge swap a=(e1,e2,r).
    Earlier code scored only the partner edge e2 conditioned on an anchor edge
    e1 and picked the first valid orientation outside the model.  This revision
    makes the orientation part of the scored action, adds graph/time/degree
    conditioning, and keeps the candidate set target-free during neural training
    and generation.
    """

    architecture = "generic_complete_action_grapher"

    def __init__(
        self,
        node_in_dim: int,
        hidden_dim: int,
        num_layer: int,
        T: int,
        *,
        max_nodes: int | None = None,
        degree_histogram_dim: int | None = None,
        time_embedding_dim: int | None = None,
        local_feature_dim: int = 8,
        dropout: float = 0.0,
        num_energy_targets: int = 2,  # accepted for backwards compatibility
        energy_hidden_dim: int = 64,  # accepted for backwards compatibility
    ):
        super().__init__()
        _ = (num_energy_targets, energy_hidden_dim)
        self.node_in_dim = int(node_in_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layer = int(num_layer)
        self.T = int(T)
        self.max_nodes = int(max_nodes or degree_histogram_dim or 64)
        self.degree_histogram_dim = int(degree_histogram_dim or self.max_nodes)
        self.time_embedding_dim = int(time_embedding_dim or hidden_dim)
        self.local_feature_dim = int(local_feature_dim)
        self.dropout = float(dropout)

        self.gin_layers = nn.ModuleList(
            [
                GINConv(
                    nn.Sequential(
                        nn.Linear(self.node_in_dim if i == 0 else self.hidden_dim, self.hidden_dim),
                        nn.ReLU(),
                        nn.Linear(self.hidden_dim, self.hidden_dim),
                    )
                )
                for i in range(self.num_layer)
            ]
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
        # h_G, h_D, e_t, and psi(G,a) = [edge_sum, edge_absdiff,
        # edge_product, r_embedding, local_features].  Each term is projected or
        # represented in hidden_dim.
        self.action_scorer = nn.Sequential(
            nn.Linear(self.hidden_dim * 8, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2 if self.hidden_dim >= 2 else 1),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2 if self.hidden_dim >= 2 else 1, 1),
        )
        nn.init.xavier_uniform_(self.orientation_embedding.weight)

    def encode_nodes(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = _pad_or_truncate_node_features(x.float(), self.node_in_dim)
        for gin in self.gin_layers:
            x = F.relu(gin(x, edge_index))
        return x

    def encode_graph(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        node_h = self.encode_nodes(x, edge_index)
        batch = node_h.new_zeros(node_h.size(0), dtype=torch.long)
        return self.graph_encoder(global_mean_pool(node_h, batch))

    def _degree_histogram_tensor(self, degree_sequence: Sequence[int], *, device: torch.device) -> torch.Tensor:
        hist = torch.zeros(self.degree_histogram_dim, dtype=torch.float32, device=device)
        for raw_degree in degree_sequence:
            degree = int(raw_degree)
            if degree < 0:
                raise ValueError(f"Invalid negative degree in degree sequence: {degree}")
            if degree >= self.degree_histogram_dim:
                raise ValueError(
                    f"Degree {degree} exceeds GraphER degree histogram width {self.degree_histogram_dim}. "
                    "Increase model.max_nodes/degree_histogram_dim or use a matching dataset checkpoint."
                )
            hist[degree] += 1.0
        return hist

    def encode_degree_sequence(self, degree_sequence: Sequence[int], *, device: torch.device) -> torch.Tensor:
        hist = self._degree_histogram_tensor(degree_sequence, device=device)
        # Normalize by graph size to make the degree embedding less sensitive to
        # absolute node count; size information is still available through the
        # total histogram mass and graph encoder.
        denom = hist.sum().clamp_min(1.0)
        return self.degree_encoder((hist / denom).view(1, -1)).squeeze(0)

    def encode_time(self, t: int | float | torch.Tensor, *, device: torch.device) -> torch.Tensor:
        t_norm = _normalized_time(t, self.T)
        t_tensor = torch.tensor([t_norm], dtype=torch.float32, device=device)
        return self.time_encoder(get_sinusoidal_embedding(t_tensor, self.time_embedding_dim)).squeeze(0)

    def _action_local_features(self, graph: nx.Graph | None, action: RewireAction, *, device: torch.device) -> torch.Tensor:
        if graph is None:
            return torch.zeros(self.local_feature_dim, dtype=torch.float32, device=device)
        n = max(int(graph.number_of_nodes()), 1)
        max_degree = max(n - 1, 1)
        (u, v), (x, y) = action.e1, action.e2
        new1, new2 = action_new_edges(action)
        values: list[float] = []
        for node in (u, v, x, y):
            values.append(float(graph.degree(int(node))) / float(max_degree))
        for a, b in (new1, new2):
            try:
                common = len(list(nx.common_neighbors(graph, int(a), int(b))))
            except Exception:
                common = 0
            values.append(float(common) / float(max(n, 1)))
        # Locality signal: minimum endpoint distance between removed edges,
        # normalized by graph size.  Missing paths are encoded as 1.0.
        distances: list[int] = []
        for a in action.e1:
            for b in action.e2:
                try:
                    distances.append(int(nx.shortest_path_length(graph, int(a), int(b))))
                except nx.NetworkXNoPath:
                    pass
        values.append(float(min(distances)) / float(max(n - 1, 1)) if distances else 1.0)
        try:
            bridges = {tuple(sorted(e)) for e in nx.bridges(graph)}
            values.append(1.0 if tuple(sorted(action.e1)) in bridges or tuple(sorted(action.e2)) in bridges else 0.0)
        except Exception:
            values.append(0.0)
        if len(values) < self.local_feature_dim:
            values.extend([0.0] * (self.local_feature_dim - len(values)))
        return torch.tensor(values[: self.local_feature_dim], dtype=torch.float32, device=device)

    def score_actions(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        actions: Sequence[RewireAction],
        *,
        t: int | float | torch.Tensor,
        degree_sequence: Sequence[int] | None = None,
        graph: nx.Graph | None = None,
    ) -> torch.Tensor:
        """Score a finite candidate set C of complete rewiring actions."""

        if not actions:
            return x.new_empty((0,))
        device = x.device
        node_h = self.encode_nodes(x, edge_index)
        batch = node_h.new_zeros(node_h.size(0), dtype=torch.long)
        graph_h = self.graph_encoder(global_mean_pool(node_h, batch)).squeeze(0)
        if degree_sequence is None:
            if graph is not None:
                degree_sequence = graph_degree_sequence(graph)
            else:
                # Fallback for ad hoc calls: infer a degree sequence from the
                # undirected edge_index.  Edge_index usually stores both
                # directions, so divide counts by two only when appropriate.
                deg = torch.bincount(edge_index[0].long(), minlength=node_h.size(0)).detach().cpu().tolist()
                degree_sequence = sorted([int(d) for d in deg], reverse=True)
        degree_h = self.encode_degree_sequence(degree_sequence, device=device)
        time_h = self.encode_time(t, device=device)

        scores: list[torch.Tensor] = []
        for action in actions:
            (u, v), (x_idx, y_idx) = action.e1, action.e2
            max_node = node_h.size(0) - 1
            if min(u, v, x_idx, y_idx) < 0 or max(u, v, x_idx, y_idx) > max_node:
                raise IndexError(f"Action {action} references nodes outside encoded graph with {node_h.size(0)} nodes.")
            huv = node_h[int(u)] + node_h[int(v)]
            hxy = node_h[int(x_idx)] + node_h[int(y_idx)]
            edge_sum = huv + hxy
            edge_absdiff = torch.abs(huv - hxy)
            edge_product = huv * hxy
            orient_idx = torch.tensor(int(action.orientation), dtype=torch.long, device=device)
            orient_h = self.orientation_embedding(orient_idx)
            local_h = self.local_encoder(self._action_local_features(graph, action, device=device).view(1, -1)).squeeze(0)
            feat = torch.cat([graph_h, degree_h, time_h, edge_sum, edge_absdiff, edge_product, orient_h, local_h], dim=-1)
            scores.append(self.action_scorer(feat).view(()))
        return torch.stack(scores, dim=0)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        first_edge: tuple[int, int] | None = None,
        candidate_edges: Sequence[tuple[int, int]] | None = None,
        t: int | float | torch.Tensor = 0,
        *,
        actions: Sequence[RewireAction] | None = None,
        degree_sequence: Sequence[int] | None = None,
        graph: nx.Graph | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        New code should pass ``actions=[RewireAction(...), ...]``.  The older
        ``first_edge`` + ``candidate_edges`` signature is retained as a thin
        compatibility layer and scores orientation-0 actions only.
        """

        if actions is None:
            if first_edge is None or candidate_edges is None:
                raise TypeError("GraphER.forward requires either actions=... or first_edge plus candidate_edges.")
            actions = [RewireAction(first_edge, edge, 0) for edge in candidate_edges]
        return self.score_actions(x, edge_index, actions, t=t, degree_sequence=degree_sequence, graph=graph)

    def save_model(self, file_path: str) -> None:
        torch.save(self.state_dict(), file_path)

    def load_model(self, file_path: str) -> None:
        self.load_state_dict(torch.load(file_path, map_location="cpu"))
        self.eval()

    def generate(
        self,
        num_samples: int,
        num_steps: int,
        dhvae_model,
        k_eigen: int,
        method: str = "havel_hakimi",
        ensure_connected: bool = True,
        k_hop: int | None = 2,
        max_candidates: int | None = None,
        degree_temperature: float = 1.0,
        action_temperature: float = 1.0,
        sample_actions: bool = True,
        max_degree_sequence_attempt_factor: int = 8,
    ):
        """Generate graphs from DH-VAE degree samples plus learned rewiring flow."""

        self.eval()
        generated_graphs: list[nx.Graph] = []
        generated_seqs: list[list[int]] = []
        initial_graphs: list[tuple[nx.Graph, list[int]]] = []
        num_degree_attempts = max(int(num_samples) * int(max_degree_sequence_attempt_factor), int(num_samples))
        degree_sequences = dhvae_model.generate(num_degree_attempts, temperature=float(degree_temperature))
        for seq in degree_sequences:
            valid, _ = check_sequence_validity(seq)
            if not valid:
                continue
            if len(seq) > self.max_nodes or (seq and max(int(d) for d in seq) >= self.degree_histogram_dim):
                continue
            try:
                g0 = initialize_graphs(method, seq)
            except Exception:
                continue
            initial_graphs.append((g0, list(map(int, seq))))
            if len(initial_graphs) >= int(num_samples):
                break

        device = next(self.parameters()).device
        temperature = max(float(action_temperature), 1e-6)
        with torch.no_grad():
            for g0, seq in initial_graphs:
                g = nx.convert_node_labels_to_integers(nx.Graph(g0), ordering="sorted")
                for step in range(int(num_steps)):
                    actions = enumerate_rewire_actions(
                        g,
                        ensure_connected=ensure_connected,
                        k_hop=k_hop,
                        max_candidates=max_candidates,
                        shuffle=True,
                    )
                    if not actions:
                        continue
                    data = graph_to_data(g, k_eigen).to(device)
                    data.x = _pad_or_truncate_node_features(data.x, self.node_in_dim)
                    t = float(step) / max(float(num_steps), 1.0)
                    scores = self.score_actions(data.x, data.edge_index, actions, t=t, degree_sequence=seq, graph=g)
                    if scores.numel() == 0:
                        continue
                    if sample_actions:
                        probs = F.softmax(scores / temperature, dim=0)
                        idx = int(torch.multinomial(probs, num_samples=1).item())
                    else:
                        idx = int(torch.argmax(scores).item())
                    out = rewire_action(g, actions[idx], ensure_connected=ensure_connected)
                    if out is not None:
                        g = nx.convert_node_labels_to_integers(out[0], ordering="sorted")
                generated_graphs.append(nx.convert_node_labels_to_integers(g, ordering="sorted"))
                generated_seqs.append(seq)
        return generated_graphs, generated_seqs

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import torch
from torch import nn

from grapher.properties.summary import SummaryConfig, distance_to_summary
from grapher.refinement.rewiring import Action, apply_action, sample_valid_double_edge_swaps
from grapher.utils.device import resolve_torch_device


def _as_float_array(value: Any) -> np.ndarray:
    if value is None:
        return np.zeros(0, dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    arr[~np.isfinite(arr)] = 0.0
    return arr


def _pad_or_trim(value: Any, width: int) -> np.ndarray:
    arr = _as_float_array(value)
    out = np.zeros(width, dtype=np.float32)
    if arr.size:
        out[: min(width, arr.size)] = arr[:width]
    return out


def _graphlet_history_to_fixed_vector(target: dict[str, Any], width: int) -> np.ndarray:
    history = target.get("graphlet_history", {}) or {}
    values: list[float] = []
    for k in sorted(history.keys(), key=lambda x: int(x)):
        hist = history.get(str(k), {}) or {}
        for key in sorted(hist.keys()):
            values.append(float(hist.get(key, 0.0)))
    return _pad_or_trim(values, width)


def _safe_scalar(value: Any, default: float = 0.0) -> float:
    try:
        val = float(value)
        return val if math.isfinite(val) else default
    except Exception:
        return default


def _common_neighbors(graph: nx.Graph, edge: tuple[int, int]) -> int:
    u, v = edge
    if u not in graph or v not in graph:
        return 0
    return len(set(graph.neighbors(u)).intersection(graph.neighbors(v)))


def _edge(edge_like: tuple[int, int] | list[int]) -> tuple[int, int]:
    u, v = int(edge_like[0]), int(edge_like[1])
    return (u, v) if u < v else (v, u)


def _graph_context_features(
    graph: nx.Graph,
    target: dict[str, Any],
    feature_cfg: dict[str, int],
) -> np.ndarray:
    n = max(int(graph.number_of_nodes()), 1)
    m = int(graph.number_of_edges())
    degrees = np.asarray([d for _, d in graph.degree()], dtype=np.float32)
    if degrees.size == 0:
        degrees = np.zeros(1, dtype=np.float32)

    density = float(nx.density(graph)) if n > 1 else 0.0
    triangles = float(sum(nx.triangles(graph).values()) / 3.0) if n else 0.0
    triangle_norm = triangles / max(n, 1)
    transitivity = float(nx.transitivity(graph)) if m > 0 else 0.0
    avg_clustering = float(nx.average_clustering(graph)) if n else 0.0

    target_n = max(_safe_scalar(target.get("num_nodes", n), n), 1.0)
    target_m = _safe_scalar(target.get("num_edges", m), m)
    target_density = _safe_scalar(target.get("density", density), density)
    target_triangle = _safe_scalar(
        target.get("triangle_count_norm", triangle_norm),
        triangle_norm,
    )

    scalar = np.asarray(
        [
            n / 256.0,
            m / max(n * n, 1),
            density,
            float(degrees.mean()) / 256.0,
            float(degrees.std()) / 256.0,
            float(degrees.max()) / 256.0,
            triangle_norm,
            transitivity,
            avg_clustering,
            target_n / 256.0,
            target_m / max(target_n * target_n, 1.0),
            target_density,
            target_triangle,
            target_density - density,
            target_triangle - triangle_norm,
        ],
        dtype=np.float32,
    )

    target_vecs = [
        _pad_or_trim(target.get("degree_hist", []), feature_cfg["degree_width"]),
        _pad_or_trim(target.get("clustering_hist", []), feature_cfg["clustering_width"]),
        _pad_or_trim(target.get("spectral_hist", []), feature_cfg["spectral_width"]),
        _pad_or_trim(target.get("motif_proxy", []), feature_cfg["motif_width"]),
        _pad_or_trim(target.get("orbit_count", []), feature_cfg["orbit_width"]),
        _graphlet_history_to_fixed_vector(target, feature_cfg.get("graphlet_width", 0)),
    ]

    return np.concatenate([scalar, *target_vecs], axis=0).astype(np.float32)


def _action_local_features(graph: nx.Graph, action: Action) -> np.ndarray:
    n = max(int(graph.number_of_nodes()), 1)

    removed, added = action
    removed = [_edge(e) for e in removed]
    added = [_edge(e) for e in added]

    degree = dict(graph.degree())
    clustering = nx.clustering(graph)

    endpoints: list[int] = []
    for e in removed:
        endpoints.extend([e[0], e[1]])
    endpoints = endpoints[:4]
    while len(endpoints) < 4:
        endpoints.append(0)

    endpoint_degrees = np.asarray(
        [degree.get(v, 0) / max(n - 1, 1) for v in endpoints],
        dtype=np.float32,
    )
    endpoint_clust = np.asarray(
        [clustering.get(v, 0.0) for v in endpoints],
        dtype=np.float32,
    )

    removed_common = np.asarray(
        [_common_neighbors(graph, e) for e in removed],
        dtype=np.float32,
    )

    graph_after_remove = graph.copy()
    for u, v in removed:
        if graph_after_remove.has_edge(u, v):
            graph_after_remove.remove_edge(u, v)

    added_common = np.asarray(
        [_common_neighbors(graph_after_remove, e) for e in added],
        dtype=np.float32,
    )

    rem_sum = float(removed_common.sum())
    add_sum = float(added_common.sum())
    delta_triangles = (add_sum - rem_sum) / max(n, 1)

    add_degree_pairs = []
    for u, v in added:
        add_degree_pairs.append(
            (degree.get(u, 0) + degree.get(v, 0)) / max(2 * (n - 1), 1)
        )
    while len(add_degree_pairs) < 2:
        add_degree_pairs.append(0.0)

    rem_degree_pairs = []
    for u, v in removed:
        rem_degree_pairs.append(
            (degree.get(u, 0) + degree.get(v, 0)) / max(2 * (n - 1), 1)
        )
    while len(rem_degree_pairs) < 2:
        rem_degree_pairs.append(0.0)

    return np.asarray(
        [
            *endpoint_degrees.tolist(),
            *endpoint_clust.tolist(),
            rem_sum / max(n, 1),
            add_sum / max(n, 1),
            float(removed_common.mean()) / max(n, 1) if removed_common.size else 0.0,
            float(added_common.mean()) / max(n, 1) if added_common.size else 0.0,
            delta_triangles,
            *rem_degree_pairs[:2],
            *add_degree_pairs[:2],
        ],
        dtype=np.float32,
    )


class CandidateMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        layers: list[nn.Module] = []
        dim = input_dim

        for _ in range(max(int(num_layers) - 1, 1)):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            dim = hidden_dim

        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class LoadedSelector:
    model: CandidateMLP
    feature_cfg: dict[str, int]
    device: torch.device

    def score_actions(
        self,
        graph: nx.Graph,
        target_summary: dict[str, Any],
        actions: list[Action],
    ) -> np.ndarray:
        if not actions:
            return np.zeros(0, dtype=np.float64)

        context = _graph_context_features(graph, target_summary, self.feature_cfg)
        features = []

        for action in actions:
            local = _action_local_features(graph, action)
            features.append(np.concatenate([context, local], axis=0))

        x = torch.tensor(np.stack(features, axis=0), dtype=torch.float32, device=self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(x).detach().cpu().numpy().astype(np.float64)

        return logits


def load_learned_selector(selector_cfg: dict[str, Any]) -> LoadedSelector:
    checkpoint_path = selector_cfg.get("checkpoint_path") or selector_cfg.get("checkpoint")
    if not checkpoint_path:
        raise ValueError("selector.checkpoint_path is required for learned_selector refiner.")

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing learned selector checkpoint: {checkpoint_path}")

    device = resolve_torch_device(selector_cfg.get("device", "auto"))
    checkpoint = torch.load(checkpoint_path, map_location=device)

    input_dim = int(checkpoint["input_dim"])
    hidden_dim = int(checkpoint.get("hidden_dim", selector_cfg.get("hidden_dim", 256)))
    num_layers = int(checkpoint.get("num_layers", selector_cfg.get("num_layers", 3)))
    dropout = float(checkpoint.get("dropout", selector_cfg.get("dropout", 0.1)))

    feature_cfg = checkpoint.get(
        "feature_cfg",
        {
            "degree_width": int(selector_cfg.get("degree_width", 64)),
            "clustering_width": int(selector_cfg.get("clustering_width", 20)),
            "spectral_width": int(selector_cfg.get("spectral_width", 20)),
            "motif_width": int(selector_cfg.get("motif_width", 5)),
            "orbit_width": int(selector_cfg.get("orbit_width", 15)),
            "graphlet_width": int(selector_cfg.get("graphlet_width", 128)),
        },
    )

    model = CandidateMLP(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model.eval()

    return LoadedSelector(model=model, feature_cfg=feature_cfg, device=device)


def _choose_by_logits(
    actions: list[Action],
    logits: np.ndarray,
    *,
    selection: str,
    temperature: float,
    rng: np.random.Generator,
) -> tuple[Action | None, float]:
    if not actions:
        return None, 0.0

    selection = str(selection).lower()

    if selection in {"greedy", "argmax"}:
        idx = int(np.argmax(logits))
        return actions[idx], float(logits[idx])

    if selection in {"softmax", "soft", "sample"}:
        temp = max(float(temperature), 1.0e-8)
        z = logits / temp
        z = z - np.max(z)
        probs = np.exp(z)
        probs = probs / probs.sum()
        idx = int(rng.choice(len(actions), p=probs))
        return actions[idx], float(logits[idx])

    raise ValueError(f"Unknown learned selector selection mode: {selection!r}")


def refine_graph_with_selector(
    graph: nx.Graph,
    target_summary: dict[str, Any],
    *,
    selector: LoadedSelector,
    summary_config: SummaryConfig | dict[str, Any] | None = None,
    energy_weights: dict[str, Any] | None = None,
    refiner_config: dict[str, Any] | None = None,
    rng: np.random.Generator | None = None,
    return_trace: bool = False,
) -> nx.Graph | tuple[nx.Graph, list[dict[str, Any]]]:
    cfg = refiner_config or {}
    generator = rng if rng is not None else np.random.default_rng(0)

    steps = int(cfg.get("steps", 20))
    candidate_budget = int(cfg.get("candidate_budget", 64))
    preserve_connectivity = bool(cfg.get("preserve_connectivity", True))
    selection = str(cfg.get("selection", "greedy"))
    temperature = float(cfg.get("temperature", 0.1))

    # Usually false for learned selector. If true, this filters using actual energy.
    accept_only_improving = bool(cfg.get("accept_only_improving", False))
    min_improvement = float(cfg.get("min_improvement", 1.0e-12))

    g = graph.copy()
    trace: list[dict[str, Any]] = []

    current_energy = None
    if return_trace or accept_only_improving:
        current_energy = distance_to_summary(g, target_summary, summary_config, energy_weights)

    for step in range(steps):
        candidates = sample_valid_double_edge_swaps(
            g,
            candidate_budget,
            generator,
            preserve_connectivity=preserve_connectivity,
        )

        if not candidates:
            if return_trace:
                trace.append(
                    {
                        "step": step,
                        "accepted": False,
                        "reason": "no_candidates",
                        "num_candidates": 0,
                    }
                )
            break

        logits = selector.score_actions(g, target_summary, candidates)

        if accept_only_improving:
            assert current_energy is not None
            deltas = []
            for action in candidates:
                candidate_energy = distance_to_summary(
                    apply_action(g, action),
                    target_summary,
                    summary_config,
                    energy_weights,
                )
                deltas.append(float(current_energy - candidate_energy))

            deltas_np = np.asarray(deltas, dtype=np.float64)
            mask = deltas_np > min_improvement

            if not np.any(mask):
                if return_trace:
                    trace.append(
                        {
                            "step": step,
                            "accepted": False,
                            "reason": "no_improving_candidate",
                            "best_delta": float(np.max(deltas_np)),
                            "num_candidates": len(candidates),
                        }
                    )
                break

            candidates = [a for a, keep in zip(candidates, mask) if bool(keep)]
            logits = logits[mask]

        chosen_action, chosen_logit = _choose_by_logits(
            candidates,
            logits,
            selection=selection,
            temperature=temperature,
            rng=generator,
        )

        if chosen_action is None:
            break

        old_energy = current_energy
        g = apply_action(g, chosen_action)

        actual_delta = None
        if return_trace or accept_only_improving:
            new_energy = distance_to_summary(g, target_summary, summary_config, energy_weights)
            if old_energy is not None:
                actual_delta = float(old_energy - new_energy)
            current_energy = new_energy

        if return_trace:
            trace.append(
                {
                    "step": step,
                    "accepted": True,
                    "num_candidates": len(candidates),
                    "chosen_logit": float(chosen_logit),
                    "actual_delta": actual_delta,
                }
            )

    if return_trace:
        return g, trace

    return g

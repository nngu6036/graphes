from __future__ import annotations

import math
from typing import Any

import networkx as nx
import numpy as np

from grapher.properties.summary import SummaryConfig, extract_summary
from grapher.refinement.rewiring import Action

VECTOR_FIELDS = (
    ("degree_hist", "degree_width"),
    ("clustering_hist", "clustering_width"),
    ("spectral_hist", "spectral_width"),
    ("motif_proxy", "motif_width"),
    ("orbit_count", "orbit_width"),
)


def safe_scalar(value: Any, default: float = 0.0) -> float:
    try:
        val = float(value)
        return val if math.isfinite(val) else default
    except (TypeError, ValueError):
        return default


def pad_or_trim(value: Any, width: int) -> np.ndarray:
    out = np.zeros(max(int(width), 0), dtype=np.float32)
    if value is None or out.size == 0:
        return out
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    arr[~np.isfinite(arr)] = 0.0
    out[: min(out.size, arr.size)] = arr[: out.size]
    return out


def _graphlet_key_order(
    current: dict[str, Any],
    target: dict[str, Any],
) -> list[tuple[str, str]]:
    current_history = current.get("graphlet_history", {}) or {}
    target_history = target.get("graphlet_history", {}) or {}
    sizes = sorted(
        {str(k) for k in current_history} | {str(k) for k in target_history},
        key=int,
    )
    keys: list[tuple[str, str]] = []
    for k in sizes:
        current_hist = current_history.get(k, {}) or {}
        target_hist = target_history.get(k, {}) or {}
        for key in sorted(set(current_hist) | set(target_hist)):
            keys.append((k, str(key)))
    return keys


def graphlet_pair_vectors(
    current: dict[str, Any],
    target: dict[str, Any],
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    keys = _graphlet_key_order(current, target)
    current_history = current.get("graphlet_history", {}) or {}
    target_history = target.get("graphlet_history", {}) or {}
    current_values = [
        safe_scalar((current_history.get(k, {}) or {}).get(key, 0.0)) for k, key in keys
    ]
    target_values = [
        safe_scalar((target_history.get(k, {}) or {}).get(key, 0.0)) for k, key in keys
    ]
    return (
        pad_or_trim(current_values, width),
        pad_or_trim(target_values, width),
    )


def _summary_config_from_features(
    target: dict[str, Any],
    feature_cfg: dict[str, Any],
) -> SummaryConfig:
    target_history = target.get("graphlet_history", {}) or {}
    sizes = sorted(int(k) for k in target_history)
    return SummaryConfig.from_dict(
        {
            "degree_hist_max_degree": max(
                int(feature_cfg.get("degree_width", 1)) - 1,
                0,
            ),
            "clustering_bins": int(feature_cfg.get("clustering_width", 20)),
            "spectral_bins": int(feature_cfg.get("spectral_width", 20)),
            "clustering_summary": int(feature_cfg.get("clustering_width", 0)) > 0,
            "spectral_summary": int(feature_cfg.get("spectral_width", 0)) > 0,
            "motif_proxy": int(feature_cfg.get("motif_width", 0)) > 0,
            "orbit_count": int(feature_cfg.get("orbit_width", 0)) > 0,
            "graphlet_history": int(feature_cfg.get("graphlet_width", 0)) > 0,
            "graphlet_k_min": min(sizes) if sizes else 3,
            "graphlet_k_max": max(sizes) if sizes else 5,
            "graphlet_connected_only": bool(
                feature_cfg.get("graphlet_connected_only", True)
            ),
            "graphlet_num_samples": feature_cfg.get("graphlet_num_samples"),
        }
    )


def _legacy_graph_context_features(
    graph: nx.Graph,
    target: dict[str, Any],
    feature_cfg: dict[str, Any],
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
    target_n = max(safe_scalar(target.get("num_nodes", n), n), 1.0)
    target_m = safe_scalar(target.get("num_edges", m), m)
    target_density = safe_scalar(target.get("density", density), density)
    target_triangle = safe_scalar(
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
    vectors = [
        pad_or_trim(target.get(field, []), int(feature_cfg.get(width_key, 0)))
        for field, width_key in VECTOR_FIELDS
    ]
    _, target_graphlet = graphlet_pair_vectors(
        {},
        target,
        int(feature_cfg.get("graphlet_width", 0)),
    )
    return np.concatenate([scalar, *vectors, target_graphlet]).astype(np.float32)


def graph_context_features(
    graph: nx.Graph,
    target: dict[str, Any],
    feature_cfg: dict[str, Any],
    *,
    current_summary: dict[str, Any] | None = None,
    summary_config: SummaryConfig | dict[str, Any] | None = None,
) -> np.ndarray:
    """Permutation-invariant state, target, and target-residual features."""

    feature_version = int(feature_cfg.get("feature_version", 1))
    if feature_version <= 1:
        return _legacy_graph_context_features(graph, target, feature_cfg)

    if current_summary is None:
        cfg = (
            summary_config
            if isinstance(summary_config, SummaryConfig)
            else SummaryConfig.from_dict(summary_config or {})
            if summary_config is not None
            else _summary_config_from_features(target, feature_cfg)
        )
        current_summary = extract_summary(graph, cfg)

    n = max(int(graph.number_of_nodes()), 1)
    m = int(graph.number_of_edges())
    target_n = max(int(target.get("num_nodes", n)), 1)
    target_m = int(target.get("num_edges", m))
    current_motif = np.asarray(current_summary.get("motif_proxy", []), dtype=float)
    target_motif = np.asarray(target.get("motif_proxy", []), dtype=float)

    current_scalar = np.asarray(
        [
            n / 256.0,
            m / max(n * n, 1),
            safe_scalar(current_summary.get("density", nx.density(graph))),
            safe_scalar(current_summary.get("triangle_count_norm", 0.0)),
            safe_scalar(
                current_motif[3] if current_motif.size > 3 else nx.transitivity(graph)
            ),
            safe_scalar(
                current_motif[4]
                if current_motif.size > 4
                else nx.average_clustering(graph)
            ),
        ],
        dtype=np.float32,
    )
    target_scalar = np.asarray(
        [
            target_n / 256.0,
            target_m / max(target_n * target_n, 1),
            safe_scalar(target.get("density", current_scalar[2])),
            safe_scalar(target.get("triangle_count_norm", current_scalar[3])),
            safe_scalar(
                target_motif[3] if target_motif.size > 3 else current_scalar[4]
            ),
            safe_scalar(
                target_motif[4] if target_motif.size > 4 else current_scalar[5]
            ),
        ],
        dtype=np.float32,
    )
    parts = [current_scalar, target_scalar, target_scalar - current_scalar]

    for field, width_key in VECTOR_FIELDS:
        width = int(feature_cfg.get(width_key, 0))
        current_vector = pad_or_trim(current_summary.get(field, []), width)
        target_vector = pad_or_trim(target.get(field, []), width)
        parts.extend([current_vector, target_vector, target_vector - current_vector])

    current_graphlet, target_graphlet = graphlet_pair_vectors(
        current_summary,
        target,
        int(feature_cfg.get("graphlet_width", 0)),
    )
    parts.extend(
        [
            current_graphlet,
            target_graphlet,
            target_graphlet - current_graphlet,
        ]
    )
    return np.concatenate(parts).astype(np.float32)


def _edge(edge_like: Any) -> tuple[int, int]:
    u, v = int(edge_like[0]), int(edge_like[1])
    return (u, v) if u < v else (v, u)


def _common_neighbors(graph: nx.Graph, edge: tuple[int, int]) -> int:
    u, v = edge
    if u not in graph or v not in graph:
        return 0
    return len(set(graph.neighbors(u)).intersection(graph.neighbors(v)))


def _unpack_action(
    action: Action | dict[str, Any] | None,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], bool]:
    if action is None:
        return [], [], True
    if isinstance(action, dict):
        is_stop = (
            str(action.get("type", action.get("action_type", ""))).lower() == "stop"
        )
        removed = [_edge(edge) for edge in action.get("removed", [])]
        added = [_edge(edge) for edge in action.get("added", [])]
        return removed, added, is_stop
    removed, added = action
    return [_edge(edge) for edge in removed], [_edge(edge) for edge in added], False


def action_local_features(
    graph: nx.Graph,
    action: Action | dict[str, Any] | None,
    *,
    feature_version: int = 2,
) -> np.ndarray:
    """Permutation-invariant local features for a swap or the STOP action."""

    removed, added, is_stop = _unpack_action(action)
    n = max(int(graph.number_of_nodes()), 1)
    degree = dict(graph.degree())
    clustering = nx.clustering(graph)

    endpoints = [node for edge in removed for node in edge]
    endpoint_degrees = sorted([degree.get(v, 0) / max(n - 1, 1) for v in endpoints])
    endpoint_clustering = sorted([clustering.get(v, 0.0) for v in endpoints])
    endpoint_degrees = pad_or_trim(endpoint_degrees, 4)
    endpoint_clustering = pad_or_trim(endpoint_clustering, 4)

    removed_common = np.asarray(
        sorted(_common_neighbors(graph, edge) for edge in removed),
        dtype=np.float32,
    )
    graph_after_remove = graph.copy()
    for u, v in removed:
        if graph_after_remove.has_edge(u, v):
            graph_after_remove.remove_edge(u, v)
    added_common = np.asarray(
        sorted(_common_neighbors(graph_after_remove, edge) for edge in added),
        dtype=np.float32,
    )
    rem_sum = float(removed_common.sum())
    add_sum = float(added_common.sum())
    delta_triangles = (add_sum - rem_sum) / max(n, 1)
    removed_degree_pairs = sorted(
        (degree.get(u, 0) + degree.get(v, 0)) / max(2 * (n - 1), 1) for u, v in removed
    )
    added_degree_pairs = sorted(
        (degree.get(u, 0) + degree.get(v, 0)) / max(2 * (n - 1), 1) for u, v in added
    )

    legacy = np.asarray(
        [
            *endpoint_degrees.tolist(),
            *endpoint_clustering.tolist(),
            rem_sum / max(n, 1),
            add_sum / max(n, 1),
            float(removed_common.mean()) / max(n, 1) if removed_common.size else 0.0,
            float(added_common.mean()) / max(n, 1) if added_common.size else 0.0,
            delta_triangles,
            *pad_or_trim(removed_degree_pairs, 2).tolist(),
            *pad_or_trim(added_degree_pairs, 2).tolist(),
        ],
        dtype=np.float32,
    )
    if int(feature_version) <= 1:
        return legacy
    return np.concatenate([legacy, np.asarray([float(is_stop)], dtype=np.float32)])

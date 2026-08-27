from __future__ import annotations

import itertools
from collections import defaultdict
from functools import lru_cache
from math import comb
from typing import Any, Mapping, Sequence

import networkx as nx
import numpy as np

from grapher.rewiring_mlp.attributed.data import (
    GRAPHLET_OVERFLOW_KEY,
    GraphletBasis,
)
from grapher.rewiring_mlp.core.rewiring import Action
from grapher.rewiring_mlp.generic.graphlet_diffusion import (
    GraphletLogitBridgeSchedule,
    graphlet_clr_to_simplex as _graphlet_clr_to_simplex,
    graphlet_logit_distance as _graphlet_logit_distance,
    graphlet_simplex_to_clr as _graphlet_simplex_to_clr,
)
from grapher.utils.motifs import canonicalize_attributed_graph_python

AttributedGraphletCounts = dict[str, dict[str, int]]


def _require_python_basis(graphlet_basis: GraphletBasis) -> None:
    if not graphlet_basis.attributed:
        raise ValueError("Attributed graphlet diffusion requires an attributed basis.")
    if not graphlet_basis.node_attribute or not graphlet_basis.edge_attribute:
        raise ValueError("Attributed graphlet basis is missing node/edge attribute names.")
    if str(graphlet_basis.attributed_backend).lower() != "python":
        raise ValueError(
            "Fast exact attributed graphlet deltas currently require "
            "graphlet_prediction.attributed_backend: python so training and "
            "generation use the same canonical-key convention."
        )


def _raw_pattern(
    graph: nx.Graph,
    nodes: Sequence[int],
    *,
    node_attribute: str,
    edge_attribute: str,
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    order = tuple(int(node) for node in nodes)
    node_labels = tuple(graph.nodes[node][node_attribute] for node in order)
    edge_labels: list[Any] = []
    for left in range(len(order)):
        for right in range(left + 1, len(order)):
            u, v = order[left], order[right]
            edge_labels.append(
                graph.edges[u, v][edge_attribute] if graph.has_edge(u, v) else None
            )
    return node_labels, tuple(edge_labels)


@lru_cache(maxsize=262144)
def _canonical_key_from_raw(
    node_labels: tuple[Any, ...],
    edge_labels: tuple[Any, ...],
    node_attribute: str,
    edge_attribute: str,
) -> str:
    graph = nx.Graph()
    graph.add_nodes_from(
        (index, {node_attribute: value})
        for index, value in enumerate(node_labels)
    )
    cursor = 0
    for left in range(len(node_labels)):
        for right in range(left + 1, len(node_labels)):
            value = edge_labels[cursor]
            cursor += 1
            if value is not None:
                graph.add_edge(left, right, **{edge_attribute: value})
    return canonicalize_attributed_graph_python(
        graph,
        node_label_attr=node_attribute,
        edge_label_attr=edge_attribute,
        max_nodes=7,
    )


def _canonical_key(
    graph: nx.Graph,
    nodes: Sequence[int],
    *,
    graphlet_basis: GraphletBasis,
) -> str:
    _require_python_basis(graphlet_basis)
    node_attribute = str(graphlet_basis.node_attribute)
    edge_attribute = str(graphlet_basis.edge_attribute)
    raw = _raw_pattern(
        graph,
        nodes,
        node_attribute=node_attribute,
        edge_attribute=edge_attribute,
    )
    return _canonical_key_from_raw(
        raw[0], raw[1], node_attribute, edge_attribute
    )


def _basis_key(key: str, graphlet_basis: GraphletBasis, size: str) -> str:
    known = set(graphlet_basis.keys_by_k[size])
    if key in known:
        return key
    if graphlet_basis.overflow_key is not None and graphlet_basis.overflow_key in known:
        return str(graphlet_basis.overflow_key)
    raise KeyError(
        f"Attributed graphlet key is outside the fixed basis and no overflow bin exists: {key}"
    )


def extract_attributed_graphlet_counts(
    graph: nx.Graph,
    *,
    graphlet_basis: GraphletBasis,
) -> AttributedGraphletCounts:
    """Count connected attributed graphlets exactly in the fixed vocabulary."""

    _require_python_basis(graphlet_basis)
    nodes = tuple(sorted(int(node) for node in graph.nodes()))
    counts_by_size: AttributedGraphletCounts = {}
    for size in graphlet_basis.sizes:
        k = int(size)
        block: defaultdict[str, int] = defaultdict(int)
        if len(nodes) >= k:
            for subset in itertools.combinations(nodes, k):
                subgraph = graph.subgraph(subset)
                if k > 1 and not nx.is_connected(subgraph):
                    continue
                key = _basis_key(
                    _canonical_key(graph, subset, graphlet_basis=graphlet_basis),
                    graphlet_basis,
                    size,
                )
                block[key] += 1
        counts_by_size[size] = dict(block)
    return counts_by_size


def attributed_graphlet_simplex_from_counts(
    counts_by_size: AttributedGraphletCounts,
    *,
    num_nodes: int,
    graphlet_basis: GraphletBasis,
) -> tuple[np.ndarray, np.ndarray]:
    values: list[float] = []
    mask: list[bool] = []
    n = int(num_nodes)
    for size in graphlet_basis.sizes:
        k = int(size)
        keys = graphlet_basis.keys_by_k[size]
        total = comb(n, k) if n >= k else 0
        if total <= 0:
            values.extend([0.0] * (len(keys) + 1))
            mask.extend([False] * (len(keys) + 1))
            continue
        counts = counts_by_size.get(size, {})
        block = np.asarray(
            [float(counts.get(key, 0)) / float(total) for key in keys],
            dtype=np.float64,
        )
        connected_mass = float(np.clip(block.sum(), 0.0, 1.0))
        full = np.concatenate(
            [block, np.asarray([max(0.0, 1.0 - connected_mass)], dtype=np.float64)]
        )
        full /= max(float(full.sum()), 1.0e-12)
        values.extend(full.tolist())
        mask.extend([True] * full.size)
    return np.asarray(values, dtype=np.float64), np.asarray(mask, dtype=np.bool_)


def extract_attributed_graphlet_simplex(
    graph: nx.Graph,
    *,
    graphlet_basis: GraphletBasis,
) -> tuple[np.ndarray, np.ndarray, AttributedGraphletCounts]:
    counts = extract_attributed_graphlet_counts(graph, graphlet_basis=graphlet_basis)
    probabilities, mask = attributed_graphlet_simplex_from_counts(
        counts,
        num_nodes=graph.number_of_nodes(),
        graphlet_basis=graphlet_basis,
    )
    return probabilities, mask, counts


def attributed_graphlet_simplex_to_clr(
    probabilities: Sequence[float] | np.ndarray,
    *,
    graphlet_basis: GraphletBasis,
    epsilon: float = 1.0e-5,
    coordinate_mask: Sequence[bool] | np.ndarray | None = None,
) -> np.ndarray:
    return _graphlet_simplex_to_clr(
        probabilities,
        graphlet_basis=graphlet_basis,  # type: ignore[arg-type]
        epsilon=epsilon,
        coordinate_mask=coordinate_mask,
    )


def attributed_graphlet_clr_to_simplex(
    logits: Sequence[float] | np.ndarray,
    *,
    graphlet_basis: GraphletBasis,
    coordinate_mask: Sequence[bool] | np.ndarray | None = None,
) -> np.ndarray:
    return _graphlet_clr_to_simplex(
        logits,
        graphlet_basis=graphlet_basis,  # type: ignore[arg-type]
        coordinate_mask=coordinate_mask,
    )


def attributed_graphlet_logit_distance(
    left_logits: Sequence[float] | np.ndarray,
    right_logits: Sequence[float] | np.ndarray,
    *,
    graphlet_basis: GraphletBasis,
    coordinate_mask: Sequence[bool] | np.ndarray | None = None,
    metric: str = "clr_rmse",
    size_weights: Mapping[str, float] | Sequence[float] | None = None,
) -> float:
    return _graphlet_logit_distance(
        left_logits,
        right_logits,
        graphlet_basis=graphlet_basis,  # type: ignore[arg-type]
        coordinate_mask=coordinate_mask,
        metric=metric,
        size_weights=size_weights,
    )


def _affected_subsets(
    graph: nx.Graph,
    candidate: nx.Graph,
    action: Action,
    k: int,
) -> set[tuple[int, ...]]:
    removed, added = action
    nodes = tuple(sorted(int(node) for node in graph.nodes()))
    affected: set[tuple[int, ...]] = set()
    for edge in tuple(removed) + tuple(added):
        u, v = int(edge[0]), int(edge[1])
        remaining = [node for node in nodes if node not in {u, v}]
        for extra in itertools.combinations(remaining, max(int(k) - 2, 0)):
            affected.add(tuple(sorted((u, v, *extra))))
    return affected


def candidate_attributed_graphlet_counts(
    graph: nx.Graph,
    candidate: nx.Graph,
    action: Action,
    *,
    current_counts: AttributedGraphletCounts,
    graphlet_basis: GraphletBasis,
) -> AttributedGraphletCounts:
    """Exact stateful local-delta update for an attributed double-edge swap."""

    _require_python_basis(graphlet_basis)
    result: AttributedGraphletCounts = {
        size: dict(current_counts.get(size, {})) for size in graphlet_basis.sizes
    }
    for size in graphlet_basis.sizes:
        k = int(size)
        delta: defaultdict[str, int] = defaultdict(int)
        for subset in _affected_subsets(graph, candidate, action, k):
            before = graph.subgraph(subset)
            if k <= 1 or nx.is_connected(before):
                before_key = _basis_key(
                    _canonical_key(graph, subset, graphlet_basis=graphlet_basis),
                    graphlet_basis,
                    size,
                )
                delta[before_key] -= 1
            after = candidate.subgraph(subset)
            if k <= 1 or nx.is_connected(after):
                after_key = _basis_key(
                    _canonical_key(candidate, subset, graphlet_basis=graphlet_basis),
                    graphlet_basis,
                    size,
                )
                delta[after_key] += 1
        updated = result[size]
        for key, value in delta.items():
            updated[key] = int(updated.get(key, 0) + value)
            if updated[key] < 0:
                raise AssertionError(
                    f"Attributed graphlet local delta produced a negative count for k={size}, key={key}."
                )
            if updated[key] == 0:
                updated.pop(key, None)
    return result


def candidate_attributed_graphlet_logits_from_counts(
    graph: nx.Graph,
    candidate: nx.Graph,
    action: Action,
    *,
    current_counts: AttributedGraphletCounts,
    graphlet_basis: GraphletBasis,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray, AttributedGraphletCounts]:
    counts = candidate_attributed_graphlet_counts(
        graph,
        candidate,
        action,
        current_counts=current_counts,
        graphlet_basis=graphlet_basis,
    )
    probabilities, mask = attributed_graphlet_simplex_from_counts(
        counts,
        num_nodes=candidate.number_of_nodes(),
        graphlet_basis=graphlet_basis,
    )
    logits = attributed_graphlet_simplex_to_clr(
        probabilities,
        graphlet_basis=graphlet_basis,
        epsilon=epsilon,
        coordinate_mask=mask,
    )
    return logits, probabilities, counts


def attributed_state_key(
    graph: nx.Graph,
    *,
    node_attribute: str,
    edge_attribute: str,
) -> tuple[Any, ...]:
    nodes = tuple(
        (int(node), graph.nodes[node].get(node_attribute))
        for node in sorted(graph.nodes())
    )
    edges = tuple(
        sorted(
            (
                min(int(u), int(v)),
                max(int(u), int(v)),
                data.get(edge_attribute),
            )
            for u, v, data in graph.edges(data=True)
        )
    )
    return nodes, edges


__all__ = [
    "AttributedGraphletCounts",
    "GraphletLogitBridgeSchedule",
    "attributed_graphlet_clr_to_simplex",
    "attributed_graphlet_logit_distance",
    "attributed_graphlet_simplex_from_counts",
    "attributed_graphlet_simplex_to_clr",
    "attributed_state_key",
    "candidate_attributed_graphlet_counts",
    "candidate_attributed_graphlet_logits_from_counts",
    "extract_attributed_graphlet_counts",
    "extract_attributed_graphlet_simplex",
]

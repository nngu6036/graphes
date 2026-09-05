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
from grapher.utils.motifs import (
    canonicalize_attributed_graph_python,
    canonicalize_attributed_simple_cycle,
    graphlet_topology_matches,
    induced_simple_cycle_node_sets,
)

AttributedGraphletCounts = dict[str, dict[str, int]]


def _require_python_basis(graphlet_basis: GraphletBasis) -> None:
    if not graphlet_basis.attributed:
        raise ValueError("Attributed graphlet diffusion requires an attributed basis.")
    if not graphlet_basis.node_attribute or not graphlet_basis.edge_attribute:
        raise ValueError(
            "Attributed graphlet basis is missing node/edge attribute names."
        )
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
    if graphlet_basis.topology_filter == "simple_cycle":
        return canonicalize_attributed_simple_cycle(
            graph.subgraph(tuple(int(node) for node in nodes)),
            node_label_attr=node_attribute,
            edge_label_attr=edge_attribute,
        )
    raw = _raw_pattern(
        graph,
        nodes,
        node_attribute=node_attribute,
        edge_attribute=edge_attribute,
    )
    return _canonical_key_from_raw(
        raw[0], raw[1], node_attribute, edge_attribute
    )


def _selected_key_or_none(
    graph: nx.Graph,
    nodes: Sequence[int],
    *,
    graphlet_basis: GraphletBasis,
    size: str,
) -> str | None:
    subgraph = graph.subgraph(tuple(int(node) for node in nodes))
    if (
        graphlet_basis.connected_only
        and len(nodes) > 1
        and not nx.is_connected(subgraph)
    ):
        return None
    if not graphlet_topology_matches(subgraph, graphlet_basis.topology_filter):
        return None
    return _basis_key(
        _canonical_key(graph, nodes, graphlet_basis=graphlet_basis),
        graphlet_basis,
        size,
    )


def _basis_key(key: str, graphlet_basis: GraphletBasis, size: str) -> str:
    known = set(graphlet_basis.keys_by_k[size])
    if key in known:
        return key
    if graphlet_basis.overflow_key is not None and graphlet_basis.overflow_key in known:
        return str(graphlet_basis.overflow_key)
    raise KeyError(
        "Attributed graphlet key is outside the fixed basis and no overflow "
        f"bin exists: {key}"
    )


def extract_attributed_graphlet_counts(
    graph: nx.Graph,
    *,
    graphlet_basis: GraphletBasis,
) -> AttributedGraphletCounts:
    """Count selected attributed graphlets exactly in the fixed vocabulary."""

    _require_python_basis(graphlet_basis)
    nodes = tuple(sorted(int(node) for node in graph.nodes()))
    counts_by_size: AttributedGraphletCounts = {}
    for size in graphlet_basis.sizes:
        k = int(size)
        block: defaultdict[str, int] = defaultdict(int)
        if len(nodes) >= k:
            subsets = (
                induced_simple_cycle_node_sets(graph, k)
                if graphlet_basis.topology_filter == "simple_cycle"
                else itertools.combinations(nodes, k)
            )
            for subset in subsets:
                key = _selected_key_or_none(
                    graph,
                    subset,
                    graphlet_basis=graphlet_basis,
                    size=size,
                )
                if key is not None:
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
        selected_mass = float(np.clip(block.sum(), 0.0, 1.0))
        full = np.concatenate(
            [block, np.asarray([max(0.0, 1.0 - selected_mass)], dtype=np.float64)]
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


def _connected_supersets_from_edge(
    graph: nx.Graph,
    edge: tuple[int, int],
    k: int,
) -> set[tuple[int, ...]]:
    """Enumerate connected ``k``-node sets containing one present edge."""

    u, v = int(edge[0]), int(edge[1])
    if not graph.has_edge(u, v):
        return set()
    subsets: set[frozenset[int]] = {frozenset((u, v))}
    for _size in range(3, int(k) + 1):
        expanded: set[frozenset[int]] = set()
        for subset in subsets:
            frontier: set[int] = set()
            for node in subset:
                frontier.update(int(value) for value in graph.neighbors(node))
            frontier.difference_update(subset)
            for node in frontier:
                expanded.add(subset | {node})
        subsets = expanded
        if not subsets:
            break
    return {tuple(sorted(subset)) for subset in subsets}


def _simple_cycle_subsets_containing_pair(
    graph: nx.Graph,
    pair: tuple[int, int],
    k: int,
) -> set[tuple[int, ...]]:
    """Enumerate induced ``C_k`` node sets containing both nodes in ``pair``.

    A changed pair can alter an induced ring in two ways: it can be a ring
    edge, or it can become/remove a chord.  Searching only for rings that use
    a changed *edge* misses the second case.  This bounded DFS enumerates
    simple length-``k`` cycles through one endpoint, retains those containing
    the other endpoint, and then checks inducedness exactly.
    """

    u, v = int(pair[0]), int(pair[1])
    k = int(k)
    if k < 3 or u == v or u not in graph or v not in graph:
        return set()
    found: set[tuple[int, ...]] = set()

    def visit(current: int, path: tuple[int, ...]) -> None:
        if len(path) == k:
            if v not in path or not graph.has_edge(current, u):
                return
            subset = tuple(sorted(path))
            if graphlet_topology_matches(graph.subgraph(subset), "simple_cycle"):
                found.add(subset)
            return

        remaining_nodes = k - len(path)
        contains_v = v in path
        for neighbor_raw in graph.neighbors(current):
            neighbor = int(neighbor_raw)
            if neighbor == u or neighbor in path:
                continue
            # When only one slot remains, it must include the second endpoint.
            if not contains_v and remaining_nodes == 1 and neighbor != v:
                continue
            visit(neighbor, path + (neighbor,))

    visit(u, (u,))
    return found


def _affected_subsets(
    graph: nx.Graph,
    candidate: nx.Graph,
    action: Action,
    k: int,
    *,
    connected_only: bool,
    topology_filter: str,
) -> set[tuple[int, ...]]:
    removed, added = action
    if topology_filter == "simple_cycle":
        affected: set[tuple[int, ...]] = set()
        changed_pairs = set(tuple(edge) for edge in tuple(removed) + tuple(added))
        for pair in changed_pairs:
            # Search both states. This covers removed/inserted ring edges and
            # rings created or destroyed by removing/inserting a chord.
            affected.update(
                _simple_cycle_subsets_containing_pair(graph, pair, k)
            )
            affected.update(
                _simple_cycle_subsets_containing_pair(candidate, pair, k)
            )
        return affected
    if connected_only:
        affected: set[tuple[int, ...]] = set()
        for edge in removed:
            affected.update(_connected_supersets_from_edge(graph, edge, k))
        for edge in added:
            affected.update(_connected_supersets_from_edge(candidate, edge, k))
        return affected

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
        for subset in _affected_subsets(
            graph,
            candidate,
            action,
            k,
            connected_only=graphlet_basis.connected_only,
            topology_filter=graphlet_basis.topology_filter,
        ):
            before_key = _selected_key_or_none(
                graph,
                subset,
                graphlet_basis=graphlet_basis,
                size=size,
            )
            if before_key is not None:
                delta[before_key] -= 1
            after_key = _selected_key_or_none(
                candidate,
                subset,
                graphlet_basis=graphlet_basis,
                size=size,
            )
            if after_key is not None:
                delta[after_key] += 1
        updated = result[size]
        for key, value in delta.items():
            updated[key] = int(updated.get(key, 0) + value)
            if updated[key] < 0:
                raise AssertionError(
                    "Attributed graphlet local delta produced a negative count "
                    f"for k={size}, key={key}."
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

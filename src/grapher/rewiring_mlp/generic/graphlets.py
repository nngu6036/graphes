from __future__ import annotations

from collections import Counter
from functools import lru_cache
from math import comb
from typing import Any

import networkx as nx
import numpy as np

from grapher.properties.summary import SummaryConfig, clustering_histogram
from grapher.rewiring_mlp.core.rewiring import Action
from grapher.utils.motifs import (
    connected_graphlet_count_dict_exact,
    default_topology_canonicalizer,
    topology_graphlet_basis,
)
from grapher.rewiring_mlp.generic.basis import TopologyGraphletBasis


TopologyGraphletCounts = dict[str, dict[str, int]]
TOPOLOGY_ORBIT_WIDTH = 15


_ORBIT_ROLES_BY_SIGNATURE: dict[
    tuple[int, int, tuple[int, ...]], tuple[tuple[int, int], ...]
] = {
    (3, 2, (1, 1, 2)): ((1, 2), (2, 1)),
    (3, 3, (2, 2, 2)): ((3, 3),),
    (4, 3, (1, 1, 2, 2)): ((4, 2), (5, 2)),
    (4, 3, (1, 1, 1, 3)): ((6, 3), (7, 1)),
    (4, 4, (2, 2, 2, 2)): ((8, 4),),
    (4, 4, (1, 2, 2, 3)): ((9, 1), (10, 2), (11, 1)),
    (4, 5, (2, 2, 3, 3)): ((12, 2), (13, 2)),
    (4, 6, (3, 3, 3, 3)): ((14, 4),),
}


@lru_cache(maxsize=2)
def _orbit_roles_by_graphlet_key(k: int) -> dict[str, tuple[tuple[int, int], ...]]:
    """Return standard ORCA orbit multiplicities for connected k-graphlets."""

    k = int(k)
    if k not in {3, 4}:
        raise ValueError("Topology orbit summaries require graphlet sizes 3 and 4.")
    result: dict[str, tuple[tuple[int, int], ...]] = {}
    for key, representative in topology_graphlet_basis(k, connected_only=True):
        signature = (
            k,
            representative.number_of_edges(),
            tuple(sorted(int(degree) for _, degree in representative.degree())),
        )
        roles = _ORBIT_ROLES_BY_SIGNATURE.get(signature)
        if roles is None:
            raise RuntimeError(
                f"No standard orbit-role mapping for k={k}, graphlet={key!r}."
            )
        result[str(key)] = roles
    return result


def extract_topology_graphlet_counts(
    graph: nx.Graph,
    *,
    graphlet_basis: TopologyGraphletBasis,
) -> TopologyGraphletCounts:
    """Count all connected topology graphlets in the fixed complete basis."""

    if graphlet_basis.attributed or not graphlet_basis.connected_only:
        raise ValueError(
            "The decoupled topology stage requires connected, unattributed "
            "graphlets."
        )
    canonicalizer = default_topology_canonicalizer()
    return {
        key: connected_graphlet_count_dict_exact(
            graph,
            int(key),
            canonicalizer=canonicalizer,
        )
        for key in graphlet_basis.sizes
    }


def _target_from_counts(
    counts_by_size: TopologyGraphletCounts,
    *,
    num_nodes: int,
    graphlet_basis: TopologyGraphletBasis,
) -> tuple[np.ndarray, np.ndarray]:
    history: dict[str, dict[str, float]] = {}
    masses: list[float] = []
    for key in graphlet_basis.sizes:
        counts = counts_by_size.get(key, {})
        total_connected = int(sum(int(value) for value in counts.values()))
        history[key] = {
            graphlet_key: (
                float(counts.get(graphlet_key, 0)) / total_connected
                if total_connected > 0
                else 0.0
            )
            for graphlet_key in graphlet_basis.keys_by_k[key]
        }
        total_subsets = comb(int(num_nodes), int(key)) if num_nodes >= int(key) else 0
        masses.append(
            float(total_connected / total_subsets) if total_subsets > 0 else 0.0
        )
    return (
        graphlet_basis.flatten_history(history).astype(np.float64),
        np.asarray(masses, dtype=np.float64),
    )


def extract_topology_graphlet_target(
    graph: nx.Graph,
    *,
    graphlet_basis: TopologyGraphletBasis,
    summary_config: SummaryConfig | dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Extract one cached target used by both teacher scoring and supervision."""

    del summary_config
    counts = extract_topology_graphlet_counts(
        graph,
        graphlet_basis=graphlet_basis,
    )
    return _target_from_counts(
        counts,
        num_nodes=graph.number_of_nodes(),
        graphlet_basis=graphlet_basis,
    )


def topology_orbit_count_vector_from_counts(
    counts_by_size: TopologyGraphletCounts,
    *,
    num_nodes: int,
    num_edges: int,
) -> np.ndarray:
    """Return the mean per-node ORCA 0--14 orbit descriptor from graphlet counts.

    The descriptor is computed from the same exact connected induced graphlet
    counts used by the topology teacher. This avoids launching ORCA for every
    candidate while retaining the standard four-node orbit convention.
    """

    n = int(num_nodes)
    output = np.zeros(TOPOLOGY_ORBIT_WIDTH, dtype=np.float64)
    if n <= 0:
        return output
    output[0] = 2.0 * float(num_edges) / float(n)
    for k in (3, 4):
        counts = counts_by_size.get(str(k))
        if counts is None:
            raise ValueError(
                "Orbit targets require graphlet_prediction to include sizes 3 and 4."
            )
        roles_by_key = _orbit_roles_by_graphlet_key(k)
        for key, count in counts.items():
            roles = roles_by_key.get(str(key))
            if roles is None:
                raise ValueError(
                    f"Unknown connected topology graphlet key for k={k}: {key!r}."
                )
            for orbit, multiplicity in roles:
                output[int(orbit)] += float(count) * float(multiplicity) / float(n)
    return output


def extract_topology_structural_target(
    graph: nx.Graph,
    *,
    graphlet_basis: TopologyGraphletBasis,
    summary_config: SummaryConfig | dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract graphlet, clustering, and orbit targets from one terminal graph."""

    cfg = (
        summary_config
        if isinstance(summary_config, SummaryConfig)
        else SummaryConfig.from_dict(summary_config or {})
    )
    counts = extract_topology_graphlet_counts(
        graph,
        graphlet_basis=graphlet_basis,
    )
    graphlet, mass = _target_from_counts(
        counts,
        num_nodes=graph.number_of_nodes(),
        graphlet_basis=graphlet_basis,
    )
    clustering = (
        clustering_histogram(graph, cfg.clustering_bins)
        if cfg.clustering_summary
        else np.zeros(0, dtype=np.float64)
    )
    orbit = (
        topology_orbit_count_vector_from_counts(
            counts,
            num_nodes=graph.number_of_nodes(),
            num_edges=graph.number_of_edges(),
        )
        if cfg.orbit_count
        else np.zeros(0, dtype=np.float64)
    )
    return graphlet, mass, clustering, orbit


def _normalized_l2(left: Any, right: Any) -> float:
    left_array = np.asarray(left, dtype=np.float64).reshape(-1)
    right_array = np.asarray(right, dtype=np.float64).reshape(-1)
    if left_array.shape != right_array.shape:
        raise ValueError(
            "Topology structural target width mismatch: "
            f"{left_array.shape} != {right_array.shape}."
        )
    if left_array.size == 0:
        return 0.0
    return float(
        np.linalg.norm(left_array - right_array) / np.sqrt(left_array.size)
    )


def topology_structural_discrepancy_from_counts(
    graph: nx.Graph,
    counts_by_size: TopologyGraphletCounts,
    *,
    graphlet_target: np.ndarray,
    graphlet_mass_target: np.ndarray,
    clustering_target: np.ndarray,
    orbit_target: np.ndarray,
    graphlet_basis: TopologyGraphletBasis,
    graphlet_weight: float = 1.0,
    graphlet_mass_weight: float = 0.0,
    clustering_weight: float = 0.0,
    orbit_weight: float = 0.0,
) -> dict[str, float]:
    """Score a graph against one frozen graph-level structural target."""

    graphlet_total, graphlet_histogram, graphlet_mass = (
        topology_graphlet_discrepancy_from_counts(
            counts_by_size,
            num_nodes=graph.number_of_nodes(),
            target=graphlet_target,
            target_mass=graphlet_mass_target,
            graphlet_basis=graphlet_basis,
            mass_weight=graphlet_mass_weight,
        )
    )
    clustering_distance = 0.0
    if float(clustering_weight) != 0.0:
        expected_clustering = np.asarray(
            clustering_target, dtype=np.float64
        ).reshape(-1)
        clustering_distance = _normalized_l2(
            clustering_histogram(graph, expected_clustering.size),
            expected_clustering,
        )
    orbit_distance = 0.0
    if float(orbit_weight) != 0.0:
        current_orbit = topology_orbit_count_vector_from_counts(
            counts_by_size,
            num_nodes=graph.number_of_nodes(),
            num_edges=graph.number_of_edges(),
        )
        expected_orbit = np.asarray(orbit_target, dtype=np.float64).reshape(-1)
        orbit_distance = _normalized_l2(
            np.log1p(np.maximum(current_orbit, 0.0)),
            np.log1p(np.maximum(expected_orbit, 0.0)),
        )
    total = (
        float(graphlet_weight) * float(graphlet_total)
        + float(clustering_weight) * float(clustering_distance)
        + float(orbit_weight) * float(orbit_distance)
    )
    return {
        "total": float(total),
        "graphlet": float(graphlet_total),
        "graphlet_histogram": float(graphlet_histogram),
        "graphlet_mass": float(graphlet_mass),
        "clustering": float(clustering_distance),
        "orbit": float(orbit_distance),
    }


def topology_graphlet_discrepancy_from_counts(
    counts_by_size: TopologyGraphletCounts,
    *,
    num_nodes: int,
    target: np.ndarray,
    target_mass: np.ndarray,
    graphlet_basis: TopologyGraphletBasis,
    mass_weight: float,
) -> tuple[float, float, float]:
    current, current_mass = _target_from_counts(
        counts_by_size,
        num_nodes=num_nodes,
        graphlet_basis=graphlet_basis,
    )
    expected = np.asarray(target, dtype=np.float64).reshape(-1)
    expected_mass = np.asarray(target_mass, dtype=np.float64).reshape(-1)
    if current.shape != expected.shape:
        raise ValueError("Graphlet target width does not match the configured basis.")
    if current_mass.shape != expected_mass.shape:
        raise ValueError("Graphlet connected-mass target width is inconsistent.")

    block_distances = [
        float(np.linalg.norm(current[start:stop] - expected[start:stop]))
        for start, stop in graphlet_basis.slices
        if stop > start
    ]
    histogram_distance = (
        float(np.mean(block_distances)) if block_distances else 0.0
    )
    mass_distance = (
        float(np.mean(np.abs(current_mass - expected_mass)))
        if current_mass.size
        else 0.0
    )
    total = histogram_distance + float(mass_weight) * mass_distance
    return float(total), histogram_distance, mass_distance


def _connected_supersets_from_edge(
    graph: nx.Graph,
    edge: tuple[int, int],
    k: int,
) -> set[frozenset[int]]:
    """Enumerate connected k-sets containing an edge by local expansion."""

    if not graph.has_edge(*edge):
        return set()
    subsets: set[frozenset[int]] = {frozenset((int(edge[0]), int(edge[1])))}
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
    return subsets if int(k) >= 2 else set()


def candidate_topology_graphlet_counts(
    graph: nx.Graph,
    candidate: nx.Graph,
    action: Action,
    *,
    current_counts: TopologyGraphletCounts,
    graphlet_basis: TopologyGraphletBasis,
) -> TopologyGraphletCounts:
    """Update exact graphlet counts using only switch-affected local subsets."""

    removed, added = action
    canonicalizer = default_topology_canonicalizer()
    result: TopologyGraphletCounts = {
        key: dict(counts) for key, counts in current_counts.items()
    }
    for key in graphlet_basis.sizes:
        k = int(key)
        affected: set[frozenset[int]] = set()
        for edge in removed:
            affected.update(_connected_supersets_from_edge(graph, edge, k))
        for edge in added:
            affected.update(_connected_supersets_from_edge(candidate, edge, k))
        delta: Counter[str] = Counter()
        for subset in affected:
            before = graph.subgraph(subset)
            after = candidate.subgraph(subset)
            if nx.is_connected(before):
                delta[canonicalizer.canonical_graph6(before)] -= 1
            if nx.is_connected(after):
                delta[canonicalizer.canonical_graph6(after)] += 1
        updated = Counter(result.get(key, {}))
        updated.update(delta)
        if any(value < 0 for value in updated.values()):
            raise AssertionError("Local graphlet delta produced a negative count.")
        result[key] = {
            graphlet_key: int(value)
            for graphlet_key, value in updated.items()
            if int(value) > 0
        }
    return result


def topology_candidate_graphlet_discrepancy(
    graph: nx.Graph,
    candidate: nx.Graph,
    action: Action,
    target: np.ndarray,
    target_mass: np.ndarray,
    *,
    current_counts: TopologyGraphletCounts,
    graphlet_basis: TopologyGraphletBasis,
    mass_weight: float = 0.0,
) -> tuple[float, float, float]:
    counts = candidate_topology_graphlet_counts(
        graph,
        candidate,
        action,
        current_counts=current_counts,
        graphlet_basis=graphlet_basis,
    )
    return topology_graphlet_discrepancy_from_counts(
        counts,
        num_nodes=candidate.number_of_nodes(),
        target=target,
        target_mass=target_mass,
        graphlet_basis=graphlet_basis,
        mass_weight=mass_weight,
    )


def topology_graphlet_discrepancy(
    graph: nx.Graph,
    target: np.ndarray,
    target_mass: np.ndarray,
    *,
    graphlet_basis: TopologyGraphletBasis,
    summary_config: SummaryConfig | dict[str, Any],
    mass_weight: float = 0.0,
) -> tuple[float, float, float]:
    """Return total, histogram, and connected-mass discrepancies.

    Reinitializing the subset sampler with the same seed gives the current
    state and every same-step candidate an identical subset plan. This is the
    common-random-numbers comparison required when graphlet counts are sampled.
    """

    del summary_config
    counts = extract_topology_graphlet_counts(
        graph,
        graphlet_basis=graphlet_basis,
    )
    return topology_graphlet_discrepancy_from_counts(
        counts,
        num_nodes=graph.number_of_nodes(),
        target=target,
        target_mass=target_mass,
        graphlet_basis=graphlet_basis,
        mass_weight=mass_weight,
    )

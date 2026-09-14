#!/usr/bin/env python3
"""Regression/benchmark checks for the exact attributed graphlet fast path.

This script does not require QM9 files. It creates deterministic QM9-like
attributed graphs, compares the optimized multi-k counter with an independent
NetworkX-subgraph reference, and verifies local swap deltas by full recounting.
"""
from __future__ import annotations

import argparse
from collections import Counter
import itertools
import random
import time

import networkx as nx

from grapher.rewiring_mlp.attributed.data import GraphletBasis
from grapher.rewiring_mlp.attributed.graphlet_diffusion import (
    candidate_attributed_graphlet_counts,
    extract_attributed_graphlet_counts,
)
from grapher.rewiring_mlp.core.rewiring import make_action
from grapher.utils.motifs import (
    attributed_graphlet_count_dict_multi,
    canonicalize_attributed_graph_python,
)


def make_graphs(count: int, seed: int) -> list[nx.Graph]:
    rng = random.Random(seed)
    graphs: list[nx.Graph] = []
    for _ in range(count):
        n = rng.randint(5, 9)
        graph = nx.Graph()
        for node in range(n):
            graph.add_node(node, atomic_num=rng.choice([6, 6, 6, 7, 8, 9]))
        for node in range(1, n):
            graph.add_edge(
                node,
                rng.randrange(node),
                bond_type=rng.choice([1, 1, 1, 2, 3]),
            )
        for left in range(n):
            for right in range(left + 1, n):
                if graph.has_edge(left, right) or rng.random() >= 0.05:
                    continue
                graph.add_edge(left, right, bond_type=rng.choice([1, 1, 2, 3]))
        graphs.append(graph)
    return graphs


def reference_counts(graph: nx.Graph, k: int) -> dict[str, int]:
    counts: Counter[str] = Counter()
    nodes = sorted(graph.nodes())
    for subset in itertools.combinations(nodes, k):
        key = canonicalize_attributed_graph_python(
            graph.subgraph(subset),
            node_label_attr="atomic_num",
            edge_label_attr="bond_type",
        )
        counts[key] += 1
    return dict(counts)


def verify_counts(graphs: list[nx.Graph]) -> None:
    for index, graph in enumerate(graphs):
        optimized = attributed_graphlet_count_dict_multi(
            graph,
            (3, 4, 5),
            node_label_attr="atomic_num",
            edge_label_attr="bond_type",
            connected_only=False,
            topology_filter="all",
            backend="python",
        )
        for k in (3, 4, 5):
            expected = reference_counts(graph, k) if graph.number_of_nodes() >= k else {}
            if optimized[k] != expected:
                raise AssertionError(f"graph={index} k={k}: optimized counts differ")


def verify_local_delta() -> None:
    graph = nx.Graph()
    for node, atom in enumerate([6, 6, 7, 8, 6, 9, 6, 7, 8]):
        graph.add_node(node, atomic_num=atom)
    for left, right, bond in [
        (0, 1, 1), (2, 3, 1), (1, 4, 1), (4, 5, 2), (5, 6, 1),
        (6, 7, 1), (7, 8, 2), (3, 8, 1), (0, 6, 1),
    ]:
        graph.add_edge(left, right, bond_type=bond)
    action = make_action([(0, 1), (2, 3)], [(0, 2), (1, 3)])
    candidate = graph.copy()
    candidate.remove_edge(0, 1)
    candidate.remove_edge(2, 3)
    candidate.add_edge(0, 2, bond_type=1)
    candidate.add_edge(1, 3, bond_type=1)
    settings = {
        "graphlet_history": True,
        "graphlet_k_min": 3,
        "graphlet_k_max": 5,
        "graphlet_connected_only": False,
        "graphlet_topology_filter": "all",
        "graphlet_num_samples": None,
        "attributed": True,
        "node_attribute": "atomic_num",
        "edge_attribute": "bond_type",
        "attributed_backend": "python",
    }
    basis = GraphletBasis.fit_from_graphs(
        [graph, candidate], settings, attributed=True, seed=42
    )
    current = extract_attributed_graphlet_counts(graph, graphlet_basis=basis)
    updated = candidate_attributed_graphlet_counts(
        graph,
        candidate,
        action,
        current_counts=current,
        graphlet_basis=basis,
    )
    expected = extract_attributed_graphlet_counts(candidate, graphlet_basis=basis)
    if updated != expected:
        raise AssertionError("local attributed graphlet delta differs from full recount")


def benchmark(graphs: list[nx.Graph]) -> float:
    started = time.perf_counter()
    subsets = 0
    for graph in graphs:
        counts = attributed_graphlet_count_dict_multi(
            graph,
            (3, 4, 5),
            node_label_attr="atomic_num",
            edge_label_attr="bond_type",
            connected_only=False,
            topology_filter="all",
            backend="python",
        )
        subsets += sum(sum(block.values()) for block in counts.values())
    elapsed = time.perf_counter() - started
    print(
        f"optimized exact multi-k: graphs={len(graphs)} subsets={subsets} "
        f"seconds={elapsed:.4f} subsets_per_second={subsets/max(elapsed,1e-12):.1f}"
    )
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    graphs = make_graphs(args.graphs, args.seed)
    verify_counts(graphs)
    verify_local_delta()
    benchmark(graphs)
    print("graphlet optimization validation: PASS")


if __name__ == "__main__":
    main()

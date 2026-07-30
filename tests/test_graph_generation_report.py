from __future__ import annotations

import networkx as nx

from scripts.evaluate_graph_generation_report import select_sample_indices


def test_stratified_samples_cover_graph_size_range() -> None:
    graphs = [nx.path_graph(size) for size in range(4, 14)]
    indices = select_sample_indices(
        graphs,
        3,
        selection="stratified",
        seed=42,
    )
    assert indices == [0, 4, 9]


def test_random_samples_are_reproducible_and_unique() -> None:
    graphs = [nx.path_graph(size) for size in range(4, 14)]
    first = select_sample_indices(graphs, 5, selection="random", seed=7)
    second = select_sample_indices(graphs, 5, selection="random", seed=7)
    assert first == second
    assert len(first) == len(set(first)) == 5

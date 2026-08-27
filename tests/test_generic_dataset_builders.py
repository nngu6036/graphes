from __future__ import annotations

import networkx as nx
import pytest

from grapher.data.builders import EgoDatasetBuilder, build_sbm_graphs, split_graphs


def _community_test_config() -> dict:
    return {
        "name": "sbm",
        "type": "sbm",
        "seed": 7,
        "num_graphs": 8,
        "communities": {
            "min_blocks": 2,
            "max_blocks": 2,
            "min_total_nodes": 12,
            "max_total_nodes": 20,
            "equal_block_sizes": True,
        },
        "edge_probs": {
            "p_in": 1.0,
            "p_out": 0.0,
            "p_inter": 0.05,
            "ensure_inter_community_edge": True,
        },
        "filters": {
            "require_connected": True,
            "reject_zero_degree": True,
            "max_attempts_per_graph": 20,
        },
    }


def test_community_declared_total_size_and_inter_edge_keys_are_honored() -> None:
    graphs = build_sbm_graphs(_community_test_config())

    assert len(graphs) == 8
    for graph in graphs:
        assert graph.number_of_nodes() in {12, 14, 16, 18, 20}
        assert nx.is_connected(graph)
        communities = nx.get_node_attributes(graph, "community")
        inter_edges = [
            (u, v) for u, v in graph.edges() if communities[u] != communities[v]
        ]
        assert len(inter_edges) >= 1
        left_size, right_size = graph.graph["block_sizes"]
        assert graph.graph["p_inter"] == 0.05
        assert graph.graph["p_inter_edge_probability"] == pytest.approx(
            0.05 * graph.number_of_nodes() / (left_size * right_size)
        )


def test_community_generation_is_deterministic_for_dataset_seed() -> None:
    left = build_sbm_graphs(_community_test_config())
    right = build_sbm_graphs(_community_test_config())

    assert [set(graph.edges()) for graph in left] == [
        set(graph.edges()) for graph in right
    ]


def test_community_protocol_splits_100_graphs_as_70_10_20() -> None:
    splits = split_graphs(
        list(range(100)),
        {
            "seed": 0,
            "split": {"train": 0.7, "val": 0.1, "test": 0.2},
        },
    )

    assert {name: len(values) for name, values in splits.items()} == {
        "train": 70,
        "val": 10,
        "test": 20,
    }


def test_ego_largest_component_and_first_selection_are_honored(monkeypatch) -> None:
    source = nx.Graph()
    source.add_edges_from([(100, 101), (101, 102), (102, 100)])
    source.add_edges_from(nx.cycle_graph(range(10, 16)).edges())
    config = {
        "name": "ego_small",
        "seed": 0,
        "num_graphs": 2,
        "ego": {
            "largest_connected_component": True,
            "selection": "first",
            "sample_with_replacement": False,
            "radius": 1,
            "min_nodes": 3,
            "max_nodes": 3,
        },
    }
    builder = EgoDatasetBuilder(config)
    monkeypatch.setattr(
        builder,
        "_build_source_graph",
        lambda _cfg, _rng: source.copy(),
    )

    graphs = builder.build_graphs()

    assert [graph.graph["center"] for graph in graphs] == [0, 1]
    assert [graph.number_of_nodes() for graph in graphs] == [3, 3]
    assert all(nx.is_connected(graph) for graph in graphs)


def test_ego_rejects_unknown_selection(monkeypatch) -> None:
    builder = EgoDatasetBuilder(
        {
            "num_graphs": 1,
            "ego": {
                "selection": "unsupported",
                "min_nodes": 1,
                "max_nodes": 2,
            },
        }
    )
    monkeypatch.setattr(
        builder,
        "_build_source_graph",
        lambda _cfg, _rng: nx.path_graph(2),
    )

    with pytest.raises(ValueError, match="ego.selection"):
        builder.build_graphs()

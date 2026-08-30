from __future__ import annotations

import networkx as nx
import pytest

from grapher.data.builders import (
    EgoDatasetBuilder,
    build_sbm_graphs,
    load_precomputed_graphs,
    split_graphs,
)


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


def test_common_protocol_splits_100_graphs_as_64_16_20() -> None:
    splits = split_graphs(
        list(range(100)),
        {
            "seed": 0,
            "split": {"seed": 0, "train": 0.64, "val": 0.16, "test": 0.2},
        },
    )

    assert {name: len(values) for name, values in splits.items()} == {
        "train": 64,
        "val": 16,
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


def test_precomputed_networkx_pickle_source_is_pinned(tmp_path) -> None:
    import pickle

    source = tmp_path / "ego.pkl"
    graphs = [nx.path_graph(4), nx.cycle_graph(5)]
    with source.open("wb") as handle:
        pickle.dump(graphs, handle)

    loaded = load_precomputed_graphs(
        {
            "name": "ego_small",
            "num_graphs": 2,
            "source": {
                "kind": "networkx_pickle",
                "path": str(source),
                "expected_graphs": 2,
            },
        }
    )

    assert loaded is not None
    assert [g.number_of_nodes() for g in loaded] == [4, 5]
    assert [g.graph["source_index"] for g in loaded] == [0, 1]


def test_split_seed_can_be_declared_inside_split_block() -> None:
    graphs = list(range(20))
    left = split_graphs(
        graphs,
        {"seed": 999, "split": {"seed": 7, "train": 0.6, "val": 0.2, "test": 0.2}},
    )
    right = split_graphs(
        graphs,
        {"seed": 1, "split": {"seed": 7, "train": 0.6, "val": 0.2, "test": 0.2}},
    )
    assert left == right


def test_precomputed_spectre_pt_source_reads_first_adjacency_payload(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    source = tmp_path / "community.pt"
    adjs = [
        torch.tensor(nx.to_numpy_array(nx.path_graph(4)), dtype=torch.float32),
        torch.tensor(nx.to_numpy_array(nx.cycle_graph(5)), dtype=torch.float32),
    ]
    torch.save((adjs, None, None, None, None, None, None, None), source)

    loaded = load_precomputed_graphs(
        {
            "name": "sbm",
            "num_graphs": 2,
            "source": {
                "kind": "spectre_pt",
                "path": str(source),
                "expected_graphs": 2,
            },
        }
    )

    assert loaded is not None
    assert [g.number_of_nodes() for g in loaded] == [4, 5]
    assert [g.number_of_edges() for g in loaded] == [3, 5]


def test_ego_small_config_does_not_alias_edge_ego_dataset() -> None:
    from pathlib import Path
    import yaml

    config_path = Path(__file__).resolve().parents[1] / "configs" / "datasets" / "ego_small.yaml"
    config = yaml.safe_load(config_path.read_text())

    assert config["num_graphs"] == 200
    assert "source" not in config
    ego = config["ego"]
    assert ego["source_model"] == "citeseer"
    assert ego["radius"] == 1
    assert ego["min_nodes"] == 4
    assert ego["max_nodes"] == 18
    assert ego["selection"] == "first"
    assert config["native_reference"]["defog"]["native_ego_graphs"] == 757

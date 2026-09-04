from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from grapher.models.gdss.codec import export_dataset, load_generated_export, profile_for


def _molecule(atom_numbers: list[int], edges: list[tuple[int, int, int]]) -> nx.Graph:
    graph = nx.Graph()
    for node, atomic_number in enumerate(atom_numbers):
        graph.add_node(node, atomic_num=atomic_number, atom_type=atomic_number)
    for u, v, bond_type in edges:
        graph.add_edge(u, v, bond_type=bond_type, bond_order=float(bond_type))
    return graph


def test_generic_export_keeps_exact_train_val_test_separate(tmp_path: Path) -> None:
    profile = profile_for("community_small")
    manifest = export_dataset(
        train_graphs=[nx.path_graph(4), nx.cycle_graph(5)],
        val_graphs=[nx.complete_graph(3)],
        test_graphs=[nx.star_graph(3)],
        profile=profile,
        output_dir=tmp_path,
    )
    with np.load(tmp_path / "train.npz", allow_pickle=False) as payload:
        assert payload["adjacency"].shape == (2, 20, 20)
        assert payload["num_nodes"].tolist() == [4, 5]
    with np.load(tmp_path / "val.npz", allow_pickle=False) as payload:
        assert payload["num_nodes"].tolist() == [3]
    with np.load(tmp_path / "test.npz", allow_pickle=False) as payload:
        assert payload["num_nodes"].tolist() == [4]
    assert manifest["training_projection"] == {
        "optimizer_split": "train",
        "monitor_split": "val",
        "test_used_during_training": False,
    }
    assert json.loads((tmp_path / "manifest.json").read_text())["format"] == "grapher_gdss_dataset_v1"


def test_molecular_export_uses_gdss_atom_channels_and_integer_bond_state(tmp_path: Path) -> None:
    profile = profile_for("qm9")
    graph = _molecule([6, 8, 7], [(0, 1, 2), (1, 2, 1)])
    export_dataset(
        train_graphs=[graph], val_graphs=[], test_graphs=[], profile=profile, output_dir=tmp_path
    )
    with np.load(tmp_path / "train.npz", allow_pickle=False) as payload:
        assert payload["node_types"][0, :3].tolist() == [0, 2, 1]
        assert payload["node_types"][0, 3:].tolist() == [-1] * 6
        assert int(payload["adjacency"][0, 0, 1]) == 2
        assert int(payload["adjacency"][0, 1, 2]) == 1


def test_molecular_export_rejects_aromatic_category_not_supported_by_release(tmp_path: Path) -> None:
    profile = profile_for("qm9")
    graph = _molecule([6, 6], [(0, 1, 4)])
    with pytest.raises(ValueError, match="expected 1, 2, or 3"):
        export_dataset(
            train_graphs=[graph], val_graphs=[], test_graphs=[], profile=profile, output_dir=tmp_path
        )


def test_generated_generic_export_decodes_native_isolate_removed_state(tmp_path: Path) -> None:
    profile = profile_for("ego_small")
    adjacency = np.zeros((2, 18, 18), dtype=np.int8)
    adjacency[0, 0, 1] = adjacency[0, 1, 0] = 1
    adjacency[1, :3, :3] = nx.to_numpy_array(nx.complete_graph(3), dtype=np.int8)
    np.fill_diagonal(adjacency[1], 0)
    path = tmp_path / "generated.npz"
    np.savez_compressed(
        path,
        adjacency=adjacency,
        num_nodes=np.asarray([2, 3], dtype=np.int64),
        sample_index=np.arange(2, dtype=np.int64),
    )
    graphs = load_generated_export(path, profile=profile)
    assert [g.number_of_edges() for g in graphs] == [1, 3]
    assert [g.graph["gdss_sample_index"] for g in graphs] == [0, 1]


def test_generated_molecular_export_is_raw_and_allows_empty_invalid_sample(tmp_path: Path) -> None:
    profile = profile_for("qm9")
    adjacency = np.zeros((2, 9, 9), dtype=np.int8)
    adjacency[0, 0, 1] = adjacency[0, 1, 0] = 3
    node_types = np.full((2, 9), -1, dtype=np.int16)
    node_types[0, :2] = [0, 2]
    path = tmp_path / "molecules.npz"
    np.savez_compressed(
        path,
        adjacency=adjacency,
        node_types=node_types,
        num_nodes=np.asarray([2, 0], dtype=np.int64),
        sample_index=np.arange(2, dtype=np.int64),
    )
    graphs = load_generated_export(path, profile=profile)
    assert graphs[0].nodes[0]["atomic_num"] == 6
    assert graphs[0].nodes[1]["atomic_num"] == 8
    assert graphs[0].edges[0, 1]["bond_type"] == 3
    assert graphs[1].number_of_nodes() == 0

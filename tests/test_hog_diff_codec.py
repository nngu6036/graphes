from __future__ import annotations

import json
import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from grapher.models.hog_diff.codec import (
    export_generic_dataset,
    export_molecular_dataset,
    load_generated_export,
    profile_for,
)


def _molecule(atom_numbers: list[int], edges: list[tuple[int, int, int]]) -> nx.Graph:
    graph = nx.Graph()
    for node, atomic_number in enumerate(atom_numbers):
        graph.add_node(node, atomic_num=atomic_number, atom_type=atomic_number)
    for u, v, bond_type in edges:
        graph.add_edge(u, v, bond_type=bond_type, bond_order=float(bond_type))
    return graph


def test_generic_projection_uses_test_then_train_and_excludes_validation(tmp_path: Path) -> None:
    profile = profile_for("community_small")
    train = [nx.path_graph(4), nx.cycle_graph(5)]
    val = [nx.complete_graph(3)]
    test = [nx.star_graph(3)]
    manifest = export_generic_dataset(
        train_graphs=train,
        val_graphs=val,
        test_graphs=test,
        profile=profile,
        output_dir=tmp_path,
    )
    raw_path = tmp_path / "community_small" / "community_small.pkl"
    with raw_path.open("rb") as handle:
        upstream = pickle.load(handle)
    assert len(upstream) == 3
    assert nx.is_isomorphic(upstream[0], test[0])
    assert nx.is_isomorphic(upstream[1], train[0])
    assert nx.is_isomorphic(upstream[2], train[1])
    assert int(manifest["upstream_training_projection"]["test_split"] * len(upstream)) == 1
    assert manifest["upstream_training_projection"]["validation_excluded"] is True
    assert json.loads((tmp_path / "manifest.json").read_text())["format"] == "grapher_hogdiff_dataset_v1"


def test_molecular_projection_matches_hogdiff_atom_and_bond_encoding(tmp_path: Path) -> None:
    profile = profile_for("qm9")
    graph = _molecule([6, 8, 7], [(0, 1, 2), (1, 2, 1)])
    manifest = export_molecular_dataset(
        train_graphs=[graph],
        val_graphs=[],
        test_graphs=[],
        profile=profile,
        output_dir=tmp_path,
    )
    neutral = tmp_path / "qm9" / "processed" / "grapher_atom_bond.npz"
    with np.load(neutral, allow_pickle=False) as payload:
        x = payload["x"]
        adjacency = payload["adjacency"]
        assert x.shape == (1, 9, 4)
        assert x[0, 0].tolist() == [1.0, 0.0, 0.0, 0.0]
        assert x[0, 1].tolist() == [0.0, 0.0, 1.0, 0.0]
        assert adjacency[0, 0, 1] == pytest.approx(2.0 / 3.0)
        assert adjacency[0, 1, 2] == pytest.approx(1.0 / 3.0)
    assert manifest["upstream_training_projection"]["training_split_only"] is True


def test_molecular_projection_rejects_aromatic_internal_category(tmp_path: Path) -> None:
    profile = profile_for("qm9")
    graph = _molecule([6, 6], [(0, 1, 4)])
    with pytest.raises(ValueError, match="single/double/triple"):
        export_molecular_dataset(
            train_graphs=[graph],
            val_graphs=[],
            test_graphs=[],
            profile=profile,
            output_dir=tmp_path,
        )


def test_generated_generic_export_decodes_order_and_padding(tmp_path: Path) -> None:
    profile = profile_for("ego_small")
    adjacency = np.zeros((2, 18, 18), dtype=np.int8)
    adjacency[0, 0, 1] = adjacency[0, 1, 0] = 1
    adjacency[1, :3, :3] = nx.to_numpy_array(nx.complete_graph(3), dtype=np.int8)
    output = tmp_path / "generated.npz"
    np.savez_compressed(
        output,
        adjacency=adjacency,
        num_nodes=np.asarray([2, 3], dtype=np.int64),
        sample_index=np.arange(2, dtype=np.int64),
    )
    graphs = load_generated_export(output, profile=profile)
    assert [g.number_of_edges() for g in graphs] == [1, 3]
    assert [g.graph["hog_diff_sample_index"] for g in graphs] == [0, 1]


def test_generated_molecular_export_decodes_raw_categories_without_correction(tmp_path: Path) -> None:
    profile = profile_for("qm9")
    adjacency = np.zeros((1, 9, 9), dtype=np.int8)
    adjacency[0, 0, 1] = adjacency[0, 1, 0] = 3
    atoms = np.full((1, 9), -1, dtype=np.int16)
    atoms[0, :2] = [0, 2]  # C, O
    output = tmp_path / "molecules.npz"
    np.savez_compressed(
        output,
        adjacency=adjacency,
        node_types=atoms,
        num_nodes=np.asarray([2], dtype=np.int64),
        sample_index=np.asarray([0], dtype=np.int64),
    )
    graph = load_generated_export(output, profile=profile)[0]
    assert graph.nodes[0]["atomic_num"] == 6
    assert graph.nodes[1]["atomic_num"] == 8
    assert graph.edges[0, 1]["bond_type"] == 3
    assert graph.edges[0, 1]["bond_order"] == 3.0

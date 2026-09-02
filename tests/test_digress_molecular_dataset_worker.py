from __future__ import annotations

import pickle

import networkx as nx
import pytest

from grapher.models.digress.workers import prepare_molecular_dataset as worker


def test_molecular_dataset_vocabularies_cover_qm9_and_zinc() -> None:
    assert worker.ATOM_VOCABULARIES == {
        "qm9": (6, 7, 8, 9),
        "zinc": (6, 7, 8, 9, 15, 16, 17, 35, 53),
    }
    assert worker.EDGE_VOCABULARIES == {
        "qm9": (1, 2, 3, 4),
        "zinc": (1, 2, 3),
    }


def test_zinc_helper_accepts_declared_atom_and_bond_vocabularies() -> None:
    atoms = [
        worker._atomic_number(
            {"atomic_num": atomic_number, "atom_type": atomic_number},
            label="zinc fixture",
            dataset="zinc",
        )
        for atomic_number in worker.ATOM_VOCABULARIES["zinc"]
    ]
    bonds = [
        worker._bond_type(
            {"bond_type": bond_type, "bond_order": float(bond_type)},
            label="zinc fixture",
            dataset="zinc",
        )
        for bond_type in worker.EDGE_VOCABULARIES["zinc"]
    ]

    assert atoms == list(worker.ATOM_VOCABULARIES["zinc"])
    assert bonds == list(worker.EDGE_VOCABULARIES["zinc"])


def test_zinc_helper_rejects_aromatic_bond_class() -> None:
    with pytest.raises(ValueError, match="bond type 4"):
        worker._bond_type(
            {"bond_type": 4, "bond_order": 1.5},
            label="zinc fixture",
            dataset="zinc",
        )


def test_zinc_conversion_writes_nine_node_and_four_edge_channels(
    tmp_path,
) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    graph = nx.Graph()
    for node, atomic_number in enumerate(worker.ATOM_VOCABULARIES["zinc"]):
        graph.add_node(
            node,
            atomic_num=atomic_number,
            atom_type=atomic_number,
        )
    graph.add_edge(0, 1, bond_type=1, bond_order=1.0)
    graph.add_edge(1, 2, bond_type=2, bond_order=2.0)
    graph.add_edge(2, 3, bond_type=3, bond_order=3.0)
    source = tmp_path / "train.pkl"
    with source.open("wb") as handle:
        pickle.dump([graph], handle)
    processed = tmp_path / "processed" / "proc_tr_no_h.pt"
    model_view = tmp_path / "model_view" / "train.pkl"

    record = worker._convert_split(
        source,
        processed,
        dataset="zinc",
        model_view_destination=model_view,
        split="train",
    )

    data, _ = torch.load(processed, weights_only=False)
    assert data.x.shape == (9, 9)
    assert data.edge_attr.shape == (6, 4)
    assert record["graph_count"] == 1
    with model_view.open("rb") as handle:
        converted = pickle.load(handle)
    assert converted[0].graph["molecular_dataset"] == "zinc"
    assert {
        attributes["bond_type"]
        for _, _, attributes in converted[0].edges(data=True)
    } == {1, 2, 3}

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from grapher.models.digress.codec import load_digress_export


def _write_export(
    path: Path,
    *,
    node_offsets,
    node_types,
    edge_offsets,
    endpoints,
    edge_types,
) -> None:
    with path.open("wb") as handle:
        np.savez_compressed(
            handle,
            node_offsets=np.asarray(node_offsets, dtype=np.int64),
            node_types=np.asarray(node_types, dtype=np.int64),
            edge_offsets=np.asarray(edge_offsets, dtype=np.int64),
            edge_endpoints=np.asarray(endpoints, dtype=np.int64).reshape((-1, 2)),
            edge_types=np.asarray(edge_types, dtype=np.int64),
        )


def test_generic_codec_preserves_order_and_isolates(tmp_path: Path) -> None:
    path = tmp_path / "samples.npz"
    _write_export(
        path,
        node_offsets=[0, 4, 7],
        node_types=[0, 0, 0, 0, 0, 0, 0],
        edge_offsets=[0, 2, 3],
        endpoints=[(0, 1), (1, 2), (0, 2)],
        edge_types=[1, 1, 1],
    )

    graphs = load_digress_export(path, dataset="comm20")

    assert [graph.number_of_nodes() for graph in graphs] == [4, 3]
    assert [set(graph.edges()) for graph in graphs] == [
        {(0, 1), (1, 2)},
        {(0, 2)},
    ]
    assert 3 in graphs[0]
    assert [graph.graph["digress_sample_index"] for graph in graphs] == [0, 1]


def test_qm9_codec_restores_atom_and_bond_attributes(tmp_path: Path) -> None:
    path = tmp_path / "qm9.npz"
    _write_export(
        path,
        node_offsets=[0, 4],
        node_types=[0, 1, 2, 3],
        edge_offsets=[0, 3],
        endpoints=[(0, 1), (1, 2), (2, 3)],
        edge_types=[1, 2, 3],
    )

    graph = load_digress_export(path, dataset="qm9")[0]

    assert [graph.nodes[i]["atomic_num"] for i in range(4)] == [6, 7, 8, 9]
    assert graph.edges[0, 1]["bond_order"] == 1.0
    assert graph.edges[1, 2]["bond_order"] == 2.0
    assert graph.edges[2, 3]["bond_order"] == 3.0
    assert graph.graph["molecular_dataset"] == "qm9"


def test_codec_rejects_mismatched_offsets(tmp_path: Path) -> None:
    path = tmp_path / "bad.npz"
    _write_export(
        path,
        node_offsets=[0, 2],
        node_types=[0, 0],
        edge_offsets=[0, 0, 0],
        endpoints=[],
        edge_types=[],
    )
    with pytest.raises(ValueError, match="different graph counts"):
        load_digress_export(path, dataset="comm20")


def test_codec_rejects_invalid_qm9_class(tmp_path: Path) -> None:
    path = tmp_path / "bad_qm9.npz"
    _write_export(
        path,
        node_offsets=[0, 1],
        node_types=[7],
        edge_offsets=[0, 0],
        endpoints=[],
        edge_types=[],
    )
    with pytest.raises(ValueError, match="invalid atom class"):
        load_digress_export(path, dataset="qm9")

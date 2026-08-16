from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from grapher.models.defog_molecular_codec import (
    MODEL_REPRESENTATION,
    MOLECULAR_EXPORT_FORMAT,
    SOURCE_REPRESENTATION,
    decode_molecular_arrays,
    encode_defog_molecular_samples,
    encode_molecular_graphs,
    load_molecular_export,
    molecular_codec_spec,
    save_molecular_export,
)


def _atom(graph: nx.Graph, node: object, atomic_number: int, **extra) -> None:
    graph.add_node(
        node,
        atomic_num=atomic_number,
        atom_type=atomic_number,
        **extra,
    )


def _bond(
    graph: nx.Graph,
    source: object,
    target: object,
    bond_type: int,
    **extra,
) -> None:
    order = 1.5 if bond_type == 4 else float(bond_type)
    graph.add_edge(
        source,
        target,
        bond_type=bond_type,
        bond_order=order,
        **extra,
    )


def test_exact_dataset_class_mappings() -> None:
    qm9 = molecular_codec_spec("QM9")
    zinc = molecular_codec_spec("zinc")

    assert qm9.atom_class_to_atomic_number == (6, 7, 8, 9)
    assert qm9.atomic_number_to_atom_class == {6: 0, 7: 1, 8: 2, 9: 3}
    assert qm9.model_bond_types == frozenset({1, 2, 3, 4})
    assert zinc.atom_class_to_atomic_number == (6, 7, 8, 9, 15, 16, 17, 35, 53)
    assert zinc.atomic_number_to_atom_class[53] == 8
    assert zinc.source_bond_types == frozenset({1, 2, 3, 4})
    assert zinc.model_bond_types == frozenset({1, 2, 3})


def test_qm9_source_round_trip_preserves_graph_node_and_isolate_order() -> None:
    first = nx.Graph()
    # NetworkX iteration order, rather than lexical or numeric sorting, defines
    # the local tensor order at the wrapper boundary.
    _atom(first, "fluorine", 9)
    _atom(first, "carbon", 6)
    _atom(first, "isolated-oxygen", 8)
    _bond(first, "fluorine", "carbon", 1)

    second = nx.Graph()
    _atom(second, 11, 7)
    _atom(second, 4, 6)
    _bond(second, 11, 4, 4)

    arrays = encode_molecular_graphs(
        [first, second],
        dataset="qm9",
        representation=SOURCE_REPRESENTATION,
    )
    decoded = decode_molecular_arrays(
        arrays,
        expected_dataset="qm9",
        expected_representation=SOURCE_REPRESENTATION,
        expected_count=2,
    )

    assert arrays["format"].item() == MOLECULAR_EXPORT_FORMAT
    assert arrays["raw_indices"].tolist() == [0, 1]
    assert [list(graph.nodes()) for graph in decoded] == [[0, 1, 2], [0, 1]]
    assert [data["atomic_num"] for _, data in decoded[0].nodes(data=True)] == [
        9,
        6,
        8,
    ]
    assert decoded[0].degree[2] == 0
    assert decoded[0].edges[0, 1] == {"bond_type": 1, "bond_order": 1.0}
    assert decoded[1].edges[0, 1] == {"bond_type": 4, "bond_order": 1.5}
    assert [graph.graph["defog_raw_index"] for graph in decoded] == [0, 1]


def test_defog_qm9_sample_maps_class_indices_to_semantic_attributes() -> None:
    atoms = np.asarray([0, 1, 2, 3], dtype=np.int64)
    edges = np.zeros((4, 4), dtype=np.int64)
    edges[0, 1] = edges[1, 0] = 1
    edges[1, 2] = edges[2, 1] = 2
    edges[2, 3] = edges[3, 2] = 4

    arrays = encode_defog_molecular_samples([[atoms, edges]], dataset="qm9")
    graphs = decode_molecular_arrays(
        arrays,
        expected_dataset="qm9",
        expected_representation=MODEL_REPRESENTATION,
    )

    assert arrays["representation"].item() == MODEL_REPRESENTATION
    assert [data["atomic_num"] for _, data in graphs[0].nodes(data=True)] == [
        6,
        7,
        8,
        9,
    ]
    assert [data["bond_type"] for *_, data in graphs[0].edges(data=True)] == [
        1,
        2,
        4,
    ]


def test_zinc_aromatic_source_is_explicitly_distinct_from_model_view() -> None:
    graph = nx.Graph()
    _atom(graph, 0, 6)
    _atom(graph, 1, 7)
    _bond(graph, 0, 1, 4)

    source_arrays = encode_molecular_graphs(
        [graph], dataset="zinc", representation=SOURCE_REPRESENTATION
    )
    source_graph = decode_molecular_arrays(
        source_arrays,
        expected_dataset="zinc",
        expected_representation=SOURCE_REPRESENTATION,
    )[0]
    assert source_graph.edges[0, 1]["bond_type"] == 4

    with pytest.raises(ValueError, match="explicit audited kekulization"):
        encode_molecular_graphs(
            [graph], dataset="zinc", representation=MODEL_REPRESENTATION
        )
    with pytest.raises(ValueError, match="does not match expected representation"):
        decode_molecular_arrays(
            source_arrays,
            expected_dataset="zinc",
            expected_representation=MODEL_REPRESENTATION,
        )


def test_zinc_model_samples_reject_aromatic_class_four() -> None:
    atoms = np.asarray([0, 1], dtype=np.int64)
    aromatic = np.asarray([[0, 4], [4, 0]], dtype=np.int64)
    with pytest.raises(ValueError, match="aromatic class 4"):
        encode_defog_molecular_samples([[atoms, aromatic]], dataset="zinc")


@pytest.mark.parametrize(
    "atoms, edges, message",
    [
        (
            np.asarray([0, 9]),
            np.zeros((2, 2), dtype=np.int64),
            "outside 0--3",
        ),
        (
            np.asarray([0, 1]),
            np.asarray([[0, 1], [0, 0]]),
            "not symmetric",
        ),
        (
            np.asarray([0, 1]),
            np.asarray([[1, 0], [0, 0]]),
            "self-loop",
        ),
        (
            np.asarray([0, 1]),
            np.zeros((3, 3), dtype=np.int64),
            "expected",
        ),
    ],
)
def test_model_encoder_rejects_invalid_dense_categorical_state(
    atoms: np.ndarray, edges: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        encode_defog_molecular_samples([[atoms, edges]], dataset="qm9")


def test_source_encoder_rejects_invalid_or_unrepresented_chemistry() -> None:
    directed = nx.DiGraph()
    _atom(directed, 0, 6)
    with pytest.raises(ValueError, match="simple undirected"):
        encode_molecular_graphs([directed], dataset="qm9")

    loop = nx.Graph()
    _atom(loop, 0, 6)
    _bond(loop, 0, 0, 1)
    with pytest.raises(ValueError, match="self-loop"):
        encode_molecular_graphs([loop], dataset="qm9")

    charged = nx.Graph()
    _atom(charged, 0, 7, formal_charge=1)
    with pytest.raises(ValueError, match="non-zero formal_charge"):
        encode_molecular_graphs([charged], dataset="qm9")

    stereo = nx.Graph()
    _atom(stereo, 0, 6)
    _atom(stereo, 1, 6)
    _bond(stereo, 0, 1, 1, stereo="E")
    with pytest.raises(ValueError, match="unsupported chemical attribute stereo"):
        encode_molecular_graphs([stereo], dataset="qm9")


def test_decoder_rejects_duplicate_edges_and_reordered_graph_indices() -> None:
    graph = nx.Graph()
    _atom(graph, 0, 6)
    _atom(graph, 1, 7)
    _bond(graph, 0, 1, 1)
    arrays = encode_molecular_graphs([graph], dataset="qm9")

    duplicate = dict(arrays)
    duplicate["edge_ptr"] = np.asarray([0, 2], dtype=np.int64)
    duplicate["edge_endpoints"] = np.asarray([[0, 1], [0, 1]], dtype=np.int64)
    duplicate["edge_bond_types"] = np.asarray([1, 1], dtype=np.int64)
    with pytest.raises(ValueError, match="duplicate edges"):
        decode_molecular_arrays(duplicate)

    two_graphs = encode_molecular_graphs([graph, graph.copy()], dataset="qm9")
    two_graphs["raw_indices"] = np.asarray([1, 0], dtype=np.int64)
    with pytest.raises(ValueError, match="omission or reordering"):
        decode_molecular_arrays(two_graphs)


def test_npz_persistence_is_pickle_free_and_checks_expected_count(
    tmp_path: Path,
) -> None:
    graph = nx.Graph()
    _atom(graph, 0, 6)
    _atom(graph, 1, 8)
    _bond(graph, 0, 1, 2)
    arrays = encode_molecular_graphs([graph], dataset="qm9")
    path = save_molecular_export(tmp_path / "molecules.npz", arrays)

    with np.load(path, allow_pickle=False) as payload:
        assert payload["format"].item() == MOLECULAR_EXPORT_FORMAT
        assert not any(payload[name].dtype.kind == "O" for name in payload.files)

    decoded = load_molecular_export(
        path,
        expected_dataset="qm9",
        expected_representation=SOURCE_REPRESENTATION,
        expected_count=1,
    )
    assert len(decoded) == 1
    with pytest.raises(ValueError, match="expected 2"):
        load_molecular_export(path, expected_count=2)

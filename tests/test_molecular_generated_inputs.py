from __future__ import annotations

import importlib.util
from pathlib import Path
import pickle
from types import SimpleNamespace

import networkx as nx
import pytest


@pytest.fixture(scope="module")
def evaluator():
    path = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_generated_molecules.py"
    spec = importlib.util.spec_from_file_location("_molecular_generated_inputs_evaluator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _args(**values):
    return SimpleNamespace(**{
        "generated_graphs": None,
        "generated_dir": None,
        "generated_smiles": None,
        **values,
    })


def _molecule(graph: nx.Graph, label: str) -> nx.Graph:
    nx.set_node_attributes(graph, 6, "atomic_num")
    nx.set_edge_attributes(graph, 1, "bond_type")
    graph.graph["record_label"] = label
    return graph


def _valid_and_invalid() -> list[nx.Graph]:
    return [
        _molecule(nx.path_graph(2), "valid_ethane"),
        _molecule(nx.star_graph(5), "invalid_carbon_valence_five"),
    ]


def _write_graphs(path: Path, graphs, wrapper=None) -> None:
    with path.open("wb") as handle:
        pickle.dump(graphs if wrapper is None else {wrapper: graphs}, handle)


@pytest.mark.parametrize("wrapper", [None, "graphs", "molecular_graphs", "generated_graphs"])
def test_baseline_pickle_retains_all_graphs_and_beats_valid_only_smiles(evaluator, tmp_path, wrapper) -> None:
    original = _valid_and_invalid()
    raw_path = tmp_path / "base_graphs.pkl"
    _write_graphs(raw_path, original, wrapper)
    (tmp_path / "generated.smi").write_text("CC\n", encoding="utf-8")

    graphs, source, complete = evaluator._resolve_generated_graphs(_args(generated_dir=str(tmp_path)))

    assert source == str(raw_path)
    assert complete is True
    assert len(graphs) == 2
    assert all(nx.utils.graphs_equal(actual, expected) for actual, expected in zip(graphs, original))
    assert max(dict(graphs[1].degree()).values()) == 5


def test_baseline_raw_graphs_keep_the_complete_validity_denominator(evaluator, tmp_path) -> None:
    pytest.importorskip("rdkit")
    _write_graphs(tmp_path / "base_graphs.pkl", _valid_and_invalid())
    (tmp_path / "generated.smi").write_text("CC\n", encoding="utf-8")

    graphs, _source, complete = evaluator._resolve_generated_graphs(_args(generated_dir=str(tmp_path)))
    validity = evaluator._validity_and_smiles(graphs)

    assert complete is True
    assert validity["num_graphs"] == 2
    assert validity["num_valid"] == validity["num_invalid"] == 1
    assert validity["invalid_indices"] == [1]
    assert validity["validity_without_correction"] == 0.5


@pytest.mark.parametrize("filenames, expected", [
    (("base_graphs.pkl", "generated_graphs.pkl", "molecular_graphs.pkl"), "molecular_graphs.pkl"),
    (("base_graphs.pkl", "generated_graphs.pkl"), "generated_graphs.pkl"),
    (("base_graphs.pkl",), "base_graphs.pkl"),
])
def test_generated_directory_keeps_grapher_file_precedence(evaluator, tmp_path, filenames, expected) -> None:
    for filename in filenames:
        _write_graphs(tmp_path / filename, [_molecule(nx.path_graph(2), filename)])
    (tmp_path / "generated.smi").write_text("CC\n", encoding="utf-8")

    graphs, source, complete = evaluator._resolve_generated_graphs(_args(generated_dir=str(tmp_path)))

    assert source == str(tmp_path / expected)
    assert complete is True
    assert [graph.graph["record_label"] for graph in graphs] == [expected]


def test_explicit_graph_input_has_priority_over_directory_and_smiles(evaluator, tmp_path) -> None:
    explicit = tmp_path / "chosen.pkl"
    _write_graphs(explicit, _valid_and_invalid())
    _write_graphs(tmp_path / "molecular_graphs.pkl", [_molecule(nx.path_graph(3), "directory")])
    smiles = tmp_path / "chosen.smi"
    smiles.write_text("CCC\n", encoding="utf-8")

    graphs, source, complete = evaluator._resolve_generated_graphs(_args(
        generated_graphs=str(explicit), generated_dir=str(tmp_path), generated_smiles=str(smiles),
    ))

    assert source == str(explicit)
    assert complete is True
    assert [graph.graph["record_label"] for graph in graphs] == [
        "valid_ethane", "invalid_carbon_valence_five",
    ]


def test_missing_explicit_graph_does_not_silently_fall_back_to_directory(evaluator, tmp_path) -> None:
    missing = tmp_path / "missing.pkl"
    _write_graphs(tmp_path / "base_graphs.pkl", _valid_and_invalid())
    with pytest.raises(FileNotFoundError) as error:
        evaluator._resolve_generated_graphs(_args(generated_graphs=str(missing), generated_dir=str(tmp_path)))
    assert str(missing) in str(error.value)


@pytest.mark.parametrize("mode", ["directory_smiles", "explicit_smiles", "explicit_smiles_empty_directory"])
def test_smiles_inputs_remain_available_with_incomplete_denominator(evaluator, tmp_path, mode) -> None:
    pytest.importorskip("rdkit")
    filename = "generated.smi" if mode == "directory_smiles" else "chosen.smi"
    smiles = tmp_path / filename
    smiles.write_text("CC\n", encoding="utf-8")
    args = _args(
        generated_dir=str(tmp_path) if mode != "explicit_smiles" else None,
        generated_smiles=str(smiles) if mode != "directory_smiles" else None,
    )

    graphs, source, complete = evaluator._resolve_generated_graphs(args)

    assert source == str(smiles)
    assert complete is False
    assert len(graphs) == 1
    assert graphs[0].number_of_nodes() == 2
    assert evaluator._validity_and_smiles(graphs)["num_valid"] == 1


@pytest.mark.parametrize("kind", ["missing", "empty", "file"])
def test_invalid_generated_directory_reports_the_actual_path(evaluator, tmp_path, kind) -> None:
    directory = tmp_path / "generated_run"
    if kind == "empty":
        directory.mkdir()
    elif kind == "file":
        directory.write_text("This is a file, not a directory.", encoding="utf-8")
    expected_error = NotADirectoryError if kind == "file" else FileNotFoundError

    with pytest.raises(expected_error) as error:
        evaluator._resolve_generated_graphs(_args(generated_dir=str(directory)))

    assert str(directory) in str(error.value)
    if kind == "empty":
        for filename in ("molecular_graphs.pkl", "generated_graphs.pkl", "base_graphs.pkl", "generated.smi"):
            assert filename in str(error.value)


def test_missing_all_input_arguments_retains_usage_error(evaluator) -> None:
    with pytest.raises(ValueError) as error:
        evaluator._resolve_generated_graphs(_args())
    for option in ("--generated-dir", "--generated-graphs", "--generated-smiles"):
        assert option in str(error.value)

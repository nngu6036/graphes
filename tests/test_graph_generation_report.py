from __future__ import annotations

import networkx as nx

from scripts.evaluate_graph_generation_report import (
    BOND_COLORS,
    _bond_type,
    _print_molecular_metrics,
    is_molecular_evaluation,
    molecular_quality_metrics,
    select_sample_indices,
)


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


def _molecule(atomic_numbers: list[int], edges: list[tuple[int, int, int]]) -> nx.Graph:
    graph = nx.Graph()
    for index, atomic_number in enumerate(atomic_numbers):
        graph.add_node(
            index,
            atomic_num=atomic_number,
            atom_type=atomic_number,
        )
    for source, target, bond_type in edges:
        graph.add_edge(
            source,
            target,
            bond_type=bond_type,
            bond_order=float(bond_type),
        )
    return graph


def test_molecular_detection_uses_dataset_name_or_atom_attributes() -> None:
    assert is_molecular_evaluation({"name": "qm9_attributed"}, [nx.path_graph(2)])
    assert is_molecular_evaluation(
        {"name": "custom"},
        [_molecule([6, 8], [(0, 1, 1)])],
    )
    assert not is_molecular_evaluation({"name": "sbm"}, [nx.path_graph(2)])


def test_molecular_quality_uses_rdkit_canonical_smiles() -> None:
    water = _molecule([8], [])
    methanol = _molecule([6, 8], [(0, 1, 1)])
    invalid_carbon = _molecule(
        [6, 9, 9, 9, 9, 9],
        [(0, index, 1) for index in range(1, 6)],
    )

    metrics, smiles, invalid_indices, errors = molecular_quality_metrics(
        [water, water.copy(), methanol, invalid_carbon],
        [water],
    )

    assert metrics["validity_without_correction"] == 0.75
    assert metrics["uniqueness_rate"] == 2 / 3
    assert metrics["novelty_rate"] == 0.5
    assert metrics["unique_valid_count"] == 2
    assert metrics["novel_unique_valid_count"] == 1
    assert len(smiles) == 3
    assert invalid_indices == [3]
    assert sum(errors.values()) == 1


def test_molecular_plot_has_distinct_colors_for_every_bond_type() -> None:
    assert len(BOND_COLORS) == 4
    assert len(set(BOND_COLORS.values())) == 4
    assert _bond_type({"bond_type": 2}) == 2
    assert _bond_type({"bond_order": 1.5}) == 4


def test_molecular_report_explains_missing_attributes(capsys) -> None:
    metrics = {
        "validity_without_correction": 0.0,
        "uniqueness_rate": 0.0,
        "novelty_rate": 0.0,
        "num_valid_generated_molecules": 0,
        "num_generated_graphs": 2,
        "unique_valid_count": 0,
        "novel_unique_valid_count": 0,
    }
    _print_molecular_metrics(metrics, {"MissingAtomType": 2})

    output = capsys.readouterr().out
    assert "MissingAtomType=2" in output
    assert "RDKit was not given molecular graphs" in output

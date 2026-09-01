from __future__ import annotations

import networkx as nx

from scripts.evaluate_graph_generation_report import (
    dataset_provenance,
    is_molecular_evaluation,
    molecular_quality_metrics,
    resolve_evaluation_counts,
    select_sample_indices,
    validate_dataset_compatibility,
)
from grapher.models.base import DatasetReference
from grapher.utils.io import save_pickle, save_yaml


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



def test_max_graphs_caps_candidates_but_not_reference() -> None:
    reference_count, generated_count = resolve_evaluation_counts(
        num_reference=40,
        num_generated=1024,
        configured_reference_cap=None,
        max_generated_graphs=20,
    )
    assert reference_count == 40
    assert generated_count == 20


def test_dataset_provenance_matches_wrapper_fingerprint(tmp_path) -> None:
    root = tmp_path / "datasets"
    dataset_dir = root / "sbm"
    dataset_dir.mkdir(parents=True)
    splits = {
        "train": [nx.path_graph(4), nx.cycle_graph(5)],
        "val": [nx.path_graph(5)],
        "test": [nx.cycle_graph(4)],
    }
    for split, graphs in splits.items():
        save_pickle(graphs, dataset_dir / f"{split}.pkl")
    save_yaml(
        {"name": "sbm", "benchmark": "community_small", "protocol_id": "unit_v1"},
        dataset_dir / "resolved_dataset_config.yaml",
    )
    config_path = tmp_path / "community_small.yaml"
    save_yaml(
        {"name": "sbm", "benchmark": "community_small", "protocol_id": "unit_v1"},
        config_path,
    )
    config = {
        "name": "sbm",
        "benchmark": "community_small",
        "root": str(root),
        "config_path": str(config_path),
    }
    observed = dataset_provenance(config, splits)
    expected = DatasetReference(
        benchmark_id="community_small",
        root=root,
        serialized_id="sbm",
    ).fingerprint()
    assert observed["fingerprint"] == expected
    assert observed["protocol_id"] == "unit_v1"
    assert observed["split_sizes"] == {"train": 2, "val": 1, "test": 1}


def test_dataset_mismatch_fails_by_default() -> None:
    provenance = {
        "benchmark_id": "community_small",
        "fingerprint": "new",
        "split_sha256": {"train": "a", "val": "b", "test": "c"},
    }
    training_manifest = {
        "dataset": {
            "benchmark_id": "community_small",
            "fingerprint": "old",
            "split_sha256": {"train": "x", "val": "b", "test": "c"},
        }
    }
    try:
        validate_dataset_compatibility(
            provenance,
            generation_manifest=None,
            training_manifest=training_manifest,
            mismatch_policy="error",
        )
    except RuntimeError as exc:
        assert "dataset split fingerprint differs" in str(exc)
        assert "train split SHA-256 differs" in str(exc)
    else:
        raise AssertionError("Expected dataset mismatch to fail evaluation.")

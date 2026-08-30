from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import pytest

from grapher.utils.io import load_pickle, load_yaml
from scripts.prepare_zinc_dataset import (
    ZincProtocol,
    ZincRecordError,
    download_bundled_zinc_source,
    prepare_zinc_dataset,
    read_zinc_smiles,
    smiles_to_zinc_graph,
)


def _zinc_config(*, expected_graphs: int = 6, seed: int = 0) -> dict:
    return {
        "name": "zinc",
        "kind": "prepared_molecular_attributed",
        "root": "outputs/datasets",
        "source": {
            "dataset": "test_zinc",
            "subset": {
                "selection": "first_valid_after_seeded_shuffle",
                "seed": seed,
                "expected_graphs": expected_graphs,
            },
        },
        "split": {
            "train": expected_graphs - 2,
            "val": 1,
            "test": 1,
        },
        "preprocessing": {
            "remove_hydrogens": True,
            "undirected": True,
            "keep_largest_fragment": True,
            "sanitize_with_rdkit": True,
            "kekulize": True,
            "retain_aromatic_bonds": False,
            "retain_formal_charge": False,
            "retain_stereochemistry": False,
            "max_nodes": 38,
        },
        "filters": {
            "require_connected": True,
            "neutral_only": True,
            "uncharged_atoms_only": True,
            "allowed_atomic_numbers": [6, 7, 8, 9, 15, 16, 17, 35, 53],
        },
        "categorical_state": {
            "node_categories": [6, 7, 8, 9, 15, 16, 17, 35, 53],
            "edge_categories": [1, 2, 3],
        },
        "bond_orders": {1: 1.0, 2: 2.0, 3: 3.0},
    }


def test_repository_zinc_config_resolves_fixed_12k_protocol() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    protocol = ZincProtocol.from_config(
        load_yaml(repository_root / "configs/datasets/zinc.yaml")
    )

    assert protocol.dataset_name == "zinc"
    assert protocol.expected_graphs == 12000
    assert protocol.seed == 0
    assert protocol.split_counts == {"train": 10000, "val": 1000, "test": 1000}
    assert protocol.uncharged_atoms_only is True
    assert protocol.allowed_bond_types == (1, 2, 3)
    assert protocol.kekulize is True
    assert protocol.retain_aromatic_bonds is False


def test_read_zinc_smiles_uses_named_or_indexed_csv_column(
    tmp_path: Path,
) -> None:
    source = tmp_path / "zinc.csv"
    source.write_text(
        "identifier,molecule\nfirst,CC\nsecond,C#N\n",
        encoding="utf-8",
    )

    assert read_zinc_smiles(source, smiles_column="molecule") == ["CC", "C#N"]
    assert read_zinc_smiles(source, smiles_column=1) == ["CC", "C#N"]


def test_zinc_conversion_kekulizes_aromatic_bonds() -> None:
    pytest.importorskip("rdkit")
    protocol = ZincProtocol.from_config(_zinc_config())
    examples = {
        "CC": {1},
        "C=C": {2},
        "C#N": {3},
        "c1ccccc1": {1, 2},
    }

    for smiles, expected_bond_types in examples.items():
        graph = smiles_to_zinc_graph(smiles, protocol)

        assert nx.is_connected(graph)
        assert {data["bond_type"] for _, _, data in graph.edges(data=True)} == (
            expected_bond_types
        )
        assert all(
            "atomic_num" in data and "atom_type" in data
            for _, data in graph.nodes(data=True)
        )


def test_zinc_conversion_applies_fragment_charge_and_size_filters() -> None:
    pytest.importorskip("rdkit")
    protocol = ZincProtocol.from_config(_zinc_config())

    largest = smiles_to_zinc_graph("CC.O", protocol)
    assert largest.number_of_nodes() == 2
    assert nx.is_connected(largest)

    with pytest.raises(ZincRecordError) as charged:
        smiles_to_zinc_graph("[NH4+]", protocol)
    assert charged.value.reason == "non_neutral"

    with pytest.raises(ZincRecordError) as zwitterion:
        smiles_to_zinc_graph("[NH3+]CC(=O)[O-]", protocol)
    assert zwitterion.value.reason == "charged_atom"

    with pytest.raises(ZincRecordError) as invalid:
        smiles_to_zinc_graph("not-a-smiles", protocol)
    assert invalid.value.reason == "parse_failure"

    small_config = _zinc_config()
    small_config["preprocessing"]["max_nodes"] = 2
    small_protocol = ZincProtocol.from_config(small_config)
    with pytest.raises(ZincRecordError) as oversized:
        smiles_to_zinc_graph("CCC", small_protocol)
    assert oversized.value.reason == "too_many_nodes"


def test_prepare_zinc_dataset_is_deterministic_and_writes_expected_artifacts(
    tmp_path: Path,
) -> None:
    pytest.importorskip("rdkit")
    source = tmp_path / "zinc.csv"
    source.write_text(
        "id,smiles\n0,CC\n1,CCC\n2,C=C\n3,C#N\n4,c1ccccc1\n5,not-a-smiles\n6,CO\n",
        encoding="utf-8",
    )
    config = _zinc_config(expected_graphs=6, seed=0)
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"

    first_report = prepare_zinc_dataset(
        source,
        config,
        root=first_root,
        smiles_column="smiles",
    )
    second_report = prepare_zinc_dataset(
        source,
        config,
        root=second_root,
        smiles_column="smiles",
    )

    expected_files = {
        "train.pkl",
        "val.pkl",
        "test.pkl",
        "resolved_dataset_config.yaml",
        "metadata.json",
        "prep_report.json",
    }
    assert expected_files <= {path.name for path in (first_root / "zinc").iterdir()}
    assert first_report["split_sizes"] == {"train": 4, "val": 1, "test": 1}
    assert first_report["filter_diagnostics"] == {
        "num_rejected": 1,
        "rejection_reasons": {"parse_failure": 1},
    }
    assert first_report["graph_statistics"]["all_connected"] is True
    assert (
        first_report["selected_records_sha256"]
        == second_report["selected_records_sha256"]
    )

    def selected_indices(root: Path) -> list[int]:
        return [
            int(graph.graph["source_index"])
            for split in ("train", "val", "test")
            for graph in load_pickle(root / "zinc" / f"{split}.pkl")
        ]

    assert selected_indices(first_root) == selected_indices(second_root)
    assert selected_indices(first_root) == [2, 4, 3, 6, 0, 1]
    metadata = json.loads(
        (first_root / "zinc" / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata == {
        "dataset": "zinc",
        "split_sizes": {"test": 1, "train": 4, "val": 1},
    }


def test_prepare_zinc_dataset_rejects_an_insufficient_valid_source(
    tmp_path: Path,
) -> None:
    pytest.importorskip("rdkit")
    source = tmp_path / "short.smi"
    source.write_text("CC\nnot-a-smiles\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Only 1 valid ZINC graphs"):
        prepare_zinc_dataset(
            source,
            _zinc_config(expected_graphs=3),
            root=tmp_path / "output",
        )


def test_bundled_zinc_download_is_an_explicit_placeholder() -> None:
    with pytest.raises(NotImplementedError, match="--smiles-file"):
        download_bundled_zinc_source()

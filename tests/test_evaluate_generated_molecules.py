from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import networkx as nx
import pytest

from grapher.utils.io import save_pickle


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_generated_molecules.py"
SPEC = importlib.util.spec_from_file_location("evaluate_generated_molecules", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _valid_cf4_graph() -> nx.Graph:
    graph = nx.Graph()
    graph.add_node(0, atomic_num=6, atom_type=6)
    for node in range(1, 5):
        graph.add_node(node, atomic_num=9, atom_type=9)
        graph.add_edge(0, node, bond_type=1, bond_order=1.0)
    return graph


def _correctable_carbon_valence_five_graph() -> nx.Graph:
    # C(=O)(F)(F)F has carbon valence five.  Reducing C=O to C-O yields a
    # sanitizable connected molecule and exercises deterministic correction.
    graph = nx.Graph()
    graph.add_node(0, atomic_num=6, atom_type=6)
    graph.add_node(1, atomic_num=8, atom_type=8)
    graph.add_edge(0, 1, bond_type=2, bond_order=2.0)
    for node in range(2, 5):
        graph.add_node(node, atomic_num=9, atom_type=9)
        graph.add_edge(0, node, bond_type=1, bond_order=1.0)
    return graph


def test_reports_validity_with_and_without_correction() -> None:
    pytest.importorskip("rdkit")
    graphs = [_valid_cf4_graph(), _correctable_carbon_valence_five_graph()]

    raw = MODULE._validity_and_smiles(graphs)
    corrected = MODULE._validity_with_correction(
        graphs,
        raw_smiles=raw["all_smiles"],
        max_steps=10,
    )

    assert raw["validity_without_correction"] == pytest.approx(0.5)
    assert corrected["validity"] == pytest.approx(1.0)
    assert corrected["validity_with_correction"] == pytest.approx(1.0)
    assert corrected["num_corrected"] == 1
    assert corrected["corrected_indices"] == [1]
    assert corrected["correction_steps"][1] == 1


def test_evaluate_outputs_validity_raw_validity_and_fcd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("rdkit")
    dataset_root = tmp_path / "datasets"
    dataset_dir = dataset_root / "qm9_attributed"
    dataset_dir.mkdir(parents=True)

    valid = _valid_cf4_graph()
    invalid_but_correctable = _correctable_carbon_valence_five_graph()
    save_pickle([valid], dataset_dir / "train.pkl")
    save_pickle([valid], dataset_dir / "test.pkl")

    generated_path = tmp_path / "molecular_graphs.pkl"
    save_pickle([valid, invalid_but_correctable], generated_path)

    monkeypatch.setattr(
        MODULE,
        "compute_fcd",
        lambda reference_smiles, generated_smiles, **kwargs: (
            1.234,
            {
                "status": "ok",
                "backend": "test",
                "num_generated": len(generated_smiles),
            },
        ),
    )

    args = argparse.Namespace(
        generated_graphs=str(generated_path),
        generated_dir=None,
        generated_smiles=None,
        dataset_root=str(dataset_root),
        dataset="qm9_attributed",
        reference_split="test",
        train_split="train",
        reference_smiles=None,
        train_smiles=None,
        max_generated=None,
        max_reference=None,
        max_train=None,
        nspdk_backend="proxy",
        nspdk_radius=1,
        nspdk_distance=2,
        no_nspdk_normalize=False,
        skip_fcd=False,
        fcd_device="cpu",
        fcd_use_corrected=True,
        require_fcd=True,
        correction_max_steps=10,
    )

    report = MODULE.evaluate(args)
    metrics = report["metrics"]

    assert metrics["validity"] == pytest.approx(0.5)
    assert metrics["validity_with_correction"] == pytest.approx(1.0)
    assert metrics["validity_without_correction"] == pytest.approx(0.5)
    assert metrics["fcd"] == pytest.approx(1.234)
    assert metrics["fcd_num_valid_generated_molecules"] == 2
    assert metrics["fcd_generated_smiles_source"] == "valid_with_correction"
    assert report["correction"]["corrected_indices"] == [1]


def test_eden_nspdk_matches_hogdiff_linear_kernel_protocol() -> None:
    pytest.importorskip("rdkit")
    from grapher.rewiring_mlp.evaluation.molecular_nspdk import eden_nspdk_mmd

    single = nx.Graph()
    single.add_node(0, atomic_num=6, atom_type=6)
    single.add_node(1, atomic_num=8, atom_type=8)
    single.add_edge(0, 1, bond_type=1, bond_order=1.0)

    double = single.copy()
    double[0][1]["bond_type"] = 2
    double[0][1]["bond_order"] = 2.0

    assert eden_nspdk_mmd([single], [single], complexity=4) == pytest.approx(0.0)
    assert eden_nspdk_mmd([single], [double], complexity=4) > 0.0


def test_eden_nspdk_singleton_scores_match_one_graph_mmd() -> None:
    pytest.importorskip("rdkit")
    from grapher.rewiring_mlp.evaluation.molecular_nspdk import (
        eden_nspdk_mmd,
        eden_nspdk_singleton_mmd,
    )

    single = nx.Graph()
    single.add_node(0, atomic_num=6, atom_type=6)
    single.add_node(1, atomic_num=8, atom_type=8)
    single.add_edge(0, 1, bond_type=1, bond_order=1.0)
    double = single.copy()
    double[0][1]["bond_type"] = 2
    double[0][1]["bond_order"] = 2.0
    scores = eden_nspdk_singleton_mmd(
        [single],
        [single, double],
        complexity=4,
    )

    assert scores.shape == (2,)
    assert scores[0] == pytest.approx(0.0)
    assert scores[1] == pytest.approx(
        eden_nspdk_mmd([single], [double], complexity=4)
    )

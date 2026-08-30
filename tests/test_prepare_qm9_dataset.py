from __future__ import annotations

from pathlib import Path

import pytest

from scripts.prepare_qm9_dataset import (
    QM9Protocol,
    _clear_dataset_outputs,
    _dataset_output_paths,
    _graphs_from_sdf_qm9,
    _rdkit_mol_to_nx,
    _read_uncharacterized_indices,
)
from grapher.utils.io import load_yaml


def test_dataset_output_paths_require_distinct_direct_children(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be different"):
        _dataset_output_paths(tmp_path, ("qm9", "qm9"))

    for unsafe_name in ("", ".", "..", "nested/qm9"):
        with pytest.raises(ValueError):
            _dataset_output_paths(tmp_path, (unsafe_name, "qm9_attributed"))


def test_clear_dataset_outputs_removes_only_selected_directories(
    tmp_path: Path,
) -> None:
    topology_path, attributed_path = _dataset_output_paths(
        tmp_path,
        ("qm9_topology", "qm9_attributed"),
    )
    topology_path.mkdir()
    attributed_path.mkdir()
    (topology_path / "stale.pkl").write_text("old", encoding="utf-8")
    (attributed_path / "stale.json").write_text("old", encoding="utf-8")
    unrelated = tmp_path / "keep_me"
    unrelated.mkdir()

    removed = _clear_dataset_outputs((topology_path, attributed_path))

    assert removed == [topology_path, attributed_path]
    assert not topology_path.exists()
    assert not attributed_path.exists()
    assert unrelated.is_dir()


def test_repository_qm9_protocol_pins_canonical_counts() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    protocol = QM9Protocol.from_config(
        load_yaml(repository_root / "configs/datasets/qm9.yaml")
    )

    assert protocol.canonical is True
    assert protocol.expected_source_records == 133885
    assert protocol.expected_excluded_records == 3054
    assert protocol.expected_graphs == 130831
    assert protocol.split_seed == 42
    assert protocol.split_counts == {
        "train": 104665,
        "val": 13083,
        "test": 13083,
    }
    assert protocol.project_formal_charge is True
    assert protocol.project_stereochemistry is True
    assert protocol.sanitize_sdf is False
    assert protocol.kekulize is False


def test_uncharacterized_parser_uses_official_one_based_layout(
    tmp_path: Path,
) -> None:
    path = tmp_path / "uncharacterized.txt"
    lines = [f"header {index}" for index in range(9)]
    lines.extend(["1 reason", "4 reason", "10 reason"])
    lines.append("footer")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    assert _read_uncharacterized_indices(
        path,
        expected_count=3,
        expected_source_records=10,
    ) == {0, 3, 9}


def test_uncharacterized_parser_rejects_count_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "uncharacterized.txt"
    lines = [f"header {index}" for index in range(9)]
    lines.extend(["1 reason", "footer"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="count mismatch"):
        _read_uncharacterized_indices(
            path,
            expected_count=2,
            expected_source_records=10,
        )


def test_qm9_conversion_audits_projected_charge_and_stereo() -> None:
    pytest.importorskip("rdkit")
    from rdkit import Chem

    charged = _rdkit_mol_to_nx(Chem.MolFromSmiles("[NH4+]"))
    stereo = _rdkit_mol_to_nx(Chem.MolFromSmiles("C[C@H](N)O"))

    assert charged.graph["projected_formal_charge_atoms"] == [[0, 1]]
    assert stereo.graph["projected_chiral_atoms"]


def test_qm9_sdf_loader_does_not_apply_rdkit_valence_sanitization(
    tmp_path: Path,
) -> None:
    pytest.importorskip("rdkit")
    from rdkit import Chem

    rw_mol = Chem.RWMol()
    carbon = rw_mol.AddAtom(Chem.Atom(6))
    for _ in range(5):
        hydrogen = rw_mol.AddAtom(Chem.Atom(1))
        rw_mol.AddBond(carbon, hydrogen, Chem.BondType.SINGLE)
    sdf = tmp_path / "raw_qm9_like.sdf"
    sdf.write_text(
        Chem.MolToMolBlock(rw_mol.GetMol()) + "\n$$$$\n",
        encoding="utf-8",
    )

    sanitized = Chem.SDMolSupplier(str(sdf), removeHs=False, sanitize=True)
    assert sanitized[0] is None

    graphs, errors, num_seen = _graphs_from_sdf_qm9(
        sdf,
        remove_h=True,
        kekulize=False,
        sanitize_sdf=False,
    )
    assert num_seen == 1
    assert errors == {}
    assert len(graphs) == 1
    assert graphs[0].number_of_nodes() == 1

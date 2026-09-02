from __future__ import annotations

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

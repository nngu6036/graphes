from __future__ import annotations

# QM9 in the standard no-hydrogen setting uses heavy atoms only.
QM9_ATOM_TYPES = (6, 7, 8, 9)  # C, N, O, F
QM9_ATOM_SYMBOLS = {6: "C", 7: "N", 8: "O", 9: "F"}

# Bond type ids used internally. They are intentionally simple integers so
# pkl files do not depend on RDKit classes.
BOND_SINGLE = 1
BOND_DOUBLE = 2
BOND_TRIPLE = 3
BOND_AROMATIC = 4
QM9_BOND_TYPES = (BOND_SINGLE, BOND_DOUBLE, BOND_TRIPLE, BOND_AROMATIC)
BOND_TYPE_NAMES = {
    BOND_SINGLE: "single",
    BOND_DOUBLE: "double",
    BOND_TRIPLE: "triple",
    BOND_AROMATIC: "aromatic",
}

# Dense edge categories used by the conditional attribute model.
NO_EDGE_INDEX = 0
EDGE_MASK_INDEX = len(QM9_BOND_TYPES) + 1


def atom_to_index(atomic_num: int, atom_types: tuple[int, ...] = QM9_ATOM_TYPES) -> int:
    return list(atom_types).index(int(atomic_num))


def index_to_atom(index: int, atom_types: tuple[int, ...] = QM9_ATOM_TYPES) -> int:
    return int(atom_types[int(index)])


def bond_type_to_index(bond_type: int, bond_types: tuple[int, ...] = QM9_BOND_TYPES) -> int:
    """Map internal bond type id to dense edge category index.

    Index 0 is reserved for no-edge. Existing bond labels therefore start at 1.
    """
    return list(bond_types).index(int(bond_type)) + 1


def index_to_bond_type(index: int, bond_types: tuple[int, ...] = QM9_BOND_TYPES) -> int:
    if int(index) <= 0:
        raise ValueError("No-edge index does not correspond to an existing bond type.")
    return int(bond_types[int(index) - 1])

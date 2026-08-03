from __future__ import annotations

# QM9 in the standard no-hydrogen setting uses heavy atoms only.
QM9_ATOM_TYPES = (6, 7, 8, 9)  # C, N, O, F

# Integer bond IDs keep serialized graphs independent of RDKit classes.
BOND_SINGLE = 1
BOND_DOUBLE = 2
BOND_TRIPLE = 3
BOND_AROMATIC = 4
QM9_BOND_TYPES = (BOND_SINGLE, BOND_DOUBLE, BOND_TRIPLE, BOND_AROMATIC)

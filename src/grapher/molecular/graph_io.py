from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import numpy as np

from grapher.molecular.constants import (
    BOND_AROMATIC,
    BOND_DOUBLE,
    BOND_SINGLE,
    BOND_TRIPLE,
    QM9_ATOM_TYPES,
    QM9_BOND_TYPES,
)


def require_rdkit():
    try:
        from rdkit import Chem  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional dependency
        raise ImportError(
            "RDKit is required for molecular dataset preparation and validity checks. "
            "Install it with conda install -c conda-forge rdkit or pip install rdkit-pypi."
        ) from exc
    return Chem


def _rdkit_bond_to_internal(bond: Any) -> int:
    Chem = require_rdkit()
    bt = bond.GetBondType()
    if bt == Chem.BondType.SINGLE:
        return BOND_SINGLE
    if bt == Chem.BondType.DOUBLE:
        return BOND_DOUBLE
    if bt == Chem.BondType.TRIPLE:
        return BOND_TRIPLE
    if bt == Chem.BondType.AROMATIC:
        return BOND_AROMATIC
    raise ValueError(f"Unsupported RDKit bond type: {bt}")


def _internal_bond_to_rdkit(bond_type: int):
    Chem = require_rdkit()
    bond_type = int(bond_type)
    if bond_type == BOND_SINGLE:
        return Chem.BondType.SINGLE
    if bond_type == BOND_DOUBLE:
        return Chem.BondType.DOUBLE
    if bond_type == BOND_TRIPLE:
        return Chem.BondType.TRIPLE
    if bond_type == BOND_AROMATIC:
        return Chem.BondType.AROMATIC
    raise ValueError(f"Unsupported internal bond type: {bond_type}")


def mol_to_nx(
    mol: Any,
    *,
    remove_h: bool = True,
    kekulize: bool = True,
    allowed_atoms: tuple[int, ...] = QM9_ATOM_TYPES,
    allowed_bonds: tuple[int, ...] = QM9_BOND_TYPES,
) -> nx.Graph:
    """Convert an RDKit molecule into an attributed NetworkX graph.

    Node attribute:
        atomic_num: integer atomic number.
    Edge attribute:
        bond_type: one of constants.BOND_*.
    """
    Chem = require_rdkit()
    mol = Chem.Mol(mol)
    if remove_h:
        mol = Chem.RemoveHs(mol)
    if kekulize:
        try:
            Chem.Kekulize(mol, clearAromaticFlags=True)
        except Exception:
            # Keep the original molecule if kekulization fails. Sanitization later
            # will decide validity.
            pass

    g = nx.Graph()
    for atom in mol.GetAtoms():
        atomic_num = int(atom.GetAtomicNum())
        if atomic_num not in allowed_atoms:
            raise ValueError(f"Atom {atomic_num} is outside allowed atom set {allowed_atoms}.")
        idx = int(atom.GetIdx())
        g.add_node(idx, atomic_num=atomic_num, atom_type=atomic_num)

    for bond in mol.GetBonds():
        u = int(bond.GetBeginAtomIdx())
        v = int(bond.GetEndAtomIdx())
        bond_type = _rdkit_bond_to_internal(bond)
        if bond_type not in allowed_bonds:
            raise ValueError(f"Bond type {bond_type} is outside allowed set {allowed_bonds}.")
        g.add_edge(u, v, bond_type=bond_type, bond_order=float(bond_type if bond_type != BOND_AROMATIC else 1.5))

    g = nx.convert_node_labels_to_integers(g, ordering="sorted")
    return g


def nx_to_topology(graph: nx.Graph) -> nx.Graph:
    """Strip atom/bond attributes and return a simple topology graph."""
    g = nx.Graph()
    g.add_nodes_from(range(graph.number_of_nodes()))
    g.add_edges_from((int(u), int(v)) for u, v in graph.edges())
    return nx.convert_node_labels_to_integers(g, ordering="sorted")


def nx_to_rdkit_mol(graph: nx.Graph, *, sanitize: bool = True):
    Chem = require_rdkit()
    mol = Chem.RWMol()
    node_map: dict[int, int] = {}
    for node, data in sorted(graph.nodes(data=True)):
        atomic_num = int(data.get("atomic_num", data.get("atom_type", 6)))
        atom = Chem.Atom(atomic_num)
        node_map[int(node)] = int(mol.AddAtom(atom))
    for u, v, data in graph.edges(data=True):
        bond_type = int(data.get("bond_type", BOND_SINGLE))
        mol.AddBond(node_map[int(u)], node_map[int(v)], _internal_bond_to_rdkit(bond_type))
    out = mol.GetMol()
    if sanitize:
        Chem.SanitizeMol(out)
    return out


def graph_to_smiles(graph: nx.Graph, *, canonical: bool = True, sanitize: bool = True) -> str | None:
    Chem = require_rdkit()
    try:
        mol = nx_to_rdkit_mol(graph, sanitize=sanitize)
        return str(Chem.MolToSmiles(mol, canonical=canonical, isomericSmiles=False))
    except Exception:
        return None


def is_valid_molecular_graph(graph: nx.Graph) -> bool:
    return graph_to_smiles(graph, sanitize=True) is not None


def read_smiles_file(path: str | Path, *, smiles_column: str | int | None = None) -> list[str]:
    """Read SMILES from .smi/.txt/.csv-like files.

    If smiles_column is None, the first whitespace-separated token of each row is used.
    For CSV with a header, pass the column name, e.g. --smiles-column smiles.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() in {".csv", ".tsv"} and smiles_column is not None:
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            if isinstance(smiles_column, int):
                rows = list(reader)
                if not rows:
                    return []
                fields = reader.fieldnames or []
                col = fields[int(smiles_column)]
                return [row[col].strip() for row in rows if row.get(col)]
            return [row[str(smiles_column)].strip() for row in reader if row.get(str(smiles_column))]

    smiles: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            token = line.split()[0].strip()
            if token.lower() in {"smiles", "smile"}:
                continue
            smiles.append(token)
    return smiles


def graphs_from_smiles(
    smiles: Iterable[str],
    *,
    remove_h: bool = True,
    kekulize: bool = True,
    allowed_atoms: tuple[int, ...] = QM9_ATOM_TYPES,
) -> tuple[list[nx.Graph], dict[str, int]]:
    Chem = require_rdkit()
    graphs: list[nx.Graph] = []
    errors: dict[str, int] = {}
    for smi in smiles:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                raise ValueError("MolFromSmiles returned None")
            graph = mol_to_nx(mol, remove_h=remove_h, kekulize=kekulize, allowed_atoms=allowed_atoms)
            if graph.number_of_nodes() == 0:
                raise ValueError("empty molecule after preprocessing")
            if graph.number_of_nodes() > 1 and not nx.is_connected(graph):
                raise ValueError("disconnected molecule")
            graphs.append(graph)
        except Exception as exc:
            name = type(exc).__name__
            errors[name] = errors.get(name, 0) + 1
    return graphs, errors


def split_graphs(graphs: list[nx.Graph], *, seed: int, train_frac: float = 0.8, val_frac: float = 0.1):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(graphs)).tolist()
    n = len(idx)
    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))
    train = [graphs[i] for i in idx[:n_train]]
    val = [graphs[i] for i in idx[n_train : n_train + n_val]]
    test = [graphs[i] for i in idx[n_train + n_val :]]
    return {"train": train, "val": val, "test": test}

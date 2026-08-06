#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import networkx as nx

from grapher.data.io import save_dataset_splits
from grapher.molecular.constants import QM9_ATOM_TYPES, QM9_BOND_TYPES
from grapher.molecular.graph_io import (
    graphs_from_smiles,
    nx_to_topology,
    read_smiles_file,
    split_graphs,
)
from grapher.utils.io import ensure_dir, save_json

TOPOLOGY_SCHEMA = {
    "node_attributes": [],
    "edge_attributes": [],
}
ATTRIBUTED_SCHEMA = {
    "node_attributes": ["atomic_num", "atom_type"],
    "edge_attributes": ["bond_type", "bond_order"],
}


def _dataset_output_paths(
    root: str | Path, dataset_names: tuple[str, str]
) -> tuple[Path, Path]:
    """Resolve two distinct dataset directories directly beneath the output root."""
    if len(set(dataset_names)) != len(dataset_names):
        raise ValueError("Topology and attributed dataset names must be different.")

    resolved_root = Path(root).resolve()
    targets: list[Path] = []
    for name in dataset_names:
        relative = Path(name)
        if not name or relative.is_absolute() or len(relative.parts) != 1:
            raise ValueError(
                f"Dataset name must be one directory name, not a path: {name!r}"
            )
        target = (resolved_root / relative).resolve()
        if target == resolved_root or target.parent != resolved_root:
            raise ValueError(f"Unsafe dataset output path: {target}")
        targets.append(target)
    return targets[0], targets[1]


def _clear_dataset_outputs(paths: tuple[Path, Path]) -> list[Path]:
    """Remove old dataset directories after new graph splits are ready to save."""
    removed: list[Path] = []
    for path in paths:
        if path.is_symlink() or path.is_file():
            path.unlink()
            removed.append(path)
        elif path.exists():
            shutil.rmtree(path)
            removed.append(path)
    return removed


def _pyg_bond_type(edge_attr) -> int:
    values = edge_attr.detach().cpu().numpy().reshape(-1).tolist()
    if len(values) >= len(QM9_BOND_TYPES):
        return int(
            QM9_BOND_TYPES[
                int(max(range(len(QM9_BOND_TYPES)), key=lambda i: values[i]))
            ]
        )
    if values:
        val = int(values[0])
        if val in QM9_BOND_TYPES:
            return val
        if 0 <= val < len(QM9_BOND_TYPES):
            return int(QM9_BOND_TYPES[val])
    return int(QM9_BOND_TYPES[0])


def _pyg_data_to_nx(data, *, remove_h: bool = True) -> nx.Graph:
    z = data.z.detach().cpu().numpy().astype(int).tolist()
    keep_old = []
    for idx, atomic_num in enumerate(z):
        if remove_h and int(atomic_num) == 1:
            continue
        if int(atomic_num) not in QM9_ATOM_TYPES:
            raise ValueError(
                f"Atom {atomic_num} is outside allowed atom set {QM9_ATOM_TYPES}."
            )
        keep_old.append(idx)

    node_map = {old: new for new, old in enumerate(keep_old)}
    graph = nx.Graph()
    for old, new in node_map.items():
        atomic_num = int(z[old])
        graph.add_node(new, atomic_num=atomic_num, atom_type=atomic_num)

    edge_index = data.edge_index.detach().cpu().numpy().astype(int)
    edge_attr = getattr(data, "edge_attr", None)
    seen: set[tuple[int, int]] = set()
    for col in range(edge_index.shape[1]):
        u_old = int(edge_index[0, col])
        v_old = int(edge_index[1, col])
        if u_old not in node_map or v_old not in node_map:
            continue
        u = int(node_map[u_old])
        v = int(node_map[v_old])
        if u == v:
            continue
        edge = (u, v) if u < v else (v, u)
        if edge in seen:
            continue
        seen.add(edge)
        bond_type = (
            _pyg_bond_type(edge_attr[col])
            if edge_attr is not None
            else int(QM9_BOND_TYPES[0])
        )
        graph.add_edge(
            edge[0],
            edge[1],
            bond_type=bond_type,
            bond_order=float(bond_type if bond_type != 4 else 1.5),
        )

    return nx.convert_node_labels_to_integers(graph, ordering="sorted")


def _graphs_from_pyg_qm9(
    root: str | Path, *, max_molecules: int | None = None, remove_h: bool = True
) -> tuple[list[nx.Graph], dict[str, int]]:
    try:
        from torch_geometric.datasets.qm9 import QM9  # type: ignore
    except ModuleNotFoundError as exc:
        raise ImportError(
            "PyTorch Geometric is required for --source pyg. Install torch-geometric "
            "or pass --source smiles --smiles-file PATH, or --source sdf --sdf-file PATH."
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            "Could not import torch_geometric.datasets.qm9.QM9. This usually means "
            "the installed PyTorch/PyG versions are incompatible or PyG was installed "
            "without source files needed by TorchScript. Try reinstalling a PyG build "
            "matching your PyTorch/CUDA version, or use --source sdf --sdf-file PATH "
            "or --source smiles --smiles-file PATH."
        ) from exc

    try:
        dataset = QM9(str(root))
    except Exception as exc:
        downloaded_sdf = Path(root) / "raw" / "gdb9.sdf"
        raise RuntimeError(
            "Could not initialize torch_geometric.datasets.QM9. PyG preprocessing may "
            "stop when RDKit cannot parse an individual molecule. If the raw download "
            f"exists, retry with --source sdf --sdf-file {downloaded_sdf}; the direct "
            "SDF loader skips invalid records. For import or TorchScript errors, reinstall "
            "a PyG build matching your PyTorch/CUDA version."
        ) from exc
    limit = (
        len(dataset) if max_molecules is None else min(int(max_molecules), len(dataset))
    )
    graphs: list[nx.Graph] = []
    errors: dict[str, int] = {}
    for idx in range(limit):
        try:
            graph = _pyg_data_to_nx(dataset[idx], remove_h=remove_h)
            if graph.number_of_nodes() == 0:
                raise ValueError("empty molecule after preprocessing")
            if graph.number_of_nodes() > 1 and not nx.is_connected(graph):
                raise ValueError("disconnected molecule")
            graphs.append(graph)
        except Exception as exc:
            name = type(exc).__name__
            errors[name] = errors.get(name, 0) + 1
    return graphs, errors


def _rdkit_bond_type(bond) -> int:
    """Map RDKit bond types to the integer convention used by QM9_BOND_TYPES."""
    try:
        from rdkit import Chem  # type: ignore
    except ModuleNotFoundError as exc:
        raise ImportError("RDKit is required for --source sdf.") from exc

    if bond.GetIsAromatic():
        val = 4
    else:
        bond_type = bond.GetBondType()
        if bond_type == Chem.BondType.SINGLE:
            val = 1
        elif bond_type == Chem.BondType.DOUBLE:
            val = 2
        elif bond_type == Chem.BondType.TRIPLE:
            val = 3
        elif bond_type == Chem.BondType.AROMATIC:
            val = 4
        else:
            val = 1

    if val in QM9_BOND_TYPES:
        return int(val)
    # Fallback for constants encoded as an ordered list rather than explicit bond orders.
    idx = max(0, min(int(val) - 1, len(QM9_BOND_TYPES) - 1))
    return int(QM9_BOND_TYPES[idx])


def _rdkit_mol_to_nx(mol, *, remove_h: bool = True, kekulize: bool = True) -> nx.Graph:
    try:
        from rdkit import Chem  # type: ignore
    except ModuleNotFoundError as exc:
        raise ImportError("RDKit is required for --source sdf.") from exc

    mol = Chem.Mol(mol)
    if kekulize:
        Chem.Kekulize(mol, clearAromaticFlags=True)

    keep_old = []
    for atom in mol.GetAtoms():
        old = int(atom.GetIdx())
        atomic_num = int(atom.GetAtomicNum())
        if remove_h and atomic_num == 1:
            continue
        if atomic_num not in QM9_ATOM_TYPES:
            raise ValueError(
                f"Atom {atomic_num} is outside allowed atom set {QM9_ATOM_TYPES}."
            )
        keep_old.append(old)

    node_map = {old: new for new, old in enumerate(keep_old)}
    graph = nx.Graph()
    for old, new in node_map.items():
        atom = mol.GetAtomWithIdx(int(old))
        atomic_num = int(atom.GetAtomicNum())
        graph.add_node(new, atomic_num=atomic_num, atom_type=atomic_num)

    for bond in mol.GetBonds():
        u_old = int(bond.GetBeginAtomIdx())
        v_old = int(bond.GetEndAtomIdx())
        if u_old not in node_map or v_old not in node_map:
            continue
        u = int(node_map[u_old])
        v = int(node_map[v_old])
        if u == v:
            continue
        bond_type = _rdkit_bond_type(bond)
        graph.add_edge(
            min(u, v),
            max(u, v),
            bond_type=bond_type,
            bond_order=float(1.5 if bond_type == 4 else bond_type),
        )

    return nx.convert_node_labels_to_integers(graph, ordering="sorted")


def _graphs_from_sdf_qm9(
    sdf_file: str | Path,
    *,
    max_molecules: int | None = None,
    remove_h: bool = True,
    kekulize: bool = True,
) -> tuple[list[nx.Graph], dict[str, int], int]:
    try:
        from rdkit import Chem  # type: ignore
    except ModuleNotFoundError as exc:
        raise ImportError(
            "RDKit is required for --source sdf. Install rdkit or use --source smiles."
        ) from exc

    sdf_file = Path(sdf_file)
    if not sdf_file.exists():
        raise FileNotFoundError(f"SDF file does not exist: {sdf_file}")

    supplier = Chem.SDMolSupplier(str(sdf_file), removeHs=False, sanitize=True)
    if supplier is None:
        raise RuntimeError(f"Could not open SDF file: {sdf_file}")

    graphs: list[nx.Graph] = []
    errors: dict[str, int] = {}
    num_seen = 0
    for mol in supplier:
        if max_molecules is not None and num_seen >= int(max_molecules):
            break
        num_seen += 1
        try:
            if mol is None:
                raise ValueError("RDKit returned None for molecule")
            graph = _rdkit_mol_to_nx(mol, remove_h=remove_h, kekulize=kekulize)
            if graph.number_of_nodes() == 0:
                raise ValueError("empty molecule after preprocessing")
            if graph.number_of_nodes() > 1 and not nx.is_connected(graph):
                raise ValueError("disconnected molecule")
            graphs.append(graph)
        except Exception as exc:
            name = type(exc).__name__
            errors[name] = errors.get(name, 0) + 1
    return graphs, errors, num_seen


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare QM9 heavy-atom topology and attributed graph splits."
    )
    parser.add_argument(
        "--source",
        choices=["auto", "pyg", "sdf", "smiles"],
        default="auto",
        help=(
            "Data source. auto uses --smiles-file when provided, otherwise --sdf-file or "
            "<pyg-root>/raw/gdb9.sdf when present, otherwise PyG QM9."
        ),
    )
    parser.add_argument(
        "--smiles-file", default=None, help="Path to .smi/.txt/.csv containing SMILES."
    )
    parser.add_argument(
        "--smiles-column", default=None, help="Optional CSV/TSV column name for SMILES."
    )
    parser.add_argument(
        "--sdf-file",
        default=None,
        help="Path to QM9 gdb9.sdf. Defaults to <pyg-root>/raw/gdb9.sdf for --source sdf.",
    )
    parser.add_argument(
        "--pyg-root",
        default="data/pyg_qm9",
        help="Root directory for torch_geometric.datasets.QM9.",
    )
    parser.add_argument("--root", default="outputs/datasets")
    parser.add_argument(
        "--topology-name",
        default="qm9_topology",
        help="Output dataset name for the topology-only representation (always written).",
    )
    parser.add_argument(
        "--attributed-name",
        default="qm9_attributed",
        help=(
            "Output dataset name for topology with atom and bond attributes "
            "(always written)."
        ),
    )
    parser.add_argument("--max-molecules", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-hydrogens", action="store_true")
    parser.add_argument("--no-kekulize", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    output_paths = _dataset_output_paths(
        root,
        (args.topology_name, args.attributed_name),
    )

    source = args.source
    default_sdf = Path(args.pyg_root) / "raw" / "gdb9.sdf"
    if source == "auto":
        if args.smiles_file:
            source = "smiles"
        elif args.sdf_file or default_sdf.exists():
            source = "sdf"
        else:
            source = "pyg"

    smiles = None
    num_input_records = None
    if source == "smiles":
        if not args.smiles_file:
            raise ValueError("--smiles-file is required when --source smiles.")
        smiles = read_smiles_file(args.smiles_file, smiles_column=args.smiles_column)
        if args.max_molecules:
            smiles = smiles[: int(args.max_molecules)]
        num_input_records = len(smiles)
        graphs, errors = graphs_from_smiles(
            smiles,
            remove_h=not args.keep_hydrogens,
            kekulize=not args.no_kekulize,
        )
        source_path = str(args.smiles_file)
    elif source == "sdf":
        sdf_file = Path(args.sdf_file) if args.sdf_file else default_sdf
        graphs, errors, num_input_records = _graphs_from_sdf_qm9(
            sdf_file,
            max_molecules=args.max_molecules,
            remove_h=not args.keep_hydrogens,
            kekulize=not args.no_kekulize,
        )
        source_path = str(sdf_file)
    else:
        graphs, errors = _graphs_from_pyg_qm9(
            args.pyg_root,
            max_molecules=args.max_molecules,
            remove_h=not args.keep_hydrogens,
        )
        source_path = f"torch_geometric.datasets.QM9:{args.pyg_root}"

    if not graphs:
        raise RuntimeError(
            f"No valid molecules parsed from {source_path}; errors={errors}"
        )

    attributed_splits = split_graphs(graphs, seed=args.seed)
    topology_splits = {
        k: [nx_to_topology(g) for g in v] for k, v in attributed_splits.items()
    }

    config_top = {
        "name": args.topology_name,
        "source": source_path,
        "source_type": source,
        "kind": "qm9_topology",
        "remove_h": not args.keep_hydrogens,
        "kekulize": not args.no_kekulize,
        "seed": args.seed,
    }
    config_attr = dict(config_top)
    config_attr["name"] = args.attributed_name
    config_attr["kind"] = "qm9_attributed"

    removed_paths = _clear_dataset_outputs(output_paths)
    for path in removed_paths:
        print(f"Removed old dataset artifacts: {path}")

    save_dataset_splits(args.topology_name, topology_splits, config_top, root=root)
    save_dataset_splits(args.attributed_name, attributed_splits, config_attr, root=root)
    prep_report = {
        "source_type": source,
        "source": source_path,
        "num_input_smiles": len(smiles) if smiles is not None else None,
        "num_input_records": num_input_records,
        "num_valid_graphs": len(graphs),
        "errors": errors,
        "topology_dataset": args.topology_name,
        "attributed_dataset": args.attributed_name,
        "split_sizes": {k: len(v) for k, v in topology_splits.items()},
        "representations": {
            "topology": TOPOLOGY_SCHEMA,
            "topology_and_attributes": ATTRIBUTED_SCHEMA,
        },
    }
    for dataset_name in (args.topology_name, args.attributed_name):
        save_json(
            prep_report,
            ensure_dir(root / dataset_name) / "qm9_prep_report.json",
        )
    print("Prepared QM9 datasets")
    print(f"  source: {source_path}")
    if smiles is not None:
        print(f"  valid molecules: {len(graphs)} / {len(smiles)}")
    elif num_input_records is not None:
        print(f"  valid molecules: {len(graphs)} / {num_input_records}")
    else:
        print(f"  valid molecules: {len(graphs)}")
    print(f"  errors: {errors}")
    print(
        f"  topology only:         {root / args.topology_name} "
        "(node attributes: none; edge attributes: none)"
    )
    print(
        f"  topology + attributes: {root / args.attributed_name} "
        "(node: atomic_num, atom_type; edge: bond_type, bond_order)"
    )


if __name__ == "__main__":
    main()

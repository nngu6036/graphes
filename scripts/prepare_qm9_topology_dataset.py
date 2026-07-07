#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import networkx as nx

from grapher.data.io import save_dataset_splits
from grapher.molecular.constants import QM9_ATOM_TYPES, QM9_BOND_TYPES
from grapher.molecular.graph_io import graphs_from_smiles, nx_to_topology, read_smiles_file, split_graphs
from grapher.utils.io import ensure_dir, save_json


def _pyg_bond_type(edge_attr) -> int:
    values = edge_attr.detach().cpu().numpy().reshape(-1).tolist()
    if len(values) >= len(QM9_BOND_TYPES):
        return int(QM9_BOND_TYPES[int(max(range(len(QM9_BOND_TYPES)), key=lambda i: values[i]))])
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
            raise ValueError(f"Atom {atomic_num} is outside allowed atom set {QM9_ATOM_TYPES}.")
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
        bond_type = _pyg_bond_type(edge_attr[col]) if edge_attr is not None else int(QM9_BOND_TYPES[0])
        graph.add_edge(edge[0], edge[1], bond_type=bond_type, bond_order=float(bond_type if bond_type != 4 else 1.5))

    return nx.convert_node_labels_to_integers(graph, ordering="sorted")


def _graphs_from_pyg_qm9(root: str | Path, *, max_molecules: int | None = None, remove_h: bool = True) -> tuple[list[nx.Graph], dict[str, int]]:
    try:
        from torch_geometric.datasets import QM9  # type: ignore
    except Exception as exc:
        raise ImportError(
            "PyTorch Geometric is required for --source pyg. Install torch-geometric "
            "or pass --source smiles --smiles-file PATH."
        ) from exc

    dataset = QM9(str(root))
    limit = len(dataset) if max_molecules is None else min(int(max_molecules), len(dataset))
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare QM9 heavy-atom topology and attributed graph splits.")
    parser.add_argument("--source", choices=["auto", "pyg", "smiles"], default="auto", help="Data source. auto uses --smiles-file when provided, otherwise PyG QM9.")
    parser.add_argument("--smiles-file", default=None, help="Path to .smi/.txt/.csv containing SMILES.")
    parser.add_argument("--smiles-column", default=None, help="Optional CSV/TSV column name for SMILES.")
    parser.add_argument("--pyg-root", default="data/pyg_qm9", help="Root directory for torch_geometric.datasets.QM9.")
    parser.add_argument("--root", default="outputs/datasets")
    parser.add_argument("--topology-name", default="qm9_topology")
    parser.add_argument("--attributed-name", default="qm9_attributed")
    parser.add_argument("--max-molecules", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-hydrogens", action="store_true")
    parser.add_argument("--no-kekulize", action="store_true")
    args = parser.parse_args()

    source = args.source
    if source == "auto":
        source = "smiles" if args.smiles_file else "pyg"

    smiles = None
    if source == "smiles":
        if not args.smiles_file:
            raise ValueError("--smiles-file is required when --source smiles.")
        smiles = read_smiles_file(args.smiles_file, smiles_column=args.smiles_column)
        if args.max_molecules:
            smiles = smiles[: int(args.max_molecules)]
        graphs, errors = graphs_from_smiles(
            smiles,
            remove_h=not args.keep_hydrogens,
            kekulize=not args.no_kekulize,
        )
        source_path = str(args.smiles_file)
    else:
        graphs, errors = _graphs_from_pyg_qm9(
            args.pyg_root,
            max_molecules=args.max_molecules,
            remove_h=not args.keep_hydrogens,
        )
        source_path = f"torch_geometric.datasets.QM9:{args.pyg_root}"

    if not graphs:
        raise RuntimeError(f"No valid molecules parsed from {source_path}; errors={errors}")

    attributed_splits = split_graphs(graphs, seed=args.seed)
    topology_splits = {k: [nx_to_topology(g) for g in v] for k, v in attributed_splits.items()}

    root = Path(args.root)
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

    save_dataset_splits(args.topology_name, topology_splits, config_top, root=root)
    save_dataset_splits(args.attributed_name, attributed_splits, config_attr, root=root)
    save_json(
        {
            "source_type": source,
            "source": source_path,
            "num_input_smiles": len(smiles) if smiles is not None else None,
            "num_valid_graphs": len(graphs),
            "errors": errors,
            "topology_dataset": args.topology_name,
            "attributed_dataset": args.attributed_name,
            "split_sizes": {k: len(v) for k, v in topology_splits.items()},
        },
        ensure_dir(root / args.topology_name) / "qm9_prep_report.json",
    )
    print("Prepared QM9 datasets")
    print(f"  source: {source_path}")
    print(f"  valid molecules: {len(graphs)}" + (f" / {len(smiles)}" if smiles is not None else ""))
    print(f"  errors: {errors}")
    print(f"  topology:   {root / args.topology_name}")
    print(f"  attributed: {root / args.attributed_name}")


if __name__ == "__main__":
    main()

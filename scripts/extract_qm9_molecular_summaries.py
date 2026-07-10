#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np

from grapher.data.io import load_dataset_splits
from grapher.molecular.constants import QM9_ATOM_TYPES, QM9_BOND_TYPES, atom_to_index, bond_type_to_index
from grapher.utils.io import ensure_dir, save_json


def molecular_topology_summary(graph: nx.Graph, *, max_cycle: int = 8) -> dict[str, object]:
    g = nx.Graph(graph)
    n = g.number_of_nodes()
    m = g.number_of_edges()
    degrees = np.asarray([d for _, d in g.degree()], dtype=np.int64)
    cycle_basis = nx.cycle_basis(g)
    cycle_hist = np.zeros(max_cycle + 1, dtype=np.float64)
    for cycle in cycle_basis:
        length = len(cycle)
        if length <= max_cycle:
            cycle_hist[length] += 1.0
    return {
        "num_nodes": int(n),
        "num_edges": int(m),
        "density": float(nx.density(g)) if n > 1 else 0.0,
        "num_leaves": int(np.sum(degrees == 1)),
        "num_branch_nodes": int(np.sum(degrees >= 3)),
        "max_degree": int(degrees.max()) if degrees.size else 0,
        "num_cycles": int(len(cycle_basis)),
        "cycle_hist": cycle_hist.tolist(),
        "is_tree": bool(nx.is_tree(g)) if n > 0 else False,
    }


def molecular_attribute_topology_summary(
    graph: nx.Graph,
    *,
    atom_types: tuple[int, ...] = QM9_ATOM_TYPES,
    bond_types: tuple[int, ...] = QM9_BOND_TYPES,
    max_degree: int = 4,
    max_cycle: int = 8,
) -> dict[str, object]:
    g = nx.convert_node_labels_to_integers(nx.Graph(graph), ordering="sorted")
    base = molecular_topology_summary(g, max_cycle=max_cycle)

    atom_hist = np.zeros(len(atom_types), dtype=np.float64)
    typed_degree_hist = np.zeros((len(atom_types), max_degree + 1), dtype=np.float64)
    leaf_atom_hist = np.zeros(len(atom_types), dtype=np.float64)
    branch_atom_hist = np.zeros(len(atom_types), dtype=np.float64)

    for node, data in g.nodes(data=True):
        atomic_num = int(data.get("atomic_num", data.get("atom_type", 6)))
        try:
            atom_idx = atom_to_index(atomic_num, atom_types)
        except ValueError:
            continue
        degree = int(g.degree(node))
        atom_hist[atom_idx] += 1.0
        typed_degree_hist[atom_idx, min(degree, max_degree)] += 1.0
        if degree == 1:
            leaf_atom_hist[atom_idx] += 1.0
        if degree >= 3:
            branch_atom_hist[atom_idx] += 1.0

    bond_hist = np.zeros(len(bond_types), dtype=np.float64)
    pair_keys = []
    for i, atom_a in enumerate(atom_types):
        for atom_b in atom_types[i:]:
            pair_keys.append((atom_a, atom_b))
    pair_to_idx = {pair: idx for idx, pair in enumerate(pair_keys)}
    atom_pair_bond_hist = np.zeros((len(pair_keys), len(bond_types)), dtype=np.float64)

    for u, v, data in g.edges(data=True):
        atom_u = int(g.nodes[u].get("atomic_num", g.nodes[u].get("atom_type", 6)))
        atom_v = int(g.nodes[v].get("atomic_num", g.nodes[v].get("atom_type", 6)))
        pair = tuple(sorted((atom_u, atom_v)))
        bond_type = int(data.get("bond_type", 1))
        try:
            bond_idx = bond_type_to_index(bond_type, bond_types) - 1
        except ValueError:
            continue
        bond_hist[bond_idx] += 1.0
        if pair in pair_to_idx:
            atom_pair_bond_hist[pair_to_idx[pair], bond_idx] += 1.0

    out = dict(base)
    out.update(
        {
            "atom_hist": atom_hist.tolist(),
            "typed_degree_hist": typed_degree_hist.reshape(-1).tolist(),
            "leaf_atom_hist": leaf_atom_hist.tolist(),
            "branch_atom_hist": branch_atom_hist.tolist(),
            "bond_hist": bond_hist.tolist(),
            "atom_pair_bond_hist": atom_pair_bond_hist.reshape(-1).tolist(),
            "atom_pair_keys": [list(pair) for pair in pair_keys],
        }
    )
    return out


def aggregate(rows):
    if not rows:
        return {}
    out = {"num_graphs": len(rows)}
    keys = [k for k, v in rows[0].items() if isinstance(v, (int, float, bool))]
    for k in keys:
        vals = np.asarray([float(r[k]) for r in rows], dtype=float)
        out[k + "_mean"] = float(vals.mean())
        out[k + "_std"] = float(vals.std())
    for k, v in rows[0].items():
        if isinstance(v, list) and v and isinstance(v[0], (int, float)):
            mat = np.asarray([r[k] for r in rows], dtype=float)
            out[k + "_mean"] = mat.mean(axis=0).tolist()
            out[k + "_std"] = mat.std(axis=0).tolist()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract topology and attribute-related molecular summaries for QM9.")
    parser.add_argument("--dataset", default="qm9_attributed")
    parser.add_argument("--root", default="outputs/datasets")
    parser.add_argument("--output-dir", default="outputs/molecular_summaries/qm9")
    parser.add_argument("--max-graphs", type=int, default=None)
    args = parser.parse_args()

    splits = load_dataset_splits(args.dataset, root=args.root, build_if_missing=False)
    out_dir = ensure_dir(args.output_dir)
    report = {}
    for split, graphs in splits.items():
        graphs = list(graphs)
        if args.max_graphs:
            graphs = graphs[: int(args.max_graphs)]
        topo_rows = [molecular_topology_summary(g) for g in graphs]
        attr_rows = [molecular_attribute_topology_summary(g) for g in graphs]
        with (out_dir / f"{split}_topology_summaries.jsonl").open("w", encoding="utf-8") as f:
            for row in topo_rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")
        with (out_dir / f"{split}_attribute_topology_summaries.jsonl").open("w", encoding="utf-8") as f:
            for row in attr_rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")
        report[split] = {
            "topology": aggregate(topo_rows),
            "attribute_topology": aggregate(attr_rows),
        }
    save_json(report, out_dir / "summary_report.json")
    print(f"Saved molecular summaries to {out_dir}")


if __name__ == "__main__":
    main()

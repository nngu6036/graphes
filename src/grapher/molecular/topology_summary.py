from __future__ import annotations

import networkx as nx
import numpy as np

from grapher.molecular.constants import QM9_ATOM_TYPES, QM9_BOND_TYPES, atom_to_index, bond_type_to_index


def molecular_topology_summary(graph: nx.Graph, *, max_cycle: int = 8) -> dict[str, object]:
    """Permutation-invariant topology summary for molecular skeletons.

    This works for both attributed and unattributed molecular topology graphs.
    It intentionally contains topology-only fields that can be generated before
    atom and bond labels are available.
    """
    g = nx.Graph(graph)
    n = g.number_of_nodes()
    m = g.number_of_edges()
    degrees = np.asarray([d for _, d in g.degree()], dtype=np.int64)
    cycle_basis = nx.cycle_basis(g)
    cycle_hist = np.zeros(max_cycle + 1, dtype=np.float64)
    for cyc in cycle_basis:
        L = len(cyc)
        if L <= max_cycle:
            cycle_hist[L] += 1.0
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
    """Attribute-related topology summary for molecular training graphs.

    These summaries are permutation-invariant but they require atom/bond labels.
    They are useful as *conditioning targets* for a molecular topology generator:
    the topology stage can be asked to generate a skeleton compatible with likely
    atom counts, typed degrees, leaf atoms, and bond orders.

    The first implementation records the fields but does not yet condition the
    generic topology generator on them. That conditioning is the next design step.
    """
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
        deg = int(g.degree(node))
        atom_hist[atom_idx] += 1.0
        typed_degree_hist[atom_idx, min(deg, max_degree)] += 1.0
        if deg == 1:
            leaf_atom_hist[atom_idx] += 1.0
        if deg >= 3:
            branch_atom_hist[atom_idx] += 1.0

    bond_hist = np.zeros(len(bond_types), dtype=np.float64)
    # unordered atom-pair x bond-type compatibility counts
    pair_keys = []
    for i, a in enumerate(atom_types):
        for j, b in enumerate(atom_types[i:], start=i):
            pair_keys.append((a, b))
    pair_to_idx = {pair: idx for idx, pair in enumerate(pair_keys)}
    atom_pair_bond_hist = np.zeros((len(pair_keys), len(bond_types)), dtype=np.float64)

    for u, v, data in g.edges(data=True):
        au = int(g.nodes[u].get("atomic_num", g.nodes[u].get("atom_type", 6)))
        av = int(g.nodes[v].get("atomic_num", g.nodes[v].get("atom_type", 6)))
        pair = tuple(sorted((au, av)))
        b = int(data.get("bond_type", 1))
        try:
            bond_idx = bond_type_to_index(b, bond_types) - 1
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

"""Strict numeric interchange for CatFlow, GSDM, EDGE and SPECTRE.

Only the parent GraphER process reads its trusted dataset pickles. Workers see
train/validation NPZ arrays, never the held-out test graphs. Padding is explicit;
zero-degree sampled vertices are not silently removed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import networkx as nx
import numpy as np


@dataclass(frozen=True)
class GraphProfile:
    benchmark_id: str
    serialized_id: str
    max_nodes: int
    atomic_numbers: tuple[int, ...] = ()
    bond_types: tuple[int, ...] = ()

    @property
    def domain(self) -> str:
        return "attributed" if self.atomic_numbers else "generic"

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "domain": self.domain}


PROFILES = {
    "community_small": GraphProfile("community_small", "sbm", 20),
    "ego_small": GraphProfile("ego_small", "ego_small", 18),
    "grid": GraphProfile("grid", "grid", 361),
    "qm9": GraphProfile("qm9", "qm9_attributed", 9, (6, 7, 8, 9), (1, 2, 3)),
    "zinc": GraphProfile("zinc", "zinc", 38, (6, 7, 8, 9, 15, 16, 17, 35, 53), (1, 2, 3)),
}


def encode_graphs(graphs: Sequence[nx.Graph], profile: GraphProfile) -> dict[str, np.ndarray]:
    count, width = len(graphs), profile.max_nodes
    adj = np.zeros((count, width, width), dtype=np.int8)
    sizes = np.zeros(count, dtype=np.int64)
    types = np.full((count, width), -1, dtype=np.int16)
    atom_map = {z: i for i, z in enumerate(profile.atomic_numbers)}
    for i, graph in enumerate(graphs):
        if not isinstance(graph, nx.Graph) or graph.is_directed() or graph.is_multigraph():
            raise ValueError(f"Graph {i} must be a simple undirected NetworkX graph.")
        if nx.number_of_selfloops(graph):
            raise ValueError(f"Graph {i} has self-loops; preprocessing must be explicit.")
        n = len(graph)
        if not 1 <= n <= width:
            raise ValueError(f"Graph {i} has {n} nodes; configured support is 1..{width}.")
        nodes = list(graph)
        indices = {v: j for j, v in enumerate(nodes)}
        sizes[i] = n
        for j, v in enumerate(nodes):
            if atom_map:
                attrs = graph.nodes[v]
                z = attrs.get("atomic_num", attrs.get("atom_type"))
                if z not in atom_map:
                    raise ValueError(f"Graph {i}, node {v}: unsupported atom {z!r}.")
                types[i, j] = atom_map[z]
            else:
                types[i, j] = 0
        for u, v, attrs in graph.edges(data=True):
            value = 1
            if atom_map:
                raw = attrs.get("bond_type", attrs.get("bond_order"))
                if raw is None or float(raw) != int(raw) or int(raw) not in profile.bond_types:
                    raise ValueError(f"Graph {i}: unsupported bond {raw!r}; use the frozen kekulized split.")
                value = int(raw)
            a, b = indices[u], indices[v]
            adj[i, a, b] = adj[i, b, a] = value
    return {"adjacency": adj, "num_nodes": sizes, "node_types": types}


def save_graphs(path: Path, graphs: Sequence[nx.Graph], profile: GraphProfile) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **encode_graphs(graphs, profile))


def decode_graphs(path: Path, profile: GraphProfile, expected_count: int) -> list[nx.Graph]:
    with np.load(path, allow_pickle=False) as values:
        required = {"adjacency", "num_nodes", "node_types"}
        if not required.issubset(values.files):
            raise ValueError(f"Export is missing arrays: {sorted(required.difference(values.files))}.")
        arrays = {key: np.asarray(values[key]) for key in required}
    a, n, x = arrays["adjacency"], arrays["num_nodes"], arrays["node_types"]
    m = profile.max_nodes
    if a.shape != (expected_count, m, m) or n.shape != (expected_count,) or x.shape != (expected_count, m):
        raise ValueError(f"Wrong export dimensions/count: adjacency={a.shape}, sizes={n.shape}, nodes={x.shape}.")
    if any(v.dtype.kind not in "iu" for v in arrays.values()):
        raise TypeError("All exported categorical arrays must have integer dtype.")
    if np.any(n < 1) or np.any(n > m):
        raise ValueError("Exported graph sizes are outside the configured support.")
    if not np.array_equal(a, a.transpose(0, 2, 1)) or np.any(np.diagonal(a, axis1=1, axis2=2)):
        raise ValueError("Exported adjacency must be symmetric with a zero diagonal.")
    permitted = (0,) + (profile.bond_types or (1,))
    if not np.isin(a, permitted).all():
        raise ValueError("Invalid edge category in export.")
    graphs = []
    for i, raw_size in enumerate(n):
        size = int(raw_size)
        if np.any(a[i, size:, :]) or np.any(a[i, :, size:]) or np.any(x[i, size:] != -1):
            raise ValueError("Nonzero edges or non-sentinel node categories in padding.")
        graph = nx.Graph()
        graph.add_nodes_from(range(size))
        for v in range(size):
            category = int(x[i, v])
            if profile.atomic_numbers:
                if not 0 <= category < len(profile.atomic_numbers):
                    raise ValueError("Invalid atom category in export.")
                z = profile.atomic_numbers[category]
                # Charges are not predicted by these releases. Do not copy target
                # charges into generated samples or apply an unreported repair.
                graph.nodes[v].update(atomic_num=z, atom_type=z, formal_charge=0)
            elif category != 0:
                raise ValueError("Generic node categories must be zero.")
        for u, v in zip(*np.nonzero(np.triu(a[i, :size, :size], 1))):
            attrs = {}
            if profile.atomic_numbers:
                bond = int(a[i, u, v])
                attrs = {"bond_type": bond, "bond_order": float(bond)}
            graph.add_edge(int(u), int(v), **attrs)
        graphs.append(graph)
    return graphs

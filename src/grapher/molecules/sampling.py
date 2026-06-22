from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import random
from typing import Any, Mapping, Sequence

import networkx as nx

from grapher.generation.molecular_rewiring import (
    EmpiricalBondTypePrior,
    deterministic_connected_havel_hakimi,
    node_type_value,
    set_edge_type,
)


@dataclass
class EmpiricalAtomTypePrior:
    """Empirical p(atomic_number | topological degree) proposal prior."""

    degree_counts: dict[int, Counter[int]]
    global_counts: Counter[int]
    allowed_node_types: list[int]
    smoothing: float = 0.0

    @classmethod
    def fit(
        cls,
        graphs: Sequence[nx.Graph],
        *,
        allowed_node_types: Sequence[int] | None = None,
        smoothing: float = 0.1,
    ) -> "EmpiricalAtomTypePrior":
        degree_counts: dict[int, Counter[int]] = defaultdict(Counter)
        global_counts: Counter[int] = Counter()
        allowed = [int(v) for v in allowed_node_types] if allowed_node_types else []
        for raw_graph in graphs:
            graph = nx.convert_node_labels_to_integers(nx.Graph(raw_graph), ordering="sorted")
            for node, degree in graph.degree():
                atom_type = node_type_value(graph, int(node))
                if allowed and atom_type not in allowed:
                    continue
                degree_counts[int(degree)][int(atom_type)] += 1
                global_counts[int(atom_type)] += 1
        if not allowed:
            allowed = sorted(global_counts)
        if not global_counts and allowed:
            for value in allowed:
                global_counts[int(value)] += 1
        if not global_counts:
            raise ValueError("Cannot fit atom prior: no molecular node types were found.")
        return cls(
            degree_counts={int(k): Counter(v) for k, v in degree_counts.items()},
            global_counts=Counter(global_counts),
            allowed_node_types=sorted(int(v) for v in allowed),
            smoothing=max(float(smoothing), 0.0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "conditioning": "p(atomic_number | topological degree)",
            "degree_counts": {str(k): {str(t): int(c) for t, c in v.items()} for k, v in self.degree_counts.items()},
            "global_counts": {str(k): int(v) for k, v in self.global_counts.items()},
            "allowed_node_types": [int(v) for v in self.allowed_node_types],
            "smoothing": float(self.smoothing),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EmpiricalAtomTypePrior":
        return cls(
            degree_counts={int(k): Counter({int(t): int(c) for t, c in v.items()}) for k, v in dict(payload.get("degree_counts", {})).items()},
            global_counts=Counter({int(k): int(v) for k, v in dict(payload.get("global_counts", {})).items()}),
            allowed_node_types=[int(v) for v in payload.get("allowed_node_types", [])],
            smoothing=float(payload.get("smoothing", 0.0)),
        )

    def _counts(self, degree: int) -> Counter[int]:
        counts = Counter(self.degree_counts.get(int(degree), {}))
        if not counts:
            counts = Counter(self.global_counts)
        for atom_type in self.allowed_node_types:
            counts[int(atom_type)] += self.smoothing
        return counts

    def sample(self, degree: int, *, rng: random.Random) -> int:
        counts = self._counts(int(degree))
        values = sorted(int(v) for v in counts)
        weights = [max(float(counts[v]), 0.0) + 1e-12 for v in values]
        return int(rng.choices(values, weights=weights, k=1)[0])


def _assign_initial_bonds(
    graph: nx.Graph,
    bond_prior: EmpiricalBondTypePrior,
    *,
    rng: random.Random,
    source_edge_type_strategy: str,
    allow_global_bond_backoff: bool,
    valence_tolerance: float,
) -> nx.Graph | None:
    source = graph.copy()
    current_valence = {int(node): 0.0 for node in source.nodes()}
    edge_types_by_order = sorted(
        bond_prior.edge_types,
        key=lambda label: (bond_prior.bond_order(int(label)), int(label)),
    )
    edges = [tuple(sorted((int(u), int(v)))) for u, v in source.edges()]
    rng.shuffle(edges)
    # Endpoint pairs with fewer empirical options are assigned first.
    edges.sort(
        key=lambda edge: len(
            bond_prior.counts_for_pair(
                node_type_value(source, edge[0]),
                node_type_value(source, edge[1]),
                allow_global_backoff=allow_global_bond_backoff,
            )
        )
    )
    source.remove_edges_from(edges)
    for u, v in edges:
        left = node_type_value(source, u)
        right = node_type_value(source, v)
        proposed = bond_prior.proposal_types(
            left,
            right,
            rng=rng,
            count=max(len(edge_types_by_order), 1),
            mode=source_edge_type_strategy,
            allow_global_backoff=allow_global_bond_backoff,
        )
        candidates = list(dict.fromkeys([int(x) for x in proposed] + [int(x) for x in edge_types_by_order]))
        feasible = []
        for label in candidates:
            order = bond_prior.bond_order(label)
            if current_valence[u] + order <= bond_prior.max_valence(left) + float(valence_tolerance) and current_valence[v] + order <= bond_prior.max_valence(right) + float(valence_tolerance):
                feasible.append(int(label))
        if not feasible:
            return None
        if str(source_edge_type_strategy).lower() == "sample" and len(feasible) > 1:
            counts = bond_prior.counts_for_pair(left, right, allow_global_backoff=allow_global_bond_backoff)
            weights = [float(counts.get(label, 0)) + bond_prior.smoothing + 1e-12 for label in feasible]
            label = int(rng.choices(feasible, weights=weights, k=1)[0])
        else:
            label = int(feasible[0])
        order = bond_prior.bond_order(label)
        set_edge_type(source, (u, v), label, bond_order=order)
        current_valence[u] += order
        current_valence[v] += order
    return source


def initialize_generated_molecular_source(
    degree_sequence: Sequence[int],
    atom_prior: EmpiricalAtomTypePrior,
    bond_prior: EmpiricalBondTypePrior,
    *,
    rng: random.Random,
    source_edge_type_strategy: str = "sample",
    allow_global_bond_backoff: bool = True,
    valence_tolerance: float = 1e-6,
    node_assignment_attempts: int = 16,
) -> nx.Graph:
    """Build a target-free molecular source for generation.

    The topology is a connected Havel-Hakimi realization of the sampled degree
    sequence.  Node types are sampled from p(atomic_number | degree), then kept
    fixed throughout the MolecularGraphER trajectory.  Initial bond labels are
    sampled from p(edge_type | endpoint atom types) and must satisfy the fitted
    valence limits.
    """

    seq = [int(d) for d in degree_sequence]
    topology = deterministic_connected_havel_hakimi(seq=seq)
    for attempt in range(max(int(node_assignment_attempts), 1)):
        graph = nx.Graph()
        graph.graph.update({"molecular_source": "generated_connected_havel_hakimi_empirical_atom_bond_prior"})
        for node in topology.nodes():
            atom_type = atom_prior.sample(int(topology.degree(int(node))), rng=rng)
            graph.add_node(
                int(node),
                atomic_number=int(atom_type),
                z=int(atom_type),
                node_label=f"atomic_number={int(atom_type)}",
                feats=[float(atom_type)],
            )
        graph.add_edges_from((int(u), int(v)) for u, v in topology.edges())
        with_bonds = _assign_initial_bonds(
            graph,
            bond_prior,
            rng=rng,
            source_edge_type_strategy=source_edge_type_strategy,
            allow_global_bond_backoff=allow_global_bond_backoff,
            valence_tolerance=valence_tolerance,
        )
        if with_bonds is not None:
            return nx.convert_node_labels_to_integers(with_bonds, ordering="sorted")
    raise RuntimeError("Could not initialize a valence-feasible generated molecular source.")

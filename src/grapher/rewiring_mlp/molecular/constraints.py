from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import networkx as nx
import numpy as np

from grapher.rewiring_mlp.molecular.constants import (
    BOND_AROMATIC,
    BOND_DOUBLE,
    BOND_SINGLE,
    BOND_TRIPLE,
    QM9_ATOM_TYPES,
)

# Neutral heavy-atom valences used by the hydrogen-suppressed QM9 graphs.
DEFAULT_MAX_VALENCE = {
    1: 1.0,
    6: 4.0,
    7: 3.0,
    8: 2.0,
    9: 1.0,
}

# QM9's categorical graph state intentionally projects formal charge.  Its
# observed/generated state can therefore contain the charged representations
# N(+1) at bond-order valence four and O(+1) at valence three even though the
# corresponding neutral-atom limits above remain lower.  Keep this envelope
# separate so generic molecular initialization retains neutral defaults.
QM9_PROJECTED_MAX_VALENCE = {
    1: 1.0,
    6: 4.0,
    7: 4.0,
    8: 3.0,
    9: 1.0,
}
DEFAULT_GENERATED_BOND_TYPES = (BOND_SINGLE, BOND_DOUBLE, BOND_TRIPLE)


def bond_order(bond_type: int) -> float:
    value = int(bond_type)
    if value == BOND_SINGLE:
        return 1.0
    if value == BOND_DOUBLE:
        return 2.0
    if value == BOND_TRIPLE:
        return 3.0
    if value == BOND_AROMATIC:
        return 1.5
    raise ValueError(f"Unsupported molecular bond type: {bond_type!r}.")


def _atomic_number(data: dict[str, Any]) -> int | None:
    value = data.get("atomic_num", data.get("atom_type"))
    return None if value is None else int(value)


def _bond_type(data: dict[str, Any]) -> int | None:
    value = data.get("bond_type")
    if value is not None:
        return int(value)
    order = data.get("bond_order")
    if order is None:
        return None
    numeric = float(order)
    if abs(numeric - 1.5) <= 1.0e-8:
        return BOND_AROMATIC
    rounded = int(round(numeric))
    return rounded if rounded in {1, 2, 3} else None


@dataclass(frozen=True)
class MolecularAttributePriors:
    atom_by_degree: dict[int, dict[int, int]]
    atom_global: dict[int, int]
    bond_by_atom_pair: dict[tuple[int, int], dict[int, int]]
    bond_global: dict[int, int]


def fit_molecular_attribute_priors(
    graphs: Sequence[nx.Graph],
    *,
    allowed_atom_types: Iterable[int] = QM9_ATOM_TYPES,
    allowed_bond_types: Iterable[int] = DEFAULT_GENERATED_BOND_TYPES,
) -> MolecularAttributePriors:
    """Fit empirical QM9 atom/bond priors without using validation or test data."""

    atom_types = {int(value) for value in allowed_atom_types}
    bond_types = {int(value) for value in allowed_bond_types}
    atom_by_degree: defaultdict[int, Counter[int]] = defaultdict(Counter)
    atom_global: Counter[int] = Counter()
    bond_by_pair: defaultdict[tuple[int, int], Counter[int]] = defaultdict(Counter)
    bond_global: Counter[int] = Counter()

    for graph in graphs:
        for node, data in graph.nodes(data=True):
            atomic_num = _atomic_number(data)
            if atomic_num in atom_types:
                degree = int(graph.degree(node))
                atom_by_degree[degree][atomic_num] += 1
                atom_global[atomic_num] += 1
        for u, v, data in graph.edges(data=True):
            category = _bond_type(data)
            atom_u = _atomic_number(graph.nodes[u])
            atom_v = _atomic_number(graph.nodes[v])
            if (
                category not in bond_types
                or atom_u not in atom_types
                or atom_v not in atom_types
            ):
                continue
            pair = tuple(sorted((int(atom_u), int(atom_v))))
            bond_by_pair[pair][int(category)] += 1
            bond_global[int(category)] += 1

    if not atom_global:
        raise ValueError(
            "Cannot initialize molecular attributes: the training graphs contain "
            "no supported atomic_num/atom_type values."
        )
    if not bond_global:
        # An all-single fallback is chemically valid and also supports datasets
        # containing only isolated heavy atoms.
        bond_global[BOND_SINGLE] = 1
    return MolecularAttributePriors(
        atom_by_degree={
            int(degree): dict(counts) for degree, counts in atom_by_degree.items()
        },
        atom_global=dict(atom_global),
        bond_by_atom_pair={
            tuple(pair): dict(counts) for pair, counts in bond_by_pair.items()
        },
        bond_global=dict(bond_global),
    )


def _draw_category(
    choices: Sequence[int],
    counts: dict[int, int],
    fallback_counts: dict[int, int],
    *,
    rng: np.random.Generator,
    sample: bool,
    smoothing: float,
) -> int:
    if not choices:
        raise ValueError("Cannot draw from an empty categorical support.")
    primary = np.asarray(
        [float(counts.get(int(value), 0)) for value in choices],
        dtype=np.float64,
    )
    fallback = np.asarray(
        [float(fallback_counts.get(int(value), 0)) for value in choices],
        dtype=np.float64,
    )
    if float(primary.sum()) > 0.0:
        fallback = fallback / max(float(fallback.sum()), 1.0)
        weights = primary + float(smoothing) * float(primary.sum()) * fallback
    else:
        weights = fallback + float(smoothing)
    if not np.isfinite(weights).all() or float(weights.sum()) <= 0.0:
        weights = np.ones(len(choices), dtype=np.float64)
    if not sample:
        return int(choices[int(np.argmax(weights))])
    weights /= float(weights.sum())
    return int(rng.choice(np.asarray(choices, dtype=np.int64), p=weights))


def initialize_molecular_attributes(
    graph: nx.Graph,
    priors: MolecularAttributePriors,
    *,
    rng: np.random.Generator,
    allowed_atom_types: Iterable[int] = QM9_ATOM_TYPES,
    allowed_bond_types: Iterable[int] = DEFAULT_GENERATED_BOND_TYPES,
    max_valence: dict[int, float] | None = None,
    sample: bool = True,
    smoothing: float = 0.05,
    force_single_bonds: bool = False,
) -> nx.Graph:
    """Attach complete, valence-feasible atom and bond categories to a topology.

    Atom types are sampled from the training distribution conditional on
    topological degree. Bonds are initialized jointly with the remaining
    valence at both endpoints, so a sampled double/triple bond cannot overfill
    either atom. No validation/test attributes are used.
    """

    out = nx.convert_node_labels_to_integers(
        nx.Graph(graph),
        first_label=0,
        ordering="sorted",
    )
    valence = {
        int(key): float(value)
        for key, value in (max_valence or DEFAULT_MAX_VALENCE).items()
    }
    atom_types = tuple(
        int(value) for value in allowed_atom_types if int(value) in valence
    )
    bond_types = tuple(
        int(value)
        for value in allowed_bond_types
        if int(value) in DEFAULT_GENERATED_BOND_TYPES
    )
    if not atom_types:
        raise ValueError("No supported molecular atom type is configured.")
    if BOND_SINGLE not in bond_types:
        raise ValueError("Molecular initialization requires the single-bond type.")

    assigned_atoms: dict[int, int] = {}
    for node in sorted(out.nodes()):
        degree = int(out.degree(node))
        feasible = [
            atomic_num
            for atomic_num in atom_types
            if valence[atomic_num] + 1.0e-8 >= degree
        ]
        if not feasible:
            raise ValueError(
                f"No configured atom type can support topological degree {degree}."
            )
        atomic_num = _draw_category(
            feasible,
            priors.atom_by_degree.get(degree, {}),
            priors.atom_global,
            rng=rng,
            sample=sample,
            smoothing=smoothing,
        )
        assigned_atoms[int(node)] = atomic_num
        out.nodes[node]["atomic_num"] = atomic_num
        out.nodes[node]["atom_type"] = atomic_num

    # Every edge starts as a single bond. The degree-compatible atom sampling
    # above makes this base assignment valence feasible.
    used_valence = {int(node): float(out.degree(node)) for node in out.nodes()}
    for u, v in out.edges():
        out.edges[u, v]["bond_type"] = BOND_SINGLE
        out.edges[u, v]["bond_order"] = 1.0

    edges = list(out.edges())
    if sample:
        rng.shuffle(edges)
    for u, v in edges:
        if force_single_bonds:
            continue
        atom_u = assigned_atoms[int(u)]
        atom_v = assigned_atoms[int(v)]
        capacity = min(
            valence[atom_u] - used_valence[int(u)],
            valence[atom_v] - used_valence[int(v)],
        )
        feasible_bonds = [
            category
            for category in bond_types
            if bond_order(category) - 1.0 <= capacity + 1.0e-8
        ]
        pair = tuple(sorted((atom_u, atom_v)))
        category = _draw_category(
            feasible_bonds,
            priors.bond_by_atom_pair.get(pair, {}),
            priors.bond_global,
            rng=rng,
            sample=sample,
            smoothing=smoothing,
        )
        order = bond_order(category)
        increment = order - 1.0
        used_valence[int(u)] += increment
        used_valence[int(v)] += increment
        out.edges[u, v]["bond_type"] = category
        out.edges[u, v]["bond_order"] = order

    errors = molecular_valence_errors(
        out,
        allowed_atom_types=atom_types,
        allowed_bond_types=bond_types,
        max_valence=valence,
    )
    if errors:
        raise RuntimeError(
            "Molecular attribute initialization produced an infeasible graph: "
            + "; ".join(errors[:3])
        )
    out.graph.update(graph.graph)
    out.graph["molecular_attribute_initializer"] = "empirical_valence_constrained"
    return out


def molecular_valence_errors(
    graph: nx.Graph,
    *,
    allowed_atom_types: Iterable[int] = QM9_ATOM_TYPES,
    allowed_bond_types: Iterable[int] = DEFAULT_GENERATED_BOND_TYPES,
    max_valence: dict[int, float] | None = None,
) -> list[str]:
    """Return missing-category, unsupported-category, and valence violations."""

    atoms = {int(value) for value in allowed_atom_types}
    bonds = {int(value) for value in allowed_bond_types}
    valence = {
        int(key): float(value)
        for key, value in (max_valence or DEFAULT_MAX_VALENCE).items()
    }
    errors: list[str] = []
    used = {int(node): 0.0 for node in graph.nodes()}

    for node, data in graph.nodes(data=True):
        atomic_num = _atomic_number(data)
        if atomic_num is None:
            errors.append(f"node {node}: missing atomic_num/atom_type")
        elif atomic_num not in atoms or atomic_num not in valence:
            errors.append(f"node {node}: unsupported atom type {atomic_num}")

    for u, v, data in graph.edges(data=True):
        category = _bond_type(data)
        if category is None:
            errors.append(f"edge ({u}, {v}): missing bond_type/bond_order")
            continue
        if category not in bonds:
            errors.append(f"edge ({u}, {v}): unsupported bond type {category}")
            continue
        order = bond_order(category)
        used[int(u)] += order
        used[int(v)] += order

    for node, data in graph.nodes(data=True):
        atomic_num = _atomic_number(data)
        if atomic_num in valence and used[int(node)] > valence[atomic_num] + 1.0e-8:
            errors.append(
                f"node {node}: bond-order valence {used[int(node)]:g} exceeds "
                f"{valence[atomic_num]:g} for atom {atomic_num}"
            )
    return errors


def is_molecular_valence_feasible(
    graph: nx.Graph,
    **kwargs: Any,
) -> bool:
    return not molecular_valence_errors(graph, **kwargs)

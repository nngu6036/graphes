from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
import random
from typing import Any, Iterable, Mapping, Sequence

import networkx as nx
import numpy as np
import torch

from grapher.generation.rewiring import (
    RewireAction,
    action_new_edges,
    action_removed_edges,
    action_signature,
    action_structural_delta,
    degree_sequence,
    deterministic_connected_havel_hakimi,
    enumerate_rewire_actions,
    laplacian_positional_encoding,
)


DEFAULT_BOND_ORDER_BY_TYPE: dict[int, float] = {
    1: 1.0,  # single
    2: 2.0,  # double
    3: 3.0,  # triple
    4: 1.5,  # aromatic
}


def canonical_edge(edge: tuple[int, int]) -> tuple[int, int]:
    return tuple(sorted((int(edge[0]), int(edge[1]))))


def node_type_value(graph: nx.Graph, node: int) -> int:
    """Return a stable molecular node type, preferring atomic number.

    Prepared QM9 and ZINC graphs retain ``atomic_number``/``z`` even after the
    generic attribute canonicalizer maps ``node_label`` to a compact id.  Using
    atomic number keeps the endpoint-conditioned bond prior interpretable and
    consistent across datasets.
    """

    data = graph.nodes[int(node)]
    for key in ("atomic_number", "z", "node_label", "atom_type"):
        value = data.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            pass
    raise KeyError(
        f"Node {node!r} is missing a usable molecular type. Expected one of "
        "atomic_number, z, node_label, atom_type."
    )


def edge_type_value(graph: nx.Graph, u: int, v: int) -> int:
    data = graph.edges[int(u), int(v)]
    value = data.get("edge_type")
    if value is None:
        value = data.get("bond_type")
    if value is None:
        raw = data.get("edge_attr")
        if torch.is_tensor(raw):
            raw = raw.detach().cpu().reshape(-1).tolist()
        if isinstance(raw, np.ndarray):
            raw = raw.reshape(-1).tolist()
        if isinstance(raw, (list, tuple)) and raw:
            value = raw[0]
    if value is None:
        raise KeyError(f"Edge {(u, v)} is missing edge_type/bond_type/edge_attr.")
    return int(value)


def set_edge_type(graph: nx.Graph, edge: tuple[int, int], edge_type: int, *, bond_order: float) -> None:
    u, v = canonical_edge(edge)
    graph.add_edge(
        u,
        v,
        edge_type=int(edge_type),
        edge_attr=[float(edge_type)],
        bond_order=float(bond_order),
    )


def parse_bond_order_mapping(raw: Mapping[Any, Any] | None) -> dict[int, float]:
    mapping = dict(DEFAULT_BOND_ORDER_BY_TYPE)
    if raw:
        for key, value in raw.items():
            mapping[int(key)] = float(value)
    return mapping


@dataclass(frozen=True)
class MolecularRewireAction:
    """A typed GraphER action ``(e1,e2,r,c1,c2)``.

    Node types are not modified. ``new_edge_types`` contains the categorical
    labels assigned to the two new edges returned by ``action_new_edges``.
    """

    topology: RewireAction
    new_edge_types: tuple[int, int]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "new_edge_types",
            (int(self.new_edge_types[0]), int(self.new_edge_types[1])),
        )

    @property
    def e1(self) -> tuple[int, int]:
        return self.topology.e1

    @property
    def e2(self) -> tuple[int, int]:
        return self.topology.e2

    @property
    def orientation(self) -> int:
        return int(self.topology.orientation)

    def as_tuple(
        self,
    ) -> tuple[tuple[int, int], tuple[int, int], int, int, int]:
        return self.e1, self.e2, self.orientation, *self.new_edge_types


def molecular_action_signature(
    action: MolecularRewireAction,
) -> tuple[tuple[int, int], tuple[int, int], int, int, int]:
    return (
        action.e1,
        action.e2,
        int(action.orientation),
        int(action.new_edge_types[0]),
        int(action.new_edge_types[1]),
    )


@dataclass
class EmpiricalBondTypePrior:
    """Empirical ``p(edge_type | unordered endpoint node types)``.

    The prior is used as a proposal distribution. The neural scorer still sees
    complete typed actions and can choose among the sampled feasible proposals.
    Valence limits are fitted from the maximum observed bond-order sum for each
    node type in the training split.
    """

    pair_counts: dict[tuple[int, int], Counter[int]]
    global_counts: Counter[int]
    node_type_counts: Counter[int]
    max_valence_by_node_type: dict[int, float]
    bond_order_by_type: dict[int, float]
    smoothing: float = 0.0

    @property
    def edge_types(self) -> list[int]:
        return sorted(int(value) for value in self.global_counts)

    @property
    def node_types(self) -> list[int]:
        return sorted(int(value) for value in self.node_type_counts)

    @classmethod
    def fit(
        cls,
        graphs: Sequence[nx.Graph],
        *,
        bond_order_by_type: Mapping[int, float] | None = None,
        max_valence_by_node_type: Mapping[int, float] | None = None,
        smoothing: float = 0.0,
    ) -> "EmpiricalBondTypePrior":
        bond_orders = parse_bond_order_mapping(bond_order_by_type)
        pair_counts: dict[tuple[int, int], Counter[int]] = defaultdict(Counter)
        global_counts: Counter[int] = Counter()
        node_type_counts: Counter[int] = Counter()
        max_valence: dict[int, float] = {}

        for raw_graph in graphs:
            graph = nx.convert_node_labels_to_integers(nx.Graph(raw_graph), ordering="sorted")
            current_valence = {int(node): 0.0 for node in graph.nodes()}
            for node in graph.nodes():
                node_type_counts[node_type_value(graph, int(node))] += 1
            for u, v in graph.edges():
                bond_type = edge_type_value(graph, int(u), int(v))
                if bond_type not in bond_orders:
                    # Unknown labels are conservatively treated as single bonds;
                    # users can override this in the model YAML.
                    bond_orders[bond_type] = 1.0
                pair = tuple(sorted((node_type_value(graph, int(u)), node_type_value(graph, int(v)))))
                pair_counts[pair][bond_type] += 1
                global_counts[bond_type] += 1
                order = float(bond_orders[bond_type])
                current_valence[int(u)] += order
                current_valence[int(v)] += order
            for node, value in current_valence.items():
                atom_type = node_type_value(graph, int(node))
                max_valence[atom_type] = max(float(value), float(max_valence.get(atom_type, 0.0)))

        if not global_counts:
            raise ValueError("Cannot fit molecular bond prior: no typed edges were found in the training graphs.")
        if max_valence_by_node_type:
            for node_type, value in max_valence_by_node_type.items():
                value = float(value)
                if value <= 0.0:
                    raise ValueError(
                        f"Maximum valence for node type {node_type!r} must be positive; got {value}."
                    )
                max_valence[int(node_type)] = value
        return cls(
            pair_counts={key: Counter(value) for key, value in pair_counts.items()},
            global_counts=Counter(global_counts),
            node_type_counts=Counter(node_type_counts),
            max_valence_by_node_type={int(k): float(v) for k, v in max_valence.items()},
            bond_order_by_type={int(k): float(v) for k, v in bond_orders.items()},
            smoothing=max(float(smoothing), 0.0),
        )

    def bond_order(self, edge_type: int) -> float:
        return float(self.bond_order_by_type.get(int(edge_type), 1.0))

    def max_valence(self, node_type: int) -> float:
        # Fall back to the largest observed valence. This is only needed for a
        # node type absent from the fitted split, which should normally not occur.
        fallback = max(self.max_valence_by_node_type.values(), default=4.0)
        return float(self.max_valence_by_node_type.get(int(node_type), fallback))

    def counts_for_pair(
        self,
        left_type: int,
        right_type: int,
        *,
        allow_global_backoff: bool = True,
    ) -> Counter[int]:
        pair = tuple(sorted((int(left_type), int(right_type))))
        counts = self.pair_counts.get(pair)
        if counts:
            return Counter(counts)
        return Counter(self.global_counts) if allow_global_backoff else Counter()

    def proposal_types(
        self,
        left_type: int,
        right_type: int,
        *,
        rng: random.Random,
        count: int = 1,
        mode: str = "sample",
        include: Iterable[int] = (),
        allow_global_backoff: bool = True,
    ) -> list[int]:
        counts = self.counts_for_pair(
            left_type,
            right_type,
            allow_global_backoff=allow_global_backoff,
        )
        requested = max(int(count), 1)
        chosen: list[int] = []
        for raw in include:
            value = int(raw)
            if value not in chosen:
                chosen.append(value)

        if not counts:
            return chosen[:requested]

        values = sorted(int(value) for value in counts)
        weights = [float(counts[value]) + self.smoothing for value in values]
        mode_key = str(mode).strip().lower()
        if mode_key in {"mode", "top", "topk"}:
            ordered = sorted(values, key=lambda value: (-weights[values.index(value)], value))
            for value in ordered:
                if value not in chosen:
                    chosen.append(value)
                if len(chosen) >= requested:
                    break
        elif mode_key == "sample":
            attempts = 0
            while len(chosen) < requested and attempts < max(20, requested * 10):
                attempts += 1
                value = int(rng.choices(values, weights=weights, k=1)[0])
                if value not in chosen:
                    chosen.append(value)
            # If the requested number exceeds the support, append the remaining
            # labels by empirical frequency.
            for value in sorted(values, key=lambda x: (-counts[x], x)):
                if value not in chosen:
                    chosen.append(value)
                if len(chosen) >= requested:
                    break
        else:
            raise ValueError(f"Unknown bond proposal mode {mode!r}; expected 'sample' or 'topk'.")
        return chosen[:requested]

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_counts": {
                f"{left}|{right}": {str(label): int(count) for label, count in counts.items()}
                for (left, right), counts in self.pair_counts.items()
            },
            "global_counts": {str(label): int(count) for label, count in self.global_counts.items()},
            "node_type_counts": {str(label): int(count) for label, count in self.node_type_counts.items()},
            "max_valence_by_node_type": {
                str(label): float(value) for label, value in self.max_valence_by_node_type.items()
            },
            "bond_order_by_type": {str(label): float(value) for label, value in self.bond_order_by_type.items()},
            "smoothing": float(self.smoothing),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EmpiricalBondTypePrior":
        pair_counts: dict[tuple[int, int], Counter[int]] = {}
        for key, counts in dict(payload.get("pair_counts", {})).items():
            left, right = str(key).split("|", 1)
            pair_counts[(int(left), int(right))] = Counter(
                {int(label): int(count) for label, count in dict(counts).items()}
            )
        return cls(
            pair_counts=pair_counts,
            global_counts=Counter(
                {int(label): int(count) for label, count in dict(payload.get("global_counts", {})).items()}
            ),
            node_type_counts=Counter(
                {int(label): int(count) for label, count in dict(payload.get("node_type_counts", {})).items()}
            ),
            max_valence_by_node_type={
                int(label): float(value)
                for label, value in dict(payload.get("max_valence_by_node_type", {})).items()
            },
            bond_order_by_type={
                int(label): float(value)
                for label, value in dict(payload.get("bond_order_by_type", {})).items()
            },
            smoothing=float(payload.get("smoothing", 0.0)),
        )


def node_valence(graph: nx.Graph, node: int, prior: EmpiricalBondTypePrior) -> float:
    total = 0.0
    for neighbour in graph.neighbors(int(node)):
        total += prior.bond_order(edge_type_value(graph, int(node), int(neighbour)))
    return float(total)


def _typed_action_valence_valid(
    graph: nx.Graph,
    action: MolecularRewireAction,
    prior: EmpiricalBondTypePrior,
    *,
    tolerance: float = 1e-6,
) -> bool:
    removed = action_removed_edges(action.topology)
    added = action_new_edges(action.topology)
    removed_order: dict[int, float] = defaultdict(float)
    added_order: dict[int, float] = defaultdict(float)

    for u, v in removed:
        order = prior.bond_order(edge_type_value(graph, u, v))
        removed_order[int(u)] += order
        removed_order[int(v)] += order
    for (u, v), edge_type in zip(added, action.new_edge_types):
        order = prior.bond_order(int(edge_type))
        added_order[int(u)] += order
        added_order[int(v)] += order

    for node in set(removed_order) | set(added_order):
        atom_type = node_type_value(graph, int(node))
        updated = node_valence(graph, int(node), prior) - removed_order[node] + added_order[node]
        if updated > prior.max_valence(atom_type) + float(tolerance):
            return False
        if updated < -float(tolerance):
            return False
    return True


def apply_molecular_rewire(
    graph: nx.Graph,
    action: MolecularRewireAction,
    prior: EmpiricalBondTypePrior,
    *,
    ensure_connected: bool = True,
    valence_tolerance: float = 1e-6,
) -> nx.Graph | None:
    """Apply a typed double-edge swap while keeping all node attributes fixed."""

    topology = action.topology
    removed = action_removed_edges(topology)
    added = action_new_edges(topology)
    if len(set(removed[0] + removed[1])) != 4:
        return None
    if not all(graph.has_edge(*edge) for edge in removed):
        return None
    if any(graph.has_edge(*edge) for edge in added):
        return None
    if not _typed_action_valence_valid(
        graph,
        action,
        prior,
        tolerance=valence_tolerance,
    ):
        return None

    candidate = graph.copy()
    candidate.remove_edge(*removed[0])
    candidate.remove_edge(*removed[1])
    for edge, edge_type in zip(added, action.new_edge_types):
        set_edge_type(
            candidate,
            edge,
            int(edge_type),
            bond_order=prior.bond_order(int(edge_type)),
        )
    if ensure_connected and candidate.number_of_nodes() > 1 and not nx.is_connected(candidate):
        return None
    return candidate


def initialize_molecular_havel_hakimi(
    target: nx.Graph,
    prior: EmpiricalBondTypePrior,
    *,
    rng: random.Random,
    proposal_mode: str = "sample",
    allow_global_backoff: bool = True,
    max_attempts: int = 8,
    valence_tolerance: float = 1e-6,
) -> nx.Graph:
    """Build the HH topology, copy node types, and initialize bond labels.

    Bond labels are sampled from the endpoint-conditioned empirical prior. A
    fixed RNG seed makes the source reproducible for a training run. If a sampled
    high-order bond would exhaust valence, lower-order feasible labels are tried.
    """

    target = nx.convert_node_labels_to_integers(nx.Graph(target), ordering="sorted")
    topology = deterministic_connected_havel_hakimi(G=target)
    node_attrs = {int(node): dict(target.nodes[int(node)]) for node in target.nodes()}
    edge_types_by_order = sorted(prior.edge_types, key=lambda label: (prior.bond_order(label), label))
    if not edge_types_by_order:
        raise ValueError("Cannot initialize molecular HH source without observed edge types.")

    for _ in range(max(int(max_attempts), 1)):
        source = nx.Graph()
        source.graph.update(target.graph)
        source.graph["molecular_source"] = "connected_havel_hakimi_empirical_bond_prior"
        source.add_nodes_from((node, dict(attrs)) for node, attrs in node_attrs.items())
        edges = [canonical_edge(edge) for edge in topology.edges()]
        # Harder endpoint pairs first; ties are randomized by the seeded RNG.
        rng.shuffle(edges)
        edges.sort(
            key=lambda edge: len(
                prior.counts_for_pair(
                    node_type_value(source, edge[0]),
                    node_type_value(source, edge[1]),
                    allow_global_backoff=allow_global_backoff,
                )
            )
        )
        success = True
        current_valence = {int(node): 0.0 for node in source.nodes()}
        for u, v in edges:
            left_type = node_type_value(source, u)
            right_type = node_type_value(source, v)
            proposed = prior.proposal_types(
                left_type,
                right_type,
                rng=rng,
                count=max(len(prior.edge_types), 1),
                mode=proposal_mode,
                allow_global_backoff=allow_global_backoff,
            )
            # Guarantee a low-order fallback while preserving empirical ordering.
            candidates = list(dict.fromkeys(proposed + edge_types_by_order))
            feasible = [
                label
                for label in candidates
                if current_valence[u] + prior.bond_order(label)
                <= prior.max_valence(left_type) + valence_tolerance
                and current_valence[v] + prior.bond_order(label)
                <= prior.max_valence(right_type) + valence_tolerance
            ]
            if not feasible:
                success = False
                break
            if proposal_mode.lower() == "sample" and len(feasible) > 1:
                counts = prior.counts_for_pair(left_type, right_type, allow_global_backoff=allow_global_backoff)
                weights = [float(counts.get(label, 0)) + prior.smoothing + 1e-12 for label in feasible]
                label = int(rng.choices(feasible, weights=weights, k=1)[0])
            else:
                label = int(feasible[0])
            order = prior.bond_order(label)
            set_edge_type(source, (u, v), label, bond_order=order)
            current_valence[u] += order
            current_valence[v] += order
        if success and source.number_of_edges() == topology.number_of_edges():
            return nx.convert_node_labels_to_integers(source, ordering="sorted")

    raise RuntimeError(
        "Could not assign valence-feasible empirical bond types to the connected Havel-Hakimi source. "
        "Try source_edge_type_strategy='topk', allow_global_bond_backoff=true, or inspect the atom/bond schema."
    )


def _target_label_for_edge(target: nx.Graph | None, edge: tuple[int, int]) -> int | None:
    if target is None or not target.has_edge(*edge):
        return None
    return edge_type_value(target, *edge)


def expand_typed_actions(
    graph: nx.Graph,
    topology_actions: Sequence[RewireAction],
    prior: EmpiricalBondTypePrior,
    *,
    rng: random.Random,
    proposals_per_edge: int = 2,
    proposal_mode: str = "sample",
    max_candidates: int | None = None,
    ensure_connected: bool = True,
    valence_tolerance: float = 1e-6,
    target: nx.Graph | None = None,
    include_target_edge_types: bool = False,
    allow_global_backoff: bool = True,
    reject_unseen_endpoint_pairs: bool = False,
) -> list[MolecularRewireAction]:
    """Expand topology actions over empirical endpoint-conditioned bond labels."""

    actions: list[MolecularRewireAction] = []
    seen: set[tuple[tuple[int, int], tuple[int, int], int, int, int]] = set()
    for topology in topology_actions:
        new_edges = action_new_edges(topology)
        label_sets: list[list[int]] = []
        skip_topology = False
        for edge in new_edges:
            left = node_type_value(graph, edge[0])
            right = node_type_value(graph, edge[1])
            raw_counts = prior.counts_for_pair(left, right, allow_global_backoff=False)
            if reject_unseen_endpoint_pairs and not raw_counts:
                skip_topology = True
                break
            target_label = _target_label_for_edge(target, edge) if include_target_edge_types else None
            include = [] if target_label is None else [target_label]
            labels = prior.proposal_types(
                left,
                right,
                rng=rng,
                count=max(int(proposals_per_edge), len(include), 1),
                mode=proposal_mode,
                include=include,
                allow_global_backoff=allow_global_backoff,
            )
            if not labels:
                skip_topology = True
                break
            label_sets.append(labels)
        if skip_topology:
            continue

        for first_label in label_sets[0]:
            for second_label in label_sets[1]:
                action = MolecularRewireAction(
                    topology=topology,
                    new_edge_types=(int(first_label), int(second_label)),
                )
                signature = molecular_action_signature(action)
                if signature in seen:
                    continue
                if apply_molecular_rewire(
                    graph,
                    action,
                    prior,
                    ensure_connected=ensure_connected,
                    valence_tolerance=valence_tolerance,
                ) is None:
                    continue
                seen.add(signature)
                actions.append(action)
                if max_candidates is not None and len(actions) >= int(max_candidates):
                    return actions
    return actions


def enumerate_molecular_rewire_actions(
    graph: nx.Graph,
    prior: EmpiricalBondTypePrior,
    *,
    rng: random.Random,
    ensure_connected: bool = True,
    k_hop: int | None = 2,
    max_candidates: int | None = None,
    topology_budget: int | None = None,
    anchor_edges: Sequence[tuple[int, int]] | None = None,
    proposals_per_edge: int = 2,
    proposal_mode: str = "sample",
    target: nx.Graph | None = None,
    include_target_edge_types: bool = False,
    allow_global_backoff: bool = True,
    reject_unseen_endpoint_pairs: bool = False,
    valence_tolerance: float = 1e-6,
    shuffle: bool = True,
) -> list[MolecularRewireAction]:
    if topology_budget is None:
        if max_candidates is None:
            topology_budget = None
        else:
            divisor = max(int(proposals_per_edge) ** 2, 1)
            topology_budget = max(int(math.ceil(int(max_candidates) / divisor)) * 2, 1)
    topology_actions = enumerate_rewire_actions(
        graph,
        ensure_connected=ensure_connected,
        k_hop=k_hop,
        max_candidates=topology_budget,
        anchor_edges=anchor_edges,
        rng=rng,
        shuffle=shuffle,
    )
    typed = expand_typed_actions(
        graph,
        topology_actions,
        prior,
        rng=rng,
        proposals_per_edge=proposals_per_edge,
        proposal_mode=proposal_mode,
        max_candidates=max_candidates,
        ensure_connected=ensure_connected,
        valence_tolerance=valence_tolerance,
        target=target,
        include_target_edge_types=include_target_edge_types,
        allow_global_backoff=allow_global_backoff,
        reject_unseen_endpoint_pairs=reject_unseen_endpoint_pairs,
    )
    if shuffle:
        rng.shuffle(typed)
    return typed


def merge_molecular_action_sets(
    *action_sets: Sequence[MolecularRewireAction],
    max_candidates: int | None = None,
) -> list[MolecularRewireAction]:
    merged: list[MolecularRewireAction] = []
    seen: set[tuple[tuple[int, int], tuple[int, int], int, int, int]] = set()
    for action_set in action_sets:
        for action in action_set:
            signature = molecular_action_signature(action)
            if signature in seen:
                continue
            seen.add(signature)
            merged.append(action)
            if max_candidates is not None and len(merged) >= int(max_candidates):
                return merged
    return merged


def typed_edge_mismatch_count(left: nx.Graph, right: nx.Graph) -> int:
    left_edges = {canonical_edge(edge) for edge in left.edges()}
    right_edges = {canonical_edge(edge) for edge in right.edges()}
    common = left_edges & right_edges
    return sum(
        int(edge_type_value(left, *edge) != edge_type_value(right, *edge))
        for edge in common
    )


def attributed_edge_discrepancy(
    current: nx.Graph,
    target: nx.Graph,
    *,
    topology_weight: float = 1.0,
    bond_type_weight: float = 1.0,
    clustering_weight: float = 0.0,
) -> float:
    current_edges = {canonical_edge(edge) for edge in current.edges()}
    target_edges = {canonical_edge(edge) for edge in target.edges()}
    topology_distance = float(len(current_edges ^ target_edges)) / float(max(2 * target.number_of_edges(), 1))
    bond_distance = float(typed_edge_mismatch_count(current, target)) / float(max(target.number_of_edges(), 1))
    clustering_distance = abs(float(nx.average_clustering(current)) - float(nx.average_clustering(target)))
    return float(
        float(topology_weight) * topology_distance
        + float(bond_type_weight) * bond_distance
        + float(clustering_weight) * clustering_distance
    )


@dataclass
class MolecularData:
    node_types: torch.Tensor
    degree_features: torch.Tensor
    edge_index: torch.Tensor
    edge_types: torch.Tensor
    pe: torch.Tensor

    def to(self, device: torch.device | str) -> "MolecularData":
        self.node_types = self.node_types.to(device)
        self.degree_features = self.degree_features.to(device)
        self.edge_index = self.edge_index.to(device)
        self.edge_types = self.edge_types.to(device)
        self.pe = self.pe.to(device)
        return self


def molecular_graph_to_data(
    graph: nx.Graph,
    *,
    node_type_to_index: Mapping[int, int],
    edge_type_to_index: Mapping[int, int],
    k_eigen: int = 4,
) -> MolecularData:
    graph = nx.convert_node_labels_to_integers(nx.Graph(graph), ordering="sorted")
    nodes = list(graph.nodes())
    node_types = torch.tensor(
        [int(node_type_to_index[node_type_value(graph, int(node))]) for node in nodes],
        dtype=torch.long,
    )
    max_degree = max(graph.number_of_nodes() - 1, 1)
    degree_features = torch.tensor(
        [[float(graph.degree(int(node))) / float(max_degree)] for node in nodes],
        dtype=torch.float32,
    )
    pe = laplacian_positional_encoding(graph, nodes, int(k_eigen))

    directed_edges: list[tuple[int, int]] = []
    directed_types: list[int] = []
    for u, v in graph.edges():
        raw_type = edge_type_value(graph, int(u), int(v))
        if raw_type not in edge_type_to_index:
            raise KeyError(f"Edge type {raw_type} is absent from the fitted molecular vocabulary.")
        type_index = int(edge_type_to_index[raw_type])
        directed_edges.extend([(int(u), int(v)), (int(v), int(u))])
        directed_types.extend([type_index, type_index])
    if directed_edges:
        edge_index = torch.tensor(directed_edges, dtype=torch.long).t().contiguous()
        edge_types = torch.tensor(directed_types, dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_types = torch.empty((0,), dtype=torch.long)
    return MolecularData(
        node_types=node_types,
        degree_features=degree_features,
        edge_index=edge_index,
        edge_types=edge_types,
        pe=pe,
    )


def molecular_local_feature_matrix(
    graph: nx.Graph,
    actions: Sequence[MolecularRewireAction],
    prior: EmpiricalBondTypePrior,
    *,
    local_feature_dim: int,
) -> torch.Tensor:
    """Compute topology, bond-order, and remaining-valence action features."""

    if not actions:
        return torch.empty((0, int(local_feature_dim)), dtype=torch.float32)
    n = max(int(graph.number_of_nodes()), 1)
    max_degree = max(n - 1, 1)
    neighbours = {int(node): {int(v) for v in graph.neighbors(node)} for node in graph.nodes()}
    degrees = {int(node): int(value) for node, value in graph.degree()}
    valences = {int(node): node_valence(graph, int(node), prior) for node in graph.nodes()}
    try:
        bridges = {canonical_edge(edge) for edge in nx.bridges(graph)}
    except Exception:
        bridges = set()

    rows: list[list[float]] = []
    for action in actions:
        topology = action.topology
        structural = action_structural_delta(
            graph,
            topology,
            neighbours=neighbours,
            degrees=degrees,
        )
        removed = action_removed_edges(topology)
        added = action_new_edges(topology)
        endpoints = [topology.e1[0], topology.e1[1], topology.e2[0], topology.e2[1]]
        values: list[float] = [float(degrees[int(node)]) / float(max_degree) for node in endpoints]
        values.extend(
            [
                float(structural.added_common_e1) / float(n),
                float(structural.added_common_e2) / float(n),
                float(structural.removed_common_e1) / float(n),
                float(structural.removed_common_e2) / float(n),
                float(structural.triangle_delta) / float(n),
                float(structural.average_clustering_delta),
                1.0 if canonical_edge(topology.e1) in bridges or canonical_edge(topology.e2) in bridges else 0.0,
            ]
        )

        removed_order: dict[int, float] = defaultdict(float)
        for edge in removed:
            order = prior.bond_order(edge_type_value(graph, *edge))
            removed_order[edge[0]] += order
            removed_order[edge[1]] += order
        added_order: dict[int, float] = defaultdict(float)
        for edge, edge_type in zip(added, action.new_edge_types):
            order = prior.bond_order(edge_type)
            added_order[edge[0]] += order
            added_order[edge[1]] += order

        for node in endpoints:
            atom_type = node_type_value(graph, int(node))
            limit = max(prior.max_valence(atom_type), 1e-12)
            values.append(float(valences[int(node)]) / limit)
        for edge_type in action.new_edge_types:
            values.append(float(prior.bond_order(edge_type)) / 3.0)
        for node in endpoints:
            atom_type = node_type_value(graph, int(node))
            limit = max(prior.max_valence(atom_type), 1e-12)
            after = valences[int(node)] - removed_order[int(node)] + added_order[int(node)]
            values.append(float(max(limit - after, 0.0)) / limit)

        if len(values) < int(local_feature_dim):
            values.extend([0.0] * (int(local_feature_dim) - len(values)))
        rows.append(values[: int(local_feature_dim)])
    return torch.tensor(rows, dtype=torch.float32)

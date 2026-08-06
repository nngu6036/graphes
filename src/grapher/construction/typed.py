from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import networkx as nx
import numpy as np

from grapher.molecular.typed_invariants import (
    TypedDegreeSignature,
    TypedInvariant,
    extract_typed_invariant,
    typed_invariant_errors,
    typed_invariant_matches_graph,
)

EndpointCompatibility = Callable[[Any, Any, Any], bool]
InitializationScore = Callable[[Any, TypedDegreeSignature, TypedDegreeSignature], float]


@dataclass(frozen=True)
class TypedConstructorConfig:
    """Search settings for exact simultaneous typed-degree realization.

    ``candidate_ranking="uniform"`` randomly orders every feasible partner.
    ``candidate_ranking="empirical"`` uses ``initialization_score`` as a
    non-negative weight proportional to the training-only initialization prior
    :math:`p_init(r | tau_i, tau_j)`.  Ranking affects search order only; a
    zero-weight candidate is never removed from the exact feasibility mask.
    """

    ensure_connected: bool = True
    randomize_assignment: bool = True
    max_restarts: int = 32
    max_backtracks: int = 50_000
    candidate_ranking: str = "uniform"
    candidate_temperature: float = 1.0
    max_ordinary_degree: int | None = None
    max_weighted_valence: dict[Any, float] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None = None) -> "TypedConstructorConfig":
        data = data or {}
        ranking = str(
            data.get(
                "candidate_ranking",
                data.get("ranking", data.get("selection", "uniform")),
            )
        ).lower()
        config = cls(
            ensure_connected=bool(data.get("ensure_connected", True)),
            randomize_assignment=bool(data.get("randomize_assignment", True)),
            max_restarts=int(data.get("max_restarts", 32)),
            max_backtracks=int(data.get("max_backtracks", 50_000)),
            candidate_ranking=ranking,
            candidate_temperature=float(data.get("candidate_temperature", 1.0)),
            max_ordinary_degree=(
                None
                if data.get("max_ordinary_degree") in {None, "", "none", "None"}
                else int(data["max_ordinary_degree"])
            ),
            max_weighted_valence=(
                None
                if data.get("max_weighted_valence") is None
                else {
                    key: float(value)
                    for key, value in dict(data["max_weighted_valence"]).items()
                }
            ),
        )
        if config.max_restarts < 0:
            raise ValueError("max_restarts must be non-negative.")
        if config.max_backtracks < 1:
            raise ValueError("max_backtracks must be positive.")
        if config.candidate_ranking not in {"uniform", "empirical"}:
            raise ValueError(
                "candidate_ranking must be either 'uniform' or 'empirical'."
            )
        if config.candidate_temperature < 0.0:
            raise ValueError("candidate_temperature must be non-negative.")
        return config


class TypedConstructionError(RuntimeError):
    """Raised when no exact realization is found within the declared budget."""

    def __init__(self, message: str, diagnostics: dict[str, Any]):
        super().__init__(message)
        self.diagnostics = diagnostics


@dataclass
class _SearchBudget:
    limit: int
    backtracks: int = 0
    search_nodes: int = 0
    candidate_trials: int = 0
    exhausted: bool = False

    def record_backtrack(self) -> None:
        self.backtracks += 1
        if self.backtracks >= self.limit:
            self.exhausted = True


def _is_compatible(
    left: TypedDegreeSignature,
    right: TypedDegreeSignature,
    edge_type: Any,
    endpoint_compatible: EndpointCompatibility | None,
) -> bool:
    if endpoint_compatible is None:
        return True
    # An undirected compatibility relation must permit both endpoint orders.
    return bool(
        endpoint_compatible(left.node_type, right.node_type, edge_type)
        and endpoint_compatible(right.node_type, left.node_type, edge_type)
    )


def _available_partners(
    graph: nx.Graph,
    residual: list[list[int]],
    signatures: Sequence[TypedDegreeSignature],
    edge_types: Sequence[Any],
    node: int,
    edge_index: int,
    endpoint_compatible: EndpointCompatibility | None,
) -> list[int]:
    edge_type = edge_types[edge_index]
    return [
        other
        for other in range(len(signatures))
        if other != node
        and residual[other][edge_index] > 0
        and not graph.has_edge(node, other)
        and _is_compatible(
            signatures[node],
            signatures[other],
            edge_type,
            endpoint_compatible,
        )
    ]


def _pair_has_residual_edge(
    graph: nx.Graph,
    residual: list[list[int]],
    signatures: Sequence[TypedDegreeSignature],
    edge_types: Sequence[Any],
    left: int,
    right: int,
    endpoint_compatible: EndpointCompatibility | None,
) -> bool:
    if left == right or graph.has_edge(left, right):
        return False
    return any(
        residual[left][column] > 0
        and residual[right][column] > 0
        and _is_compatible(
            signatures[left],
            signatures[right],
            edge_type,
            endpoint_compatible,
        )
        for column, edge_type in enumerate(edge_types)
    )


def _completion_possible(
    graph: nx.Graph,
    residual: list[list[int]],
    signatures: Sequence[TypedDegreeSignature],
    edge_types: Sequence[Any],
    *,
    ensure_connected: bool,
    endpoint_compatible: EndpointCompatibility | None,
) -> bool:
    """Apply inexpensive necessary masks before descending the search tree."""

    node_count = len(signatures)
    residual_degrees = [sum(row) for row in residual]
    if any(value < 0 for row in residual for value in row):
        return False

    # Each type must remain graphical even before accounting for occupied and
    # compatibility-forbidden pairs.  This is necessary, not sufficient.
    for column, _edge_type in enumerate(edge_types):
        sequence = [row[column] for row in residual]
        if sum(sequence) % 2:
            return False
        if any(value >= node_count for value in sequence):
            return False
        if any(sequence) and not nx.is_graphical(sequence, method="eg"):
            return False

        available_pairs = 0
        for left in range(node_count):
            partners = _available_partners(
                graph,
                residual,
                signatures,
                edge_types,
                left,
                column,
                endpoint_compatible,
            )
            if residual[left][column] > len(partners):
                return False
            available_pairs += len(partners)
        if sum(sequence) // 2 > available_pairs // 2:
            return False

    if sum(residual_degrees) % 2:
        return False
    if any(residual_degrees) and not nx.is_graphical(residual_degrees, method="eg"):
        return False

    # Since at most one edge of any type may occupy a pair, aggregate residual
    # demand at a node cannot exceed the number of still-compatible partners.
    for node, demand in enumerate(residual_degrees):
        if demand == 0:
            continue
        possible = sum(
            _pair_has_residual_edge(
                graph,
                residual,
                signatures,
                edge_types,
                node,
                other,
                endpoint_compatible,
            )
            for other in range(node_count)
        )
        if demand > possible:
            return False

    if not ensure_connected or node_count <= 1:
        return True
    components = [set(value) for value in nx.connected_components(graph)]
    if len(components) == 1:
        return True
    remaining_edges = sum(residual_degrees) // 2
    if remaining_edges < len(components) - 1:
        return False

    # Contract each current component and require the graph of possible future
    # cross-component edges to remain connected.
    component_of: dict[int, int] = {}
    for index, component in enumerate(components):
        for node in component:
            component_of[int(node)] = index
    possible_components = nx.Graph()
    possible_components.add_nodes_from(range(len(components)))
    for left in range(node_count):
        for right in range(left + 1, node_count):
            left_component = component_of[left]
            right_component = component_of[right]
            if left_component == right_component:
                continue
            if _pair_has_residual_edge(
                graph,
                residual,
                signatures,
                edge_types,
                left,
                right,
                endpoint_compatible,
            ):
                possible_components.add_edge(left_component, right_component)
    return nx.is_connected(possible_components)


def _select_active_demand(
    graph: nx.Graph,
    residual: list[list[int]],
    signatures: Sequence[TypedDegreeSignature],
    edge_types: Sequence[Any],
    endpoint_compatible: EndpointCompatibility | None,
    rng: np.random.Generator,
) -> tuple[int, int, list[int]] | None:
    choices: list[tuple[int, int, int, int, list[int]]] = []
    for node, row in enumerate(residual):
        for column, demand in enumerate(row):
            if demand <= 0:
                continue
            partners = _available_partners(
                graph,
                residual,
                signatures,
                edge_types,
                node,
                column,
                endpoint_compatible,
            )
            # Minimum slack is the fail-first residual-demand heuristic.  The
            # negative demand then prioritizes the largest constrained cell.
            choices.append((len(partners) - demand, -demand, node, column, partners))
    if not choices:
        return None
    best_key = min((item[0], item[1]) for item in choices)
    tied = [item for item in choices if (item[0], item[1]) == best_key]
    chosen = tied[int(rng.integers(0, len(tied)))]
    return chosen[2], chosen[3], chosen[4]


def _order_partners(
    partners: list[int],
    *,
    node: int,
    edge_index: int,
    signatures: Sequence[TypedDegreeSignature],
    edge_types: Sequence[Any],
    config: TypedConstructorConfig,
    initialization_score: InitializationScore | None,
    rng: np.random.Generator,
) -> list[int]:
    ordered = list(partners)
    rng.shuffle(ordered)
    if config.candidate_ranking == "uniform":
        return ordered
    if initialization_score is None:
        raise ValueError("candidate_ranking='empirical' requires initialization_score.")
    weights = np.asarray(
        [
            float(
                initialization_score(
                    edge_types[edge_index],
                    signatures[node],
                    signatures[other],
                )
            )
            for other in ordered
        ],
        dtype=np.float64,
    )
    if not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise ValueError(
            "initialization_score must return finite non-negative weights."
        )
    if config.candidate_temperature == 0.0:
        return [
            ordered[index] for index in np.argsort(-weights, kind="stable").tolist()
        ]
    # A Gumbel ranking samples without replacement from weights while retaining
    # every zero-weight candidate as an exact-search fallback.
    priorities = np.log(np.maximum(weights, 1.0e-12)) / max(
        config.candidate_temperature, 1.0e-12
    ) + rng.gumbel(size=len(ordered))
    return [ordered[index] for index in np.argsort(-priorities).tolist()]


def _search_realization(
    graph: nx.Graph,
    residual: list[list[int]],
    signatures: Sequence[TypedDegreeSignature],
    edge_types: Sequence[Any],
    *,
    config: TypedConstructorConfig,
    endpoint_compatible: EndpointCompatibility | None,
    initialization_score: InitializationScore | None,
    rng: np.random.Generator,
    budget: _SearchBudget,
) -> bool:
    budget.search_nodes += 1
    if not _completion_possible(
        graph,
        residual,
        signatures,
        edge_types,
        ensure_connected=config.ensure_connected,
        endpoint_compatible=endpoint_compatible,
    ):
        return False
    active = _select_active_demand(
        graph,
        residual,
        signatures,
        edge_types,
        endpoint_compatible,
        rng,
    )
    if active is None:
        return (
            not config.ensure_connected
            or graph.number_of_nodes() <= 1
            or nx.is_connected(graph)
        )
    node, edge_index, partners = active
    partners = _order_partners(
        partners,
        node=node,
        edge_index=edge_index,
        signatures=signatures,
        edge_types=edge_types,
        config=config,
        initialization_score=initialization_score,
        rng=rng,
    )
    edge_type = edge_types[edge_index]
    for other in partners:
        if budget.exhausted:
            return False
        budget.candidate_trials += 1
        graph.add_edge(node, other, **{graph.graph["edge_attribute"]: edge_type})
        residual[node][edge_index] -= 1
        residual[other][edge_index] -= 1
        if _search_realization(
            graph,
            residual,
            signatures,
            edge_types,
            config=config,
            endpoint_compatible=endpoint_compatible,
            initialization_score=initialization_score,
            rng=rng,
            budget=budget,
        ):
            return True
        residual[node][edge_index] += 1
        residual[other][edge_index] += 1
        graph.remove_edge(node, other)
        budget.record_backtrack()
    return False


def _assigned_invariant(
    invariant: TypedInvariant,
    assignment: Sequence[int],
) -> TypedInvariant:
    return TypedInvariant(
        signatures=tuple(invariant.signatures[int(index)] for index in assignment),
        edge_types=invariant.edge_types,
        node_attribute=invariant.node_attribute,
        edge_attribute=invariant.edge_attribute,
    )


def _new_search_graph(invariant: TypedInvariant) -> nx.Graph:
    graph = nx.Graph()
    for node, signature in enumerate(invariant.signatures):
        graph.add_node(node, **{invariant.node_attribute: signature.node_type})
    graph.graph.update(
        {
            "constructor": "typed_backtracking",
            "edge_attribute": invariant.edge_attribute,
            "typed_edge_types": list(invariant.edge_types),
        }
    )
    return graph


def _validate_result(
    graph: nx.Graph,
    original: TypedInvariant,
    assigned: TypedInvariant,
    *,
    ensure_connected: bool,
    endpoint_compatible: EndpointCompatibility | None,
) -> None:
    if graph.is_directed() or graph.is_multigraph() or nx.number_of_selfloops(graph):
        raise AssertionError("Typed construction returned a non-simple graph.")
    if ensure_connected and graph.number_of_nodes() > 1 and not nx.is_connected(graph):
        raise AssertionError("Typed construction returned a disconnected graph.")
    if not typed_invariant_matches_graph(graph, assigned):
        raise AssertionError(
            "Typed construction did not realize the indexed assignment."
        )
    observed = extract_typed_invariant(
        graph,
        edge_types=original.edge_types,
        node_attribute=original.node_attribute,
        edge_attribute=original.edge_attribute,
    )
    if Counter(observed.signatures) != Counter(original.signatures):
        raise AssertionError("Typed construction changed the signature multiset.")
    if endpoint_compatible is not None:
        for left, right, data in graph.edges(data=True):
            edge_type = data[original.edge_attribute]
            if not _is_compatible(
                assigned.signatures[int(left)],
                assigned.signatures[int(right)],
                edge_type,
                endpoint_compatible,
            ):
                raise AssertionError(
                    "Typed construction violated endpoint compatibility."
                )


def construct_typed_graph(
    invariant: TypedInvariant,
    config: TypedConstructorConfig | dict[str, Any] | None = None,
    rng: np.random.Generator | None = None,
    *,
    endpoint_compatible: EndpointCompatibility | None = None,
    initialization_score: InitializationScore | None = None,
) -> tuple[nx.Graph, dict[str, Any]]:
    """Construct an exact simple typed realization or raise with diagnostics.

    With ``randomize_assignment=False`` the input's indexed signatures are
    preserved exactly.  With it enabled, output node ``i`` receives input
    signature ``diagnostics["assignment"][i]``; the complete signature
    multiset remains exact.
    """

    cfg = (
        config
        if isinstance(config, TypedConstructorConfig)
        else TypedConstructorConfig.from_dict(config)
    )
    if cfg.candidate_ranking == "empirical" and initialization_score is None:
        raise ValueError("candidate_ranking='empirical' requires initialization_score.")
    generator = rng if rng is not None else np.random.default_rng(0)
    precheck_errors = typed_invariant_errors(
        invariant,
        require_connected=cfg.ensure_connected,
        max_ordinary_degree=cfg.max_ordinary_degree,
        max_weighted_valence=cfg.max_weighted_valence,
        endpoint_compatible=endpoint_compatible,
    )
    diagnostics: dict[str, Any] = {
        "success": False,
        "trials": 0,
        "restarts": 0,
        "backtracks": 0,
        "candidate_trials": 0,
        "search_nodes": 0,
        "failure_reason": None,
        "precheck_errors": list(precheck_errors),
        "candidate_ranking": cfg.candidate_ranking,
        "assignment": None,
    }
    if precheck_errors:
        diagnostics["failure_reason"] = "precheck_failed"
        raise TypedConstructionError(
            "Typed invariant failed preliminary feasibility checks: "
            + "; ".join(precheck_errors[:3]),
            diagnostics,
        )

    budget_was_exhausted = False
    last_assignment: list[int] | None = None
    for trial in range(cfg.max_restarts + 1):
        assignment = list(range(invariant.num_nodes))
        if cfg.randomize_assignment:
            assignment = [int(value) for value in generator.permutation(assignment)]
        last_assignment = assignment
        assigned = _assigned_invariant(invariant, assignment)
        graph = _new_search_graph(assigned)
        residual = [
            [int(value) for value in signature.edge_degrees]
            for signature in assigned.signatures
        ]
        budget = _SearchBudget(limit=cfg.max_backtracks)
        found = _search_realization(
            graph,
            residual,
            assigned.signatures,
            assigned.edge_types,
            config=cfg,
            endpoint_compatible=endpoint_compatible,
            initialization_score=initialization_score,
            rng=generator,
            budget=budget,
        )
        diagnostics["trials"] = trial + 1
        diagnostics["restarts"] = trial
        diagnostics["backtracks"] += budget.backtracks
        diagnostics["candidate_trials"] += budget.candidate_trials
        diagnostics["search_nodes"] += budget.search_nodes
        diagnostics["assignment"] = list(assignment)
        budget_was_exhausted = budget_was_exhausted or budget.exhausted
        if found:
            _validate_result(
                graph,
                invariant,
                assigned,
                ensure_connected=cfg.ensure_connected,
                endpoint_compatible=endpoint_compatible,
            )
            graph.graph["typed_assignment_permutation"] = list(assignment)
            diagnostics["success"] = True
            diagnostics["failure_reason"] = None
            return graph, diagnostics
        if not budget.exhausted:
            # An untruncated depth-first traversal proves infeasibility; a
            # different random ordering cannot alter that result.
            break

    diagnostics["assignment"] = last_assignment
    diagnostics["failure_reason"] = (
        "search_budget_exhausted" if budget_was_exhausted else "no_typed_realization"
    )
    raise TypedConstructionError(
        "Could not construct an exact typed realization "
        f"({diagnostics['failure_reason']}).",
        diagnostics,
    )


def construct_typed_molecular_graph(
    invariant: TypedInvariant,
    config: TypedConstructorConfig | dict[str, Any] | None = None,
    rng: np.random.Generator | None = None,
    *,
    endpoint_compatible: EndpointCompatibility | None = None,
    initialization_score: InitializationScore | None = None,
) -> tuple[nx.Graph, dict[str, Any]]:
    """Explicit molecular alias for :func:`construct_typed_graph`."""

    return construct_typed_graph(
        invariant,
        config,
        rng,
        endpoint_compatible=endpoint_compatible,
        initialization_score=initialization_score,
    )

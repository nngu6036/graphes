"""Research-study utilities for controlled Graph-ER experiments.

The functions in this module are deliberately pure: they consume graphs or
already collected run records and return JSON-compatible diagnostics.  They do
not load checkpoints, mutate graphs supplied by callers, or write artifacts.
This keeps the scientific comparisons reusable from scripts and tests.
"""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from itertools import combinations, pairwise
from math import comb, sqrt
from typing import Any

import networkx as nx
import numpy as np

from grapher.evaluation.metrics import mmd_rbf

DEFAULT_EVALUATION_SEEDS = (42, 43, 44)
EXACT_REACHABILITY_MAX_NODES = 8
DEFAULT_REACHABILITY_MAX_STATES = 100_000

GraphDescriptor = Callable[[nx.Graph], Sequence[float] | np.ndarray]
EdgeCompatibility = Callable[[Hashable, Hashable, Mapping[str, Any]], bool]
MoveFilter = Callable[
    [
        nx.Graph,
        tuple[tuple[Hashable, Hashable], tuple[Hashable, Hashable]],
        tuple[tuple[Hashable, Hashable], tuple[Hashable, Hashable]],
        nx.Graph,
    ],
    bool,
]


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        return float(bool(value))
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric, got {value!r}.") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    return result


def generation_error_decomposition(
    stage_metrics: Mapping[str, Mapping[str, float] | float],
    *,
    stage_order: Sequence[str] | None = None,
    metric_higher_is_better: Mapping[str, bool] | None = None,
) -> dict[str, Any]:
    """Decompose metric changes over named generation stages.

    ``stage_metrics`` should contain matched evaluations (the same sampled
    invariants and graph identities at every stage).  Values may either be one
    scalar error per stage or equally keyed metric mappings.  A transition's
    ``error_change`` is positive when the later stage is worse.  Metrics are
    treated as lower-is-better unless listed in ``metric_higher_is_better``.

    This is a telescoping attribution, not a causal claim: interaction effects
    require the corresponding oracle/learned factorial stages to be supplied.
    """

    if len(stage_metrics) < 2:
        raise ValueError("At least two named stages are required.")
    order = list(stage_metrics) if stage_order is None else list(stage_order)
    if len(order) != len(set(order)):
        raise ValueError("stage_order contains duplicate stage names.")
    if set(order) != set(stage_metrics):
        missing = sorted(set(stage_metrics) - set(order))
        unknown = sorted(set(order) - set(stage_metrics))
        raise ValueError(
            f"stage_order must name every stage exactly once; missing={missing}, "
            f"unknown={unknown}."
        )

    normalized: dict[str, dict[str, float]] = {}
    scalar_mode = all(
        not isinstance(value, Mapping) for value in stage_metrics.values()
    )
    if any(isinstance(value, Mapping) for value in stage_metrics.values()) and not all(
        isinstance(value, Mapping) for value in stage_metrics.values()
    ):
        raise TypeError("Do not mix scalar and metric-mapping stage values.")
    for stage in order:
        raw = stage_metrics[stage]
        values = {"error": raw} if scalar_mode else dict(raw)  # type: ignore[arg-type]
        if not values:
            raise ValueError(f"Stage {stage!r} has no metrics.")
        normalized[stage] = {
            str(metric): _finite_float(value, name=f"{stage}.{metric}")
            for metric, value in values.items()
        }

    metric_names = list(normalized[order[0]])
    expected = set(metric_names)
    for stage in order[1:]:
        if set(normalized[stage]) != expected:
            raise ValueError(
                "Every stage must report the same metrics; "
                f"{stage!r} reports {sorted(normalized[stage])}, expected "
                f"{sorted(expected)}."
            )
    directions = dict(metric_higher_is_better or {})

    transitions: list[dict[str, Any]] = []
    for before, after in pairwise(order):
        raw_delta = {
            metric: normalized[after][metric] - normalized[before][metric]
            for metric in metric_names
        }
        error_change = {
            metric: (-delta if directions.get(metric, False) else delta)
            for metric, delta in raw_delta.items()
        }
        transitions.append(
            {
                "from": before,
                "to": after,
                "delta": raw_delta,
                "error_change": error_change,
                "improvement": {
                    metric: -value for metric, value in error_change.items()
                },
            }
        )

    total_delta = {
        metric: normalized[order[-1]][metric] - normalized[order[0]][metric]
        for metric in metric_names
    }
    total_error_change = {
        metric: (-delta if directions.get(metric, False) else delta)
        for metric, delta in total_delta.items()
    }
    telescoping_error = {
        metric: abs(
            total_delta[metric] - sum(item["delta"][metric] for item in transitions)
        )
        for metric in metric_names
    }
    return {
        "stage_order": order,
        "metrics": metric_names,
        "stages": normalized,
        "transitions": transitions,
        "total_delta": total_delta,
        "total_error_change": total_error_change,
        "telescoping_error": telescoping_error,
    }


def _default_graph_descriptor(graph: nx.Graph) -> np.ndarray:
    node_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()
    degrees = np.asarray([degree for _, degree in graph.degree()], dtype=np.float64)
    return np.asarray(
        [
            node_count,
            edge_count,
            nx.density(graph) if node_count > 1 else 0.0,
            nx.average_clustering(graph) if node_count else 0.0,
            nx.transitivity(graph) if node_count > 2 else 0.0,
            float(sum(nx.triangles(graph).values()) / 3.0) if node_count else 0.0,
            float(np.var(degrees)) if degrees.size else 0.0,
            nx.number_connected_components(graph) if node_count else 0.0,
        ],
        dtype=np.float64,
    )


def _descriptor_rows(
    graphs: Sequence[nx.Graph], descriptor: GraphDescriptor
) -> np.ndarray:
    rows = [
        np.asarray(descriptor(graph), dtype=np.float64).reshape(-1) for graph in graphs
    ]
    if not rows:
        raise ValueError("Each constructor must supply at least one graph.")
    width = rows[0].size
    if width == 0 or any(row.size != width for row in rows):
        raise ValueError("The graph descriptor must return one fixed non-empty width.")
    matrix = np.vstack(rows)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("The graph descriptor returned a non-finite value.")
    return matrix


def _aggregate_constructor_diagnostics(
    raw: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    if raw is None:
        return {
            "attempts": None,
            "accepted": None,
            "rejected": None,
            "restarts": None,
            "acceptance_rate": None,
            "restarts_per_accepted": None,
            "rejection_reasons": {},
        }
    records = [raw] if isinstance(raw, Mapping) else list(raw)
    totals: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    for record in records:
        for key in ("attempts", "accepted", "rejected", "restarts"):
            if key in record and record[key] is not None:
                totals[key] += _finite_float(record[key], name=key)
        for reason, count in dict(record.get("rejection_reasons", {})).items():
            reasons[str(reason)] += _finite_float(
                count, name=f"rejection_reasons.{reason}"
            )
    attempts = float(totals["attempts"])
    accepted = float(totals["accepted"])
    rejected = float(totals["rejected"])
    if attempts == 0.0 and accepted + rejected > 0.0:
        attempts = accepted + rejected
    return {
        "attempts": attempts,
        "accepted": accepted,
        "rejected": rejected,
        "restarts": float(totals["restarts"]),
        "acceptance_rate": accepted / attempts if attempts > 0.0 else None,
        "restarts_per_accepted": (
            float(totals["restarts"]) / accepted if accepted > 0.0 else None
        ),
        "rejection_reasons": dict(sorted(reasons.items())),
    }


def constructor_bias_comparison(
    samples_by_constructor: Mapping[str, Sequence[nx.Graph]],
    *,
    reference: Sequence[nx.Graph] | None = None,
    descriptor: GraphDescriptor | None = None,
    diagnostics: Mapping[str, Mapping[str, Any] | Sequence[Mapping[str, Any]]]
    | None = None,
) -> dict[str, Any]:
    """Compare structural and rejection/restart bias across constructors."""

    if len(samples_by_constructor) < 2:
        raise ValueError("At least two constructors are required for a bias study.")
    descriptor_fn = descriptor or _default_graph_descriptor
    names = list(samples_by_constructor)
    matrices = {
        name: _descriptor_rows(list(samples_by_constructor[name]), descriptor_fn)
        for name in names
    }
    reference_matrix = (
        _descriptor_rows(list(reference), descriptor_fn)
        if reference is not None
        else None
    )
    per_constructor: dict[str, Any] = {}
    for name in names:
        matrix = matrices[name]
        hashes = [
            nx.weisfeiler_lehman_graph_hash(graph)
            for graph in samples_by_constructor[name]
        ]
        per_constructor[name] = {
            "num_graphs": int(matrix.shape[0]),
            "descriptor_mean": matrix.mean(axis=0).tolist(),
            "descriptor_std": matrix.std(axis=0, ddof=0).tolist(),
            "wl_uniqueness_rate": float(len(set(hashes)) / len(hashes)),
            "reference_mmd": (
                mmd_rbf(reference_matrix, matrix)
                if reference_matrix is not None
                else None
            ),
            "construction_diagnostics": _aggregate_constructor_diagnostics(
                None if diagnostics is None else diagnostics.get(name)
            ),
        }

    pairwise = []
    for left, right in combinations(names, 2):
        left_hashes = {
            nx.weisfeiler_lehman_graph_hash(graph)
            for graph in samples_by_constructor[left]
        }
        right_hashes = {
            nx.weisfeiler_lehman_graph_hash(graph)
            for graph in samples_by_constructor[right]
        }
        union = left_hashes | right_hashes
        pairwise.append(
            {
                "left": left,
                "right": right,
                "descriptor_mmd": mmd_rbf(matrices[left], matrices[right]),
                "mean_absolute_descriptor_shift": float(
                    np.mean(np.abs(matrices[left].mean(0) - matrices[right].mean(0)))
                ),
                "wl_jaccard": (
                    float(len(left_hashes & right_hashes) / len(union))
                    if union
                    else 1.0
                ),
            }
        )
    return {
        "constructors": per_constructor,
        "pairwise": pairwise,
        "uses_reference": reference is not None,
    }


def _node_order(graph: nx.Graph) -> tuple[list[Hashable], dict[Hashable, int]]:
    nodes = sorted(graph.nodes(), key=lambda node: (type(node).__name__, repr(node)))
    return nodes, {node: index for index, node in enumerate(nodes)}


def _edge_state_key(
    graph: nx.Graph,
    node_index: Mapping[Hashable, int],
    *,
    edge_type_attr: str | None,
) -> tuple[tuple[Any, ...], ...]:
    state = []
    for left, right, data in graph.edges(data=True):
        u, v = sorted((node_index[left], node_index[right]))
        if edge_type_attr is None:
            state.append((u, v))
        else:
            state.append((u, v, repr(data.get(edge_type_attr))))
    return tuple(sorted(state))


def _has_edge_type(graph: nx.Graph, edge_type_attr: str) -> bool:
    return any(edge_type_attr in data for _, _, data in graph.edges(data=True))


def _swap_neighbours(
    graph: nx.Graph,
    *,
    preserve_connected: bool,
    same_edge_type: bool,
    edge_type_attr: str,
    locality_radius: int | None,
    edge_compatibility: EdgeCompatibility | None,
    graph_mask: Callable[[nx.Graph], bool] | None,
    move_filter: MoveFilter | None,
    counters: Counter[str],
) -> Iterable[nx.Graph]:
    edges = list(graph.edges())
    for first, second in combinations(edges, 2):
        a, b = first
        c, d = second
        if len({a, b, c, d}) != 4:
            continue
        first_data = dict(graph.get_edge_data(a, b) or {})
        second_data = dict(graph.get_edge_data(c, d) or {})
        if same_edge_type and first_data.get(edge_type_attr) != second_data.get(
            edge_type_attr
        ):
            counters["different_edge_type"] += 2
            continue
        for added in (((a, c), (b, d)), ((a, d), (b, c))):
            counters["proposals"] += 1
            if any(graph.has_edge(left, right) for left, right in added):
                counters["parallel_edge"] += 1
                continue
            if locality_radius is not None:
                local = True
                for left, right in added:
                    try:
                        distance = nx.shortest_path_length(graph, left, right)
                    except nx.NetworkXNoPath:
                        local = False
                        break
                    if distance > locality_radius:
                        local = False
                        break
                if not local:
                    counters["locality"] += 1
                    continue
            proposed = graph.copy()
            proposed.remove_edges_from((first, second))
            proposed.add_edge(*added[0], **first_data)
            proposed.add_edge(*added[1], **second_data)
            if edge_compatibility is not None and any(
                not edge_compatibility(left, right, proposed.edges[left, right])
                for left, right in added
            ):
                counters["compatibility"] += 1
                continue
            if (
                preserve_connected
                and proposed.number_of_nodes() > 0
                and not nx.is_connected(proposed)
            ):
                counters["disconnected"] += 1
                continue
            if graph_mask is not None and not graph_mask(proposed):
                counters["graph_mask"] += 1
                continue
            removed_pair = (first, second)
            added_pair = (added[0], added[1])
            if move_filter is not None and not move_filter(
                graph, removed_pair, added_pair, proposed
            ):
                counters["move_filter"] += 1
                continue
            counters["valid_proposals"] += 1
            yield proposed


def _enumerate_reachable(
    initial: nx.Graph,
    *,
    preserve_connected: bool,
    same_edge_type: bool,
    edge_type_attr: str,
    locality_radius: int | None,
    edge_compatibility: EdgeCompatibility | None,
    graph_mask: Callable[[nx.Graph], bool] | None,
    move_filter: MoveFilter | None,
    max_states: int,
) -> tuple[dict[tuple[tuple[Any, ...], ...], int], Counter[str]]:
    _, node_index = _node_order(initial)
    state_attr = edge_type_attr if _has_edge_type(initial, edge_type_attr) else None
    initial_key = _edge_state_key(initial, node_index, edge_type_attr=state_attr)
    distances = {initial_key: 0}
    queue: deque[nx.Graph] = deque([initial.copy()])
    counters: Counter[str] = Counter()
    while queue:
        current = queue.popleft()
        current_key = _edge_state_key(current, node_index, edge_type_attr=state_attr)
        for proposed in _swap_neighbours(
            current,
            preserve_connected=preserve_connected,
            same_edge_type=same_edge_type,
            edge_type_attr=edge_type_attr,
            locality_radius=locality_radius,
            edge_compatibility=edge_compatibility,
            graph_mask=graph_mask,
            move_filter=move_filter,
            counters=counters,
        ):
            key = _edge_state_key(proposed, node_index, edge_type_attr=state_attr)
            if key in distances:
                counters["duplicate_state"] += 1
                continue
            if len(distances) >= max_states:
                raise RuntimeError(
                    "Exact rewiring enumeration exceeded max_states="
                    f"{max_states}; increase the explicit budget or use an empirical study."
                )
            distances[key] = distances[current_key] + 1
            queue.append(proposed)
    return distances, counters


def constrained_rewiring_reachability(
    initial: nx.Graph,
    *,
    target: nx.Graph | None = None,
    preserve_connected: bool = True,
    same_edge_type: bool = False,
    edge_type_attr: str = "bond_type",
    locality_radius: int | None = None,
    edge_compatibility: EdgeCompatibility | None = None,
    graph_mask: Callable[[nx.Graph], bool] | None = None,
    move_filter: MoveFilter | None = None,
    exact_node_limit: int = EXACT_REACHABILITY_MAX_NODES,
    max_states: int = DEFAULT_REACHABILITY_MAX_STATES,
) -> dict[str, Any]:
    """Exactly enumerate small-graph constrained 2-switch reachability.

    Coverage is measured against the exact connected realization component
    reachable with the invariant-preserving rules (connectivity and optional
    same-edge-type swaps), but without locality, compatibility, or domain-mask
    restrictions.  This distinguishes a restrictive proposal/mask policy from
    the underlying invariant's realization space.
    """

    if (
        initial.is_directed()
        or initial.is_multigraph()
        or nx.number_of_selfloops(initial)
    ):
        raise ValueError("Reachability requires a simple undirected graph.")
    node_count = initial.number_of_nodes()
    if node_count > EXACT_REACHABILITY_MAX_NODES:
        raise NotImplementedError(
            "Exact constrained-rewiring enumeration is intentionally limited to "
            f"at most {EXACT_REACHABILITY_MAX_NODES} nodes (received {node_count}); "
            "use an empirical multi-start reachability estimate for larger graphs."
        )
    if node_count > exact_node_limit:
        raise NotImplementedError(
            "Exact constrained-rewiring enumeration is intentionally limited to "
            f"at most {exact_node_limit} nodes (received {node_count}); use an "
            "empirical multi-start reachability estimate for larger graphs."
        )
    if exact_node_limit > EXACT_REACHABILITY_MAX_NODES:
        raise ValueError(
            f"exact_node_limit cannot exceed the declared supported limit "
            f"{EXACT_REACHABILITY_MAX_NODES}."
        )
    if max_states < 1:
        raise ValueError("max_states must be positive.")
    if locality_radius is not None and locality_radius < 1:
        raise ValueError("locality_radius must be positive when supplied.")
    if preserve_connected and node_count > 0 and not nx.is_connected(initial):
        raise ValueError("The initial graph violates preserve_connected=True.")
    if same_edge_type and any(
        edge_type_attr not in data for _, _, data in initial.edges(data=True)
    ):
        raise ValueError(
            f"Every edge must define {edge_type_attr!r} for same-edge-type swaps."
        )
    if graph_mask is not None and not graph_mask(initial):
        raise ValueError("The initial graph does not pass graph_mask.")
    if target is not None:
        if set(target.nodes()) != set(initial.nodes()):
            raise ValueError(
                "Initial and target graphs must have identical node labels."
            )
        if dict(target.degree()) != dict(initial.degree()):
            raise ValueError("Initial and target graphs must share indexed degrees.")

    baseline_distances, baseline_counts = _enumerate_reachable(
        initial,
        preserve_connected=preserve_connected,
        same_edge_type=same_edge_type,
        edge_type_attr=edge_type_attr,
        locality_radius=None,
        edge_compatibility=None,
        graph_mask=None,
        move_filter=None,
        max_states=max_states,
    )
    constrained_distances, constrained_counts = _enumerate_reachable(
        initial,
        preserve_connected=preserve_connected,
        same_edge_type=same_edge_type,
        edge_type_attr=edge_type_attr,
        locality_radius=locality_radius,
        edge_compatibility=edge_compatibility,
        graph_mask=graph_mask,
        move_filter=move_filter,
        max_states=max_states,
    )
    _, node_index = _node_order(initial)
    state_attr = edge_type_attr if _has_edge_type(initial, edge_type_attr) else None
    target_key = (
        _edge_state_key(target, node_index, edge_type_attr=state_attr)
        if target is not None
        else None
    )
    constrained_count = len(constrained_distances)
    feasible_count = len(baseline_distances)
    return {
        "exact": True,
        "node_count": node_count,
        "reachable_count": constrained_count,
        "feasible_component_count": feasible_count,
        "coverage": float(constrained_count / feasible_count),
        "target_reachable": (
            target_key in constrained_distances if target_key is not None else None
        ),
        "target_distance": (
            constrained_distances.get(target_key) if target_key is not None else None
        ),
        "constraints": {
            "preserve_connected": preserve_connected,
            "same_edge_type": same_edge_type,
            "locality_radius": locality_radius,
            "edge_compatibility": edge_compatibility is not None,
            "graph_mask": graph_mask is not None,
            "move_filter": move_filter is not None,
        },
        "search": {
            "constrained": dict(sorted(constrained_counts.items())),
            "baseline": dict(sorted(baseline_counts.items())),
        },
    }


exact_rewiring_reachability = constrained_rewiring_reachability


def prediction_consistency_residuals(
    pair_marginals: np.ndarray | Sequence[Any],
    fixed_invariant: np.ndarray | Sequence[Any],
    *,
    graphlet_histograms: Mapping[int | str, Sequence[float] | np.ndarray] | None = None,
    graphlet_edge_counts: Mapping[int | str, Sequence[float] | np.ndarray]
    | None = None,
) -> dict[str, Any]:
    """Measure pair/invariant and pair/graphlet consistency residuals.

    ``pair_marginals`` has shape ``[n, n]`` or ``[n, n, edge_types]`` and must
    exclude the no-edge category.  ``fixed_invariant`` is the corresponding
    indexed incident-degree target of shape ``[n]`` or ``[n, edge_types]``.
    For graphlet cross-checks, each histogram needs an aligned vector giving
    the edge count represented by each vocabulary entry; unknown overflow
    entries may use ``NaN`` and are omitted from that cross-check.
    """

    pair = np.asarray(pair_marginals, dtype=np.float64)
    invariant = np.asarray(fixed_invariant, dtype=np.float64)
    if pair.ndim == 2:
        pair = pair[..., None]
    if invariant.ndim == 1:
        invariant = invariant[..., None]
    if pair.ndim != 3 or pair.shape[0] != pair.shape[1]:
        raise ValueError("pair_marginals must have shape [n,n] or [n,n,c].")
    if invariant.shape != (pair.shape[0], pair.shape[2]):
        raise ValueError(
            "fixed_invariant must have shape [n,c] matching pair_marginals."
        )
    if not np.all(np.isfinite(pair)) or not np.all(np.isfinite(invariant)):
        raise ValueError("Predictions and fixed invariants must be finite.")

    induced = pair.sum(axis=1) - np.diagonal(pair, axis1=0, axis2=1).T
    residual = induced - invariant
    symmetry = pair - pair.transpose(1, 0, 2)
    diagonal = np.stack(
        [np.diag(pair[..., index]) for index in range(pair.shape[2])], axis=1
    )
    bound_violation = np.maximum(-pair, 0.0) + np.maximum(pair - 1.0, 0.0)
    type_incidence = invariant.sum(axis=0)
    handshake_residual = np.abs(type_incidence / 2.0 - np.round(type_incidence / 2.0))

    graphlet_results: dict[str, Any] = {}
    if graphlet_histograms is not None:
        if graphlet_edge_counts is None:
            raise ValueError(
                "graphlet_edge_counts is required with graphlet_histograms."
            )
        n = pair.shape[0]
        upper_pair_mass = float(np.triu(pair.sum(axis=-1), k=1).sum())
        pair_density = upper_pair_mass / comb(n, 2) if n >= 2 else 0.0
        for raw_k, raw_histogram in graphlet_histograms.items():
            key = str(raw_k)
            k = int(raw_k)
            histogram = np.asarray(raw_histogram, dtype=np.float64).reshape(-1)
            candidate_keys: list[int | str] = [raw_k, key]
            if key.lstrip("-").isdigit():
                candidate_keys.append(int(key))
            matching_key = next(
                (
                    candidate
                    for candidate in candidate_keys
                    if candidate in graphlet_edge_counts
                ),
                None,
            )
            if matching_key is None:
                raise ValueError(f"Missing graphlet edge counts for k={key}.")
            edge_counts_raw = graphlet_edge_counts[matching_key]
            edge_counts = np.asarray(edge_counts_raw, dtype=np.float64).reshape(-1)
            if histogram.shape != edge_counts.shape:
                raise ValueError(f"Histogram and edge-count widths differ for k={key}.")
            if not np.all(np.isfinite(histogram)):
                raise ValueError(
                    f"Graphlet histogram k={key} contains non-finite values."
                )
            known = np.isfinite(edge_counts)
            known_mass = float(histogram[known].sum())
            expected_edges = (
                float(np.dot(histogram[known] / known_mass, edge_counts[known]))
                if known_mass > 0.0
                else None
            )
            pair_expected_edges = pair_density * comb(k, 2)
            graphlet_results[key] = {
                "mass": float(histogram.sum()),
                "normalization_residual": abs(float(histogram.sum()) - 1.0),
                "negative_mass": float(np.maximum(-histogram, 0.0).sum()),
                "known_mass": known_mass,
                "expected_edges": expected_edges,
                "pair_expected_edges": pair_expected_edges,
                "pair_graphlet_edge_residual": (
                    abs(expected_edges - pair_expected_edges)
                    if expected_edges is not None
                    else None
                ),
            }

    return {
        "induced_invariant": induced.tolist(),
        "node_type_residual": residual.tolist(),
        "invariant_l1": float(np.mean(np.abs(residual))) if residual.size else 0.0,
        "invariant_l2": float(np.sqrt(np.mean(residual**2))) if residual.size else 0.0,
        "invariant_max": float(np.max(np.abs(residual))) if residual.size else 0.0,
        "symmetry_l1": float(np.mean(np.abs(symmetry))) if symmetry.size else 0.0,
        "diagonal_mass": float(np.abs(diagonal).sum()),
        "probability_bound_violation": float(bound_violation.sum()),
        "handshake_residual_by_type": handshake_residual.tolist(),
        "graphlets": graphlet_results,
    }


def project_predictions_to_feasible_target(*args: Any, **kwargs: Any) -> Any:
    """Project joint predictions onto a realizable target set.

    A projection objective and a tractable description of the joint feasible
    pair/graphlet polytope have not yet been specified by the method.
    """

    del args, kwargs
    raise NotImplementedError(
        "Feasible-target projection is not implemented: the optimization "
        "objective and joint realizability constraints must be specified first."
    )


def local_summary_collision_diagnostics(
    graphs: Sequence[nx.Graph],
    summaries: Sequence[Sequence[float] | np.ndarray],
    *,
    targets: Sequence[Sequence[float] | float] | np.ndarray | None = None,
    decimals: int = 8,
    node_match: Callable[[Mapping[str, Any], Mapping[str, Any]], bool] | None = None,
    edge_match: Callable[[Mapping[str, Any], Mapping[str, Any]], bool] | None = None,
) -> dict[str, Any]:
    """Find summary collisions and quantify irreducible target uncertainty."""

    if len(graphs) != len(summaries):
        raise ValueError("graphs and summaries must contain the same number of items.")
    if not graphs:
        raise ValueError("At least one graph is required.")
    rows = [np.asarray(summary, dtype=np.float64).reshape(-1) for summary in summaries]
    width = rows[0].size
    if width == 0 or any(row.size != width for row in rows):
        raise ValueError("Every summary must have the same non-empty width.")
    matrix = np.vstack(rows)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Summaries must be finite.")

    grouped: dict[tuple[float, ...], list[int]] = {}
    for index, row in enumerate(matrix):
        key = tuple(np.round(row, decimals=decimals).tolist())
        grouped.setdefault(key, []).append(index)
    colliding_pairs = 0
    nonisomorphic_pairs = 0
    isomorphic_pairs = 0
    collision_groups = []
    for key, indices in grouped.items():
        if len(indices) < 2:
            continue
        group_nonisomorphic = 0
        for left, right in combinations(indices, 2):
            colliding_pairs += 1
            if nx.is_isomorphic(
                graphs[left],
                graphs[right],
                node_match=node_match,
                edge_match=edge_match,
            ):
                isomorphic_pairs += 1
            else:
                nonisomorphic_pairs += 1
                group_nonisomorphic += 1
        collision_groups.append(
            {
                "summary": list(key),
                "indices": indices,
                "pair_count": len(indices) * (len(indices) - 1) // 2,
                "nonisomorphic_pair_count": group_nonisomorphic,
            }
        )
    total_pairs = len(graphs) * (len(graphs) - 1) // 2
    result: dict[str, Any] = {
        "num_graphs": len(graphs),
        "num_distinct_summaries": len(grouped),
        "collision_group_count": len(collision_groups),
        "colliding_pair_count": colliding_pairs,
        "nonisomorphic_collision_pair_count": nonisomorphic_pairs,
        "isomorphic_duplicate_pair_count": isomorphic_pairs,
        "collision_rate_all_pairs": (
            float(nonisomorphic_pairs / total_pairs) if total_pairs else 0.0
        ),
        "nonisomorphic_rate_within_collisions": (
            float(nonisomorphic_pairs / colliding_pairs) if colliding_pairs else 0.0
        ),
        "collision_groups": collision_groups,
    }
    if targets is not None:
        target_matrix = np.asarray(targets, dtype=np.float64)
        if target_matrix.ndim == 1:
            target_matrix = target_matrix[:, None]
        if target_matrix.shape[0] != len(graphs) or not np.all(
            np.isfinite(target_matrix)
        ):
            raise ValueError("targets must be finite with one row per graph.")
        center = target_matrix.mean(axis=0, keepdims=True)
        total_sse = float(np.sum((target_matrix - center) ** 2))
        within_sse = 0.0
        for indices in grouped.values():
            values = target_matrix[indices]
            within_sse += float(np.sum((values - values.mean(axis=0)) ** 2))
        result["target_sufficiency"] = {
            "total_variation_sse": total_sse,
            "unresolved_within_summary_sse": within_sse,
            "explained_variation_fraction": (
                1.0 - within_sse / total_sse
                if total_sse > 0.0
                else (1.0 if within_sse == 0.0 else 0.0)
            ),
        }
    else:
        result["target_sufficiency"] = None
    return result


local_summary_sufficiency = local_summary_collision_diagnostics


def hierarchical_graph_summary(*args: Any, **kwargs: Any) -> Any:
    """Return a coarsened/hierarchical graph summary once specified."""

    del args, kwargs
    raise NotImplementedError(
        "Hierarchical summaries are not implemented because the coarsening "
        "operator, hierarchy depth, and cross-level target statistics are not "
        "yet specified."
    )


def _bond_order(value: Any) -> float | None:
    if isinstance(value, (int, float, np.integer, np.floating)):
        result = float(value)
        return result if np.isfinite(result) and result >= 0.0 else None
    normalized = str(value).strip().upper()
    lookup = {
        "SINGLE": 1.0,
        "S": 1.0,
        "1": 1.0,
        "DOUBLE": 2.0,
        "D": 2.0,
        "2": 2.0,
        "TRIPLE": 3.0,
        "T": 3.0,
        "3": 3.0,
        "AROMATIC": 1.5,
        "A": 1.5,
        "1.5": 1.5,
    }
    return lookup.get(normalized)


def molecular_validity_limitation_audit(
    graphs: Sequence[nx.Graph],
    *,
    atom_type_attr: str = "atom_type",
    bond_type_attr: str = "bond_type",
    formal_charge_attr: str = "formal_charge",
    aromatic_attr: str = "aromatic",
    stereochemistry_attr: str = "stereo",
    allowed_weighted_valence: Mapping[Any, float] | None = None,
) -> dict[str, Any]:
    """Audit what graph-level molecular checks can and cannot establish.

    This routine performs no post-hoc repair and deliberately makes no claim
    about chemical stability or synthesizability.
    """

    if not graphs:
        raise ValueError("At least one molecular graph is required.")
    counts: Counter[str] = Counter()
    weighted_valence_valid: list[bool] = []
    for graph in graphs:
        simple = (
            not graph.is_directed()
            and not graph.is_multigraph()
            and nx.number_of_selfloops(graph) == 0
        )
        connected = graph.number_of_nodes() > 0 and nx.is_connected(graph)
        counts["simple"] += int(simple)
        counts["connected"] += int(connected)
        counts["atom_type_complete"] += int(
            all(atom_type_attr in data for _, data in graph.nodes(data=True))
        )
        counts["bond_type_complete"] += int(
            all(bond_type_attr in data for _, _, data in graph.edges(data=True))
        )
        counts["formal_charge_complete"] += int(
            all(formal_charge_attr in data for _, data in graph.nodes(data=True))
        )
        counts["atom_aromaticity_complete"] += int(
            all(aromatic_attr in data for _, data in graph.nodes(data=True))
        )
        counts["bond_aromaticity_complete"] += int(
            all(aromatic_attr in data for _, _, data in graph.edges(data=True))
        )
        counts["stereochemistry_complete"] += int(
            all(stereochemistry_attr in data for _, _, data in graph.edges(data=True))
        )

        bond_orders: dict[tuple[Hashable, Hashable], float] = {}
        orders_known = True
        for left, right, data in graph.edges(data=True):
            order = _bond_order(data.get(bond_type_attr))
            if order is None:
                orders_known = False
                break
            bond_orders[(left, right)] = order
        counts["bond_orders_interpretable"] += int(orders_known)
        if allowed_weighted_valence is not None and orders_known:
            graph_valid = True
            for node, data in graph.nodes(data=True):
                atom_type = data.get(atom_type_attr)
                if atom_type not in allowed_weighted_valence:
                    graph_valid = False
                    break
                valence = sum(
                    order
                    for (left, right), order in bond_orders.items()
                    if node == left or node == right
                )
                if valence > float(allowed_weighted_valence[atom_type]) + 1.0e-8:
                    graph_valid = False
                    break
            weighted_valence_valid.append(graph_valid)

    total = float(len(graphs))
    rates = {
        key + "_rate": float(value / total) for key, value in sorted(counts.items())
    }
    return {
        "num_molecules": len(graphs),
        **rates,
        "weighted_valence_check_available": allowed_weighted_valence is not None,
        "weighted_valence_valid_rate": (
            float(np.mean(weighted_valence_valid)) if weighted_valence_valid else None
        ),
        "posthoc_repair_applied": False,
        "typed_degree_guarantee_scope": [
            "bond-count incidence by represented bond type",
            "weighted valence when bond orders are interpretable",
        ],
        "not_guaranteed": [
            "aromaticity consistency",
            "formal-charge consistency",
            "stereochemistry",
            "chemical stability",
            "synthetic accessibility",
            "three-dimensional realizability",
        ],
    }


audit_molecular_validity_limits = molecular_validity_limitation_audit


def assess_stability_and_synthesizability(*args: Any, **kwargs: Any) -> Any:
    """Call a validated stability/synthesizability oracle once one is declared."""

    del args, kwargs
    raise NotImplementedError(
        "No validated chemical-stability, synthesizability, or 3D oracle is "
        "declared; graph and valence checks cannot substitute for one."
    )


stability_synthesizability_oracle = assess_stability_and_synthesizability


def _flatten_numeric_metrics(
    values: Mapping[str, Any], *, prefix: str = "", skip: frozenset[str] = frozenset()
) -> dict[str, float]:
    flattened: dict[str, float] = {}
    for raw_key, value in values.items():
        key = str(raw_key)
        if key in skip:
            continue
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, Mapping):
            flattened.update(_flatten_numeric_metrics(value, prefix=name, skip=skip))
        elif isinstance(value, (int, float, np.integer, np.floating, bool, np.bool_)):
            flattened[name] = _finite_float(value, name=name)
    return flattened


def aggregate_three_seed_results(
    results: Mapping[int, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
    *,
    seeds: Sequence[int] = DEFAULT_EVALUATION_SEEDS,
    seed_key: str = "seed",
    ddof: int = 1,
) -> dict[str, Any]:
    """Validate and aggregate the fixed 42/43/44 evaluation protocol."""

    expected = tuple(int(seed) for seed in seeds)
    if expected != DEFAULT_EVALUATION_SEEDS:
        raise ValueError(
            f"The fixed protocol requires seeds {DEFAULT_EVALUATION_SEEDS}, got {expected}."
        )
    if isinstance(results, Mapping):
        by_seed = {int(seed): dict(values) for seed, values in results.items()}
    else:
        by_seed: dict[int, dict[str, Any]] = {}
        for record in results:
            if seed_key not in record:
                raise ValueError(f"Every result record must define {seed_key!r}.")
            seed = int(record[seed_key])
            if seed in by_seed:
                raise ValueError(f"Duplicate result for seed {seed}.")
            by_seed[seed] = dict(record)
    if set(by_seed) != set(expected):
        missing = sorted(set(expected) - set(by_seed))
        extra = sorted(set(by_seed) - set(expected))
        raise ValueError(
            f"Expected exactly seeds {expected}; missing={missing}, extra={extra}."
        )
    flattened = {
        seed: _flatten_numeric_metrics(values, skip=frozenset({seed_key}))
        for seed, values in by_seed.items()
    }
    metric_names = set(flattened[expected[0]])
    for seed in expected[1:]:
        if set(flattened[seed]) != metric_names:
            raise ValueError(
                "All seeds must report the same numeric metrics; "
                f"seed {seed} differs from seed {expected[0]}."
            )
    if ddof < 0 or ddof >= len(expected):
        raise ValueError(
            "ddof must be between zero and two for three-seed aggregation."
        )
    aggregate: dict[str, Any] = {}
    for metric in sorted(metric_names):
        values = np.asarray(
            [flattened[seed][metric] for seed in expected], dtype=np.float64
        )
        aggregate[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=ddof)),
            "values": values.tolist(),
            "n": len(expected),
        }
    return {
        "seeds": list(expected),
        "per_seed": {str(seed): by_seed[seed] for seed in expected},
        "aggregate": aggregate,
        "ddof": ddof,
    }


_PIPELINE_FIELD_GROUPS: dict[str, tuple[str, ...]] = {
    "predictor_nll": ("predictor_nll", "nll"),
    "predictor_macro_f1": ("predictor_macro_f1", "macro_f1"),
    "graphlet_error": ("graphlet_error",),
    "consistency_residual": ("consistency_residual",),
    "invariant_feasible": ("invariant_feasible", "invariant_feasibility"),
    "constructor_success": ("constructor_success",),
    "accepted_swaps": ("accepted_swaps",),
    "runtime_seconds": ("runtime_seconds", "runtime"),
    "fallback_used": ("fallback_used", "silent_fallback"),
}


def _first_present(record: Mapping[str, Any], names: Sequence[str]) -> Any | None:
    for name in names:
        if name in record:
            return record[name]
    return None


def _mean_summary(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "min": float(array.min()),
        "max": float(array.max()),
        "n": int(array.size),
    }


def aggregate_pipeline_diagnostics(
    records: Sequence[Mapping[str, Any]],
    *,
    require_complete: bool = True,
    allow_fallback: bool = False,
) -> dict[str, Any]:
    """Aggregate all declared pipeline diagnostics without hiding fallback."""

    if not records:
        raise ValueError("At least one pipeline diagnostic record is required.")
    missing: set[str] = set()
    collected: dict[str, list[float]] = {key: [] for key in _PIPELINE_FIELD_GROUPS}
    rejection_reasons: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    count_fields_seen: Counter[str] = Counter()
    fallback_count = 0
    for index, record in enumerate(records):
        for canonical, aliases in _PIPELINE_FIELD_GROUPS.items():
            value = _first_present(record, aliases)
            if value is None:
                missing.add(canonical)
                continue
            numeric = _finite_float(value, name=f"record[{index}].{canonical}")
            collected[canonical].append(numeric)
            if canonical == "fallback_used" and bool(numeric):
                fallback_count += 1
        for key in (
            "candidate_proposals",
            "candidate_passes",
            "accepted_swaps",
            "stopped",
            "stop_opportunities",
            "generation_attempts",
            "generation_successes",
        ):
            if key in record:
                totals[key] += _finite_float(record[key], name=f"record[{index}].{key}")
                count_fields_seen[key] += 1
        for reason, count in dict(record.get("rejection_reasons", {})).items():
            rejection_reasons[str(reason)] += _finite_float(
                count, name=f"record[{index}].rejection_reasons.{reason}"
            )

    direct_rate_fields = {
        "candidate_pass_rate": ("candidate_pass_rate",),
        "proposals_per_accepted_swap": ("proposals_per_accepted_swap",),
        "stop_rate": ("stop_rate", "stop"),
        "end_to_end_yield": ("end_to_end_yield",),
    }
    direct_rates: dict[str, list[float]] = {key: [] for key in direct_rate_fields}
    for canonical, aliases in direct_rate_fields.items():
        for index, record in enumerate(records):
            value = _first_present(record, aliases)
            if value is not None:
                direct_rates[canonical].append(
                    _finite_float(value, name=f"record[{index}].{canonical}")
                )
        if not direct_rates[canonical]:
            derivable = {
                "candidate_pass_rate": count_fields_seen["candidate_proposals"]
                == len(records)
                and count_fields_seen["candidate_passes"] == len(records)
                and totals["candidate_proposals"] > 0,
                "proposals_per_accepted_swap": count_fields_seen["candidate_proposals"]
                == len(records)
                and count_fields_seen["accepted_swaps"] == len(records)
                and totals["accepted_swaps"] > 0,
                "stop_rate": count_fields_seen["stopped"] == len(records)
                and count_fields_seen["stop_opportunities"] == len(records)
                and totals["stop_opportunities"] > 0,
                "end_to_end_yield": count_fields_seen["generation_attempts"]
                == len(records)
                and count_fields_seen["generation_successes"] == len(records)
                and totals["generation_attempts"] > 0,
            }[canonical]
            if not derivable:
                missing.add(canonical)

    if not any("rejection_reasons" in record for record in records):
        missing.add("rejection_reasons")
    if require_complete and missing:
        raise ValueError(f"Incomplete pipeline diagnostics; missing {sorted(missing)}.")
    if fallback_count and not allow_fallback:
        raise ValueError(
            f"{fallback_count} record(s) used fallback; final metrics must exclude them."
        )

    metrics = {
        key: _mean_summary(values)
        for key, values in collected.items()
        if values and key != "fallback_used"
    }
    if (
        count_fields_seen["candidate_proposals"] == len(records)
        and count_fields_seen["candidate_passes"] == len(records)
        and totals["candidate_proposals"] > 0
    ):
        candidate_pass_rate = totals["candidate_passes"] / totals["candidate_proposals"]
    else:
        candidate_pass_rate = (
            float(np.mean(direct_rates["candidate_pass_rate"]))
            if direct_rates["candidate_pass_rate"]
            else None
        )
    if (
        count_fields_seen["candidate_proposals"] == len(records)
        and count_fields_seen["accepted_swaps"] == len(records)
        and totals["accepted_swaps"] > 0
    ):
        proposals_per_accepted = (
            totals["candidate_proposals"] / totals["accepted_swaps"]
        )
    else:
        proposals_per_accepted = (
            float(np.mean(direct_rates["proposals_per_accepted_swap"]))
            if direct_rates["proposals_per_accepted_swap"]
            else None
        )
    if (
        count_fields_seen["stopped"] == len(records)
        and count_fields_seen["stop_opportunities"] == len(records)
        and totals["stop_opportunities"] > 0
    ):
        stop_rate = totals["stopped"] / totals["stop_opportunities"]
    else:
        stop_rate = (
            float(np.mean(direct_rates["stop_rate"]))
            if direct_rates["stop_rate"]
            else None
        )
    if (
        count_fields_seen["generation_attempts"] == len(records)
        and count_fields_seen["generation_successes"] == len(records)
        and totals["generation_attempts"] > 0
    ):
        end_yield = totals["generation_successes"] / totals["generation_attempts"]
    else:
        end_yield = (
            float(np.mean(direct_rates["end_to_end_yield"]))
            if direct_rates["end_to_end_yield"]
            else None
        )
    metrics.update(
        {
            "candidate_pass_rate": candidate_pass_rate,
            "proposals_per_accepted_swap": proposals_per_accepted,
            "stop_rate": stop_rate,
            "end_to_end_yield": end_yield,
        }
    )
    return {
        "num_records": len(records),
        "metrics": metrics,
        "totals": {key: float(value) for key, value in sorted(totals.items())},
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "fallback_count": fallback_count,
        "fallback_records_included": bool(fallback_count),
        "missing_fields": sorted(missing),
    }


def paired_ablation_comparison(
    control: Sequence[Mapping[str, Any]],
    treatment: Sequence[Mapping[str, Any]],
    *,
    id_key: str = "sample_id",
    pairing_keys: Sequence[str] = ("seed", "invariant_id", "initial_graph_id"),
    metrics: Sequence[str] | None = None,
    metric_higher_is_better: Mapping[str, bool] | None = None,
    tie_tolerance: float = 1.0e-12,
) -> dict[str, Any]:
    """Validate controlled pairing and compare an ablation sample by sample."""

    if not control or not treatment:
        raise ValueError("Both ablation arms must contain records.")

    def index_arm(
        records: Sequence[Mapping[str, Any]], arm: str
    ) -> dict[Any, Mapping[str, Any]]:
        indexed: dict[Any, Mapping[str, Any]] = {}
        for record in records:
            if id_key not in record:
                raise ValueError(f"Every {arm} record must define {id_key!r}.")
            identity = record[id_key]
            if identity in indexed:
                raise ValueError(f"Duplicate {arm} {id_key}={identity!r}.")
            indexed[identity] = record
        return indexed

    left = index_arm(control, "control")
    right = index_arm(treatment, "treatment")
    if set(left) != set(right):
        raise ValueError("Control and treatment must contain identical sample IDs.")
    identities = sorted(left, key=lambda value: (type(value).__name__, repr(value)))
    validated_pairing_keys: set[str] = set()
    for identity in identities:
        for key in pairing_keys:
            present_left = key in left[identity]
            present_right = key in right[identity]
            if present_left != present_right:
                raise ValueError(
                    f"Pair {identity!r} defines {key!r} in only one ablation arm."
                )
            if present_left:
                validated_pairing_keys.add(key)
                if left[identity][key] != right[identity][key]:
                    raise ValueError(
                        f"Pair {identity!r} does not share {key!r}: "
                        f"{left[identity][key]!r} != {right[identity][key]!r}."
                    )
    if metrics is None:
        candidate_metrics = set(left[identities[0]]) & set(right[identities[0]])
        excluded = {id_key, *pairing_keys}
        metric_names = sorted(
            key
            for key in candidate_metrics - excluded
            if isinstance(left[identities[0]][key], (int, float, np.number))
            and isinstance(right[identities[0]][key], (int, float, np.number))
        )
    else:
        metric_names = list(metrics)
    if not metric_names:
        raise ValueError("No numeric ablation metrics were selected.")
    directions = dict(metric_higher_is_better or {})
    comparison: dict[str, Any] = {}
    for metric in metric_names:
        control_values = np.asarray(
            [
                _finite_float(left[identity].get(metric), name=f"control.{metric}")
                for identity in identities
            ],
            dtype=np.float64,
        )
        treatment_values = np.asarray(
            [
                _finite_float(right[identity].get(metric), name=f"treatment.{metric}")
                for identity in identities
            ],
            dtype=np.float64,
        )
        delta = treatment_values - control_values
        improvement = delta if directions.get(metric, False) else -delta
        standard_error = (
            float(delta.std(ddof=1) / sqrt(delta.size)) if delta.size > 1 else 0.0
        )
        comparison[metric] = {
            "control_mean": float(control_values.mean()),
            "treatment_mean": float(treatment_values.mean()),
            "mean_delta": float(delta.mean()),
            "mean_improvement": float(improvement.mean()),
            "delta_std": float(delta.std(ddof=1)) if delta.size > 1 else 0.0,
            "delta_ci95": [
                float(delta.mean() - 1.96 * standard_error),
                float(delta.mean() + 1.96 * standard_error),
            ],
            "wins": int(np.sum(improvement > tie_tolerance)),
            "ties": int(np.sum(np.abs(improvement) <= tie_tolerance)),
            "losses": int(np.sum(improvement < -tie_tolerance)),
            "paired_deltas": delta.tolist(),
        }
    return {
        "num_pairs": len(identities),
        "sample_ids": identities,
        "validated_pairing_keys": sorted(validated_pairing_keys),
        "metrics": comparison,
    }


ablation_pairing_comparison = paired_ablation_comparison


def quality_cost_pareto_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    quality_keys: str | Sequence[str],
    cost_keys: str | Sequence[str],
    higher_is_better: Mapping[str, bool] | None = None,
    id_key: str = "name",
) -> dict[str, Any]:
    """Return the exact nondominated frontier for quality/cost sweep records.

    Every objective is minimized by default (appropriate for MMD/error and
    runtime/proposal cost).  Put maximization objectives such as validity or
    yield in ``higher_is_better``.
    """

    if not records:
        raise ValueError("At least one quality-cost record is required.")
    qualities = [quality_keys] if isinstance(quality_keys, str) else list(quality_keys)
    costs = [cost_keys] if isinstance(cost_keys, str) else list(cost_keys)
    objectives = qualities + costs
    if not qualities or not costs or len(objectives) != len(set(objectives)):
        raise ValueError("Provide distinct, non-empty quality and cost objective keys.")
    directions = dict(higher_is_better or {})
    utilities = np.empty((len(records), len(objectives)), dtype=np.float64)
    for row, record in enumerate(records):
        for column, key in enumerate(objectives):
            if key not in record:
                raise ValueError(f"Record {row} is missing objective {key!r}.")
            value = _finite_float(record[key], name=f"record[{row}].{key}")
            utilities[row, column] = value if directions.get(key, False) else -value

    dominated_by: list[list[int]] = [[] for _ in records]
    dominates: list[list[int]] = [[] for _ in records]
    for left, right in combinations(range(len(records)), 2):
        left_ge = np.all(utilities[left] >= utilities[right])
        right_ge = np.all(utilities[right] >= utilities[left])
        left_gt = np.any(utilities[left] > utilities[right])
        right_gt = np.any(utilities[right] > utilities[left])
        if left_ge and left_gt:
            dominates[left].append(right)
            dominated_by[right].append(left)
        elif right_ge and right_gt:
            dominates[right].append(left)
            dominated_by[left].append(right)
    frontier_indices = [
        index for index, parents in enumerate(dominated_by) if not parents
    ]

    def frontier_sort_key(index: int) -> tuple[float, ...]:
        # Best cost first, then best quality, respecting each direction.
        keys = costs + qualities
        values = []
        for key in keys:
            value = _finite_float(records[index][key], name=key)
            values.append(-value if directions.get(key, False) else value)
        return tuple(values)

    frontier_indices.sort(key=frontier_sort_key)
    annotated = []
    for index, record in enumerate(records):
        annotated.append(
            {
                "index": index,
                "id": record.get(id_key, index),
                "is_pareto": not dominated_by[index],
                "dominated_by": dominated_by[index],
                "dominates": dominates[index],
                "objectives": {
                    key: _finite_float(record[key], name=f"record[{index}].{key}")
                    for key in objectives
                },
            }
        )
    return {
        "quality_keys": qualities,
        "cost_keys": costs,
        "frontier_indices": frontier_indices,
        "frontier_ids": [
            records[index].get(id_key, index) for index in frontier_indices
        ],
        "frontier": [dict(records[index]) for index in frontier_indices],
        "records": annotated,
    }


pareto_summary = quality_cost_pareto_summary

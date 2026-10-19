"""Stochastic degree-constrained realization of a generated Laplacian.

This module converts one continuous spectral endpoint into a simple graph in
three explicit stages:

1. infer a target edge count from the generated Laplacian trace (or a declared
   control source);
2. project the continuous Laplacian diagonal to an indexed graphical degree
   sequence with that edge count; and
3. solve a stochastic, score-weighted degree-constrained realization problem.

The stochastic stage uses perturb-and-MAP: independent Gumbel noise is added to
normalized spectral edge scores, and a binary linear program selects the
maximum-utility simple graph subject to exact node-degree constraints.  When
requested, connectivity is enforced by iterative cut generation.

The procedure is a generation-time decoder.  It does not retrain the spectral
model and it is not claimed to sample exactly from a Gibbs distribution over
all degree-sequence realizations.
"""
from __future__ import annotations

import copy
import math
import time
from collections.abc import Mapping
from typing import Any

import networkx as nx
import numpy as np

from grapher.rewiring_mlp.generic.basis import TopologyGraphletBasis
from grapher.rewiring_mlp.generic.graphlets import (
    extract_topology_graphlet_counts,
    topology_structural_discrepancy_from_counts,
)

try:  # Imported lazily by the generation path, but explicit here for tests.
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix, csr_matrix, vstack
except Exception as exc:  # pragma: no cover - exercised only in incomplete envs
    Bounds = LinearConstraint = milp = None  # type: ignore[assignment]
    coo_matrix = csr_matrix = vstack = None  # type: ignore[assignment]
    _SCIPY_IMPORT_ERROR: Exception | None = exc
else:
    _SCIPY_IMPORT_ERROR = None


FORMAT = "grapher_degree_constrained_spectral_realization_v1"


def _finite_float_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def default_config() -> dict[str, Any]:
    return {
        "enabled": False,
        "method": "gumbel_milp",
        "degree_source": "laplacian_diagonal",
        "edge_count_source": "spectral_trace",
        "edge_count_rounding": "nearest",
        "require_connected": True,
        "score_transform": "zscore",
        "score_clip": 8.0,
        "spectral_weight": 1.0,
        "gumbel_temperature": 0.15,
        "projection_degree_l1_weight": 1.0,
        "projection_spectral_tiebreak_weight": 1.0e-4,
        "projection_graphicality_check": True,
        "max_connectivity_cut_rounds": 16,
        "solver_time_limit_seconds": 5.0,
        "solver_mip_rel_gap": 0.0,
        "seed_offset": 810001,
        "failure_policy": "error",
        "save_threshold_control": True,
        "candidate_selection": {
            "enabled": False,
            "num_candidates": 8,
            "selection": "best",
            "selection_temperature": 0.10,
            "graphlet_weight": 1.0,
            "graphlet_mass_weight": 0.25,
            "clustering_weight": 0.0,
            "orbit_weight": 0.25,
        },
    }


def _deep_update(base: dict[str, Any], changes: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in changes.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            base[key] = _deep_update(dict(base[key]), value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def resolve_config(sample_options: Mapping[str, Any] | None) -> dict[str, Any]:
    sample_options = dict(sample_options or {})
    raw = sample_options.get("degree_constrained_realization", {}) or {}
    if not isinstance(raw, Mapping):
        raise TypeError("sample.degree_constrained_realization must be a mapping")
    cfg = _deep_update(default_config(), raw)
    validate_config(cfg)
    return cfg


def enabled(config: Mapping[str, Any] | None) -> bool:
    return bool((config or {}).get("enabled", False))


def validate_config(config: Mapping[str, Any]) -> None:
    if not bool(config.get("enabled", False)):
        return
    if _SCIPY_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Stochastic degree-constrained realization requires scipy.optimize.milp"
        ) from _SCIPY_IMPORT_ERROR
    if str(config.get("method", "gumbel_milp")).lower() != "gumbel_milp":
        raise ValueError("degree_constrained_realization.method must be gumbel_milp")
    if str(config.get("degree_source", "laplacian_diagonal")).lower() not in {
        "laplacian_diagonal",
        "threshold_graph",
    }:
        raise ValueError(
            "degree_constrained_realization.degree_source must be "
            "laplacian_diagonal or threshold_graph"
        )
    if str(config.get("edge_count_source", "spectral_trace")).lower() not in {
        "spectral_trace",
        "threshold_graph",
    }:
        raise ValueError(
            "degree_constrained_realization.edge_count_source must be "
            "spectral_trace or threshold_graph"
        )
    if str(config.get("edge_count_rounding", "nearest")).lower() not in {
        "nearest",
        "floor",
        "ceil",
    }:
        raise ValueError("edge_count_rounding must be nearest, floor, or ceil")
    if str(config.get("score_transform", "zscore")).lower() not in {
        "none",
        "zscore",
        "minmax",
        "rank",
    }:
        raise ValueError("score_transform must be none, zscore, minmax, or rank")
    if not math.isfinite(float(config.get("score_clip", 8.0))) or float(
        config.get("score_clip", 8.0)
    ) <= 0.0:
        raise ValueError("score_clip must be positive and finite")
    for name in (
        "spectral_weight",
        "gumbel_temperature",
        "projection_degree_l1_weight",
        "projection_spectral_tiebreak_weight",
        "solver_time_limit_seconds",
        "solver_mip_rel_gap",
    ):
        value = float(config.get(name, 0.0))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    if float(config.get("projection_degree_l1_weight", 1.0)) <= 0.0:
        raise ValueError("projection_degree_l1_weight must be positive")
    if int(config.get("max_connectivity_cut_rounds", 0)) < 0:
        raise ValueError("max_connectivity_cut_rounds must be non-negative")
    if str(config.get("failure_policy", "error")).lower() not in {
        "error",
        "threshold",
    }:
        raise ValueError("failure_policy must be error or threshold")

    selection = config.get("candidate_selection", {}) or {}
    if not isinstance(selection, Mapping):
        raise TypeError("degree_constrained_realization.candidate_selection must be a mapping")
    if bool(selection.get("enabled", False)):
        num_candidates = int(selection.get("num_candidates", 8))
        if num_candidates < 2:
            raise ValueError("candidate_selection.num_candidates must be >= 2 when enabled")
        mode = str(selection.get("selection", "best")).lower()
        if mode not in {"best", "softmax"}:
            raise ValueError("candidate_selection.selection must be best or softmax")
        selection_temperature = float(selection.get("selection_temperature", 0.10))
        if not math.isfinite(selection_temperature) or selection_temperature <= 0.0:
            raise ValueError("candidate_selection.selection_temperature must be positive and finite")
        weights = []
        for name, default in (
            ("graphlet_weight", 1.0),
            ("graphlet_mass_weight", 0.25),
            ("clustering_weight", 0.0),
            ("orbit_weight", 0.25),
        ):
            value = float(selection.get(name, default))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"candidate_selection.{name} must be finite and non-negative")
            weights.append(value)
        if not any(value > 0.0 for value in weights):
            raise ValueError("candidate_selection requires at least one positive structural weight")


def _edge_index(n: int) -> tuple[np.ndarray, np.ndarray]:
    if n < 0:
        raise ValueError("n must be non-negative")
    return np.triu_indices(n, 1)


def _degree_incidence(n: int, edge_u: np.ndarray, edge_v: np.ndarray) -> csr_matrix:
    edge_count = int(edge_u.size)
    columns = np.arange(edge_count, dtype=np.int64)
    rows = np.concatenate((edge_u, edge_v)).astype(np.int64, copy=False)
    cols = np.concatenate((columns, columns))
    data = np.ones(rows.size, dtype=np.float64)
    return coo_matrix((data, (rows, cols)), shape=(n, edge_count)).tocsr()


def _normalize_scores(values: np.ndarray, mode: str, clip: float) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("score vector must be one-dimensional")
    x = np.nan_to_num(x, nan=0.0, posinf=clip, neginf=-clip)
    mode = str(mode).lower()
    if x.size == 0:
        return x
    if mode == "none":
        out = x
    elif mode == "zscore":
        std = float(x.std())
        out = (x - float(x.mean())) / std if std > 1.0e-12 else np.zeros_like(x)
    elif mode == "minmax":
        lo, hi = float(x.min()), float(x.max())
        out = (x - lo) / (hi - lo) if hi - lo > 1.0e-12 else np.zeros_like(x)
    elif mode == "rank":
        order = np.argsort(x, kind="mergesort")
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(x.size, dtype=np.float64)
        out = ranks / float(max(x.size - 1, 1))
    else:  # validated earlier
        raise ValueError(f"Unknown score transform {mode!r}")
    return np.clip(out, -float(clip), float(clip))


def _round_edge_count(value: float, mode: str) -> int:
    mode = str(mode).lower()
    if mode == "floor":
        return int(math.floor(value))
    if mode == "ceil":
        return int(math.ceil(value))
    return int(math.floor(value + 0.5))


def _target_edge_count(
    laplacian: np.ndarray,
    threshold_graph: nx.Graph,
    *,
    config: Mapping[str, Any],
) -> tuple[int, dict[str, Any]]:
    n = int(laplacian.shape[0])
    max_edges = n * (n - 1) // 2
    require_connected = bool(config.get("require_connected", True)) and n > 1
    minimum = n - 1 if require_connected else 0
    source = str(config.get("edge_count_source", "spectral_trace")).lower()
    if source == "threshold_graph":
        raw = float(threshold_graph.number_of_edges())
    else:
        raw = 0.5 * float(np.trace(laplacian))
    rounded = _round_edge_count(raw, str(config.get("edge_count_rounding", "nearest")))
    clamped = int(min(max(rounded, minimum), max_edges))
    return clamped, {
        "source": source,
        "raw_edge_count": raw,
        "rounded_edge_count": int(rounded),
        "minimum_edge_count": int(minimum),
        "maximum_edge_count": int(max_edges),
        "clamped_edge_count": clamped,
        "clamped": bool(clamped != rounded),
    }


def _continuous_box_projection(
    raw: np.ndarray,
    *,
    total: int,
    lower: int,
    upper: int,
) -> np.ndarray:
    """Project to a bounded real vector with the requested sum."""
    raw = np.asarray(raw, dtype=np.float64)
    n = int(raw.size)
    if total < n * lower or total > n * upper:
        raise ValueError("requested degree sum is outside box bounds")
    lo = float(lower) - float(np.max(raw)) - 1.0
    hi = float(upper) - float(np.min(raw)) + 1.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        value = np.clip(raw + mid, lower, upper)
        if float(value.sum()) < float(total):
            lo = mid
        else:
            hi = mid
    return np.clip(raw + 0.5 * (lo + hi), lower, upper)


def _integer_box_projection(
    raw: np.ndarray,
    *,
    total: int,
    lower: int,
    upper: int,
) -> np.ndarray:
    projected = _continuous_box_projection(raw, total=total, lower=lower, upper=upper)
    integer = np.floor(projected + 1.0e-12).astype(np.int64)
    remainder = int(total - int(integer.sum()))
    if remainder > 0:
        fractional = projected - integer
        eligible = np.where(integer < upper)[0]
        order = eligible[np.argsort(-fractional[eligible], kind="mergesort")]
        for index in order[:remainder]:
            integer[int(index)] += 1
    elif remainder < 0:
        fractional = projected - integer
        eligible = np.where(integer > lower)[0]
        order = eligible[np.argsort(fractional[eligible], kind="mergesort")]
        for index in order[: -remainder]:
            integer[int(index)] -= 1
    if int(integer.sum()) != int(total):
        raise RuntimeError("integer degree projection failed to preserve requested sum")
    return integer


def _milp_options(config: Mapping[str, Any]) -> dict[str, Any]:
    options: dict[str, Any] = {"disp": False, "presolve": True}
    time_limit = float(config.get("solver_time_limit_seconds", 5.0))
    if time_limit > 0.0:
        options["time_limit"] = time_limit
    gap = float(config.get("solver_mip_rel_gap", 0.0))
    if gap >= 0.0:
        options["mip_rel_gap"] = gap
    return options


def _solve_milp(
    objective: np.ndarray,
    integrality: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    matrix: csr_matrix,
    constraint_lb: np.ndarray,
    constraint_ub: np.ndarray,
    *,
    config: Mapping[str, Any],
):
    if milp is None or Bounds is None or LinearConstraint is None:
        raise RuntimeError("scipy.optimize.milp is unavailable") from _SCIPY_IMPORT_ERROR
    return milp(
        c=np.asarray(objective, dtype=np.float64),
        integrality=np.asarray(integrality, dtype=np.int8),
        bounds=Bounds(
            np.asarray(lower_bounds, dtype=np.float64),
            np.asarray(upper_bounds, dtype=np.float64),
        ),
        constraints=LinearConstraint(
            matrix,
            np.asarray(constraint_lb, dtype=np.float64),
            np.asarray(constraint_ub, dtype=np.float64),
        ),
        options=_milp_options(config),
    )


def _solution_graph(
    n: int,
    edge_u: np.ndarray,
    edge_v: np.ndarray,
    solution: np.ndarray,
) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    chosen = np.where(np.asarray(solution[: edge_u.size]) > 0.5)[0]
    graph.add_edges_from((int(edge_u[k]), int(edge_v[k])) for k in chosen)
    return graph


def _connectivity_cut_rows(
    graph: nx.Graph,
    edge_u: np.ndarray,
    edge_v: np.ndarray,
    variable_count: int,
) -> list[csr_matrix]:
    rows: list[csr_matrix] = []
    for component in nx.connected_components(graph):
        component_set = set(component)
        if len(component_set) == graph.number_of_nodes():
            continue
        mask = np.fromiter(
            (
                (int(u) in component_set) ^ (int(v) in component_set)
                for u, v in zip(edge_u.tolist(), edge_v.tolist())
            ),
            dtype=bool,
            count=edge_u.size,
        )
        indices = np.where(mask)[0]
        data = np.ones(indices.size, dtype=np.float64)
        row = coo_matrix(
            (data, (np.zeros(indices.size, dtype=np.int64), indices)),
            shape=(1, variable_count),
        ).tocsr()
        rows.append(row)
    return rows


def _project_degree_sequence_with_milp(
    raw_degree: np.ndarray,
    target_edges: int,
    normalized_scores: np.ndarray,
    *,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    n = int(raw_degree.size)
    edge_u, edge_v = _edge_index(n)
    edge_count = int(edge_u.size)
    incidence = _degree_incidence(n, edge_u, edge_v)
    variable_count = edge_count + n

    objective = np.zeros(variable_count, dtype=np.float64)
    objective[:edge_count] = -float(
        config.get("projection_spectral_tiebreak_weight", 1.0e-4)
    ) * normalized_scores
    objective[edge_count:] = float(config.get("projection_degree_l1_weight", 1.0))
    integrality = np.zeros(variable_count, dtype=np.int8)
    integrality[:edge_count] = 1
    lower = np.zeros(variable_count, dtype=np.float64)
    upper = np.full(variable_count, np.inf, dtype=np.float64)
    upper[:edge_count] = 1.0

    edge_count_row = csr_matrix(
        (
            np.ones(edge_count, dtype=np.float64),
            (np.zeros(edge_count, dtype=np.int64), np.arange(edge_count)),
        ),
        shape=(1, variable_count),
    )
    incidence_padded = csr_matrix(
        (incidence.data, incidence.indices, incidence.indptr),
        shape=(n, edge_count),
    )
    incidence_full = csr_matrix(
        np.hstack((incidence_padded.toarray(), np.zeros((n, n), dtype=np.float64)))
    )
    slack_identity = csr_matrix(
        np.hstack((np.zeros((n, edge_count), dtype=np.float64), np.eye(n)))
    )
    rows = [edge_count_row]
    lb = [float(target_edges)]
    ub = [float(target_edges)]

    require_connected = bool(config.get("require_connected", True)) and n > 1
    if require_connected:
        rows.append(incidence_full)
        lb.extend([1.0] * n)
        ub.extend([np.inf] * n)

    # |degree_i - raw_i| <= slack_i.
    rows.append(incidence_full - slack_identity)
    lb.extend([-np.inf] * n)
    ub.extend(raw_degree.tolist())
    rows.append(-incidence_full - slack_identity)
    lb.extend([-np.inf] * n)
    ub.extend((-raw_degree).tolist())

    matrix = vstack(rows, format="csr")
    result = _solve_milp(
        objective,
        integrality,
        lower,
        upper,
        matrix,
        np.asarray(lb),
        np.asarray(ub),
        config=config,
    )
    if not bool(result.success) or result.x is None:
        raise RuntimeError(
            "degree-sequence projection MILP failed: "
            f"status={getattr(result, 'status', None)} message={getattr(result, 'message', '')}"
        )
    graph = _solution_graph(n, edge_u, edge_v, np.asarray(result.x))
    degree = np.asarray([graph.degree(i) for i in range(n)], dtype=np.int64)
    return degree, {
        "method": "milp_l1_projection",
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "objective": float(result.fun),
        "mip_gap": _finite_float_or_none(getattr(result, "mip_gap", None)),
        "mip_node_count": int(getattr(result, "mip_node_count", 0) or 0),
    }


def project_degree_sequence(
    raw_degree: np.ndarray,
    target_edges: int,
    normalized_scores: np.ndarray,
    *,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    raw_degree = np.asarray(raw_degree, dtype=np.float64)
    n = int(raw_degree.size)
    lower = 1 if bool(config.get("require_connected", True)) and n > 1 else 0
    upper = max(n - 1, 0)
    raw_degree = np.clip(np.nan_to_num(raw_degree, nan=0.0), lower, upper)
    candidate = _integer_box_projection(
        raw_degree,
        total=2 * int(target_edges),
        lower=lower,
        upper=upper,
    )
    graphical = bool(nx.is_graphical(candidate.tolist(), method="eg"))
    if graphical and bool(config.get("projection_graphicality_check", True)):
        return candidate, {
            "method": "bounded_rounding_graphical",
            "solver_status": None,
            "solver_message": None,
            "objective": None,
            "mip_gap": None,
            "mip_node_count": None,
        }
    if graphical and not bool(config.get("projection_graphicality_check", True)):
        return candidate, {
            "method": "bounded_rounding_unchecked",
            "solver_status": None,
            "solver_message": None,
            "objective": None,
            "mip_gap": None,
            "mip_node_count": None,
        }
    return _project_degree_sequence_with_milp(
        raw_degree,
        target_edges,
        normalized_scores,
        config=config,
    )


def _solve_exact_degree_realization(
    degree: np.ndarray,
    normalized_scores: np.ndarray,
    *,
    rng: np.random.Generator,
    config: Mapping[str, Any],
) -> tuple[nx.Graph, dict[str, Any]]:
    degree = np.asarray(degree, dtype=np.int64)
    n = int(degree.size)
    edge_u, edge_v = _edge_index(n)
    edge_count = int(edge_u.size)
    incidence = _degree_incidence(n, edge_u, edge_v)

    temperature = float(config.get("gumbel_temperature", 0.15))
    uniform = rng.uniform(
        low=np.finfo(np.float64).eps,
        high=1.0 - np.finfo(np.float64).eps,
        size=edge_count,
    )
    gumbel = -np.log(-np.log(uniform))
    utility = (
        float(config.get("spectral_weight", 1.0)) * normalized_scores
        + temperature * gumbel
    )
    objective = -utility
    integrality = np.ones(edge_count, dtype=np.int8)
    lower = np.zeros(edge_count, dtype=np.float64)
    upper = np.ones(edge_count, dtype=np.float64)
    base_matrix = incidence
    base_lb = degree.astype(np.float64)
    base_ub = degree.astype(np.float64)

    cut_rows: list[csr_matrix] = []
    cut_rounds = 0
    result = None
    graph = None
    require_connected = bool(config.get("require_connected", True)) and n > 1
    max_rounds = int(config.get("max_connectivity_cut_rounds", 16))
    while True:
        if cut_rows:
            matrix = vstack([base_matrix, *cut_rows], format="csr")
            lb = np.concatenate((base_lb, np.ones(len(cut_rows), dtype=np.float64)))
            ub = np.concatenate((base_ub, np.full(len(cut_rows), np.inf)))
        else:
            matrix = base_matrix
            lb, ub = base_lb, base_ub
        result = _solve_milp(
            objective,
            integrality,
            lower,
            upper,
            matrix,
            lb,
            ub,
            config=config,
        )
        if not bool(result.success) or result.x is None:
            raise RuntimeError(
                "degree-constrained realization MILP failed: "
                f"status={getattr(result, 'status', None)} message={getattr(result, 'message', '')}"
            )
        graph = _solution_graph(n, edge_u, edge_v, np.asarray(result.x))
        if not require_connected or nx.is_connected(graph):
            break
        if cut_rounds >= max_rounds:
            raise RuntimeError(
                "degree-constrained realization remained disconnected after "
                f"{max_rounds} connectivity-cut rounds"
            )
        new_rows = _connectivity_cut_rows(graph, edge_u, edge_v, edge_count)
        if not new_rows:
            raise RuntimeError("failed to construct connectivity cuts")
        cut_rows.extend(new_rows)
        cut_rounds += 1

    assert graph is not None and result is not None
    chosen = np.asarray(result.x[:edge_count]) > 0.5
    return graph, {
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "objective": float(result.fun),
        "mip_gap": _finite_float_or_none(getattr(result, "mip_gap", None)),
        "mip_node_count": int(getattr(result, "mip_node_count", 0) or 0),
        "connectivity_cut_rounds": int(cut_rounds),
        "connectivity_cuts": int(len(cut_rows)),
        "selected_spectral_score_mean": (
            float(normalized_scores[chosen].mean()) if bool(chosen.any()) else 0.0
        ),
        "selected_gumbel_mean": float(gumbel[chosen].mean()) if bool(chosen.any()) else 0.0,
        "temperature": temperature,
    }


def _candidate_selection_enabled(config: Mapping[str, Any]) -> bool:
    raw = config.get("candidate_selection", {}) or {}
    return isinstance(raw, Mapping) and bool(raw.get("enabled", False))


def _candidate_summary_score(
    graph: nx.Graph,
    *,
    summary_target: Mapping[str, Any],
    summary_meta: Mapping[str, Any],
    selection_config: Mapping[str, Any],
) -> dict[str, float]:
    if not bool(summary_meta.get("enabled", False)):
        raise RuntimeError(
            "summary candidate selection requires a checkpoint with structure_summary.enabled=true"
        )
    basis_payload = summary_meta.get("graphlet_basis")
    if not isinstance(basis_payload, Mapping):
        raise RuntimeError("structure-summary checkpoint is missing graphlet_basis metadata")
    basis = TopologyGraphletBasis.from_dict(dict(basis_payload))
    counts = extract_topology_graphlet_counts(graph, graphlet_basis=basis)

    graphlet_target = np.asarray(
        summary_target.get("graphlet_histogram", []), dtype=np.float64
    ).reshape(-1)
    mass_target = np.asarray(
        summary_target.get("graphlet_mass", []), dtype=np.float64
    ).reshape(-1)
    clustering_target = np.asarray(
        summary_target.get("clustering_histogram", []), dtype=np.float64
    ).reshape(-1)
    orbit_target = np.asarray(
        summary_target.get("orbit_mean_counts", []), dtype=np.float64
    ).reshape(-1)

    if graphlet_target.size != basis.width:
        raise ValueError(
            f"summary candidate selection expected graphlet width {basis.width}, "
            f"got {graphlet_target.size}"
        )
    if mass_target.size != len(basis.sizes):
        raise ValueError(
            f"summary candidate selection expected graphlet mass width {len(basis.sizes)}, "
            f"got {mass_target.size}"
        )

    clustering_weight = float(selection_config.get("clustering_weight", 0.0))
    orbit_weight = float(selection_config.get("orbit_weight", 0.25))
    if clustering_weight > 0.0 and clustering_target.size == 0:
        raise ValueError(
            "candidate_selection.clustering_weight is positive but the checkpoint has no "
            "clustering summary prediction"
        )
    if orbit_weight > 0.0 and orbit_target.size == 0:
        raise ValueError(
            "candidate_selection.orbit_weight is positive but the checkpoint has no orbit "
            "summary prediction"
        )

    return topology_structural_discrepancy_from_counts(
        graph,
        counts,
        graphlet_target=graphlet_target,
        graphlet_mass_target=mass_target,
        clustering_target=clustering_target,
        orbit_target=orbit_target,
        graphlet_basis=basis,
        graphlet_weight=float(selection_config.get("graphlet_weight", 1.0)),
        graphlet_mass_weight=float(selection_config.get("graphlet_mass_weight", 0.25)),
        clustering_weight=clustering_weight,
        orbit_weight=orbit_weight,
    )


def _candidate_edge_signature(graph: nx.Graph) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((min(int(u), int(v)), max(int(u), int(v))) for u, v in graph.edges()))


def _candidate_isomorphism_unique_count(graphs: list[nx.Graph]) -> int:
    representatives: list[nx.Graph] = []
    for graph in graphs:
        if not any(nx.is_isomorphic(graph, representative) for representative in representatives):
            representatives.append(graph)
    return len(representatives)


def _generate_and_select_candidates(
    degree: np.ndarray,
    normalized_scores: np.ndarray,
    *,
    rng: np.random.Generator,
    config: Mapping[str, Any],
    summary_target: Mapping[str, Any],
    summary_meta: Mapping[str, Any],
) -> tuple[nx.Graph, dict[str, Any], dict[str, Any]]:
    selection_config = dict(config.get("candidate_selection", {}) or {})
    num_candidates = int(selection_config.get("num_candidates", 8))
    candidates: list[nx.Graph] = []
    solver_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, float]] = []

    for candidate_index in range(num_candidates):
        graph, solver = _solve_exact_degree_realization(
            degree, normalized_scores, rng=rng, config=config
        )
        score = _candidate_summary_score(
            graph,
            summary_target=summary_target,
            summary_meta=summary_meta,
            selection_config=selection_config,
        )
        candidates.append(graph)
        solver_rows.append(solver)
        score_rows.append(score)

    energies = np.asarray([row["total"] for row in score_rows], dtype=np.float64)
    mode = str(selection_config.get("selection", "best")).lower()
    probabilities: np.ndarray | None = None
    if mode == "softmax":
        temperature = float(selection_config.get("selection_temperature", 0.10))
        logits = -(energies - float(energies.min())) / temperature
        logits -= float(logits.max())
        probabilities = np.exp(logits)
        probabilities /= float(probabilities.sum())
        selected = int(rng.choice(num_candidates, p=probabilities))
    else:
        selected = int(np.argmin(energies))

    selected_solver = dict(solver_rows[selected])
    selected_solver["candidate_selection_applied"] = True
    selected_solver["selected_candidate_index"] = selected

    selection_diagnostics: dict[str, Any] = {
        "enabled": True,
        "num_candidates": num_candidates,
        "selection": mode,
        "selection_temperature": float(selection_config.get("selection_temperature", 0.10)),
        "selected_candidate_index": selected,
        "selected_is_best": bool(selected == int(np.argmin(energies))),
        "selected_summary_energy": float(energies[selected]),
        "best_summary_energy": float(energies.min()),
        "mean_summary_energy": float(energies.mean()),
        "max_summary_energy": float(energies.max()),
        "candidate_edge_unique_count": len({_candidate_edge_signature(graph) for graph in candidates}),
        "candidate_isomorphism_unique_count": _candidate_isomorphism_unique_count(candidates),
        "selection_probabilities": probabilities.tolist() if probabilities is not None else None,
        "weights": {
            "graphlet": float(selection_config.get("graphlet_weight", 1.0)),
            "graphlet_mass": float(selection_config.get("graphlet_mass_weight", 0.25)),
            "clustering": float(selection_config.get("clustering_weight", 0.0)),
            "orbit": float(selection_config.get("orbit_weight", 0.25)),
        },
        "candidates": [
            {
                "candidate_index": index,
                "summary": {key: float(value) for key, value in score.items()},
                "solver_objective": _finite_float_or_none(solver.get("objective")),
                "selected_spectral_score_mean": _finite_float_or_none(
                    solver.get("selected_spectral_score_mean")
                ),
                "selected_gumbel_mean": _finite_float_or_none(
                    solver.get("selected_gumbel_mean")
                ),
            }
            for index, (score, solver) in enumerate(zip(score_rows, solver_rows))
        ],
    }
    return candidates[selected], selected_solver, selection_diagnostics


def realize_graph(
    laplacian: np.ndarray,
    adjacency_scores: np.ndarray,
    *,
    threshold: float,
    rng: np.random.Generator,
    config: Mapping[str, Any],
    summary_target: Mapping[str, Any] | None = None,
    summary_meta: Mapping[str, Any] | None = None,
) -> tuple[nx.Graph, nx.Graph, list[int], dict[str, Any]]:
    """Return final graph, threshold control, target degrees, and diagnostics."""
    validate_config(config)
    started = time.monotonic()
    lap = np.asarray(laplacian, dtype=np.float64)
    scores = np.asarray(adjacency_scores, dtype=np.float64)
    if lap.ndim != 2 or lap.shape[0] != lap.shape[1]:
        raise ValueError("laplacian must be square")
    if scores.shape != lap.shape:
        raise ValueError("adjacency_scores must match laplacian shape")
    n = int(lap.shape[0])
    lap = 0.5 * (lap + lap.T)
    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)

    threshold_adjacency = scores > float(threshold)
    np.fill_diagonal(threshold_adjacency, False)
    threshold_graph = nx.from_numpy_array(
        threshold_adjacency.astype(np.int8), create_using=nx.Graph
    )

    edge_u, edge_v = _edge_index(n)
    normalized_scores = _normalize_scores(
        scores[edge_u, edge_v],
        str(config.get("score_transform", "zscore")),
        float(config.get("score_clip", 8.0)),
    )
    target_edges, edge_count_diagnostics = _target_edge_count(
        lap,
        threshold_graph,
        config=config,
    )
    degree_source = str(config.get("degree_source", "laplacian_diagonal")).lower()
    if degree_source == "threshold_graph":
        raw_degree = np.asarray(
            [threshold_graph.degree(i) for i in range(n)], dtype=np.float64
        )
    else:
        raw_degree = np.diag(lap).astype(np.float64, copy=True)

    fallback = False
    failure_reason = None
    try:
        projected_degree, projection = project_degree_sequence(
            raw_degree,
            target_edges,
            normalized_scores,
            config=config,
        )
        candidate_selection = {"enabled": False}
        if _candidate_selection_enabled(config):
            if summary_target is None or summary_meta is None:
                raise RuntimeError(
                    "degree-constrained candidate selection requires summary_target and summary_meta"
                )
            graph, solver, candidate_selection = _generate_and_select_candidates(
                projected_degree,
                normalized_scores,
                rng=rng,
                config=config,
                summary_target=summary_target,
                summary_meta=summary_meta,
            )
        else:
            graph, solver = _solve_exact_degree_realization(
                projected_degree,
                normalized_scores,
                rng=rng,
                config=config,
            )
    except Exception as exc:
        if str(config.get("failure_policy", "error")).lower() != "threshold":
            raise
        fallback = True
        failure_reason = f"{type(exc).__name__}: {exc}"
        graph = threshold_graph.copy()
        projected_degree = np.asarray(
            [graph.degree(i) for i in range(n)], dtype=np.int64
        )
        projection = {
            "method": "threshold_fallback",
            "solver_status": None,
            "solver_message": failure_reason,
            "objective": None,
            "mip_gap": None,
            "mip_node_count": None,
        }
        candidate_selection = {
            "enabled": bool(_candidate_selection_enabled(config)),
            "failed": True,
            "failure_reason": failure_reason,
        }
        solver = {
            "solver_status": None,
            "solver_message": failure_reason,
            "objective": None,
            "mip_gap": None,
            "mip_node_count": None,
            "connectivity_cut_rounds": 0,
            "connectivity_cuts": 0,
            "selected_spectral_score_mean": None,
            "selected_gumbel_mean": None,
            "temperature": float(config.get("gumbel_temperature", 0.15)),
        }

    actual_degree = np.asarray([graph.degree(i) for i in range(n)], dtype=np.int64)
    threshold_edges = {tuple(sorted(edge)) for edge in threshold_graph.edges()}
    final_edges = {tuple(sorted(edge)) for edge in graph.edges()}
    degree_error = projected_degree.astype(np.float64) - raw_degree
    trace = float(np.trace(lap))
    diagnostics: dict[str, Any] = {
        "format": FORMAT,
        "num_nodes": n,
        "degree_source": degree_source,
        "edge_count": edge_count_diagnostics,
        "raw_degree": raw_degree.tolist(),
        "projected_degree": projected_degree.tolist(),
        "actual_degree": actual_degree.tolist(),
        "degree_exact": bool(np.array_equal(actual_degree, projected_degree)),
        "projected_degree_graphical": bool(
            nx.is_graphical(projected_degree.tolist(), method="eg")
        ),
        "projection_method": projection.get("method"),
        "projection": projection,
        "realization_solver": solver,
        "candidate_selection": candidate_selection,
        "connected": bool(n <= 1 or nx.is_connected(graph)),
        "require_connected": bool(config.get("require_connected", True)),
        "fallback_used": fallback,
        "failure_reason": failure_reason,
        "trace": trace,
        "realized_degree_sum": int(actual_degree.sum()),
        "trace_degree_sum_residual": float(actual_degree.sum() - trace),
        "raw_to_projected_degree_l1": float(np.abs(degree_error).sum()),
        "raw_to_projected_degree_mae": float(np.abs(degree_error).mean()) if n else 0.0,
        "raw_to_projected_degree_rmse": float(np.sqrt(np.mean(degree_error**2))) if n else 0.0,
        "threshold_num_edges": int(threshold_graph.number_of_edges()),
        "realized_num_edges": int(graph.number_of_edges()),
        "edge_symmetric_difference_from_threshold": int(
            len(threshold_edges.symmetric_difference(final_edges))
        ),
        "changed_from_threshold": bool(threshold_edges != final_edges),
        "isomorphic_to_threshold": bool(nx.is_isomorphic(graph, threshold_graph)),
        "runtime_seconds": float(time.monotonic() - started),
    }
    return graph, threshold_graph, projected_degree.tolist(), diagnostics


def aggregate_diagnostics(
    config: Mapping[str, Any], rows: list[Mapping[str, Any]]
) -> dict[str, Any]:
    if not enabled(config):
        return {
            "enabled": False,
            "method": str(config.get("method", "gumbel_milp")),
        }
    if not rows:
        return {
            "enabled": True,
            "method": str(config.get("method", "gumbel_milp")),
            "num_graphs": 0,
        }

    def mean_bool(key: str) -> float:
        return float(np.mean([bool(row.get(key, False)) for row in rows]))

    def mean_float(key: str) -> float:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        return float(np.mean(values)) if values else 0.0

    projection_counts: dict[str, int] = {}
    for row in rows:
        method = str(row.get("projection_method", "unknown"))
        projection_counts[method] = projection_counts.get(method, 0) + 1
    cut_rounds = [
        int((row.get("realization_solver", {}) or {}).get("connectivity_cut_rounds", 0))
        for row in rows
    ]
    selection_cfg = dict(config.get("candidate_selection", {}) or {})
    selection_rows = [
        dict(row.get("candidate_selection", {}) or {})
        for row in rows
        if bool((row.get("candidate_selection", {}) or {}).get("enabled", False))
    ]

    def selection_mean(key: str) -> float | None:
        values = [float(row[key]) for row in selection_rows if row.get(key) is not None]
        return float(np.mean(values)) if values else None

    return {
        "enabled": True,
        "format": FORMAT,
        "method": str(config.get("method", "gumbel_milp")),
        "degree_source": str(config.get("degree_source", "laplacian_diagonal")),
        "edge_count_source": str(config.get("edge_count_source", "spectral_trace")),
        "require_connected": bool(config.get("require_connected", True)),
        "score_transform": str(config.get("score_transform", "zscore")),
        "spectral_weight": float(config.get("spectral_weight", 1.0)),
        "gumbel_temperature": float(config.get("gumbel_temperature", 0.15)),
        "candidate_selection_enabled": bool(selection_cfg.get("enabled", False)),
        "candidate_selection_num_candidates": int(selection_cfg.get("num_candidates", 8)),
        "candidate_selection_mode": str(selection_cfg.get("selection", "best")),
        "candidate_selection_mean_selected_summary_energy": selection_mean("selected_summary_energy"),
        "candidate_selection_mean_best_summary_energy": selection_mean("best_summary_energy"),
        "candidate_selection_mean_candidate_summary_energy": selection_mean("mean_summary_energy"),
        "candidate_selection_mean_edge_unique_count": selection_mean("candidate_edge_unique_count"),
        "candidate_selection_mean_isomorphism_unique_count": selection_mean("candidate_isomorphism_unique_count"),
        "candidate_selection_selected_best_rate": (
            float(np.mean([bool(row.get("selected_is_best", False)) for row in selection_rows]))
            if selection_rows else None
        ),
        "num_graphs": len(rows),
        "success_rate": float(np.mean([not bool(row.get("fallback_used", False)) for row in rows])),
        "fallback_rate": mean_bool("fallback_used"),
        "degree_exact_rate": mean_bool("degree_exact"),
        "connectedness_rate": mean_bool("connected"),
        "changed_from_threshold_rate": mean_bool("changed_from_threshold"),
        "nonisomorphic_from_threshold_rate": float(
            np.mean([not bool(row.get("isomorphic_to_threshold", False)) for row in rows])
        ),
        "mean_edge_symmetric_difference_from_threshold": mean_float(
            "edge_symmetric_difference_from_threshold"
        ),
        "mean_raw_to_projected_degree_mae": mean_float(
            "raw_to_projected_degree_mae"
        ),
        "mean_raw_to_projected_degree_rmse": mean_float(
            "raw_to_projected_degree_rmse"
        ),
        "edge_count_clamped_rate": float(
            np.mean([bool((row.get("edge_count", {}) or {}).get("clamped", False)) for row in rows])
        ),
        "mean_raw_spectral_edge_count": float(
            np.mean([float((row.get("edge_count", {}) or {}).get("raw_edge_count", 0.0)) for row in rows])
        ),
        "mean_target_edge_count": float(
            np.mean([float((row.get("edge_count", {}) or {}).get("clamped_edge_count", 0.0)) for row in rows])
        ),
        "mean_threshold_edge_count": mean_float("threshold_num_edges"),
        "mean_realized_edge_count": mean_float("realized_num_edges"),
        "mean_abs_trace_degree_sum_residual": float(
            np.mean([abs(float(row.get("trace_degree_sum_residual", 0.0))) for row in rows])
        ),
        "mean_runtime_seconds": mean_float("runtime_seconds"),
        "mean_connectivity_cut_rounds": float(np.mean(cut_rounds)),
        "max_connectivity_cut_rounds_realized": int(max(cut_rounds)),
        "projection_method_counts": projection_counts,
        "config": copy.deepcopy(dict(config)),
    }

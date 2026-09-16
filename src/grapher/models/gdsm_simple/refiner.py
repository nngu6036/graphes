"""Degree-preserving spectral refinement for GSDM-Simple generation.

The refiner starts from the binary graph produced by the usual
``U diag(lambda) U^T`` threshold reconstruction and freezes its indexed degree
sequence.  Two generation-only modes are supported:

``lambda_only``
    Legacy S1 ablation.  Accept swaps only when the discrete graph's adjacency
    eigenvalues move closer to the spectrum predicted by the diffusion model.

``source_preserving``
    Conservative refinement for an already-strong GSDM sample.  The predicted
    eigenvalues are still required to improve, but candidate ranking also keeps
    the graph close to the *sampled empirical eigenbasis* used by GSDM and to
    the threshold source graph itself.  The eigenbasis is represented with a
    low-rank projector so eigenvector sign changes do not matter.

This module changes generation only; it does not alter GSDM-Simple training.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np

from grapher.rewiring_mlp.generic.rewiring import (
    propose_valid_topology_swaps,
    topology_state_key,
)


@dataclass(frozen=True)
class SpectralRewireConfig:
    # ``lambda_only`` is kept as the dataclass default for backwards-compatible
    # direct construction in tests/old callers.  The maintained Community-small
    # YAML explicitly selects ``source_preserving``.
    mode: str = "lambda_only"
    max_steps: int = 32
    proposal_budget: int = 256
    valid_candidate_budget: int = 128
    min_relative_improvement: float = 1.0e-6
    relative_improvement_epsilon: float = 1.0e-12
    preserve_connectivity_if_source_connected: bool = True
    reject_revisited_states: bool = True

    # Source-preserving objective.
    lambda_weight: float = 1.0
    projector_weight: float = 1.0
    source_weight: float = 0.1
    projector_rank: int = 4
    projector_normalization_floor: float = 0.05
    projector_relative_worsening_tolerance: float = 0.10
    require_lambda_improvement: bool = True

    # Batch-level gate.  The wrapper computes the threshold from the complete
    # generated batch before refinement so the same percentile semantics hold
    # independently of generation batch size.
    gate_enabled: bool = False
    gate_initial_lambda_quantile: float = 0.75

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> "SpectralRewireConfig":
        values = dict(raw or {})
        config = cls(
            mode=str(values.get("mode", cls.mode)).strip().lower(),
            max_steps=int(values.get("max_steps", cls.max_steps)),
            proposal_budget=int(values.get("proposal_budget", cls.proposal_budget)),
            valid_candidate_budget=int(
                values.get("valid_candidate_budget", cls.valid_candidate_budget)
            ),
            min_relative_improvement=float(
                values.get("min_relative_improvement", cls.min_relative_improvement)
            ),
            relative_improvement_epsilon=float(
                values.get(
                    "relative_improvement_epsilon",
                    cls.relative_improvement_epsilon,
                )
            ),
            preserve_connectivity_if_source_connected=bool(
                values.get(
                    "preserve_connectivity_if_source_connected",
                    cls.preserve_connectivity_if_source_connected,
                )
            ),
            reject_revisited_states=bool(
                values.get("reject_revisited_states", cls.reject_revisited_states)
            ),
            lambda_weight=float(values.get("lambda_weight", cls.lambda_weight)),
            projector_weight=float(
                values.get("projector_weight", cls.projector_weight)
            ),
            source_weight=float(values.get("source_weight", cls.source_weight)),
            projector_rank=int(values.get("projector_rank", cls.projector_rank)),
            projector_normalization_floor=float(
                values.get(
                    "projector_normalization_floor",
                    cls.projector_normalization_floor,
                )
            ),
            projector_relative_worsening_tolerance=float(
                values.get(
                    "projector_relative_worsening_tolerance",
                    cls.projector_relative_worsening_tolerance,
                )
            ),
            require_lambda_improvement=bool(
                values.get("require_lambda_improvement", cls.require_lambda_improvement)
            ),
            gate_enabled=bool(values.get("gate_enabled", cls.gate_enabled)),
            gate_initial_lambda_quantile=float(
                values.get(
                    "gate_initial_lambda_quantile",
                    cls.gate_initial_lambda_quantile,
                )
            ),
        )
        if config.mode not in {"lambda_only", "source_preserving"}:
            raise ValueError("rewiring.mode must be 'lambda_only' or 'source_preserving'")
        if config.max_steps < 0:
            raise ValueError("rewiring.max_steps must be non-negative")
        if config.proposal_budget == 0:
            raise ValueError("rewiring.proposal_budget must be positive or -1 for exhaustive search")
        if config.valid_candidate_budget == 0:
            raise ValueError(
                "rewiring.valid_candidate_budget must be positive or -1 for exhaustive search"
            )
        if config.min_relative_improvement < 0:
            raise ValueError("rewiring.min_relative_improvement must be non-negative")
        if config.relative_improvement_epsilon <= 0:
            raise ValueError("rewiring.relative_improvement_epsilon must be positive")
        if config.lambda_weight < 0 or config.projector_weight < 0 or config.source_weight < 0:
            raise ValueError("rewiring objective weights must be non-negative")
        if config.mode == "source_preserving" and (
            config.lambda_weight + config.projector_weight + config.source_weight <= 0
        ):
            raise ValueError("source-preserving rewiring requires at least one positive objective weight")
        if config.projector_rank < 1:
            raise ValueError("rewiring.projector_rank must be positive")
        if config.projector_normalization_floor <= 0:
            raise ValueError("rewiring.projector_normalization_floor must be positive")
        if config.projector_relative_worsening_tolerance < 0:
            raise ValueError("rewiring.projector_relative_worsening_tolerance must be non-negative")
        if not 0.0 <= config.gate_initial_lambda_quantile <= 1.0:
            raise ValueError("rewiring.gate_initial_lambda_quantile must be in [0, 1]")
        return config


def normalized_adjacency_eigenvalues(graph: nx.Graph) -> np.ndarray:
    """Sorted adjacency eigenvalues divided by sqrt(n), matching training."""

    n = int(graph.number_of_nodes())
    if n < 1:
        raise ValueError("Cannot compute a spectrum for an empty graph")
    nodes = sorted(graph.nodes())
    adjacency = nx.to_numpy_array(graph, nodelist=nodes, dtype=np.float64)
    values = np.linalg.eigvalsh(adjacency)
    return np.asarray(values, dtype=np.float64) / np.sqrt(float(n))


def adjacency_spectral_rmse(
    graph: nx.Graph,
    target_normalized_eigenvalues: np.ndarray,
) -> float:
    """Permutation-invariant RMSE between sorted normalized spectra."""

    target = np.sort(np.asarray(target_normalized_eigenvalues, dtype=np.float64).reshape(-1))
    current = normalized_adjacency_eigenvalues(graph)
    if target.shape != current.shape:
        raise ValueError(
            f"Spectrum length mismatch: target={target.shape}, graph={current.shape}"
        )
    return float(np.sqrt(np.mean(np.square(current - target))))


def _projector_mode_indices(
    target_normalized_eigenvalues: np.ndarray,
    rank: int,
) -> np.ndarray:
    """Select the dominant GSDM spectral modes by |lambda|.

    GSDM reconstructs ``U diag(lambda_hat) U^T`` with columns of ``U`` paired
    with the sorted generated eigenvalues.  We therefore select positions using
    ``lambda_hat`` and use the *same positions* for candidate graph eigenvectors.
    This preserves the intended mode correspondence while making the projector
    invariant to sign flips of individual eigenvectors.
    """

    values = np.asarray(target_normalized_eigenvalues, dtype=np.float64).reshape(-1)
    if values.size < 1:
        raise ValueError("Cannot select projector modes from an empty spectrum")
    k = min(int(rank), int(values.size))
    order = np.argsort(np.abs(values), kind="stable")[::-1]
    return np.sort(order[:k]).astype(np.int64)


def target_eigenspace_projector(
    sampled_eigenvectors: np.ndarray,
    target_normalized_eigenvalues: np.ndarray,
    rank: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the low-rank projector carried by GSDM's sampled empirical ``U``."""

    basis = np.asarray(sampled_eigenvectors, dtype=np.float64)
    target = np.asarray(target_normalized_eigenvalues, dtype=np.float64).reshape(-1)
    if basis.ndim != 2 or basis.shape[0] != basis.shape[1]:
        raise ValueError("sampled_eigenvectors must be a square eigenvector matrix")
    if basis.shape[0] != target.size:
        raise ValueError("sampled_eigenvectors and target spectrum must have the same size")
    indices = _projector_mode_indices(target, rank)
    modes = basis[:, indices]
    projector = modes @ modes.T
    projector = 0.5 * (projector + projector.T)
    return projector, indices


def graph_eigenspace_projector(
    graph: nx.Graph,
    mode_indices: np.ndarray,
) -> np.ndarray:
    """Projector for the corresponding sorted adjacency-eigenvector positions."""

    nodes = sorted(graph.nodes())
    adjacency = nx.to_numpy_array(graph, nodelist=nodes, dtype=np.float64)
    _values, vectors = np.linalg.eigh(adjacency)
    indices = np.asarray(mode_indices, dtype=np.int64).reshape(-1)
    if indices.size < 1 or indices.min() < 0 or indices.max() >= vectors.shape[1]:
        raise ValueError("Invalid adjacency eigenspace mode indices")
    modes = vectors[:, indices]
    projector = modes @ modes.T
    return 0.5 * (projector + projector.T)


def projector_chordal_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Normalized chordal distance between equal-rank orthogonal projectors."""

    p = np.asarray(left, dtype=np.float64)
    q = np.asarray(right, dtype=np.float64)
    if p.shape != q.shape or p.ndim != 2 or p.shape[0] != p.shape[1]:
        raise ValueError("Projector matrices must be square and have matching shapes")
    rank = max(float(np.trace(p)), 1.0)
    return float(np.linalg.norm(p - q, ord="fro") / np.sqrt(2.0 * rank))


def source_edge_distance(graph: nx.Graph, source: nx.Graph) -> float:
    """Fraction of the source edge set changed by the candidate.

    Degree-preserving swaps keep edge count fixed, so ``|E Δ E0| / (2|E0|)`` is
    in [0, 1] and equals zero exactly at the threshold source graph.
    """

    def canonical_edges(value: nx.Graph) -> set[tuple[int, int]]:
        return {
            (min(int(u), int(v)), max(int(u), int(v)))
            for u, v in value.edges()
        }

    source_edges = canonical_edges(source)
    graph_edges = canonical_edges(graph)
    if not source_edges:
        return 0.0 if not graph_edges else 1.0
    return float(len(source_edges.symmetric_difference(graph_edges)) / (2.0 * len(source_edges)))


def initial_lambda_gate(
    initial_errors: list[float] | np.ndarray,
    *,
    enabled: bool,
    quantile: float,
) -> tuple[np.ndarray, float | None]:
    """Return a batch-level high-residual refinement mask and its threshold."""

    values = np.asarray(initial_errors, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return np.zeros(0, dtype=np.bool_), None
    if not enabled:
        return np.ones(values.size, dtype=np.bool_), None
    threshold = float(np.quantile(values, float(quantile)))
    # Include ties at the threshold for deterministic behavior.
    return values >= threshold - 1.0e-15, threshold


def _source_preserving_metrics(
    graph: nx.Graph,
    source: nx.Graph,
    target_spectrum: np.ndarray,
    target_projector: np.ndarray,
    mode_indices: np.ndarray,
    *,
    initial_lambda_error: float,
    initial_projector_error: float,
    config: SpectralRewireConfig,
) -> dict[str, float]:
    lambda_error = adjacency_spectral_rmse(graph, target_spectrum)
    projector = graph_eigenspace_projector(graph, mode_indices)
    projector_error = projector_chordal_distance(projector, target_projector)
    source_distance = source_edge_distance(graph, source)
    lambda_scale = max(initial_lambda_error, config.relative_improvement_epsilon)
    projector_scale = max(
        initial_projector_error,
        config.projector_normalization_floor,
    )
    lambda_relative = lambda_error / lambda_scale
    projector_relative = projector_error / projector_scale
    energy = (
        config.lambda_weight * lambda_relative
        + config.projector_weight * projector_relative
        + config.source_weight * source_distance
    )
    return {
        "lambda_error": float(lambda_error),
        "projector_error": float(projector_error),
        "source_distance": float(source_distance),
        "lambda_relative": float(lambda_relative),
        "projector_relative": float(projector_relative),
        "energy": float(energy),
    }


def refine_graph_toward_adjacency_spectrum(
    graph: nx.Graph,
    target_normalized_eigenvalues: np.ndarray,
    *,
    rng: np.random.Generator,
    config: SpectralRewireConfig,
    target_eigenvectors: np.ndarray | None = None,
    gate_passed: bool = True,
    gate_threshold: float | None = None,
) -> tuple[nx.Graph, dict[str, Any]]:
    """Refine a GSDM threshold graph with degree-preserving swaps.

    ``lambda_only`` reproduces the previous S1 objective.

    ``source_preserving`` additionally uses the empirical eigenbasis that was
    sampled by GSDM to construct the threshold graph.  The low-rank projector
    provides a sign-invariant eigenspace target, while an edge symmetric-
    difference penalty discourages unnecessary movement away from the already
    realistic threshold graph.  Every accepted source-preserving move is also
    required to improve the generated eigenvalue target when
    ``require_lambda_improvement`` is enabled.
    """

    if graph.is_directed() or graph.is_multigraph():
        raise ValueError("GSDM-Simple spectral rewiring requires a simple undirected graph")
    if nx.number_of_selfloops(graph):
        raise ValueError("GSDM-Simple spectral rewiring does not support self-loops")

    current = nx.convert_node_labels_to_integers(graph, ordering="sorted")
    source = current.copy()
    source_degrees = np.asarray([current.degree(node) for node in sorted(current.nodes())])
    source_connected = bool(current.number_of_nodes() <= 1 or nx.is_connected(current))
    preserve_connectivity = bool(
        config.preserve_connectivity_if_source_connected and source_connected
    )

    target = np.sort(
        np.asarray(target_normalized_eigenvalues, dtype=np.float64).reshape(-1)
    )
    if target.size != current.number_of_nodes():
        raise ValueError(
            "Predicted spectrum length must equal the reconstructed graph's node count"
        )

    initial_error = adjacency_spectral_rmse(current, target)
    current_error = initial_error
    target_projector: np.ndarray | None = None
    mode_indices: np.ndarray | None = None
    initial_projector_error: float | None = None
    current_metrics: dict[str, float] | None = None

    if config.mode == "source_preserving":
        if target_eigenvectors is None:
            raise ValueError(
                "source_preserving rewiring requires the sampled GSDM eigenbasis"
            )
        target_projector, mode_indices = target_eigenspace_projector(
            target_eigenvectors,
            target,
            config.projector_rank,
        )
        source_projector = graph_eigenspace_projector(source, mode_indices)
        initial_projector_error = projector_chordal_distance(
            source_projector,
            target_projector,
        )
        current_metrics = _source_preserving_metrics(
            current,
            source,
            target,
            target_projector,
            mode_indices,
            initial_lambda_error=initial_error,
            initial_projector_error=initial_projector_error,
            config=config,
        )

    if not gate_passed:
        diagnostics = {
            "enabled": True,
            "mode": config.mode,
            "gate_passed": False,
            "gate_threshold": gate_threshold,
            "distance": "normalized_adjacency_eigenvalue_rmse",
            "normalization": "eigenvalues_div_sqrt_num_nodes",
            "initial_error": float(initial_error),
            "final_error": float(initial_error),
            "absolute_improvement": 0.0,
            "relative_improvement": 0.0,
            "accepted_steps": 0,
            "accepted_errors": [],
            "accepted_relative_improvements": [],
            "all_accepted_steps_improve": True,
            "degree_preserved": True,
            "source_connected": source_connected,
            "final_connected": source_connected,
            "connectivity_preserved_when_required": True,
            "preserve_connectivity_applied": preserve_connectivity,
            "total_proposals": 0,
            "total_valid_candidates": 0,
            "stop_reason": "below_gate",
            "config": _config_dict(config),
        }
        if current_metrics is not None:
            diagnostics.update(
                {
                    "projector_rank_effective": int(len(mode_indices)),
                    "projector_mode_indices": mode_indices.tolist(),
                    "initial_projector_error": float(initial_projector_error),
                    "final_projector_error": float(initial_projector_error),
                    "initial_source_distance": 0.0,
                    "final_source_distance": 0.0,
                    "initial_energy": float(current_metrics["energy"]),
                    "final_energy": float(current_metrics["energy"]),
                }
            )
        return current, diagnostics

    accepted_errors: list[float] = []
    accepted_relative_improvements: list[float] = []
    accepted_projector_errors: list[float] = []
    accepted_source_distances: list[float] = []
    accepted_energies: list[float] = []
    total_proposals = 0
    total_valid_candidates = 0
    projector_constraint_rejections = 0
    lambda_constraint_rejections = 0
    visited: set[bytes] = {topology_state_key(current)} if config.reject_revisited_states else set()
    stop_reason = "max_steps"

    for _step in range(config.max_steps):
        candidates, candidate_graphs, proposal_diag = propose_valid_topology_swaps(
            current,
            proposal_budget=config.proposal_budget,
            valid_candidate_budget=config.valid_candidate_budget,
            preserve_connectivity=preserve_connectivity,
            rng=rng,
            excluded_states=visited if config.reject_revisited_states else None,
        )
        total_proposals += int(proposal_diag.get("num_proposals", 0))
        total_valid_candidates += int(proposal_diag.get("num_valid_candidates", 0))
        if not candidates:
            stop_reason = "no_valid_candidates"
            break

        best_graph = None
        best_error = current_error
        best_metrics: dict[str, float] | None = None

        if config.mode == "lambda_only":
            for action in candidates:
                candidate = candidate_graphs[action]
                error = adjacency_spectral_rmse(candidate, target)
                if error < best_error:
                    best_error = error
                    best_graph = candidate
        else:
            assert current_metrics is not None
            assert target_projector is not None
            assert mode_indices is not None
            assert initial_projector_error is not None
            best_energy = float(current_metrics["energy"])
            projector_limit = initial_projector_error + (
                config.projector_relative_worsening_tolerance
                * max(initial_projector_error, config.projector_normalization_floor)
            )
            for action in candidates:
                candidate = candidate_graphs[action]
                metrics = _source_preserving_metrics(
                    candidate,
                    source,
                    target,
                    target_projector,
                    mode_indices,
                    initial_lambda_error=initial_error,
                    initial_projector_error=initial_projector_error,
                    config=config,
                )
                if config.require_lambda_improvement and not (
                    metrics["lambda_error"] < current_metrics["lambda_error"] - 1.0e-15
                ):
                    lambda_constraint_rejections += 1
                    continue
                if metrics["projector_error"] > projector_limit + 1.0e-15:
                    projector_constraint_rejections += 1
                    continue
                if metrics["energy"] < best_energy - 1.0e-15:
                    best_energy = float(metrics["energy"])
                    best_error = float(metrics["lambda_error"])
                    best_metrics = metrics
                    best_graph = candidate

        if best_graph is None:
            stop_reason = "no_improving_candidate"
            break

        if config.mode == "lambda_only":
            relative = (current_error - best_error) / max(
                abs(current_error), config.relative_improvement_epsilon
            )
        else:
            assert current_metrics is not None and best_metrics is not None
            relative = (current_metrics["energy"] - best_metrics["energy"]) / max(
                abs(current_metrics["energy"]), config.relative_improvement_epsilon
            )

        if relative + 1.0e-15 < config.min_relative_improvement:
            stop_reason = "below_relative_improvement_threshold"
            break

        current = best_graph
        current_error = best_error
        accepted_errors.append(float(current_error))
        accepted_relative_improvements.append(float(relative))
        if config.mode == "source_preserving":
            assert best_metrics is not None
            current_metrics = best_metrics
            accepted_projector_errors.append(float(best_metrics["projector_error"]))
            accepted_source_distances.append(float(best_metrics["source_distance"]))
            accepted_energies.append(float(best_metrics["energy"]))
        if config.reject_revisited_states:
            visited.add(topology_state_key(current))
    else:
        stop_reason = "max_steps"

    final_degrees = np.asarray([current.degree(node) for node in sorted(current.nodes())])
    degree_preserved = bool(np.array_equal(source_degrees, final_degrees))
    if not degree_preserved:
        raise RuntimeError("Degree-preserving rewiring changed the indexed degree sequence")
    final_connected = bool(current.number_of_nodes() <= 1 or nx.is_connected(current))

    diagnostics: dict[str, Any] = {
        "enabled": True,
        "mode": config.mode,
        "gate_passed": True,
        "gate_threshold": gate_threshold,
        "distance": "normalized_adjacency_eigenvalue_rmse",
        "normalization": "eigenvalues_div_sqrt_num_nodes",
        "initial_error": float(initial_error),
        "final_error": float(current_error),
        "absolute_improvement": float(initial_error - current_error),
        "relative_improvement": float(
            (initial_error - current_error)
            / max(abs(initial_error), config.relative_improvement_epsilon)
        ),
        "accepted_steps": len(accepted_errors),
        "accepted_errors": accepted_errors,
        "accepted_relative_improvements": accepted_relative_improvements,
        "all_accepted_steps_improve": bool(
            all(
                after < before
                for before, after in zip(
                    [initial_error] + accepted_errors[:-1], accepted_errors
                )
            )
        ),
        "degree_preserved": degree_preserved,
        "source_connected": source_connected,
        "final_connected": final_connected,
        "connectivity_preserved_when_required": bool(
            (not preserve_connectivity) or final_connected
        ),
        "preserve_connectivity_applied": preserve_connectivity,
        "total_proposals": total_proposals,
        "total_valid_candidates": total_valid_candidates,
        "stop_reason": stop_reason,
        "config": _config_dict(config),
    }
    if config.mode == "source_preserving":
        assert current_metrics is not None
        assert mode_indices is not None
        assert initial_projector_error is not None
        diagnostics.update(
            {
                "projector_distance": "normalized_chordal",
                "projector_rank_effective": int(len(mode_indices)),
                "projector_mode_indices": mode_indices.tolist(),
                "initial_projector_error": float(initial_projector_error),
                "final_projector_error": float(current_metrics["projector_error"]),
                "initial_source_distance": 0.0,
                "final_source_distance": float(current_metrics["source_distance"]),
                "initial_energy": float(
                    _source_preserving_metrics(
                        source,
                        source,
                        target,
                        target_projector,
                        mode_indices,
                        initial_lambda_error=initial_error,
                        initial_projector_error=initial_projector_error,
                        config=config,
                    )["energy"]
                ),
                "final_energy": float(current_metrics["energy"]),
                "accepted_projector_errors": accepted_projector_errors,
                "accepted_source_distances": accepted_source_distances,
                "accepted_energies": accepted_energies,
                "projector_constraint_rejections": projector_constraint_rejections,
                "lambda_constraint_rejections": lambda_constraint_rejections,
            }
        )
    return current, diagnostics


def _config_dict(config: SpectralRewireConfig) -> dict[str, Any]:
    return {
        "mode": config.mode,
        "max_steps": config.max_steps,
        "proposal_budget": config.proposal_budget,
        "valid_candidate_budget": config.valid_candidate_budget,
        "min_relative_improvement": config.min_relative_improvement,
        "relative_improvement_epsilon": config.relative_improvement_epsilon,
        "preserve_connectivity_if_source_connected": config.preserve_connectivity_if_source_connected,
        "reject_revisited_states": config.reject_revisited_states,
        "lambda_weight": config.lambda_weight,
        "projector_weight": config.projector_weight,
        "source_weight": config.source_weight,
        "projector_rank": config.projector_rank,
        "projector_normalization_floor": config.projector_normalization_floor,
        "projector_relative_worsening_tolerance": config.projector_relative_worsening_tolerance,
        "require_lambda_improvement": config.require_lambda_improvement,
        "gate_enabled": config.gate_enabled,
        "gate_initial_lambda_quantile": config.gate_initial_lambda_quantile,
    }


__all__ = [
    "SpectralRewireConfig",
    "adjacency_spectral_rmse",
    "graph_eigenspace_projector",
    "initial_lambda_gate",
    "normalized_adjacency_eigenvalues",
    "projector_chordal_distance",
    "refine_graph_toward_adjacency_spectrum",
    "source_edge_distance",
    "target_eigenspace_projector",
]

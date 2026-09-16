"""Degree-preserving spectral refinement for GSDM-Simple generation.

The refiner starts from the binary graph produced by the usual
``U diag(lambda) U^T`` threshold reconstruction.  It freezes that graph's
indexed degree sequence and searches its degree fibre with ordinary double-edge
swaps.  A swap is accepted only when it moves the *discrete graph's* adjacency
eigenvalues closer to the spectrum predicted by the diffusion model.

This is intentionally a generation-only GraphER extension: it does not alter
the GSDM-Simple denoiser or training objective.
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
    max_steps: int = 32
    proposal_budget: int = 256
    valid_candidate_budget: int = 128
    min_relative_improvement: float = 1.0e-6
    relative_improvement_epsilon: float = 1.0e-12
    preserve_connectivity_if_source_connected: bool = True
    reject_revisited_states: bool = True

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> "SpectralRewireConfig":
        values = dict(raw or {})
        config = cls(
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
        )
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


def refine_graph_toward_adjacency_spectrum(
    graph: nx.Graph,
    target_normalized_eigenvalues: np.ndarray,
    *,
    rng: np.random.Generator,
    config: SpectralRewireConfig,
) -> tuple[nx.Graph, dict[str, Any]]:
    """Move a binary graph toward a predicted spectrum with degree-preserving swaps.

    The indexed degree sequence of ``graph`` is frozen exactly.  If the source is
    connected and ``preserve_connectivity_if_source_connected`` is enabled, every
    accepted candidate must also remain connected.  A disconnected threshold
    reconstruction is still allowed to search its degree fibre; connectivity is
    not invented as a hard constraint in that case.
    """

    if graph.is_directed() or graph.is_multigraph():
        raise ValueError("GSDM-Simple spectral rewiring requires a simple undirected graph")
    if nx.number_of_selfloops(graph):
        raise ValueError("GSDM-Simple spectral rewiring does not support self-loops")

    current = nx.convert_node_labels_to_integers(graph, ordering="sorted")
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
    accepted_errors: list[float] = []
    accepted_relative_improvements: list[float] = []
    total_proposals = 0
    total_valid_candidates = 0
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
        for action in candidates:
            candidate = candidate_graphs[action]
            error = adjacency_spectral_rmse(candidate, target)
            if error < best_error:
                best_error = error
                best_graph = candidate

        if best_graph is None:
            stop_reason = "no_improving_candidate"
            break

        relative = (current_error - best_error) / max(
            abs(current_error), config.relative_improvement_epsilon
        )
        if relative + 1.0e-15 < config.min_relative_improvement:
            stop_reason = "below_relative_improvement_threshold"
            break

        current = best_graph
        current_error = best_error
        accepted_errors.append(float(current_error))
        accepted_relative_improvements.append(float(relative))
        if config.reject_revisited_states:
            visited.add(topology_state_key(current))
    else:
        stop_reason = "max_steps"

    final_degrees = np.asarray([current.degree(node) for node in sorted(current.nodes())])
    degree_preserved = bool(np.array_equal(source_degrees, final_degrees))
    if not degree_preserved:
        raise RuntimeError("Degree-preserving rewiring changed the indexed degree sequence")
    final_connected = bool(current.number_of_nodes() <= 1 or nx.is_connected(current))

    diagnostics = {
        "enabled": True,
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
            all(after < before for before, after in zip([initial_error] + accepted_errors[:-1], accepted_errors))
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
        "config": {
            "max_steps": config.max_steps,
            "proposal_budget": config.proposal_budget,
            "valid_candidate_budget": config.valid_candidate_budget,
            "min_relative_improvement": config.min_relative_improvement,
            "relative_improvement_epsilon": config.relative_improvement_epsilon,
            "preserve_connectivity_if_source_connected": config.preserve_connectivity_if_source_connected,
            "reject_revisited_states": config.reject_revisited_states,
        },
    }
    return current, diagnostics


__all__ = [
    "SpectralRewireConfig",
    "adjacency_spectral_rmse",
    "normalized_adjacency_eigenvalues",
    "refine_graph_toward_adjacency_spectrum",
]

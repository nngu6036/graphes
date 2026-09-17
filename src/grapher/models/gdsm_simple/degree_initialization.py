"""Degree-informed adjacency-spectrum anchors and degree-exact realization.

A sampled U and arbitrary graphical d need not share any exact spectrum. The
ridge fit below is only an anchor. Its residuals are reported, and the discrete
graph is separately realized with exactly the sampled indexed degrees.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any

import networkx as nx
import numpy as np

from grapher.models.dhvae_hh.havel_hakimi import construct_indexed_havel_hakimi
from grapher.rewiring_mlp.generic.rewiring import propose_valid_topology_swaps


def degree_basis_anchor(basis: np.ndarray, degrees: np.ndarray, *, ridge: float = 1e-3,
                        diagonal_weight: float = 1., seed_top_k: int | None = None) -> tuple[np.ndarray, dict[str, float]]:
    """Fit sorted adjacency eigenvalues with zero trace and a degree moment.

    A1 = U diag(U^T 1) lambda = d, diag(A) = (U*U)lambda = 0.
    Ordering and trace are enforced in a convex least-squares fit. Finally scale
    to sum(lambda^2)=sum(d), a necessary unweighted-graph moment. Scaling can
    increase the degree/diagonal residual; these are explicitly recorded.
    """
    from scipy.optimize import LinearConstraint, minimize

    u = np.asarray(basis, dtype=np.float64)
    d = np.asarray(degrees, dtype=np.float64)
    n = len(d)
    if n < 2 or u.shape != (n, n) or not np.isfinite(u).all() or not np.isfinite(d).all():
        raise ValueError("Expected finite square basis and degree vector, n>=2.")
    if np.any(d < 0) or np.any(d >= n) or np.any(d != np.rint(d)) or not nx.is_graphical(d.astype(int).tolist()):
        raise ValueError("Degree sequence must be integer, graphical, and in [0,n-1].")
    if not np.allclose(u.T @ u, np.eye(n), atol=2e-5):
        raise ValueError("Sampled eigenbasis is not orthonormal.")
    if seed_top_k is not None and int(seed_top_k) < 1:
        raise ValueError("seed_top_k must be null or positive.")
    if ridge <= 0 or diagonal_weight < 0:
        raise ValueError("ridge>0 and diagonal_weight>=0 required.")
    b = u * (u.T @ np.ones(n))[None, :]
    c = u * u
    h = b.T @ b + diagonal_weight * c.T @ c + ridge * np.eye(n)
    rhs = b.T @ d
    # KKT solution under trace zero supplies a stable starting point.
    kkt = np.block([[h, np.ones((n, 1))], [np.ones((1, n)), np.zeros((1, 1))]])
    guess = np.linalg.solve(kkt, np.r_[rhs, 0.])[:n]
    guess = np.sort(guess - guess.mean())
    differences = np.eye(n)[1:] - np.eye(n)[:-1]
    result = minimize(lambda x: .5 * x @ h @ x - rhs @ x, guess,
                      jac=lambda x: h @ x - rhs, method="SLSQP",
                      constraints=[LinearConstraint(np.ones((1, n)), 0., 0.), LinearConstraint(differences, 0., np.inf)],
                      options={"maxiter": 300, "ftol": 1e-10})
    values = np.asarray(result.x if result.success else guess, dtype=np.float64)
    values = np.sort(values - values.mean())
    desired = float(d.sum())
    if desired > 0 and np.linalg.norm(values) < 1e-10:
        values = np.linspace(-1., 1., n)
    if np.linalg.norm(values) > 0:
        values *= np.sqrt(desired) / np.linalg.norm(values)
    full = values.copy()
    if seed_top_k is not None and int(seed_top_k) < n:
        selected = np.argsort(np.abs(values), kind="stable")[-int(seed_top_k):]
        values = np.where(np.isin(np.arange(n), selected), values, 0.)
        # Keep spectral positions: negative selected modes, zeros, positive modes.
    stats = {
        "solver_success": bool(result.success),
        "row_sum_rmse": float(np.sqrt(np.mean((b @ full - d)**2))),
        "diagonal_rmse": float(np.sqrt(np.mean((c @ full)**2))),
        "trace_residual": float(abs(full.sum())),
        "second_moment_residual": float(abs(full @ full - desired)),
        "anchor_trace_residual": float(abs(values.sum())),
        "anchor_second_moment_residual": float(abs(values @ values - desired)),
        "num_seeded_modes": int(min(seed_top_k, n) if seed_top_k is not None else n),
    }
    return values / np.sqrt(n), stats


def conditioning_features(basis: np.ndarray, degrees: np.ndarray, anchor: np.ndarray, max_nodes: int) -> np.ndarray:
    """Size-padded d, anchor, IPR(U), and squared U^T1; invariant to column signs."""
    n = len(degrees)
    if n > max_nodes:
        raise ValueError("Graph size exceeds the trained maximum.")
    blocks = [np.sort(degrees)[::-1] / max(max_nodes - 1, 1), anchor,
              np.sum(np.asarray(basis)**4, axis=0), (np.asarray(basis).T @ np.ones(n))**2 / n]
    return np.concatenate([np.pad(np.asarray(v, dtype=np.float32), (0, max_nodes-n)) for v in blocks])


def align_basis_rows(basis: np.ndarray, bank_degrees: np.ndarray) -> np.ndarray:
    """Descending empirical degree rank; target d uses the same rank convention."""
    return np.asarray(basis, dtype=np.float64)[np.argsort(-np.asarray(bank_degrees), kind="stable")]


def validate_degree_bank(vectorizer: Any, training_sequences: list[list[int]]) -> None:
    """Reject stale degree checkpoints; check the full multiset, not membership."""
    actual = getattr(vectorizer, "empirical_degree_sequences", None)
    if not actual:
        raise ValueError("Degree checkpoint lacks empirical training sequences; cannot verify its training split.")
    canonical = lambda seqs: Counter(tuple(sorted(map(int, s), reverse=True)) for s in seqs)
    if canonical(actual) != canonical(training_sequences):
        raise ValueError("DH-VAE training degree multiset differs from this GSDM training split. Retrain the degree prior on the current split.")


def realize_degrees(degrees: np.ndarray, soft: np.ndarray, *, ensure_connected: bool,
                    config: Mapping[str, Any], rng: np.random.Generator) -> tuple[nx.Graph, dict[str, Any]]:
    """HH feasibility construction followed by bounded U/lambda edge-score fitting.

    This is not the diffusion prior. It realizes the first clean-state estimate
    as a simple graph, with d exact, before intermediate structural rewiring.
    """
    graph = construct_indexed_havel_hakimi(list(map(int, degrees)), ensure_connected=ensure_connected, rng=rng)
    soft = np.asarray(soft, dtype=np.float64)
    def cost(g):
        # Fixed edge count makes this equivalent to squared Frobenius fitting.
        return -float(sum(soft[u, v] + soft[v, u] for u, v in g.edges()))
    before, now, accepted = cost(graph), cost(graph), 0
    for _ in range(int(config.get("realization_fit_steps", 4))):
        actions, candidates, _ = propose_valid_topology_swaps(
            graph, proposal_budget=int(config.get("proposal_budget", 128)),
            valid_candidate_budget=int(config.get("valid_candidate_budget", 64)),
            preserve_connectivity=ensure_connected, rng=rng,
        )
        best, best_cost = None, now
        for action in actions:
            c = cost(candidates[action])
            if c < best_cost - 1e-9:
                best, best_cost = candidates[action], c
        if best is None:
            break
        graph, now, accepted = best, best_cost, accepted+1
    if list(dict(graph.degree()).values()) != list(map(int, degrees)):
        raise AssertionError("Degree realization changed indexed degrees.")
    return graph, {"constructor": "indexed_havel_hakimi_then_basis_score_swaps", "accepted_fit_steps": accepted,
                   "initial_score_cost": before, "final_score_cost": now, "degree_preserved": True}

"""Exact size-three summaries and generation-only constrained local search.

Orbit order is ORCA's first four columns: edge endpoint, induced-P3 endpoint,
induced-P3 centre, triangle vertex. No size-four/five orbit claim is made.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import networkx as nx
import numpy as np

from grapher.models.gdsm_simple.refiner import adjacency_spectral_rmse, source_edge_distance
from grapher.rewiring_mlp.generic.rewiring import propose_valid_topology_swaps, topology_state_key


def structure_enabled(extensions: Mapping[str, Any]) -> bool:
    value = extensions.get("structural_summary", "none")
    return isinstance(value, Mapping) and bool(value.get("enabled", True))


def validate_structure_options(options: Mapping[str, Any]) -> None:
    if not bool(options.get("sample", {}).get("sort_eigenvalues", True)):
        raise ValueError("Structure3 requires sample.sort_eigenvalues=true.")
    ext = options.get("extensions", {})
    summary = ext.get("structural_summary", {})
    if not structure_enabled(ext):
        raise ValueError("Structured checkpoints require structural_summary.enabled=true; disable rewiring for a no-guidance ablation.")
    if int(summary.get("graphlet_size", 3)) != 3:
        raise ValueError("This stage supports graphlet_size=3 only.")
    if summary.get("orbit_scope", "up_to_3") != "up_to_3":
        raise ValueError("orbit_scope must be 'up_to_3' (ORCA columns 0..3).")
    if int(summary.get("clustering_bins", 100)) < 2:
        raise ValueError("clustering_bins must be >= 2.")
    if not ext.get("degree_conditioning", False):
        raise ValueError("The structured denoiser requires degree_conditioning=true.")
    if ext.get("hh_initialization", False):
        raise ValueError("Use initialization.mode; the legacy hh_initialization flag is not used here.")
    init = ext.get("initialization", {})
    if init.get("mode", "degree_basis") not in {"degree_basis", "gaussian"}:
        raise ValueError("initialization.mode must be degree_basis or gaussian.")
    if float(init.get("ridge", 1e-3)) <= 0 or float(init.get("diagonal_weight", 1.0)) < 0:
        raise ValueError("Initialization requires ridge>0 and diagonal_weight>=0.")
    if init.get("seed_top_k") is not None and int(init["seed_top_k"]) < 1:
        raise ValueError("seed_top_k must be null or positive.")
    prior = init.get("degree_generator", {})
    if prior.get("type", "empirical") not in {"empirical", "dhvae"}:
        raise ValueError("degree_generator.type must be empirical or dhvae.")
    guidance = ext.get("structure_guidance", {})
    for name, default in (("every", 100), ("max_steps_per_event", 2), ("realization_fit_steps", 4)):
        value = int(guidance.get(name, default))
        if value < (1 if name == "every" else 0):
            raise ValueError(f"structure_guidance.{name} is invalid.")
    for name in ("proposal_budget", "valid_candidate_budget"):
        if int(guidance.get(name, 128)) == 0:
            raise ValueError(f"structure_guidance.{name} must be nonzero (negative means exhaustive).")
    for name, default in (("start_fraction", .35), ("spectrum_feedback", .05)):
        if not 0 <= float(guidance.get(name, default)) <= 1:
            raise ValueError(f"structure_guidance.{name} must be in [0,1].")
    for name, default in (("lambda_weight", .25), ("clustering_weight", 1.), ("orbit_weight", .25), ("graphlet_weight", .25), ("source_weight", .02), ("min_improvement", 1e-8)):
        if not np.isfinite(float(guidance.get(name, default))) or float(guidance.get(name, default)) < 0:
            raise ValueError(f"structure_guidance.{name} must be finite and nonnegative.")
    weights = summary.get("loss_weights", {})
    if any(not np.isfinite(float(weights.get(k, d))) or float(weights.get(k, d)) < 0
           for k, d in (("clustering", 1.), ("orbit", .5), ("graphlet", .5))):
        raise ValueError("Summary loss weights must be finite and nonnegative.")


def graph_summary3(graph: nx.Graph, clustering_bins: int = 100) -> dict[str, np.ndarray]:
    """Exact induced counts; O(n^3) dense triangle algebra, suitable for small graphs."""
    if graph.is_directed() or graph.is_multigraph() or nx.number_of_selfloops(graph):
        raise ValueError("Summaries require a loop-free simple undirected graph.")
    n = len(graph)
    if n == 0 or clustering_bins < 2:
        raise ValueError("Expected a nonempty graph and at least two clustering bins.")
    a = nx.to_numpy_array(graph, nodelist=sorted(graph.nodes()), dtype=np.float64)
    d = a.sum(1)
    triangles = np.rint(np.sum((a @ a) * a, axis=1) / 2.0)
    wedges = d * (d - 1) / 2
    centre = np.maximum(wedges - triangles, 0)
    endpoint = np.maximum(a @ (d - 1) - 2 * triangles, 0)
    orbit_nodes = np.stack((d, endpoint, centre, triangles), axis=1)
    orbits = orbit_nodes.mean(0)
    counts = np.array([centre.sum(), triangles.sum() / 3.0], dtype=np.float64)
    histogram = counts / max(float(counts.sum()), 1.0)
    clustering = np.divide(triangles, wedges, out=np.zeros_like(d), where=wedges > 0)
    chist = np.histogram(clustering, bins=clustering_bins, range=(0., 1.))[0].astype(np.float64) / n
    return {
        "clustering_histogram": chist,
        "orbit_mean": orbits,
        "orbit_log_mean": np.log1p(orbits),
        "orbit_node_counts": orbit_nodes,
        "graphlet_counts": counts,
        "graphlet_histogram": histogram,
    }


def reconcile_summary3(raw: Mapping[str, np.ndarray], degrees: np.ndarray) -> dict[str, np.ndarray]:
    """Project global count predictions onto necessary degree/count identities.

    This reconciles algebraic identities, not graph realizability. In particular,
    triangle counts can remain noninteger and clustering may be incompatible.
    """
    d = np.asarray(degrees, dtype=np.float64)
    n = len(d)
    w = float(np.sum(d * (d - 1) / 2))
    h = np.maximum(np.asarray(raw["graphlet_histogram"], dtype=np.float64), 0)
    h = h / max(float(h.sum()), 1e-12)
    orbit = np.expm1(np.clip(raw["orbit_log_mean"], 0, 30))
    # T/(P3+T)=q and P3+3T=W => T=qW/(1+2q).
    t_graphlet = float(h[1] * w / max(1 + 2 * h[1], 1e-12))
    t_orbit = float(orbit[3] * n / 3)
    t = float(np.clip((t_graphlet + t_orbit) / 2, 0, w / 3))
    p = max(w - 3 * t, 0.)
    counts = np.array([p, t])
    result = {
        "clustering_histogram": np.asarray(raw["clustering_histogram"], dtype=np.float64).copy(),
        "orbit_log_mean": np.log1p([float(d.mean()), 2*p/n, p/n, 3*t/n]),
        "graphlet_histogram": counts / max(float(counts.sum()), 1e-12),
    }
    # Nodes of degree 0/1 must have zero local clustering.
    c = np.maximum(result["clustering_histogram"], 0)
    c /= max(float(c.sum()), 1e-12)
    floor = float(np.mean(d < 2))
    if c[0] < floor:
        c[1:] *= (1 - floor) / max(1 - c[0], 1e-12)
        c[0] = floor
    result["clustering_histogram"] = c
    return result


def structure_distances(current: Mapping[str, np.ndarray], target: Mapping[str, np.ndarray]) -> dict[str, float]:
    return {
        "clustering": float(np.mean(np.abs(np.cumsum(current["clustering_histogram"] - target["clustering_histogram"])))),
        "orbit": float(np.sqrt(np.mean((current["orbit_log_mean"] - target["orbit_log_mean"])**2))),
        "graphlet": float(.5 * np.abs(current["graphlet_histogram"] - target["graphlet_histogram"]).sum()),
    }


def refine_structure3(
    graph: nx.Graph, target_spectrum: np.ndarray, target_summary: Mapping[str, np.ndarray],
    *, source: nx.Graph, config: Mapping[str, Any], rng: np.random.Generator,
    visited: set[bytes] | None = None,
) -> tuple[nx.Graph, dict[str, Any]]:
    """Greedy valid swaps against a *frozen* target during one DDIM event."""
    current = graph.copy()
    degrees = dict(current.degree())
    connected = nx.is_connected(current)
    bins = len(target_summary["clustering_histogram"])
    visited = visited if visited is not None else set()
    visited.add(topology_state_key(current))
    weights = {k: float(config.get(k + "_weight", default)) for k, default in (("clustering", 1.), ("orbit", .25), ("graphlet", .25))}
    lw, sw = float(config.get("lambda_weight", .25)), float(config.get("source_weight", .02))

    def score(g):
        metrics = structure_distances(graph_summary3(g, bins), target_summary)
        metrics["structure"] = sum(weights[k] * metrics[k] for k in weights)
        metrics["lambda"] = adjacency_spectral_rmse(g, target_spectrum) if lw else 0.
        metrics["source"] = source_edge_distance(g, source) if sw else 0.
        metrics["energy"] = metrics["structure"] + lw * metrics["lambda"] + sw * metrics["source"]
        return metrics

    before = score(current)
    now = before
    accepted, traces, proposed, valid = 0, [], 0, 0
    minimum = float(config.get("min_improvement", 1e-8))
    for _ in range(int(config.get("max_steps_per_event", 2))):
        actions, candidates, diag = propose_valid_topology_swaps(
            current, proposal_budget=int(config.get("proposal_budget", 128)),
            valid_candidate_budget=int(config.get("valid_candidate_budget", 64)),
            preserve_connectivity=connected, rng=rng, excluded_states=visited,
        )
        proposed += diag["num_proposals"]
        valid += diag["num_valid_candidates"]
        best, best_metrics = None, now
        for action in actions:
            candidate = candidates[action]
            metrics = score(candidate)
            if bool(config.get("require_structural_improvement", True)) and metrics["structure"] >= now["structure"] - minimum:
                continue
            if metrics["energy"] < best_metrics["energy"] - minimum:
                best, best_metrics = candidate, metrics
        if best is None:
            break
        traces.append({"before": now, "after": best_metrics})
        current, now = best, best_metrics
        accepted += 1
        visited.add(topology_state_key(current))
    if dict(current.degree()) != degrees or nx.number_of_selfloops(current) or (connected and not nx.is_connected(current)):
        raise AssertionError("A structural swap violated the frozen graph constraints.")
    return current, {
        "initial": before, "final": now, "accepted_steps": accepted,
        "num_proposals": proposed, "num_valid_candidates": valid, "accepted_trace": traces,
        "degree_preserved": True, "source_connected": connected, "final_connected": nx.is_connected(current),
        "all_accepted_steps_improve_energy": all(t["after"]["energy"] < t["before"]["energy"] for t in traces),
        "all_accepted_steps_improve_structure": all(t["after"]["structure"] < t["before"]["structure"] for t in traces),
    }

"""Generic GraphER generation with joint soft-edge and Laplacian-spectrum diffusion.

This is the generic analogue of the attributed joint typed edge sampler.  The
neural state contains two continuous bridges:

* symmetric binary edge-category logits over {no-edge, edge}; and
* combinatorial-Laplacian eigenvalues.

The hard graph never follows the soft bridge directly.  After sampling a clean
endpoint prediction, ordinary degree-preserving double-edge swaps realize that
endpoint while preserving simplicity and connectivity.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math

import networkx as nx
import numpy as np
import torch

from grapher.rewiring_mlp.generic.clustering import (
    clustering_histogram_wasserstein,
    extract_clustering_histogram,
)
from grapher.rewiring_mlp.generic.induced_graphlets import (
    InducedGraphletCounter,
    histogram_distance as induced_histogram_distance,
    validate_histogram as validate_induced_histogram,
)
from grapher.rewiring_mlp.generic.orbit import extract_orbit_summary, orbit_summary_distance
from grapher.rewiring_mlp.generic.rewiring import propose_valid_topology_swaps, topology_state_key
from grapher.rewiring_mlp.generic.soft_edge_bridge import (
    advance_edges,
    edge_probabilities,
    labels_to_logits,
)
from grapher.rewiring_mlp.generic.spectral import (
    laplacian_eigenvalues,
    spectral_distance,
    spectral_scale,
)
from grapher.rewiring_mlp.generic.spectral_data import (
    TopologySpectralExample,
    collate_spectral_examples,
)


def _adjacency_labels(graph: nx.Graph) -> torch.Tensor:
    nodes = list(range(graph.number_of_nodes()))
    a = nx.to_numpy_array(graph, nodelist=nodes, weight=None, dtype=np.int64)
    a = (a > 0).astype(np.int64)
    np.fill_diagonal(a, 0)
    return torch.from_numpy(a).unsqueeze(0)


def _source_edge_logits(graph: nx.Graph, smoothing: float, device: torch.device) -> torch.Tensor:
    labels = _adjacency_labels(graph).to(device)
    mask = torch.ones((1, graph.number_of_nodes()), dtype=torch.bool, device=device)
    return labels_to_logits(labels, mask, smoothing)


def _spectral_noise(values: np.ndarray, *, rng: np.random.Generator) -> np.ndarray:
    """Zero-first, zero-sum Gaussian direction for a fixed-trace Laplacian spectrum."""
    n = int(values.size)
    if n <= 1:
        return np.zeros(n, dtype=np.float64)
    noise = np.zeros(n, dtype=np.float64)
    noise[1:] = rng.normal(size=n - 1)
    noise[1:] -= float(noise[1:].mean())
    rms = float(np.sqrt(np.mean(noise[1:] ** 2)))
    if rms > 1e-12:
        noise[1:] /= rms
    return noise


def _advance_spectrum(
    current: np.ndarray,
    clean: np.ndarray,
    t: float,
    s: float,
    *,
    sigma: float,
    scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if not (0.0 <= t < s <= 1.0):
        raise ValueError("Spectral bridge transition requires 0 <= t < s <= 1.")
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("spectral_sigma must be finite and nonnegative.")
    if current.shape != clean.shape:
        raise ValueError("Current and predicted clean spectra must have the same shape.")
    alpha = (s - t) / max(1.0 - t, 1e-12)
    std = float(sigma) * float(scale) * math.sqrt(max((s - t) * (1.0 - s) / max(1.0 - t, 1e-12), 0.0))
    result = current + alpha * (clean - current)
    if std > 0:
        result = result + std * _spectral_noise(result, rng=rng)
    # The bridge is a continuous spectral state.  Preserve the two exact
    # invariants imposed during training; do not sort/project intermediate values.
    if result.size:
        result[0] = 0.0
        trace = float(current.sum())
        if result.size > 1:
            result[1:] += (trace - float(result.sum())) / (result.size - 1)
    return np.asarray(result, dtype=np.float64)


@dataclass(frozen=True)
class JointEdgeSpectralRefinerConfig:
    steps: int = 32
    proposal_budget: int = 1024
    valid_candidate_budget: int = 256
    preserve_connectivity: bool = True
    normalization: str = "mean_degree"
    component_normalization: str = "initial"
    distance: str = "rmse"
    epsilon: float = 1e-6
    min_improvement: float = 1e-8
    min_relative_improvement: float = 0.0
    edge_weight: float = 1.0
    spectral_weight: float = 0.1
    clustering_weight: float = 0.25
    orbit_weight: float = 1.0
    induced_graphlet_weight: float = 0.1
    orbit_distance: str = "log_rmse"
    clustering_statistic: str = "histogram"
    clustering_histogram_bins: int | None = None
    debug_enabled: bool = False
    debug_print_every: int = 1
    debug_top_candidates: int = 5
    debug_spectrum_values: int = 16

    # Compatibility metadata used by the existing generation report.
    guidance_mode: str = "edge_spectral_clustering_orbit_graphlet"
    bridge_schedule: str = "brownian_bridge"
    prediction_horizon_mode: str = "fixed"
    prediction_horizon_schedule: str = "constant"
    prediction_horizon_initial_k: int = 1
    prediction_horizon_final_k: int = 1
    refresh_on_prediction_plateau: bool = False
    induced_graphlet_distance: str = "total_variation"
    induced_graphlet_k: int | None = None
    induced_graphlet_scope: str | None = None
    cycle_weight: float = 0.0
    cycle_k: int | None = None
    cycle_distance: str | None = None
    compute_candidate_spectral_diagnostics: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None, *, model=None) -> "JointEdgeSpectralRefinerConfig":
        values = dict(data or {})
        mode = str(values.pop("mode", "joint_edge_spectral"))
        if mode not in {"joint_edge_spectral", "edge_spectral", "joint_diffusion"}:
            raise ValueError("Edge+spectral checkpoints require topology_refiner.mode=joint_edge_spectral.")
        weights = dict(values.pop("weights", {}) or {})
        allowed = {
            "steps", "proposal_budget", "valid_candidate_budget", "preserve_connectivity",
            "normalization", "component_normalization", "distance", "epsilon", "min_improvement",
            "min_relative_improvement", "orbit_distance", "debug_enabled",
            "debug_print_every", "debug_top_candidates", "debug_spectrum_values",
        }
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"Unsupported joint edge+spectral refiner settings: {sorted(unknown)}")
        unknown_weights = set(weights) - {"edge", "spectral", "clustering", "orbit", "graphlet"}
        if unknown_weights:
            raise ValueError(f"Unsupported joint refiner weights: {sorted(unknown_weights)}")
        kwargs = {key: values[key] for key in allowed if key in values}
        kwargs.update(
            edge_weight=float(weights.get("edge", 1.0)),
            spectral_weight=float(weights.get("spectral", 0.1)),
            clustering_weight=float(weights.get("clustering", 0.25)),
            orbit_weight=float(weights.get("orbit", 1.0)),
            induced_graphlet_weight=float(weights.get("graphlet", 0.1)),
        )
        if model is not None:
            kwargs["clustering_histogram_bins"] = (
                int(model.clustering_histogram_bins)
                if getattr(model, "predict_clustering_histogram", False) else None
            )
            spec = getattr(model, "induced_graphlet_spec", None)
            if spec is not None:
                kwargs["induced_graphlet_k"] = int(spec.max_k)
                kwargs["induced_graphlet_scope"] = str(spec.scope)
        cfg = cls(**kwargs)
        if cfg.steps < 0 or cfg.proposal_budget == 0 or cfg.valid_candidate_budget == 0:
            raise ValueError("steps must be nonnegative and candidate budgets must be nonzero.")
        if not cfg.preserve_connectivity:
            raise ValueError("Joint edge+spectral generation requires preserve_connectivity=true.")
        if cfg.normalization not in {"mean_degree", "trace", "none", "raw", "degree_sum"}:
            raise ValueError("Unsupported spectral normalization.")
        if cfg.component_normalization not in {"initial", "none"}:
            raise ValueError("component_normalization must be initial or none.")
        for name in ("edge_weight", "spectral_weight", "clustering_weight", "orbit_weight", "induced_graphlet_weight"):
            value = float(getattr(cfg, name))
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        if sum((cfg.edge_weight, cfg.spectral_weight, cfg.clustering_weight, cfg.orbit_weight, cfg.induced_graphlet_weight)) <= 0:
            raise ValueError("At least one joint refiner weight must be positive.")
        if cfg.clustering_weight > 0 and not getattr(model, "predict_clustering_histogram", False):
            raise ValueError("Clustering guidance requires a trained clustering histogram head.")
        if cfg.orbit_weight > 0 and not getattr(model, "predict_orbit_summary", False):
            raise ValueError("Orbit guidance requires a trained orbit-summary head.")
        if cfg.induced_graphlet_weight > 0 and not getattr(model, "predict_induced_graphlet_histogram", False):
            raise ValueError("Graphlet guidance requires a trained induced-graphlet head.")
        return cfg


@torch.no_grad()
def _predict(model, source: nx.Graph, edge_state: torch.Tensor, spectrum_state: np.ndarray, time: float):
    device = next(model.parameters()).device
    n = source.number_of_nodes()
    source_spectrum = laplacian_eigenvalues(source).astype(np.float32)
    smoothing = float(getattr(model, "edge_smoothing", 0.01))
    source_logits = _source_edge_logits(source, smoothing, device)[0].detach().cpu().numpy().astype(np.float32)
    current_logits = edge_state[0].detach().cpu().numpy().astype(np.float32)
    labels = _adjacency_labels(source)[0].numpy().astype(np.int64)
    example = TopologySpectralExample(
        current_graph=source,
        time=float(time),
        current_spectrum=np.asarray(spectrum_state, dtype=np.float32),
        source_spectrum=source_spectrum,
        clean_spectrum_target=np.zeros(n, dtype=np.float32),
        current_edge_logits=current_logits,
        source_edge_logits=source_logits,
        clean_edge_logits_target=source_logits,
        clean_edge_labels_target=labels,
    )
    batch = collate_spectral_examples([example]).to(device)
    return model(batch), batch


@torch.no_grad()
def sample_soft_endpoint(model, source: nx.Graph, config: dict[str, Any], *, seed: int):
    if not getattr(model, "predict_edge_state", False):
        raise ValueError("Joint soft-edge sampling requires a checkpoint trained with edge_diffusion.enabled=true.")
    model.eval()
    device = next(model.parameters()).device
    edge_cfg = dict(config.get("edge_diffusion", {}) or {})
    diff_cfg = dict(config.get("summary_diffusion", {}) or {})
    steps = int(edge_cfg.get("sampling_steps", 32))
    if steps < 2:
        raise ValueError("edge_diffusion.sampling_steps must be at least 2.")
    sigma_edge = float(edge_cfg.get("sigma", 1.0))
    sigma_spec = float(diff_cfg.get("spectral_sigma", 0.2))
    smoothing = float(edge_cfg.get("smoothing", getattr(model, "edge_smoothing", 0.01)))
    if not math.isclose(smoothing, float(getattr(model, "edge_smoothing", smoothing)), rel_tol=0, abs_tol=1e-12):
        raise ValueError("edge_diffusion.smoothing differs from the trained checkpoint.")

    mask = torch.ones((1, source.number_of_nodes()), dtype=torch.bool, device=device)
    labels = _adjacency_labels(source).to(device)
    edge_state = labels_to_logits(labels, mask, smoothing)
    spectrum_state = laplacian_eigenvalues(source).astype(np.float64)
    scale = spectral_scale(source, mode=str(config.get("spectral_prediction", {}).get("normalization", "mean_degree")))
    rng = np.random.default_rng(int(seed))
    generator = torch.Generator(device=device).manual_seed(int(seed))
    trajectory = []
    last_outputs = None
    for step in range(steps):
        t = step / steps
        s = (step + 1) / steps
        outputs, _batch = _predict(model, source, edge_state, spectrum_state, t)
        last_outputs = outputs
        clean_edge = outputs["clean_edge_logits"]
        clean_spectrum = outputs["clean_spectrum"][0, : source.number_of_nodes()].detach().cpu().numpy().astype(np.float64)
        edge_state = advance_edges(edge_state, clean_edge, t, s, mask, sigma_edge, generator=generator)
        spectrum_state = _advance_spectrum(
            spectrum_state, clean_spectrum, t, s,
            sigma=sigma_spec, scale=scale, rng=rng,
        )
        trajectory.append({
            "step": step + 1,
            "time": s,
            "edge_logit_rms": float(edge_state.square().mean().sqrt().cpu()),
            "spectral_state_rms": float(np.sqrt(np.mean(np.square(spectrum_state)))) if spectrum_state.size else 0.0,
        })

    final_outputs, _ = _predict(model, source, edge_state, spectrum_state, 1.0)
    n = source.number_of_nodes()
    targets: dict[str, Any] = {
        "edge_probabilities": edge_probabilities(edge_state, mask)[0, :n, :n].detach().cpu().numpy(),
        "spectrum": np.asarray(spectrum_state, dtype=np.float64),
    }
    if getattr(model, "predict_clustering_histogram", False):
        targets["clustering_histogram"] = final_outputs["clean_clustering_histogram"][0].detach().cpu().numpy()
    if getattr(model, "predict_orbit_summary", False):
        targets["orbit"] = final_outputs["clean_orbit_summary"][0].detach().cpu().numpy()
    if getattr(model, "predict_induced_graphlet_histogram", False):
        targets["induced_histogram"] = final_outputs["clean_induced_graphlet_histogram"][0].detach().cpu().numpy()
        targets["induced_graphlet_metadata"] = model.induced_graphlet_spec.metadata()
    return targets, {
        "prediction_calls": steps + 1,
        "sampling_steps": steps,
        "edge_sigma": sigma_edge,
        "spectral_sigma": sigma_spec,
        "trajectory": trajectory,
        "independent_laplacian_eigenvalue_diffusion": True,
    }


def _edge_energy(graph: nx.Graph, probabilities: np.ndarray) -> float:
    n = graph.number_of_nodes()
    labels = nx.to_numpy_array(graph, nodelist=list(range(n)), weight=None, dtype=np.int64)
    labels = (labels > 0).astype(np.int64)
    u, v = np.triu_indices(n, 1)
    if not len(u):
        return 0.0
    return float(-np.log(np.maximum(probabilities[u, v, labels[u, v]], 1e-12)).mean())


def refine_graph(source: nx.Graph, targets: dict[str, Any], model, config: dict[str, Any], *, rng: np.random.Generator, prediction_calls: int = 0):
    cfg = JointEdgeSpectralRefinerConfig.from_dict(config.get("topology_refiner"), model=model)
    current = source.copy()
    source_degrees = dict(source.degree())
    probs = np.asarray(targets["edge_probabilities"], dtype=np.float64)
    n = len(source)
    if probs.shape != (n, n, 2) or not np.isfinite(probs).all():
        raise ValueError("Generic soft endpoint probabilities have the wrong shape or contain nonfinite values.")
    target_spectrum = np.asarray(targets["spectrum"], dtype=np.float64)
    if target_spectrum.shape != (n,):
        raise ValueError("Predicted Laplacian target width differs from graph size.")
    target_hist = targets.get("clustering_histogram")
    target_orbit = targets.get("orbit")
    spec = getattr(model, "induced_graphlet_spec", None)
    target_graphlet = targets.get("induced_histogram")
    if cfg.induced_graphlet_weight > 0:
        if targets.get("induced_graphlet_metadata") != spec.metadata():
            raise ValueError("Induced graphlet target catalogue differs from the checkpoint.")
        validate_induced_histogram(target_graphlet, spec)
    graphlet_counter = InducedGraphletCounter(current, spec) if cfg.induced_graphlet_weight > 0 else None
    scale = spectral_scale(source, mode=cfg.normalization)

    weights = {
        "edge": cfg.edge_weight,
        "spectral": cfg.spectral_weight,
        "clustering": cfg.clustering_weight,
        "orbit": cfg.orbit_weight,
        "graphlet": cfg.induced_graphlet_weight,
    }
    weights = {k: v for k, v in weights.items() if v > 0}

    def discrepancies(graph: nx.Graph, *, graphlet_hist=None):
        out: dict[str, float] = {}
        if "edge" in weights:
            out["edge"] = _edge_energy(graph, probs)
        if "spectral" in weights:
            out["spectral"] = spectral_distance(
                laplacian_eigenvalues(graph), target_spectrum,
                metric=cfg.distance, scale=scale,
            )
        if "clustering" in weights:
            out["clustering"] = clustering_histogram_wasserstein(
                extract_clustering_histogram(graph, int(model.clustering_histogram_bins)), target_hist
            )
        if "orbit" in weights:
            out["orbit"] = orbit_summary_distance(extract_orbit_summary(graph), target_orbit, distance=cfg.orbit_distance)
        if "graphlet" in weights:
            if graphlet_hist is None:
                graphlet_hist = graphlet_counter.histogram() if graph is current else graphlet_counter.candidate_histogram(graph)
            out["graphlet"] = induced_histogram_distance(graphlet_hist, target_graphlet, spec)
        return out

    initial = discrepancies(current)
    scales = {key: max(value, cfg.epsilon) for key, value in initial.items()}
    # Spectral normalization is separately controlled by cfg.normalization.  For
    # component balancing, initial-residual scaling is the default.
    component_norm = cfg.component_normalization
    if component_norm not in {"initial", "none"}:
        raise ValueError("topology_refiner.component_normalization must be initial or none.")
    if component_norm == "none":
        scales = {key: 1.0 for key in initial}

    def energy(values):
        return sum(weights[key] * values[key] / scales[key] for key in values)

    current_d = initial
    current_e = energy(current_d)
    trace: list[dict[str, Any]] = []
    visited: set[bytes] = {topology_state_key(current)}
    for step in range(cfg.steps):
        actions, candidates, proposal_diag = propose_valid_topology_swaps(
            current,
            proposal_budget=cfg.proposal_budget,
            valid_candidate_budget=cfg.valid_candidate_budget,
            preserve_connectivity=True,
            rng=rng,
            excluded_states=visited,
        )
        if not actions:
            trace.append({"accepted": False, "terminal_stop": True, "reason": "no_valid_degree_preserving_swap",
                          "prediction_calls": prediction_calls, "num_proposals": int(proposal_diag.get("num_proposals",0)),
                          "num_valid_candidates": 0, "candidate_rejection_reasons": proposal_diag.get("candidate_rejection_reasons",{})})
            break
        best = None
        for action in actions:
            candidate = candidates[action]
            graphlet_hist = graphlet_counter.candidate_histogram(candidate) if graphlet_counter is not None else None
            d = discrepancies(candidate, graphlet_hist=graphlet_hist)
            e = energy(d)
            if best is None or e < best[0]:
                best = (e, action, candidate, d, graphlet_hist)
        assert best is not None
        best_e, action, candidate, candidate_d, candidate_hist = best
        gain = current_e - best_e
        rel = gain / max(abs(current_e), cfg.epsilon)
        row: dict[str, Any] = {
            "step": step + 1,
            "accepted": bool(gain > cfg.min_improvement and rel > cfg.min_relative_improvement),
            "prediction_calls": prediction_calls,
            "num_proposals": int(proposal_diag.get("num_proposals", len(actions))),
            "num_valid_candidates": len(actions),
            "candidate_rejection_reasons": proposal_diag.get("candidate_rejection_reasons", {}),
            "energy_before": current_e,
            "energy_after": best_e,
            "energy_gain": gain,
            "energy_improvement": gain,
            "relative_improvement": rel,
            "relative_energy_improvement": rel,
            "removed": action[0],
            "added": action[1],
        }
        for key in current_d:
            row[f"current_{key}_discrepancy"] = current_d[key]
            row[f"candidate_{key}_discrepancy"] = candidate_d[key]
            row[f"{key}_gain"] = current_d[key] - candidate_d[key]
        if not row["accepted"]:
            row.update(terminal_stop=True, reason="no_sampled_frozen_target_improvement")
            trace.append(row)
            break
        if dict(candidate.degree()) != source_degrees:
            raise AssertionError("Generic hard refiner changed indexed degrees.")
        if not nx.is_connected(candidate):
            raise AssertionError("Generic hard refiner disconnected the graph.")
        current = candidate
        visited.add(topology_state_key(current))
        current_d = candidate_d
        current_e = best_e
        if graphlet_counter is not None:
            graphlet_counter = InducedGraphletCounter(current, spec)
        trace.append(row)
    return current, trace


__all__ = ["JointEdgeSpectralRefinerConfig", "sample_soft_endpoint", "refine_graph"]

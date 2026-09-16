"""Generic GraphER generation with joint soft-edge and Laplacian-spectrum diffusion.

This is the generic analogue of the attributed joint typed edge sampler.  The
neural state contains two continuous bridges:

* symmetric binary edge-category logits over {no-edge, edge}; and
* a configurable spectral state: Laplacian eigenvalues, legacy multiscale
  heat kernels, or a joint eigenvalue + low-frequency eigenspace projector.

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
from grapher.rewiring_mlp.generic.heat_kernel import (
    heat_kernel_distance,
    heat_kernel_stack,
    validate_heat_kernel_stack,
)
from grapher.rewiring_mlp.generic.eigenspace import (
    effective_projector_rank,
    eigenspace_projector_distance,
    laplacian_eigenspace_projector,
    project_to_eigenspace_projector,
    validate_eigenspace_projector,
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


def _advance_heat_kernel(
    current: np.ndarray,
    clean: np.ndarray,
    t: float,
    s: float,
    *,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if not (0.0 <= t < s <= 1.0):
        raise ValueError("Heat-kernel bridge transition requires 0 <= t < s <= 1.")
    current = validate_heat_kernel_stack(current)
    clean = validate_heat_kernel_stack(clean)
    if current.shape != clean.shape:
        raise ValueError("Current and predicted clean heat kernels must have the same shape.")
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("heat_kernel_sigma must be finite and nonnegative.")
    alpha = (s - t) / max(1.0 - t, 1.0e-12)
    std = float(sigma) * math.sqrt(
        max((s - t) * (1.0 - s) / max(1.0 - t, 1.0e-12), 0.0)
    )
    result = current + alpha * (clean - current)
    if std > 0:
        noise = rng.normal(size=result.shape).astype(np.float64)
        noise = 0.5 * (noise + noise.transpose(1, 0, 2))
        for channel in range(noise.shape[-1]):
            rms = float(np.sqrt(np.mean(noise[..., channel] ** 2)))
            if rms > 1.0e-12:
                noise[..., channel] /= rms
        result = result + std * noise
    return 0.5 * (result + result.transpose(1, 0, 2))


def _advance_projector(
    current: np.ndarray,
    clean: np.ndarray,
    t: float,
    s: float,
    *,
    sigma: float,
    rank: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if not (0.0 <= t < s <= 1.0):
        raise ValueError("Projector bridge transition requires 0 <= t < s <= 1.")
    current = np.asarray(current, dtype=np.float64)
    clean = np.asarray(clean, dtype=np.float64)
    if current.shape != clean.shape or current.ndim != 2 or current.shape[0] != current.shape[1]:
        raise ValueError("Current and predicted clean projectors must have matching [n,n] shape.")
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("projector_sigma must be finite and nonnegative.")
    alpha = (s - t) / max(1.0 - t, 1e-12)
    std = float(sigma) * math.sqrt(
        max((s - t) * (1.0 - s) / max(1.0 - t, 1e-12), 0.0)
    )
    result = current + alpha * (clean - current)
    n = result.shape[0]
    if std > 0.0 and n:
        noise = rng.normal(size=result.shape).astype(np.float64)
        noise = 0.5 * (noise + noise.T)
        centering = np.eye(n, dtype=np.float64) - np.ones((n, n), dtype=np.float64) / float(n)
        noise = centering @ noise @ centering
        rms = float(np.sqrt(np.mean(noise ** 2)))
        if rms > 1.0e-12:
            noise /= rms
        result = result + std * noise
        result = centering @ result @ centering
    result = 0.5 * (result + result.T)
    # At t=1 the transition is exactly the model's clean projector. Keep
    # intermediate states continuous/off-manifold, mirroring the training bridge.
    if s >= 1.0 - 1.0e-12:
        result = project_to_eigenspace_projector(result, rank=rank)
    return result


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
        edge_weight = float(weights.get("edge", 1.0))
        spectral_weight = float(weights.get("spectral", 0.1))
        clustering_weight = float(weights.get("clustering", 0.25))
        orbit_weight = float(weights.get("orbit", 1.0))
        induced_graphlet_weight = float(weights.get("graphlet", 0.1))
        active_components = [
            name
            for name, weight in (
                ("edge", edge_weight),
                ("spectral", spectral_weight),
                ("clustering", clustering_weight),
                ("orbit", orbit_weight),
                ("graphlet", induced_graphlet_weight),
            )
            if weight > 0.0
        ]
        kwargs.update(
            edge_weight=edge_weight,
            spectral_weight=spectral_weight,
            clustering_weight=clustering_weight,
            orbit_weight=orbit_weight,
            induced_graphlet_weight=induced_graphlet_weight,
            guidance_mode="_".join(active_components),
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
def _predict(
    model,
    source: nx.Graph,
    edge_state: torch.Tensor,
    spectrum_state: np.ndarray,
    time: float,
    heat_kernel_state: np.ndarray | None = None,
    projector_state: np.ndarray | None = None,
):
    device = next(model.parameters()).device
    n = source.number_of_nodes()
    source_spectrum = laplacian_eigenvalues(source).astype(np.float32)
    smoothing = float(getattr(model, "edge_smoothing", 0.01))
    source_logits = _source_edge_logits(source, smoothing, device)[0].detach().cpu().numpy().astype(np.float32)
    current_logits = edge_state[0].detach().cpu().numpy().astype(np.float32)
    labels = _adjacency_labels(source)[0].numpy().astype(np.int64)
    representation = str(getattr(model, "spectral_representation", "eigenvalues"))
    source_heat = clean_heat = None
    source_projector = clean_projector = None
    if representation == "heat_kernel":
        source_heat = heat_kernel_stack(
            source,
            times=getattr(model, "heat_kernel_times", (0.25, 1.0, 4.0)),
            normalization=getattr(model, "input_normalization", "mean_degree"),
        ).astype(np.float32)
        if heat_kernel_state is None:
            heat_kernel_state = source_heat
        clean_heat = np.zeros_like(source_heat, dtype=np.float32)
    elif representation == "lambda_projector":
        source_projector = laplacian_eigenspace_projector(
            source, rank=int(getattr(model, "projector_rank", 4))
        ).astype(np.float32)
        if projector_state is None:
            projector_state = source_projector
        clean_projector = np.zeros_like(source_projector, dtype=np.float32)
    example = TopologySpectralExample(
        current_graph=source,
        time=float(time),
        current_spectrum=np.asarray(spectrum_state, dtype=np.float32),
        source_spectrum=source_spectrum,
        clean_spectrum_target=np.zeros(n, dtype=np.float32),
        current_heat_kernel=(None if heat_kernel_state is None else np.asarray(heat_kernel_state, dtype=np.float32)),
        source_heat_kernel=source_heat,
        clean_heat_kernel_target=clean_heat,
        current_projector=(None if projector_state is None else np.asarray(projector_state, dtype=np.float32)),
        source_projector=source_projector,
        clean_projector_target=clean_projector,
        current_edge_logits=(current_logits if getattr(model, "predict_edge_state", False) else None),
        source_edge_logits=(source_logits if getattr(model, "predict_edge_state", False) else None),
        clean_edge_logits_target=(source_logits if getattr(model, "predict_edge_state", False) else None),
        clean_edge_labels_target=(labels if getattr(model, "predict_edge_state", False) else None),
    )
    batch = collate_spectral_examples([example]).to(device)
    return model(batch), batch


@torch.no_grad()
def sample_soft_endpoint(model, source: nx.Graph, config: dict[str, Any], *, seed: int):
    model.eval()
    device = next(model.parameters()).device
    edge_cfg = dict(config.get("edge_diffusion", {}) or {})
    diff_cfg = dict(config.get("summary_diffusion", {}) or {})
    steps = int(edge_cfg.get("sampling_steps", 32))
    if steps < 2:
        raise ValueError("edge_diffusion.sampling_steps must be at least 2.")
    sigma_edge = float(edge_cfg.get("sigma", 1.0))
    sigma_spec = float(diff_cfg.get("spectral_sigma", 0.2))
    sigma_heat = float(diff_cfg.get("heat_kernel_sigma", sigma_spec))
    sigma_projector = float(diff_cfg.get("projector_sigma", sigma_spec))
    smoothing = float(edge_cfg.get("smoothing", getattr(model, "edge_smoothing", 0.01)))
    if not math.isclose(smoothing, float(getattr(model, "edge_smoothing", smoothing)), rel_tol=0, abs_tol=1e-12):
        raise ValueError("edge_diffusion.smoothing differs from the trained checkpoint.")

    representation = str(getattr(model, "spectral_representation", "eigenvalues"))
    mask = torch.ones((1, source.number_of_nodes()), dtype=torch.bool, device=device)
    labels = _adjacency_labels(source).to(device)
    edge_state = labels_to_logits(labels, mask, smoothing)
    spectrum_state = laplacian_eigenvalues(source).astype(np.float64)
    heat_state = None
    projector_state = None
    if representation == "heat_kernel":
        heat_state = heat_kernel_stack(
            source,
            times=getattr(model, "heat_kernel_times", (0.25, 1.0, 4.0)),
            normalization=str(config.get("spectral_prediction", {}).get(
                "heat_kernel_normalization", getattr(model, "input_normalization", "mean_degree")
            )),
        )
    elif representation == "lambda_projector":
        projector_state = laplacian_eigenspace_projector(
            source, rank=int(getattr(model, "projector_rank", 4))
        )
    scale = spectral_scale(source, mode=str(config.get("spectral_prediction", {}).get("normalization", "mean_degree")))
    rng = np.random.default_rng(int(seed))
    generator = torch.Generator(device=device).manual_seed(int(seed))
    trajectory = []
    for step in range(steps):
        t = step / steps
        next_t = (step + 1) / steps
        outputs, _batch = _predict(
            model, source, edge_state, spectrum_state, t, heat_state, projector_state
        )
        if getattr(model, "predict_edge_state", False):
            clean_edge = outputs["clean_edge_logits"]
            edge_state = advance_edges(edge_state, clean_edge, t, next_t, mask, sigma_edge, generator=generator)
        if representation == "heat_kernel":
            clean_heat = outputs["clean_heat_kernel"][0, : source.number_of_nodes(), : source.number_of_nodes()].detach().cpu().numpy().astype(np.float64)
            heat_state = _advance_heat_kernel(
                heat_state, clean_heat, t, next_t, sigma=sigma_heat, rng=rng
            )
            state_rms = float(np.sqrt(np.mean(np.square(heat_state)))) if heat_state.size else 0.0
        else:
            clean_spectrum = outputs["clean_spectrum"][0, : source.number_of_nodes()].detach().cpu().numpy().astype(np.float64)
            spectrum_state = _advance_spectrum(
                spectrum_state, clean_spectrum, t, next_t,
                sigma=sigma_spec, scale=scale, rng=rng,
            )
            if representation == "lambda_projector":
                clean_projector = outputs["clean_projector"][0, : source.number_of_nodes(), : source.number_of_nodes()].detach().cpu().numpy().astype(np.float64)
                projector_state = _advance_projector(
                    projector_state, clean_projector, t, next_t,
                    sigma=sigma_projector, rank=int(getattr(model, "projector_rank", 4)), rng=rng,
                )
                projector_rms = float(np.sqrt(np.mean(np.square(projector_state)))) if projector_state.size else 0.0
                state_rms = float(np.sqrt(
                    0.5 * np.mean(np.square(spectrum_state)) + 0.5 * np.mean(np.square(projector_state))
                ))
            else:
                projector_rms = None
                state_rms = float(np.sqrt(np.mean(np.square(spectrum_state)))) if spectrum_state.size else 0.0
        trajectory.append({
            "step": step + 1,
            "time": next_t,
            "edge_logit_rms": float(edge_state.square().mean().sqrt().cpu()),
            "spectral_state_rms": state_rms,
            "projector_state_rms": (projector_rms if representation == "lambda_projector" else None),
            "spectral_representation": representation,
        })

    final_outputs, _ = _predict(
        model, source, edge_state, spectrum_state, 1.0, heat_state, projector_state
    )
    n = source.number_of_nodes()
    targets: dict[str, Any] = {
        "edge_probabilities": edge_probabilities(edge_state, mask)[0, :n, :n].detach().cpu().numpy(),
        "spectral_representation": representation,
    }
    if representation == "heat_kernel":
        targets["heat_kernel"] = np.asarray(heat_state, dtype=np.float64)
        targets["heat_kernel_times"] = list(getattr(model, "heat_kernel_times", (0.25, 1.0, 4.0)))
    else:
        targets["spectrum"] = np.asarray(spectrum_state, dtype=np.float64)
        if representation == "lambda_projector":
            targets["projector"] = np.asarray(projector_state, dtype=np.float64)
            targets["projector_rank"] = int(getattr(model, "projector_rank", 4))
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
        "spectral_sigma": (sigma_spec if representation in {"eigenvalues", "lambda_projector"} else None),
        "heat_kernel_sigma": (sigma_heat if representation == "heat_kernel" else None),
        "projector_sigma": (sigma_projector if representation == "lambda_projector" else None),
        "projector_rank": (int(getattr(model, "projector_rank", 4)) if representation == "lambda_projector" else None),
        "spectral_representation": representation,
        "heat_kernel_times": (list(getattr(model, "heat_kernel_times", ())) if representation == "heat_kernel" else None),
        "trajectory": trajectory,
        "independent_laplacian_eigenvalue_diffusion": representation == "eigenvalues",
        "joint_laplacian_heat_kernel_diffusion": representation == "heat_kernel",
        "joint_laplacian_lambda_projector_diffusion": representation == "lambda_projector",
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
    representation = str(targets.get("spectral_representation", getattr(model, "spectral_representation", "eigenvalues")))
    target_spectrum = None
    target_heat_kernel = None
    target_projector = None
    if representation == "heat_kernel":
        target_heat_kernel = validate_heat_kernel_stack(
            targets.get("heat_kernel"), n=n, num_scales=len(getattr(model, "heat_kernel_times", ()))
        )
    else:
        target_spectrum = np.asarray(targets["spectrum"], dtype=np.float64)
        if target_spectrum.shape != (n,):
            raise ValueError("Predicted Laplacian target width differs from graph size.")
        if representation == "lambda_projector":
            target_projector = validate_eigenspace_projector(
                targets.get("projector"), n=n
            )
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
    spectral_cfg = dict(config.get("spectral_prediction", {}) or {})
    lambda_weight = float(spectral_cfg.get("lambda_weight", 1.0))
    projector_weight = float(spectral_cfg.get("projector_weight", 1.0))
    projector_distance_name = str(spectral_cfg.get("projector_distance", "chordal"))
    projector_rank = int(getattr(model, "projector_rank", spectral_cfg.get("projector_rank", 4)))
    if lambda_weight < 0.0 or projector_weight < 0.0 or lambda_weight + projector_weight <= 0.0:
        raise ValueError("lambda_weight and projector_weight must be nonnegative with positive sum.")

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
            if representation == "heat_kernel":
                candidate_heat = heat_kernel_stack(
                    graph,
                    times=getattr(model, "heat_kernel_times", (0.25, 1.0, 4.0)),
                    normalization=str(config.get("spectral_prediction", {}).get(
                        "heat_kernel_normalization", getattr(model, "input_normalization", "mean_degree")
                    )),
                )
                out["spectral"] = heat_kernel_distance(
                    candidate_heat, target_heat_kernel, metric=cfg.distance
                )
            elif representation == "lambda_projector":
                lambda_discrepancy = spectral_distance(
                    laplacian_eigenvalues(graph), target_spectrum,
                    metric=cfg.distance, scale=scale,
                )
                candidate_projector = laplacian_eigenspace_projector(
                    graph, rank=projector_rank
                )
                projector_discrepancy = eigenspace_projector_distance(
                    candidate_projector, target_projector, rank=projector_rank,
                    metric=projector_distance_name,
                )
                out["spectral"] = (
                    lambda_weight * lambda_discrepancy
                    + projector_weight * projector_discrepancy
                ) / max(lambda_weight + projector_weight, 1.0e-12)
                out["lambda"] = lambda_discrepancy
                out["projector"] = projector_discrepancy
            else:
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
        return sum(weights[key] * values[key] / scales[key] for key in weights)

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

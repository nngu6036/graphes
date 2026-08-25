from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import networkx as nx
import numpy as np
import torch

from grapher.rewiring_mlp.core.rewiring import Action
from grapher.rewiring_mlp.generic.basis import TopologyGraphletBasis
from grapher.rewiring_mlp.generic.data import normalize_topology_graph
from grapher.rewiring_mlp.generic.graphlet_diffusion import (
    GraphletLogitBridgeSchedule,
    candidate_graphlet_logits_from_counts,
    extract_topology_graphlet_simplex,
    graphlet_clr_to_simplex,
    graphlet_simplex_from_counts,
    graphlet_logit_distance,
    graphlet_simplex_to_clr,
)
from grapher.rewiring_mlp.generic.graphlets import TopologyGraphletCounts
from grapher.rewiring_mlp.generic.rewiring import (
    propose_valid_topology_swaps,
    topology_state_key,
)
from grapher.rewiring_mlp.generic.spectral import (
    laplacian_eigenvalues,
    spectral_distance,
    spectral_scale,
    spectrum_moments,
)
from grapher.rewiring_mlp.generic.spectral_data import (
    TopologySpectralExample,
    collate_spectral_examples,
)
from grapher.rewiring_mlp.generic.spectral_model import (
    TopologySpectralGraphletTransformerPredictor,
)
from grapher.rewiring_mlp.generic.spectral_refiner import SpectralRefinerConfig


@dataclass(frozen=True)
class SpectralGraphletPrediction:
    clean_spectrum: np.ndarray
    current_spectrum: np.ndarray
    clean_graphlet_logits: np.ndarray
    clean_graphlet_probabilities: np.ndarray
    current_graphlet_logits: np.ndarray
    current_graphlet_probabilities: np.ndarray
    graphlet_coordinate_mask: np.ndarray
    trace: float
    second_moment: float


@dataclass(frozen=True)
class SpectralGraphletRefinerConfig(SpectralRefinerConfig):
    """Dual global-spectrum/local-graphlet guidance configuration."""

    graphlet_distance: str = "clr_rmse"
    graphlet_logit_epsilon: float = 1.0e-5
    graphlet_size_weights: dict[str, float] = field(default_factory=dict)
    graphlet_bridge_schedule: str = "cosine"
    graphlet_bridge_min_clean_mix: float = 0.05
    graphlet_bridge_max_clean_mix: float = 1.0
    graphlet_bridge_power: float = 2.0
    graphlet_plateau_expand_factor: float = 1.35

    guidance_weight_schedule: str = "cosine"
    spectral_weight_initial: float = 1.0
    spectral_weight_final: float = 0.5
    graphlet_weight_initial: float = 0.10
    graphlet_weight_final: float = 1.25
    guidance_weight_power: float = 1.5

    @property
    def graphlet_bridge(self) -> GraphletLogitBridgeSchedule:
        return GraphletLogitBridgeSchedule(
            schedule=self.graphlet_bridge_schedule,
            min_clean_mix=self.graphlet_bridge_min_clean_mix,
            max_clean_mix=self.graphlet_bridge_max_clean_mix,
            power=self.graphlet_bridge_power,
        )

    def guidance_weights_at(self, progress: float) -> tuple[float, float]:
        p = float(np.clip(progress, 0.0, 1.0))
        if self.guidance_weight_schedule == "linear":
            shaped = p
        elif self.guidance_weight_schedule == "cosine":
            shaped = 0.5 - 0.5 * np.cos(np.pi * p)
        elif self.guidance_weight_schedule == "power":
            shaped = p ** float(self.guidance_weight_power)
        else:
            raise ValueError(
                f"Unknown global_to_local schedule: {self.guidance_weight_schedule!r}."
            )
        spectral = self.spectral_weight_initial + (
            self.spectral_weight_final - self.spectral_weight_initial
        ) * shaped
        graphlet = self.graphlet_weight_initial + (
            self.graphlet_weight_final - self.graphlet_weight_initial
        ) * shaped
        return float(spectral), float(graphlet)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None = None) -> "SpectralGraphletRefinerConfig":
        values = dict(data or {})
        mode = str(values.get("mode", "spectral_graphlet")).lower()
        if mode not in {
            "spectral_graphlet",
            "spectral_graphlet_diffusion",
            "dual_diffusion",
            "spectral+graphlet",
        }:
            raise ValueError(
                "SpectralGraphletRefinerConfig requires topology_refiner.mode: spectral_graphlet."
            )
        base_values = dict(values)
        base_values["mode"] = "spectral"
        base = SpectralRefinerConfig.from_dict(base_values)
        graphlet = dict(values.get("graphlet_guidance", {}) or {})
        global_to_local = dict(values.get("global_to_local", {}) or {})
        bridge = GraphletLogitBridgeSchedule.from_dict(graphlet)
        size_weights_raw = graphlet.get("size_weights", {}) or {}
        if not isinstance(size_weights_raw, Mapping):
            raise ValueError("graphlet_guidance.size_weights must be a mapping keyed by graphlet order.")
        size_weights = {str(key): float(value) for key, value in size_weights_raw.items()}
        config = cls(
            **base.__dict__,
            graphlet_distance=str(graphlet.get("distance", "clr_rmse")).lower(),
            graphlet_logit_epsilon=float(graphlet.get("logit_epsilon", 1.0e-5)),
            graphlet_size_weights=size_weights,
            graphlet_bridge_schedule=bridge.schedule,
            graphlet_bridge_min_clean_mix=bridge.min_clean_mix,
            graphlet_bridge_max_clean_mix=bridge.max_clean_mix,
            graphlet_bridge_power=bridge.power,
            graphlet_plateau_expand_factor=float(
                graphlet.get("plateau_expand_factor", base.plateau_expand_factor)
            ),
            guidance_weight_schedule=str(global_to_local.get("schedule", "cosine")).lower(),
            spectral_weight_initial=float(global_to_local.get("spectral_initial", 1.0)),
            spectral_weight_final=float(global_to_local.get("spectral_final", 0.5)),
            graphlet_weight_initial=float(global_to_local.get("graphlet_initial", 0.10)),
            graphlet_weight_final=float(global_to_local.get("graphlet_final", 1.25)),
            guidance_weight_power=float(global_to_local.get("power", 1.5)),
        )
        if config.graphlet_logit_epsilon <= 0.0 or not np.isfinite(config.graphlet_logit_epsilon):
            raise ValueError("graphlet_guidance.logit_epsilon must be finite and positive.")
        if config.graphlet_plateau_expand_factor <= 1.0 and config.expand_on_plateau:
            raise ValueError("graphlet_guidance.plateau_expand_factor must exceed 1.")
        if config.guidance_weight_schedule not in {"linear", "cosine", "power"}:
            raise ValueError("global_to_local.schedule must be linear, cosine, or power.")
        weights = [
            config.spectral_weight_initial,
            config.spectral_weight_final,
            config.graphlet_weight_initial,
            config.graphlet_weight_final,
        ]
        if any((not np.isfinite(value)) or value < 0.0 for value in weights):
            raise ValueError("Global-to-local guidance weights must be finite and nonnegative.")
        if config.guidance_weight_power <= 0.0 or not np.isfinite(config.guidance_weight_power):
            raise ValueError("global_to_local.power must be finite and positive.")
        return config


@torch.no_grad()
def predict_clean_spectrum_and_graphlets(
    model: TopologySpectralGraphletTransformerPredictor,
    graph: nx.Graph,
    *,
    graphlet_basis: TopologyGraphletBasis,
    time: float,
    device: torch.device | str,
    graphlet_logit_epsilon: float,
    conditioning_graph: nx.Graph | None = None,
    source_spectrum: np.ndarray | None = None,
    source_graphlet_probabilities: np.ndarray | None = None,
    source_graphlet_logits: np.ndarray | None = None,
    current_graphlet_counts: TopologyGraphletCounts | None = None,
) -> SpectralGraphletPrediction:
    model.eval()
    n = graph.number_of_nodes()
    context_graph = graph if conditioning_graph is None else conditioning_graph
    current_spectrum = laplacian_eigenvalues(graph)
    source_spectrum_values = (
        laplacian_eigenvalues(context_graph)
        if source_spectrum is None
        else np.asarray(source_spectrum, dtype=np.float64).reshape(-1)
    )
    if current_graphlet_counts is None:
        current_prob, current_mask, _ = extract_topology_graphlet_simplex(
            graph,
            graphlet_basis=graphlet_basis,
        )
    else:
        current_prob, current_mask = graphlet_simplex_from_counts(
            current_graphlet_counts,
            num_nodes=graph.number_of_nodes(),
            graphlet_basis=graphlet_basis,
        )
    current_logits = graphlet_simplex_to_clr(
        current_prob,
        graphlet_basis=graphlet_basis,
        epsilon=graphlet_logit_epsilon,
        coordinate_mask=current_mask,
    )
    if source_graphlet_probabilities is None or source_graphlet_logits is None:
        source_prob, source_mask, _ = extract_topology_graphlet_simplex(
            context_graph, graphlet_basis=graphlet_basis
        )
        if not np.array_equal(source_mask, current_mask):
            raise AssertionError("Source/current graphlet masks must match for equal-size graphs.")
        source_logits = graphlet_simplex_to_clr(
            source_prob,
            graphlet_basis=graphlet_basis,
            epsilon=graphlet_logit_epsilon,
            coordinate_mask=source_mask,
        )
    else:
        source_prob = np.asarray(source_graphlet_probabilities, dtype=np.float64).reshape(-1)
        source_logits = np.asarray(source_graphlet_logits, dtype=np.float64).reshape(-1)
    example = TopologySpectralExample(
        current_graph=context_graph,
        time=float(time),
        current_spectrum=current_spectrum.astype(np.float32),
        source_spectrum=source_spectrum_values.astype(np.float32),
        clean_spectrum_target=np.zeros(n, dtype=np.float32),
        current_graphlet_probabilities=current_prob.astype(np.float32),
        source_graphlet_probabilities=source_prob.astype(np.float32),
        clean_graphlet_probabilities_target=current_prob.astype(np.float32),
        current_graphlet_logits=current_logits.astype(np.float32),
        source_graphlet_logits=source_logits.astype(np.float32),
        clean_graphlet_logits_target=current_logits.astype(np.float32),
        graphlet_coordinate_mask=current_mask.astype(np.bool_),
    )
    batch = collate_spectral_examples([example]).to(device)
    outputs = model(batch)
    clean_spectrum = outputs["clean_spectrum"][0, :n].detach().cpu().numpy().astype(np.float64)
    clean_logits = outputs["clean_graphlet_logits"][0].detach().cpu().numpy().astype(np.float64)
    clean_prob = outputs["clean_graphlet_probabilities"][0].detach().cpu().numpy().astype(np.float64)
    trace, second = spectrum_moments(clean_spectrum)
    return SpectralGraphletPrediction(
        clean_spectrum=clean_spectrum,
        current_spectrum=current_spectrum,
        clean_graphlet_logits=clean_logits,
        clean_graphlet_probabilities=clean_prob,
        current_graphlet_logits=current_logits,
        current_graphlet_probabilities=current_prob,
        graphlet_coordinate_mask=current_mask,
        trace=trace,
        second_moment=second,
    )


def _select_row(
    rows: Sequence[dict[str, Any]],
    *,
    config: SpectralGraphletRefinerConfig,
    rng: np.random.Generator,
) -> tuple[int | None, float, list[float]]:
    improvements = np.asarray([float(row["energy_improvement"]) for row in rows], dtype=np.float64)
    relative = np.asarray(
        [float(row["relative_energy_improvement"]) for row in rows], dtype=np.float64
    )
    scores = np.concatenate([improvements, np.asarray([0.0], dtype=np.float64)])
    if config.accept_only_improving:
        eligible = improvements > float(config.min_improvement)
        eligible &= relative > float(config.min_relative_improvement)
        scores[:-1][~eligible] = -np.inf
        if np.any(np.isfinite(scores[:-1])):
            scores[-1] = -np.inf
    finite = np.isfinite(scores)
    probabilities = np.zeros_like(scores)
    shifted = scores.copy()
    shifted[finite] -= float(np.max(shifted[finite]))
    probabilities[finite] = np.exp(shifted[finite] / float(config.temperature))
    probabilities /= max(float(probabilities.sum()), 1.0e-12)
    if config.selection in {"greedy", "argmax"}:
        best = float(np.max(scores))
        maxima = np.flatnonzero(np.isclose(scores, best, atol=1.0e-12))
        selected = int(rng.choice(maxima))
    else:
        selected = int(rng.choice(len(scores), p=probabilities))
    stop_index = len(rows)
    return (
        None if selected == stop_index else selected,
        float(probabilities[-1]),
        probabilities.tolist(),
    )


def prepare_spectral_graphlet_candidate_states(
    graph: nx.Graph,
    candidates: Sequence[Action],
    *,
    candidate_graphs: dict[Action, nx.Graph],
    current_graphlet_counts: TopologyGraphletCounts,
    graphlet_basis: TopologyGraphletBasis,
    config: SpectralGraphletRefinerConfig,
) -> dict[str, Any]:
    """Compute target-independent candidate summaries exactly once.

    Plateau expansion changes only the continuous denoising target.  Candidate
    spectra and candidate graphlet states depend only on the current graph and
    swap action, so recomputing them after every bridge expansion multiplies
    the most expensive work without changing the candidates.  This cache is
    local to one rewiring decision and is rebuilt only after the graph changes.
    """

    current_spectrum = laplacian_eigenvalues(graph)
    scale = spectral_scale(graph, mode=config.normalization)
    current_prob, current_mask = graphlet_simplex_from_counts(
        current_graphlet_counts,
        num_nodes=graph.number_of_nodes(),
        graphlet_basis=graphlet_basis,
    )
    current_logits = graphlet_simplex_to_clr(
        current_prob,
        graphlet_basis=graphlet_basis,
        epsilon=config.graphlet_logit_epsilon,
        coordinate_mask=current_mask,
    )
    prepared_rows: list[dict[str, Any]] = []
    for action in candidates:
        candidate = candidate_graphs[action]
        spectrum = laplacian_eigenvalues(candidate)
        candidate_logits, candidate_prob, candidate_counts = (
            candidate_graphlet_logits_from_counts(
                graph,
                candidate,
                action,
                current_counts=current_graphlet_counts,
                graphlet_basis=graphlet_basis,
                epsilon=config.graphlet_logit_epsilon,
            )
        )
        prepared_rows.append(
            {
                "action": action,
                "candidate_graph": candidate,
                "candidate_spectrum": spectrum,
                "candidate_graphlet_logits": candidate_logits,
                "candidate_graphlet_probabilities": candidate_prob,
                "candidate_graphlet_counts": candidate_counts,
            }
        )
    return {
        "current_spectrum": current_spectrum,
        "current_graphlet_probabilities": current_prob,
        "current_graphlet_logits": current_logits,
        "current_graphlet_mask": current_mask,
        "spectral_scale": float(scale),
        "rows": prepared_rows,
    }


def score_prepared_spectral_graphlet_candidates(
    prepared: Mapping[str, Any],
    *,
    graphlet_basis: TopologyGraphletBasis,
    clean_spectrum: np.ndarray,
    next_spectrum_target: np.ndarray,
    clean_graphlet_logits: np.ndarray,
    next_graphlet_logits_target: np.ndarray,
    graphlet_coordinate_mask: np.ndarray,
    spectral_weight: float,
    graphlet_weight: float,
    config: SpectralGraphletRefinerConfig,
) -> list[dict[str, Any]]:
    """Rescore cached candidate summaries against one bridge target."""

    current_spectrum = np.asarray(prepared["current_spectrum"], dtype=np.float64)
    current_logits = np.asarray(prepared["current_graphlet_logits"], dtype=np.float64)
    scale = float(prepared["spectral_scale"])
    current_spectral = spectral_distance(
        current_spectrum,
        next_spectrum_target,
        metric=config.distance,
        scale=scale,
        low_frequency_weight=config.low_frequency_weight,
        low_frequency_cutoff=config.low_frequency_cutoff,
    )
    current_clean_spectral = spectral_distance(
        current_spectrum,
        clean_spectrum,
        metric=config.distance,
        scale=scale,
        low_frequency_weight=config.low_frequency_weight,
        low_frequency_cutoff=config.low_frequency_cutoff,
    )
    current_graphlet = graphlet_logit_distance(
        current_logits,
        next_graphlet_logits_target,
        graphlet_basis=graphlet_basis,
        coordinate_mask=graphlet_coordinate_mask,
        metric=config.graphlet_distance,
        size_weights=config.graphlet_size_weights,
    )
    current_clean_graphlet = graphlet_logit_distance(
        current_logits,
        clean_graphlet_logits,
        graphlet_basis=graphlet_basis,
        coordinate_mask=graphlet_coordinate_mask,
        metric=config.graphlet_distance,
        size_weights=config.graphlet_size_weights,
    )
    current_energy = spectral_weight * current_spectral + graphlet_weight * current_graphlet
    current_clean_energy = (
        spectral_weight * current_clean_spectral + graphlet_weight * current_clean_graphlet
    )

    rows: list[dict[str, Any]] = []
    for cached in prepared["rows"]:
        spectrum = np.asarray(cached["candidate_spectrum"], dtype=np.float64)
        candidate_logits = np.asarray(cached["candidate_graphlet_logits"], dtype=np.float64)
        candidate_spectral = spectral_distance(
            spectrum,
            next_spectrum_target,
            metric=config.distance,
            scale=scale,
            low_frequency_weight=config.low_frequency_weight,
            low_frequency_cutoff=config.low_frequency_cutoff,
        )
        candidate_clean_spectral = spectral_distance(
            spectrum,
            clean_spectrum,
            metric=config.distance,
            scale=scale,
            low_frequency_weight=config.low_frequency_weight,
            low_frequency_cutoff=config.low_frequency_cutoff,
        )
        candidate_graphlet = graphlet_logit_distance(
            candidate_logits,
            next_graphlet_logits_target,
            graphlet_basis=graphlet_basis,
            coordinate_mask=graphlet_coordinate_mask,
            metric=config.graphlet_distance,
            size_weights=config.graphlet_size_weights,
        )
        candidate_clean_graphlet = graphlet_logit_distance(
            candidate_logits,
            clean_graphlet_logits,
            graphlet_basis=graphlet_basis,
            coordinate_mask=graphlet_coordinate_mask,
            metric=config.graphlet_distance,
            size_weights=config.graphlet_size_weights,
        )
        candidate_energy = spectral_weight * candidate_spectral + graphlet_weight * candidate_graphlet
        candidate_clean_energy = (
            spectral_weight * candidate_clean_spectral
            + graphlet_weight * candidate_clean_graphlet
        )
        gain = float(current_energy - candidate_energy)
        relative = float(
            gain / max(abs(current_energy), float(config.relative_improvement_epsilon))
        )
        row = dict(cached)
        row.update(
            {
                "spectral_weight": float(spectral_weight),
                "graphlet_weight": float(graphlet_weight),
                "current_spectral_discrepancy": float(current_spectral),
                "candidate_spectral_discrepancy": float(candidate_spectral),
                "current_clean_spectral_discrepancy": float(current_clean_spectral),
                "candidate_clean_spectral_discrepancy": float(candidate_clean_spectral),
                "current_graphlet_discrepancy": float(current_graphlet),
                "candidate_graphlet_discrepancy": float(candidate_graphlet),
                "current_clean_graphlet_discrepancy": float(current_clean_graphlet),
                "candidate_clean_graphlet_discrepancy": float(candidate_clean_graphlet),
                "current_energy": float(current_energy),
                "candidate_energy": float(candidate_energy),
                "current_clean_energy": float(current_clean_energy),
                "candidate_clean_energy": float(candidate_clean_energy),
                "spectral_gain": float(current_spectral - candidate_spectral),
                "clean_spectral_gain": float(current_clean_spectral - candidate_clean_spectral),
                "graphlet_gain": float(current_graphlet - candidate_graphlet),
                "clean_graphlet_gain": float(current_clean_graphlet - candidate_clean_graphlet),
                "projection_residual": float(candidate_energy),
                "spectral_projection_residual": float(candidate_spectral),
                "graphlet_projection_residual": float(candidate_graphlet),
                "energy_improvement": gain,
                "relative_energy_improvement": relative,
            }
        )
        rows.append(row)
    return rows


def score_spectral_graphlet_candidates(
    graph: nx.Graph,
    candidates: Sequence[Action],
    *,
    candidate_graphs: dict[Action, nx.Graph],
    current_graphlet_counts: TopologyGraphletCounts,
    graphlet_basis: TopologyGraphletBasis,
    clean_spectrum: np.ndarray,
    next_spectrum_target: np.ndarray,
    clean_graphlet_logits: np.ndarray,
    next_graphlet_logits_target: np.ndarray,
    graphlet_coordinate_mask: np.ndarray,
    spectral_weight: float,
    graphlet_weight: float,
    config: SpectralGraphletRefinerConfig,
) -> list[dict[str, Any]]:
    """Backward-compatible one-shot candidate preparation and scoring."""

    prepared = prepare_spectral_graphlet_candidate_states(
        graph,
        candidates,
        candidate_graphs=candidate_graphs,
        current_graphlet_counts=current_graphlet_counts,
        graphlet_basis=graphlet_basis,
        config=config,
    )
    return score_prepared_spectral_graphlet_candidates(
        prepared,
        graphlet_basis=graphlet_basis,
        clean_spectrum=clean_spectrum,
        next_spectrum_target=next_spectrum_target,
        clean_graphlet_logits=clean_graphlet_logits,
        next_graphlet_logits_target=next_graphlet_logits_target,
        graphlet_coordinate_mask=graphlet_coordinate_mask,
        spectral_weight=spectral_weight,
        graphlet_weight=graphlet_weight,
        config=config,
    )

def _format_spectrum(values: np.ndarray, limit: int) -> str:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    limit = max(int(limit), 1)
    if array.size <= limit:
        body = ", ".join(f"{value:.4f}" for value in array)
    else:
        left = max(limit // 2, 1)
        right = max(limit - left, 1)
        body = ", ".join(f"{value:.4f}" for value in array[:left])
        body += ", ..., " + ", ".join(f"{value:.4f}" for value in array[-right:])
    return f"[{body}]"


def _format_graphlet_probabilities(
    values: np.ndarray,
    *,
    graphlet_basis: TopologyGraphletBasis,
    top_k: int = 3,
) -> str:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    parts: list[str] = []
    for key, (start, stop) in zip(graphlet_basis.sizes, graphlet_basis.simplex_slices):
        block = vector[start:stop]
        names = list(graphlet_basis.keys_by_k[key]) + ["disconnected"]
        ranked = np.argsort(-block)[: max(int(top_k), 1)]
        body = ",".join(f"{names[int(i)]}:{block[int(i)]:.3f}" for i in ranked)
        parts.append(f"k={key}[{body}]")
    return " ".join(parts)


def _debug_print(
    config: SpectralGraphletRefinerConfig,
    decision_step: int,
    prefix: str,
    message: str,
) -> None:
    if not config.debug_enabled or decision_step % config.debug_print_every != 0:
        return
    print(f"[GraphER/SpectralGraphlet]{prefix} {message}", flush=True)


def refine_graph_with_spectral_graphlet_predictions(
    graph: nx.Graph,
    *,
    model: TopologySpectralGraphletTransformerPredictor,
    graphlet_basis: TopologyGraphletBasis,
    refiner_config: SpectralGraphletRefinerConfig | dict[str, Any] | None = None,
    device: torch.device | str = "cpu",
    rng: np.random.Generator | None = None,
    return_trace: bool = False,
    prediction_fn: Any | None = None,
    debug_context: str | None = None,
) -> nx.Graph | tuple[nx.Graph, list[dict[str, Any]]]:
    """Denoise global spectrum and local graphlet logits through valid swaps.

    The neural network predicts clean x0 targets.  Separate bridge schedules
    derive the desired next spectrum and graphlet CLR state.  Candidate graphlet
    states are computed by exact local-delta updates; the actual graph never
    leaves the degree-preserving, simple, connected realization space.
    """

    cfg = (
        refiner_config
        if isinstance(refiner_config, SpectralGraphletRefinerConfig)
        else SpectralGraphletRefinerConfig.from_dict(refiner_config)
    )
    if tuple(model.graphlet_block_widths) != tuple(graphlet_basis.simplex_block_widths):
        raise ValueError("Joint predictor graphlet widths do not match the configured graphlet basis.")
    generator = rng if rng is not None else np.random.default_rng(0)
    current = normalize_topology_graph(graph)
    conditioning_graph = current.copy()
    source_spectrum = laplacian_eigenvalues(conditioning_graph)
    source_probabilities, source_graphlet_mask, source_graphlet_counts = (
        extract_topology_graphlet_simplex(
            conditioning_graph, graphlet_basis=graphlet_basis
        )
    )
    source_graphlet_logits = graphlet_simplex_to_clr(
        source_probabilities,
        graphlet_basis=graphlet_basis,
        epsilon=cfg.graphlet_logit_epsilon,
        coordinate_mask=source_graphlet_mask,
    )
    if current.number_of_nodes() > 1 and not nx.is_connected(current):
        raise ValueError("Spectral+graphlet refinement requires a connected source graph.")
    initial_degrees = [int(current.degree(node)) for node in sorted(current.nodes())]
    # The refinement starts from the same graph as the fixed source context,
    # so reuse its exact count cache instead of recounting graphlets.  Every
    # accepted candidate below then advances this cache with an exact local
    # delta update.
    current_counts: TopologyGraphletCounts = {
        key: dict(counts) for key, counts in source_graphlet_counts.items()
    }
    visited = {topology_state_key(current)}
    trace: list[dict[str, Any]] = []
    prediction: SpectralGraphletPrediction | None = None
    accepted_steps = 0
    accepted_since_prediction = 0
    prediction_calls = 0
    prediction_block = -1
    prediction_horizon = 1
    prediction_progress = 0.0
    prediction_time = 0.0
    decision_step = 0
    prefix = f" {debug_context}" if debug_context else ""

    while accepted_steps < cfg.steps:
        prediction_refreshed = False
        if prediction is None or accepted_since_prediction >= prediction_horizon:
            prediction_progress = float(accepted_steps / max(cfg.steps - 1, 1))
            prediction_time = prediction_progress
            prediction_horizon = cfg.prediction_horizon_at(prediction_progress)
            if prediction_fn is None:
                prediction = predict_clean_spectrum_and_graphlets(
                    model,
                    current,
                    graphlet_basis=graphlet_basis,
                    time=prediction_time,
                    device=device,
                    graphlet_logit_epsilon=cfg.graphlet_logit_epsilon,
                    conditioning_graph=conditioning_graph,
                    source_spectrum=source_spectrum,
                    source_graphlet_probabilities=source_probabilities,
                    source_graphlet_logits=source_graphlet_logits,
                    current_graphlet_counts=current_counts,
                )
            else:
                prediction = prediction_fn(
                    model,
                    current,
                    graphlet_basis=graphlet_basis,
                    time=prediction_time,
                    device=device,
                    graphlet_logit_epsilon=cfg.graphlet_logit_epsilon,
                    conditioning_graph=conditioning_graph,
                    source_spectrum=source_spectrum,
                    source_graphlet_probabilities=source_probabilities,
                    source_graphlet_logits=source_graphlet_logits,
                    current_graphlet_counts=current_counts,
                )
            prediction_calls += 1
            prediction_block += 1
            accepted_since_prediction = 0
            prediction_refreshed = True
            spectral_weight, graphlet_weight = cfg.guidance_weights_at(prediction_progress)
            _debug_print(
                cfg,
                decision_step,
                prefix,
                (
                    f"prediction_refresh call={prediction_calls} block={prediction_block} "
                    f"accepted={accepted_steps}/{cfg.steps} t={prediction_time:.4f} "
                    f"horizon={prediction_horizon} weights=(spectral={spectral_weight:.3f},"
                    f"graphlet={graphlet_weight:.3f})"
                ),
            )
            _debug_print(
                cfg,
                decision_step,
                prefix,
                "lambda_clean_hat=" + _format_spectrum(
                    prediction.clean_spectrum, cfg.debug_spectrum_values
                ),
            )
            _debug_print(
                cfg,
                decision_step,
                prefix,
                "graphlet_clean_hat=" + _format_graphlet_probabilities(
                    prediction.clean_graphlet_probabilities,
                    graphlet_basis=graphlet_basis,
                    top_k=3,
                ),
            )
            invariant_trace = float(sum(initial_degrees))
            invariant_second = float(
                sum(value * value for value in initial_degrees) + invariant_trace
            )
            _debug_print(
                cfg,
                decision_step,
                prefix,
                (
                    f"clean_hat_moments trace={prediction.trace:.6f} "
                    f"trace_invariant={invariant_trace:.6f} "
                    f"second={prediction.second_moment:.6f} "
                    f"second_invariant={invariant_second:.6f} "
                    f"second_residual={prediction.second_moment - invariant_second:+.6f}"
                ),
            )

        assert prediction is not None
        progress = float(accepted_steps / max(cfg.steps - 1, 1))
        spectral_weight, graphlet_weight = cfg.guidance_weights_at(progress)
        current_spectrum = laplacian_eigenvalues(current)
        current_prob, current_mask = graphlet_simplex_from_counts(
            current_counts,
            num_nodes=current.number_of_nodes(),
            graphlet_basis=graphlet_basis,
        )
        current_logits = graphlet_simplex_to_clr(
            current_prob,
            graphlet_basis=graphlet_basis,
            epsilon=cfg.graphlet_logit_epsilon,
            coordinate_mask=current_mask,
        )
        spectral_mix = cfg.bridge.clean_mix_for_step(
            accepted_step=accepted_steps,
            total_steps=max(cfg.steps, 1),
        )
        graphlet_mix = cfg.graphlet_bridge.clean_mix_for_step(
            accepted_step=accepted_steps,
            total_steps=max(cfg.steps, 1),
        )
        candidates, candidate_graphs, proposal_diagnostics = propose_valid_topology_swaps(
            current,
            proposal_budget=cfg.proposal_budget,
            valid_candidate_budget=cfg.valid_candidate_budget,
            preserve_connectivity=cfg.preserve_connectivity,
            rng=generator,
            excluded_states=visited if cfg.reject_revisited_states else None,
        )
        if not candidates:
            trace.append(
                {
                    "step": decision_step,
                    "accepted_step": accepted_steps,
                    "accepted": False,
                    "reason": "explicit_stop_no_candidates",
                    "terminal_stop": True,
                    "prediction_refreshed": prediction_refreshed,
                    "prediction_calls": prediction_calls,
                    "prediction_block": prediction_block,
                    "prediction_horizon": prediction_horizon,
                    "prediction_progress": prediction_progress,
                    "prediction_time": prediction_time,
                    "inner_step": accepted_since_prediction,
                    "spectral_weight": spectral_weight,
                    "graphlet_weight": graphlet_weight,
                    "spectral_clean_mix": spectral_mix,
                    "graphlet_clean_mix": graphlet_mix,
                    **proposal_diagnostics,
                }
            )
            _debug_print(cfg, decision_step, prefix, "stop reason=no_valid_candidates")
            break

        # Candidate graph structure is fixed throughout plateau expansion.
        # Precompute each candidate spectrum and exact local-delta graphlet
        # state once, then only rescore the cached summaries as the continuous
        # bridge target moves toward the predicted clean endpoint.
        prepared_candidates = prepare_spectral_graphlet_candidate_states(
            current,
            candidates,
            candidate_graphs=candidate_graphs,
            current_graphlet_counts=current_counts,
            graphlet_basis=graphlet_basis,
            config=cfg,
        )

        expansion_count = 0
        rows: list[dict[str, Any]] = []
        selected: int | None = None
        stop_probability = 1.0
        probabilities: list[float] = []
        next_spectrum = current_spectrum.copy()
        next_graphlet_logits = current_logits.copy()
        while True:
            next_spectrum = cfg.bridge.target(
                current_spectrum,
                prediction.clean_spectrum,
                spectral_mix,
            )
            next_graphlet_logits = cfg.graphlet_bridge.target(
                current_logits,
                prediction.clean_graphlet_logits,
                graphlet_mix,
            )
            rows = score_prepared_spectral_graphlet_candidates(
                prepared_candidates,
                graphlet_basis=graphlet_basis,
                clean_spectrum=prediction.clean_spectrum,
                next_spectrum_target=next_spectrum,
                clean_graphlet_logits=prediction.clean_graphlet_logits,
                next_graphlet_logits_target=next_graphlet_logits,
                graphlet_coordinate_mask=current_mask,
                spectral_weight=spectral_weight,
                graphlet_weight=graphlet_weight,
                config=cfg,
            )
            selected, stop_probability, probabilities = _select_row(
                rows,
                config=cfg,
                rng=generator,
            )
            if selected is not None:
                break
            if (
                not cfg.expand_on_plateau
                or expansion_count >= cfg.max_plateau_expansions
                or (
                    spectral_mix >= cfg.bridge_max_clean_mix - 1.0e-12
                    and graphlet_mix >= cfg.graphlet_bridge_max_clean_mix - 1.0e-12
                )
            ):
                break
            spectral_mix = min(
                cfg.bridge_max_clean_mix,
                max(spectral_mix * cfg.plateau_expand_factor, spectral_mix + 1.0e-6),
            )
            graphlet_mix = min(
                cfg.graphlet_bridge_max_clean_mix,
                max(
                    graphlet_mix * cfg.graphlet_plateau_expand_factor,
                    graphlet_mix + 1.0e-6,
                ),
            )
            expansion_count += 1

        if cfg.debug_enabled and decision_step % cfg.debug_print_every == 0:
            current_graphlet_distance = graphlet_logit_distance(
                current_logits,
                next_graphlet_logits,
                graphlet_basis=graphlet_basis,
                coordinate_mask=current_mask,
                metric=cfg.graphlet_distance,
                size_weights=cfg.graphlet_size_weights,
            )
            current_spectral_distance = spectral_distance(
                current_spectrum,
                next_spectrum,
                metric=cfg.distance,
                scale=spectral_scale(current, mode=cfg.normalization),
                low_frequency_weight=cfg.low_frequency_weight,
                low_frequency_cutoff=cfg.low_frequency_cutoff,
            )
            target_prob = graphlet_clr_to_simplex(
                next_graphlet_logits,
                graphlet_basis=graphlet_basis,
                coordinate_mask=current_mask,
            )
            _debug_print(
                cfg,
                decision_step,
                prefix,
                (
                    f"step={decision_step} accepted={accepted_steps}/{cfg.steps} "
                    f"weights=(spectral={spectral_weight:.3f},graphlet={graphlet_weight:.3f}) "
                    f"mix=(spectral={spectral_mix:.3f},graphlet={graphlet_mix:.3f}) "
                    f"expansions={expansion_count} d_spec_next={current_spectral_distance:.6f} "
                    f"d_graphlet_next={current_graphlet_distance:.6f}"
                ),
            )
            _debug_print(cfg, decision_step, prefix, "lambda_current=" + _format_spectrum(current_spectrum, cfg.debug_spectrum_values))
            _debug_print(cfg, decision_step, prefix, "lambda_next_target=" + _format_spectrum(next_spectrum, cfg.debug_spectrum_values))
            _debug_print(
                cfg,
                decision_step,
                prefix,
                "graphlet_current=" + _format_graphlet_probabilities(
                    current_prob,
                    graphlet_basis=graphlet_basis,
                ),
            )
            _debug_print(
                cfg,
                decision_step,
                prefix,
                "graphlet_next_target=" + _format_graphlet_probabilities(
                    target_prob,
                    graphlet_basis=graphlet_basis,
                ),
            )
            _debug_print(
                cfg,
                decision_step,
                prefix,
                (
                    f"candidates proposals={proposal_diagnostics.get('num_proposals', 0)} "
                    f"valid={proposal_diagnostics.get('num_valid_candidates', 0)} "
                    f"pass_rate={proposal_diagnostics.get('candidate_pass_rate', 0.0):.4f} "
                    f"rejections={proposal_diagnostics.get('candidate_rejection_reasons', {})}"
                ),
            )
            if cfg.debug_top_candidates > 0:
                ranked = sorted(
                    enumerate(rows),
                    key=lambda item: float(item[1]["energy_improvement"]),
                    reverse=True,
                )[: cfg.debug_top_candidates]
                for rank, (row_index, row) in enumerate(ranked, start=1):
                    _debug_print(
                        cfg,
                        decision_step,
                        prefix,
                        (
                            f"candidate_rank={rank} index={row_index} "
                            f"gain={row['energy_improvement']:.6f} "
                            f"spec_gain={row['spectral_gain']:.6f} "
                            f"graphlet_gain={row['graphlet_gain']:.6f} "
                            f"d_spec={row['candidate_spectral_discrepancy']:.6f} "
                            f"d_graphlet={row['candidate_graphlet_discrepancy']:.6f}"
                        ),
                    )

        if selected is None:
            refresh = bool(cfg.refresh_on_prediction_plateau and accepted_since_prediction > 0)
            best_gain = max((float(row["energy_improvement"]) for row in rows), default=0.0)
            reason = "prediction_plateau_refresh" if refresh else "explicit_stop_below_joint_improvement_threshold"
            trace.append(
                {
                    "step": decision_step,
                    "accepted_step": accepted_steps,
                    "accepted": False,
                    "reason": reason,
                    "terminal_stop": not refresh,
                    "prediction_refreshed": prediction_refreshed,
                    "prediction_calls": prediction_calls,
                    "prediction_block": prediction_block,
                    "prediction_horizon": prediction_horizon,
                    "prediction_progress": prediction_progress,
                    "prediction_time": prediction_time,
                    "inner_step": accepted_since_prediction,
                    "stop_probability": stop_probability,
                    "selection_probabilities": probabilities,
                    "spectral_weight": spectral_weight,
                    "graphlet_weight": graphlet_weight,
                    "spectral_clean_mix": spectral_mix,
                    "graphlet_clean_mix": graphlet_mix,
                    "bridge_expansions": expansion_count,
                    "best_energy_improvement": best_gain,
                    **proposal_diagnostics,
                }
            )
            _debug_print(
                cfg,
                decision_step,
                prefix,
                f"stop_or_refresh reason={reason} best_gain={best_gain:.6f}",
            )
            decision_step += 1
            if refresh:
                prediction = None
                continue
            break

        chosen = rows[selected]
        candidate = chosen["candidate_graph"]
        if [int(candidate.degree(node)) for node in sorted(candidate.nodes())] != initial_degrees:
            raise AssertionError("A spectral+graphlet rewiring action changed indexed degrees.")
        if cfg.preserve_connectivity and candidate.number_of_nodes() > 1 and not nx.is_connected(candidate):
            raise AssertionError("A spectral+graphlet rewiring action broke connectivity.")
        current = candidate
        current_counts = chosen["candidate_graphlet_counts"]
        visited.add(topology_state_key(current))
        accepted_steps += 1
        accepted_since_prediction += 1
        _debug_print(
            cfg,
            decision_step,
            prefix,
            (
                f"ACCEPT accepted_step={accepted_steps}/{cfg.steps} "
                f"gain={chosen['energy_improvement']:.6f} "
                f"spec_gain={chosen['spectral_gain']:.6f} "
                f"graphlet_gain={chosen['graphlet_gain']:.6f} "
                f"projection_residual={chosen['projection_residual']:.6f}"
            ),
        )
        _debug_print(
            cfg,
            decision_step,
            prefix,
            "lambda_after_accept=" + _format_spectrum(
                np.asarray(chosen["candidate_spectrum"], dtype=np.float64),
                cfg.debug_spectrum_values,
            ),
        )
        _debug_print(
            cfg,
            decision_step,
            prefix,
            "graphlet_after_accept=" + _format_graphlet_probabilities(
                np.asarray(chosen["candidate_graphlet_probabilities"], dtype=np.float64),
                graphlet_basis=graphlet_basis,
            ),
        )
        trace.append(
            {
                "step": decision_step,
                "accepted_step": accepted_steps,
                "accepted": True,
                "reason": "spectral_graphlet_denoising_swap",
                "terminal_stop": False,
                "action": chosen["action"],
                "prediction_refreshed": prediction_refreshed,
                "prediction_calls": prediction_calls,
                "prediction_block": prediction_block,
                "prediction_horizon": prediction_horizon,
                "prediction_progress": prediction_progress,
                "prediction_time": prediction_time,
                "inner_step": accepted_since_prediction,
                "stop_probability": stop_probability,
                "selected_action_probability": probabilities[selected],
                "spectral_weight": float(spectral_weight),
                "graphlet_weight": float(graphlet_weight),
                "spectral_clean_mix": float(spectral_mix),
                "graphlet_clean_mix": float(graphlet_mix),
                "clean_mix": float(spectral_mix),
                "bridge_expansions": int(expansion_count),
                "current_spectral_discrepancy": float(chosen["current_spectral_discrepancy"]),
                "candidate_spectral_discrepancy": float(chosen["candidate_spectral_discrepancy"]),
                "current_clean_spectral_discrepancy": float(chosen["current_clean_spectral_discrepancy"]),
                "candidate_clean_spectral_discrepancy": float(chosen["candidate_clean_spectral_discrepancy"]),
                "current_graphlet_discrepancy": float(chosen["current_graphlet_discrepancy"]),
                "candidate_graphlet_discrepancy": float(chosen["candidate_graphlet_discrepancy"]),
                "current_clean_graphlet_discrepancy": float(chosen["current_clean_graphlet_discrepancy"]),
                "candidate_clean_graphlet_discrepancy": float(chosen["candidate_clean_graphlet_discrepancy"]),
                "spectral_gain": float(chosen["spectral_gain"]),
                "clean_spectral_gain": float(chosen["clean_spectral_gain"]),
                "graphlet_gain": float(chosen["graphlet_gain"]),
                "clean_graphlet_gain": float(chosen["clean_graphlet_gain"]),
                "spectral_projection_residual": float(chosen["spectral_projection_residual"]),
                "graphlet_projection_residual": float(chosen["graphlet_projection_residual"]),
                "projection_residual": float(chosen["projection_residual"]),
                "energy_improvement": float(chosen["energy_improvement"]),
                "relative_energy_improvement": float(chosen["relative_energy_improvement"]),
                **proposal_diagnostics,
            }
        )
        decision_step += 1

    if return_trace:
        return current, trace
    return current

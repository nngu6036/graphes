from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import networkx as nx
import numpy as np
import torch

from grapher.rewiring_mlp.core.rewiring import Action
from grapher.rewiring_mlp.generic.data import normalize_topology_graph
from grapher.rewiring_mlp.generic.rewiring import (
    propose_valid_topology_swaps,
    topology_state_key,
)
from grapher.rewiring_mlp.generic.spectral import (
    SpectralBridgeSchedule,
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
    TopologySpectralTransformerPredictor,
)


@dataclass(frozen=True)
class SpectralPrediction:
    clean_spectrum: np.ndarray
    current_spectrum: np.ndarray
    trace: float
    second_moment: float


@dataclass(frozen=True)
class SpectralRefinerConfig:
    steps: int = 24
    # Denominator for the `time` feature.  Must match the horizon used to label
    # training states (`topology_trajectory.steps`), NOT the generation step
    # budget, otherwise the predictor sees a rescaled time input.  ``None``
    # falls back to ``steps`` for backward compatibility with old checkpoints.
    time_horizon: int | None = None
    proposal_budget: int = 128
    valid_candidate_budget: int = 32
    preserve_connectivity: bool = True
    selection: str = "greedy"
    temperature: float = 0.1
    accept_only_improving: bool = True
    min_improvement: float = 1.0e-8
    min_relative_improvement: float = 0.0
    relative_improvement_epsilon: float = 1.0e-12
    reject_revisited_states: bool = True

    distance: str = "rmse"
    normalization: str = "mean_degree"
    low_frequency_weight: float = 1.0
    low_frequency_cutoff: int = 0

    bridge_schedule: str = "linear"
    bridge_min_clean_mix: float = 0.15
    bridge_max_clean_mix: float = 1.0
    bridge_power: float = 2.0
    expand_on_plateau: bool = True
    plateau_expand_factor: float = 2.0
    max_plateau_expansions: int = 4
    # F1: greedy one-swap descent stalls at a local minimum after a few moves.
    # When every candidate fails the improvement threshold even after clean-mix
    # expansion, optionally take the least-harmful valid swap instead of
    # stopping.  `sideways_tolerance` is in the same normalized units as the
    # spectral distance.
    allow_sideways_moves: bool = False
    max_consecutive_sideways: int = 0
    sideways_tolerance: float = 0.0
    # Sideways moves can end on a worse state than the best one visited, so the
    # best state is tracked and returned by default.
    return_best_state: bool = True

    refresh_prediction_every: int = 1
    prediction_horizon_mode: str = "fixed"
    prediction_horizon_initial_k: int = 1
    prediction_horizon_final_k: int = 1
    prediction_horizon_schedule: str = "constant"
    refresh_on_prediction_plateau: bool = True

    debug_enabled: bool = False
    debug_print_every: int = 1
    debug_top_candidates: int = 3
    debug_spectrum_values: int = 12
    debug_store_spectra: bool = False

    @property
    def resolved_time_horizon(self) -> int:
        """Denominator for the `time` feature, defaulting to the step budget."""

        if self.time_horizon is None:
            return max(int(self.steps), 1)
        return max(int(self.time_horizon), 1)

    @property
    def bridge(self) -> SpectralBridgeSchedule:
        return SpectralBridgeSchedule(
            schedule=self.bridge_schedule,
            min_clean_mix=self.bridge_min_clean_mix,
            max_clean_mix=self.bridge_max_clean_mix,
            power=self.bridge_power,
        )

    def prediction_horizon_at(self, progress: float) -> int:
        if self.prediction_horizon_mode == "fixed":
            return int(self.refresh_prediction_every)
        clipped = float(np.clip(progress, 0.0, 1.0))
        start = float(self.prediction_horizon_initial_k)
        end = float(self.prediction_horizon_final_k)
        if self.prediction_horizon_schedule == "linear":
            value = start + (end - start) * clipped
        elif self.prediction_horizon_schedule == "cosine":
            cooling = 0.5 * (1.0 + np.cos(np.pi * clipped))
            value = end + (start - end) * cooling
        elif self.prediction_horizon_schedule == "exponential":
            value = start * ((end / start) ** clipped)
        else:
            raise ValueError(
                f"Unknown prediction-horizon schedule: {self.prediction_horizon_schedule!r}."
            )
        return max(1, int(np.floor(value + 0.5)))

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any] | None = None,
    ) -> "SpectralRefinerConfig":
        values = dict(data or {})
        mode = str(values.get("mode", "spectral")).lower()
        if mode not in {"spectral", "spectral_energy", "spectral_guidance"}:
            raise ValueError(
                "SpectralRefinerConfig requires topology_refiner.mode: spectral."
            )
        legacy_budget = int(values.get("candidate_budget", 32))
        valid_budget = int(values.get("valid_candidate_budget", legacy_budget))
        proposal_budget = int(
            values.get(
                "proposal_budget",
                valid_budget if valid_budget < 0 else max(valid_budget, 1) * 4,
            )
        )

        guidance = dict(values.get("spectral_guidance", {}) or {})
        bridge = SpectralBridgeSchedule.from_dict(guidance)
        legacy_refresh = int(values.get("refresh_prediction_every", 1))
        horizon_data = values.get("prediction_horizon")
        if horizon_data is None:
            horizon_mode = "fixed"
            horizon_initial = legacy_refresh
            horizon_final = legacy_refresh
            horizon_schedule = "constant"
            refresh_on_prediction_plateau = bool(
                values.get("refresh_on_plateau", True)
            )
        else:
            if not isinstance(horizon_data, dict):
                raise ValueError(
                    "topology_refiner.prediction_horizon must be a mapping."
                )
            horizon = dict(horizon_data)
            horizon_mode = str(horizon.get("mode", "annealed")).lower()
            if horizon_mode in {"adaptive", "anneal"}:
                horizon_mode = "annealed"
            if horizon_mode == "fixed":
                fixed_k = int(
                    horizon.get("k", horizon.get("initial_k", legacy_refresh))
                )
                horizon_initial = fixed_k
                horizon_final = fixed_k
                horizon_schedule = "constant"
                legacy_refresh = fixed_k
            else:
                horizon_initial = int(horizon.get("initial_k", legacy_refresh))
                horizon_final = int(horizon.get("final_k", 1))
                horizon_schedule = str(horizon.get("schedule", "exponential")).lower()
                if horizon_schedule in {"geometric", "exp"}:
                    horizon_schedule = "exponential"
                legacy_refresh = horizon_initial
            refresh_on_prediction_plateau = bool(
                horizon.get("refresh_on_plateau", True)
            )

        debug = dict(values.get("debug", {}) or {})
        raw_time_horizon = values.get("time_horizon")
        config = cls(
            steps=int(values.get("steps", 24)),
            time_horizon=(
                None if raw_time_horizon is None else int(raw_time_horizon)
            ),
            proposal_budget=proposal_budget,
            valid_candidate_budget=valid_budget,
            preserve_connectivity=bool(values.get("preserve_connectivity", True)),
            selection=str(values.get("selection", "greedy")).lower(),
            temperature=float(values.get("temperature", 0.1)),
            accept_only_improving=bool(values.get("accept_only_improving", True)),
            min_improvement=float(values.get("min_improvement", 1.0e-8)),
            min_relative_improvement=float(
                values.get("min_relative_improvement", 0.0)
            ),
            relative_improvement_epsilon=float(
                values.get("relative_improvement_epsilon", 1.0e-12)
            ),
            reject_revisited_states=bool(values.get("reject_revisited_states", True)),
            distance=str(guidance.get("distance", "rmse")).lower(),
            normalization=str(guidance.get("normalization", "mean_degree")).lower(),
            low_frequency_weight=float(
                guidance.get("low_frequency_weight", 1.0)
            ),
            low_frequency_cutoff=int(guidance.get("low_frequency_cutoff", 0)),
            bridge_schedule=bridge.schedule,
            bridge_min_clean_mix=bridge.min_clean_mix,
            bridge_max_clean_mix=bridge.max_clean_mix,
            bridge_power=bridge.power,
            expand_on_plateau=bool(guidance.get("expand_on_plateau", True)),
            plateau_expand_factor=float(
                guidance.get("plateau_expand_factor", 2.0)
            ),
            max_plateau_expansions=int(
                guidance.get("max_plateau_expansions", 4)
            ),
            allow_sideways_moves=bool(
                guidance.get("allow_sideways_moves", False)
            ),
            max_consecutive_sideways=int(
                guidance.get("max_consecutive_sideways", 0)
            ),
            sideways_tolerance=float(guidance.get("sideways_tolerance", 0.0)),
            return_best_state=bool(values.get("return_best_state", True)),
            refresh_prediction_every=legacy_refresh,
            prediction_horizon_mode=horizon_mode,
            prediction_horizon_initial_k=horizon_initial,
            prediction_horizon_final_k=horizon_final,
            prediction_horizon_schedule=horizon_schedule,
            refresh_on_prediction_plateau=refresh_on_prediction_plateau,
            debug_enabled=bool(debug.get("enabled", False)),
            debug_print_every=max(int(debug.get("print_every", 1)), 1),
            debug_top_candidates=max(int(debug.get("top_candidates", 3)), 0),
            debug_spectrum_values=max(int(debug.get("spectrum_values", 12)), 1),
            debug_store_spectra=bool(debug.get("store_spectra", False)),
        )
        if config.steps < 0:
            raise ValueError("topology_refiner.steps must be non-negative.")
        if config.time_horizon is not None and config.time_horizon <= 0:
            raise ValueError("topology_refiner.time_horizon must be positive.")
        if config.proposal_budget == 0 or config.valid_candidate_budget == 0:
            raise ValueError("Topology proposal budgets must be non-zero.")
        if config.selection not in {"greedy", "argmax", "softmax", "sample"}:
            raise ValueError("Spectral topology selection must be greedy or softmax.")
        if config.temperature <= 0.0:
            raise ValueError("topology_refiner.temperature must be positive.")
        if not config.preserve_connectivity:
            raise ValueError("Spectral GraphER requires connectivity-preserving swaps.")
        if not config.accept_only_improving:
            raise ValueError(
                "Spectral GraphER currently accepts only positive-improvement swaps."
            )
        if not np.isfinite(config.min_improvement) or config.min_improvement < 0.0:
            raise ValueError("min_improvement must be finite and nonnegative.")
        if (
            not np.isfinite(config.min_relative_improvement)
            or config.min_relative_improvement < 0.0
        ):
            raise ValueError(
                "min_relative_improvement must be finite and nonnegative."
            )
        if (
            not np.isfinite(config.relative_improvement_epsilon)
            or config.relative_improvement_epsilon <= 0.0
        ):
            raise ValueError(
                "relative_improvement_epsilon must be finite and positive."
            )
        if config.distance not in {"rmse", "mse", "mae", "l1", "l2"}:
            raise ValueError("spectral_guidance.distance must be rmse, mse, or mae.")
        if config.low_frequency_weight <= 0.0:
            raise ValueError("low_frequency_weight must be positive.")
        if config.low_frequency_cutoff < 0:
            raise ValueError("low_frequency_cutoff must be nonnegative.")
        if config.plateau_expand_factor <= 1.0 and config.expand_on_plateau:
            raise ValueError(
                "plateau_expand_factor must exceed 1 when expansion is enabled."
            )
        if config.max_plateau_expansions < 0:
            raise ValueError("max_plateau_expansions must be nonnegative.")
        if config.max_consecutive_sideways < 0:
            raise ValueError(
                "spectral_guidance.max_consecutive_sideways must be nonnegative."
            )
        if (
            not np.isfinite(config.sideways_tolerance)
            or config.sideways_tolerance < 0.0
        ):
            raise ValueError(
                "spectral_guidance.sideways_tolerance must be finite and "
                "nonnegative."
            )
        if config.allow_sideways_moves and config.max_consecutive_sideways <= 0:
            raise ValueError(
                "spectral_guidance.allow_sideways_moves requires "
                "max_consecutive_sideways > 0."
            )
        if config.refresh_prediction_every <= 0:
            raise ValueError("refresh_prediction_every must be positive.")
        if config.prediction_horizon_mode not in {"fixed", "annealed"}:
            raise ValueError("prediction_horizon.mode must be fixed or annealed.")
        if config.prediction_horizon_initial_k <= 0 or config.prediction_horizon_final_k <= 0:
            raise ValueError("Prediction horizons must be positive.")
        if (
            config.prediction_horizon_mode == "annealed"
            and config.prediction_horizon_initial_k < config.prediction_horizon_final_k
        ):
            raise ValueError("Annealed horizons require initial_k >= final_k.")
        if config.prediction_horizon_mode == "annealed" and (
            config.prediction_horizon_schedule not in {"linear", "cosine", "exponential"}
        ):
            raise ValueError(
                "prediction_horizon.schedule must be linear, cosine, or exponential."
            )
        return config


@torch.no_grad()
def predict_clean_spectrum(
    model: TopologySpectralTransformerPredictor,
    graph: nx.Graph,
    *,
    time: float,
    device: torch.device | str,
    conditioning_graph: nx.Graph | None = None,
    source_spectrum: np.ndarray | None = None,
    **_: Any,
) -> SpectralPrediction:
    """Predict all clean eigenvalues jointly for one variable-size graph."""

    model.eval()
    n = graph.number_of_nodes()
    context_graph = graph if conditioning_graph is None else conditioning_graph
    current_spectrum = laplacian_eigenvalues(graph)
    source_values = (
        laplacian_eigenvalues(context_graph)
        if source_spectrum is None
        else np.asarray(source_spectrum, dtype=np.float64).reshape(-1)
    )
    batch = collate_spectral_examples(
        [
            TopologySpectralExample(
                current_graph=context_graph,
                time=float(time),
                current_spectrum=current_spectrum.astype(np.float32),
                source_spectrum=source_values.astype(np.float32),
                clean_spectrum_target=np.zeros(n, dtype=np.float32),
            )
        ]
    ).to(device)
    outputs = model(batch)
    predicted = outputs["clean_spectrum"][0, :n].detach().cpu().numpy().astype(
        np.float64
    )
    current = current_spectrum.astype(np.float64)
    trace, second = spectrum_moments(predicted)
    return SpectralPrediction(
        clean_spectrum=predicted,
        current_spectrum=current,
        trace=trace,
        second_moment=second,
    )


def score_spectral_candidates(
    graph: nx.Graph,
    candidates: Sequence[Action],
    *,
    clean_spectrum: np.ndarray,
    next_spectrum_target: np.ndarray,
    config: SpectralRefinerConfig,
    candidate_graphs: dict[Action, nx.Graph],
) -> list[dict[str, Any]]:
    current_spectrum = laplacian_eigenvalues(graph)
    scale = spectral_scale(graph, mode=config.normalization)
    current_local = spectral_distance(
        current_spectrum,
        next_spectrum_target,
        metric=config.distance,
        scale=scale,
        low_frequency_weight=config.low_frequency_weight,
        low_frequency_cutoff=config.low_frequency_cutoff,
    )
    current_clean = spectral_distance(
        current_spectrum,
        clean_spectrum,
        metric=config.distance,
        scale=scale,
        low_frequency_weight=config.low_frequency_weight,
        low_frequency_cutoff=config.low_frequency_cutoff,
    )
    rows: list[dict[str, Any]] = []
    for action in candidates:
        candidate = candidate_graphs[action]
        spectrum = laplacian_eigenvalues(candidate)
        candidate_local = spectral_distance(
            spectrum,
            next_spectrum_target,
            metric=config.distance,
            scale=scale,
            low_frequency_weight=config.low_frequency_weight,
            low_frequency_cutoff=config.low_frequency_cutoff,
        )
        candidate_clean = spectral_distance(
            spectrum,
            clean_spectrum,
            metric=config.distance,
            scale=scale,
            low_frequency_weight=config.low_frequency_weight,
            low_frequency_cutoff=config.low_frequency_cutoff,
        )
        local_gain = float(current_local - candidate_local)
        clean_gain = float(current_clean - candidate_clean)
        relative = float(
            local_gain
            / max(abs(current_local), float(config.relative_improvement_epsilon))
        )
        rows.append(
            {
                "action": action,
                "candidate_graph": candidate,
                "candidate_spectrum": spectrum,
                "current_spectral_discrepancy": float(current_local),
                "candidate_spectral_discrepancy": float(candidate_local),
                "current_clean_spectral_discrepancy": float(current_clean),
                "candidate_clean_spectral_discrepancy": float(candidate_clean),
                "spectral_gain": local_gain,
                "clean_spectral_gain": clean_gain,
                "projection_residual": float(candidate_local),
                "energy_improvement": local_gain,
                "relative_energy_improvement": relative,
            }
        )
    return rows


def _select_row(
    rows: Sequence[dict[str, Any]],
    *,
    config: SpectralRefinerConfig,
    rng: np.random.Generator,
) -> tuple[int | None, float, list[float]]:
    improvements = np.asarray(
        [float(row["energy_improvement"]) for row in rows],
        dtype=np.float64,
    )
    relative = np.asarray(
        [float(row["relative_energy_improvement"]) for row in rows],
        dtype=np.float64,
    )
    scores = np.concatenate([improvements, np.asarray([0.0])])
    if config.accept_only_improving:
        eligible = improvements > float(config.min_improvement)
        eligible &= relative > float(config.min_relative_improvement)
        scores[:-1][~eligible] = -np.inf
        if np.any(np.isfinite(scores[:-1])):
            scores[-1] = -np.inf
    finite = np.isfinite(scores)
    shifted = scores.copy()
    shifted[finite] -= float(np.max(shifted[finite]))
    probabilities = np.zeros_like(scores)
    probabilities[finite] = np.exp(shifted[finite] / float(config.temperature))
    probabilities /= float(probabilities.sum())
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


def _select_sideways_row(
    rows: Sequence[dict[str, Any]],
    *,
    config: SpectralRefinerConfig,
) -> int | None:
    """Return the least-harmful candidate when no swap clears the threshold.

    Used only as a plateau escape: the returned move does not improve the
    spectral objective, but it lets the constrained walk leave a one-swap local
    minimum instead of terminating there.  Candidates are already tabu-filtered
    upstream, so any returned index is a state that has not been visited.
    """

    if not rows:
        return None
    tolerance = -float(config.sideways_tolerance)
    eligible = [
        index
        for index, row in enumerate(rows)
        if float(row["energy_improvement"]) >= tolerance
    ]
    if not eligible:
        return None
    return max(eligible, key=lambda index: float(rows[index]["energy_improvement"]))


def _format_spectrum(values: np.ndarray, limit: int) -> str:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    limit = max(int(limit), 1)
    if array.size <= limit:
        body = ", ".join(f"{value:.4f}" for value in array)
    else:
        left_count = max(limit // 2, 1)
        right_count = max(limit - left_count, 1)
        body = ", ".join(f"{value:.4f}" for value in array[:left_count])
        body += ", ..., "
        body += ", ".join(f"{value:.4f}" for value in array[-right_count:])
    return f"[{body}]"


def _format_action(action: Action) -> str:
    removed, added = action
    return f"remove={list(removed)} add={list(added)}"


def _debug_print(
    config: SpectralRefinerConfig,
    decision_step: int,
    prefix: str,
    message: str,
) -> None:
    if not config.debug_enabled:
        return
    if decision_step % config.debug_print_every != 0:
        return
    print(f"[GraphER/Spectral]{prefix} {message}", flush=True)


def refine_graph_with_spectral_predictions(
    graph: nx.Graph,
    *,
    model: TopologySpectralTransformerPredictor,
    refiner_config: SpectralRefinerConfig | dict[str, Any] | None = None,
    device: torch.device | str = "cpu",
    rng: np.random.Generator | None = None,
    return_trace: bool = False,
    prediction_fn: Any | None = None,
    debug_context: str | None = None,
) -> nx.Graph | tuple[nx.Graph, list[dict[str, Any]]]:
    """Project clean-spectrum denoising guidance through valid edge swaps."""

    cfg = (
        refiner_config
        if isinstance(refiner_config, SpectralRefinerConfig)
        else SpectralRefinerConfig.from_dict(refiner_config)
    )
    generator = rng if rng is not None else np.random.default_rng(0)
    predictor = prediction_fn or predict_clean_spectrum
    current = normalize_topology_graph(graph)
    conditioning_graph = current.copy()
    source_spectrum = laplacian_eigenvalues(conditioning_graph)
    if current.number_of_nodes() > 1 and not nx.is_connected(current):
        raise ValueError("Spectral refinement requires a connected source graph.")
    initial_degrees = [int(current.degree(node)) for node in sorted(current.nodes())]
    visited = {topology_state_key(current)}
    trace: list[dict[str, Any]] = []
    prediction: SpectralPrediction | None = None
    accepted_steps = 0
    accepted_since_prediction = 0
    # F1: bounded plateau escape.  `best_graph` is tracked separately because a
    # sideways move may terminate on a worse state than the best one visited.
    consecutive_sideways = 0
    sideways_accepted = 0
    best_graph = current.copy()
    best_clean_distance = float("inf")
    decision_step = 0
    prediction_calls = 0
    prediction_block = -1
    prediction_horizon = 1
    prediction_progress = 0.0
    prediction_time = 0.0
    prefix = f"[{debug_context}]" if debug_context else ""

    while accepted_steps < cfg.steps:
        prediction_refreshed = False
        if prediction is None or accepted_since_prediction >= prediction_horizon:
            prediction_progress = float(accepted_steps / max(cfg.steps - 1, 1))
            # V2 predictors are trained on normalized diffusion progress: 0 is
            # the HH/base source endpoint and 1 is the clean endpoint.  This is
            # independent of the number of rewiring projection steps.
            prediction_time = prediction_progress
            prediction_horizon = cfg.prediction_horizon_at(prediction_progress)
            if prediction_fn is None:
                prediction = predictor(
                    model,
                    current,
                    time=prediction_time,
                    device=device,
                    conditioning_graph=conditioning_graph,
                    source_spectrum=source_spectrum,
                )
            else:
                # Preserve the lightweight custom-predictor testing/debug API.
                prediction = predictor(
                    model,
                    current,
                    time=prediction_time,
                    device=device,
                )
            if prediction.clean_spectrum.size != current.number_of_nodes():
                raise ValueError(
                    "Spectral predictor returned the wrong number of eigenvalues."
                )
            prediction_calls += 1
            prediction_block += 1
            accepted_since_prediction = 0
            prediction_refreshed = True
            _debug_print(
                cfg,
                decision_step,
                prefix,
                (
                    f"prediction_refresh call={prediction_calls} block={prediction_block} "
                    f"accepted={accepted_steps}/{cfg.steps} t={prediction_time:.4f} "
                    f"horizon={prediction_horizon}"
                ),
            )
            _debug_print(
                cfg,
                decision_step,
                prefix,
                "lambda_clean_hat="
                + _format_spectrum(
                    prediction.clean_spectrum, cfg.debug_spectrum_values
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
        current_spectrum = laplacian_eigenvalues(current)
        clean_spectrum = prediction.clean_spectrum
        base_mix = cfg.bridge.clean_mix_for_step(
            accepted_step=accepted_steps,
            total_steps=max(cfg.steps, 1),
        )
        clean_mix = base_mix

        candidates, candidate_graphs, proposal_diagnostics = (
            propose_valid_topology_swaps(
                current,
                proposal_budget=cfg.proposal_budget,
                valid_candidate_budget=cfg.valid_candidate_budget,
                preserve_connectivity=cfg.preserve_connectivity,
                rng=generator,
                excluded_states=visited if cfg.reject_revisited_states else None,
            )
        )
        if not candidates:
            _debug_print(
                cfg,
                decision_step,
                prefix,
                (
                    "stop reason=no_valid_candidates "
                    f"proposals={proposal_diagnostics.get('num_proposals', 0)} "
                    f"rejections={proposal_diagnostics.get('candidate_rejection_reasons', {})}"
                ),
            )
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
                    "clean_mix": clean_mix,
                    **proposal_diagnostics,
                }
            )
            break

        rows: list[dict[str, Any]] = []
        next_target = current_spectrum.copy()
        stop_probability = 1.0
        probabilities: list[float] = []
        selected: int | None = None
        expansion_count = 0
        while True:
            next_target = cfg.bridge.target(
                current_spectrum,
                clean_spectrum,
                clean_mix,
            )
            rows = score_spectral_candidates(
                current,
                candidates,
                clean_spectrum=clean_spectrum,
                next_spectrum_target=next_target,
                config=cfg,
                candidate_graphs=candidate_graphs,
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
                or clean_mix >= cfg.bridge_max_clean_mix - 1.0e-12
                or expansion_count >= cfg.max_plateau_expansions
            ):
                break
            clean_mix = min(
                cfg.bridge_max_clean_mix,
                max(clean_mix * cfg.plateau_expand_factor, clean_mix + 1.0e-6),
            )
            expansion_count += 1

        should_debug = cfg.debug_enabled and decision_step % cfg.debug_print_every == 0
        if should_debug:
            scale = spectral_scale(current, mode=cfg.normalization)
            current_clean_distance = spectral_distance(
                current_spectrum,
                clean_spectrum,
                metric=cfg.distance,
                scale=scale,
                low_frequency_weight=cfg.low_frequency_weight,
                low_frequency_cutoff=cfg.low_frequency_cutoff,
            )
            current_next_distance = spectral_distance(
                current_spectrum,
                next_target,
                metric=cfg.distance,
                scale=scale,
                low_frequency_weight=cfg.low_frequency_weight,
                low_frequency_cutoff=cfg.low_frequency_cutoff,
            )
            _debug_print(
                cfg,
                decision_step,
                prefix,
                (
                    f"step={decision_step} accepted={accepted_steps}/{cfg.steps} "
                    f"mix={clean_mix:.4f} base_mix={base_mix:.4f} "
                    f"bridge_expansions={expansion_count} "
                    f"d(current,clean)={current_clean_distance:.6f} "
                    f"d(current,next)={current_next_distance:.6f}"
                ),
            )
            _debug_print(
                cfg,
                decision_step,
                prefix,
                "lambda_current="
                + _format_spectrum(current_spectrum, cfg.debug_spectrum_values),
            )
            _debug_print(
                cfg,
                decision_step,
                prefix,
                "lambda_next_target="
                + _format_spectrum(next_target, cfg.debug_spectrum_values),
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
            if cfg.debug_top_candidates > 0 and rows:
                ranked = sorted(
                    enumerate(rows),
                    key=lambda pair: float(pair[1]["energy_improvement"]),
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
                            f"rel_gain={row['relative_energy_improvement']:.6f} "
                            f"d_next={row['candidate_spectral_discrepancy']:.6f} "
                            f"d_clean={row['candidate_clean_spectral_discrepancy']:.6f} "
                            + _format_action(row["action"])
                        ),
                    )

        spectra_payload: dict[str, Any] = {}
        if cfg.debug_store_spectra:
            spectra_payload = {
                "current_spectrum": current_spectrum.tolist(),
                "predicted_clean_spectrum": clean_spectrum.tolist(),
                "next_spectrum_target": next_target.tolist(),
            }

        move_kind = "improving"
        if selected is None:
            # F1: before treating this as terminal, optionally take the
            # least-harmful valid swap so the walk can cross a plateau.
            if (
                cfg.allow_sideways_moves
                and consecutive_sideways < cfg.max_consecutive_sideways
            ):
                sideways_index = _select_sideways_row(rows, config=cfg)
                if sideways_index is not None:
                    selected = sideways_index
                    move_kind = "sideways"
                    _debug_print(
                        cfg,
                        decision_step,
                        prefix,
                        (
                            f"SIDEWAYS consecutive={consecutive_sideways + 1}/"
                            f"{cfg.max_consecutive_sideways} "
                            f"gain={rows[selected]['energy_improvement']:.6f} "
                            f"tolerance={cfg.sideways_tolerance:.6g}"
                        ),
                    )

        if selected is None:
            refresh_after_plateau = bool(
                cfg.refresh_on_prediction_plateau
                and accepted_since_prediction > 0
            )
            reason = (
                "prediction_plateau_refresh"
                if refresh_after_plateau
                else "explicit_stop_below_spectral_improvement_threshold"
            )
            best_gain = max(
                (float(row["energy_improvement"]) for row in rows),
                default=0.0,
            )
            _debug_print(
                cfg,
                decision_step,
                prefix,
                (
                    f"stop_or_refresh reason={reason} best_gain={best_gain:.6f} "
                    f"threshold={cfg.min_improvement:.6g} "
                    f"relative_threshold={cfg.min_relative_improvement:.6g}"
                ),
            )
            trace.append(
                {
                    "step": decision_step,
                    "accepted_step": accepted_steps,
                    "accepted": False,
                    "reason": reason,
                    "terminal_stop": not refresh_after_plateau,
                    "prediction_refreshed": prediction_refreshed,
                    "prediction_calls": prediction_calls,
                    "prediction_block": prediction_block,
                    "prediction_horizon": prediction_horizon,
                    "prediction_progress": prediction_progress,
                    "prediction_time": prediction_time,
                    "inner_step": accepted_since_prediction,
                    "stop_probability": stop_probability,
                    "selection_probabilities": probabilities,
                    "clean_mix": clean_mix,
                    "base_clean_mix": base_mix,
                    "bridge_expansions": expansion_count,
                    "current_spectral_discrepancy": (
                        float(rows[0]["current_spectral_discrepancy"])
                        if rows
                        else None
                    ),
                    "current_clean_spectral_discrepancy": (
                        float(rows[0]["current_clean_spectral_discrepancy"])
                        if rows
                        else None
                    ),
                    "best_energy_improvement": best_gain,
                    "min_improvement": float(cfg.min_improvement),
                    "min_relative_improvement": float(
                        cfg.min_relative_improvement
                    ),
                    **spectra_payload,
                    **proposal_diagnostics,
                }
            )
            decision_step += 1
            if refresh_after_plateau:
                prediction = None
                continue
            break

        chosen = rows[selected]
        candidate = chosen["candidate_graph"]
        if [int(candidate.degree(node)) for node in sorted(candidate.nodes())] != (
            initial_degrees
        ):
            raise AssertionError("A spectral rewiring action changed indexed degrees.")
        if (
            cfg.preserve_connectivity
            and candidate.number_of_nodes() > 1
            and not nx.is_connected(candidate)
        ):
            raise AssertionError("A spectral rewiring action broke connectivity.")

        current = candidate
        visited.add(topology_state_key(current))
        accepted_steps += 1
        accepted_since_prediction += 1
        if move_kind == "sideways":
            consecutive_sideways += 1
            sideways_accepted += 1
        else:
            consecutive_sideways = 0
        # Track the best state by distance to the predicted clean spectrum, so
        # a plateau escape that ends worse cannot degrade the returned graph.
        candidate_clean = float(chosen["candidate_clean_spectral_discrepancy"])
        if candidate_clean < best_clean_distance:
            best_clean_distance = candidate_clean
            best_graph = current.copy()
        _debug_print(
            cfg,
            decision_step,
            prefix,
            (
                f"ACCEPT accepted_step={accepted_steps}/{cfg.steps} "
                f"gain={chosen['energy_improvement']:.6f} "
                f"clean_gain={chosen['clean_spectral_gain']:.6f} "
                f"projection_residual={chosen['projection_residual']:.6f} "
                + _format_action(chosen["action"])
            ),
        )
        _debug_print(
            cfg,
            decision_step,
            prefix,
            "lambda_after_accept="
            + _format_spectrum(
                np.asarray(chosen["candidate_spectrum"], dtype=np.float64),
                cfg.debug_spectrum_values,
            ),
        )
        trace.append(
            {
                "step": decision_step,
                "accepted_step": accepted_steps,
                "accepted": True,
                "reason": "spectral_denoising_swap",
                "move_kind": move_kind,
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
                "clean_mix": clean_mix,
                "base_clean_mix": base_mix,
                "bridge_expansions": expansion_count,
                "current_spectral_discrepancy": float(
                    chosen["current_spectral_discrepancy"]
                ),
                "candidate_spectral_discrepancy": float(
                    chosen["candidate_spectral_discrepancy"]
                ),
                "current_clean_spectral_discrepancy": float(
                    chosen["current_clean_spectral_discrepancy"]
                ),
                "candidate_clean_spectral_discrepancy": float(
                    chosen["candidate_clean_spectral_discrepancy"]
                ),
                "spectral_gain": float(chosen["spectral_gain"]),
                "clean_spectral_gain": float(chosen["clean_spectral_gain"]),
                "projection_residual": float(chosen["projection_residual"]),
                "energy_improvement": float(chosen["energy_improvement"]),
                "relative_energy_improvement": float(
                    chosen["relative_energy_improvement"]
                ),
                "min_improvement": float(cfg.min_improvement),
                "min_relative_improvement": float(cfg.min_relative_improvement),
                **spectra_payload,
                **proposal_diagnostics,
            }
        )
        decision_step += 1

    # F1: with sideways moves enabled the final state is not necessarily the
    # best one visited.  Returning the best keeps the plateau escape strictly
    # non-harmful.  Degree preservation holds for every visited state, so this
    # cannot violate the fibre constraint.
    result = current
    if cfg.return_best_state and sideways_accepted > 0:
        result = best_graph
        if [int(result.degree(node)) for node in sorted(result.nodes())] != (
            initial_degrees
        ):
            raise AssertionError("The best spectral state changed indexed degrees.")

    if return_trace:
        return result, trace
    return result

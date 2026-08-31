from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import networkx as nx
import numpy as np
import torch

from grapher.rewiring_mlp.core.rewiring import Action
from grapher.rewiring_mlp.generic.basis import TopologyGraphletBasis
from grapher.rewiring_mlp.generic.data import normalize_topology_graph
from grapher.rewiring_mlp.generic.flow_data import (
    TopologyFlowGraphletExample,
    collate_flow_graphlet_examples,
)
from grapher.rewiring_mlp.generic.flow_model import TopologyFlowGraphletPredictor
from grapher.rewiring_mlp.generic.graphlet_diffusion import (
    GraphletLogitBridgeSchedule,
    candidate_graphlet_logits_from_counts,
    extract_topology_graphlet_simplex,
    graphlet_logit_distance,
    graphlet_simplex_to_clr,
)
from grapher.rewiring_mlp.generic.graphlets import TopologyGraphletCounts
from grapher.rewiring_mlp.generic.rewiring import (
    propose_valid_topology_swaps,
    topology_state_key,
)
from grapher.rewiring_mlp.generic.spectral_refiner import SpectralRefinerConfig


@dataclass(frozen=True)
class FlowGraphletPrediction:
    flow_velocity: np.ndarray
    clean_graphlet_logits: np.ndarray
    clean_graphlet_probabilities: np.ndarray
    graphlet_coordinate_mask: np.ndarray
    soft_degree_residual: float


@dataclass(frozen=True)
class FlowGraphletRefinerConfig(SpectralRefinerConfig):
    """Projected flow-matching + graphlet guidance.

    The inherited fields provide the common topology search/prediction-horizon
    controls. Spectral distance fields are unused. Candidate improvement is
    instead a weighted sum of edge-flow alignment and exact local graphlet gain.
    """

    flow_weight_initial: float = 1.0
    flow_weight_final: float = 1.0
    graphlet_weight_initial: float = 0.5
    graphlet_weight_final: float = 2.0
    guidance_weight_schedule: str = "cosine"
    guidance_weight_power: float = 1.0
    flow_normalize_per_swap: bool = True

    graphlet_distance: str = "clr_rmse"
    graphlet_logit_epsilon: float = 1.0e-5
    graphlet_size_weights: dict[str, float] = field(default_factory=dict)
    graphlet_bridge_schedule: str = "cosine"
    graphlet_bridge_min_clean_mix: float = 0.15
    graphlet_bridge_max_clean_mix: float = 1.0
    graphlet_bridge_power: float = 1.5

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
                f"Unknown flow global_to_local schedule: {self.guidance_weight_schedule!r}."
            )
        flow = self.flow_weight_initial + (
            self.flow_weight_final - self.flow_weight_initial
        ) * shaped
        graphlet = self.graphlet_weight_initial + (
            self.graphlet_weight_final - self.graphlet_weight_initial
        ) * shaped
        return float(flow), float(graphlet)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None = None) -> "FlowGraphletRefinerConfig":
        values = dict(data or {})
        mode = str(values.get("mode", "flow_graphlet")).lower()
        if mode not in {
            "flow_graphlet",
            "flow_matching_graphlet",
            "edge_flow_graphlet",
            "flow+graphlet",
        }:
            raise ValueError(
                "FlowGraphletRefinerConfig requires topology_refiner.mode: flow_graphlet."
            )

        # Reuse the mature common candidate-budget, selection, connectivity,
        # stopping, and prediction-horizon parsing. The spectral-specific fields
        # in the base object are inert in this refiner.
        base_values = dict(values)
        base_values["mode"] = "spectral"
        base_values.setdefault("spectral_guidance", {})
        base = SpectralRefinerConfig.from_dict(base_values)

        flow = dict(values.get("flow_guidance", {}) or {})
        graphlet = dict(values.get("graphlet_guidance", {}) or {})
        global_to_local = dict(values.get("global_to_local", {}) or {})
        bridge = GraphletLogitBridgeSchedule.from_dict(graphlet)
        size_weights_raw = graphlet.get("size_weights", {}) or {}
        if not isinstance(size_weights_raw, Mapping):
            raise ValueError(
                "graphlet_guidance.size_weights must be a mapping keyed by graphlet order."
            )
        size_weights = {str(key): float(value) for key, value in size_weights_raw.items()}

        config = cls(
            **base.__dict__,
            flow_weight_initial=float(
                global_to_local.get("flow_initial", flow.get("weight", 1.0))
            ),
            flow_weight_final=float(
                global_to_local.get("flow_final", flow.get("weight", 1.0))
            ),
            graphlet_weight_initial=float(global_to_local.get("graphlet_initial", 0.5)),
            graphlet_weight_final=float(global_to_local.get("graphlet_final", 2.0)),
            guidance_weight_schedule=str(
                global_to_local.get("schedule", "cosine")
            ).lower(),
            guidance_weight_power=float(global_to_local.get("power", 1.0)),
            flow_normalize_per_swap=bool(flow.get("normalize_per_swap", True)),
            graphlet_distance=str(graphlet.get("distance", "clr_rmse")).lower(),
            graphlet_logit_epsilon=float(graphlet.get("logit_epsilon", 1.0e-5)),
            graphlet_size_weights=size_weights,
            graphlet_bridge_schedule=bridge.schedule,
            graphlet_bridge_min_clean_mix=bridge.min_clean_mix,
            graphlet_bridge_max_clean_mix=bridge.max_clean_mix,
            graphlet_bridge_power=bridge.power,
        )
        if config.guidance_weight_schedule not in {"linear", "cosine", "power"}:
            raise ValueError("global_to_local.schedule must be linear, cosine, or power.")
        if config.guidance_weight_power <= 0.0 or not np.isfinite(
            config.guidance_weight_power
        ):
            raise ValueError("global_to_local.power must be finite and positive.")
        for value in (
            config.flow_weight_initial,
            config.flow_weight_final,
            config.graphlet_weight_initial,
            config.graphlet_weight_final,
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError("Flow/graphlet guidance weights must be finite and nonnegative.")
        if config.graphlet_logit_epsilon <= 0.0 or not np.isfinite(
            config.graphlet_logit_epsilon
        ):
            raise ValueError("graphlet_guidance.logit_epsilon must be finite and positive.")
        return config


@torch.no_grad()
def predict_flow_and_graphlets(
    model: TopologyFlowGraphletPredictor,
    *,
    source_graph: nx.Graph,
    soft_edge_probabilities: np.ndarray,
    graphlet_basis: TopologyGraphletBasis,
    time: float,
    device: torch.device | str,
) -> FlowGraphletPrediction:
    model.eval()
    source = normalize_topology_graph(source_graph)
    n = source.number_of_nodes()
    soft = np.asarray(soft_edge_probabilities, dtype=np.float32)
    if soft.shape != (n, n):
        raise ValueError("Soft flow state shape must match the source graph size.")
    # Inference targets are dummies; only masks/shapes are consumed by forward().
    source_prob, mask, _ = extract_topology_graphlet_simplex(
        source,
        graphlet_basis=graphlet_basis,
    )
    source_logits = graphlet_simplex_to_clr(
        source_prob,
        graphlet_basis=graphlet_basis,
        epsilon=1.0e-5,
        coordinate_mask=mask,
    )
    example = TopologyFlowGraphletExample(
        source_graph=source,
        time=float(np.clip(time, 0.0, 1.0)),
        current_edge_probabilities=soft,
        flow_target=np.zeros((n, n), dtype=np.float32),
        clean_graphlet_probabilities_target=source_prob.astype(np.float32),
        clean_graphlet_logits_target=source_logits.astype(np.float32),
        graphlet_coordinate_mask=mask.astype(np.bool_),
    )
    batch = collate_flow_graphlet_examples([example]).to(device)
    outputs = model(batch)
    velocity = outputs["flow_velocity"][0, :n, :n].detach().cpu().numpy().astype(np.float64)
    clean_logits = outputs["clean_graphlet_logits"][0].detach().cpu().numpy().astype(np.float64)
    clean_probabilities = (
        outputs["clean_graphlet_probabilities"][0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    residual = float(np.max(np.abs(velocity.sum(axis=1)))) if n else 0.0
    return FlowGraphletPrediction(
        flow_velocity=velocity,
        clean_graphlet_logits=clean_logits,
        clean_graphlet_probabilities=clean_probabilities,
        graphlet_coordinate_mask=mask.astype(np.bool_),
        soft_degree_residual=residual,
    )


def _flow_alignment(velocity: np.ndarray, action: Action, *, normalize: bool) -> float:
    removed, added = action
    value = sum(float(velocity[u, v]) for u, v in added) - sum(
        float(velocity[u, v]) for u, v in removed
    )
    # A perfectly matched double-edge swap under targets in {-1,0,+1} scores 4.
    if normalize:
        value /= 4.0
    return float(value)


def _select_row(
    rows: list[dict[str, Any]],
    *,
    config: FlowGraphletRefinerConfig,
    rng: np.random.Generator,
) -> tuple[int | None, float, list[float]]:
    improvements = np.asarray(
        [float(row["energy_improvement"]) for row in rows], dtype=np.float64
    )
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
    if np.any(finite):
        shifted = scores.copy()
        shifted[finite] -= float(np.max(shifted[finite]))
        probabilities[finite] = np.exp(shifted[finite] / float(config.temperature))
        probabilities /= max(float(probabilities.sum()), 1.0e-12)
    else:
        probabilities[-1] = 1.0
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


def score_flow_graphlet_candidates(
    graph: nx.Graph,
    candidates: list[Action],
    *,
    candidate_graphs: dict[Action, nx.Graph],
    current_graphlet_counts: TopologyGraphletCounts,
    graphlet_basis: TopologyGraphletBasis,
    velocity: np.ndarray,
    clean_graphlet_logits: np.ndarray,
    graphlet_coordinate_mask: np.ndarray,
    accepted_step: int,
    config: FlowGraphletRefinerConfig,
) -> list[dict[str, Any]]:
    current_prob, current_mask, _ = extract_topology_graphlet_simplex(
        graph,
        graphlet_basis=graphlet_basis,
    )
    if not np.array_equal(current_mask, graphlet_coordinate_mask):
        raise AssertionError("Current/predicted graphlet coordinate masks must match.")
    current_logits = graphlet_simplex_to_clr(
        current_prob,
        graphlet_basis=graphlet_basis,
        epsilon=config.graphlet_logit_epsilon,
        coordinate_mask=current_mask,
    )
    clean_mix = config.graphlet_bridge.clean_mix_for_step(
        accepted_step=accepted_step,
        total_steps=max(config.steps, 1),
    )
    graphlet_target = config.graphlet_bridge.target(
        current_logits,
        clean_graphlet_logits,
        clean_mix,
    )
    current_graphlet_distance = graphlet_logit_distance(
        current_logits,
        graphlet_target,
        graphlet_basis=graphlet_basis,
        coordinate_mask=current_mask,
        metric=config.graphlet_distance,
        size_weights=config.graphlet_size_weights,
    )
    progress = float(accepted_step / max(config.steps, 1))
    flow_weight, graphlet_weight = config.guidance_weights_at(progress)
    current_energy_scale = (
        graphlet_weight * current_graphlet_distance + flow_weight + 1.0e-12
    )

    rows: list[dict[str, Any]] = []
    for action in candidates:
        candidate = candidate_graphs[action]
        candidate_logits, _candidate_prob, candidate_counts = (
            candidate_graphlet_logits_from_counts(
                graph,
                candidate,
                action,
                current_counts=current_graphlet_counts,
                graphlet_basis=graphlet_basis,
                epsilon=config.graphlet_logit_epsilon,
            )
        )
        candidate_graphlet_distance = graphlet_logit_distance(
            candidate_logits,
            graphlet_target,
            graphlet_basis=graphlet_basis,
            coordinate_mask=current_mask,
            metric=config.graphlet_distance,
            size_weights=config.graphlet_size_weights,
        )
        graphlet_gain = current_graphlet_distance - candidate_graphlet_distance
        flow_gain = _flow_alignment(
            velocity,
            action,
            normalize=config.flow_normalize_per_swap,
        )
        improvement = flow_weight * flow_gain + graphlet_weight * graphlet_gain
        relative = improvement / max(abs(current_energy_scale), config.relative_improvement_epsilon)
        rows.append(
            {
                "action": action,
                "candidate_graph": candidate,
                "candidate_graphlet_counts": candidate_counts,
                "flow_gain": float(flow_gain),
                "graphlet_gain": float(graphlet_gain),
                "energy_improvement": float(improvement),
                "relative_energy_improvement": float(relative),
                "flow_weight": float(flow_weight),
                "graphlet_weight": float(graphlet_weight),
                "graphlet_clean_mix": float(clean_mix),
                "current_graphlet_distance": float(current_graphlet_distance),
                "candidate_graphlet_distance": float(candidate_graphlet_distance),
            }
        )
    return rows


def refine_graph_with_flow_graphlet_predictions(
    graph: nx.Graph,
    *,
    model: TopologyFlowGraphletPredictor,
    graphlet_basis: TopologyGraphletBasis,
    refiner_config: FlowGraphletRefinerConfig | dict[str, Any] | None,
    device: torch.device | str,
    rng: np.random.Generator,
    return_trace: bool = False,
    debug_context: str | None = None,
) -> tuple[nx.Graph, list[dict[str, Any]]] | nx.Graph:
    config = (
        refiner_config
        if isinstance(refiner_config, FlowGraphletRefinerConfig)
        else FlowGraphletRefinerConfig.from_dict(refiner_config)
    )
    current = normalize_topology_graph(graph)
    source = current.copy()
    n = current.number_of_nodes()
    soft_state = nx.to_numpy_array(
        source, nodelist=list(range(n)), dtype=np.float64
    )
    source_degree = soft_state.sum(axis=1)
    visited = {topology_state_key(current)} if config.reject_revisited_states else set()
    trace: list[dict[str, Any]] = []
    current_prob, current_mask, current_counts = extract_topology_graphlet_simplex(
        current,
        graphlet_basis=graphlet_basis,
    )
    del current_prob, current_mask

    cached_prediction: FlowGraphletPrediction | None = None
    prediction_calls = 0
    prediction_horizon = 1
    accepted_steps = 0
    accepted_since_prediction = 0

    while accepted_steps < int(config.steps):
        progress = float(accepted_steps / max(config.steps, 1))
        refresh = (
            cached_prediction is None
            or accepted_since_prediction >= prediction_horizon
        )
        if refresh:
            if cached_prediction is not None and accepted_since_prediction > 0:
                dt = float(accepted_since_prediction / max(config.steps, 1))
                soft_state = soft_state + dt * cached_prediction.flow_velocity
            prediction_horizon = config.prediction_horizon_at(progress)
            cached_prediction = predict_flow_and_graphlets(
                model,
                source_graph=source,
                soft_edge_probabilities=soft_state,
                graphlet_basis=graphlet_basis,
                time=progress,
                device=device,
            )
            prediction_calls += 1
            accepted_since_prediction = 0
            soft_degree_residual = float(
                np.max(np.abs(soft_state.sum(axis=1) - source_degree))
            ) if n else 0.0
        else:
            soft_degree_residual = float(
                np.max(np.abs(soft_state.sum(axis=1) - source_degree))
            ) if n else 0.0

        excluded = visited if config.reject_revisited_states else None
        actions, candidate_graphs, proposal_diag = propose_valid_topology_swaps(
            current,
            proposal_budget=config.proposal_budget,
            valid_candidate_budget=config.valid_candidate_budget,
            preserve_connectivity=config.preserve_connectivity,
            rng=rng,
            excluded_states=excluded,
        )
        if not actions:
            trace.append(
                {
                    "step": accepted_steps,
                    "accepted": False,
                    "terminal_stop": True,
                    "reason": "no_valid_candidates",
                    "prediction_refreshed": refresh,
                    "prediction_calls": prediction_calls,
                    "prediction_horizon": prediction_horizon,
                    "soft_degree_residual": soft_degree_residual,
                    **proposal_diag,
                }
            )
            break

        assert cached_prediction is not None
        rows = score_flow_graphlet_candidates(
            current,
            actions,
            candidate_graphs=candidate_graphs,
            current_graphlet_counts=current_counts,
            graphlet_basis=graphlet_basis,
            velocity=cached_prediction.flow_velocity,
            clean_graphlet_logits=cached_prediction.clean_graphlet_logits,
            graphlet_coordinate_mask=cached_prediction.graphlet_coordinate_mask,
            accepted_step=accepted_steps,
            config=config,
        )
        selected_index, stop_probability, probabilities = _select_row(
            rows,
            config=config,
            rng=rng,
        )
        if selected_index is None:
            # If a frozen K-step prediction has already driven at least one swap,
            # refresh immediately before declaring a plateau. This mirrors the
            # generation-only adaptive-horizon behavior of the spectral refiner.
            if config.refresh_on_prediction_plateau and accepted_since_prediction > 0:
                trace.append(
                    {
                        "step": accepted_steps,
                        "accepted": False,
                        "terminal_stop": False,
                        "reason": "prediction_plateau_refresh",
                        "prediction_refreshed": False,
                        "prediction_calls": prediction_calls,
                        "prediction_horizon": prediction_horizon,
                        "stop_probability": stop_probability,
                        "soft_degree_residual": soft_degree_residual,
                        **proposal_diag,
                    }
                )
                # Force a new neural query; the accepted interval is integrated
                # into the continuous soft state at the top of the next loop.
                prediction_horizon = accepted_since_prediction
                continue
            best_improvement = max(
                (float(row["energy_improvement"]) for row in rows), default=0.0
            )
            trace.append(
                {
                    "step": accepted_steps,
                    "accepted": False,
                    "terminal_stop": True,
                    "reason": "no_positive_flow_graphlet_projection",
                    "prediction_refreshed": refresh,
                    "prediction_calls": prediction_calls,
                    "prediction_horizon": prediction_horizon,
                    "stop_probability": stop_probability,
                    "best_candidate_improvement": best_improvement,
                    "soft_degree_residual": soft_degree_residual,
                    **proposal_diag,
                }
            )
            break

        selected = rows[selected_index]
        current = selected["candidate_graph"]
        current_counts = selected["candidate_graphlet_counts"]
        if config.reject_revisited_states:
            visited.add(topology_state_key(current))
        accepted_steps += 1
        accepted_since_prediction += 1
        row = {
            "step": accepted_steps - 1,
            "accepted": True,
            "terminal_stop": False,
            "reason": "accepted",
            "prediction_refreshed": refresh,
            "prediction_calls": prediction_calls,
            "prediction_horizon": prediction_horizon,
            "accepted_since_prediction": accepted_since_prediction,
            "stop_probability": stop_probability,
            "action_probabilities": probabilities,
            "soft_degree_residual": soft_degree_residual,
            "predicted_flow_degree_tangent_residual": cached_prediction.soft_degree_residual,
            **proposal_diag,
            **{
                key: value
                for key, value in selected.items()
                if key not in {"candidate_graph", "candidate_graphlet_counts"}
            },
        }
        trace.append(row)
        if config.debug_enabled and (
            accepted_steps == 1 or accepted_steps % config.debug_print_every == 0
        ):
            prefix = "[GraphER/FlowGraphlet]"
            context = f" {debug_context}" if debug_context else ""
            print(
                f"{prefix}{context} step={accepted_steps}/{config.steps} "
                f"flow_gain={row['flow_gain']:.6g} graphlet_gain={row['graphlet_gain']:.6g} "
                f"combined={row['energy_improvement']:.6g} horizon={prediction_horizon}",
                flush=True,
            )

    if return_trace:
        return current, trace
    return current

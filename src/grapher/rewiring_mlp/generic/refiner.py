from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import networkx as nx
import numpy as np
import torch

from grapher.properties.summary import SummaryConfig
from grapher.rewiring_mlp.core.rewiring import Action
from grapher.rewiring_mlp.generic.data import (
    TopologyGraphletExample,
    collate_topology_examples,
    normalize_topology_graph,
)
from grapher.rewiring_mlp.generic.basis import TopologyGraphletBasis
from grapher.rewiring_mlp.generic.graphlets import (
    candidate_topology_graphlet_counts,
    extract_topology_graphlet_counts,
    topology_structural_discrepancy_from_counts,
)
from grapher.rewiring_mlp.generic.model import TopologyGraphletPredictor
from grapher.rewiring_mlp.generic.rewiring import (
    propose_valid_topology_swaps,
    topology_state_key,
)


@dataclass(frozen=True)
class TopologyPrediction:
    graphlet_target: np.ndarray
    graphlet_mass_target: np.ndarray
    graphlet_history: dict[str, dict[str, float]]
    graphlet_connected_mass: dict[str, float]
    clustering_target: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64)
    )
    orbit_target: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64)
    )


@dataclass(frozen=True)
class TopologyRefinerConfig:
    steps: int = 80
    proposal_budget: int = 512
    valid_candidate_budget: int = 128
    preserve_connectivity: bool = True
    selection: str = "greedy"
    temperature: float = 0.1
    graphlet_weight: float = 1.0
    graphlet_mass_weight: float = 0.0
    clustering_weight: float = 0.0
    orbit_weight: float = 0.0
    accept_only_improving: bool = True
    min_improvement: float = 1.0e-8
    sample_graphlet: bool = False
    refresh_prediction_every: int = 1
    reject_revisited_states: bool = True

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any] | None = None,
    ) -> "TopologyRefinerConfig":
        values = dict(data or {})
        mode = str(values.get("mode", "energy")).lower()
        if mode != "energy":
            raise NotImplementedError(
                "The decoupled topology reference path currently implements exact "
                "structural-summary energy selection only. Legacy pair-aware "
                "selector checkpoints are incompatible and must not be reused."
            )
        legacy_budget = int(values.get("candidate_budget", 128))
        valid_budget = int(values.get("valid_candidate_budget", legacy_budget))
        proposal_budget = int(
            values.get(
                "proposal_budget",
                valid_budget if valid_budget < 0 else max(valid_budget, 1) * 4,
            )
        )
        config = cls(
            steps=int(values.get("steps", 80)),
            proposal_budget=proposal_budget,
            valid_candidate_budget=valid_budget,
            preserve_connectivity=bool(values.get("preserve_connectivity", True)),
            selection=str(values.get("selection", "greedy")).lower(),
            temperature=float(values.get("temperature", 0.1)),
            graphlet_weight=float(values.get("graphlet_weight", 1.0)),
            graphlet_mass_weight=float(values.get("graphlet_mass_weight", 0.0)),
            clustering_weight=float(values.get("clustering_weight", 0.0)),
            orbit_weight=float(values.get("orbit_weight", 0.0)),
            accept_only_improving=bool(
                values.get("accept_only_improving", True)
            ),
            min_improvement=float(values.get("min_improvement", 1.0e-8)),
            sample_graphlet=bool(values.get("sample_graphlet", False)),
            refresh_prediction_every=int(
                values.get("refresh_prediction_every", 1)
            ),
            reject_revisited_states=bool(
                values.get("reject_revisited_states", True)
            ),
        )
        if config.steps < 0:
            raise ValueError("topology_refiner.steps must be non-negative.")
        if config.proposal_budget == 0 or config.valid_candidate_budget == 0:
            raise ValueError("Topology proposal budgets must be non-zero.")
        if config.selection not in {"greedy", "argmax", "softmax", "sample"}:
            raise ValueError("Topology selection must be greedy or softmax sampling.")
        if config.temperature <= 0.0:
            raise ValueError("topology_refiner.temperature must be positive.")
        for name, value in {
            "graphlet_weight": config.graphlet_weight,
            "graphlet_mass_weight": config.graphlet_mass_weight,
            "clustering_weight": config.clustering_weight,
            "orbit_weight": config.orbit_weight,
        }.items():
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"topology_refiner.{name} must be finite and nonnegative."
                )
        if not any(
            value > 0.0
            for value in (
                config.graphlet_weight,
                config.clustering_weight,
                config.orbit_weight,
            )
        ):
            raise ValueError("At least one topology structural weight must be active.")
        if not np.isfinite(config.min_improvement) or config.min_improvement < 0.0:
            raise ValueError(
                "topology_refiner.min_improvement must be finite and nonnegative."
            )
        if not config.preserve_connectivity:
            raise ValueError(
                "The decoupled generic topology path requires connectivity-"
                "preserving swaps."
            )
        if not config.accept_only_improving:
            raise ValueError(
                "The decoupled topology path accepts only positive-improvement "
                "swaps."
            )
        if config.refresh_prediction_every <= 0:
            raise ValueError("refresh_prediction_every must be positive.")
        return config


@torch.no_grad()
def predict_topology_target(
    model: TopologyGraphletPredictor,
    graph: nx.Graph,
    *,
    time: float,
    graphlet_basis: TopologyGraphletBasis,
    device: torch.device | str,
    rng: np.random.Generator,
    sample_graphlet: bool = False,
) -> TopologyPrediction:
    """Predict the state-conditioned graphlet, clustering, and orbit targets."""

    model.eval()
    batch = collate_topology_examples(
        [
            TopologyGraphletExample(
                current_graph=graph,
                time=float(time),
                graphlet_target=np.zeros(graphlet_basis.width, dtype=np.float32),
                graphlet_mass_target=np.zeros(
                    len(graphlet_basis.sizes), dtype=np.float32
                ),
                clustering_target=np.zeros(
                    model.clustering_width, dtype=np.float32
                ),
                orbit_target=np.zeros(model.orbit_width, dtype=np.float32),
            )
        ]
    ).to(device)
    outputs = model(batch)
    alpha = outputs["graphlet_alpha"][0].detach().cpu().numpy()
    graphlet_target = np.zeros(graphlet_basis.width, dtype=np.float64)
    for start, stop in graphlet_basis.slices:
        block = np.maximum(alpha[start:stop], 1.0e-12)
        graphlet_target[start:stop] = (
            rng.dirichlet(block)
            if sample_graphlet
            else block / float(block.sum())
        )
    mass_ab = outputs["graphlet_mass_ab"][0].detach().cpu().numpy()
    graphlet_mass = np.asarray(
        [
            rng.beta(max(float(a), 1.0e-12), max(float(b), 1.0e-12))
            if sample_graphlet
            else float(a / max(a + b, 1.0e-12))
            for a, b in mass_ab
        ],
        dtype=np.float64,
    )
    clustering_target = np.zeros(model.clustering_width, dtype=np.float64)
    if model.clustering_width > 0:
        clustering_alpha = np.maximum(
            outputs["clustering_alpha"][0].detach().cpu().numpy(),
            1.0e-12,
        )
        clustering_target = (
            rng.dirichlet(clustering_alpha)
            if sample_graphlet
            else clustering_alpha / float(clustering_alpha.sum())
        )
    orbit_target = np.zeros(model.orbit_width, dtype=np.float64)
    if model.orbit_width > 0:
        orbit_target = np.expm1(
            outputs["orbit_log_mean"][0].detach().cpu().numpy()
        ).clip(min=0.0)
    return TopologyPrediction(
        graphlet_target=graphlet_target,
        graphlet_mass_target=graphlet_mass,
        graphlet_history=graphlet_basis.unflatten_history(graphlet_target),
        graphlet_connected_mass={
            key: float(value)
            for key, value in zip(graphlet_basis.sizes, graphlet_mass)
        },
        clustering_target=clustering_target,
        orbit_target=orbit_target,
    )


def score_topology_candidates(
    graph: nx.Graph,
    candidates: Sequence[Action],
    prediction: TopologyPrediction,
    *,
    graphlet_basis: TopologyGraphletBasis,
    summary_config: SummaryConfig,
    config: TopologyRefinerConfig | dict[str, Any] | None = None,
    candidate_graphs: dict[Action, nx.Graph] | None = None,
) -> list[dict[str, Any]]:
    """Score candidates against one frozen graph-level structural prediction."""

    del summary_config
    cfg = (
        config
        if isinstance(config, TopologyRefinerConfig)
        else TopologyRefinerConfig.from_dict(config)
    )
    if cfg.clustering_weight > 0.0 and prediction.clustering_target.size == 0:
        raise ValueError(
            "clustering_weight is active but the checkpoint has no clustering head."
        )
    if cfg.orbit_weight > 0.0 and prediction.orbit_target.size == 0:
        raise ValueError("orbit_weight is active but the checkpoint has no orbit head.")

    current_counts = extract_topology_graphlet_counts(
        graph,
        graphlet_basis=graphlet_basis,
    )

    def score(
        candidate: nx.Graph,
        counts: dict[str, dict[str, int]],
    ) -> dict[str, float]:
        return topology_structural_discrepancy_from_counts(
            candidate,
            counts,
            graphlet_target=prediction.graphlet_target,
            graphlet_mass_target=prediction.graphlet_mass_target,
            clustering_target=prediction.clustering_target,
            orbit_target=prediction.orbit_target,
            graphlet_basis=graphlet_basis,
            graphlet_weight=cfg.graphlet_weight,
            graphlet_mass_weight=cfg.graphlet_mass_weight,
            clustering_weight=cfg.clustering_weight,
            orbit_weight=cfg.orbit_weight,
        )

    current_score = score(graph, current_counts)
    rows: list[dict[str, Any]] = []
    for action in candidates:
        if candidate_graphs is None or action not in candidate_graphs:
            raise ValueError(
                "Topology candidate materialization is required for scoring."
            )
        candidate = candidate_graphs[action]
        candidate_counts = candidate_topology_graphlet_counts(
            graph,
            candidate,
            action,
            current_counts=current_counts,
            graphlet_basis=graphlet_basis,
        )
        candidate_score = score(candidate, candidate_counts)
        graphlet_gain = float(
            current_score["graphlet"] - candidate_score["graphlet"]
        )
        clustering_gain = float(
            current_score["clustering"] - candidate_score["clustering"]
        )
        orbit_gain = float(current_score["orbit"] - candidate_score["orbit"])
        structural_gain = float(current_score["total"] - candidate_score["total"])
        rows.append(
            {
                "action": action,
                "candidate_graph": candidate,
                "current_structural_discrepancy": float(current_score["total"]),
                "candidate_structural_discrepancy": float(
                    candidate_score["total"]
                ),
                "current_graphlet_discrepancy": float(current_score["graphlet"]),
                "candidate_graphlet_discrepancy": float(
                    candidate_score["graphlet"]
                ),
                "current_histogram_discrepancy": float(
                    current_score["graphlet_histogram"]
                ),
                "candidate_histogram_discrepancy": float(
                    candidate_score["graphlet_histogram"]
                ),
                "current_mass_discrepancy": float(current_score["graphlet_mass"]),
                "candidate_mass_discrepancy": float(
                    candidate_score["graphlet_mass"]
                ),
                "current_clustering_discrepancy": float(
                    current_score["clustering"]
                ),
                "candidate_clustering_discrepancy": float(
                    candidate_score["clustering"]
                ),
                "current_orbit_discrepancy": float(current_score["orbit"]),
                "candidate_orbit_discrepancy": float(candidate_score["orbit"]),
                "graphlet_gain": graphlet_gain,
                "clustering_gain": clustering_gain,
                "orbit_gain": orbit_gain,
                "structural_gain": structural_gain,
                "energy_improvement": structural_gain,
            }
        )
    return rows


def _select_row(
    rows: Sequence[dict[str, Any]],
    *,
    config: TopologyRefinerConfig,
    rng: np.random.Generator,
) -> tuple[int | None, float, list[float]]:
    improvements = np.asarray(
        [float(row["energy_improvement"]) for row in rows],
        dtype=np.float64,
    )
    scores = np.concatenate([improvements, np.asarray([0.0])])
    if config.accept_only_improving:
        scores[:-1][improvements <= float(config.min_improvement)] = -np.inf
        if np.any(np.isfinite(scores[:-1])):
            # STOP is feasible only when the constrained candidate set has no
            # positive-improvement move.
            scores[-1] = -np.inf
    finite = np.isfinite(scores)
    shifted = scores.copy()
    shifted[finite] -= float(np.max(shifted[finite]))
    probabilities = np.zeros_like(scores)
    probabilities[finite] = np.exp(shifted[finite] / float(config.temperature))
    probabilities /= float(probabilities.sum())
    if config.selection in {"greedy", "argmax"}:
        best = float(np.max(scores))
        maximizers = np.flatnonzero(np.isclose(scores, best, atol=1.0e-12))
        selected = int(rng.choice(maximizers))
    else:
        selected = int(rng.choice(len(scores), p=probabilities))
    stop_index = len(rows)
    return (
        None if selected == stop_index else selected,
        float(probabilities[-1]),
        probabilities.tolist(),
    )


def refine_graph_with_topology_predictions(
    graph: nx.Graph,
    *,
    model: TopologyGraphletPredictor,
    graphlet_basis: TopologyGraphletBasis,
    summary_config: SummaryConfig,
    refiner_config: TopologyRefinerConfig | dict[str, Any] | None = None,
    device: torch.device | str = "cpu",
    rng: np.random.Generator | None = None,
    return_trace: bool = False,
    prediction_fn: Any | None = None,
) -> nx.Graph | tuple[nx.Graph, list[dict[str, Any]]]:
    """Apply degree-preserving structural correction to a generic topology."""

    cfg = (
        refiner_config
        if isinstance(refiner_config, TopologyRefinerConfig)
        else TopologyRefinerConfig.from_dict(refiner_config)
    )
    generator = rng if rng is not None else np.random.default_rng(0)
    predictor = prediction_fn or predict_topology_target
    current = normalize_topology_graph(graph)
    if current.number_of_nodes() > 1 and not nx.is_connected(current):
        raise ValueError("Topology refinement requires a connected source graph.")
    initial_degrees = [int(current.degree(node)) for node in sorted(current.nodes())]
    visited = {topology_state_key(current)}
    trace: list[dict[str, Any]] = []
    prediction: TopologyPrediction | None = None
    accepted_since_prediction = cfg.refresh_prediction_every
    prediction_calls = 0

    for step in range(cfg.steps):
        prediction_refreshed = False
        if (
            prediction is None
            or accepted_since_prediction >= cfg.refresh_prediction_every
        ):
            prediction = predictor(
                model,
                current,
                time=float(step / max(cfg.steps, 1)),
                graphlet_basis=graphlet_basis,
                device=device,
                rng=generator,
                sample_graphlet=cfg.sample_graphlet,
            )
            prediction_calls += 1
            accepted_since_prediction = 0
            prediction_refreshed = True

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
            trace.append(
                {
                    "step": step,
                    "accepted": False,
                    "reason": "explicit_stop_no_candidates",
                    "prediction_refreshed": prediction_refreshed,
                    "prediction_calls": prediction_calls,
                    **proposal_diagnostics,
                }
            )
            break
        rows = score_topology_candidates(
            current,
            candidates,
            prediction,
            graphlet_basis=graphlet_basis,
            summary_config=summary_config,
            config=cfg,
            candidate_graphs=candidate_graphs,
        )
        selected, stop_probability, probabilities = _select_row(
            rows,
            config=cfg,
            rng=generator,
        )
        if selected is None:
            trace.append(
                {
                    "step": step,
                    "accepted": False,
                    "reason": "explicit_stop_no_positive_structural_gain",
                    "prediction_refreshed": prediction_refreshed,
                    "prediction_calls": prediction_calls,
                    "stop_probability": stop_probability,
                    "selection_probabilities": probabilities,
                    "current_graphlet_discrepancy": float(
                        rows[0]["current_graphlet_discrepancy"]
                    ),
                    "current_structural_discrepancy": float(
                        rows[0]["current_structural_discrepancy"]
                    ),
                    **proposal_diagnostics,
                }
            )
            break

        chosen = rows[selected]
        candidate = chosen["candidate_graph"]
        if [int(candidate.degree(node)) for node in sorted(candidate.nodes())] != (
            initial_degrees
        ):
            raise AssertionError("A topology rewiring action changed indexed degrees.")
        if (
            cfg.preserve_connectivity
            and candidate.number_of_nodes() > 1
            and not nx.is_connected(candidate)
        ):
            raise AssertionError("A topology rewiring action broke connectivity.")
        current = candidate
        visited.add(topology_state_key(current))
        accepted_since_prediction += 1
        trace.append(
            {
                "step": step,
                "accepted": True,
                "reason": "structural_improving_swap",
                "action": chosen["action"],
                "prediction_refreshed": prediction_refreshed,
                "prediction_calls": prediction_calls,
                "stop_probability": stop_probability,
                "selected_action_probability": probabilities[selected],
                "current_graphlet_discrepancy": float(
                    chosen["current_graphlet_discrepancy"]
                ),
                "candidate_graphlet_discrepancy": float(
                    chosen["candidate_graphlet_discrepancy"]
                ),
                "current_structural_discrepancy": float(
                    chosen["current_structural_discrepancy"]
                ),
                "candidate_structural_discrepancy": float(
                    chosen["candidate_structural_discrepancy"]
                ),
                "graphlet_gain": float(chosen["graphlet_gain"]),
                "clustering_gain": float(chosen["clustering_gain"]),
                "orbit_gain": float(chosen["orbit_gain"]),
                "structural_gain": float(chosen["structural_gain"]),
                "energy_improvement": float(chosen["energy_improvement"]),
                **proposal_diagnostics,
            }
        )

    if return_trace:
        return current, trace
    return current

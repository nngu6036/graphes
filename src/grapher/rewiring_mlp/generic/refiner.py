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
    # ``temperature`` remains as the backward-compatible constant-temperature
    # knob.  New soft rewiring configs should use the start/end schedule below.
    temperature: float = 0.1
    temperature_start: float = 0.1
    temperature_end: float = 0.1
    temperature_schedule: str = "constant"
    score_normalization: str = "none"
    stop_action: bool = True
    graphlet_weight: float = 1.0
    graphlet_mass_weight: float = 0.0
    clustering_weight: float = 0.0
    orbit_weight: float = 0.0
    accept_only_improving: bool = True
    min_improvement: float = 1.0e-8
    min_relative_improvement: float = 0.0
    # Relaxed soft selection may temporarily accept a worse target score.  A
    # negative relative limit disables that bound.
    max_target_worsening: float = 0.0
    max_relative_target_worsening: float = -1.0
    relative_improvement_epsilon: float = 1.0e-12
    sample_graphlet: bool = False
    # Conservative higher-order trust region.  -1 disables an individual
    # bound.  Counts refer to *existing motif instances destroyed* by the two
    # removed edges, not merely the net count after newly created motifs.
    motif_guard_enabled: bool = False
    max_destroyed_triangles: int = -1
    max_destroyed_cycles_4: int = -1
    max_destroyed_cycles_5: int = -1
    # Backward-compatible fixed prediction horizon. New configs should prefer
    # the nested ``prediction_horizon`` block parsed below.
    refresh_prediction_every: int = 1
    prediction_horizon_mode: str = "fixed"
    prediction_horizon_initial_k: int = 1
    prediction_horizon_final_k: int = 1
    prediction_horizon_schedule: str = "constant"
    refresh_on_plateau: bool = False
    reject_revisited_states: bool = True

    def temperature_at(self, progress: float) -> float:
        clipped = float(np.clip(progress, 0.0, 1.0))
        start = float(self.temperature_start)
        end = float(self.temperature_end)
        schedule = self.temperature_schedule
        if schedule == "constant":
            return start
        if schedule == "linear":
            return start + (end - start) * clipped
        if schedule == "cosine":
            cooling = 0.5 * (1.0 + np.cos(np.pi * clipped))
            return end + (start - end) * cooling
        raise ValueError(f"Unknown soft-rewiring temperature schedule: {schedule!r}.")

    def prediction_horizon_at(self, progress: float) -> int:
        """Return the frozen-prediction rewiring budget at normalized progress."""
        if self.prediction_horizon_mode == "fixed":
            return int(self.refresh_prediction_every)

        clipped = float(np.clip(progress, 0.0, 1.0))
        start = float(self.prediction_horizon_initial_k)
        end = float(self.prediction_horizon_final_k)
        schedule = self.prediction_horizon_schedule
        if schedule == "linear":
            value = start + (end - start) * clipped
        elif schedule == "cosine":
            cooling = 0.5 * (1.0 + np.cos(np.pi * clipped))
            value = end + (start - end) * cooling
        elif schedule == "exponential":
            value = start * ((end / start) ** clipped)
        else:
            raise ValueError(
                f"Unknown prediction-horizon schedule: {schedule!r}."
            )
        return max(1, int(np.floor(value + 0.5)))

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

        soft = values.get("soft_selection", {}) or {}
        if not isinstance(soft, dict):
            raise ValueError("topology_refiner.soft_selection must be a mapping.")
        legacy_temperature = float(values.get("temperature", 0.1))
        temperature_start = float(soft.get("temperature_start", legacy_temperature))
        temperature_end = float(soft.get("temperature_end", temperature_start))
        temperature_schedule = str(soft.get("temperature_schedule", "constant")).lower()
        score_normalization = str(soft.get("score_normalization", "none")).lower()
        stop_action = bool(soft.get("stop_action", True))
        max_target_worsening = float(soft.get("max_target_worsening", 0.0))
        max_relative_target_worsening = float(
            soft.get("max_relative_target_worsening", -1.0)
        )

        motif = values.get("motif_guard", {}) or {}
        if not isinstance(motif, dict):
            raise ValueError("topology_refiner.motif_guard must be a mapping.")
        cycle_limits = motif.get("max_destroyed_cycles", {}) or {}
        if not isinstance(cycle_limits, dict):
            raise ValueError(
                "topology_refiner.motif_guard.max_destroyed_cycles must be a mapping."
            )

        legacy_refresh = int(values.get("refresh_prediction_every", 1))
        horizon_data = values.get("prediction_horizon")
        if horizon_data is None:
            prediction_horizon_mode = "fixed"
            prediction_horizon_initial_k = legacy_refresh
            prediction_horizon_final_k = legacy_refresh
            prediction_horizon_schedule = "constant"
            refresh_on_plateau = bool(values.get("refresh_on_plateau", False))
        else:
            if not isinstance(horizon_data, dict):
                raise ValueError(
                    "topology_refiner.prediction_horizon must be a mapping."
                )
            horizon = dict(horizon_data)
            prediction_horizon_mode = str(horizon.get("mode", "annealed")).lower()
            if prediction_horizon_mode in {"adaptive", "anneal"}:
                prediction_horizon_mode = "annealed"
            if prediction_horizon_mode == "fixed":
                fixed_k = int(horizon.get("k", horizon.get("initial_k", legacy_refresh)))
                prediction_horizon_initial_k = fixed_k
                prediction_horizon_final_k = fixed_k
                prediction_horizon_schedule = "constant"
                legacy_refresh = fixed_k
            else:
                prediction_horizon_initial_k = int(horizon.get("initial_k", legacy_refresh))
                prediction_horizon_final_k = int(horizon.get("final_k", 1))
                prediction_horizon_schedule = str(horizon.get("schedule", "exponential")).lower()
                if prediction_horizon_schedule in {"geometric", "exp"}:
                    prediction_horizon_schedule = "exponential"
                legacy_refresh = prediction_horizon_initial_k
            refresh_on_plateau = bool(horizon.get("refresh_on_plateau", True))

        config = cls(
            steps=int(values.get("steps", 80)),
            proposal_budget=proposal_budget,
            valid_candidate_budget=valid_budget,
            preserve_connectivity=bool(values.get("preserve_connectivity", True)),
            selection=str(values.get("selection", "greedy")).lower(),
            temperature=legacy_temperature,
            temperature_start=temperature_start,
            temperature_end=temperature_end,
            temperature_schedule=temperature_schedule,
            score_normalization=score_normalization,
            stop_action=stop_action,
            graphlet_weight=float(values.get("graphlet_weight", 1.0)),
            graphlet_mass_weight=float(values.get("graphlet_mass_weight", 0.0)),
            clustering_weight=float(values.get("clustering_weight", 0.0)),
            orbit_weight=float(values.get("orbit_weight", 0.0)),
            accept_only_improving=bool(values.get("accept_only_improving", True)),
            min_improvement=float(values.get("min_improvement", 1.0e-8)),
            min_relative_improvement=float(values.get("min_relative_improvement", 0.0)),
            max_target_worsening=max_target_worsening,
            max_relative_target_worsening=max_relative_target_worsening,
            relative_improvement_epsilon=float(values.get("relative_improvement_epsilon", 1.0e-12)),
            sample_graphlet=bool(values.get("sample_graphlet", False)),
            motif_guard_enabled=bool(motif.get("enabled", False)),
            max_destroyed_triangles=int(motif.get("max_destroyed_triangles", -1)),
            max_destroyed_cycles_4=int(cycle_limits.get("4", cycle_limits.get(4, -1))),
            max_destroyed_cycles_5=int(cycle_limits.get("5", cycle_limits.get(5, -1))),
            refresh_prediction_every=legacy_refresh,
            prediction_horizon_mode=prediction_horizon_mode,
            prediction_horizon_initial_k=prediction_horizon_initial_k,
            prediction_horizon_final_k=prediction_horizon_final_k,
            prediction_horizon_schedule=prediction_horizon_schedule,
            refresh_on_plateau=refresh_on_plateau,
            reject_revisited_states=bool(values.get("reject_revisited_states", True)),
        )
        if config.steps < 0:
            raise ValueError("topology_refiner.steps must be non-negative.")
        if config.proposal_budget == 0 or config.valid_candidate_budget == 0:
            raise ValueError("Topology proposal budgets must be non-zero.")
        if config.selection not in {"greedy", "argmax", "softmax", "sample", "softmax_safe"}:
            raise ValueError("Topology selection must be greedy or softmax sampling.")
        if config.temperature_start <= 0.0 or config.temperature_end <= 0.0:
            raise ValueError("Soft-rewiring temperatures must be positive.")
        if config.temperature_schedule not in {"constant", "linear", "cosine"}:
            raise ValueError(
                "topology_refiner.soft_selection.temperature_schedule must be constant, linear, or cosine."
            )
        if config.score_normalization not in {"none", "std"}:
            raise ValueError(
                "topology_refiner.soft_selection.score_normalization must be none or std."
            )
        for name, value in {
            "graphlet_weight": config.graphlet_weight,
            "graphlet_mass_weight": config.graphlet_mass_weight,
            "clustering_weight": config.clustering_weight,
            "orbit_weight": config.orbit_weight,
        }.items():
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"topology_refiner.{name} must be finite and nonnegative.")
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
            raise ValueError("topology_refiner.min_improvement must be finite and nonnegative.")
        if not np.isfinite(config.min_relative_improvement) or config.min_relative_improvement < 0.0:
            raise ValueError(
                "topology_refiner.min_relative_improvement must be finite and nonnegative."
            )
        if not np.isfinite(config.max_target_worsening) or config.max_target_worsening < 0.0:
            raise ValueError(
                "topology_refiner.soft_selection.max_target_worsening must be finite and nonnegative."
            )
        if (
            not np.isfinite(config.max_relative_target_worsening)
            or config.max_relative_target_worsening < -1.0
        ):
            raise ValueError(
                "topology_refiner.soft_selection.max_relative_target_worsening must be >= -1."
            )
        if not np.isfinite(config.relative_improvement_epsilon) or config.relative_improvement_epsilon <= 0.0:
            raise ValueError(
                "topology_refiner.relative_improvement_epsilon must be finite and positive."
            )
        if not config.preserve_connectivity:
            raise ValueError(
                "The decoupled generic topology path requires connectivity-preserving swaps."
            )
        for name, value in {
            "max_destroyed_triangles": config.max_destroyed_triangles,
            "max_destroyed_cycles_4": config.max_destroyed_cycles_4,
            "max_destroyed_cycles_5": config.max_destroyed_cycles_5,
        }.items():
            if value < -1:
                raise ValueError(f"topology_refiner.motif_guard.{name} must be >= -1.")
        if config.refresh_prediction_every <= 0:
            raise ValueError("refresh_prediction_every must be positive.")
        if config.prediction_horizon_mode not in {"fixed", "annealed"}:
            raise ValueError(
                "topology_refiner.prediction_horizon.mode must be fixed or annealed."
            )
        if config.prediction_horizon_initial_k <= 0 or config.prediction_horizon_final_k <= 0:
            raise ValueError("prediction-horizon values must be positive.")
        if (
            config.prediction_horizon_mode == "annealed"
            and config.prediction_horizon_initial_k < config.prediction_horizon_final_k
        ):
            raise ValueError("Annealed prediction horizons require initial_k >= final_k.")
        if config.prediction_horizon_mode == "annealed" and (
            config.prediction_horizon_schedule not in {"linear", "cosine", "exponential"}
        ):
            raise ValueError(
                "topology_refiner.prediction_horizon.schedule must be linear, cosine, or exponential."
            )
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
        relative_structural_gain = float(
            structural_gain
            / max(
                abs(float(current_score["total"])),
                float(cfg.relative_improvement_epsilon),
            )
        )
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
                "relative_energy_improvement": relative_structural_gain,
            }
        )
    return rows



def _cycle_signatures_containing_removed_edges(
    graph: nx.Graph,
    action: Action,
    *,
    length: int,
) -> set[frozenset[tuple[int, int]]]:
    """Return existing simple cycle instances destroyed by ``action``.

    A simple k-cycle containing a removed edge (u,v) is equivalent to a simple
    (k-1)-edge path from u to v after that edge is excluded.  We enumerate only
    those local paths, then canonicalize a cycle by its undirected edge set.
    For the small generic benchmarks and k in {3,4,5}, this is substantially
    cheaper than enumerating every cycle in every candidate graph.
    """

    k = int(length)
    if k < 3:
        raise ValueError("Protected cycle length must be >= 3.")
    removed, _added = action
    cycles: set[frozenset[tuple[int, int]]] = set()

    for edge in removed:
        u, v = int(edge[0]), int(edge[1])
        blocked = {tuple(sorted((u, v)))}

        def dfs(current: int, path: list[int]) -> None:
            edges_used = len(path) - 1
            if edges_used == k - 1:
                if current != v:
                    return
                cycle_edges = {
                    tuple(sorted((path[i], path[i + 1])))
                    for i in range(len(path) - 1)
                }
                cycle_edges.add(tuple(sorted((u, v))))
                if len(cycle_edges) == k:
                    cycles.add(frozenset(cycle_edges))
                return
            if current == v:
                return
            for nxt_raw in graph.neighbors(current):
                nxt = int(nxt_raw)
                e = tuple(sorted((current, nxt)))
                if e in blocked:
                    continue
                # v may only be entered on the final path edge.
                if nxt == v and edges_used + 1 != k - 1:
                    continue
                if nxt != v and nxt in path:
                    continue
                dfs(nxt, path + [nxt])

        dfs(u, [u])
    return cycles


def _filter_candidates_by_motif_guard(
    graph: nx.Graph,
    candidates: Sequence[Action],
    candidate_graphs: dict[Action, nx.Graph],
    *,
    config: TopologyRefinerConfig,
) -> tuple[list[Action], dict[Action, nx.Graph], dict[Action, dict[str, int]], dict[str, Any]]:
    """Apply the conservative short-motif trust region before target scoring."""

    if not config.motif_guard_enabled:
        zero = {
            action: {
                "destroyed_triangles": 0,
                "destroyed_cycles_4": 0,
                "destroyed_cycles_5": 0,
            }
            for action in candidates
        }
        return list(candidates), dict(candidate_graphs), zero, {
            "num_motif_guard_rejections": 0,
            "motif_guard_rejection_reasons": {},
        }

    retained: list[Action] = []
    retained_graphs: dict[Action, nx.Graph] = {}
    damage_by_action: dict[Action, dict[str, int]] = {}
    rejections: dict[str, int] = {}
    cache: dict[tuple[tuple[tuple[int, int], tuple[int, int]], int], int] = {}

    for action in candidates:
        removed_key = tuple(sorted(tuple(sorted(edge)) for edge in action[0]))
        damage: dict[str, int] = {}
        for length, name in (
            (3, "destroyed_triangles"),
            (4, "destroyed_cycles_4"),
            (5, "destroyed_cycles_5"),
        ):
            key = (removed_key, length)
            if key not in cache:
                cache[key] = len(
                    _cycle_signatures_containing_removed_edges(
                        graph, action, length=length
                    )
                )
            damage[name] = int(cache[key])

        reason = None
        if (
            config.max_destroyed_triangles >= 0
            and damage["destroyed_triangles"] > config.max_destroyed_triangles
        ):
            reason = "triangle_guard"
        elif (
            config.max_destroyed_cycles_4 >= 0
            and damage["destroyed_cycles_4"] > config.max_destroyed_cycles_4
        ):
            reason = "cycle4_guard"
        elif (
            config.max_destroyed_cycles_5 >= 0
            and damage["destroyed_cycles_5"] > config.max_destroyed_cycles_5
        ):
            reason = "cycle5_guard"

        if reason is not None:
            rejections[reason] = int(rejections.get(reason, 0) + 1)
            continue
        retained.append(action)
        retained_graphs[action] = candidate_graphs[action]
        damage_by_action[action] = damage

    return retained, retained_graphs, damage_by_action, {
        "num_motif_guard_rejections": int(len(candidates) - len(retained)),
        "motif_guard_rejection_reasons": dict(sorted(rejections.items())),
    }

def _select_row(
    rows: Sequence[dict[str, Any]],
    *,
    config: TopologyRefinerConfig,
    rng: np.random.Generator,
    progress: float,
) -> tuple[int | None, float, list[float], float, int]:
    """Select a candidate or STOP using greedy or relaxed softmax selection.

    Candidate score is the structural target gain, so STOP has gain zero.  In
    relaxed mode small negative gains remain eligible up to the configured
    worsening budget; this permits temporary moves away from an imperfect
    predicted target while the motif guard limits local structural damage.
    """

    improvements = np.asarray(
        [float(row["energy_improvement"]) for row in rows], dtype=np.float64
    )
    relative_improvements = np.asarray(
        [float(row["relative_energy_improvement"]) for row in rows],
        dtype=np.float64,
    )
    scores = np.concatenate([improvements, np.asarray([0.0], dtype=np.float64)])

    if config.accept_only_improving:
        eligible = improvements > float(config.min_improvement)
        eligible &= relative_improvements > float(config.min_relative_improvement)
        scores[:-1][~eligible] = -np.inf
        if np.any(np.isfinite(scores[:-1])):
            # Preserve historical greedy semantics: STOP is offered only on a
            # plateau when strict positive improvement is requested.
            scores[-1] = -np.inf
    else:
        eligible = improvements >= -float(config.max_target_worsening)
        if float(config.max_relative_target_worsening) >= 0.0:
            eligible &= (
                relative_improvements
                >= -float(config.max_relative_target_worsening)
            )
        scores[:-1][~eligible] = -np.inf
        if not config.stop_action and np.any(np.isfinite(scores[:-1])):
            scores[-1] = -np.inf

    finite = np.isfinite(scores)
    if not np.any(finite):
        # Defensive fallback: never force an invalid/wild move.
        probabilities = np.zeros_like(scores)
        probabilities[-1] = 1.0
        return None, 1.0, probabilities.tolist(), config.temperature_at(progress), 0

    normalized = scores.copy()
    if config.score_normalization == "std":
        finite_values = normalized[finite]
        scale = float(np.std(finite_values))
        if np.isfinite(scale) and scale > 1.0e-12:
            normalized[finite] /= scale

    temperature = float(config.temperature_at(progress))
    shifted = normalized.copy()
    shifted[finite] -= float(np.max(shifted[finite]))
    probabilities = np.zeros_like(scores)
    probabilities[finite] = np.exp(shifted[finite] / temperature)
    total = float(probabilities.sum())
    if not np.isfinite(total) or total <= 0.0:
        probabilities[:] = 0.0
        probabilities[-1] = 1.0
    else:
        probabilities /= total

    if config.selection in {"greedy", "argmax"}:
        best = float(np.max(normalized[finite]))
        maximizers = np.flatnonzero(
            finite & np.isclose(normalized, best, atol=1.0e-12)
        )
        selected = int(rng.choice(maximizers))
    else:
        selected = int(rng.choice(len(scores), p=probabilities))
    stop_index = len(rows)
    return (
        None if selected == stop_index else selected,
        float(probabilities[-1]),
        probabilities.tolist(),
        temperature,
        int(np.isfinite(scores[:-1]).sum()),
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
    accepted_steps = 0
    accepted_since_prediction = 0
    decision_step = 0
    prediction_calls = 0
    prediction_block = -1
    prediction_horizon = 1
    prediction_progress = 0.0
    prediction_time = 0.0

    while accepted_steps < cfg.steps:
        prediction_refreshed = False
        if (
            prediction is None
            or accepted_since_prediction >= prediction_horizon
        ):
            prediction_progress = float(
                accepted_steps / max(cfg.steps - 1, 1)
            )
            # Preserve the predictor's existing t/T convention. The separate
            # cooling progress above reaches one before the final action so the
            # annealed K schedule can attain ``final_k`` without changing the
            # time-conditioning scale seen during training.
            prediction_time = float(accepted_steps / max(cfg.steps, 1))
            prediction_horizon = cfg.prediction_horizon_at(prediction_progress)
            prediction = predictor(
                model,
                current,
                time=prediction_time,
                graphlet_basis=graphlet_basis,
                device=device,
                rng=generator,
                sample_graphlet=cfg.sample_graphlet,
            )
            prediction_calls += 1
            prediction_block += 1
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
                    **proposal_diagnostics,
                }
            )
            break

        candidates, candidate_graphs, motif_damage, motif_diagnostics = (
            _filter_candidates_by_motif_guard(
                current, candidates, candidate_graphs, config=cfg
            )
        )
        proposal_diagnostics = {**proposal_diagnostics, **motif_diagnostics}
        if not candidates:
            trace.append(
                {
                    "step": decision_step,
                    "accepted_step": accepted_steps,
                    "accepted": False,
                    "reason": "explicit_stop_motif_guard_empty",
                    "terminal_stop": True,
                    "prediction_refreshed": prediction_refreshed,
                    "prediction_calls": prediction_calls,
                    "prediction_block": prediction_block,
                    "prediction_horizon": prediction_horizon,
                    "prediction_progress": prediction_progress,
                    "prediction_time": prediction_time,
                    "inner_step": accepted_since_prediction,
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
        for row in rows:
            row.update(motif_damage.get(row["action"], {}))

        selection_progress = float(accepted_steps / max(cfg.steps - 1, 1))
        selected, stop_probability, probabilities, selection_temperature, num_eligible = _select_row(
            rows,
            config=cfg,
            rng=generator,
            progress=selection_progress,
        )
        if selected is None:
            refresh_after_plateau = bool(
                cfg.refresh_on_plateau and accepted_since_prediction > 0
            )
            trace.append(
                {
                    "step": decision_step,
                    "accepted_step": accepted_steps,
                    "accepted": False,
                    "reason": (
                        "prediction_plateau_refresh"
                        if refresh_after_plateau
                        else (
                            "explicit_stop_softmax"
                            if not cfg.accept_only_improving
                            else "explicit_stop_below_improvement_threshold"
                        )
                    ),
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
                    "selection_temperature": float(selection_temperature),
                    "num_eligible_candidates": int(num_eligible),
                    "max_target_worsening": float(cfg.max_target_worsening),
                    "max_relative_target_worsening": float(cfg.max_relative_target_worsening),
                    "current_graphlet_discrepancy": float(
                        rows[0]["current_graphlet_discrepancy"]
                    ),
                    "current_structural_discrepancy": float(
                        rows[0]["current_structural_discrepancy"]
                    ),
                    "best_energy_improvement": float(
                        max(row["energy_improvement"] for row in rows)
                    ),
                    "best_relative_energy_improvement": float(
                        max(row["relative_energy_improvement"] for row in rows)
                    ),
                    "min_improvement": float(cfg.min_improvement),
                    "min_relative_improvement": float(
                        cfg.min_relative_improvement
                    ),
                    **proposal_diagnostics,
                }
            )
            decision_step += 1
            if refresh_after_plateau:
                # The frozen prediction has reached a local plateau before its
                # scheduled K budget. Re-estimate the target from the new state
                # rather than terminating the whole generation trajectory.
                prediction = None
                continue
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
        accepted_steps += 1
        accepted_since_prediction += 1
        trace.append(
            {
                "step": decision_step,
                "accepted_step": accepted_steps,
                "accepted": True,
                "reason": (
                    "structural_softmax_swap"
                    if not cfg.accept_only_improving
                    else "structural_improving_swap"
                ),
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
                "selection_temperature": float(selection_temperature),
                "num_eligible_candidates": int(num_eligible),
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
                "relative_energy_improvement": float(
                    chosen["relative_energy_improvement"]
                ),
                "min_improvement": float(cfg.min_improvement),
                "min_relative_improvement": float(
                    cfg.min_relative_improvement
                ),
                "max_target_worsening": float(cfg.max_target_worsening),
                "max_relative_target_worsening": float(cfg.max_relative_target_worsening),
                "destroyed_triangles": int(chosen.get("destroyed_triangles", 0)),
                "destroyed_cycles_4": int(chosen.get("destroyed_cycles_4", 0)),
                "destroyed_cycles_5": int(chosen.get("destroyed_cycles_5", 0)),
                **proposal_diagnostics,
            }
        )
        decision_step += 1

    if return_trace:
        return current, trace
    return current

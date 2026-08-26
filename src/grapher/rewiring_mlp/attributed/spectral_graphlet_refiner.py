from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import networkx as nx
import numpy as np
import torch

from grapher.rewiring_mlp.attributed.data import GraphCategoryVocabulary, GraphletBasis
from grapher.rewiring_mlp.attributed.graphlet_diffusion import (
    AttributedGraphletCounts,
    GraphletLogitBridgeSchedule,
    attributed_graphlet_clr_to_simplex,
    attributed_graphlet_logit_distance,
    attributed_graphlet_simplex_from_counts,
    attributed_graphlet_simplex_to_clr,
    attributed_state_key,
    candidate_attributed_graphlet_logits_from_counts,
    extract_attributed_graphlet_simplex,
)
from grapher.rewiring_mlp.attributed.spectral import (
    attributed_laplacian_spectra,
    attributed_spectral_distance,
    attributed_spectral_scales,
    attributed_spectrum_moments,
    normalize_attributed_graph,
)
from grapher.rewiring_mlp.attributed.spectral_data import (
    AttributedSpectralExample,
    collate_attributed_spectral_examples,
)
from grapher.rewiring_mlp.attributed.spectral_model import (
    AttributedSpectralGraphletTransformerPredictor,
)
from grapher.rewiring_mlp.core.rewiring import (
    Action,
    candidate_actions_from_edge_pair,
    is_valid_action,
)
from grapher.rewiring_mlp.generic.spectral import SpectralBridgeSchedule
from grapher.rewiring_mlp.molecular.constraints import (
    bond_order,
    is_molecular_valence_feasible,
)
from grapher.rewiring_mlp.molecular.graph_io import is_valid_molecular_graph
from grapher.rewiring_mlp.molecular.typed_invariants import (
    TypedInvariant,
    extract_typed_invariant,
    typed_invariant_matches_graph,
)


@dataclass(frozen=True)
class AttributedSpectralGraphletPrediction:
    clean_spectra: np.ndarray
    current_spectra: np.ndarray
    clean_graphlet_logits: np.ndarray
    clean_graphlet_probabilities: np.ndarray
    current_graphlet_logits: np.ndarray
    current_graphlet_probabilities: np.ndarray
    graphlet_coordinate_mask: np.ndarray
    spectral_moments: dict[str, list[float]]


@dataclass(frozen=True)
class AttributedSpectralGraphletRefinerConfig:
    steps: int = 48
    proposal_budget: int = 256
    valid_candidate_budget: int = 64
    preserve_connectivity: bool = True
    selection: str = "greedy"
    temperature: float = 0.1
    min_improvement: float = 1.0e-9
    min_relative_improvement: float = 1.0e-6
    relative_improvement_epsilon: float = 1.0e-12
    reject_revisited_states: bool = True
    # Candidate energies are defined by time-dependent predictions and bridge
    # targets, so values from different reverse steps are not globally
    # comparable. Return the last accepted projected state by default.
    return_best_state: bool = False

    spectral_distance: str = "rmse"
    spectral_normalization: str = "mean_degree"
    spectral_channel_weights: tuple[float, float] = (1.0, 1.0)
    low_frequency_weight: float = 1.5
    low_frequency_cutoff: int = 4
    spectral_bridge_schedule: str = "cosine"
    spectral_min_clean_mix: float = 0.15
    spectral_max_clean_mix: float = 1.0
    spectral_bridge_power: float = 2.0

    graphlet_distance: str = "clr_rmse"
    graphlet_logit_epsilon: float = 1.0e-4
    graphlet_size_weights: dict[str, float] = field(default_factory=dict)
    graphlet_bridge_schedule: str = "cosine"
    graphlet_min_clean_mix: float = 0.10
    graphlet_max_clean_mix: float = 1.0
    graphlet_bridge_power: float = 2.0

    expand_on_plateau: bool = True
    plateau_expand_factor: float = 1.35
    max_plateau_expansions: int = 6

    guidance_weight_schedule: str = "cosine"
    spectral_weight_initial: float = 1.0
    spectral_weight_final: float = 0.25
    graphlet_weight_initial: float = 0.25
    graphlet_weight_final: float = 2.0
    guidance_weight_power: float = 1.5

    prediction_horizon_mode: str = "annealed"
    prediction_horizon_initial_k: int = 4
    prediction_horizon_final_k: int = 1
    prediction_horizon_schedule: str = "cosine"
    refresh_on_prediction_plateau: bool = True

    require_same_edge_type_pair: bool = True
    preserve_removed_edge_type: bool = True
    enforce_molecular_valence: bool = True
    molecular_allowed_bond_types: tuple[int, ...] = (1, 2, 3)
    molecular_max_valence: dict[int, float] = field(default_factory=dict)
    rdkit_candidate_check: bool = True
    rdkit_shortlist: int = 8
    require_rdkit_source_validity: bool = False

    debug_enabled: bool = False
    debug_print_every: int = 1
    debug_top_candidates: int = 3

    @property
    def spectral_bridge(self) -> SpectralBridgeSchedule:
        return SpectralBridgeSchedule(
            schedule=self.spectral_bridge_schedule,
            min_clean_mix=self.spectral_min_clean_mix,
            max_clean_mix=self.spectral_max_clean_mix,
            power=self.spectral_bridge_power,
        )

    @property
    def graphlet_bridge(self) -> GraphletLogitBridgeSchedule:
        return GraphletLogitBridgeSchedule(
            schedule=self.graphlet_bridge_schedule,
            min_clean_mix=self.graphlet_min_clean_mix,
            max_clean_mix=self.graphlet_max_clean_mix,
            power=self.graphlet_bridge_power,
        )

    def guidance_weights_at(self, progress: float) -> tuple[float, float]:
        p = float(np.clip(progress, 0.0, 1.0))
        if self.guidance_weight_schedule == "linear":
            shaped = p
        elif self.guidance_weight_schedule == "cosine":
            shaped = 0.5 - 0.5 * np.cos(np.pi * p)
        elif self.guidance_weight_schedule == "power":
            shaped = p ** self.guidance_weight_power
        else:
            raise ValueError(f"Unknown global_to_local schedule {self.guidance_weight_schedule!r}.")
        spectral = self.spectral_weight_initial + (
            self.spectral_weight_final - self.spectral_weight_initial
        ) * shaped
        graphlet = self.graphlet_weight_initial + (
            self.graphlet_weight_final - self.graphlet_weight_initial
        ) * shaped
        return float(spectral), float(graphlet)

    def prediction_horizon_at(self, progress: float) -> int:
        if self.prediction_horizon_mode == "fixed":
            return max(int(self.prediction_horizon_initial_k), 1)
        p = float(np.clip(progress, 0.0, 1.0))
        start = float(self.prediction_horizon_initial_k)
        end = float(self.prediction_horizon_final_k)
        if self.prediction_horizon_schedule == "linear":
            value = start + (end - start) * p
        elif self.prediction_horizon_schedule == "cosine":
            cooling = 0.5 * (1.0 + np.cos(np.pi * p))
            value = end + (start - end) * cooling
        elif self.prediction_horizon_schedule == "exponential":
            value = start * ((end / start) ** p)
        else:
            raise ValueError(
                f"Unknown prediction horizon schedule {self.prediction_horizon_schedule!r}."
            )
        return max(1, int(np.floor(value + 0.5)))

    @classmethod
    def from_dict(
        cls, data: dict[str, Any] | None = None
    ) -> "AttributedSpectralGraphletRefinerConfig":
        values = dict(data or {})
        mode = str(values.get("mode", "attributed_spectral_graphlet")).lower()
        if mode not in {
            "attributed_spectral_graphlet",
            "spectral_graphlet",
            "spectral_graphlet_diffusion",
        }:
            raise ValueError(
                "Attributed refiner mode must be attributed_spectral_graphlet."
            )
        spectral = dict(values.get("spectral_guidance", {}) or {})
        graphlet = dict(values.get("graphlet_guidance", {}) or {})
        global_to_local = dict(values.get("global_to_local", {}) or {})
        horizon = dict(values.get("prediction_horizon", {}) or {})
        molecular = dict(values.get("molecular", {}) or {})
        debug = dict(values.get("debug", {}) or {})
        channel_weights = spectral.get("channel_weights", [1.0, 1.0])
        if isinstance(channel_weights, Mapping):
            channel_weights = [
                channel_weights.get("topology", 1.0),
                channel_weights.get("bond_weighted", channel_weights.get("bond", 1.0)),
            ]
        if len(channel_weights) != 2:
            raise ValueError("spectral_guidance.channel_weights requires topology and bond weights.")
        size_weights = graphlet.get("size_weights", {}) or {}
        if not isinstance(size_weights, Mapping):
            raise ValueError("graphlet_guidance.size_weights must be a mapping.")
        cfg = cls(
            steps=int(values.get("steps", 48)),
            proposal_budget=int(values.get("proposal_budget", 256)),
            valid_candidate_budget=int(values.get("valid_candidate_budget", 64)),
            preserve_connectivity=bool(values.get("preserve_connectivity", True)),
            selection=str(values.get("selection", "greedy")).lower(),
            temperature=float(values.get("temperature", 0.1)),
            min_improvement=float(values.get("min_improvement", 1.0e-9)),
            min_relative_improvement=float(values.get("min_relative_improvement", 1.0e-6)),
            relative_improvement_epsilon=float(values.get("relative_improvement_epsilon", 1.0e-12)),
            reject_revisited_states=bool(values.get("reject_revisited_states", True)),
            return_best_state=bool(values.get("return_best_state", False)),
            spectral_distance=str(spectral.get("distance", "rmse")).lower(),
            spectral_normalization=str(spectral.get("normalization", "mean_degree")).lower(),
            spectral_channel_weights=(float(channel_weights[0]), float(channel_weights[1])),
            low_frequency_weight=float(spectral.get("low_frequency_weight", 1.5)),
            low_frequency_cutoff=int(spectral.get("low_frequency_cutoff", 4)),
            spectral_bridge_schedule=str(spectral.get("schedule", "cosine")).lower(),
            spectral_min_clean_mix=float(spectral.get("min_clean_mix", 0.15)),
            spectral_max_clean_mix=float(spectral.get("max_clean_mix", 1.0)),
            spectral_bridge_power=float(spectral.get("power", 2.0)),
            graphlet_distance=str(graphlet.get("distance", "clr_rmse")).lower(),
            graphlet_logit_epsilon=float(graphlet.get("logit_epsilon", 1.0e-4)),
            graphlet_size_weights={str(k): float(v) for k, v in size_weights.items()},
            graphlet_bridge_schedule=str(graphlet.get("schedule", "cosine")).lower(),
            graphlet_min_clean_mix=float(graphlet.get("min_clean_mix", 0.10)),
            graphlet_max_clean_mix=float(graphlet.get("max_clean_mix", 1.0)),
            graphlet_bridge_power=float(graphlet.get("power", 2.0)),
            expand_on_plateau=bool(values.get("expand_on_plateau", spectral.get("expand_on_plateau", True))),
            plateau_expand_factor=float(values.get("plateau_expand_factor", spectral.get("plateau_expand_factor", 1.35))),
            max_plateau_expansions=int(values.get("max_plateau_expansions", spectral.get("max_plateau_expansions", 6))),
            guidance_weight_schedule=str(global_to_local.get("schedule", "cosine")).lower(),
            spectral_weight_initial=float(global_to_local.get("spectral_initial", 1.0)),
            spectral_weight_final=float(global_to_local.get("spectral_final", 0.25)),
            graphlet_weight_initial=float(global_to_local.get("graphlet_initial", 0.25)),
            graphlet_weight_final=float(global_to_local.get("graphlet_final", 2.0)),
            guidance_weight_power=float(global_to_local.get("power", 1.5)),
            prediction_horizon_mode=str(horizon.get("mode", "annealed")).lower(),
            prediction_horizon_initial_k=int(horizon.get("initial_k", horizon.get("k", 4))),
            prediction_horizon_final_k=int(horizon.get("final_k", 1)),
            prediction_horizon_schedule=str(horizon.get("schedule", "cosine")).lower(),
            refresh_on_prediction_plateau=bool(horizon.get("refresh_on_plateau", True)),
            require_same_edge_type_pair=bool(molecular.get("require_same_edge_type_pair", True)),
            preserve_removed_edge_type=bool(molecular.get("preserve_removed_edge_type", True)),
            enforce_molecular_valence=bool(
                molecular.get(
                    "enforce_molecular_valence",
                    molecular.get("enforce_valence", True),
                )
            ),
            molecular_allowed_bond_types=tuple(int(v) for v in molecular.get("allowed_bond_types", [1, 2, 3])),
            molecular_max_valence={
                int(key): float(value)
                for key, value in dict(molecular.get("max_valence", {}) or {}).items()
            },
            rdkit_candidate_check=bool(
                molecular.get(
                    "rdkit_candidate_check",
                    molecular.get("require_rdkit_candidate", True),
                )
            ),
            rdkit_shortlist=max(
                int(
                    molecular.get(
                        "rdkit_shortlist",
                        molecular.get("rdkit_shortlist_size", 8),
                    )
                ),
                1,
            ),
            require_rdkit_source_validity=bool(molecular.get("require_rdkit_source_validity", False)),
            debug_enabled=bool(debug.get("enabled", False)),
            debug_print_every=max(int(debug.get("print_every", 1)), 1),
            debug_top_candidates=max(int(debug.get("top_candidates", 3)), 0),
        )
        if cfg.steps < 0 or cfg.proposal_budget == 0 or cfg.valid_candidate_budget == 0:
            raise ValueError("Attributed refiner steps/budgets must be nonnegative/nonzero.")
        if not cfg.preserve_connectivity:
            raise ValueError("Attributed spectral GraphER requires connectivity preservation.")
        if cfg.selection not in {"greedy", "argmax", "softmax", "sample"}:
            raise ValueError("selection must be greedy or softmax/sample.")
        if cfg.temperature <= 0:
            raise ValueError("temperature must be positive.")
        if cfg.require_same_edge_type_pair != cfg.preserve_removed_edge_type:
            raise ValueError("Strict typed rewiring requires same-type pairing and type preservation together.")
        if cfg.prediction_horizon_mode not in {"fixed", "annealed"}:
            raise ValueError("prediction_horizon.mode must be fixed or annealed.")
        if cfg.prediction_horizon_initial_k <= 0 or cfg.prediction_horizon_final_k <= 0:
            raise ValueError("Prediction horizons must be positive.")
        return cfg


def _debug(config: AttributedSpectralGraphletRefinerConfig, step: int, text: str) -> None:
    if config.debug_enabled and step % config.debug_print_every == 0:
        print(f"[GraphER/AttributedSpectralGraphlet] {text}", flush=True)


def _edge_category(
    graph: nx.Graph,
    edge: tuple[int, int],
    vocabulary: GraphCategoryVocabulary,
) -> int:
    return int(vocabulary.edge_index(graph.edges[edge[0], edge[1]]))


def _apply_same_type_action(
    graph: nx.Graph,
    action: Action,
    vocabulary: GraphCategoryVocabulary,
) -> nx.Graph:
    removed, added = action
    categories = [_edge_category(graph, edge, vocabulary) for edge in removed]
    if len(set(categories)) != 1:
        raise ValueError("Attributed rewiring must remove two edges of the same type.")
    category = categories[0]
    edge_value = vocabulary.edge_value(category)
    source_attributes = dict(graph.edges[removed[0]])
    if vocabulary.edge_attribute:
        source_attributes[vocabulary.edge_attribute] = edge_value
    if vocabulary.edge_attribute == "bond_type":
        source_attributes["bond_order"] = bond_order(int(edge_value))
    candidate = graph.copy()
    for edge in removed:
        candidate.remove_edge(*edge)
    for edge in added:
        candidate.add_edge(*edge, **source_attributes)
    return candidate


def _propose_same_type_candidates(
    graph: nx.Graph,
    *,
    vocabulary: GraphCategoryVocabulary,
    config: AttributedSpectralGraphletRefinerConfig,
    rng: np.random.Generator,
    excluded_states: set[tuple[Any, ...]] | None,
) -> tuple[list[Action], dict[Action, nx.Graph], dict[str, Any]]:
    groups: dict[int, list[tuple[int, int]]] = {}
    for u, v, data in graph.edges(data=True):
        category = int(vocabulary.edge_index(data))
        groups.setdefault(category, []).append((min(int(u), int(v)), max(int(u), int(v))))
    groups = {key: value for key, value in groups.items() if len(value) >= 2}
    if not groups:
        return [], {}, {"num_proposals": 0, "num_valid_candidates": 0, "candidate_pass_rate": 0.0, "candidate_rejection_reasons": {"no_same_type_pair": 1}}
    group_keys = list(groups)
    group_weights = np.asarray(
        [len(groups[key]) * (len(groups[key]) - 1) / 2 for key in group_keys],
        dtype=np.float64,
    )
    group_weights /= group_weights.sum()
    target = int(config.valid_candidate_budget)
    proposal_limit = int(config.proposal_budget)
    if target < 0 or proposal_limit < 0:
        proposal_limit = sum(len(edges) * (len(edges) - 1) for edges in groups.values())
        target = proposal_limit
    seen: set[Action] = set()
    candidates: list[Action] = []
    candidate_graphs: dict[Action, nx.Graph] = {}
    rejections: dict[str, int] = {}
    attempts = 0
    max_attempts = max(100, proposal_limit * 50)
    while attempts < max_attempts and len(seen) < proposal_limit and len(candidates) < target:
        attempts += 1
        key = group_keys[int(rng.choice(len(group_keys), p=group_weights))]
        edges = groups[key]
        indices = rng.choice(len(edges), size=2, replace=False)
        actions = candidate_actions_from_edge_pair(edges[int(indices[0])], edges[int(indices[1])])
        rng.shuffle(actions)
        for action in actions:
            if action in seen:
                continue
            seen.add(action)
            if not is_valid_action(graph, action, preserve_connectivity=config.preserve_connectivity):
                rejections["topology_or_connectivity"] = rejections.get("topology_or_connectivity", 0) + 1
                continue
            try:
                candidate = _apply_same_type_action(graph, action, vocabulary)
            except (KeyError, TypeError, ValueError):
                rejections["attribute_assignment"] = rejections.get("attribute_assignment", 0) + 1
                continue
            if config.enforce_molecular_valence and not is_molecular_valence_feasible(
                candidate,
                allowed_atom_types=vocabulary.node_values,
                allowed_bond_types=config.molecular_allowed_bond_types,
                max_valence=config.molecular_max_valence or None,
            ):
                rejections["molecular_valence"] = rejections.get("molecular_valence", 0) + 1
                continue
            if excluded_states is not None:
                state = attributed_state_key(
                    candidate,
                    node_attribute=str(vocabulary.node_attribute),
                    edge_attribute=str(vocabulary.edge_attribute),
                )
                if state in excluded_states:
                    rejections["revisited_state"] = rejections.get("revisited_state", 0) + 1
                    continue
            candidates.append(action)
            candidate_graphs[action] = candidate
            if len(candidates) >= target:
                break
    proposals = len(seen)
    return candidates, candidate_graphs, {
        "num_proposals": proposals,
        "num_valid_candidates": len(candidates),
        "candidate_pass_rate": float(len(candidates) / max(proposals, 1)),
        "candidate_rejection_reasons": rejections,
    }


@torch.no_grad()
def predict_clean_attributed_summaries(
    model: AttributedSpectralGraphletTransformerPredictor,
    graph: nx.Graph,
    *,
    conditioning_graph: nx.Graph,
    source_spectra: np.ndarray,
    source_graphlet_probabilities: np.ndarray,
    source_graphlet_logits: np.ndarray,
    current_graphlet_counts: AttributedGraphletCounts,
    vocabulary: GraphCategoryVocabulary,
    graphlet_basis: GraphletBasis,
    time: float,
    device: torch.device | str,
    graphlet_logit_epsilon: float,
) -> AttributedSpectralGraphletPrediction:
    model.eval()
    current_spectra = attributed_laplacian_spectra(
        graph, edge_attribute=str(vocabulary.edge_attribute)
    )
    current_prob, current_mask = attributed_graphlet_simplex_from_counts(
        current_graphlet_counts,
        num_nodes=graph.number_of_nodes(),
        graphlet_basis=graphlet_basis,
    )
    current_logits = attributed_graphlet_simplex_to_clr(
        current_prob,
        graphlet_basis=graphlet_basis,
        epsilon=graphlet_logit_epsilon,
        coordinate_mask=current_mask,
    )
    example = AttributedSpectralExample(
        conditioning_graph=conditioning_graph,
        time=float(time),
        current_spectra=current_spectra.astype(np.float32),
        source_spectra=np.asarray(source_spectra, dtype=np.float32),
        clean_spectra_target=np.zeros_like(current_spectra, dtype=np.float32),
        current_graphlet_probabilities=current_prob.astype(np.float32),
        source_graphlet_probabilities=np.asarray(source_graphlet_probabilities, dtype=np.float32),
        clean_graphlet_probabilities_target=np.zeros_like(current_prob, dtype=np.float32),
        current_graphlet_logits=current_logits.astype(np.float32),
        source_graphlet_logits=np.asarray(source_graphlet_logits, dtype=np.float32),
        clean_graphlet_logits_target=np.zeros_like(current_logits, dtype=np.float32),
        graphlet_coordinate_mask=current_mask.astype(np.bool_),
    )
    batch = collate_attributed_spectral_examples([example], vocabulary).to(device)
    outputs = model(batch)
    clean_spectra = outputs["clean_spectra"][0, :, : graph.number_of_nodes()].detach().cpu().numpy().astype(np.float64)
    clean_logits = outputs["clean_graphlet_logits"][0].detach().cpu().numpy().astype(np.float64)
    clean_prob = outputs["clean_graphlet_probabilities"][0].detach().cpu().numpy().astype(np.float64)
    return AttributedSpectralGraphletPrediction(
        clean_spectra=clean_spectra,
        current_spectra=current_spectra,
        clean_graphlet_logits=clean_logits,
        clean_graphlet_probabilities=clean_prob,
        current_graphlet_logits=current_logits,
        current_graphlet_probabilities=current_prob,
        graphlet_coordinate_mask=current_mask,
        spectral_moments=attributed_spectrum_moments(clean_spectra),
    )


def _prepare_candidate_states(
    graph: nx.Graph,
    candidates: Sequence[Action],
    *,
    candidate_graphs: dict[Action, nx.Graph],
    current_counts: AttributedGraphletCounts,
    vocabulary: GraphCategoryVocabulary,
    graphlet_basis: GraphletBasis,
    config: AttributedSpectralGraphletRefinerConfig,
) -> dict[str, Any]:
    current_spectra = attributed_laplacian_spectra(
        graph, edge_attribute=str(vocabulary.edge_attribute)
    )
    scales = attributed_spectral_scales(
        graph,
        mode=config.spectral_normalization,
        edge_attribute=str(vocabulary.edge_attribute),
    )
    current_prob, current_mask = attributed_graphlet_simplex_from_counts(
        current_counts,
        num_nodes=graph.number_of_nodes(),
        graphlet_basis=graphlet_basis,
    )
    current_logits = attributed_graphlet_simplex_to_clr(
        current_prob,
        graphlet_basis=graphlet_basis,
        epsilon=config.graphlet_logit_epsilon,
        coordinate_mask=current_mask,
    )
    rows: list[dict[str, Any]] = []
    for action in candidates:
        candidate = candidate_graphs[action]
        candidate_logits, candidate_prob, candidate_counts = (
            candidate_attributed_graphlet_logits_from_counts(
                graph,
                candidate,
                action,
                current_counts=current_counts,
                graphlet_basis=graphlet_basis,
                epsilon=config.graphlet_logit_epsilon,
            )
        )
        rows.append(
            {
                "action": action,
                "candidate_graph": candidate,
                "candidate_spectra": attributed_laplacian_spectra(
                    candidate, edge_attribute=str(vocabulary.edge_attribute)
                ),
                "candidate_graphlet_logits": candidate_logits,
                "candidate_graphlet_probabilities": candidate_prob,
                "candidate_graphlet_counts": candidate_counts,
                "rdkit_valid": None,
            }
        )
    return {
        "current_spectra": current_spectra,
        "spectral_scales": scales,
        "current_graphlet_logits": current_logits,
        "current_graphlet_probabilities": current_prob,
        "current_graphlet_mask": current_mask,
        "rows": rows,
    }


def _score_prepared(
    prepared: Mapping[str, Any],
    *,
    graphlet_basis: GraphletBasis,
    clean_spectra: np.ndarray,
    next_spectra: np.ndarray,
    clean_graphlet_logits: np.ndarray,
    next_graphlet_logits: np.ndarray,
    graphlet_mask: np.ndarray,
    spectral_weight: float,
    graphlet_weight: float,
    config: AttributedSpectralGraphletRefinerConfig,
) -> list[dict[str, Any]]:
    current_spectra = np.asarray(prepared["current_spectra"], dtype=np.float64)
    current_logits = np.asarray(prepared["current_graphlet_logits"], dtype=np.float64)
    scales = np.asarray(prepared["spectral_scales"], dtype=np.float64)
    current_spec, current_spec_channels = attributed_spectral_distance(
        current_spectra,
        next_spectra,
        scales=scales,
        metric=config.spectral_distance,
        channel_weights=config.spectral_channel_weights,
        low_frequency_weight=config.low_frequency_weight,
        low_frequency_cutoff=config.low_frequency_cutoff,
    )
    current_clean_spec, _ = attributed_spectral_distance(
        current_spectra,
        clean_spectra,
        scales=scales,
        metric=config.spectral_distance,
        channel_weights=config.spectral_channel_weights,
        low_frequency_weight=config.low_frequency_weight,
        low_frequency_cutoff=config.low_frequency_cutoff,
    )
    current_graphlet = attributed_graphlet_logit_distance(
        current_logits,
        next_graphlet_logits,
        graphlet_basis=graphlet_basis,
        coordinate_mask=graphlet_mask,
        metric=config.graphlet_distance,
        size_weights=config.graphlet_size_weights,
    )
    current_clean_graphlet = attributed_graphlet_logit_distance(
        current_logits,
        clean_graphlet_logits,
        graphlet_basis=graphlet_basis,
        coordinate_mask=graphlet_mask,
        metric=config.graphlet_distance,
        size_weights=config.graphlet_size_weights,
    )
    current_energy = spectral_weight * current_spec + graphlet_weight * current_graphlet
    rows: list[dict[str, Any]] = []
    for cached in prepared["rows"]:
        candidate_spec, candidate_channels = attributed_spectral_distance(
            cached["candidate_spectra"],
            next_spectra,
            scales=scales,
            metric=config.spectral_distance,
            channel_weights=config.spectral_channel_weights,
            low_frequency_weight=config.low_frequency_weight,
            low_frequency_cutoff=config.low_frequency_cutoff,
        )
        candidate_clean_spec, _ = attributed_spectral_distance(
            cached["candidate_spectra"],
            clean_spectra,
            scales=scales,
            metric=config.spectral_distance,
            channel_weights=config.spectral_channel_weights,
            low_frequency_weight=config.low_frequency_weight,
            low_frequency_cutoff=config.low_frequency_cutoff,
        )
        candidate_graphlet = attributed_graphlet_logit_distance(
            cached["candidate_graphlet_logits"],
            next_graphlet_logits,
            graphlet_basis=graphlet_basis,
            coordinate_mask=graphlet_mask,
            metric=config.graphlet_distance,
            size_weights=config.graphlet_size_weights,
        )
        candidate_clean_graphlet = attributed_graphlet_logit_distance(
            cached["candidate_graphlet_logits"],
            clean_graphlet_logits,
            graphlet_basis=graphlet_basis,
            coordinate_mask=graphlet_mask,
            metric=config.graphlet_distance,
            size_weights=config.graphlet_size_weights,
        )
        energy = spectral_weight * candidate_spec + graphlet_weight * candidate_graphlet
        gain = float(current_energy - energy)
        row = dict(cached)
        # Keep a reference to the candidate-state cache. Plateau expansion
        # changes only the continuous denoising target, so RDKit validity and
        # all discrete candidate summaries must be reused rather than
        # recomputed.
        row["_candidate_cache"] = cached
        row.update(
            {
                "current_energy": float(current_energy),
                "candidate_energy": float(energy),
                "energy_improvement": gain,
                "relative_energy_improvement": float(
                    gain / max(abs(current_energy), config.relative_improvement_epsilon)
                ),
                "spectral_gain": float(current_spec - candidate_spec),
                "topology_spectral_gain": float(
                    current_spec_channels[0] - candidate_channels[0]
                ),
                "bond_spectral_gain": float(
                    current_spec_channels[1] - candidate_channels[1]
                ),
                "graphlet_gain": float(current_graphlet - candidate_graphlet),
                "clean_spectral_gain": float(current_clean_spec - candidate_clean_spec),
                "clean_graphlet_gain": float(current_clean_graphlet - candidate_clean_graphlet),
                "spectral_projection_residual": float(candidate_spec),
                "topology_spectral_projection_residual": float(candidate_channels[0]),
                "bond_spectral_projection_residual": float(candidate_channels[1]),
                "graphlet_projection_residual": float(candidate_graphlet),
                "projection_residual": float(energy),
                "current_topology_spectral_residual": float(current_spec_channels[0]),
                "current_bond_spectral_residual": float(current_spec_channels[1]),
            }
        )
        rows.append(row)
    return rows


def _choose_candidate(
    rows: list[dict[str, Any]],
    *,
    config: AttributedSpectralGraphletRefinerConfig,
    rng: np.random.Generator,
) -> tuple[int | None, dict[str, int]]:
    eligible = [
        index
        for index, row in enumerate(rows)
        if float(row["energy_improvement"]) > config.min_improvement
        and float(row["relative_energy_improvement"]) > config.min_relative_improvement
    ]
    if not eligible:
        return None, {"rdkit_checked": 0, "rdkit_rejected": 0}
    eligible.sort(key=lambda index: float(rows[index]["energy_improvement"]), reverse=True)
    shortlist = eligible[: config.rdkit_shortlist]
    checked = rejected = 0
    valid: list[int] = []
    for index in shortlist:
        row = rows[index]
        if config.rdkit_candidate_check:
            checked += 1
            cache = row.get("_candidate_cache", row)
            if cache.get("rdkit_valid") is None:
                cache["rdkit_valid"] = bool(
                    is_valid_molecular_graph(row["candidate_graph"])
                )
            row["rdkit_valid"] = cache["rdkit_valid"]
            if not bool(cache["rdkit_valid"]):
                rejected += 1
                continue
        valid.append(index)
    if not valid:
        return None, {"rdkit_checked": checked, "rdkit_rejected": rejected}
    if config.selection in {"greedy", "argmax"}:
        best_gain = max(float(rows[index]["energy_improvement"]) for index in valid)
        maxima = [index for index in valid if np.isclose(float(rows[index]["energy_improvement"]), best_gain)]
        selected = int(rng.choice(maxima))
    else:
        scores = np.asarray([float(rows[index]["energy_improvement"]) for index in valid], dtype=np.float64)
        scores -= scores.max()
        probabilities = np.exp(scores / config.temperature)
        probabilities /= probabilities.sum()
        selected = valid[int(rng.choice(len(valid), p=probabilities))]
    return selected, {"rdkit_checked": checked, "rdkit_rejected": rejected}


def refine_attributed_graph_with_spectral_graphlet_diffusion(
    model: AttributedSpectralGraphletTransformerPredictor,
    graph: nx.Graph,
    *,
    vocabulary: GraphCategoryVocabulary,
    graphlet_basis: GraphletBasis,
    config: AttributedSpectralGraphletRefinerConfig | dict[str, Any] | None = None,
    device: torch.device | str = "cpu",
    rng: np.random.Generator | None = None,
    return_trace: bool = False,
    debug_context: str = "",
) -> nx.Graph | tuple[nx.Graph, list[dict[str, Any]]]:
    cfg = config if isinstance(config, AttributedSpectralGraphletRefinerConfig) else AttributedSpectralGraphletRefinerConfig.from_dict(config)
    generator = rng if rng is not None else np.random.default_rng(0)
    current = normalize_attributed_graph(graph)
    conditioning_graph = current.copy()
    invariant = extract_typed_invariant(
        current,
        edge_types=vocabulary.edge_values,
        node_attribute=str(vocabulary.node_attribute),
        edge_attribute=str(vocabulary.edge_attribute),
    )
    if cfg.require_rdkit_source_validity and not is_valid_molecular_graph(current):
        raise ValueError("Attributed source graph failed RDKit sanitization.")
    source_spectra = attributed_laplacian_spectra(
        conditioning_graph, edge_attribute=str(vocabulary.edge_attribute)
    )
    source_prob, source_mask, source_counts = extract_attributed_graphlet_simplex(
        conditioning_graph, graphlet_basis=graphlet_basis
    )
    source_logits = attributed_graphlet_simplex_to_clr(
        source_prob,
        graphlet_basis=graphlet_basis,
        epsilon=cfg.graphlet_logit_epsilon,
        coordinate_mask=source_mask,
    )
    current_counts = {key: dict(value) for key, value in source_counts.items()}
    visited = {
        attributed_state_key(
            current,
            node_attribute=str(vocabulary.node_attribute),
            edge_attribute=str(vocabulary.edge_attribute),
        )
    }
    trace: list[dict[str, Any]] = []
    prediction: AttributedSpectralGraphletPrediction | None = None
    prediction_calls = 0
    accepted_steps = 0
    accepted_since_prediction = 0
    horizon = 1
    decision_step = 0
    best_graph = current.copy()
    best_energy = float("inf")
    prefix = f" {debug_context}" if debug_context else ""

    while accepted_steps < cfg.steps:
        progress = float(accepted_steps / max(cfg.steps - 1, 1))
        prediction_refreshed = False
        if prediction is None or accepted_since_prediction >= horizon:
            horizon = cfg.prediction_horizon_at(progress)
            prediction = predict_clean_attributed_summaries(
                model,
                current,
                conditioning_graph=conditioning_graph,
                source_spectra=source_spectra,
                source_graphlet_probabilities=source_prob,
                source_graphlet_logits=source_logits,
                current_graphlet_counts=current_counts,
                vocabulary=vocabulary,
                graphlet_basis=graphlet_basis,
                time=progress,
                device=device,
                graphlet_logit_epsilon=cfg.graphlet_logit_epsilon,
            )
            prediction_calls += 1
            accepted_since_prediction = 0
            prediction_refreshed = True
            sw, gw = cfg.guidance_weights_at(progress)
            _debug(
                cfg,
                decision_step,
                f"{prefix} prediction_refresh call={prediction_calls} accepted={accepted_steps}/{cfg.steps} "
                f"progress={progress:.4f} horizon={horizon} weights=(spectral={sw:.3f},graphlet={gw:.3f})",
            )

        assert prediction is not None
        spectral_weight, graphlet_weight = cfg.guidance_weights_at(progress)
        current_spectra = attributed_laplacian_spectra(
            current, edge_attribute=str(vocabulary.edge_attribute)
        )
        current_prob, current_mask = attributed_graphlet_simplex_from_counts(
            current_counts,
            num_nodes=current.number_of_nodes(),
            graphlet_basis=graphlet_basis,
        )
        current_logits = attributed_graphlet_simplex_to_clr(
            current_prob,
            graphlet_basis=graphlet_basis,
            epsilon=cfg.graphlet_logit_epsilon,
            coordinate_mask=current_mask,
        )
        spectral_mix = cfg.spectral_bridge.clean_mix_for_step(
            accepted_step=accepted_steps, total_steps=max(cfg.steps, 1)
        )
        graphlet_mix = cfg.graphlet_bridge.clean_mix_for_step(
            accepted_step=accepted_steps, total_steps=max(cfg.steps, 1)
        )
        candidates, candidate_graphs, proposal_diag = _propose_same_type_candidates(
            current,
            vocabulary=vocabulary,
            config=cfg,
            rng=generator,
            excluded_states=visited if cfg.reject_revisited_states else None,
        )
        if not candidates:
            trace.append(
                {
                    "step": decision_step,
                    "accepted": False,
                    "reason": "no_valid_same_type_candidates",
                    "prediction_calls": prediction_calls,
                    **proposal_diag,
                }
            )
            break
        prepared = _prepare_candidate_states(
            current,
            candidates,
            candidate_graphs=candidate_graphs,
            current_counts=current_counts,
            vocabulary=vocabulary,
            graphlet_basis=graphlet_basis,
            config=cfg,
        )

        expansion = 0
        rows: list[dict[str, Any]] = []
        selected: int | None = None
        rdkit_diag = {"rdkit_checked": 0, "rdkit_rejected": 0}
        while True:
            next_spectra = np.stack(
                [
                    cfg.spectral_bridge.target(
                        current_spectra[channel], prediction.clean_spectra[channel], spectral_mix
                    )
                    for channel in range(2)
                ]
            )
            next_graphlet_logits = cfg.graphlet_bridge.target(
                current_logits, prediction.clean_graphlet_logits, graphlet_mix
            )
            rows = _score_prepared(
                prepared,
                graphlet_basis=graphlet_basis,
                clean_spectra=prediction.clean_spectra,
                next_spectra=next_spectra,
                clean_graphlet_logits=prediction.clean_graphlet_logits,
                next_graphlet_logits=next_graphlet_logits,
                graphlet_mask=current_mask,
                spectral_weight=spectral_weight,
                graphlet_weight=graphlet_weight,
                config=cfg,
            )
            selected, rdkit_diag = _choose_candidate(rows, config=cfg, rng=generator)
            if selected is not None:
                break
            if (
                not cfg.expand_on_plateau
                or expansion >= cfg.max_plateau_expansions
                or (
                    spectral_mix >= cfg.spectral_max_clean_mix - 1.0e-12
                    and graphlet_mix >= cfg.graphlet_max_clean_mix - 1.0e-12
                )
            ):
                break
            spectral_mix = min(
                cfg.spectral_max_clean_mix,
                max(spectral_mix * cfg.plateau_expand_factor, spectral_mix + 1.0e-6),
            )
            graphlet_mix = min(
                cfg.graphlet_max_clean_mix,
                max(graphlet_mix * cfg.plateau_expand_factor, graphlet_mix + 1.0e-6),
            )
            expansion += 1

        if selected is None:
            refresh = cfg.refresh_on_prediction_plateau and accepted_since_prediction > 0
            trace.append(
                {
                    "step": decision_step,
                    "accepted": False,
                    "reason": "prediction_plateau_refresh" if refresh else "below_improvement_threshold",
                    "prediction_calls": prediction_calls,
                    "bridge_expansions": expansion,
                    **proposal_diag,
                    **rdkit_diag,
                }
            )
            decision_step += 1
            if refresh:
                prediction = None
                continue
            break

        chosen = rows[selected]
        candidate = chosen["candidate_graph"]
        if not typed_invariant_matches_graph(candidate, invariant):
            raise AssertionError("Attributed rewiring changed the indexed typed-degree invariant.")
        if cfg.preserve_connectivity and candidate.number_of_nodes() > 1 and not nx.is_connected(candidate):
            raise AssertionError("Attributed rewiring broke connectivity.")
        current = candidate
        current_counts = chosen["candidate_graphlet_counts"]
        visited.add(
            attributed_state_key(
                current,
                node_attribute=str(vocabulary.node_attribute),
                edge_attribute=str(vocabulary.edge_attribute),
            )
        )
        accepted_steps += 1
        accepted_since_prediction += 1
        if float(chosen["candidate_energy"]) < best_energy:
            best_energy = float(chosen["candidate_energy"])
            best_graph = current.copy()
        _debug(
            cfg,
            decision_step,
            f"{prefix} ACCEPT accepted_step={accepted_steps}/{cfg.steps} gain={chosen['energy_improvement']:.6f} "
            f"spec_gain={chosen['spectral_gain']:.6f} graphlet_gain={chosen['graphlet_gain']:.6f} "
            f"residual={chosen['projection_residual']:.6f}",
        )
        if cfg.debug_enabled and cfg.debug_top_candidates:
            ranked = sorted(rows, key=lambda row: float(row["energy_improvement"]), reverse=True)
            for rank, row in enumerate(ranked[: cfg.debug_top_candidates], start=1):
                _debug(
                    cfg,
                    decision_step,
                    f"{prefix} candidate_rank={rank} gain={row['energy_improvement']:.6f} "
                    f"top_spec={row['topology_spectral_projection_residual']:.6f} "
                    f"bond_spec={row['bond_spectral_projection_residual']:.6f} "
                    f"graphlet={row['graphlet_projection_residual']:.6f}",
                )
        trace.append(
            {
                "step": decision_step,
                "accepted_step": accepted_steps,
                "accepted": True,
                "reason": "attributed_spectral_graphlet_denoising_swap",
                "action": chosen["action"],
                "prediction_refreshed": prediction_refreshed,
                "prediction_calls": prediction_calls,
                "prediction_horizon": horizon,
                "progress": progress,
                "spectral_weight": spectral_weight,
                "graphlet_weight": graphlet_weight,
                "spectral_clean_mix": spectral_mix,
                "graphlet_clean_mix": graphlet_mix,
                "bridge_expansions": expansion,
                "energy_improvement": float(chosen["energy_improvement"]),
                "relative_energy_improvement": float(chosen["relative_energy_improvement"]),
                "spectral_gain": float(chosen["spectral_gain"]),
                "topology_spectral_gain": float(chosen["topology_spectral_gain"]),
                "bond_spectral_gain": float(chosen["bond_spectral_gain"]),
                "graphlet_gain": float(chosen["graphlet_gain"]),
                "clean_spectral_gain": float(chosen["clean_spectral_gain"]),
                "clean_graphlet_gain": float(chosen["clean_graphlet_gain"]),
                "projection_residual": float(chosen["projection_residual"]),
                "spectral_projection_residual": float(chosen["spectral_projection_residual"]),
                "topology_spectral_projection_residual": float(chosen["topology_spectral_projection_residual"]),
                "bond_spectral_projection_residual": float(chosen["bond_spectral_projection_residual"]),
                "graphlet_projection_residual": float(chosen["graphlet_projection_residual"]),
                **proposal_diag,
                **rdkit_diag,
            }
        )
        decision_step += 1

    result = best_graph if cfg.return_best_state and accepted_steps > 0 else current
    if not typed_invariant_matches_graph(result, invariant):
        raise AssertionError("Final attributed graph does not preserve the source typed invariant.")
    if return_trace:
        return result, trace
    return result


__all__ = [
    "AttributedSpectralGraphletPrediction",
    "AttributedSpectralGraphletRefinerConfig",
    "predict_clean_attributed_summaries",
    "refine_attributed_graph_with_spectral_graphlet_diffusion",
]

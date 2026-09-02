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
    batched_attributed_laplacian_spectra,
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
    canonical_edge,
)
from grapher.rewiring_mlp.generic.rewiring import _fast_valid_action
from grapher.rewiring_mlp.generic.spectral import SpectralBridgeSchedule
from grapher.rewiring_mlp.molecular.constraints import (
    bond_order,
    is_molecular_valence_feasible,
)
from grapher.rewiring_mlp.molecular.graph_io import is_valid_molecular_graph
from grapher.rewiring_mlp.molecular.typed_invariants import (
    AttributedRewiringInvariant,
    TypedInvariant,
    attributed_rewiring_invariant_matches_graph,
    extract_attributed_rewiring_invariant,
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
class AttributedInvariantSummaryPrediction:
    clean_spectra: np.ndarray
    clean_graphlet_logits: np.ndarray
    clean_graphlet_probabilities: np.ndarray
    graphlet_coordinate_mask: np.ndarray
    spectral_moments: dict[str, list[float]]


@dataclass(frozen=True)
class AttributedSpectralGraphletRefinerConfig:
    steps: int = 48
    proposal_budget: int = 256
    valid_candidate_budget: int = 64
    adaptive_candidate_search: bool = False
    initial_proposal_budget: int = 256
    initial_valid_candidate_budget: int = 64
    final_proposal_budget: int = 256
    final_valid_candidate_budget: int = 64
    candidate_budget_schedule: str = "cosine"
    candidate_budget_power: float = 1.0
    spectrum_backend: str = "auto"
    spectrum_batch_size: int = 256
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

    # Attributed action family.  The default revised kernel may select two
    # edges of different categories and reassign the two removed categories
    # across either topological reconnection.  This preserves indexed ordinary
    # degrees, node categories, and global edge-category counts while allowing
    # per-node typed degrees / weighted valence to move subject to validity.
    require_same_edge_type_pair: bool = False
    preserve_removed_edge_type: bool = False  # legacy strict-mode alias
    preserve_global_edge_type_counts: bool = True
    enumerate_edge_type_permutations: bool = True
    preserve_node_types: bool = True
    preserve_ordinary_degree: bool = True
    preserve_typed_degree: bool = False
    preserve_weighted_valence: bool = False
    enforce_molecular_valence: bool = True
    molecular_allowed_bond_types: tuple[int, ...] = (1, 2, 3)
    molecular_max_valence: dict[int, float] = field(default_factory=dict)
    rdkit_candidate_check: bool = True
    rdkit_infer_projected_formal_charges: bool = False
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

    def candidate_budgets_at(self, progress: float) -> tuple[int, int]:
        if not self.adaptive_candidate_search:
            return int(self.proposal_budget), int(self.valid_candidate_budget)
        p = float(np.clip(progress, 0.0, 1.0))
        if self.candidate_budget_schedule == "linear":
            shaped = p
        elif self.candidate_budget_schedule == "cosine":
            shaped = 0.5 - 0.5 * np.cos(np.pi * p)
        elif self.candidate_budget_schedule == "power":
            shaped = p ** float(self.candidate_budget_power)
        else:
            raise ValueError(
                f"Unknown candidate_search.schedule {self.candidate_budget_schedule!r}."
            )
        proposal = int(round(
            self.initial_proposal_budget
            + (self.final_proposal_budget - self.initial_proposal_budget) * shaped
        ))
        valid = int(round(
            self.initial_valid_candidate_budget
            + (self.final_valid_candidate_budget - self.initial_valid_candidate_budget) * shaped
        ))
        # The legacy top-level budgets remain hard maxima.
        if self.proposal_budget > 0:
            proposal = min(proposal, int(self.proposal_budget))
        if self.valid_candidate_budget > 0:
            valid = min(valid, int(self.valid_candidate_budget))
        return max(proposal, 1), max(valid, 1)

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
        candidate_search = dict(values.get("candidate_search", {}) or {})
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
            adaptive_candidate_search=bool(candidate_search.get("adaptive", False)),
            initial_proposal_budget=int(candidate_search.get("initial_proposal_budget", values.get("proposal_budget", 256))),
            initial_valid_candidate_budget=int(candidate_search.get("initial_valid_candidate_budget", values.get("valid_candidate_budget", 64))),
            final_proposal_budget=int(candidate_search.get("final_proposal_budget", values.get("proposal_budget", 256))),
            final_valid_candidate_budget=int(candidate_search.get("final_valid_candidate_budget", values.get("valid_candidate_budget", 64))),
            candidate_budget_schedule=str(candidate_search.get("schedule", "cosine")).lower(),
            candidate_budget_power=float(candidate_search.get("power", 1.0)),
            spectrum_backend=str(candidate_search.get("spectrum_backend", "auto")).lower(),
            spectrum_batch_size=max(int(candidate_search.get("spectrum_batch_size", 256)), 1),
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
            require_same_edge_type_pair=bool(molecular.get("require_same_edge_type_pair", False)),
            preserve_removed_edge_type=bool(molecular.get("preserve_removed_edge_type", False)),
            preserve_global_edge_type_counts=bool(
                molecular.get("preserve_global_edge_type_counts", True)
            ),
            enumerate_edge_type_permutations=bool(
                molecular.get("enumerate_edge_type_permutations", True)
            ),
            preserve_node_types=bool(molecular.get("preserve_node_types", True)),
            preserve_ordinary_degree=bool(molecular.get("preserve_ordinary_degree", True)),
            preserve_typed_degree=bool(molecular.get("preserve_typed_degree", False)),
            preserve_weighted_valence=bool(
                molecular.get("preserve_weighted_valence", False)
            ),
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
            rdkit_infer_projected_formal_charges=bool(
                molecular.get("rdkit_infer_projected_formal_charges", False)
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
        if not cfg.preserve_global_edge_type_counts:
            raise ValueError(
                "Attributed GraphER currently preserves the global edge-category histogram; "
                "preserve_global_edge_type_counts must be true."
            )
        if cfg.preserve_typed_degree and not cfg.require_same_edge_type_pair:
            raise ValueError(
                "Per-node typed-degree preservation requires same-edge-type pairing. "
                "Disable preserve_typed_degree to use bond-reassigning cross-type swaps."
            )
        if cfg.preserve_weighted_valence and not cfg.require_same_edge_type_pair:
            raise ValueError(
                "Exact per-node weighted-valence preservation requires same-edge-type pairing. "
                "Use enforce_molecular_valence for the revised cross-type kernel."
            )
        if not cfg.preserve_node_types or not cfg.preserve_ordinary_degree:
            raise ValueError(
                "The implemented attributed rewiring kernel always preserves node types and "
                "indexed ordinary degrees; both preservation flags must remain true."
            )
        if cfg.prediction_horizon_mode not in {"fixed", "annealed"}:
            raise ValueError("prediction_horizon.mode must be fixed or annealed.")
        if cfg.prediction_horizon_initial_k <= 0 or cfg.prediction_horizon_final_k <= 0:
            raise ValueError("Prediction horizons must be positive.")
        if cfg.candidate_budget_schedule not in {"linear", "cosine", "power"}:
            raise ValueError("candidate_search.schedule must be linear, cosine, or power.")
        if cfg.candidate_budget_power <= 0.0:
            raise ValueError("candidate_search.power must be positive.")
        if cfg.spectrum_backend not in {"auto", "torch", "numpy", "np"}:
            raise ValueError("candidate_search.spectrum_backend must be auto, torch, or numpy.")
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


@dataclass(frozen=True)
class AttributedRewireAction:
    """A double-edge topology move plus an assignment of edge categories.

    ``topology_action`` fixes the two removed and two inserted unordered edges.
    ``removed_edge_categories`` and ``added_edge_categories`` are category
    indices aligned with the canonical edge order inside that topology action.
    The revised kernel requires the two category multisets to be identical, so
    global edge-category counts are preserved even when the two removed edges
    have different types.
    """

    topology_action: Action
    removed_edge_categories: tuple[int, int]
    added_edge_categories: tuple[int, int]

    def __post_init__(self) -> None:
        if len(self.removed_edge_categories) != 2 or len(self.added_edge_categories) != 2:
            raise ValueError("Attributed rewiring actions require two removed and two added categories.")
        if sorted(self.removed_edge_categories) != sorted(self.added_edge_categories):
            raise ValueError("Attributed rewiring must preserve the global edge-category multiset.")

    @property
    def removed(self):
        return self.topology_action[0]

    @property
    def added(self):
        return self.topology_action[1]


def _attributed_action_trace(
    action: AttributedRewireAction,
    vocabulary: GraphCategoryVocabulary,
) -> dict[str, Any]:
    return {
        "removed": [list(edge) for edge in action.removed],
        "added": [list(edge) for edge in action.added],
        "removed_edge_types": [
            vocabulary.edge_value(category) for category in action.removed_edge_categories
        ],
        "added_edge_types": [
            vocabulary.edge_value(category) for category in action.added_edge_categories
        ],
    }


def _edge_attribute_template(
    graph: nx.Graph,
    edge: tuple[int, int],
    *,
    category: int,
    vocabulary: GraphCategoryVocabulary,
) -> dict[str, Any]:
    attributes = dict(graph.edges[edge])
    edge_value = vocabulary.edge_value(category)
    if vocabulary.edge_attribute:
        attributes[vocabulary.edge_attribute] = edge_value
    if vocabulary.edge_attribute == "bond_type":
        attributes["bond_order"] = bond_order(int(edge_value))
    return attributes


def _apply_attributed_action(
    graph: nx.Graph,
    action: AttributedRewireAction,
    vocabulary: GraphCategoryVocabulary,
) -> nx.Graph:
    """Apply a topology swap while reassigning the two removed edge types."""

    observed_removed = tuple(
        _edge_category(graph, edge, vocabulary) for edge in action.removed
    )
    if observed_removed != action.removed_edge_categories:
        raise ValueError(
            "Attributed action removed-edge categories no longer match the current graph."
        )
    if sorted(action.added_edge_categories) != sorted(observed_removed):
        raise ValueError("Attributed action would change global edge-category counts.")

    # Preserve any non-category edge metadata by taking a template from a
    # removed edge of the same category.  The category/bond order themselves
    # are always overwritten explicitly below.
    templates: dict[int, dict[str, Any]] = {}
    for edge, category in zip(action.removed, observed_removed):
        templates.setdefault(
            int(category),
            _edge_attribute_template(
                graph, edge, category=int(category), vocabulary=vocabulary
            ),
        )

    candidate = graph.copy()
    for edge in action.removed:
        candidate.remove_edge(*edge)
    for edge, category in zip(action.added, action.added_edge_categories):
        attributes = dict(templates[int(category)])
        edge_value = vocabulary.edge_value(int(category))
        if vocabulary.edge_attribute:
            attributes[vocabulary.edge_attribute] = edge_value
        if vocabulary.edge_attribute == "bond_type":
            attributes["bond_order"] = bond_order(int(edge_value))
        candidate.add_edge(*edge, **attributes)
    return candidate


def _typed_actions_for_topology_action(
    graph: nx.Graph,
    topology_action: Action,
    *,
    vocabulary: GraphCategoryVocabulary,
    config: AttributedSpectralGraphletRefinerConfig,
) -> list[AttributedRewireAction]:
    removed_categories = tuple(
        _edge_category(graph, edge, vocabulary) for edge in topology_action[0]
    )
    if config.require_same_edge_type_pair and len(set(removed_categories)) != 1:
        return []

    assignments = [removed_categories]
    if (
        config.enumerate_edge_type_permutations
        and removed_categories[0] != removed_categories[1]
    ):
        assignments.append((removed_categories[1], removed_categories[0]))

    out: list[AttributedRewireAction] = []
    for added_categories in assignments:
        out.append(
            AttributedRewireAction(
                topology_action=topology_action,
                removed_edge_categories=(
                    int(removed_categories[0]),
                    int(removed_categories[1]),
                ),
                added_edge_categories=(
                    int(added_categories[0]),
                    int(added_categories[1]),
                ),
            )
        )
    return out


def _fast_molecular_valence_after_action(
    graph: nx.Graph,
    action: AttributedRewireAction,
    *,
    vocabulary: GraphCategoryVocabulary,
    config: AttributedSpectralGraphletRefinerConfig,
) -> bool:
    """Check the four affected weighted valences without materializing a graph."""

    affected = {node for edge in action.removed for node in edge}
    current: dict[int, float] = {}
    for node in affected:
        current[node] = float(
            sum(
                bond_order(int(data[str(vocabulary.edge_attribute)]))
                for _, _, data in graph.edges(node, data=True)
            )
        )
    delta = {node: 0.0 for node in affected}
    for edge, category in zip(action.removed, action.removed_edge_categories):
        order = bond_order(int(vocabulary.edge_value(category)))
        delta[edge[0]] -= order
        delta[edge[1]] -= order
    for edge, category in zip(action.added, action.added_edge_categories):
        order = bond_order(int(vocabulary.edge_value(category)))
        delta.setdefault(edge[0], 0.0)
        delta.setdefault(edge[1], 0.0)
        delta[edge[0]] += order
        delta[edge[1]] += order
    for node, change in delta.items():
        atom = int(graph.nodes[node][str(vocabulary.node_attribute)])
        if atom not in set(int(v) for v in vocabulary.node_values):
            return False
        limit = (
            float(config.molecular_max_valence[atom])
            if atom in config.molecular_max_valence
            else None
        )
        value = current.get(node, 0.0) + float(change)
        if value < -1.0e-8:
            return False
        if limit is not None and value > limit + 1.0e-8:
            return False
    return True


def _propose_attributed_candidates(
    graph: nx.Graph,
    *,
    vocabulary: GraphCategoryVocabulary,
    config: AttributedSpectralGraphletRefinerConfig,
    rng: np.random.Generator,
    excluded_states: set[tuple[Any, ...]] | None,
    proposal_budget: int | None = None,
    valid_candidate_budget: int | None = None,
) -> tuple[
    list[AttributedRewireAction],
    dict[AttributedRewireAction, nx.Graph],
    dict[str, Any],
]:
    """Propose valid typed double-edge swaps.

    In revised mode, any two distinct edges with four distinct endpoints may be
    selected.  For each of the two topological reconnections, both assignments
    of the two removed edge categories are enumerated.  Hence two differently
    typed removed edges yield up to four attributed successors while same-type
    edges yield the usual two.
    """

    edges = sorted(
        (min(int(u), int(v)), max(int(u), int(v))) for u, v in graph.edges()
    )
    if len(edges) < 2:
        return [], {}, {
            "num_proposals": 0,
            "num_valid_candidates": 0,
            "candidate_pass_rate": 0.0,
            "candidate_rejection_reasons": {"too_few_edges": 1},
        }

    edge_set = set(edges)
    nodes = sorted(int(node) for node in graph.nodes())
    adjacency = {node: {int(v) for v in graph.neighbors(node)} for node in nodes}

    target = int(
        config.valid_candidate_budget
        if valid_candidate_budget is None
        else valid_candidate_budget
    )
    proposal_limit = int(
        config.proposal_budget if proposal_budget is None else proposal_budget
    )
    complete = target < 0 or proposal_limit < 0
    if complete:
        pair_indices = [
            (i, j) for i in range(len(edges)) for j in range(i + 1, len(edges))
        ]
        rng.shuffle(pair_indices)
        target = 10**18
        proposal_limit = 10**18
    else:
        pair_indices = []

    seen: set[AttributedRewireAction] = set()
    candidates: list[AttributedRewireAction] = []
    candidate_graphs: dict[AttributedRewireAction, nx.Graph] = {}
    rejections: dict[str, int] = {}
    attempts = 0
    pair_cursor = 0
    max_attempts = (
        len(pair_indices)
        if complete
        else max(100, int(proposal_limit) * 50)
    )

    while (
        attempts < max_attempts
        and len(seen) < proposal_limit
        and len(candidates) < target
    ):
        attempts += 1
        if complete:
            if pair_cursor >= len(pair_indices):
                break
            left, right = pair_indices[pair_cursor]
            pair_cursor += 1
        else:
            indices = rng.choice(len(edges), size=2, replace=False)
            left, right = int(indices[0]), int(indices[1])
        e1, e2 = edges[left], edges[right]
        if len({*e1, *e2}) != 4:
            rejections["shared_endpoint"] = rejections.get("shared_endpoint", 0) + 1
            continue
        if config.require_same_edge_type_pair:
            if _edge_category(graph, e1, vocabulary) != _edge_category(graph, e2, vocabulary):
                rejections["different_edge_type"] = rejections.get("different_edge_type", 0) + 1
                continue

        topology_actions = candidate_actions_from_edge_pair(e1, e2)
        rng.shuffle(topology_actions)
        for topology_action in topology_actions:
            if not _fast_valid_action(
                graph,
                topology_action,
                edge_set=edge_set,
                nodes=nodes,
                adjacency=adjacency,
                preserve_connectivity=config.preserve_connectivity,
            ):
                rejections["topology_or_connectivity"] = rejections.get(
                    "topology_or_connectivity", 0
                ) + 1
                continue
            typed_actions = _typed_actions_for_topology_action(
                graph,
                topology_action,
                vocabulary=vocabulary,
                config=config,
            )
            rng.shuffle(typed_actions)
            for action in typed_actions:
                if action in seen:
                    continue
                seen.add(action)
                if len(seen) > proposal_limit:
                    break
                if config.enforce_molecular_valence and not _fast_molecular_valence_after_action(
                    graph, action, vocabulary=vocabulary, config=config
                ):
                    rejections["molecular_valence_fast"] = rejections.get(
                        "molecular_valence_fast", 0
                    ) + 1
                    continue
                try:
                    candidate = _apply_attributed_action(graph, action, vocabulary)
                except (KeyError, TypeError, ValueError):
                    rejections["attribute_assignment"] = rejections.get(
                        "attribute_assignment", 0
                    ) + 1
                    continue
                if config.enforce_molecular_valence and not is_molecular_valence_feasible(
                    candidate,
                    allowed_atom_types=vocabulary.node_values,
                    allowed_bond_types=config.molecular_allowed_bond_types,
                    max_valence=config.molecular_max_valence or None,
                ):
                    rejections["molecular_valence"] = rejections.get(
                        "molecular_valence", 0
                    ) + 1
                    continue
                if excluded_states is not None:
                    state = attributed_state_key(
                        candidate,
                        node_attribute=str(vocabulary.node_attribute),
                        edge_attribute=str(vocabulary.edge_attribute),
                    )
                    if state in excluded_states:
                        rejections["revisited_state"] = rejections.get(
                            "revisited_state", 0
                        ) + 1
                        continue
                candidates.append(action)
                candidate_graphs[action] = candidate
                if len(candidates) >= target:
                    break
            if len(candidates) >= target or len(seen) >= proposal_limit:
                break

    proposals = len(seen)
    return candidates, candidate_graphs, {
        "proposal_budget": int(proposal_limit),
        "valid_candidate_budget": int(target),
        "num_proposals": proposals,
        "num_valid_candidates": len(candidates),
        "candidate_pass_rate": float(len(candidates) / max(proposals, 1)),
        "candidate_rejection_reasons": rejections,
        "rewiring_kernel": (
            "same_edge_type" if config.require_same_edge_type_pair
            else "bond_reassigning_cross_type"
        ),
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


@torch.no_grad()
def predict_attributed_invariant_summary(
    model: AttributedSpectralGraphletTransformerPredictor,
    graph: nx.Graph,
    *,
    vocabulary: GraphCategoryVocabulary,
    graphlet_basis: GraphletBasis,
    device: torch.device | str,
    graphlet_logit_epsilon: float,
) -> AttributedInvariantSummaryPrediction:
    """Predict a clean summary using only the hard molecular rewiring invariant."""

    if not model.invariant_summary_enabled:
        raise ValueError(
            "Source enrichment requires a checkpoint trained with the hard-invariant summary head."
        )
    context = normalize_attributed_graph(graph)
    spectra = attributed_laplacian_spectra(
        context, edge_attribute=str(vocabulary.edge_attribute)
    )
    prob, mask, _ = extract_attributed_graphlet_simplex(
        context, graphlet_basis=graphlet_basis
    )
    logits = attributed_graphlet_simplex_to_clr(
        prob,
        graphlet_basis=graphlet_basis,
        epsilon=graphlet_logit_epsilon,
        coordinate_mask=mask,
    )
    example = AttributedSpectralExample(
        conditioning_graph=context,
        time=0.0,
        current_spectra=spectra.astype(np.float32),
        source_spectra=spectra.astype(np.float32),
        clean_spectra_target=np.zeros_like(spectra, dtype=np.float32),
        current_graphlet_probabilities=prob.astype(np.float32),
        source_graphlet_probabilities=prob.astype(np.float32),
        clean_graphlet_probabilities_target=np.zeros_like(prob, dtype=np.float32),
        current_graphlet_logits=logits.astype(np.float32),
        source_graphlet_logits=logits.astype(np.float32),
        clean_graphlet_logits_target=np.zeros_like(logits, dtype=np.float32),
        graphlet_coordinate_mask=mask.astype(np.bool_),
    )
    batch = collate_attributed_spectral_examples([example], vocabulary).to(device)
    outputs = model.invariant_summary(batch)
    n = context.number_of_nodes()
    clean_spectra = (
        outputs["invariant_clean_spectra"][0, :, :n]
        .detach().cpu().numpy().astype(np.float64)
    )
    clean_logits = (
        outputs["invariant_clean_graphlet_logits"][0]
        .detach().cpu().numpy().astype(np.float64)
    )
    clean_prob = (
        outputs["invariant_clean_graphlet_probabilities"][0]
        .detach().cpu().numpy().astype(np.float64)
    )
    return AttributedInvariantSummaryPrediction(
        clean_spectra=clean_spectra,
        clean_graphlet_logits=clean_logits,
        clean_graphlet_probabilities=clean_prob,
        graphlet_coordinate_mask=mask,
        spectral_moments=attributed_spectrum_moments(clean_spectra),
    )


def _prepare_candidate_states(
    graph: nx.Graph,
    candidates: Sequence[AttributedRewireAction],
    *,
    candidate_graphs: dict[AttributedRewireAction, nx.Graph],
    current_counts: AttributedGraphletCounts,
    vocabulary: GraphCategoryVocabulary,
    graphlet_basis: GraphletBasis,
    config: AttributedSpectralGraphletRefinerConfig,
    device: torch.device | str = "cpu",
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
    materialized = [candidate_graphs[action] for action in candidates]
    candidate_spectra_batch = batched_attributed_laplacian_spectra(
        materialized,
        edge_attribute=str(vocabulary.edge_attribute),
        device=device,
        backend=config.spectrum_backend,
        batch_size=config.spectrum_batch_size,
    )
    for action, candidate_spectra in zip(candidates, candidate_spectra_batch):
        candidate = candidate_graphs[action]
        candidate_logits, candidate_prob, candidate_counts = (
            candidate_attributed_graphlet_logits_from_counts(
                graph,
                candidate,
                action.topology_action,
                current_counts=current_counts,
                graphlet_basis=graphlet_basis,
                epsilon=config.graphlet_logit_epsilon,
            )
        )
        rows.append(
            {
                "action": action,
                "candidate_graph": candidate,
                "candidate_spectra": np.asarray(candidate_spectra, dtype=np.float64),
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
                    is_valid_molecular_graph(
                        row["candidate_graph"],
                        infer_projected_formal_charges=(
                            config.rdkit_infer_projected_formal_charges
                        ),
                    )
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


def enrich_attributed_graph_with_invariant_summary(
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
    """Enrich a typed source by matching a fixed hard-invariant-conditioned summary.

    Unlike the main reverse-summary refiner, this stage predicts exactly one
    target S(I) from the source's preserved molecular invariant and optimizes a
    fixed energy against it.  Therefore energies are comparable across steps
    and ``return_best_state`` has its literal global-within-search meaning.
    """

    cfg = (
        config
        if isinstance(config, AttributedSpectralGraphletRefinerConfig)
        else AttributedSpectralGraphletRefinerConfig.from_dict(config)
    )
    generator = rng if rng is not None else np.random.default_rng(0)
    current = normalize_attributed_graph(graph)
    invariant = extract_attributed_rewiring_invariant(
        current,
        edge_types=vocabulary.edge_values,
        node_attribute=str(vocabulary.node_attribute),
        edge_attribute=str(vocabulary.edge_attribute),
    )
    if cfg.require_rdkit_source_validity and not is_valid_molecular_graph(
        current,
        infer_projected_formal_charges=cfg.rdkit_infer_projected_formal_charges,
    ):
        raise ValueError("Attributed source graph failed RDKit sanitization before enrichment.")

    prediction = predict_attributed_invariant_summary(
        model,
        current,
        vocabulary=vocabulary,
        graphlet_basis=graphlet_basis,
        device=device,
        graphlet_logit_epsilon=cfg.graphlet_logit_epsilon,
    )
    current_prob, current_mask, current_counts_raw = extract_attributed_graphlet_simplex(
        current, graphlet_basis=graphlet_basis
    )
    current_counts = {key: dict(value) for key, value in current_counts_raw.items()}
    current_logits = attributed_graphlet_simplex_to_clr(
        current_prob,
        graphlet_basis=graphlet_basis,
        epsilon=cfg.graphlet_logit_epsilon,
        coordinate_mask=current_mask,
    )
    current_spectra = attributed_laplacian_spectra(
        current, edge_attribute=str(vocabulary.edge_attribute)
    )
    scales = attributed_spectral_scales(
        current,
        mode=cfg.spectral_normalization,
        edge_attribute=str(vocabulary.edge_attribute),
    )
    spectral_weight = float(cfg.spectral_weight_initial)
    graphlet_weight = float(cfg.graphlet_weight_initial)
    initial_spec, _ = attributed_spectral_distance(
        current_spectra,
        prediction.clean_spectra,
        scales=scales,
        metric=cfg.spectral_distance,
        channel_weights=cfg.spectral_channel_weights,
        low_frequency_weight=cfg.low_frequency_weight,
        low_frequency_cutoff=cfg.low_frequency_cutoff,
    )
    initial_graphlet = attributed_graphlet_logit_distance(
        current_logits,
        prediction.clean_graphlet_logits,
        graphlet_basis=graphlet_basis,
        coordinate_mask=current_mask,
        metric=cfg.graphlet_distance,
        size_weights=cfg.graphlet_size_weights,
    )
    best_energy = spectral_weight * initial_spec + graphlet_weight * initial_graphlet
    best_graph = current.copy()
    visited = {
        attributed_state_key(
            current,
            node_attribute=str(vocabulary.node_attribute),
            edge_attribute=str(vocabulary.edge_attribute),
        )
    }
    trace: list[dict[str, Any]] = []
    prefix = f" {debug_context}" if debug_context else ""

    for step in range(max(int(cfg.steps), 0)):
        progress = float(step / max(cfg.steps - 1, 1))
        proposal_budget, valid_candidate_budget = cfg.candidate_budgets_at(progress)
        candidates, candidate_graphs, proposal_diag = _propose_attributed_candidates(
            current,
            vocabulary=vocabulary,
            config=cfg,
            rng=generator,
            excluded_states=visited if cfg.reject_revisited_states else None,
            proposal_budget=proposal_budget,
            valid_candidate_budget=valid_candidate_budget,
        )
        if not candidates:
            trace.append(
                {
                    "step": step,
                    "accepted": False,
                    "reason": "source_enrichment_no_valid_candidate",
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
            device=device,
        )
        rows = _score_prepared(
            prepared,
            graphlet_basis=graphlet_basis,
            clean_spectra=prediction.clean_spectra,
            next_spectra=prediction.clean_spectra,
            clean_graphlet_logits=prediction.clean_graphlet_logits,
            next_graphlet_logits=prediction.clean_graphlet_logits,
            graphlet_mask=current_mask,
            spectral_weight=spectral_weight,
            graphlet_weight=graphlet_weight,
            config=cfg,
        )
        selected, rdkit_diag = _choose_candidate(rows, config=cfg, rng=generator)
        if selected is None:
            trace.append(
                {
                    "step": step,
                    "accepted": False,
                    "reason": "source_enrichment_plateau",
                    "best_energy": float(best_energy),
                    **proposal_diag,
                    **rdkit_diag,
                }
            )
            break
        chosen = rows[selected]
        candidate = chosen["candidate_graph"]
        if not attributed_rewiring_invariant_matches_graph(candidate, invariant):
            raise AssertionError("Source enrichment changed the hard attributed rewiring invariant.")
        current = candidate
        current_counts = {
            key: dict(value) for key, value in chosen["candidate_graphlet_counts"].items()
        }
        visited.add(
            attributed_state_key(
                current,
                node_attribute=str(vocabulary.node_attribute),
                edge_attribute=str(vocabulary.edge_attribute),
            )
        )
        candidate_energy = float(chosen["candidate_energy"])
        if candidate_energy < best_energy:
            best_energy = candidate_energy
            best_graph = current.copy()
        trace.append(
            {
                "step": step,
                "accepted": True,
                "reason": "invariant_summary_source_enrichment_swap",
                "energy": candidate_energy,
                "best_energy": float(best_energy),
                "energy_improvement": float(chosen["energy_improvement"]),
                "relative_energy_improvement": float(chosen["relative_energy_improvement"]),
                "action": _attributed_action_trace(chosen["action"], vocabulary),
                **proposal_diag,
                **rdkit_diag,
            }
        )
        _debug(
            cfg,
            step,
            f"{prefix} source_enrichment accepted={step + 1}/{cfg.steps} "
            f"energy={candidate_energy:.6f} best={best_energy:.6f}",
        )

    result = best_graph if cfg.return_best_state else current
    if return_trace:
        return result, trace
    return result


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
    conditioning_graph: nx.Graph | None = None,
) -> nx.Graph | tuple[nx.Graph, list[dict[str, Any]]]:
    cfg = config if isinstance(config, AttributedSpectralGraphletRefinerConfig) else AttributedSpectralGraphletRefinerConfig.from_dict(config)
    generator = rng if rng is not None else np.random.default_rng(0)
    current = normalize_attributed_graph(graph)
    context_graph = normalize_attributed_graph(
        graph if conditioning_graph is None else conditioning_graph
    )
    if context_graph.number_of_nodes() != current.number_of_nodes():
        raise ValueError("Attributed conditioning graph and current graph must have equal size.")
    rewiring_invariant = extract_attributed_rewiring_invariant(
        current,
        edge_types=vocabulary.edge_values,
        node_attribute=str(vocabulary.node_attribute),
        edge_attribute=str(vocabulary.edge_attribute),
    )
    typed_invariant = (
        extract_typed_invariant(
            current,
            edge_types=vocabulary.edge_values,
            node_attribute=str(vocabulary.node_attribute),
            edge_attribute=str(vocabulary.edge_attribute),
        )
        if cfg.preserve_typed_degree
        else None
    )
    source_weighted_valence = (
        tuple(
            sum(
                bond_order(int(data[str(vocabulary.edge_attribute)]))
                for _, _, data in current.edges(node, data=True)
            )
            for node in sorted(current.nodes())
        )
        if cfg.preserve_weighted_valence
        else None
    )
    if cfg.require_rdkit_source_validity and not is_valid_molecular_graph(
        current,
        infer_projected_formal_charges=(
            cfg.rdkit_infer_projected_formal_charges
        ),
    ):
        raise ValueError("Attributed source graph failed RDKit sanitization.")
    source_spectra = attributed_laplacian_spectra(
        context_graph, edge_attribute=str(vocabulary.edge_attribute)
    )
    source_prob, source_mask, _source_counts = extract_attributed_graphlet_simplex(
        context_graph, graphlet_basis=graphlet_basis
    )
    source_logits = attributed_graphlet_simplex_to_clr(
        source_prob,
        graphlet_basis=graphlet_basis,
        epsilon=cfg.graphlet_logit_epsilon,
        coordinate_mask=source_mask,
    )
    if current is context_graph or attributed_state_key(
        current, node_attribute=str(vocabulary.node_attribute), edge_attribute=str(vocabulary.edge_attribute)
    ) == attributed_state_key(
        context_graph, node_attribute=str(vocabulary.node_attribute), edge_attribute=str(vocabulary.edge_attribute)
    ):
        _, _, current_counts_raw = extract_attributed_graphlet_simplex(
            context_graph, graphlet_basis=graphlet_basis
        )
    else:
        _, _, current_counts_raw = extract_attributed_graphlet_simplex(
            current, graphlet_basis=graphlet_basis
        )
    current_counts = {key: dict(value) for key, value in current_counts_raw.items()}
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
                conditioning_graph=context_graph,
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
        proposal_budget, valid_candidate_budget = cfg.candidate_budgets_at(progress)
        candidates, candidate_graphs, proposal_diag = _propose_attributed_candidates(
            current,
            vocabulary=vocabulary,
            config=cfg,
            rng=generator,
            excluded_states=visited if cfg.reject_revisited_states else None,
            proposal_budget=proposal_budget,
            valid_candidate_budget=valid_candidate_budget,
        )
        if not candidates:
            trace.append(
                {
                    "step": decision_step,
                    "accepted": False,
                    "reason": "no_valid_attributed_candidates",
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
            device=device,
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
        if not attributed_rewiring_invariant_matches_graph(candidate, rewiring_invariant):
            raise AssertionError(
                "Attributed rewiring changed node types, indexed ordinary degrees, "
                "or global edge-category counts."
            )
        if typed_invariant is not None and not typed_invariant_matches_graph(
            candidate, typed_invariant
        ):
            raise AssertionError("Strict attributed rewiring changed the indexed typed-degree invariant.")
        if source_weighted_valence is not None:
            candidate_weighted_valence = tuple(
                sum(
                    bond_order(int(data[str(vocabulary.edge_attribute)]))
                    for _, _, data in candidate.edges(node, data=True)
                )
                for node in sorted(candidate.nodes())
            )
            if not np.allclose(
                candidate_weighted_valence, source_weighted_valence, atol=1.0e-8
            ):
                raise AssertionError("Strict attributed rewiring changed per-node weighted valence.")
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
                "reason": "attributed_spectral_graphlet_bond_reassigning_swap",
                "action": _attributed_action_trace(chosen["action"], vocabulary),
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
    if not attributed_rewiring_invariant_matches_graph(result, rewiring_invariant):
        raise AssertionError(
            "Final attributed graph does not preserve node types, indexed ordinary "
            "degrees, and global edge-category counts."
        )
    if typed_invariant is not None and not typed_invariant_matches_graph(
        result, typed_invariant
    ):
        raise AssertionError("Final attributed graph does not preserve the strict typed invariant.")
    if return_trace:
        return result, trace
    return result


__all__ = [
    "AttributedRewireAction",
    "AttributedSpectralGraphletPrediction",
    "AttributedInvariantSummaryPrediction",
    "AttributedSpectralGraphletRefinerConfig",
    "predict_clean_attributed_summaries",
    "predict_attributed_invariant_summary",
    "enrich_attributed_graph_with_invariant_summary",
    "refine_attributed_graph_with_spectral_graphlet_diffusion",
]

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias

import networkx as nx
import numpy as np
import torch
from torch import nn

from grapher.rewiring_mlp.core.rewiring import Action, is_valid_action
from grapher.utils.device import resolve_torch_device

SELECTOR_CHECKPOINT_FORMAT = "learned_candidate_selector_v1"
SelectorMode: TypeAlias = Literal["energy", "policy", "hybrid"]

CANDIDATE_FEATURE_NAMES = (
    "energy_improvement",
    "pair_categorical_gain",
    "pair_probability_gain",
    "graphlet_gain",
    "validity_slack",
    "unique_endpoint_fraction",
    "endpoint_degree_mean",
    "endpoint_degree_std",
    "removed_degree_product_mean",
    "added_degree_product_mean",
    "degree_product_gain",
    "removed_common_neighbors_mean",
    "added_common_neighbors_mean",
    "common_neighbor_gain",
    "removed_bridge_fraction",
)

GRAPH_CONTEXT_FEATURE_NAMES = (
    "log1p_num_nodes",
    "density",
    "normalized_mean_degree",
    "normalized_degree_std",
    "normalized_max_degree",
    "transitivity",
    "connected",
    "time",
    "remaining_step_fraction",
    "current_energy",
)


def _finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real scalar.") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _first_present(
    data: Mapping[str, Any],
    names: Sequence[str],
    *,
    default: float | None = None,
) -> float:
    for name in names:
        if name in data:
            return _finite_float(data[name], name)
    if default is not None:
        return float(default)
    raise KeyError(f"Missing candidate diagnostic; expected one of {tuple(names)!r}.")


@dataclass(frozen=True)
class CandidateDiagnostics:
    """Step-local diagnostics used by the learned action policy.

    ``energy_improvement`` is the fixed-current-step energy before the action
    minus the energy after it. Pair and graphlet gains may be individual terms
    from that energy. ``validity_slack`` is a caller-defined continuous margin;
    hard-invalid candidates must be removed before building selector features.
    """

    energy_improvement: float
    pair_categorical_gain: float = 0.0
    pair_probability_gain: float = 0.0
    graphlet_gain: float = 0.0
    validity_slack: float = 0.0

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            _finite_float(value, name)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> CandidateDiagnostics:
        """Read either selector-native or current refiner diagnostic names."""

        if not isinstance(data, Mapping):
            raise TypeError("Each candidate diagnostic must be a mapping.")
        return cls(
            energy_improvement=_first_present(
                data,
                ("energy_improvement", "energy_gain", "hybrid_score"),
            ),
            pair_categorical_gain=_first_present(
                data,
                ("pair_categorical_gain", "categorical_gain"),
                default=0.0,
            ),
            pair_probability_gain=_first_present(
                data,
                ("pair_probability_gain", "probability_gain", "pair_gain"),
                default=0.0,
            ),
            graphlet_gain=_first_present(
                data,
                ("graphlet_gain", "graphlet_improvement"),
                default=0.0,
            ),
            validity_slack=_first_present(
                data,
                ("validity_slack",),
                default=0.0,
            ),
        )


@dataclass
class SelectorFeatures:
    """Unbatched action features; STOP is added by the selector, not here."""

    actions: tuple[Action, ...]
    candidate_features: torch.Tensor
    graph_context: torch.Tensor
    candidate_feature_names: tuple[str, ...] = CANDIDATE_FEATURE_NAMES
    graph_context_feature_names: tuple[str, ...] = GRAPH_CONTEXT_FEATURE_NAMES

    def __post_init__(self) -> None:
        if self.candidate_features.ndim != 2:
            raise ValueError("candidate_features must have shape [C, F].")
        if self.graph_context.ndim != 1:
            raise ValueError("graph_context must have shape [G].")
        if len(self.actions) != self.candidate_features.shape[0]:
            raise ValueError("Action count and candidate feature count differ.")
        if self.candidate_features.shape[1] != len(self.candidate_feature_names):
            raise ValueError(
                "Candidate feature schema width does not match its tensor."
            )
        if self.graph_context.shape[0] != len(self.graph_context_feature_names):
            raise ValueError("Graph context schema width does not match its tensor.")

    def to(self, device: torch.device | str) -> SelectorFeatures:
        return SelectorFeatures(
            actions=self.actions,
            candidate_features=self.candidate_features.to(device),
            graph_context=self.graph_context.to(device),
            candidate_feature_names=self.candidate_feature_names,
            graph_context_feature_names=self.graph_context_feature_names,
        )


@dataclass
class SelectorBatch:
    candidate_features: torch.Tensor
    graph_context: torch.Tensor
    candidate_mask: torch.Tensor
    action_counts: tuple[int, ...]

    def to(self, device: torch.device | str) -> SelectorBatch:
        return SelectorBatch(
            candidate_features=self.candidate_features.to(device),
            graph_context=self.graph_context.to(device),
            candidate_mask=self.candidate_mask.to(device),
            action_counts=self.action_counts,
        )


def _validate_graph(graph: nx.Graph) -> None:
    if not isinstance(graph, nx.Graph):
        raise TypeError("graph must be a NetworkX graph.")
    if graph.is_directed() or graph.is_multigraph():
        raise ValueError("Selector features require a simple undirected graph.")
    if nx.number_of_selfloops(graph):
        raise ValueError("Selector features do not support self-loops.")


def _graph_context_value(
    diagnostics: Mapping[str, Any],
    name: str,
    default: float,
    *,
    unit_interval: bool = False,
) -> float:
    value = _finite_float(diagnostics.get(name, default), name)
    if unit_interval and not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1].")
    return value


def build_graph_context_features(
    graph: nx.Graph,
    diagnostics: Mapping[str, Any] | None = None,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build a permutation-invariant graph/step context vector."""

    _validate_graph(graph)
    values = diagnostics or {}
    if not isinstance(values, Mapping):
        raise TypeError("graph diagnostics must be a mapping.")
    n = graph.number_of_nodes()
    degrees = np.asarray([degree for _, degree in graph.degree()], dtype=np.float64)
    degree_scale = max(n - 1, 1)
    connected = bool(n > 0 and (n == 1 or nx.is_connected(graph)))
    context = [
        float(np.log1p(n)),
        float(nx.density(graph)) if n > 1 else 0.0,
        float(degrees.mean() / degree_scale) if degrees.size else 0.0,
        float(degrees.std() / degree_scale) if degrees.size else 0.0,
        float(degrees.max() / degree_scale) if degrees.size else 0.0,
        float(nx.transitivity(graph)) if n >= 3 else 0.0,
        float(connected),
        _graph_context_value(values, "time", 0.0, unit_interval=True),
        _graph_context_value(
            values,
            "remaining_step_fraction",
            1.0,
            unit_interval=True,
        ),
        _graph_context_value(values, "current_energy", 0.0),
    ]
    return torch.tensor(context, dtype=dtype, device=device)


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _action_structural_features(
    graph: nx.Graph,
    action: Action,
    bridges: set[tuple[Any, Any]],
) -> list[float]:
    removed, added = action
    n = graph.number_of_nodes()
    degree_scale = max(n - 1, 1)
    product_scale = float(degree_scale * degree_scale)
    neighbor_scale = max(n - 2, 1)
    endpoints = {node for edge in removed for node in edge}
    endpoint_degrees = [float(graph.degree(node)) for node in endpoints]
    removed_products = [
        float(graph.degree(u) * graph.degree(v)) / product_scale for u, v in removed
    ]
    added_products = [
        float(graph.degree(u) * graph.degree(v)) / product_scale for u, v in added
    ]
    removed_common = [
        len(list(nx.common_neighbors(graph, u, v))) / neighbor_scale for u, v in removed
    ]
    added_common = [
        len(list(nx.common_neighbors(graph, u, v))) / neighbor_scale for u, v in added
    ]
    canonical_bridges = {
        (min(u, v), max(u, v))
        if isinstance(u, int) and isinstance(v, int)
        else frozenset((u, v))
        for u, v in bridges
    }

    def is_bridge(edge: tuple[Any, Any]) -> bool:
        u, v = edge
        key: Any = (
            (min(u, v), max(u, v))
            if isinstance(u, int) and isinstance(v, int)
            else frozenset((u, v))
        )
        return key in canonical_bridges

    removed_product_mean = _mean(removed_products)
    added_product_mean = _mean(added_products)
    removed_common_mean = _mean(removed_common)
    added_common_mean = _mean(added_common)
    return [
        len(endpoints) / max(n, 1),
        _mean(endpoint_degrees) / degree_scale,
        float(np.std(endpoint_degrees)) / degree_scale if endpoint_degrees else 0.0,
        removed_product_mean,
        added_product_mean,
        added_product_mean - removed_product_mean,
        removed_common_mean,
        added_common_mean,
        added_common_mean - removed_common_mean,
        _mean([float(is_bridge(edge)) for edge in removed]),
    ]


def build_selector_features(
    graph: nx.Graph,
    actions: Sequence[Action],
    diagnostics: Sequence[CandidateDiagnostics | Mapping[str, Any]],
    *,
    graph_diagnostics: Mapping[str, Any] | None = None,
    preserve_connectivity: bool = True,
    validate_actions: bool = True,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> SelectorFeatures:
    """Build order-preserving features for already proposed valid actions.

    The same current graph and step diagnostics are used for every action.
    Thus a caller can compute the predictor and current energy once, then pass
    fixed pair/graphlet changes for all same-step candidates.
    """

    _validate_graph(graph)
    action_tuple = tuple(actions)
    if len(action_tuple) != len(diagnostics):
        raise ValueError("actions and diagnostics must have the same length.")
    if len(set(action_tuple)) != len(action_tuple):
        raise ValueError("Candidate actions must be deduplicated before scoring.")
    parsed = [
        value
        if isinstance(value, CandidateDiagnostics)
        else CandidateDiagnostics.from_mapping(value)
        for value in diagnostics
    ]
    bridges = set(nx.bridges(graph)) if graph.number_of_nodes() else set()
    rows: list[list[float]] = []
    for index, (action, diagnostic) in enumerate(zip(action_tuple, parsed)):
        if validate_actions and not is_valid_action(
            graph,
            action,
            preserve_connectivity=preserve_connectivity,
        ):
            raise ValueError(f"Candidate action at index {index} is not valid.")
        row = [
            float(diagnostic.energy_improvement),
            float(diagnostic.pair_categorical_gain),
            float(diagnostic.pair_probability_gain),
            float(diagnostic.graphlet_gain),
            float(diagnostic.validity_slack),
            *_action_structural_features(graph, action, bridges),
        ]
        if len(row) != len(CANDIDATE_FEATURE_NAMES):
            raise AssertionError("Internal selector feature schema mismatch.")
        rows.append(row)
    candidate_features = torch.tensor(rows, dtype=dtype, device=device)
    if not rows:
        candidate_features = torch.empty(
            (0, len(CANDIDATE_FEATURE_NAMES)),
            dtype=dtype,
            device=device,
        )
    graph_context = build_graph_context_features(
        graph,
        graph_diagnostics,
        dtype=dtype,
        device=device,
    )
    return SelectorFeatures(
        actions=action_tuple,
        candidate_features=candidate_features,
        graph_context=graph_context,
    )


def collate_selector_features(features: Sequence[SelectorFeatures]) -> SelectorBatch:
    """Pad variable candidate sets; padded actions are masked before softmax."""

    if not features:
        raise ValueError("Cannot collate an empty selector batch.")
    candidate_width = features[0].candidate_features.shape[1]
    context_width = features[0].graph_context.shape[0]
    dtype = features[0].candidate_features.dtype
    device = features[0].candidate_features.device
    for value in features:
        if value.candidate_features.shape[1] != candidate_width:
            raise ValueError("Candidate feature widths differ across examples.")
        if value.graph_context.shape[0] != context_width:
            raise ValueError("Graph context widths differ across examples.")
        if (
            value.candidate_features.dtype != dtype
            or value.graph_context.dtype != dtype
        ):
            raise ValueError("Selector feature dtypes differ across examples.")
        if (
            value.candidate_features.device != device
            or value.graph_context.device != device
        ):
            raise ValueError("Selector feature devices differ across examples.")
    counts = tuple(len(value.actions) for value in features)
    max_candidates = max(counts, default=0)
    batch_size = len(features)
    candidate_tensor = torch.zeros(
        (batch_size, max_candidates, candidate_width),
        dtype=dtype,
        device=device,
    )
    candidate_mask = torch.zeros(
        (batch_size, max_candidates),
        dtype=torch.bool,
        device=device,
    )
    contexts = []
    for index, value in enumerate(features):
        count = counts[index]
        candidate_tensor[index, :count] = value.candidate_features
        candidate_mask[index, :count] = True
        contexts.append(value.graph_context)
    return SelectorBatch(
        candidate_features=candidate_tensor,
        graph_context=torch.stack(contexts),
        candidate_mask=candidate_mask,
        action_counts=counts,
    )


class LearnedCandidateSelector(nn.Module):
    """Shared candidate scorer with a graph-conditioned learned STOP token."""

    def __init__(
        self,
        *,
        candidate_feature_dim: int,
        graph_context_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if int(candidate_feature_dim) <= 0:
            raise ValueError("candidate_feature_dim must be positive.")
        if int(graph_context_dim) <= 0:
            raise ValueError("graph_context_dim must be positive.")
        if int(hidden_dim) <= 0:
            raise ValueError("hidden_dim must be positive.")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        self.candidate_feature_dim = int(candidate_feature_dim)
        self.graph_context_dim = int(graph_context_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout_p = float(dropout)
        self.candidate_encoder = nn.Sequential(
            nn.Linear(self.candidate_feature_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(self.graph_context_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        )
        # Candidate and STOP embeddings pass through this exact same scorer.
        self.shared_scorer = nn.Sequential(
            nn.Linear(3 * self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.hidden_dim, 1),
        )
        self.stop_embedding = nn.Parameter(torch.zeros(self.hidden_dim))
        nn.init.normal_(self.stop_embedding, mean=0.0, std=0.02)

    def model_config(self) -> dict[str, Any]:
        return {
            "candidate_feature_dim": self.candidate_feature_dim,
            "graph_context_dim": self.graph_context_dim,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout_p,
        }

    def _score(
        self,
        candidate_encoding: torch.Tensor,
        context_encoding: torch.Tensor,
    ) -> torch.Tensor:
        context = context_encoding.unsqueeze(1).expand_as(candidate_encoding)
        scorer_input = torch.cat(
            [candidate_encoding, context, candidate_encoding * context],
            dim=-1,
        )
        return self.shared_scorer(scorer_input).squeeze(-1)

    def forward(
        self,
        candidate_features: torch.Tensor,
        graph_context: torch.Tensor,
        candidate_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return action logits with the finite STOP logit in the last column."""

        if not isinstance(candidate_features, torch.Tensor) or not isinstance(
            graph_context, torch.Tensor
        ):
            raise TypeError("candidate_features and graph_context must be tensors.")
        unbatched = candidate_features.ndim == 2
        if unbatched:
            if graph_context.ndim != 1:
                raise ValueError("Unbatched graph_context must have shape [G].")
            candidate_features = candidate_features.unsqueeze(0)
            graph_context = graph_context.unsqueeze(0)
            if candidate_mask is not None:
                candidate_mask = candidate_mask.unsqueeze(0)
        elif candidate_features.ndim == 3:
            if graph_context.ndim != 2:
                raise ValueError("Batched graph_context must have shape [B, G].")
        else:
            raise ValueError("candidate_features must have shape [C,F] or [B,C,F].")
        if (
            not candidate_features.is_floating_point()
            or not graph_context.is_floating_point()
        ):
            raise TypeError("Selector inputs must be floating-point tensors.")
        if candidate_features.device != graph_context.device:
            raise ValueError("Selector inputs must be on the same device.")
        if candidate_features.dtype != graph_context.dtype:
            raise ValueError("Selector inputs must have the same dtype.")
        batch_size, candidate_count, feature_dim = candidate_features.shape
        if feature_dim != self.candidate_feature_dim:
            raise ValueError(
                f"Expected candidate feature width {self.candidate_feature_dim}, got {feature_dim}."
            )
        if graph_context.shape != (batch_size, self.graph_context_dim):
            raise ValueError(
                "graph_context shape does not match the batch or configured width."
            )
        if (
            not torch.isfinite(candidate_features).all()
            or not torch.isfinite(graph_context).all()
        ):
            raise ValueError("Selector inputs must be finite.")
        if candidate_mask is None:
            mask = torch.ones(
                (batch_size, candidate_count),
                dtype=torch.bool,
                device=candidate_features.device,
            )
        else:
            if not isinstance(candidate_mask, torch.Tensor):
                raise TypeError("candidate_mask must be a tensor.")
            if candidate_mask.shape != (batch_size, candidate_count):
                raise ValueError(
                    "candidate_mask shape does not match candidate features."
                )
            if candidate_mask.device != candidate_features.device:
                raise ValueError("candidate_mask must be on the selector input device.")
            mask = candidate_mask.bool()

        context_encoding = self.context_encoder(graph_context)
        candidate_encoding = self.candidate_encoder(candidate_features)
        candidate_logits = self._score(candidate_encoding, context_encoding)
        candidate_logits = candidate_logits.masked_fill(~mask, float("-inf"))
        stop_encoding = self.stop_embedding.view(1, 1, -1).expand(
            batch_size,
            1,
            self.hidden_dim,
        )
        stop_logit = self._score(stop_encoding, context_encoding)
        logits = torch.cat([candidate_logits, stop_logit], dim=-1)
        return logits[0] if unbatched else logits

    def distribution_loss(
        self,
        logits: torch.Tensor,
        teacher_distribution: torch.Tensor,
        *,
        action_mask: torch.Tensor | None = None,
        objective: str = "cross_entropy",
        reduction: str = "mean",
    ) -> torch.Tensor:
        return selector_distribution_loss(
            logits,
            teacher_distribution,
            action_mask=action_mask,
            objective=objective,
            reduction=reduction,
        )


def build_teacher_distribution(
    energy_improvements: Sequence[float] | torch.Tensor,
    *,
    temperature: float = 0.0,
    positive_epsilon: float = 0.0,
) -> torch.Tensor:
    """Create a hard/soft improving-action teacher with STOP in the last slot."""

    if isinstance(energy_improvements, torch.Tensor):
        values = energy_improvements
        if values.ndim != 1:
            raise ValueError("energy_improvements must be one-dimensional.")
        if not values.is_floating_point():
            values = values.float()
    else:
        values = torch.as_tensor(list(energy_improvements), dtype=torch.float32)
    if not torch.isfinite(values).all():
        raise ValueError("energy_improvements must be finite.")
    temperature = _finite_float(temperature, "temperature")
    positive_epsilon = _finite_float(positive_epsilon, "positive_epsilon")
    if temperature < 0.0:
        raise ValueError("temperature must be non-negative.")
    improving = values > positive_epsilon
    distribution = torch.zeros(
        values.numel() + 1,
        dtype=values.dtype,
        device=values.device,
    )
    if not torch.any(improving):
        distribution[-1] = 1.0
        return distribution
    if temperature == 0.0:
        maximum = values[improving].max()
        winners = improving & torch.isclose(values, maximum)
        distribution[:-1][winners] = 1.0 / winners.sum().to(values.dtype)
    else:
        distribution[:-1][improving] = torch.softmax(
            values[improving] / temperature,
            dim=0,
        )
    return distribution


def _expanded_action_mask(
    action_mask: torch.Tensor | None,
    logits: torch.Tensor,
) -> torch.Tensor:
    if action_mask is None:
        return torch.ones_like(logits, dtype=torch.bool)
    if not isinstance(action_mask, torch.Tensor):
        raise TypeError("action_mask must be a tensor.")
    mask = action_mask.bool()
    if mask.shape == logits.shape:
        if not torch.all(mask[..., -1]):
            raise ValueError("STOP must always be active in action_mask.")
        return mask
    if mask.shape == logits.shape[:-1] + (logits.shape[-1] - 1,):
        stop = torch.ones(
            mask.shape[:-1] + (1,),
            dtype=torch.bool,
            device=mask.device,
        )
        return torch.cat([mask, stop], dim=-1)
    raise ValueError("action_mask shape must cover candidates or candidates plus STOP.")


def selector_distribution_loss(
    logits: torch.Tensor,
    teacher_distribution: torch.Tensor,
    *,
    action_mask: torch.Tensor | None = None,
    objective: str = "cross_entropy",
    reduction: str = "mean",
) -> torch.Tensor:
    """Cross-entropy or KL against a normalized teacher including STOP."""

    if not isinstance(logits, torch.Tensor) or not isinstance(
        teacher_distribution, torch.Tensor
    ):
        raise TypeError("logits and teacher_distribution must be tensors.")
    unbatched = logits.ndim == 1
    if unbatched:
        logits = logits.unsqueeze(0)
        teacher_distribution = teacher_distribution.unsqueeze(0)
        if action_mask is not None and action_mask.ndim == 1:
            action_mask = action_mask.unsqueeze(0)
    if logits.ndim != 2 or teacher_distribution.shape != logits.shape:
        raise ValueError("logits and teacher_distribution must have shape [B, C+1].")
    if logits.shape[-1] < 1:
        raise ValueError("Selector logits must contain STOP.")
    if logits.device != teacher_distribution.device:
        raise ValueError("Loss tensors must be on the same device.")
    mask = _expanded_action_mask(action_mask, logits).to(logits.device)
    if torch.isnan(logits).any() or torch.isposinf(logits).any():
        raise ValueError("Selector logits cannot contain NaN or positive infinity.")
    if not torch.isfinite(logits[mask]).all():
        raise ValueError("Every active action, including STOP, needs a finite logit.")
    teacher = teacher_distribution.to(dtype=logits.dtype)
    if not torch.isfinite(teacher).all() or torch.any(teacher < 0.0):
        raise ValueError("Teacher probabilities must be finite and non-negative.")
    if torch.any(teacher.masked_select(~mask) > 1.0e-7):
        raise ValueError("Teacher assigns probability to a masked candidate.")
    totals = teacher.sum(dim=-1)
    if not torch.allclose(totals, torch.ones_like(totals), atol=1.0e-5, rtol=1.0e-5):
        raise ValueError("Each teacher distribution must sum to one.")
    masked_logits = logits.masked_fill(~mask, float("-inf"))
    log_probabilities = torch.log_softmax(masked_logits, dim=-1)
    cross_entropy_terms = torch.where(
        teacher > 0.0,
        -teacher * log_probabilities,
        torch.zeros_like(teacher),
    )
    cross_entropy = cross_entropy_terms.sum(dim=-1)
    normalized_objective = str(objective).lower()
    if normalized_objective in {"cross_entropy", "ce"}:
        losses = cross_entropy
    elif normalized_objective in {"kl", "kl_divergence"}:
        teacher_entropy_terms = torch.where(
            teacher > 0.0,
            teacher * torch.log(teacher),
            torch.zeros_like(teacher),
        )
        losses = cross_entropy + teacher_entropy_terms.sum(dim=-1)
    else:
        raise ValueError("objective must be 'cross_entropy' or 'kl'.")
    if reduction == "none":
        return losses[0] if unbatched else losses
    if reduction == "sum":
        return losses.sum()
    if reduction == "mean":
        return losses.mean()
    raise ValueError("reduction must be 'none', 'sum', or 'mean'.")


def _as_batched_scores(
    value: torch.Tensor | None,
    *,
    name: str,
    includes_stop: bool,
) -> tuple[torch.Tensor | None, bool]:
    if value is None:
        return None, True
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor.")
    if value.ndim == 1:
        batched = value.unsqueeze(0)
        unbatched = True
    elif value.ndim == 2:
        batched = value
        unbatched = False
    else:
        suffix = "C+1" if includes_stop else "C"
        raise ValueError(f"{name} must have shape [{suffix}] or [B,{suffix}].")
    return batched, unbatched


def combine_selector_scores(
    *,
    mode: SelectorMode | str,
    policy_logits: torch.Tensor | None = None,
    energy_improvements: torch.Tensor | None = None,
    candidate_mask: torch.Tensor | None = None,
    policy_weight: float = 1.0,
    energy_weight: float = 1.0,
    policy_shortlist_size: int | None = None,
    positive_improvement_only: bool = True,
    positive_epsilon: float = 0.0,
    stop_energy: float = 0.0,
) -> torch.Tensor:
    """Dispatch energy-, policy-, or policy-shortlisted hybrid scoring.

    In hybrid mode policy logits first produce an order-equivariant shortlist
    (ties at the cutoff are retained), then fixed-current-step improvements
    rescore that shortlist. STOP remains available regardless of candidates.
    """

    normalized_mode = str(mode).lower()
    if normalized_mode not in {"energy", "policy", "hybrid"}:
        raise ValueError("mode must be 'energy', 'policy', or 'hybrid'.")
    policy, policy_unbatched = _as_batched_scores(
        policy_logits,
        name="policy_logits",
        includes_stop=True,
    )
    energy, energy_unbatched = _as_batched_scores(
        energy_improvements,
        name="energy_improvements",
        includes_stop=False,
    )
    if normalized_mode in {"policy", "hybrid"} and policy is None:
        raise ValueError(f"{normalized_mode} mode requires policy_logits.")
    if normalized_mode in {"energy", "hybrid"} and energy is None:
        raise ValueError(f"{normalized_mode} mode requires energy_improvements.")
    reference = policy if policy is not None else energy
    assert reference is not None
    batch_size = reference.shape[0]
    candidate_count = (
        policy.shape[1] - 1 if policy is not None else energy.shape[1]  # type: ignore[union-attr]
    )
    if policy is not None and policy.shape != (batch_size, candidate_count + 1):
        raise ValueError("policy_logits has an inconsistent shape.")
    if energy is not None and energy.shape != (batch_size, candidate_count):
        raise ValueError("energy_improvements shape does not match policy candidates.")
    if (
        policy is not None
        and energy is not None
        and (policy.device != energy.device or policy.dtype != energy.dtype)
    ):
        raise ValueError("Policy and energy tensors must share device and dtype.")
    device = reference.device
    dtype = reference.dtype
    if not reference.is_floating_point():
        raise TypeError("Selector scores must be floating-point tensors.")
    if policy is not None and (
        torch.isnan(policy).any() or torch.isposinf(policy).any()
    ):
        raise ValueError("policy_logits cannot contain NaN or positive infinity.")
    if energy is not None and not torch.isfinite(energy).all():
        raise ValueError("energy_improvements must be finite.")
    if candidate_mask is None:
        mask = torch.ones(
            (batch_size, candidate_count),
            dtype=torch.bool,
            device=device,
        )
    else:
        candidate_mask_batched = (
            candidate_mask.unsqueeze(0) if candidate_mask.ndim == 1 else candidate_mask
        )
        if candidate_mask_batched.shape != (batch_size, candidate_count):
            raise ValueError("candidate_mask shape does not match candidate scores.")
        mask = candidate_mask_batched.to(device=device, dtype=torch.bool)
    if policy is not None and not torch.isfinite(policy[:, -1]).all():
        raise ValueError("STOP policy logits must be finite.")

    policy_weight = _finite_float(policy_weight, "policy_weight")
    energy_weight = _finite_float(energy_weight, "energy_weight")
    positive_epsilon = _finite_float(positive_epsilon, "positive_epsilon")
    stop_energy = _finite_float(stop_energy, "stop_energy")
    if policy_shortlist_size is not None and int(policy_shortlist_size) <= 0:
        raise ValueError("policy_shortlist_size must be positive when set.")

    if normalized_mode == "policy":
        assert policy is not None
        candidate_scores = policy_weight * policy[:, :-1]
        stop_scores = policy_weight * policy[:, -1:]
    elif normalized_mode == "energy":
        assert energy is not None
        candidate_scores = energy_weight * energy
        stop_scores = torch.full(
            (batch_size, 1),
            energy_weight * stop_energy,
            dtype=dtype,
            device=device,
        )
    else:
        assert policy is not None and energy is not None
        candidate_scores = policy_weight * policy[:, :-1] + energy_weight * energy
        stop_scores = policy_weight * policy[:, -1:] + energy_weight * stop_energy
        if policy_shortlist_size is not None:
            shortlist = torch.zeros_like(mask)
            for batch_index in range(batch_size):
                active = mask[batch_index]
                active_scores = policy[batch_index, :-1][active]
                if active_scores.numel() == 0:
                    continue
                k = min(int(policy_shortlist_size), active_scores.numel())
                threshold = torch.topk(active_scores, k=k).values[-1]
                shortlist[batch_index] = active & (
                    policy[batch_index, :-1] >= threshold
                )
            mask = mask & shortlist
    if normalized_mode in {"energy", "hybrid"} and positive_improvement_only:
        assert energy is not None
        mask = mask & (energy > positive_epsilon)
    candidate_scores = candidate_scores.masked_fill(~mask, float("-inf"))
    combined = torch.cat([candidate_scores, stop_scores], dim=-1)
    unbatched = policy_unbatched if policy is not None else energy_unbatched
    return combined[0] if unbatched else combined


@dataclass
class SelectorDecision:
    action: Action | None
    index: int
    stopped: bool
    reason: str
    probabilities: torch.Tensor
    scores: torch.Tensor
    mode: str


def select_action(
    actions: Sequence[Action],
    scores: torch.Tensor,
    *,
    mode: SelectorMode | str,
    temperature: float = 1.0,
    deterministic: bool = True,
    generator: torch.Generator | None = None,
) -> SelectorDecision:
    """Choose one action or explicit STOP from a single graph's scores."""

    action_tuple = tuple(actions)
    if len(set(action_tuple)) != len(action_tuple):
        raise ValueError("Candidate actions must be unique.")
    if not isinstance(scores, torch.Tensor) or scores.ndim != 1:
        raise ValueError("scores must be a one-dimensional tensor.")
    if scores.numel() != len(action_tuple) + 1:
        raise ValueError("scores must contain one value per action plus STOP.")
    if torch.isnan(scores).any() or torch.isposinf(scores).any():
        raise ValueError("scores cannot contain NaN or positive infinity.")
    if not torch.isfinite(scores[-1]):
        raise ValueError("STOP score must be finite.")
    temperature = _finite_float(temperature, "temperature")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")
    probabilities = torch.softmax(scores / temperature, dim=-1)
    if deterministic:
        index = int(torch.argmax(scores).item())
    else:
        index = int(torch.multinomial(probabilities, 1, generator=generator).item())
    stopped = index == len(action_tuple)
    return SelectorDecision(
        action=None if stopped else action_tuple[index],
        index=index,
        stopped=stopped,
        reason="learned_stop" if stopped else "selected_action",
        probabilities=probabilities.detach().clone(),
        scores=scores.detach().clone(),
        mode=str(mode).lower(),
    )


@torch.no_grad()
def select_with_selector(
    model: LearnedCandidateSelector,
    features: SelectorFeatures,
    *,
    mode: SelectorMode | str,
    energy_improvements: torch.Tensor | None = None,
    policy_weight: float = 1.0,
    energy_weight: float = 1.0,
    policy_shortlist_size: int | None = None,
    positive_improvement_only: bool = True,
    positive_epsilon: float = 0.0,
    temperature: float = 1.0,
    deterministic: bool = True,
    generator: torch.Generator | None = None,
) -> SelectorDecision:
    """High-level policy/hybrid decision API for one candidate set."""

    model.eval()
    policy_logits = model(features.candidate_features, features.graph_context)
    if energy_improvements is None and str(mode).lower() in {"energy", "hybrid"}:
        energy_improvements = features.candidate_features[
            :, CANDIDATE_FEATURE_NAMES.index("energy_improvement")
        ]
    scores = combine_selector_scores(
        mode=mode,
        policy_logits=policy_logits,
        energy_improvements=energy_improvements,
        policy_weight=policy_weight,
        energy_weight=energy_weight,
        policy_shortlist_size=policy_shortlist_size,
        positive_improvement_only=positive_improvement_only,
        positive_epsilon=positive_epsilon,
    )
    return select_action(
        features.actions,
        scores,
        mode=mode,
        temperature=temperature,
        deterministic=deterministic,
        generator=generator,
    )


def save_selector_checkpoint(
    model: LearnedCandidateSelector,
    path: str | Path,
    *,
    config: Mapping[str, Any] | None = None,
    report: Mapping[str, Any] | None = None,
    feature_schema: Mapping[str, Sequence[str]] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = dict(
        feature_schema
        or {
            "candidate": list(CANDIDATE_FEATURE_NAMES),
            "graph_context": list(GRAPH_CONTEXT_FEATURE_NAMES),
        }
    )
    if len(schema.get("candidate", ())) != model.candidate_feature_dim:
        raise ValueError("Checkpoint candidate feature schema/model width mismatch.")
    if len(schema.get("graph_context", ())) != model.graph_context_dim:
        raise ValueError("Checkpoint graph context schema/model width mismatch.")
    torch.save(
        {
            "format": SELECTOR_CHECKPOINT_FORMAT,
            "model_state_dict": model.state_dict(),
            "model_config": model.model_config(),
            "feature_schema": schema,
            "config": dict(config or {}),
            "report": dict(report or {}),
        },
        path,
    )


def load_selector_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "auto",
) -> tuple[LearnedCandidateSelector, dict[str, Any]]:
    resolved_device = (
        resolve_torch_device(device) if isinstance(device, str) else device
    )
    checkpoint = torch.load(Path(path), map_location=resolved_device)
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("format") != SELECTOR_CHECKPOINT_FORMAT
    ):
        raise ValueError(
            f"Checkpoint is not a learned candidate selector ({SELECTOR_CHECKPOINT_FORMAT})."
        )
    model_config = checkpoint.get("model_config")
    state_dict = checkpoint.get("model_state_dict")
    schema = checkpoint.get("feature_schema")
    if not isinstance(model_config, dict) or not isinstance(state_dict, dict):
        raise TypeError(
            "Selector checkpoint is missing model configuration or weights."
        )
    if not isinstance(schema, dict):
        raise TypeError("Selector checkpoint is missing its feature schema.")
    if len(schema.get("candidate", ())) != int(model_config["candidate_feature_dim"]):
        raise ValueError("Checkpoint candidate feature schema/model mismatch.")
    if len(schema.get("graph_context", ())) != int(model_config["graph_context_dim"]):
        raise ValueError("Checkpoint graph context schema/model mismatch.")
    model = LearnedCandidateSelector(**model_config).to(resolved_device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, checkpoint

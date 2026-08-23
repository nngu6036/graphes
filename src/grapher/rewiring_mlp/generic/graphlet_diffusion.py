from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Any, Mapping, Sequence

import networkx as nx
import numpy as np

from grapher.rewiring_mlp.core.rewiring import Action
from grapher.rewiring_mlp.generic.basis import TopologyGraphletBasis
from grapher.rewiring_mlp.generic.graphlets import (
    TopologyGraphletCounts,
    candidate_topology_graphlet_counts,
    extract_topology_graphlet_counts,
)


@dataclass(frozen=True)
class GraphletLogitBridgeSchedule:
    """Deterministic x0-style bridge in graphlet CLR/logit coordinates.

    The actual graph state is always discrete and valid.  The bridge target is
    only a soft denoising target used to rank valid degree-preserving swaps.
    """

    schedule: str = "cosine"
    min_clean_mix: float = 0.05
    max_clean_mix: float = 1.0
    power: float = 2.0

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None = None) -> "GraphletLogitBridgeSchedule":
        values = dict(data or {})
        schedule = str(values.get("schedule", "cosine")).lower()
        if schedule not in {"linear", "cosine", "power"}:
            raise ValueError("graphlet_guidance.schedule must be linear, cosine, or power.")
        minimum = float(values.get("min_clean_mix", 0.05))
        maximum = float(values.get("max_clean_mix", 1.0))
        power = float(values.get("power", 2.0))
        if not (0.0 <= minimum <= maximum <= 1.0):
            raise ValueError("Graphlet clean-mix bounds must satisfy 0 <= min <= max <= 1.")
        if power <= 0.0 or not np.isfinite(power):
            raise ValueError("graphlet_guidance.power must be finite and positive.")
        return cls(schedule=schedule, min_clean_mix=minimum, max_clean_mix=maximum, power=power)

    def clean_mix_for_step(self, *, accepted_step: int, total_steps: int) -> float:
        if total_steps <= 1:
            progress = 1.0
        else:
            progress = float(np.clip(accepted_step / max(total_steps - 1, 1), 0.0, 1.0))
        if self.schedule == "linear":
            shaped = progress
        elif self.schedule == "cosine":
            shaped = 0.5 - 0.5 * np.cos(np.pi * progress)
        else:
            shaped = progress ** self.power
        return float(self.min_clean_mix + (self.max_clean_mix - self.min_clean_mix) * shaped)

    @staticmethod
    def target(current_logits: np.ndarray, clean_logits: np.ndarray, clean_mix: float) -> np.ndarray:
        alpha = float(np.clip(clean_mix, 0.0, 1.0))
        return (1.0 - alpha) * np.asarray(current_logits, dtype=np.float64) + alpha * np.asarray(
            clean_logits, dtype=np.float64
        )


def graphlet_simplex_from_counts(
    counts_by_size: TopologyGraphletCounts,
    *,
    num_nodes: int,
    graphlet_basis: TopologyGraphletBasis,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-k graphlet probabilities including one disconnected bin.

    For each order k, connected induced graphlet counts are divided by C(n,k).
    The final coordinate is the probability that a uniformly selected k-node
    subset is disconnected.  Every valid block therefore lies on a probability
    simplex and preserves connected-subgraph mass instead of normalizing it away.

    Returns ``(probabilities, coordinate_mask)``.  A block is masked when n < k.
    """

    values: list[float] = []
    mask: list[bool] = []
    n = int(num_nodes)
    for key in graphlet_basis.sizes:
        k = int(key)
        keys = graphlet_basis.keys_by_k[key]
        total = comb(n, k) if n >= k else 0
        if total <= 0:
            values.extend([0.0] * (len(keys) + 1))
            mask.extend([False] * (len(keys) + 1))
            continue
        counts = counts_by_size.get(key, {})
        block = np.asarray(
            [float(counts.get(graphlet_key, 0)) / float(total) for graphlet_key in keys],
            dtype=np.float64,
        )
        connected_mass = float(np.clip(block.sum(), 0.0, 1.0))
        disconnected = max(0.0, 1.0 - connected_mass)
        full = np.concatenate([block, np.asarray([disconnected], dtype=np.float64)])
        # Numerical count arithmetic should already sum to one; normalize only
        # to remove floating-point drift.
        full /= max(float(full.sum()), 1.0e-12)
        values.extend(full.tolist())
        mask.extend([True] * full.size)
    return np.asarray(values, dtype=np.float64), np.asarray(mask, dtype=np.bool_)


def extract_topology_graphlet_simplex(
    graph: nx.Graph,
    *,
    graphlet_basis: TopologyGraphletBasis,
) -> tuple[np.ndarray, np.ndarray, TopologyGraphletCounts]:
    counts = extract_topology_graphlet_counts(graph, graphlet_basis=graphlet_basis)
    probabilities, mask = graphlet_simplex_from_counts(
        counts,
        num_nodes=graph.number_of_nodes(),
        graphlet_basis=graphlet_basis,
    )
    return probabilities, mask, counts


def graphlet_simplex_to_clr(
    probabilities: Sequence[float] | np.ndarray,
    *,
    graphlet_basis: TopologyGraphletBasis,
    epsilon: float = 1.0e-5,
    coordinate_mask: Sequence[bool] | np.ndarray | None = None,
) -> np.ndarray:
    """Map graphlet simplexes to centered log-ratio coordinates blockwise."""

    eps = float(epsilon)
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError("graphlet logit epsilon must be finite and positive.")
    vector = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if vector.size != graphlet_basis.simplex_width:
        raise ValueError(
            f"Expected {graphlet_basis.simplex_width} graphlet simplex values, received {vector.size}."
        )
    if coordinate_mask is None:
        valid_mask = np.ones(vector.size, dtype=np.bool_)
    else:
        valid_mask = np.asarray(coordinate_mask, dtype=np.bool_).reshape(-1)
        if valid_mask.shape != vector.shape:
            raise ValueError("Graphlet coordinate mask shape mismatch.")
    output = np.zeros_like(vector)
    for start, stop in graphlet_basis.simplex_slices:
        block_mask = valid_mask[start:stop]
        if not np.any(block_mask):
            continue
        block = np.maximum(vector[start:stop], 0.0)
        block = block / max(float(block.sum()), 1.0e-12)
        log_values = np.log(block + eps)
        log_values -= float(log_values.mean())
        output[start:stop] = log_values
    return output


def graphlet_clr_to_simplex(
    logits: Sequence[float] | np.ndarray,
    *,
    graphlet_basis: TopologyGraphletBasis,
    coordinate_mask: Sequence[bool] | np.ndarray | None = None,
) -> np.ndarray:
    """Inverse CLR up to the additive gauge, implemented blockwise by softmax."""

    vector = np.asarray(logits, dtype=np.float64).reshape(-1)
    if vector.size != graphlet_basis.simplex_width:
        raise ValueError(
            f"Expected {graphlet_basis.simplex_width} graphlet logits, received {vector.size}."
        )
    if coordinate_mask is None:
        valid_mask = np.ones(vector.size, dtype=np.bool_)
    else:
        valid_mask = np.asarray(coordinate_mask, dtype=np.bool_).reshape(-1)
        if valid_mask.shape != vector.shape:
            raise ValueError("Graphlet coordinate mask shape mismatch.")
    output = np.zeros_like(vector)
    for start, stop in graphlet_basis.simplex_slices:
        block_mask = valid_mask[start:stop]
        if not np.any(block_mask):
            continue
        block = vector[start:stop]
        shifted = block - float(np.max(block))
        weights = np.exp(shifted)
        output[start:stop] = weights / max(float(weights.sum()), 1.0e-12)
    return output


def _size_weight_map(
    graphlet_basis: TopologyGraphletBasis,
    size_weights: Mapping[str, float] | Sequence[float] | None,
) -> dict[str, float]:
    if size_weights is None:
        return {key: 1.0 for key in graphlet_basis.sizes}
    if isinstance(size_weights, Mapping):
        result = {key: float(size_weights.get(key, 1.0)) for key in graphlet_basis.sizes}
    else:
        values = [float(value) for value in size_weights]
        if len(values) != len(graphlet_basis.sizes):
            raise ValueError("graphlet size_weights sequence must match configured graphlet orders.")
        result = dict(zip(graphlet_basis.sizes, values))
    if any((not np.isfinite(value)) or value < 0.0 for value in result.values()):
        raise ValueError("graphlet size weights must be finite and nonnegative.")
    if not any(value > 0.0 for value in result.values()):
        raise ValueError("At least one graphlet size weight must be positive.")
    return result


def graphlet_logit_distance(
    left_logits: Sequence[float] | np.ndarray,
    right_logits: Sequence[float] | np.ndarray,
    *,
    graphlet_basis: TopologyGraphletBasis,
    coordinate_mask: Sequence[bool] | np.ndarray | None = None,
    metric: str = "rmse",
    size_weights: Mapping[str, float] | Sequence[float] | None = None,
) -> float:
    """Weighted mean distance between per-order CLR graphlet blocks."""

    left = np.asarray(left_logits, dtype=np.float64).reshape(-1)
    right = np.asarray(right_logits, dtype=np.float64).reshape(-1)
    if left.shape != right.shape or left.size != graphlet_basis.simplex_width:
        raise ValueError("Graphlet logit distance received inconsistent vector widths.")
    if coordinate_mask is None:
        valid_mask = np.ones(left.size, dtype=np.bool_)
    else:
        valid_mask = np.asarray(coordinate_mask, dtype=np.bool_).reshape(-1)
        if valid_mask.shape != left.shape:
            raise ValueError("Graphlet logit distance mask shape mismatch.")
    metric_name = str(metric).lower()
    if metric_name in {"clr_rmse", "rmse", "l2"}:
        metric_name = "rmse"
    elif metric_name in {"clr_mae", "mae", "l1"}:
        metric_name = "mae"
    elif metric_name != "mse":
        raise ValueError("graphlet logit distance must be rmse, mse, or mae.")
    weights = _size_weight_map(graphlet_basis, size_weights)
    total = 0.0
    normalizer = 0.0
    for key, (start, stop) in zip(graphlet_basis.sizes, graphlet_basis.simplex_slices):
        mask = valid_mask[start:stop]
        if not np.any(mask):
            continue
        delta = left[start:stop][mask] - right[start:stop][mask]
        if metric_name == "rmse":
            value = float(np.sqrt(np.mean(delta * delta)))
        elif metric_name == "mse":
            value = float(np.mean(delta * delta))
        else:
            value = float(np.mean(np.abs(delta)))
        weight = float(weights[key])
        total += weight * value
        normalizer += weight
    return float(total / max(normalizer, 1.0e-12))


def candidate_graphlet_logits_from_counts(
    graph: nx.Graph,
    candidate: nx.Graph,
    action: Action,
    *,
    current_counts: TopologyGraphletCounts,
    graphlet_basis: TopologyGraphletBasis,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray, TopologyGraphletCounts]:
    """Incrementally update counts for one swap and return its graphlet CLR state."""

    candidate_counts = candidate_topology_graphlet_counts(
        graph,
        candidate,
        action,
        current_counts=current_counts,
        graphlet_basis=graphlet_basis,
    )
    probabilities, mask = graphlet_simplex_from_counts(
        candidate_counts,
        num_nodes=candidate.number_of_nodes(),
        graphlet_basis=graphlet_basis,
    )
    logits = graphlet_simplex_to_clr(
        probabilities,
        graphlet_basis=graphlet_basis,
        epsilon=epsilon,
        coordinate_mask=mask,
    )
    return logits, probabilities, candidate_counts

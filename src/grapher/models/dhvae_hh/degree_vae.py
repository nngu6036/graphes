from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from grapher.utils.device import resolve_torch_device


def _sorted_degree_sequence(graph: nx.Graph) -> list[int]:
    """Return degrees without importing GraphER's summary implementation."""

    return sorted((int(degree) for _, degree in graph.degree()), reverse=True)


def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    total = float(x.sum())
    if total <= 0:
        out = np.zeros_like(x, dtype=np.float64)
        if out.size:
            out[0] = 1.0
        return out
    return x / total


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    logits = logits - np.max(logits)
    probs = np.exp(logits)
    return _normalize(probs)


def _degree_counts_to_sequence(counts: np.ndarray) -> list[int]:
    seq: list[int] = []
    for degree, count in enumerate(np.asarray(counts, dtype=np.int64).reshape(-1)):
        seq.extend([int(degree)] * int(count))
    return sorted(seq, reverse=True)


def _degree_sequence_to_counts(sequence: list[int], width: int) -> np.ndarray:
    out = np.zeros(int(width), dtype=np.int64)
    for degree in sequence:
        degree = int(degree)
        if 0 <= degree < width:
            out[degree] += 1
    return out


def _integer_counts_from_probs(n: int, probs: np.ndarray) -> np.ndarray:
    probs = _normalize(probs)
    raw = probs * int(n)
    counts = np.floor(raw).astype(np.int64)
    remaining = int(n) - int(counts.sum())
    if remaining > 0:
        order = np.argsort(-(raw - counts))
        for i in order[:remaining]:
            counts[int(i)] += 1
    elif remaining < 0:
        order = np.argsort(raw - counts)
        for i in order[:-remaining]:
            if counts[int(i)] > 0:
                counts[int(i)] -= 1
    return counts


def _sample_even_degree_counts(
    n: int,
    probs: np.ndarray,
    rng: np.random.Generator,
    *,
    max_attempts: int = 32,
) -> tuple[np.ndarray, int]:
    """Sample a multinomial histogram conditional on an even degree sum."""

    attempts = max(int(max_attempts), 1)
    counts = rng.multinomial(int(n), _normalize(probs))
    for draw in range(1, attempts + 1):
        if int(np.dot(np.arange(counts.size), counts)) % 2 == 0:
            return counts, draw
        if draw < attempts:
            counts = rng.multinomial(int(n), _normalize(probs))
    return counts, attempts


def _logsumexp_np(values: list[float]) -> float:
    if not values:
        return float("-inf")
    array = np.asarray(values, dtype=np.float64)
    maximum = float(np.max(array))
    if not np.isfinite(maximum):
        return float("-inf")
    return maximum + float(np.log(np.exp(array - maximum).sum()))


def _sample_degree_counts_exact_sum(
    n: int,
    probs: np.ndarray,
    total_degree: int,
    rng: np.random.Generator,
    *,
    min_degree: int = 0,
    max_degree: int | None = None,
    deterministic: bool = False,
) -> np.ndarray | None:
    """Sample iid categorical degrees conditioned on an exact degree sum.

    Dynamic programming computes the conditional normalizer for the remaining
    nodes and degree sum.  This keeps the decoder distribution intact while
    enforcing ``sum_i d_i = total_degree`` exactly, so parity and edge count do
    not need post-hoc repair.
    """

    n = int(n)
    total_degree = int(total_degree)
    probs = _normalize(np.asarray(probs, dtype=np.float64).reshape(-1))
    if n < 0 or total_degree < 0 or probs.size == 0:
        return None
    upper = probs.size - 1 if max_degree is None else min(int(max_degree), probs.size - 1)
    lower = max(int(min_degree), 0)
    if lower > upper:
        return None
    if total_degree < n * lower or total_degree > n * upper:
        return None

    support = [degree for degree in range(lower, upper + 1) if probs[degree] > 0.0]
    if not support:
        return None
    log_probs = np.full(probs.size, float("-inf"), dtype=np.float64)
    for degree in support:
        log_probs[degree] = float(np.log(max(probs[degree], 1.0e-300)))

    # dp[k, s] is log mass of k remaining iid draws summing to s.
    dp = np.full((n + 1, total_degree + 1), float("-inf"), dtype=np.float64)
    dp[0, 0] = 0.0
    for k in range(1, n + 1):
        min_sum = max(0, k * lower)
        max_sum = min(total_degree, k * upper)
        for degree_sum in range(min_sum, max_sum + 1):
            values = [
                float(log_probs[degree] + dp[k - 1, degree_sum - degree])
                for degree in support
                if degree <= degree_sum and np.isfinite(dp[k - 1, degree_sum - degree])
            ]
            dp[k, degree_sum] = _logsumexp_np(values)
    if not np.isfinite(dp[n, total_degree]):
        return None

    counts = np.zeros(probs.size, dtype=np.int64)
    remaining_nodes = n
    remaining_sum = total_degree
    while remaining_nodes > 0:
        candidates: list[int] = []
        log_weights: list[float] = []
        for degree in support:
            if degree > remaining_sum:
                continue
            suffix = dp[remaining_nodes - 1, remaining_sum - degree]
            if not np.isfinite(suffix):
                continue
            candidates.append(degree)
            log_weights.append(float(log_probs[degree] + suffix))
        if not candidates:
            return None
        if deterministic:
            chosen = candidates[int(np.argmax(np.asarray(log_weights)))]
        else:
            weights = np.asarray(log_weights, dtype=np.float64)
            weights -= float(np.max(weights))
            weights = np.exp(weights)
            weights /= float(weights.sum())
            chosen = int(rng.choice(np.asarray(candidates, dtype=np.int64), p=weights))
        counts[chosen] += 1
        remaining_sum -= chosen
        remaining_nodes -= 1
    return counts


def connected_feasible_degree_sequence(sequence: list[int]) -> bool:
    n = len(sequence)
    if n <= 1:
        return True
    if min(sequence) <= 0:
        return False
    if sum(sequence) < 2 * (n - 1):
        return False
    return nx.is_graphical(
        sorted([int(d) for d in sequence], reverse=True), method="eg"
    )


def repair_degree_sequence(
    degree_sequence: list[int],
    *,
    n: int,
    require_connected: bool = True,
    rng: np.random.Generator | None = None,
    max_iterations: int = 10000,
) -> list[int]:
    """Project a sampled sequence to a graphical fallback after rejection fails."""

    generator = rng if rng is not None else np.random.default_rng(0)
    min_degree = 1 if require_connected and n > 1 else 0
    max_degree = max(n - 1, 0)
    sequence = [
        int(np.clip(degree, min_degree, max_degree)) for degree in degree_sequence[:n]
    ]
    sequence.extend([min_degree] * (n - len(sequence)))
    if n <= 1:
        return [0] * n

    def fix_parity() -> None:
        if sum(sequence) % 2 == 0:
            return
        order = generator.permutation(n)
        for index in order:
            if sequence[int(index)] < max_degree:
                sequence[int(index)] += 1
                return
        for index in order:
            if sequence[int(index)] > min_degree:
                sequence[int(index)] -= 1
                return

    if require_connected:
        while sum(sequence) < 2 * (n - 1):
            candidates = [
                index for index, degree in enumerate(sequence) if degree < max_degree
            ]
            if not candidates:
                break
            sequence[int(generator.choice(candidates))] += 1
    fix_parity()

    for _ in range(max_iterations):
        if nx.is_graphical(sorted(sequence, reverse=True), method="eg"):
            return sorted(sequence, reverse=True)
        candidates = [
            index for index, degree in enumerate(sequence) if degree > min_degree
        ]
        if candidates:
            largest = max(sequence[index] for index in candidates)
            largest_indices = [
                index for index in candidates if sequence[index] == largest
            ]
            sequence[int(generator.choice(largest_indices))] -= 1
        else:
            sequence = [1] * n
            sequence[0] = sequence[1] = 2 if n > 2 else 1
        fix_parity()

    if require_connected and n >= 2:
        sequence = [2] * n
        sequence[0] = sequence[-1] = 1
    return sorted(sequence, reverse=True)


@dataclass
class DegreeVectorizer:
    """Vectorizer and post-processor for DegreeHistogramVAE."""

    min_nodes: int
    max_nodes: int
    max_degree: int
    max_edges: int
    require_connected: bool = True
    empirical_node_counts: list[int] | None = None
    empirical_edge_counts: list[int] | None = None
    empirical_degree_sequences: list[list[int]] | None = None

    @classmethod
    def fit(
        cls,
        graphs: list[nx.Graph],
        *,
        max_degree: int | None = None,
        require_connected: bool = True,
    ) -> "DegreeVectorizer":
        if not graphs:
            raise ValueError("Cannot fit DegreeVectorizer on an empty graph list.")
        node_counts = [int(g.number_of_nodes()) for g in graphs]
        edge_counts = [int(g.number_of_edges()) for g in graphs]
        degree_sequences = [_sorted_degree_sequence(g) for g in graphs]
        observed_max_degree = max(max(seq) if seq else 0 for seq in degree_sequences)
        max_degree = observed_max_degree if max_degree is None else int(max_degree)
        max_edges = max(int(g.number_of_edges()) for g in graphs)
        return cls(
            min_nodes=int(min(node_counts)),
            max_nodes=int(max(node_counts)),
            max_degree=int(max(max_degree, 1)),
            max_edges=int(max(max_edges, 1)),
            require_connected=bool(require_connected),
            empirical_node_counts=[int(x) for x in node_counts],
            empirical_edge_counts=[int(x) for x in edge_counts],
            empirical_degree_sequences=[
                [int(d) for d in seq] for seq in degree_sequences
            ],
        )

    @property
    def node_count_classes(self) -> int:
        return int(self.max_nodes - self.min_nodes + 1)

    @property
    def degree_dim(self) -> int:
        return int(self.max_degree + 1)

    @property
    def input_dim(self) -> int:
        # q(z | h_D, n): edge count is exactly determined by h_D and is not
        # an independent encoder input.
        return int(1 + self.degree_dim)

    @property
    def edge_count_classes(self) -> int:
        return int(self.max_edges + 1)

    def head_dims(self) -> dict[str, int]:
        return {
            "num_nodes": self.node_count_classes,
            "num_edges": self.edge_count_classes,
            "degree": self.degree_dim,
        }

    def node_index(self, n: int) -> int:
        return int(np.clip(int(n) - self.min_nodes, 0, self.node_count_classes - 1))

    def node_count_from_index(self, idx: int) -> int:
        return int(self.min_nodes + int(np.clip(idx, 0, self.node_count_classes - 1)))

    def degree_hist_from_sequence(self, sequence: list[int]) -> np.ndarray:
        return _normalize(
            _degree_sequence_to_counts(sequence, self.degree_dim).astype(np.float64)
        )

    def to_feature_vector(self, graph: nx.Graph) -> np.ndarray:
        seq = _sorted_degree_sequence(graph)
        n = int(graph.number_of_nodes())
        degree = self.degree_hist_from_sequence(seq)
        size = np.asarray([n / max(float(self.max_nodes), 1.0)], dtype=np.float64)
        return np.concatenate([size, degree]).astype(np.float32)

    def to_targets(self, graph: nx.Graph) -> dict[str, np.ndarray | np.int64]:
        seq = _sorted_degree_sequence(graph)
        n = int(graph.number_of_nodes())
        m = int(graph.number_of_edges())
        return {
            "num_nodes": np.int64(self.node_index(n)),
            "num_nodes_count": np.int64(n),
            "num_edges_count": np.int64(m),
            "degree": self.degree_hist_from_sequence(seq).astype(np.float32),
            "mean_degree": np.asarray(
                [(2.0 * m / n) if n > 0 else 0.0], dtype=np.float32
            ),
        }

    def to_training_arrays(
        self, graphs: list[nx.Graph]
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        x = np.stack([self.to_feature_vector(g) for g in graphs]).astype(np.float32)
        items = [self.to_targets(g) for g in graphs]
        targets: dict[str, np.ndarray] = {}
        for key in items[0]:
            targets[key] = np.asarray([item[key] for item in items])
        return x, targets

    def empirical_node_distribution(self) -> np.ndarray:
        counts = np.zeros(self.node_count_classes, dtype=np.float64)
        for n in self.empirical_node_counts or []:
            counts[self.node_index(int(n))] += 1.0
        return _normalize(counts)

    def sample_empirical_node_count(self, rng: np.random.Generator) -> int:
        probs = self.empirical_node_distribution()
        idx = int(rng.choice(np.arange(self.node_count_classes), p=probs))
        return self.node_count_from_index(idx)

    def empirical_edge_distribution(self, n: int) -> np.ndarray:
        """Empirical p(m | n), with nearest-size fallback for sparse sizes."""

        node_counts = self.empirical_node_counts or []
        edge_counts = self.empirical_edge_counts or []
        probs = np.zeros(self.edge_count_classes, dtype=np.float64)
        if node_counts and len(node_counts) == len(edge_counts):
            distances = np.asarray([abs(int(x) - int(n)) for x in node_counts])
            best_distance = int(distances.min()) if distances.size else 0
            for observed_n, observed_m, distance in zip(node_counts, edge_counts, distances):
                if int(distance) == best_distance and 0 <= int(observed_m) < probs.size:
                    probs[int(observed_m)] += 1.0
        if probs.sum() <= 0:
            low = max(int(n) - 1, 0) if self.require_connected and int(n) > 1 else 0
            high = min(int(n) * (int(n) - 1) // 2, self.max_edges)
            if low <= high:
                probs[low : high + 1] = 1.0
        return _normalize(probs)

    def sample_empirical_edge_count(self, n: int, rng: np.random.Generator) -> int:
        probs = self.empirical_edge_distribution(int(n))
        return int(rng.choice(np.arange(self.edge_count_classes), p=probs))

    def empirical_nearest_degree_sequence(
        self, n: int, rng: np.random.Generator | None = None
    ) -> list[int]:
        generator = rng if rng is not None else np.random.default_rng(0)
        sequences = self.empirical_degree_sequences or []
        if not sequences:
            if n <= 1:
                return [0] * int(n)
            seq = [2] * int(n)
            seq[0] = seq[-1] = 1
            return sorted(seq, reverse=True)

        distances = np.asarray(
            [abs(len(seq) - int(n)) for seq in sequences], dtype=np.int64
        )
        best = np.flatnonzero(distances == distances.min())
        chosen = sequences[int(generator.choice(best))]
        seq = [int(d) for d in chosen]
        if len(seq) == int(n):
            return sorted(seq, reverse=True)
        if len(seq) > int(n):
            seq = seq[: int(n)]
        else:
            min_degree = 1 if self.require_connected and n > 1 else 0
            seq = seq + [min_degree] * (int(n) - len(seq))
        return repair_degree_sequence(
            seq, n=int(n), require_connected=self.require_connected, rng=generator
        )

    def outputs_to_summaries(
        self,
        outputs: dict[str, Any],
        *,
        rng: np.random.Generator | None = None,
        deterministic: bool = False,
        sample_num_nodes: str = "empirical",
        sample_num_edges: str = "model",
        exact_degree_sum_conditioning: bool = True,
        max_resample: int = 200,
        fallback: str = "empirical_nearest_n",
        parity_conditioned: bool = True,
        max_parity_resample: int = 32,
        postprocess_policy: str = "repair",
        include_diagnostics: bool = False,
    ) -> list[dict[str, Any]]:
        policy = str(postprocess_policy).strip().lower()
        if policy not in {"repair", "reject_only"}:
            raise ValueError(
                "postprocess_policy must be either 'repair' or 'reject_only'."
            )
        generator = rng if rng is not None else np.random.default_rng(0)
        arrays: dict[str, np.ndarray] = {}
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                arrays[key] = value.detach().cpu().numpy()
            else:
                arrays[key] = np.asarray(value)
        batch = int(next(iter(arrays.values())).shape[0])
        summaries: list[dict[str, Any]] = []
        for i in range(batch):
            n_probs = _softmax_np(arrays["num_nodes_logits"][i])
            conditioned_nodes = arrays.get("conditioned_num_nodes")
            if conditioned_nodes is not None:
                n = int(np.asarray(conditioned_nodes[i]).reshape(-1)[0])
            elif str(sample_num_nodes).lower() == "empirical":
                n = self.sample_empirical_node_count(generator)
            else:
                if deterministic:
                    n_idx = int(np.argmax(n_probs))
                else:
                    n_idx = int(
                        generator.choice(np.arange(self.node_count_classes), p=n_probs)
                    )
                n = self.node_count_from_index(n_idx)

            target_num_edges: int | None = None
            conditioned_edges = arrays.get("conditioned_num_edges")
            if conditioned_edges is not None:
                target_num_edges = int(np.asarray(conditioned_edges[i]).reshape(-1)[0])
            elif str(sample_num_edges).lower() == "empirical":
                target_num_edges = self.sample_empirical_edge_count(n, generator)
            elif "num_edges_logits" in arrays:
                edge_probs = _softmax_np(arrays["num_edges_logits"][i]).astype(np.float64)
                low_edges = max(n - 1, 0) if self.require_connected and n > 1 else 0
                high_edges = min(n * (n - 1) // 2, self.max_edges)
                edge_probs[:low_edges] = 0.0
                if high_edges + 1 < edge_probs.size:
                    edge_probs[high_edges + 1 :] = 0.0
                edge_probs = _normalize(edge_probs)
                if deterministic:
                    target_num_edges = int(np.argmax(edge_probs))
                else:
                    target_num_edges = int(
                        generator.choice(np.arange(edge_probs.size), p=edge_probs)
                    )

            degree_probs = _softmax_np(arrays["degree_logits"][i]).astype(np.float64)
            if n < degree_probs.size:
                degree_probs[n:] = 0.0
            if self.require_connected and n > 1:
                degree_probs[0] = 0.0
            degree_probs = _normalize(degree_probs)

            degree_sequence: list[int] | None = None
            first_raw_sequence: list[int] | None = None
            last_raw_sequence: list[int] | None = None
            attempts_used = 0
            repair_used = False
            fallback_used = False
            accepted_without_postprocessing = False
            parity_draws_total = 0
            attempt_limit = 1 if deterministic else max(int(max_resample), 1)
            exact_sum_used = bool(
                exact_degree_sum_conditioning and target_num_edges is not None
            )
            for attempt in range(attempt_limit):
                if exact_sum_used:
                    counts = _sample_degree_counts_exact_sum(
                        n,
                        degree_probs,
                        2 * int(target_num_edges),
                        generator,
                        min_degree=(1 if self.require_connected and n > 1 else 0),
                        max_degree=max(n - 1, 0),
                        deterministic=deterministic,
                    )
                    parity_draws = 1
                    if counts is None:
                        break
                elif deterministic:
                    counts = _integer_counts_from_probs(n, degree_probs)
                    parity_draws = 1
                elif parity_conditioned:
                    counts, parity_draws = _sample_even_degree_counts(
                        n,
                        degree_probs,
                        generator,
                        max_attempts=max_parity_resample,
                    )
                else:
                    counts = generator.multinomial(int(n), degree_probs)
                    parity_draws = 1
                parity_draws_total += int(parity_draws)
                raw_seq = _degree_counts_to_sequence(counts)
                if first_raw_sequence is None:
                    first_raw_sequence = list(raw_seq)
                last_raw_sequence = list(raw_seq)
                raw_valid = nx.is_graphical(raw_seq, method="eg") and (
                    not self.require_connected
                    or connected_feasible_degree_sequence(raw_seq)
                )
                if raw_valid:
                    degree_sequence = raw_seq
                    attempts_used = attempt + 1
                    accepted_without_postprocessing = True
                    break

            # Projection is a last resort. The previous implementation repaired
            # every proposal inside the retry loop, so it almost never performed
            # true rejection sampling and hid the native decoder quality.
            if degree_sequence is None:
                attempts_used = attempt_limit
                if policy == "repair" and last_raw_sequence is not None:
                    repaired = repair_degree_sequence(
                        last_raw_sequence,
                        n=int(n),
                        require_connected=self.require_connected,
                        rng=generator,
                    )
                    if nx.is_graphical(repaired, method="eg") and (
                        not self.require_connected
                        or connected_feasible_degree_sequence(repaired)
                    ):
                        degree_sequence = repaired
                        repair_used = sorted(last_raw_sequence, reverse=True) != sorted(
                            repaired, reverse=True
                        )
                if degree_sequence is None:
                    if fallback == "error":
                        raise RuntimeError(
                            "Degree generator exhausted its samples without a "
                            "valid graphical, connected-feasible degree sequence."
                        )
                    degree_sequence = self.empirical_nearest_degree_sequence(
                        n, generator
                    )
                    fallback_used = True

            counts = _degree_sequence_to_counts(
                degree_sequence, self.degree_dim
            ).astype(np.float64)
            degree_hist = _normalize(counts)
            num_edges = int(sum(degree_sequence) // 2)
            density = (2.0 * num_edges / (int(n) * (int(n) - 1))) if int(n) > 1 else 0.0
            summary = {
                "num_nodes": int(n),
                "num_edges": int(num_edges),
                "degree_sequence": sorted(
                    [int(d) for d in degree_sequence], reverse=True
                ),
                "degree_hist": degree_hist.astype(np.float64),
                "density": float(density),
            }
            if include_diagnostics:
                raw_seq = first_raw_sequence
                raw_graphical = bool(
                    raw_seq is not None and nx.is_graphical(raw_seq, method="eg")
                )
                raw_connected_feasible = bool(
                    raw_seq is not None and connected_feasible_degree_sequence(raw_seq)
                )
                raw_even_sum = bool(raw_seq is not None and sum(raw_seq) % 2 == 0)
                raw_degree_bounds = bool(
                    raw_seq is not None
                    and len(raw_seq) == int(n)
                    and all(0 <= int(d) < int(n) for d in raw_seq)
                )
                repair_l1 = (
                    int(
                        np.abs(
                            np.asarray(
                                sorted(last_raw_sequence or raw_seq, reverse=True),
                                dtype=np.int64,
                            )
                            - np.asarray(
                                sorted(degree_sequence, reverse=True),
                                dtype=np.int64,
                            )
                        ).sum()
                    )
                    if raw_seq is not None
                    else 0
                )
                summary["sampling_diagnostics"] = {
                    "raw_graphical": raw_graphical,
                    "raw_connected_feasible": raw_connected_feasible,
                    "raw_even_degree_sum": raw_even_sum,
                    "raw_degree_bounds_valid": raw_degree_bounds,
                    "repair_used": bool(repair_used),
                    "repair_l1_adjustment": repair_l1,
                    "fallback_used": bool(fallback_used),
                    "attempts_used": int(attempts_used),
                    "parity_draws": int(parity_draws_total),
                    "parity_redraws": int(max(parity_draws_total - attempts_used, 0)),
                    "accepted_without_postprocessing": bool(
                        accepted_without_postprocessing
                    ),
                    "postprocess_policy": policy,
                    "target_num_edges": (
                        int(target_num_edges) if target_num_edges is not None else None
                    ),
                    "exact_degree_sum_conditioned": bool(exact_sum_used),
                    "raw_edge_count_matches_target": bool(
                        raw_seq is not None
                        and target_num_edges is not None
                        and sum(raw_seq) == 2 * int(target_num_edges)
                    ),
                    "first_raw_degree_sequence": (
                        [int(d) for d in raw_seq] if raw_seq is not None else []
                    ),
                }
            summaries.append(summary)
        return summaries

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: str | Path) -> "DegreeVectorizer":
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int | None = None,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        dim = int(input_dim)
        for _ in range(max(int(num_layers), 1)):
            layers.append(nn.Linear(dim, int(hidden_dim)))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
            dim = int(hidden_dim)
        if output_dim is not None:
            layers.append(nn.Linear(dim, int(output_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DegreeHistogramVAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int,
        head_dims: dict[str, int],
        *,
        min_nodes: int,
        max_nodes: int,
        max_degree: int,
        max_edges: int | None = None,
        require_connected: bool = True,
        size_condition_dim: int = 16,
        edge_condition_dim: int = 16,
        use_edge_count_conditioning: bool = False,
        prior_condition_on_edges: bool = False,
        prior_type: str = "conditional_gmm",
        prior_components: int = 4,
        prior_hidden_dim: int | None = None,
        prior_logvar_min: float = -6.0,
        prior_logvar_max: float = 4.0,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        prior_type = str(prior_type).lower()
        if prior_type not in {
            "standard_normal",
            "conditional_gaussian",
            "conditional_gmm",
        }:
            raise ValueError(
                "prior_type must be standard_normal, conditional_gaussian, "
                "or conditional_gmm."
            )
        if prior_type == "conditional_gaussian":
            prior_components = 1
        if int(prior_components) < 1:
            raise ValueError("prior_components must be at least one.")
        self.use_edge_count_conditioning = bool(use_edge_count_conditioning)
        self.prior_condition_on_edges = bool(
            prior_condition_on_edges and self.use_edge_count_conditioning
        )
        self.architecture_version = 4 if self.use_edge_count_conditioning else 3
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.head_dims = dict(head_dims)
        self.min_nodes = int(min_nodes)
        self.max_nodes = int(max_nodes)
        self.max_degree = int(max_degree)
        self.max_edges = int(
            max_edges
            if max_edges is not None
            else max_nodes * (max_nodes - 1) // 2
        )
        self.require_connected = bool(require_connected)
        self.size_condition_dim = int(size_condition_dim)
        self.edge_condition_dim = int(edge_condition_dim)
        self.prior_type = prior_type
        self.prior_components = int(prior_components)
        self.prior_hidden_dim = int(prior_hidden_dim or hidden_dim)
        self.prior_logvar_min = float(prior_logvar_min)
        self.prior_logvar_max = float(prior_logvar_max)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.encoder = MLP(
            input_dim, hidden_dim, num_layers=num_layers, dropout=dropout
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.prior_decoder = MLP(
            latent_dim, hidden_dim, num_layers=num_layers, dropout=dropout
        )
        self.num_nodes_head = nn.Linear(hidden_dim, head_dims["num_nodes"])
        self.size_encoder = MLP(
            2,
            hidden_dim,
            output_dim=self.size_condition_dim,
            num_layers=1,
            dropout=dropout,
        )
        if self.use_edge_count_conditioning:
            self.edge_count_decoder = MLP(
                2,
                hidden_dim,
                num_layers=max(num_layers - 1, 1),
                dropout=dropout,
            )
            self.num_edges_head = nn.Linear(hidden_dim, head_dims["num_edges"])
            self.edge_encoder = MLP(
                2,
                hidden_dim,
                output_dim=self.edge_condition_dim,
                num_layers=1,
                dropout=dropout,
            )
        if self.prior_type != "standard_normal":
            self.conditional_prior = MLP(
                4 if self.prior_condition_on_edges else 2,
                self.prior_hidden_dim,
                output_dim=self.prior_components * (1 + 2 * self.latent_dim),
                num_layers=1,
                dropout=dropout,
            )
            final = self.conditional_prior.net[-1]
            if isinstance(final, nn.Linear):
                nn.init.zeros_(final.weight)
                nn.init.zeros_(final.bias)
                if self.prior_components > 1:
                    with torch.no_grad():
                        bias = final.bias.reshape(
                            self.prior_components,
                            1 + 2 * self.latent_dim,
                        )
                        bias[:, 1] = torch.linspace(
                            -0.1,
                            0.1,
                            self.prior_components,
                            device=bias.device,
                        )
        self.degree_decoder = MLP(
            latent_dim
            + self.size_condition_dim
            + (self.edge_condition_dim if self.use_edge_count_conditioning else 0),
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.degree_head = nn.Linear(hidden_dim, head_dims["degree"])

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.mu(h), self.logvar(h).clamp(min=-10.0, max=10.0)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def node_count_logits(self, z: torch.Tensor) -> torch.Tensor:
        return self.num_nodes_head(self.prior_decoder(z))

    def _size_features(self, node_counts: torch.Tensor) -> torch.Tensor:
        n = node_counts.to(dtype=torch.float32).reshape(-1, 1)
        linear = n / max(float(self.max_nodes), 1.0)
        logarithmic = torch.log1p(n) / max(float(np.log1p(self.max_nodes)), 1.0)
        return torch.cat([linear, logarithmic], dim=-1)

    def _edge_features(
        self, node_counts: torch.Tensor, edge_counts: torch.Tensor
    ) -> torch.Tensor:
        n = node_counts.to(dtype=torch.float32).reshape(-1, 1)
        m = edge_counts.to(dtype=torch.float32).reshape(-1, 1)
        scaled_m = m / max(float(self.max_edges), 1.0)
        possible = torch.clamp(n * (n - 1.0) / 2.0, min=1.0)
        density = m / possible
        return torch.cat([scaled_m, density], dim=-1)

    def edge_count_logits(self, node_counts: torch.Tensor) -> torch.Tensor:
        if not self.use_edge_count_conditioning:
            raise RuntimeError("This DH-VAE checkpoint has no edge-count head.")
        node_counts = node_counts.to(
            device=next(self.parameters()).device, dtype=torch.long
        ).reshape(-1)
        hidden = self.edge_count_decoder(self._size_features(node_counts))
        logits = self.num_edges_head(hidden)
        edges = torch.arange(logits.shape[-1], device=logits.device).unsqueeze(0)
        if self.require_connected:
            lower = torch.where(
                node_counts > 1, node_counts - 1, torch.zeros_like(node_counts)
            ).unsqueeze(1)
        else:
            lower = torch.zeros_like(node_counts).unsqueeze(1)
        upper = torch.minimum(
            node_counts * (node_counts - 1) // 2,
            torch.full_like(node_counts, self.max_edges),
        ).unsqueeze(1)
        invalid = (edges < lower) | (edges > upper)
        return logits.masked_fill(invalid, -1.0e9)

    def prior_parameters(
        self,
        node_counts: torch.Tensor,
        edge_counts: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        node_counts = node_counts.to(
            device=next(self.parameters()).device, dtype=torch.long
        ).reshape(-1)
        if self.prior_condition_on_edges:
            if edge_counts is None:
                edge_counts = torch.argmax(self.edge_count_logits(node_counts), dim=-1)
            edge_counts = edge_counts.to(
                device=node_counts.device, dtype=torch.long
            ).reshape(-1)
        batch = int(node_counts.shape[0])
        if self.prior_type == "standard_normal":
            logits = torch.zeros(batch, 1, device=node_counts.device)
            means = torch.zeros(batch, 1, self.latent_dim, device=node_counts.device)
            logvars = torch.zeros_like(means)
        else:
            prior_features = self._size_features(node_counts)
            if self.prior_condition_on_edges:
                prior_features = torch.cat(
                    [prior_features, self._edge_features(node_counts, edge_counts)],
                    dim=-1,
                )
            raw = self.conditional_prior(prior_features)
            raw = raw.reshape(batch, self.prior_components, 1 + 2 * self.latent_dim)
            logits = raw[..., 0]
            means = raw[..., 1 : 1 + self.latent_dim]
            logvars = raw[..., 1 + self.latent_dim :].clamp(
                min=self.prior_logvar_min, max=self.prior_logvar_max
            )
        return {
            "prior_logits": logits,
            "prior_means": means,
            "prior_logvars": logvars,
        }

    def sample_prior(
        self,
        node_counts: torch.Tensor,
        *,
        edge_counts: torch.Tensor | None = None,
        prior_mode: str = "model",
    ) -> torch.Tensor:
        prior_mode = str(prior_mode).lower()
        node_counts = node_counts.to(
            device=next(self.parameters()).device, dtype=torch.long
        ).reshape(-1)
        if prior_mode == "standard_normal":
            return torch.randn(
                node_counts.shape[0],
                self.latent_dim,
                device=node_counts.device,
            )
        if prior_mode != "model":
            raise ValueError("prior_mode must be 'model' or 'standard_normal'.")
        params = self.prior_parameters(node_counts, edge_counts)
        components = torch.distributions.Categorical(
            logits=params["prior_logits"]
        ).sample()
        rows = torch.arange(node_counts.shape[0], device=node_counts.device)
        means = params["prior_means"][rows, components]
        logvars = params["prior_logvars"][rows, components]
        return means + torch.randn_like(means) * torch.exp(0.5 * logvars)

    def decode(
        self,
        z: torch.Tensor,
        node_counts: torch.Tensor,
        edge_counts: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        node_counts = node_counts.to(device=z.device, dtype=torch.long).reshape(-1)
        if node_counts.shape[0] != z.shape[0]:
            raise ValueError("node_counts must have one value per latent sample.")
        if bool(
            torch.any(node_counts < self.min_nodes)
            or torch.any(node_counts > self.max_nodes)
        ):
            raise ValueError(
                f"node_counts must lie in [{self.min_nodes}, {self.max_nodes}]."
            )

        node_logits = self.node_count_logits(z)
        size_embedding = self.size_encoder(self._size_features(node_counts))
        edge_logits = None
        decoder_inputs = [z, size_embedding]
        if self.use_edge_count_conditioning:
            edge_logits = self.edge_count_logits(node_counts)
            if edge_counts is None:
                edge_counts = torch.argmax(edge_logits, dim=-1)
            edge_counts = edge_counts.to(device=z.device, dtype=torch.long).reshape(-1)
            if edge_counts.shape[0] != z.shape[0]:
                raise ValueError("edge_counts must have one value per latent sample.")
            edge_embedding = self.edge_encoder(
                self._edge_features(node_counts, edge_counts)
            )
            decoder_inputs.append(edge_embedding)
        degree_hidden = self.degree_decoder(torch.cat(decoder_inputs, dim=-1))
        degree_logits = self.degree_head(degree_hidden)
        degrees = torch.arange(
            degree_logits.shape[-1], device=z.device, dtype=torch.long
        )
        invalid = degrees.unsqueeze(0) >= node_counts.unsqueeze(1)
        degree_logits = degree_logits.masked_fill(invalid, -1.0e9)
        degree_probs = F.softmax(degree_logits, dim=-1)
        expected_mean_degree = torch.sum(
            degree_probs * degrees.to(dtype=degree_probs.dtype).unsqueeze(0),
            dim=-1,
            keepdim=True,
        )
        result = {
            "num_nodes_logits": node_logits,
            "degree_logits": degree_logits,
            "conditioned_num_nodes": node_counts,
            "expected_mean_degree": expected_mean_degree,
        }
        if edge_logits is not None:
            result["num_edges_logits"] = edge_logits
            result["conditioned_num_edges"] = edge_counts
        return result

    def forward(
        self,
        x: torch.Tensor,
        node_counts: torch.Tensor | None = None,
        edge_counts: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        if node_counts is None:
            node_counts = torch.round(x[:, 0] * float(self.max_nodes)).long()
        outputs = self.decode(z, node_counts, edge_counts)
        outputs.update(self.prior_parameters(node_counts, edge_counts))
        outputs["latent_z"] = z
        return outputs, mu, logvar

    @torch.no_grad()
    def reconstruct_outputs(
        self,
        x: torch.Tensor,
        node_counts: torch.Tensor,
        *,
        edge_counts: torch.Tensor | None = None,
        use_mean: bool = True,
    ) -> dict[str, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = mu if use_mean else self.reparameterize(mu, logvar)
        return self.decode(z, node_counts, edge_counts)

    @torch.no_grad()
    def sample_outputs(
        self,
        num_samples: int,
        *,
        node_counts: torch.Tensor | np.ndarray | list[int] | None = None,
        edge_counts: torch.Tensor | np.ndarray | list[int] | None = None,
        deterministic_node_count: bool = False,
        deterministic_edge_count: bool = False,
        prior_mode: str = "model",
        device: torch.device | str | None = None,
    ) -> dict[str, torch.Tensor]:
        if device is None:
            device = next(self.parameters()).device
        if node_counts is None:
            size_z = torch.randn(int(num_samples), self.latent_dim, device=device)
            logits = self.node_count_logits(size_z)
            if deterministic_node_count:
                indices = torch.argmax(logits, dim=-1)
            else:
                indices = torch.distributions.Categorical(logits=logits).sample()
            node_counts = indices + self.min_nodes
        else:
            node_counts = torch.as_tensor(node_counts, dtype=torch.long, device=device)
        if self.use_edge_count_conditioning:
            edge_logits = self.edge_count_logits(node_counts)
            if edge_counts is None:
                if deterministic_edge_count:
                    edge_counts = torch.argmax(edge_logits, dim=-1)
                else:
                    edge_counts = torch.distributions.Categorical(
                        logits=edge_logits
                    ).sample()
            else:
                edge_counts = torch.as_tensor(
                    edge_counts, dtype=torch.long, device=device
                )
        z = self.sample_prior(
            node_counts, edge_counts=edge_counts, prior_mode=prior_mode
        )
        return self.decode(z, node_counts, edge_counts)

    def model_config(self) -> dict[str, Any]:
        return {
            "architecture_version": self.architecture_version,
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "head_dims": self.head_dims,
            "min_nodes": self.min_nodes,
            "max_nodes": self.max_nodes,
            "max_degree": self.max_degree,
            "max_edges": self.max_edges,
            "require_connected": self.require_connected,
            "size_condition_dim": self.size_condition_dim,
            "edge_condition_dim": self.edge_condition_dim,
            "use_edge_count_conditioning": self.use_edge_count_conditioning,
            "prior_condition_on_edges": self.prior_condition_on_edges,
            "prior_type": self.prior_type,
            "prior_components": self.prior_components,
            "prior_hidden_dim": self.prior_hidden_dim,
            "prior_logvar_min": self.prior_logvar_min,
            "prior_logvar_max": self.prior_logvar_max,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
        }


def soft_histogram_ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    return -(target * log_probs).sum(dim=-1).mean()


def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=-1))


def _rbf_mmd_torch(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    sigma: float = 0.25,
) -> torch.Tensor:
    """Biased differentiable RBF MMD used as a prior-distribution surrogate."""

    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("RBF MMD inputs must be 2-D with matching feature width.")
    scale = max(float(sigma), 1.0e-6)
    gamma = 1.0 / (2.0 * scale * scale)
    k_xx = torch.exp(-gamma * torch.cdist(x, x, p=2).pow(2)).mean()
    k_yy = torch.exp(-gamma * torch.cdist(y, y, p=2).pow(2)).mean()
    k_xy = torch.exp(-gamma * torch.cdist(x, y, p=2).pow(2)).mean()
    return k_xx + k_yy - 2.0 * k_xy


def aggregate_prior_moment_loss(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    prior_logits: torch.Tensor,
    prior_means: torch.Tensor,
    prior_logvars: torch.Tensor,
) -> torch.Tensor:
    """Match first/second moments of q(z) and the learned conditional prior.

    The usual per-sample KL remains the principal regularizer.  This auxiliary
    aggregate term directly reduces the train-time aggregate-posterior / prior
    gap that can otherwise hurt unconditional sampling.
    """

    weights = F.softmax(prior_logits, dim=-1).unsqueeze(-1)
    prior_first = torch.sum(weights * prior_means, dim=1)
    prior_second = torch.sum(
        weights * (torch.exp(prior_logvars) + prior_means.pow(2)), dim=1
    )
    q_first = mu
    q_second = torch.exp(logvar) + mu.pow(2)
    first_loss = F.mse_loss(prior_first.mean(dim=0), q_first.mean(dim=0))
    second_loss = F.mse_loss(prior_second.mean(dim=0), q_second.mean(dim=0))
    return first_loss + second_loss


def conditional_prior_kl(
    z: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    prior_logits: torch.Tensor,
    prior_means: torch.Tensor,
    prior_logvars: torch.Tensor,
) -> torch.Tensor:
    """Monte Carlo KL(q(z|x) || p(z|n)) for a diagonal Gaussian-mixture prior."""

    log_two_pi = float(np.log(2.0 * np.pi))
    log_q = -0.5 * torch.sum(
        log_two_pi + logvar + (z - mu).pow(2) * torch.exp(-logvar),
        dim=-1,
    )
    expanded_z = z.unsqueeze(1)
    component_log_prob = -0.5 * torch.sum(
        log_two_pi
        + prior_logvars
        + (expanded_z - prior_means).pow(2) * torch.exp(-prior_logvars),
        dim=-1,
    )
    log_p = torch.logsumexp(
        F.log_softmax(prior_logits, dim=-1) + component_log_prob,
        dim=-1,
    )
    return torch.mean(log_q - log_p)


def degree_vae_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    mu: torch.Tensor,
    logvar: torch.Tensor,
    *,
    beta: float = 5.0e-3,
    weights: dict[str, float] | None = None,
    prior_outputs: dict[str, torch.Tensor] | None = None,
    prior_distribution_sigma: float = 0.25,
) -> tuple[torch.Tensor, dict[str, float]]:
    weights = weights or {}
    n_loss = F.cross_entropy(outputs["num_nodes_logits"], targets["num_nodes"].long())
    if "num_edges_logits" in outputs and "num_edges_count" in targets:
        edge_count_loss = F.cross_entropy(
            outputs["num_edges_logits"], targets["num_edges_count"].long()
        )
    else:
        edge_count_loss = torch.zeros((), device=mu.device, dtype=mu.dtype)
    degree_loss = soft_histogram_ce(outputs["degree_logits"], targets["degree"])
    predicted_mean_degree = outputs["expected_mean_degree"].reshape(-1)
    target_mean_degree = targets["mean_degree"].reshape(-1)
    degree_scale = max(float(outputs["degree_logits"].shape[-1] - 1), 1.0)
    moment_loss = F.mse_loss(
        predicted_mean_degree / degree_scale,
        target_mean_degree / degree_scale,
    )
    if {
        "latent_z",
        "prior_logits",
        "prior_means",
        "prior_logvars",
    }.issubset(outputs):
        kld = conditional_prior_kl(
            outputs["latent_z"],
            mu,
            logvar,
            outputs["prior_logits"],
            outputs["prior_means"],
            outputs["prior_logvars"],
        )
    else:
        kld = kl_loss(mu, logvar)

    if {"prior_logits", "prior_means", "prior_logvars"}.issubset(outputs):
        aggregate_moment = aggregate_prior_moment_loss(
            mu,
            logvar,
            outputs["prior_logits"],
            outputs["prior_means"],
            outputs["prior_logvars"],
        )
    else:
        aggregate_moment = torch.zeros((), device=mu.device, dtype=mu.dtype)

    if prior_outputs is not None and "degree_logits" in prior_outputs:
        prior_degree = F.softmax(prior_outputs["degree_logits"], dim=-1)
        target_degree = targets["degree"].to(dtype=prior_degree.dtype)
        prior_distribution = _rbf_mmd_torch(
            prior_degree,
            target_degree,
            sigma=prior_distribution_sigma,
        )
    else:
        prior_distribution = torch.zeros((), device=mu.device, dtype=mu.dtype)

    total = (
        float(weights.get("num_nodes", 1.0)) * n_loss
        + float(weights.get("num_edges", 0.0)) * edge_count_loss
        + float(weights.get("degree", 5.0)) * degree_loss
        + float(weights.get("degree_moment", weights.get("edge_scalar", 0.1)))
        * moment_loss
        + float(weights.get("aggregate_prior_moment", 0.0)) * aggregate_moment
        + float(weights.get("prior_distribution", 0.0)) * prior_distribution
        + float(beta) * kld
    )
    metrics = {
        "loss": float(total.detach().cpu()),
        "num_nodes_loss": float(n_loss.detach().cpu()),
        "num_edges_loss": float(edge_count_loss.detach().cpu()),
        "degree_loss": float(degree_loss.detach().cpu()),
        "degree_moment_loss": float(moment_loss.detach().cpu()),
        "aggregate_prior_moment_loss": float(aggregate_moment.detach().cpu()),
        "prior_distribution_loss": float(prior_distribution.detach().cpu()),
        "kl_loss": float(kld.detach().cpu()),
    }
    return total, metrics


def build_degree_vae(
    vectorizer: DegreeVectorizer,
    *,
    latent_dim: int = 32,
    hidden_dim: int = 128,
    size_condition_dim: int = 16,
    edge_condition_dim: int = 16,
    use_edge_count_conditioning: bool = False,
    prior_condition_on_edges: bool = False,
    prior_type: str = "conditional_gmm",
    prior_components: int = 4,
    prior_hidden_dim: int | None = None,
    prior_logvar_min: float = -6.0,
    prior_logvar_max: float = 4.0,
    num_layers: int = 2,
    dropout: float = 0.0,
) -> DegreeHistogramVAE:
    return DegreeHistogramVAE(
        input_dim=vectorizer.input_dim,
        latent_dim=int(latent_dim),
        hidden_dim=int(hidden_dim),
        head_dims=vectorizer.head_dims(),
        min_nodes=vectorizer.min_nodes,
        max_nodes=vectorizer.max_nodes,
        max_degree=vectorizer.max_degree,
        max_edges=vectorizer.max_edges,
        require_connected=vectorizer.require_connected,
        size_condition_dim=int(size_condition_dim),
        edge_condition_dim=int(edge_condition_dim),
        use_edge_count_conditioning=bool(use_edge_count_conditioning),
        prior_condition_on_edges=bool(prior_condition_on_edges),
        prior_type=str(prior_type),
        prior_components=int(prior_components),
        prior_hidden_dim=prior_hidden_dim,
        prior_logvar_min=float(prior_logvar_min),
        prior_logvar_max=float(prior_logvar_max),
        num_layers=int(num_layers),
        dropout=float(dropout),
    )


def save_degree_vae_checkpoint(
    path: str | Path,
    model: DegreeHistogramVAE,
    vectorizer: DegreeVectorizer,
    *,
    config: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model.model_config(),
            "vectorizer": vectorizer.__dict__,
            "config": config or {},
            "metrics": metrics or {},
        },
        path,
    )


def load_degree_vae_checkpoint(
    path: str | Path, *, device: torch.device | str = "auto"
) -> tuple[DegreeHistogramVAE, DegreeVectorizer, dict[str, Any]]:
    resolved_device = resolve_torch_device(device)
    checkpoint = torch.load(path, map_location=resolved_device)
    model_config = dict(checkpoint.get("model_config", {}))
    architecture_version = int(model_config.get("architecture_version", 1))
    if architecture_version == 1:
        raise RuntimeError(
            "This checkpoint uses the old unconditional DH-VAE decoder. "
            "Retrain it with scripts/train_degree_generator.py so decoding is "
            "conditioned on graph size."
        )
    if architecture_version == 2:
        # Version 2 is the corrected size-conditioned decoder with a fixed
        # standard-normal prior. Keep it loadable for baseline evaluation.
        model_config.setdefault("prior_type", "standard_normal")
        model_config.setdefault("prior_components", 1)
        model_config.setdefault("use_edge_count_conditioning", False)
        model_config.setdefault("prior_condition_on_edges", False)
        model_config.setdefault("require_connected", True)
    elif architecture_version == 3:
        # Version 3 adds the learned conditional latent prior but still decodes
        # a degree histogram without an explicit edge-count condition.
        model_config.setdefault("use_edge_count_conditioning", False)
        model_config.setdefault("prior_condition_on_edges", False)
        model_config.setdefault("require_connected", True)
    elif architecture_version != 4:
        raise RuntimeError(
            f"Unsupported DH-VAE architecture version {architecture_version}."
        )
    model_config.pop("architecture_version", None)
    vectorizer = DegreeVectorizer(**checkpoint["vectorizer"])
    model = DegreeHistogramVAE(**model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(resolved_device)
    model.eval()
    return model, vectorizer, checkpoint

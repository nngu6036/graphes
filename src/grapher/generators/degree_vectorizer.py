from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from grapher.generators.summary_vectorizer import repair_degree_sequence
from grapher.properties.summary import degree_histogram, sorted_degree_sequence


def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    total = float(x.sum())
    if total <= 0:
        out = np.zeros_like(x, dtype=np.float64)
        if out.size:
            out[0] = 1.0
        return out
    return x / total


def _pad(x: np.ndarray, width: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    out = np.zeros(int(width), dtype=np.float64)
    if x.size:
        out[: min(x.size, int(width))] = x[: int(width)]
    return out


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
    for d in sequence:
        d = int(d)
        if 0 <= d < width:
            out[d] += 1
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
        for i in order[: -remaining]:
            if counts[int(i)] > 0:
                counts[int(i)] -= 1
    return counts


def connected_feasible_degree_sequence(sequence: list[int]) -> bool:
    n = len(sequence)
    if n <= 1:
        return True
    if min(sequence) <= 0:
        return False
    if sum(sequence) < 2 * (n - 1):
        return False
    return nx.is_graphical(sorted([int(d) for d in sequence], reverse=True), method="eg")


@dataclass
class DegreeVectorizer:
    """Vectorizer and post-processor for a degree-sequence generator.

    The degree generator models only permutation-invariant degree information.
    It saves empirical node counts and training degree sequences so sampling can
    use empirical graph sizes and robust fallback degree sequences.
    """

    min_nodes: int
    max_nodes: int
    max_degree: int
    max_edges: int
    require_connected: bool = True
    empirical_node_counts: list[int] | None = None
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
        degree_sequences = [sorted_degree_sequence(g) for g in graphs]
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
            empirical_degree_sequences=[[int(d) for d in seq] for seq in degree_sequences],
        )

    @property
    def node_count_classes(self) -> int:
        return int(self.max_nodes - self.min_nodes + 1)

    @property
    def degree_dim(self) -> int:
        return int(self.max_degree + 1)

    @property
    def input_dim(self) -> int:
        # Scalars: normalized n and normalized edge count.
        return int(2 + self.degree_dim)

    def head_dims(self) -> dict[str, int]:
        return {
            "num_nodes": self.node_count_classes,
            "degree": self.degree_dim,
            "edge_scalar": 1,
        }

    def node_index(self, n: int) -> int:
        return int(np.clip(int(n) - self.min_nodes, 0, self.node_count_classes - 1))

    def node_count_from_index(self, idx: int) -> int:
        return int(self.min_nodes + int(np.clip(idx, 0, self.node_count_classes - 1)))

    def degree_hist_from_sequence(self, sequence: list[int]) -> np.ndarray:
        return _normalize(_degree_sequence_to_counts(sequence, self.degree_dim).astype(np.float64))

    def to_feature_vector(self, graph: nx.Graph) -> np.ndarray:
        seq = sorted_degree_sequence(graph)
        n = int(graph.number_of_nodes())
        m = int(graph.number_of_edges())
        degree = self.degree_hist_from_sequence(seq)
        scalars = np.asarray(
            [
                n / max(float(self.max_nodes), 1.0),
                m / max(float(self.max_edges), 1.0),
            ],
            dtype=np.float64,
        )
        return np.concatenate([scalars, degree]).astype(np.float32)

    def to_targets(self, graph: nx.Graph) -> dict[str, np.ndarray | np.int64]:
        seq = sorted_degree_sequence(graph)
        n = int(graph.number_of_nodes())
        m = int(graph.number_of_edges())
        return {
            "num_nodes": np.int64(self.node_index(n)),
            "degree": self.degree_hist_from_sequence(seq).astype(np.float32),
            "edge_scalar": np.asarray([m / max(float(self.max_edges), 1.0)], dtype=np.float32),
        }

    def to_training_arrays(self, graphs: list[nx.Graph]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
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

    def empirical_nearest_degree_sequence(self, n: int, rng: np.random.Generator | None = None) -> list[int]:
        generator = rng if rng is not None else np.random.default_rng(0)
        sequences = self.empirical_degree_sequences or []
        if not sequences:
            # Last-resort path-like connected sequence.
            if n <= 1:
                return [0] * int(n)
            seq = [2] * int(n)
            seq[0] = seq[-1] = 1
            return sorted(seq, reverse=True)
        distances = np.asarray([abs(len(seq) - int(n)) for seq in sequences], dtype=np.int64)
        best = np.flatnonzero(distances == distances.min())
        chosen = sequences[int(generator.choice(best))]
        seq = [int(d) for d in chosen]
        if len(seq) == int(n):
            return sorted(seq, reverse=True)
        # Adapt a nearby empirical sequence and repair.
        if len(seq) > int(n):
            seq = seq[: int(n)]
        else:
            min_degree = 1 if self.require_connected and n > 1 else 0
            seq = seq + [min_degree] * (int(n) - len(seq))
        return repair_degree_sequence(seq, n=int(n), require_connected=self.require_connected, rng=generator)

    def outputs_to_summaries(
        self,
        outputs: dict[str, Any],
        *,
        rng: np.random.Generator | None = None,
        deterministic: bool = False,
        sample_num_nodes: str = "empirical",
        max_resample: int = 200,
        fallback: str = "empirical_nearest_n",
    ) -> list[dict[str, Any]]:
        import torch

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
            if str(sample_num_nodes).lower() == "empirical":
                n = self.sample_empirical_node_count(generator)
            else:
                if deterministic:
                    n_idx = int(np.argmax(n_probs))
                else:
                    n_idx = int(generator.choice(np.arange(self.node_count_classes), p=n_probs))
                n = self.node_count_from_index(n_idx)

            degree_probs = _softmax_np(arrays["degree_logits"][i]).astype(np.float64)
            if n < degree_probs.size:
                degree_probs[n:] = 0.0
            if self.require_connected and n > 1:
                degree_probs[0] = 0.0
            degree_probs = _normalize(degree_probs)

            degree_sequence: list[int] | None = None
            for _ in range(max(int(max_resample), 1)):
                if deterministic:
                    counts = _integer_counts_from_probs(n, degree_probs)
                else:
                    counts = generator.multinomial(int(n), degree_probs)
                seq = _degree_counts_to_sequence(counts)
                seq = repair_degree_sequence(seq, n=int(n), require_connected=self.require_connected, rng=generator)
                if nx.is_graphical(seq, method="eg") and (not self.require_connected or connected_feasible_degree_sequence(seq)):
                    degree_sequence = seq
                    break
            if degree_sequence is None:
                if fallback == "error":
                    raise RuntimeError("Degree generator failed to sample a graphical degree sequence.")
                degree_sequence = self.empirical_nearest_degree_sequence(n, generator)

            counts = _degree_sequence_to_counts(degree_sequence, self.degree_dim).astype(np.float64)
            degree_hist = _normalize(counts)
            num_edges = int(sum(degree_sequence) // 2)
            density = (2.0 * num_edges / (int(n) * (int(n) - 1))) if int(n) > 1 else 0.0
            summaries.append(
                {
                    "num_nodes": int(n),
                    "num_edges": int(num_edges),
                    "degree_sequence": sorted([int(d) for d in degree_sequence], reverse=True),
                    "degree_hist": degree_hist.astype(np.float64),
                    "density": float(density),
                }
            )
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

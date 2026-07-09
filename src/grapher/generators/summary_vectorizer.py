from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from grapher.properties.summary import SummaryConfig, extract_summary
from grapher.utils.motifs import (
    flatten_graphlet_history,
    graphlet_keys_by_size,
    unflatten_graphlet_history,
)


@dataclass
class SummaryVectorizer:
    """Convert graph-summary dictionaries to fixed arrays and back.

    The learned summary generator works on fixed-size tensors, but the rest of
    the pipeline expects a summary dictionary with a valid degree sequence. This
    class owns both conversions and the post-processing needed before the coarse
    graph constructor is called.

    New graphlet-history fields are optional and default to empty, so older
    checkpoints remain loadable.
    """

    min_nodes: int
    max_nodes: int
    max_edges: int
    max_degree: int
    clustering_bins: int
    spectral_bins: int
    motif_dim: int
    orbit_dim: int
    motif_scale: list[float]
    orbit_scale: list[float]
    scalar_scale: list[float]
    require_connected: bool = True
    graphlet_k_min: int = 3
    graphlet_k_max: int = 5
    graphlet_keys_by_k: dict[str, list[str]] | None = None
    graphlet_dim: int = 0

    @classmethod
    def fit(
        cls,
        summaries: list[dict[str, Any]],
        config: SummaryConfig | dict[str, Any] | None = None,
        *,
        require_connected: bool = True,
    ) -> "SummaryVectorizer":
        if not summaries:
            raise ValueError("Cannot fit SummaryVectorizer on an empty summary list.")
        cfg = config if isinstance(config, SummaryConfig) else SummaryConfig.from_dict(config or {})
        min_nodes = min(int(s["num_nodes"]) for s in summaries)
        max_nodes = max(int(s["num_nodes"]) for s in summaries)
        max_edges = max(int(s.get("num_edges", 0)) for s in summaries)
        max_degree = cfg.degree_hist_max_degree
        if max_degree is None:
            max_degree = max(max([0] + [int(d) for d in s.get("degree_sequence", [])]) for s in summaries)
        clustering_bins = int(cfg.clustering_bins)
        spectral_bins = int(cfg.spectral_bins)
        motif_dim = max((np.asarray(s.get("motif_proxy", []), dtype=np.float64).size for s in summaries), default=0)
        orbit_dim = max((np.asarray(s.get("orbit_count", []), dtype=np.float64).size for s in summaries), default=0)

        motif_logs = np.zeros((len(summaries), motif_dim), dtype=np.float64)
        orbit_logs = np.zeros((len(summaries), orbit_dim), dtype=np.float64)
        for i, s in enumerate(summaries):
            motif = np.log1p(_pad(np.asarray(s.get("motif_proxy", []), dtype=np.float64), motif_dim))
            orbit = np.log1p(_pad(np.asarray(s.get("orbit_count", []), dtype=np.float64), orbit_dim))
            if motif_dim:
                motif_logs[i] = motif
            if orbit_dim:
                orbit_logs[i] = orbit
        motif_scale = np.maximum(motif_logs.max(axis=0), 1.0).tolist() if motif_dim else []
        orbit_scale = np.maximum(orbit_logs.max(axis=0), 1.0).tolist() if orbit_dim else []
        triangle_max = max(float(s.get("triangle_count_norm", 0.0)) for s in summaries)
        scalar_scale = [1.0, max(triangle_max, 1.0)]

        graphlet_keys = graphlet_keys_by_size([s.get("graphlet_history", {}) or {} for s in summaries]) if cfg.graphlet_history else {}
        # Ensure all requested k values exist, even if no graphlets of that size appeared.
        if cfg.graphlet_history:
            for k in range(int(cfg.graphlet_k_min), int(cfg.graphlet_k_max) + 1):
                graphlet_keys.setdefault(str(k), [])
        graphlet_dim = int(sum(len(v) for v in graphlet_keys.values()))

        return cls(
            min_nodes=int(min_nodes),
            max_nodes=int(max_nodes),
            max_edges=int(max(max_edges, 1)),
            max_degree=int(max(max_degree or 1, 1)),
            clustering_bins=clustering_bins,
            spectral_bins=spectral_bins,
            motif_dim=int(motif_dim),
            orbit_dim=int(orbit_dim),
            motif_scale=motif_scale,
            orbit_scale=orbit_scale,
            scalar_scale=scalar_scale,
            require_connected=bool(require_connected),
            graphlet_k_min=int(cfg.graphlet_k_min),
            graphlet_k_max=int(cfg.graphlet_k_max),
            graphlet_keys_by_k={str(k): list(v) for k, v in graphlet_keys.items()},
            graphlet_dim=graphlet_dim,
        )

    @classmethod
    def fit_from_graphs(
        cls,
        graphs: list[nx.Graph],
        config: SummaryConfig | dict[str, Any] | None = None,
        *,
        require_connected: bool = True,
    ) -> "SummaryVectorizer":
        cfg = config if isinstance(config, SummaryConfig) else SummaryConfig.from_dict(config or {}, graphs)
        summaries = [extract_summary(g, cfg) for g in graphs]
        return cls.fit(summaries, cfg, require_connected=require_connected)

    @property
    def node_count_classes(self) -> int:
        return int(self.max_nodes - self.min_nodes + 1)

    @property
    def degree_dim(self) -> int:
        return int(self.max_degree + 1)

    @property
    def input_dim(self) -> int:
        return int(4 + self.degree_dim + self.clustering_bins + self.spectral_bins + self.motif_dim + self.orbit_dim + self.graphlet_dim)

    def head_dims(self) -> dict[str, int]:
        return {
            "num_nodes": self.node_count_classes,
            "degree": self.degree_dim,
            "clustering": self.clustering_bins,
            "spectral": self.spectral_bins,
            "motif": self.motif_dim,
            "orbit": self.orbit_dim,
            "graphlet": self.graphlet_dim,
            "scalar": 2,
        }

    def graphlet_slices(self) -> dict[str, slice]:
        out: dict[str, slice] = {}
        pos = 0
        for k in sorted((self.graphlet_keys_by_k or {}).keys(), key=lambda x: int(x)):
            width = len((self.graphlet_keys_by_k or {}).get(k, []))
            out[str(k)] = slice(pos, pos + width)
            pos += width
        return out

    def graphlet_to_vector(self, summary: dict[str, Any]) -> np.ndarray:
        return flatten_graphlet_history(summary.get("graphlet_history", {}) or {}, self.graphlet_keys_by_k or {})

    def to_feature_vector(self, summary: dict[str, Any]) -> np.ndarray:
        n = float(summary.get("num_nodes", 0.0))
        m = float(summary.get("num_edges", 0.0))
        density = float(summary.get("density", 0.0))
        triangle = float(summary.get("triangle_count_norm", 0.0))
        scalars = np.asarray(
            [
                n / max(float(self.max_nodes), 1.0),
                m / max(float(self.max_edges), 1.0),
                density,
                triangle / max(float(self.scalar_scale[1]), 1.0),
            ],
            dtype=np.float64,
        )
        degree = _normalize(_pad(np.asarray(summary.get("degree_hist", []), dtype=np.float64), self.degree_dim))
        clustering = _normalize(_pad(np.asarray(summary.get("clustering_hist", []), dtype=np.float64), self.clustering_bins))
        spectral = _normalize(_pad(np.asarray(summary.get("spectral_hist", []), dtype=np.float64), self.spectral_bins))
        motif = np.log1p(_pad(np.asarray(summary.get("motif_proxy", []), dtype=np.float64), self.motif_dim))
        orbit = np.log1p(_pad(np.asarray(summary.get("orbit_count", []), dtype=np.float64), self.orbit_dim))
        graphlet = _pad(self.graphlet_to_vector(summary), self.graphlet_dim)
        if self.motif_dim:
            motif = motif / np.asarray(self.motif_scale, dtype=np.float64)
        if self.orbit_dim:
            orbit = orbit / np.asarray(self.orbit_scale, dtype=np.float64)
        return np.concatenate([scalars, degree, clustering, spectral, motif, orbit, graphlet]).astype(np.float32)

    def to_targets(self, summary: dict[str, Any]) -> dict[str, np.ndarray | np.int64]:
        node_index = int(np.clip(int(summary["num_nodes"]) - self.min_nodes, 0, self.node_count_classes - 1))
        degree = _normalize(_pad(np.asarray(summary.get("degree_hist", []), dtype=np.float64), self.degree_dim))
        clustering = _normalize(_pad(np.asarray(summary.get("clustering_hist", []), dtype=np.float64), self.clustering_bins))
        spectral = _normalize(_pad(np.asarray(summary.get("spectral_hist", []), dtype=np.float64), self.spectral_bins))
        motif = np.log1p(_pad(np.asarray(summary.get("motif_proxy", []), dtype=np.float64), self.motif_dim))
        orbit = np.log1p(_pad(np.asarray(summary.get("orbit_count", []), dtype=np.float64), self.orbit_dim))
        graphlet = _pad(self.graphlet_to_vector(summary), self.graphlet_dim)
        if self.motif_dim:
            motif = motif / np.asarray(self.motif_scale, dtype=np.float64)
        if self.orbit_dim:
            orbit = orbit / np.asarray(self.orbit_scale, dtype=np.float64)
        scalars = np.asarray(
            [
                float(summary.get("density", 0.0)),
                float(summary.get("triangle_count_norm", 0.0)) / max(float(self.scalar_scale[1]), 1.0),
            ],
            dtype=np.float64,
        )
        return {
            "num_nodes": np.int64(node_index),
            "degree": degree.astype(np.float32),
            "clustering": clustering.astype(np.float32),
            "spectral": spectral.astype(np.float32),
            "motif": motif.astype(np.float32),
            "orbit": orbit.astype(np.float32),
            "graphlet": graphlet.astype(np.float32),
            "scalar": scalars.astype(np.float32),
        }

    def to_training_arrays(self, summaries: list[dict[str, Any]]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        x = np.stack([self.to_feature_vector(s) for s in summaries]).astype(np.float32)
        target_items = [self.to_targets(s) for s in summaries]
        targets: dict[str, np.ndarray] = {}
        for key in target_items[0]:
            targets[key] = np.asarray([item[key] for item in target_items])
        return x, targets

    def outputs_to_summaries(
        self,
        outputs: dict[str, Any],
        *,
        rng: np.random.Generator | None = None,
        deterministic: bool = False,
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
        summaries = []
        for i in range(batch):
            n_logits = arrays["num_nodes_logits"][i]
            n_probs = _softmax_np(n_logits)
            if deterministic:
                node_index = int(np.argmax(n_probs))
            else:
                node_index = int(generator.choice(np.arange(self.node_count_classes), p=n_probs))
            n = int(self.min_nodes + node_index)

            degree_probs = _softmax_np(arrays["degree_logits"][i]).astype(np.float64)
            if n < degree_probs.size:
                degree_probs[n:] = 0.0
            if self.require_connected and n > 1:
                degree_probs[0] = 0.0
            degree_probs = _normalize(degree_probs)
            if deterministic:
                degree_counts = _integer_counts_from_probs(n, degree_probs)
            else:
                degree_counts = generator.multinomial(n, degree_probs)
            degree_sequence = _degree_counts_to_sequence(degree_counts)
            degree_sequence = repair_degree_sequence(degree_sequence, n=n, require_connected=self.require_connected, rng=generator)
            degree_hist = _degree_sequence_to_hist(degree_sequence, self.degree_dim)

            clustering = _normalize(_softmax_np(arrays["clustering_logits"][i]))
            spectral = _normalize(_softmax_np(arrays["spectral_logits"][i]))

            motif = np.zeros(self.motif_dim, dtype=np.float64)
            if self.motif_dim:
                motif_scaled = np.maximum(arrays["motif_log"][i], 0.0)
                motif = np.expm1(motif_scaled * np.asarray(self.motif_scale, dtype=np.float64))
            orbit = np.zeros(self.orbit_dim, dtype=np.float64)
            if self.orbit_dim:
                orbit_scaled = np.maximum(arrays["orbit_log"][i], 0.0)
                orbit = np.expm1(orbit_scaled * np.asarray(self.orbit_scale, dtype=np.float64))

            graphlet_history = {}
            if self.graphlet_dim:
                raw = np.maximum(arrays.get("graphlet", np.zeros((batch, self.graphlet_dim), dtype=np.float64))[i], 0.0)
                # Normalize each k-slice independently so every h_k is a frequency vector.
                graphlet_vec = np.zeros(self.graphlet_dim, dtype=np.float64)
                for _, sl in self.graphlet_slices().items():
                    graphlet_vec[sl] = _normalize(raw[sl]) if sl.stop > sl.start else raw[sl]
                graphlet_history = unflatten_graphlet_history(graphlet_vec, self.graphlet_keys_by_k or {})

            scalar = arrays.get("scalar", np.zeros((batch, 2), dtype=np.float64))[i]
            triangle = max(float(scalar[1]) * max(float(self.scalar_scale[1]), 1.0), 0.0) if scalar.size > 1 else 0.0
            num_edges = int(sum(degree_sequence) // 2)
            density = (2.0 * num_edges / (n * (n - 1))) if n > 1 else 0.0
            summaries.append(
                {
                    "num_nodes": n,
                    "num_edges": num_edges,
                    "degree_sequence": sorted([int(d) for d in degree_sequence], reverse=True),
                    "density": float(density),
                    "triangle_count_norm": float(triangle),
                    "degree_hist": degree_hist,
                    "clustering_hist": clustering.astype(np.float64),
                    "spectral_hist": spectral.astype(np.float64),
                    "motif_proxy": motif.astype(np.float64),
                    "orbit_count": orbit.astype(np.float64),
                    "graphlet_history": graphlet_history,
                }
            )
        return summaries

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: str | Path) -> "SummaryVectorizer":
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


def _pad(x: np.ndarray, width: int) -> np.ndarray:
    if width <= 0:
        return np.zeros(0, dtype=np.float64)
    out = np.zeros(width, dtype=np.float64)
    flat = np.asarray(x, dtype=np.float64).reshape(-1)
    out[: min(width, flat.size)] = flat[:width]
    return out


def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.maximum(x, 0.0)
    total = float(np.sum(x))
    if total <= 0.0:
        out = np.zeros_like(x, dtype=np.float64)
        if out.size:
            out[0] = 1.0
        return out
    return x / total


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - np.max(logits)
    exp = np.exp(logits)
    return _normalize(exp)


def _integer_counts_from_probs(n: int, probs: np.ndarray) -> np.ndarray:
    raw = np.asarray(probs, dtype=np.float64) * int(n)
    counts = np.floor(raw).astype(int)
    remainder = int(n - counts.sum())
    if remainder > 0:
        order = np.argsort(-(raw - counts))
        for idx in order[:remainder]:
            counts[int(idx)] += 1
    elif remainder < 0:
        order = np.argsort(raw - counts)
        for idx in order[: -remainder]:
            if counts[int(idx)] > 0:
                counts[int(idx)] -= 1
    return counts


def _degree_counts_to_sequence(counts: np.ndarray) -> list[int]:
    seq: list[int] = []
    for degree, count in enumerate(np.asarray(counts, dtype=int).reshape(-1)):
        seq.extend([int(degree)] * int(max(count, 0)))
    return seq


def _degree_sequence_to_hist(degree_sequence: list[int], width: int) -> np.ndarray:
    hist = np.zeros(width, dtype=np.float64)
    for degree in degree_sequence:
        d = int(np.clip(degree, 0, width - 1))
        hist[d] += 1.0
    return _normalize(hist)


def repair_degree_sequence(
    degree_sequence: list[int],
    *,
    n: int,
    require_connected: bool = True,
    rng: np.random.Generator | None = None,
    max_iterations: int = 10000,
) -> list[int]:
    """Repair sampled degrees into a graphical sequence.

    The repair is intentionally conservative. It clips impossible degrees,
    enforces a positive minimum degree when connected graphs are required, fixes
    parity, and then adjusts degrees until NetworkX's Erdős-Gallai checker
    accepts the sequence.
    """

    generator = rng if rng is not None else np.random.default_rng(0)
    min_degree = 1 if require_connected and n > 1 else 0
    max_degree = max(n - 1, 0)
    seq = [int(np.clip(d, min_degree, max_degree)) for d in degree_sequence[:n]]
    while len(seq) < n:
        seq.append(min_degree)
    if n <= 1:
        return [0] * n

    def fix_parity() -> None:
        if sum(seq) % 2 == 0:
            return
        order = generator.permutation(n)
        for idx in order:
            i = int(idx)
            if seq[i] < max_degree:
                seq[i] += 1
                return
        for idx in order:
            i = int(idx)
            if seq[i] > min_degree:
                seq[i] -= 1
                return

    # Connected simple graphs need at least n - 1 edges.
    if require_connected:
        while sum(seq) < 2 * (n - 1):
            candidates = [i for i, d in enumerate(seq) if d < max_degree]
            if not candidates:
                break
            seq[int(generator.choice(candidates))] += 1
    fix_parity()

    for _ in range(max_iterations):
        if nx.is_graphical(sorted(seq, reverse=True), method="eg"):
            return sorted([int(d) for d in seq], reverse=True)
        # If the sequence is too heavy/non-graphical, reduce a high degree.
        candidates = [i for i, d in enumerate(seq) if d > min_degree]
        if candidates:
            max_d = max(seq[i] for i in candidates)
            high = [i for i in candidates if seq[i] == max_d]
            seq[int(generator.choice(high))] -= 1
        else:
            # Fallback for path-like connected sequence.
            seq = [1] * n
            seq[0] = seq[1] = 2 if n > 2 else 1
        fix_parity()

    # Last-resort connected sequence: a path degree sequence.
    if require_connected and n >= 2:
        seq = [2] * n
        seq[0] = 1
        seq[-1] = 1
        return sorted(seq, reverse=True)
    return sorted(seq, reverse=True)

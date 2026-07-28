from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from grapher.properties.summary import SummaryConfig, extract_summary
from grapher.utils.device import resolve_torch_device
from grapher.utils.motifs import (
    flatten_graphlet_history,
    graphlet_keys_by_size,
    unflatten_graphlet_history,
)


@dataclass
class SummaryVectorizer:
    """Convert graph-summary dictionaries to fixed arrays and back."""

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
    ) -> SummaryVectorizer:
        if not summaries:
            raise ValueError("Cannot fit SummaryVectorizer on an empty summary list.")
        cfg = (
            config
            if isinstance(config, SummaryConfig)
            else SummaryConfig.from_dict(config or {})
        )
        min_nodes = min(int(s["num_nodes"]) for s in summaries)
        max_nodes = max(int(s["num_nodes"]) for s in summaries)
        max_edges = max(int(s.get("num_edges", 0)) for s in summaries)
        max_degree = cfg.degree_hist_max_degree
        if max_degree is None:
            max_degree = max(
                max([0] + [int(d) for d in s.get("degree_sequence", [])])
                for s in summaries
            )
        clustering_bins = int(cfg.clustering_bins)
        spectral_bins = int(cfg.spectral_bins)
        motif_dim = max(
            (
                np.asarray(s.get("motif_proxy", []), dtype=np.float64).size
                for s in summaries
            ),
            default=0,
        )
        orbit_dim = max(
            (
                np.asarray(s.get("orbit_count", []), dtype=np.float64).size
                for s in summaries
            ),
            default=0,
        )

        motif_logs = np.zeros((len(summaries), motif_dim), dtype=np.float64)
        orbit_logs = np.zeros((len(summaries), orbit_dim), dtype=np.float64)
        for i, s in enumerate(summaries):
            motif = np.log1p(
                _pad(np.asarray(s.get("motif_proxy", []), dtype=np.float64), motif_dim)
            )
            orbit = np.log1p(
                _pad(np.asarray(s.get("orbit_count", []), dtype=np.float64), orbit_dim)
            )
            if motif_dim:
                motif_logs[i] = motif
            if orbit_dim:
                orbit_logs[i] = orbit
        motif_scale = (
            np.maximum(motif_logs.max(axis=0), 1.0).tolist() if motif_dim else []
        )
        orbit_scale = (
            np.maximum(orbit_logs.max(axis=0), 1.0).tolist() if orbit_dim else []
        )
        triangle_max = max(float(s.get("triangle_count_norm", 0.0)) for s in summaries)
        scalar_scale = [1.0, max(triangle_max, 1.0)]

        graphlet_keys = (
            graphlet_keys_by_size(
                [s.get("graphlet_history", {}) or {} for s in summaries]
            )
            if cfg.graphlet_history
            else {}
        )
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
    ) -> SummaryVectorizer:
        cfg = (
            config
            if isinstance(config, SummaryConfig)
            else SummaryConfig.from_dict(config or {}, graphs)
        )
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
        return int(
            4
            + self.degree_dim
            + self.clustering_bins
            + self.spectral_bins
            + self.motif_dim
            + self.orbit_dim
            + self.graphlet_dim
        )

    @property
    def condition_dim(self) -> int:
        """Width of the fixed-degree condition used by the target-summary CVAE."""

        return int(2 + self.degree_dim)

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
        return flatten_graphlet_history(
            summary.get("graphlet_history", {}) or {}, self.graphlet_keys_by_k or {}
        )

    def normalize_graphlet_vector(self, values: np.ndarray) -> np.ndarray:
        """Normalize every graphlet-size block as its own categorical law."""

        vector = _pad(np.asarray(values, dtype=np.float64), self.graphlet_dim)
        for graphlet_slice in self.graphlet_slices().values():
            if graphlet_slice.stop > graphlet_slice.start:
                vector[graphlet_slice] = _normalize(vector[graphlet_slice])
        return vector

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
        degree = _normalize(
            _pad(
                np.asarray(summary.get("degree_hist", []), dtype=np.float64),
                self.degree_dim,
            )
        )
        clustering = _normalize(
            _pad(
                np.asarray(summary.get("clustering_hist", []), dtype=np.float64),
                self.clustering_bins,
            )
        )
        spectral = _normalize(
            _pad(
                np.asarray(summary.get("spectral_hist", []), dtype=np.float64),
                self.spectral_bins,
            )
        )
        motif = np.log1p(
            _pad(
                np.asarray(summary.get("motif_proxy", []), dtype=np.float64),
                self.motif_dim,
            )
        )
        orbit = np.log1p(
            _pad(
                np.asarray(summary.get("orbit_count", []), dtype=np.float64),
                self.orbit_dim,
            )
        )
        graphlet = self.normalize_graphlet_vector(
            self.graphlet_to_vector(summary)
        )
        if self.motif_dim:
            motif = motif / np.asarray(self.motif_scale, dtype=np.float64)
        if self.orbit_dim:
            orbit = orbit / np.asarray(self.orbit_scale, dtype=np.float64)
        return np.concatenate(
            [scalars, degree, clustering, spectral, motif, orbit, graphlet]
        ).astype(np.float32)

    def to_condition_vector(self, summary: dict[str, Any]) -> np.ndarray:
        """Encode only invariants fixed before refinement.

        The number of nodes, number of edges, and degree histogram completely
        describe the information that the target-summary generator is allowed
        to condition on.  The Havel--Hakimi graph is intentionally omitted
        because it is a deterministic realization of the same degree sequence.
        """

        n = float(summary.get("num_nodes", len(summary.get("degree_sequence", []))))
        sequence = [int(d) for d in summary.get("degree_sequence", [])]
        m = float(summary.get("num_edges", sum(sequence) // 2))
        degree = np.asarray(summary.get("degree_hist", []), dtype=np.float64)
        if degree.size == 0 and sequence:
            degree = _degree_sequence_to_hist(sequence, self.degree_dim)
        degree = _normalize(_pad(degree, self.degree_dim))
        return np.concatenate(
            [
                np.asarray(
                    [
                        n / max(float(self.max_nodes), 1.0),
                        m / max(float(self.max_edges), 1.0),
                    ],
                    dtype=np.float64,
                ),
                degree,
            ]
        ).astype(np.float32)

    def to_targets(self, summary: dict[str, Any]) -> dict[str, np.ndarray | np.int64]:
        node_index = int(
            np.clip(
                int(summary["num_nodes"]) - self.min_nodes,
                0,
                self.node_count_classes - 1,
            )
        )
        degree = _normalize(
            _pad(
                np.asarray(summary.get("degree_hist", []), dtype=np.float64),
                self.degree_dim,
            )
        )
        clustering = _normalize(
            _pad(
                np.asarray(summary.get("clustering_hist", []), dtype=np.float64),
                self.clustering_bins,
            )
        )
        spectral = _normalize(
            _pad(
                np.asarray(summary.get("spectral_hist", []), dtype=np.float64),
                self.spectral_bins,
            )
        )
        motif = np.log1p(
            _pad(
                np.asarray(summary.get("motif_proxy", []), dtype=np.float64),
                self.motif_dim,
            )
        )
        orbit = np.log1p(
            _pad(
                np.asarray(summary.get("orbit_count", []), dtype=np.float64),
                self.orbit_dim,
            )
        )
        graphlet = self.normalize_graphlet_vector(
            self.graphlet_to_vector(summary)
        )
        if self.motif_dim:
            motif = motif / np.asarray(self.motif_scale, dtype=np.float64)
        if self.orbit_dim:
            orbit = orbit / np.asarray(self.orbit_scale, dtype=np.float64)
        scalars = np.asarray(
            [
                float(summary.get("density", 0.0)),
                float(summary.get("triangle_count_norm", 0.0))
                / max(float(self.scalar_scale[1]), 1.0),
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

    def to_training_arrays(
        self, summaries: list[dict[str, Any]]
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        x = np.stack([self.to_feature_vector(s) for s in summaries]).astype(np.float32)
        target_items = [self.to_targets(s) for s in summaries]
        targets: dict[str, np.ndarray] = {}
        for key in target_items[0]:
            targets[key] = np.asarray([item[key] for item in target_items])
        return x, targets

    def to_condition_array(self, summaries: list[dict[str, Any]]) -> np.ndarray:
        return np.stack([self.to_condition_vector(s) for s in summaries]).astype(
            np.float32
        )

    def outputs_to_summaries(
        self,
        outputs: dict[str, Any],
        *,
        rng: np.random.Generator | None = None,
        deterministic: bool = False,
        condition_summaries: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        generator = rng if rng is not None else np.random.default_rng(0)
        arrays: dict[str, np.ndarray] = {}
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                arrays[key] = value.detach().cpu().numpy()
            else:
                arrays[key] = np.asarray(value)
        batch = int(next(iter(arrays.values())).shape[0])
        if condition_summaries is not None and len(condition_summaries) != batch:
            raise ValueError(
                "condition_summaries must contain one degree condition per output "
                f"(expected {batch}, got {len(condition_summaries)})."
            )
        summaries = []
        for i in range(batch):
            if condition_summaries is not None:
                condition_summary = condition_summaries[i]
                degree_sequence = sorted(
                    [int(d) for d in condition_summary.get("degree_sequence", [])],
                    reverse=True,
                )
                n = int(condition_summary.get("num_nodes", len(degree_sequence)))
                if len(degree_sequence) != n:
                    raise ValueError(
                        "Conditional summary has inconsistent num_nodes and "
                        "degree_sequence."
                    )
                degree_hist = np.asarray(
                    condition_summary.get("degree_hist", []),
                    dtype=np.float64,
                )
                if degree_hist.size == 0:
                    degree_hist = _degree_sequence_to_hist(
                        degree_sequence,
                        self.degree_dim,
                    )
                else:
                    degree_hist = _normalize(_pad(degree_hist, self.degree_dim))
            else:
                n_logits = arrays["num_nodes_logits"][i]
                n_probs = _softmax_np(n_logits)
                if deterministic:
                    node_index = int(np.argmax(n_probs))
                else:
                    node_index = int(
                        generator.choice(
                            np.arange(self.node_count_classes),
                            p=n_probs,
                        )
                    )
                n = int(self.min_nodes + node_index)

                degree_probs = _softmax_np(arrays["degree_logits"][i]).astype(
                    np.float64
                )
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
                degree_sequence = repair_degree_sequence(
                    degree_sequence,
                    n=n,
                    require_connected=self.require_connected,
                    rng=generator,
                )
                degree_hist = _degree_sequence_to_hist(
                    degree_sequence,
                    self.degree_dim,
                )

            clustering = _normalize(_softmax_np(arrays["clustering_logits"][i]))
            spectral = _normalize(_softmax_np(arrays["spectral_logits"][i]))

            motif = np.zeros(self.motif_dim, dtype=np.float64)
            if self.motif_dim:
                motif_scaled = np.maximum(arrays["motif_log"][i], 0.0)
                motif = np.expm1(
                    motif_scaled * np.asarray(self.motif_scale, dtype=np.float64)
                )
            orbit = np.zeros(self.orbit_dim, dtype=np.float64)
            if self.orbit_dim:
                orbit_scaled = np.maximum(arrays["orbit_log"][i], 0.0)
                orbit = np.expm1(
                    orbit_scaled * np.asarray(self.orbit_scale, dtype=np.float64)
                )

            graphlet_history = {}
            if self.graphlet_dim:
                logits = arrays.get(
                    "graphlet_logits",
                    np.zeros((batch, self.graphlet_dim), dtype=np.float64),
                )[i]
                graphlet_vec = np.zeros(self.graphlet_dim, dtype=np.float64)
                for sl in self.graphlet_slices().values():
                    graphlet_vec[sl] = (
                        _softmax_np(logits[sl])
                        if sl.stop > sl.start
                        else logits[sl]
                    )
                graphlet_history = unflatten_graphlet_history(
                    graphlet_vec, self.graphlet_keys_by_k or {}
                )

            scalar = arrays.get("scalar", np.zeros((batch, 2), dtype=np.float64))[i]
            triangle = (
                max(float(scalar[1]) * max(float(self.scalar_scale[1]), 1.0), 0.0)
                if scalar.size > 1
                else 0.0
            )
            num_edges = int(sum(degree_sequence) // 2)
            density = (2.0 * num_edges / (n * (n - 1))) if n > 1 else 0.0
            summary = {
                "num_nodes": n,
                "num_edges": num_edges,
                "degree_sequence": sorted(
                    [int(d) for d in degree_sequence], reverse=True
                ),
                "density": float(density),
                "triangle_count_norm": float(triangle),
                "degree_hist": degree_hist,
                "clustering_hist": clustering.astype(np.float64),
                "spectral_hist": spectral.astype(np.float64),
                "motif_proxy": motif.astype(np.float64),
                "orbit_count": orbit.astype(np.float64),
                "graphlet_history": graphlet_history,
            }
            if condition_summaries is not None:
                summary = _apply_degree_condition(
                    summary, condition_summaries[i], self.degree_dim
                )
            summaries.append(summary)
        return summaries

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: str | Path) -> SummaryVectorizer:
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
        for idx in order[:-remainder]:
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


def _apply_degree_condition(
    structural_summary: dict[str, Any],
    condition_summary: dict[str, Any],
    degree_width: int,
) -> dict[str, Any]:
    """Overwrite invariant fields with the exact sampled degree condition."""

    out = dict(structural_summary)
    sequence = sorted(
        [int(d) for d in condition_summary.get("degree_sequence", [])],
        reverse=True,
    )
    n = int(condition_summary.get("num_nodes", len(sequence)))
    if not sequence and n > 0:
        raise ValueError("A conditional target summary requires degree_sequence.")
    if len(sequence) != n:
        raise ValueError(
            "Degree condition is inconsistent: "
            f"num_nodes={n}, len(degree_sequence)={len(sequence)}."
        )
    m = int(sum(sequence) // 2)
    degree_hist = np.asarray(condition_summary.get("degree_hist", []), dtype=np.float64)
    if degree_hist.size == 0:
        degree_hist = _degree_sequence_to_hist(sequence, degree_width)
    else:
        degree_hist = _normalize(_pad(degree_hist, degree_width))
    out.update(
        {
            "num_nodes": n,
            "num_edges": m,
            "degree_sequence": sequence,
            "degree_hist": degree_hist,
            "density": float((2.0 * m / (n * (n - 1))) if n > 1 else 0.0),
        }
    )
    return out


def repair_degree_sequence(
    degree_sequence: list[int],
    *,
    n: int,
    require_connected: bool = True,
    rng: np.random.Generator | None = None,
    max_iterations: int = 10000,
) -> list[int]:
    """Repair sampled degrees into a graphical sequence."""

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
        candidates = [i for i, d in enumerate(seq) if d > min_degree]
        if candidates:
            max_d = max(seq[i] for i in candidates)
            high = [i for i in candidates if seq[i] == max_d]
            seq[int(generator.choice(high))] -= 1
        else:
            seq = [1] * n
            seq[0] = seq[1] = 2 if n > 2 else 1
        fix_parity()

    if require_connected and n >= 2:
        seq = [2] * n
        seq[0] = 1
        seq[-1] = 1
        return sorted(seq, reverse=True)
    return sorted(seq, reverse=True)


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


class SummaryVAE(nn.Module):
    """MLP VAE over permutation-invariant graph summaries."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int,
        head_dims: dict[str, int],
        *,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.head_dims = dict(head_dims)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.encoder = MLP(
            input_dim, hidden_dim, num_layers=num_layers, dropout=dropout
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = MLP(
            latent_dim, hidden_dim, num_layers=num_layers, dropout=dropout
        )
        self.num_nodes_head = nn.Linear(hidden_dim, head_dims["num_nodes"])
        self.degree_head = nn.Linear(hidden_dim, head_dims["degree"])
        self.clustering_head = nn.Linear(hidden_dim, head_dims["clustering"])
        self.spectral_head = nn.Linear(hidden_dim, head_dims["spectral"])
        self.motif_head = (
            nn.Linear(hidden_dim, head_dims.get("motif", 0))
            if head_dims.get("motif", 0) > 0
            else None
        )
        self.orbit_head = (
            nn.Linear(hidden_dim, head_dims.get("orbit", 0))
            if head_dims.get("orbit", 0) > 0
            else None
        )
        self.graphlet_head = (
            nn.Linear(hidden_dim, head_dims.get("graphlet", 0))
            if head_dims.get("graphlet", 0) > 0
            else None
        )
        self.scalar_head = nn.Linear(hidden_dim, head_dims.get("scalar", 2))

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.mu(h), self.logvar(h).clamp(min=-10.0, max=10.0)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.decoder(z)
        batch = z.shape[0]
        device = z.device
        out = {
            "num_nodes_logits": self.num_nodes_head(h),
            "degree_logits": self.degree_head(h),
            "clustering_logits": self.clustering_head(h),
            "spectral_logits": self.spectral_head(h),
            "scalar": F.softplus(self.scalar_head(h)),
        }
        if self.motif_head is not None:
            out["motif_log"] = F.softplus(self.motif_head(h))
        else:
            out["motif_log"] = torch.zeros(batch, 0, device=device)
        if self.orbit_head is not None:
            out["orbit_log"] = F.softplus(self.orbit_head(h))
        else:
            out["orbit_log"] = torch.zeros(batch, 0, device=device)
        out["graphlet_logits"] = (
            self.graphlet_head(h)
            if self.graphlet_head is not None
            else torch.zeros(batch, 0, device=device)
        )
        return out

    def forward(
        self, x: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    @torch.no_grad()
    def sample_outputs(
        self, num_samples: int, *, device: torch.device | str | None = None
    ) -> dict[str, torch.Tensor]:
        if device is None:
            device = next(self.parameters()).device
        z = torch.randn(int(num_samples), self.latent_dim, device=device)
        return self.decode(z)

    def model_config(self) -> dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "head_dims": self.head_dims,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
        }


class ConditionalSummaryVAE(nn.Module):
    """Conditional VAE for structural targets given a fixed degree sequence.

    The encoder observes the complete training summary and the fixed-degree
    condition.  The decoder receives ``(z, condition)`` and predicts only the
    structural target used to guide rewiring.  Legacy output heads remain
    present for checkpoint compatibility and diagnostics; generated invariant
    fields are overwritten by the exact condition during decoding.
    """

    def __init__(
        self,
        input_dim: int,
        condition_dim: int,
        latent_dim: int,
        hidden_dim: int,
        head_dims: dict[str, int],
        *,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.condition_dim = int(condition_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.head_dims = dict(head_dims)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)

        self.encoder = MLP(
            self.input_dim + self.condition_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = MLP(
            self.latent_dim + self.condition_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.num_nodes_head = nn.Linear(hidden_dim, head_dims["num_nodes"])
        self.degree_head = nn.Linear(hidden_dim, head_dims["degree"])
        self.clustering_head = nn.Linear(hidden_dim, head_dims["clustering"])
        self.spectral_head = nn.Linear(hidden_dim, head_dims["spectral"])
        self.motif_head = (
            nn.Linear(hidden_dim, head_dims.get("motif", 0))
            if head_dims.get("motif", 0) > 0
            else None
        )
        self.orbit_head = (
            nn.Linear(hidden_dim, head_dims.get("orbit", 0))
            if head_dims.get("orbit", 0) > 0
            else None
        )
        self.graphlet_head = (
            nn.Linear(hidden_dim, head_dims.get("graphlet", 0))
            if head_dims.get("graphlet", 0) > 0
            else None
        )
        self.scalar_head = nn.Linear(hidden_dim, head_dims.get("scalar", 2))

    def encode(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(torch.cat([x, condition], dim=-1))
        return self.mu(h), self.logvar(h).clamp(min=-10.0, max=10.0)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return SummaryVAE.reparameterize(mu, logvar)

    def decode(
        self,
        z: torch.Tensor,
        condition: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        h = self.decoder(torch.cat([z, condition], dim=-1))
        batch = z.shape[0]
        device = z.device
        out = {
            "num_nodes_logits": self.num_nodes_head(h),
            "degree_logits": self.degree_head(h),
            "clustering_logits": self.clustering_head(h),
            "spectral_logits": self.spectral_head(h),
            "scalar": F.softplus(self.scalar_head(h)),
        }
        out["motif_log"] = (
            F.softplus(self.motif_head(h))
            if self.motif_head is not None
            else torch.zeros(batch, 0, device=device)
        )
        out["orbit_log"] = (
            F.softplus(self.orbit_head(h))
            if self.orbit_head is not None
            else torch.zeros(batch, 0, device=device)
        )
        out["graphlet_logits"] = (
            self.graphlet_head(h)
            if self.graphlet_head is not None
            else torch.zeros(batch, 0, device=device)
        )
        return out

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x, condition)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, condition), mu, logvar

    @torch.no_grad()
    def sample_outputs(
        self,
        condition: torch.Tensor,
        *,
        device: torch.device | str | None = None,
    ) -> dict[str, torch.Tensor]:
        if device is None:
            device = next(self.parameters()).device
        condition = condition.to(device)
        z = torch.randn(condition.shape[0], self.latent_dim, device=device)
        return self.decode(z, condition)

    def model_config(self) -> dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "condition_dim": self.condition_dim,
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "head_dims": self.head_dims,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
        }


def soft_histogram_ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    return -(target * log_probs).sum(dim=-1).mean()


def graphlet_histogram_ce(
    logits: torch.Tensor,
    target: torch.Tensor,
    graphlet_slices: list[tuple[int, int]] | None = None,
) -> torch.Tensor:
    """Average categorical cross-entropy equally across graphlet sizes."""

    if logits.numel() == 0:
        return torch.zeros((), device=logits.device)
    slices = graphlet_slices or [(0, int(logits.shape[-1]))]
    losses = [
        soft_histogram_ce(logits[:, start:stop], target[:, start:stop])
        for start, stop in slices
        if stop > start
    ]
    return (
        torch.stack(losses).mean()
        if losses
        else torch.zeros((), device=logits.device)
    )


def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=-1))


def summary_vae_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    mu: torch.Tensor,
    logvar: torch.Tensor,
    *,
    beta: float = 1.0e-3,
    weights: dict[str, float] | None = None,
    graphlet_slices: list[tuple[int, int]] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    weights = weights or {}
    n_loss = F.cross_entropy(outputs["num_nodes_logits"], targets["num_nodes"].long())
    degree_loss = soft_histogram_ce(outputs["degree_logits"], targets["degree"])
    clustering_loss = soft_histogram_ce(
        outputs["clustering_logits"], targets["clustering"]
    )
    spectral_loss = soft_histogram_ce(outputs["spectral_logits"], targets["spectral"])
    motif_loss = (
        F.mse_loss(outputs["motif_log"], targets["motif"])
        if outputs["motif_log"].numel()
        else torch.zeros((), device=mu.device)
    )
    orbit_loss = (
        F.mse_loss(outputs["orbit_log"], targets["orbit"])
        if outputs["orbit_log"].numel()
        else torch.zeros((), device=mu.device)
    )
    graphlet_loss = graphlet_histogram_ce(
        outputs.get(
            "graphlet_logits",
            torch.zeros((mu.shape[0], 0), device=mu.device),
        ),
        targets["graphlet"],
        graphlet_slices,
    )
    density_loss = F.mse_loss(outputs["scalar"][:, 0], targets["scalar"][:, 0])
    triangle_loss = F.mse_loss(outputs["scalar"][:, 1], targets["scalar"][:, 1])
    density_weight = float(weights.get("density", weights.get("scalar", 1.0)))
    triangle_weight = float(weights.get("triangle", weights.get("scalar", 1.0)))
    scalar_loss = density_weight * density_loss + triangle_weight * triangle_loss
    kld = kl_loss(mu, logvar)
    total = (
        float(weights.get("num_nodes", 1.0)) * n_loss
        + float(weights.get("degree", 1.0)) * degree_loss
        + float(weights.get("clustering", 1.0)) * clustering_loss
        + float(weights.get("spectral", 1.0)) * spectral_loss
        + float(weights.get("motif", 1.0)) * motif_loss
        + float(weights.get("orbit", 1.0)) * orbit_loss
        + float(weights.get("graphlet", 1.0)) * graphlet_loss
        + scalar_loss
        + float(beta) * kld
    )
    metrics = {
        "loss": float(total.detach().cpu()),
        "num_nodes_loss": float(n_loss.detach().cpu()),
        "degree_loss": float(degree_loss.detach().cpu()),
        "clustering_loss": float(clustering_loss.detach().cpu()),
        "spectral_loss": float(spectral_loss.detach().cpu()),
        "motif_loss": float(motif_loss.detach().cpu()),
        "orbit_loss": float(orbit_loss.detach().cpu()),
        "graphlet_loss": float(graphlet_loss.detach().cpu()),
        "scalar_loss": float(scalar_loss.detach().cpu()),
        "density_loss": float(density_loss.detach().cpu()),
        "triangle_loss": float(triangle_loss.detach().cpu()),
        "kl_loss": float(kld.detach().cpu()),
    }
    return total, metrics


def build_summary_vae(
    vectorizer: SummaryVectorizer,
    *,
    latent_dim: int = 32,
    hidden_dim: int = 128,
    num_layers: int = 2,
    dropout: float = 0.0,
) -> SummaryVAE:
    return SummaryVAE(
        input_dim=vectorizer.input_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        head_dims=vectorizer.head_dims(),
        num_layers=num_layers,
        dropout=dropout,
    )


def build_conditional_summary_vae(
    vectorizer: SummaryVectorizer,
    *,
    latent_dim: int = 32,
    hidden_dim: int = 128,
    num_layers: int = 2,
    dropout: float = 0.0,
) -> ConditionalSummaryVAE:
    return ConditionalSummaryVAE(
        input_dim=vectorizer.input_dim,
        condition_dim=vectorizer.condition_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        head_dims=vectorizer.head_dims(),
        num_layers=num_layers,
        dropout=dropout,
    )


def save_summary_vae_checkpoint(
    path: str | Path,
    model: SummaryVAE | ConditionalSummaryVAE,
    vectorizer: SummaryVectorizer,
    *,
    config: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 2,
            "graphlet_decoder": "per_size_categorical_v1",
            "model_type": (
                "conditional_summary_vae"
                if isinstance(model, ConditionalSummaryVAE)
                else "summary_vae"
            ),
            "model_state_dict": model.state_dict(),
            "model_config": model.model_config(),
            "vectorizer": vectorizer.__dict__,
            "config": config or {},
            "metrics": metrics or {},
        },
        path,
    )


def load_summary_vae_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str = "auto",
) -> tuple[SummaryVAE | ConditionalSummaryVAE, SummaryVectorizer, dict[str, Any]]:
    resolved_device = resolve_torch_device(device)
    checkpoint = torch.load(path, map_location=resolved_device)
    vectorizer = SummaryVectorizer(**checkpoint["vectorizer"])
    if vectorizer.graphlet_dim > 0 and int(checkpoint.get("format_version", 1)) < 2:
        raise RuntimeError(
            "This checkpoint used the legacy unnormalized graphlet-MSE decoder. "
            "Retrain the target-summary generator with the per-size categorical "
            "graphlet decoder."
        )
    model_type = str(checkpoint.get("model_type", "summary_vae")).lower()
    if model_type == "conditional_summary_vae":
        model = ConditionalSummaryVAE(**checkpoint["model_config"])
    elif model_type == "summary_vae":
        model = SummaryVAE(**checkpoint["model_config"])
    else:
        raise ValueError(f"Unknown summary checkpoint model_type: {model_type!r}")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(resolved_device)
    model.eval()
    return model, vectorizer, checkpoint

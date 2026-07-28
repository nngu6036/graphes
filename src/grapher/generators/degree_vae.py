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

from grapher.generators.summary_vae import repair_degree_sequence
from grapher.properties.summary import sorted_degree_sequence
from grapher.utils.device import resolve_torch_device


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
    """Vectorizer and post-processor for DegreeHistogramVAE."""

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
        # q(z | h_D, n): edge count is exactly determined by h_D and is not
        # an independent encoder input.
        return int(1 + self.degree_dim)

    def head_dims(self) -> dict[str, int]:
        return {
            "num_nodes": self.node_count_classes,
            "degree": self.degree_dim,
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
        degree = self.degree_hist_from_sequence(seq)
        size = np.asarray(
            [n / max(float(self.max_nodes), 1.0)], dtype=np.float64
        )
        return np.concatenate([size, degree]).astype(np.float32)

    def to_targets(self, graph: nx.Graph) -> dict[str, np.ndarray | np.int64]:
        seq = sorted_degree_sequence(graph)
        n = int(graph.number_of_nodes())
        m = int(graph.number_of_edges())
        return {
            "num_nodes": np.int64(self.node_index(n)),
            "num_nodes_count": np.int64(n),
            "degree": self.degree_hist_from_sequence(seq).astype(np.float32),
            "mean_degree": np.asarray(
                [(2.0 * m / n) if n > 0 else 0.0], dtype=np.float32
            ),
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
        include_diagnostics: bool = False,
    ) -> list[dict[str, Any]]:
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
                    n_idx = int(generator.choice(np.arange(self.node_count_classes), p=n_probs))
                n = self.node_count_from_index(n_idx)

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
            attempt_limit = 1 if deterministic else max(int(max_resample), 1)
            for attempt in range(attempt_limit):
                if deterministic:
                    counts = _integer_counts_from_probs(n, degree_probs)
                else:
                    counts = generator.multinomial(int(n), degree_probs)
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
                if last_raw_sequence is not None:
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
                        repair_used = (
                            sorted(last_raw_sequence, reverse=True)
                            != sorted(repaired, reverse=True)
                        )
                if degree_sequence is None:
                    if fallback == "error":
                        raise RuntimeError(
                            "Degree generator failed to sample or repair a "
                            "graphical degree sequence."
                        )
                    degree_sequence = self.empirical_nearest_degree_sequence(
                        n, generator
                    )
                    fallback_used = True

            counts = _degree_sequence_to_counts(degree_sequence, self.degree_dim).astype(np.float64)
            degree_hist = _normalize(counts)
            num_edges = int(sum(degree_sequence) // 2)
            density = (2.0 * num_edges / (int(n) * (int(n) - 1))) if int(n) > 1 else 0.0
            summary = {
                "num_nodes": int(n),
                "num_edges": int(num_edges),
                "degree_sequence": sorted([int(d) for d in degree_sequence], reverse=True),
                "degree_hist": degree_hist.astype(np.float64),
                "density": float(density),
            }
            if include_diagnostics:
                raw_seq = first_raw_sequence
                raw_graphical = bool(
                    raw_seq is not None and nx.is_graphical(raw_seq, method="eg")
                )
                raw_connected_feasible = bool(
                    raw_seq is not None
                    and connected_feasible_degree_sequence(raw_seq)
                )
                raw_even_sum = bool(
                    raw_seq is not None and sum(raw_seq) % 2 == 0
                )
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
                    "accepted_without_postprocessing": bool(
                        accepted_without_postprocessing
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
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int | None = None, num_layers: int = 2, dropout: float = 0.0):
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
        size_condition_dim: int = 16,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.architecture_version = 2
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.head_dims = dict(head_dims)
        self.min_nodes = int(min_nodes)
        self.max_nodes = int(max_nodes)
        self.max_degree = int(max_degree)
        self.size_condition_dim = int(size_condition_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.encoder = MLP(input_dim, hidden_dim, num_layers=num_layers, dropout=dropout)
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
        self.degree_decoder = MLP(
            latent_dim + self.size_condition_dim,
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
        logarithmic = torch.log1p(n) / max(
            float(np.log1p(self.max_nodes)), 1.0
        )
        return torch.cat([linear, logarithmic], dim=-1)

    def decode(
        self, z: torch.Tensor, node_counts: torch.Tensor
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
        degree_hidden = self.degree_decoder(torch.cat([z, size_embedding], dim=-1))
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
        return {
            "num_nodes_logits": node_logits,
            "degree_logits": degree_logits,
            "conditioned_num_nodes": node_counts,
            "expected_mean_degree": expected_mean_degree,
        }

    def forward(
        self, x: torch.Tensor, node_counts: torch.Tensor | None = None
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        if node_counts is None:
            node_counts = torch.round(x[:, 0] * float(self.max_nodes)).long()
        return self.decode(z, node_counts), mu, logvar

    @torch.no_grad()
    def reconstruct_outputs(
        self,
        x: torch.Tensor,
        node_counts: torch.Tensor,
        *,
        use_mean: bool = True,
    ) -> dict[str, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = mu if use_mean else self.reparameterize(mu, logvar)
        return self.decode(z, node_counts)

    @torch.no_grad()
    def sample_outputs(
        self,
        num_samples: int,
        *,
        node_counts: torch.Tensor | np.ndarray | list[int] | None = None,
        deterministic_node_count: bool = False,
        device: torch.device | str | None = None,
    ) -> dict[str, torch.Tensor]:
        if device is None:
            device = next(self.parameters()).device
        z = torch.randn(int(num_samples), self.latent_dim, device=device)
        if node_counts is None:
            logits = self.node_count_logits(z)
            if deterministic_node_count:
                indices = torch.argmax(logits, dim=-1)
            else:
                indices = torch.distributions.Categorical(logits=logits).sample()
            node_counts = indices + self.min_nodes
        else:
            node_counts = torch.as_tensor(
                node_counts, dtype=torch.long, device=device
            )
        return self.decode(z, node_counts)

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
            "size_condition_dim": self.size_condition_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
        }


def soft_histogram_ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    return -(target * log_probs).sum(dim=-1).mean()


def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=-1))


def degree_vae_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    mu: torch.Tensor,
    logvar: torch.Tensor,
    *,
    beta: float = 5.0e-3,
    weights: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    weights = weights or {}
    n_loss = F.cross_entropy(outputs["num_nodes_logits"], targets["num_nodes"].long())
    degree_loss = soft_histogram_ce(outputs["degree_logits"], targets["degree"])
    predicted_mean_degree = outputs["expected_mean_degree"].reshape(-1)
    target_mean_degree = targets["mean_degree"].reshape(-1)
    degree_scale = max(float(outputs["degree_logits"].shape[-1] - 1), 1.0)
    moment_loss = F.mse_loss(
        predicted_mean_degree / degree_scale,
        target_mean_degree / degree_scale,
    )
    kld = kl_loss(mu, logvar)
    total = (
        float(weights.get("num_nodes", 1.0)) * n_loss
        + float(weights.get("degree", 5.0)) * degree_loss
        + float(
            weights.get("degree_moment", weights.get("edge_scalar", 0.1))
        )
        * moment_loss
        + float(beta) * kld
    )
    metrics = {
        "loss": float(total.detach().cpu()),
        "num_nodes_loss": float(n_loss.detach().cpu()),
        "degree_loss": float(degree_loss.detach().cpu()),
        "degree_moment_loss": float(moment_loss.detach().cpu()),
        "kl_loss": float(kld.detach().cpu()),
    }
    return total, metrics


def build_degree_vae(
    vectorizer: DegreeVectorizer,
    *,
    latent_dim: int = 32,
    hidden_dim: int = 128,
    size_condition_dim: int = 16,
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
        size_condition_dim=int(size_condition_dim),
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


def load_degree_vae_checkpoint(path: str | Path, *, device: torch.device | str = "auto") -> tuple[DegreeHistogramVAE, DegreeVectorizer, dict[str, Any]]:
    resolved_device = resolve_torch_device(device)
    checkpoint = torch.load(path, map_location=resolved_device)
    model_config = dict(checkpoint.get("model_config", {}))
    if int(model_config.get("architecture_version", 1)) != 2:
        raise RuntimeError(
            "This checkpoint uses the old unconditional DH-VAE decoder. "
            "Retrain it with scripts/train_degree_generator.py so decoding is "
            "conditioned on graph size."
        )
    model_config.pop("architecture_version", None)
    vectorizer = DegreeVectorizer(**checkpoint["vectorizer"])
    model = DegreeHistogramVAE(**model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(resolved_device)
    model.eval()
    return model, vectorizer, checkpoint

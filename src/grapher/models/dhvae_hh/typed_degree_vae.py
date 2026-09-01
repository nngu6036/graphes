from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from grapher.models.dhvae_hh.degree_vae import (
    MLP,
    conditional_prior_kl,
    kl_loss,
    soft_histogram_ce,
)
from grapher.rewiring_mlp.molecular.typed_invariants import (
    TypedDegreeSignature,
    TypedInvariant,
    extract_typed_invariant,
    typed_invariant_errors,
)
from grapher.utils.device import resolve_torch_device

TYPED_CHECKPOINT_FORMAT = "typed_signature_histogram_vae_v1"


def _stable_multinomial_probabilities(logits: np.ndarray) -> np.ndarray:
    """Return a float64 probability vector safe for NumPy multinomial draws.

    Torch decoder logits are usually float32. Normalizing them in float32 and then
    passing the result to ``Generator.multinomial`` can fail because NumPy casts
    the probabilities to float64 and strictly checks that ``sum(p[:-1]) <= 1``.
    Tiny float32 round-off can therefore make an otherwise valid softmax fail.
    Normalize in float64 and make the final entry the exact residual mass.
    """

    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("Typed signature logits must contain at least one class.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Typed signature logits contain non-finite values.")

    shifted = values - np.max(values)
    probabilities = np.exp(shifted)
    probabilities = np.clip(probabilities, 0.0, None)
    total = float(np.sum(probabilities, dtype=np.float64))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Typed signature logits produced invalid probability mass.")
    probabilities /= total

    if probabilities.size == 1:
        probabilities[0] = 1.0
        return probabilities

    # NumPy's multinomial samples the last category from the residual mass and
    # validates only the sum of p[:-1]. Keep that prefix strictly below 1 and
    # assign the last category from an exact float64 residual.
    prefix = float(np.sum(probabilities[:-1], dtype=np.float64))
    if prefix >= 1.0:
        target = float(np.nextafter(1.0, 0.0))
        if prefix > 0.0:
            probabilities[:-1] *= target / prefix
        prefix = float(np.sum(probabilities[:-1], dtype=np.float64))
    probabilities[-1] = max(0.0, 1.0 - prefix)

    # A final normalization protects against platform-specific summation drift.
    total = float(np.sum(probabilities, dtype=np.float64))
    probabilities /= total
    prefix = float(np.sum(probabilities[:-1], dtype=np.float64))
    if prefix > 1.0:
        target = float(np.nextafter(1.0, 0.0))
        probabilities[:-1] *= target / prefix
        prefix = float(np.sum(probabilities[:-1], dtype=np.float64))
        probabilities[-1] = 1.0 - prefix
    return probabilities


@dataclass(frozen=True)
class TypedSignatureVocabulary:
    signatures: tuple[TypedDegreeSignature, ...]
    edge_types: tuple[Any, ...]
    node_attribute: str = "atomic_num"
    edge_attribute: str = "bond_type"

    @classmethod
    def fit(
        cls,
        graphs: Sequence[nx.Graph],
        *,
        edge_types: Sequence[Any],
        node_attribute: str = "atomic_num",
        edge_attribute: str = "bond_type",
    ) -> "TypedSignatureVocabulary":
        values: set[TypedDegreeSignature] = set()
        for graph in graphs:
            invariant = extract_typed_invariant(
                graph,
                edge_types=edge_types,
                node_attribute=node_attribute,
                edge_attribute=edge_attribute,
            )
            values.update(invariant.signatures)
        if not values:
            raise ValueError("Cannot fit a typed-signature vocabulary on no records.")
        ordered = tuple(
            sorted(values, key=lambda item: (repr(item.node_type), item.edge_degrees))
        )
        return cls(
            signatures=ordered,
            edge_types=tuple(edge_types),
            node_attribute=node_attribute,
            edge_attribute=edge_attribute,
        )

    def index(self, signature: TypedDegreeSignature) -> int:
        try:
            return self.signatures.index(signature)
        except ValueError as exc:
            raise ValueError(f"Unknown typed signature {signature!r}.") from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "signatures": [signature.to_dict() for signature in self.signatures],
            "edge_types": list(self.edge_types),
            "node_attribute": self.node_attribute,
            "edge_attribute": self.edge_attribute,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TypedSignatureVocabulary":
        return cls(
            signatures=tuple(
                TypedDegreeSignature.from_dict(value)
                for value in data.get("signatures", [])
            ),
            edge_types=tuple(data.get("edge_types", [])),
            node_attribute=str(data.get("node_attribute", "atomic_num")),
            edge_attribute=str(data.get("edge_attribute", "bond_type")),
        )


@dataclass
class TypedSignatureVectorizer:
    vocabulary: TypedSignatureVocabulary
    min_nodes: int
    max_nodes: int
    empirical_node_counts: list[int]
    empirical_invariants: list[dict[str, Any]]
    require_connected: bool = True
    max_ordinary_degree: int | None = None
    max_weighted_valence: dict[Any, float] | None = None

    @classmethod
    def fit(
        cls,
        graphs: Sequence[nx.Graph],
        *,
        edge_types: Sequence[Any],
        node_attribute: str = "atomic_num",
        edge_attribute: str = "bond_type",
        require_connected: bool = True,
        max_ordinary_degree: int | None = None,
        max_weighted_valence: dict[Any, float] | None = None,
    ) -> "TypedSignatureVectorizer":
        if not graphs:
            raise ValueError("Cannot fit a typed vectorizer on an empty graph list.")
        vocabulary = TypedSignatureVocabulary.fit(
            graphs,
            edge_types=edge_types,
            node_attribute=node_attribute,
            edge_attribute=edge_attribute,
        )
        invariants = [
            extract_typed_invariant(
                graph,
                edge_types=edge_types,
                node_attribute=node_attribute,
                edge_attribute=edge_attribute,
            )
            for graph in graphs
        ]
        counts = [invariant.num_nodes for invariant in invariants]
        return cls(
            vocabulary=vocabulary,
            min_nodes=min(counts),
            max_nodes=max(counts),
            empirical_node_counts=counts,
            empirical_invariants=[invariant.to_dict() for invariant in invariants],
            require_connected=bool(require_connected),
            max_ordinary_degree=max_ordinary_degree,
            max_weighted_valence=max_weighted_valence,
        )

    @property
    def signature_dim(self) -> int:
        return len(self.vocabulary.signatures)

    @property
    def input_dim(self) -> int:
        return 1 + self.signature_dim

    @property
    def node_count_classes(self) -> int:
        return self.max_nodes - self.min_nodes + 1

    def sample_empirical_node_count(self, rng: np.random.Generator) -> int:
        return int(rng.choice(np.asarray(self.empirical_node_counts, dtype=np.int64)))

    def invariant_histogram(self, invariant: TypedInvariant) -> np.ndarray:
        counts = np.zeros(self.signature_dim, dtype=np.float64)
        for signature in invariant.signatures:
            counts[self.vocabulary.index(signature)] += 1.0
        return counts / max(float(counts.sum()), 1.0)

    def to_training_arrays(
        self, graphs: Sequence[nx.Graph]
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        features: list[np.ndarray] = []
        node_indices: list[int] = []
        node_counts: list[int] = []
        histograms: list[np.ndarray] = []
        incidence: list[np.ndarray] = []
        for graph in graphs:
            invariant = extract_typed_invariant(
                graph,
                edge_types=self.vocabulary.edge_types,
                node_attribute=self.vocabulary.node_attribute,
                edge_attribute=self.vocabulary.edge_attribute,
            )
            histogram = self.invariant_histogram(invariant)
            n = invariant.num_nodes
            features.append(
                np.concatenate(
                    [np.asarray([n / max(self.max_nodes, 1)]), histogram]
                ).astype(np.float32)
            )
            node_indices.append(n - self.min_nodes)
            node_counts.append(n)
            histograms.append(histogram.astype(np.float32))
            incidence.append(
                np.asarray(
                    [
                        sum(
                            signature.edge_degrees[column]
                            for signature in invariant.signatures
                        )
                        / max(n, 1)
                        for column in range(len(invariant.edge_types))
                    ],
                    dtype=np.float32,
                )
            )
        return np.stack(features), {
            "num_nodes": np.asarray(node_indices, dtype=np.int64),
            "num_nodes_count": np.asarray(node_counts, dtype=np.int64),
            "signature": np.stack(histograms),
            "incidence": np.stack(incidence),
        }

    def _invariant_from_counts(self, counts: np.ndarray) -> TypedInvariant:
        signatures: list[TypedDegreeSignature] = []
        for signature, count in zip(self.vocabulary.signatures, counts):
            signatures.extend([signature] * int(count))
        return TypedInvariant(
            signatures=tuple(signatures),
            edge_types=self.vocabulary.edge_types,
            node_attribute=self.vocabulary.node_attribute,
            edge_attribute=self.vocabulary.edge_attribute,
        )

    def outputs_to_summaries(
        self,
        outputs: dict[str, torch.Tensor],
        *,
        rng: np.random.Generator,
        deterministic: bool = False,
        max_resample: int = 1000,
        fallback: str = "error",
        include_diagnostics: bool = False,
    ) -> list[dict[str, Any]]:
        logits = outputs["signature_logits"].detach().cpu().numpy()
        node_counts = outputs["conditioned_num_nodes"].detach().cpu().numpy()
        summaries: list[dict[str, Any]] = []
        for row, node_count in zip(logits, node_counts):
            probabilities = _stable_multinomial_probabilities(row)
            accepted: TypedInvariant | None = None
            first_errors: list[str] = []
            attempts = 0
            for attempts in range(1, max(int(max_resample), 1) + 1):
                counts = (
                    _integer_counts(int(node_count), probabilities)
                    if deterministic
                    else rng.multinomial(int(node_count), probabilities)
                )
                candidate = self._invariant_from_counts(counts)
                errors = typed_invariant_errors(
                    candidate,
                    require_connected=self.require_connected,
                    max_ordinary_degree=self.max_ordinary_degree,
                    max_weighted_valence=self.max_weighted_valence,
                )
                if attempts == 1:
                    first_errors = errors
                if not errors:
                    accepted = candidate
                    break
            fallback_used = False
            if accepted is None:
                if str(fallback).lower() == "error":
                    raise RuntimeError(
                        "Typed invariant sampling exhausted its feasibility budget: "
                        + "; ".join(first_errors[:3])
                    )
                if str(fallback).lower() not in {"empirical", "empirical_nearest_n"}:
                    raise ValueError(f"Unknown typed fallback policy {fallback!r}.")
                candidates = [
                    TypedInvariant.from_dict(value)
                    for value in self.empirical_invariants
                ]
                accepted = min(
                    candidates,
                    key=lambda value: abs(value.num_nodes - int(node_count)),
                )
                fallback_used = True
            histogram = self.invariant_histogram(accepted)
            summary: dict[str, Any] = {
                "num_nodes": accepted.num_nodes,
                "num_edges": int(sum(accepted.degree_sequence) // 2),
                "degree_sequence": accepted.degree_sequence,
                "typed_invariant": accepted.to_dict(),
                "typed_signature_hist": histogram,
            }
            if include_diagnostics:
                summary["sampling_diagnostics"] = {
                    "attempts_used": int(attempts),
                    "first_raw_feasible": not first_errors,
                    "first_raw_errors": first_errors,
                    "fallback_used": fallback_used,
                    "accepted_without_postprocessing": not fallback_used,
                }
            summaries.append(summary)
        return summaries

    def to_dict(self) -> dict[str, Any]:
        return {
            "vocabulary": self.vocabulary.to_dict(),
            "min_nodes": self.min_nodes,
            "max_nodes": self.max_nodes,
            "empirical_node_counts": self.empirical_node_counts,
            "empirical_invariants": self.empirical_invariants,
            "require_connected": self.require_connected,
            "max_ordinary_degree": self.max_ordinary_degree,
            "max_weighted_valence": self.max_weighted_valence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TypedSignatureVectorizer":
        return cls(
            vocabulary=TypedSignatureVocabulary.from_dict(data["vocabulary"]),
            min_nodes=int(data["min_nodes"]),
            max_nodes=int(data["max_nodes"]),
            empirical_node_counts=[
                int(value) for value in data["empirical_node_counts"]
            ],
            empirical_invariants=list(data.get("empirical_invariants", [])),
            require_connected=bool(data.get("require_connected", True)),
            max_ordinary_degree=data.get("max_ordinary_degree"),
            max_weighted_valence=(
                {
                    int(key) if str(key).lstrip("-").isdigit() else key: float(value)
                    for key, value in data["max_weighted_valence"].items()
                }
                if data.get("max_weighted_valence")
                else None
            ),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))


def _integer_counts(n: int, probabilities: np.ndarray) -> np.ndarray:
    raw = np.asarray(probabilities, dtype=np.float64) * int(n)
    counts = np.floor(raw).astype(np.int64)
    for index in np.argsort(-(raw - counts))[: int(n) - int(counts.sum())]:
        counts[int(index)] += 1
    return counts


class TypedSignatureHistogramVAE(nn.Module):
    """Size-conditioned VAE over complete typed-degree signature classes."""

    def __init__(
        self,
        *,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int,
        signature_degrees: Sequence[int],
        signature_incidence: Sequence[Sequence[int]],
        min_nodes: int,
        max_nodes: int,
        size_condition_dim: int = 16,
        prior_type: str = "conditional_gmm",
        prior_components: int = 4,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if str(prior_type) not in {
            "standard_normal",
            "conditional_gaussian",
            "conditional_gmm",
        }:
            raise ValueError("Unsupported typed VAE prior_type.")
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.min_nodes = int(min_nodes)
        self.max_nodes = int(max_nodes)
        self.size_condition_dim = int(size_condition_dim)
        self.prior_type = str(prior_type)
        self.prior_components = (
            1 if self.prior_type == "conditional_gaussian" else int(prior_components)
        )
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        degrees = torch.as_tensor(signature_degrees, dtype=torch.long)
        incidence = torch.as_tensor(signature_incidence, dtype=torch.float32)
        if (
            degrees.ndim != 1
            or incidence.ndim != 2
            or incidence.shape[0] != degrees.shape[0]
        ):
            raise ValueError("Invalid typed signature tensors.")
        self.register_buffer("signature_degrees", degrees)
        self.register_buffer("signature_incidence", incidence)
        self.encoder = MLP(
            input_dim, hidden_dim, num_layers=num_layers, dropout=dropout
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.prior_decoder = MLP(
            latent_dim, hidden_dim, num_layers=num_layers, dropout=dropout
        )
        self.num_nodes_head = nn.Linear(hidden_dim, self.max_nodes - self.min_nodes + 1)
        self.size_encoder = MLP(
            2,
            hidden_dim,
            output_dim=size_condition_dim,
            num_layers=1,
            dropout=dropout,
        )
        self.signature_decoder = MLP(
            latent_dim + size_condition_dim,
            hidden_dim,
            output_dim=len(signature_degrees),
            num_layers=num_layers,
            dropout=dropout,
        )
        if self.prior_type != "standard_normal":
            self.conditional_prior = MLP(
                2,
                hidden_dim,
                output_dim=self.prior_components * (1 + 2 * latent_dim),
                num_layers=1,
                dropout=dropout,
            )

    def _size_features(self, node_counts: torch.Tensor) -> torch.Tensor:
        values = node_counts.float().reshape(-1, 1)
        return torch.cat(
            [
                values / max(float(self.max_nodes), 1.0),
                torch.log1p(values) / max(float(np.log1p(self.max_nodes)), 1.0),
            ],
            dim=-1,
        )

    def encode(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(inputs)
        return self.mu(hidden), self.logvar(hidden).clamp(-10.0, 10.0)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def prior_parameters(self, node_counts: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = node_counts.shape[0]
        if self.prior_type == "standard_normal":
            logits = torch.zeros(batch, 1, device=node_counts.device)
            means = torch.zeros(batch, 1, self.latent_dim, device=node_counts.device)
            logvars = torch.zeros_like(means)
        else:
            raw = self.conditional_prior(self._size_features(node_counts)).reshape(
                batch, self.prior_components, 1 + 2 * self.latent_dim
            )
            logits = raw[..., 0]
            means = raw[..., 1 : 1 + self.latent_dim]
            logvars = raw[..., 1 + self.latent_dim :].clamp(-6.0, 4.0)
        return {
            "prior_logits": logits,
            "prior_means": means,
            "prior_logvars": logvars,
        }

    def decode(
        self, latent: torch.Tensor, node_counts: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        node_counts = node_counts.long().reshape(-1)
        size = self.size_encoder(self._size_features(node_counts))
        signature_logits = self.signature_decoder(torch.cat([latent, size], dim=-1))
        invalid = self.signature_degrees.unsqueeze(0) >= node_counts.unsqueeze(1)
        signature_logits = signature_logits.masked_fill(invalid, -1.0e9)
        probabilities = torch.softmax(signature_logits, dim=-1)
        expected_incidence = probabilities @ self.signature_incidence
        return {
            "num_nodes_logits": self.num_nodes_head(self.prior_decoder(latent)),
            "signature_logits": signature_logits,
            "expected_incidence": expected_incidence,
            "conditioned_num_nodes": node_counts,
        }

    def forward(
        self, inputs: torch.Tensor, node_counts: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(inputs)
        latent = self.reparameterize(mu, logvar)
        outputs = self.decode(latent, node_counts)
        outputs.update(self.prior_parameters(node_counts))
        outputs["latent_z"] = latent
        return outputs, mu, logvar

    @torch.no_grad()
    def sample_outputs(
        self,
        num_samples: int,
        *,
        node_counts: Sequence[int] | torch.Tensor | None = None,
        device: torch.device | str | None = None,
    ) -> dict[str, torch.Tensor]:
        device = device or next(self.parameters()).device
        if node_counts is None:
            latent = torch.randn(int(num_samples), self.latent_dim, device=device)
            indices = torch.distributions.Categorical(
                logits=self.num_nodes_head(self.prior_decoder(latent))
            ).sample()
            node_counts = indices + self.min_nodes
        node_counts = torch.as_tensor(node_counts, dtype=torch.long, device=device)
        params = self.prior_parameters(node_counts)
        components = torch.distributions.Categorical(
            logits=params["prior_logits"]
        ).sample()
        rows = torch.arange(node_counts.shape[0], device=node_counts.device)
        means = params["prior_means"][rows, components]
        logvars = params["prior_logvars"][rows, components]
        latent = means + torch.randn_like(means) * torch.exp(0.5 * logvars)
        return self.decode(latent, node_counts)

    def model_config(self) -> dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "signature_degrees": self.signature_degrees.detach().cpu().tolist(),
            "signature_incidence": self.signature_incidence.detach().cpu().tolist(),
            "min_nodes": self.min_nodes,
            "max_nodes": self.max_nodes,
            "size_condition_dim": self.size_condition_dim,
            "prior_type": self.prior_type,
            "prior_components": self.prior_components,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
        }


def typed_signature_vae_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    mu: torch.Tensor,
    logvar: torch.Tensor,
    *,
    beta: float = 0.005,
    weights: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    weights = weights or {}
    node_loss = F.cross_entropy(
        outputs["num_nodes_logits"], targets["num_nodes"].long()
    )
    signature_loss = soft_histogram_ce(
        outputs["signature_logits"], targets["signature"]
    )
    incidence_loss = F.mse_loss(outputs["expected_incidence"], targets["incidence"])
    if "latent_z" in outputs:
        latent_loss = conditional_prior_kl(
            outputs["latent_z"],
            mu,
            logvar,
            outputs["prior_logits"],
            outputs["prior_means"],
            outputs["prior_logvars"],
        )
    else:
        latent_loss = kl_loss(mu, logvar)
    total = (
        float(weights.get("num_nodes", 1.0)) * node_loss
        + float(weights.get("signature", 5.0)) * signature_loss
        + float(weights.get("incidence", 0.1)) * incidence_loss
        + float(beta) * latent_loss
    )
    return total, {
        "loss": float(total.detach().cpu()),
        "num_nodes_loss": float(node_loss.detach().cpu()),
        "signature_loss": float(signature_loss.detach().cpu()),
        "incidence_loss": float(incidence_loss.detach().cpu()),
        "kl_loss": float(latent_loss.detach().cpu()),
    }


def build_typed_signature_vae(
    vectorizer: TypedSignatureVectorizer,
    **kwargs: Any,
) -> TypedSignatureHistogramVAE:
    signatures = vectorizer.vocabulary.signatures
    return TypedSignatureHistogramVAE(
        input_dim=vectorizer.input_dim,
        signature_degrees=[signature.degree for signature in signatures],
        signature_incidence=[signature.edge_degrees for signature in signatures],
        min_nodes=vectorizer.min_nodes,
        max_nodes=vectorizer.max_nodes,
        **kwargs,
    )


def save_typed_signature_checkpoint(
    path: str | Path,
    model: TypedSignatureHistogramVAE,
    vectorizer: TypedSignatureVectorizer,
    *,
    config: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": TYPED_CHECKPOINT_FORMAT,
            "model_state_dict": model.state_dict(),
            "model_config": model.model_config(),
            "vectorizer": vectorizer.to_dict(),
            "config": config or {},
            "metrics": metrics or {},
        },
        path,
    )


def load_typed_signature_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str = "auto",
) -> tuple[TypedSignatureHistogramVAE, TypedSignatureVectorizer, dict[str, Any]]:
    resolved = resolve_torch_device(device)
    checkpoint = torch.load(path, map_location=resolved)
    if checkpoint.get("format") != TYPED_CHECKPOINT_FORMAT:
        raise ValueError("Checkpoint is not a typed-signature histogram VAE.")
    vectorizer = TypedSignatureVectorizer.from_dict(checkpoint["vectorizer"])
    model = TypedSignatureHistogramVAE(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(resolved).eval()
    return model, vectorizer, checkpoint

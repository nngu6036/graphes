from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from grapher.generators.degree_vae import (
    MLP,
    conditional_prior_kl,
    kl_loss,
    soft_histogram_ce,
)
from grapher.molecular.constraints import bond_order
from grapher.utils.device import resolve_torch_device

TYPED_CHECKPOINT_FORMAT = "typed_signature_histogram_vae_v1"


@dataclass(frozen=True)
class TypedDegreeSignature:
    """One immutable node category and its degree in every edge category."""

    node_type: Any
    edge_degrees: tuple[int, ...]

    def __post_init__(self) -> None:
        if any(int(value) < 0 for value in self.edge_degrees):
            raise ValueError("Typed degrees must be non-negative.")

    @property
    def degree(self) -> int:
        return int(sum(int(value) for value in self.edge_degrees))

    def weighted_degree(self, edge_types: Sequence[Any]) -> float:
        if len(edge_types) != len(self.edge_degrees):
            raise ValueError("edge_types and edge_degrees must have equal length.")
        return float(
            sum(
                float(count) * bond_order(int(edge_type))
                for edge_type, count in zip(edge_types, self.edge_degrees)
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_type": self.node_type,
            "edge_degrees": [int(value) for value in self.edge_degrees],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TypedDegreeSignature":
        return cls(
            node_type=data["node_type"],
            edge_degrees=tuple(int(value) for value in data["edge_degrees"]),
        )


@dataclass(frozen=True)
class TypedInvariant:
    """Indexed typed-degree records preserved by strict molecular rewiring."""

    signatures: tuple[TypedDegreeSignature, ...]
    edge_types: tuple[Any, ...]
    node_attribute: str = "atomic_num"
    edge_attribute: str = "bond_type"

    def __post_init__(self) -> None:
        width = len(self.edge_types)
        if len(set(self.edge_types)) != width:
            raise ValueError("edge_types must be unique.")
        if any(len(signature.edge_degrees) != width for signature in self.signatures):
            raise ValueError("Every signature must contain one degree per edge type.")

    @property
    def num_nodes(self) -> int:
        return len(self.signatures)

    @property
    def degree_sequence(self) -> list[int]:
        return sorted((signature.degree for signature in self.signatures), reverse=True)

    @property
    def edge_counts(self) -> dict[Any, int]:
        return {
            edge_type: int(
                sum(signature.edge_degrees[index] for signature in self.signatures) // 2
            )
            for index, edge_type in enumerate(self.edge_types)
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "signatures": [signature.to_dict() for signature in self.signatures],
            "edge_types": list(self.edge_types),
            "node_attribute": self.node_attribute,
            "edge_attribute": self.edge_attribute,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TypedInvariant":
        return cls(
            signatures=tuple(
                TypedDegreeSignature.from_dict(value)
                for value in data.get("signatures", [])
            ),
            edge_types=tuple(data.get("edge_types", [])),
            node_attribute=str(data.get("node_attribute", "atomic_num")),
            edge_attribute=str(data.get("edge_attribute", "bond_type")),
        )


def extract_typed_invariant(
    graph: nx.Graph,
    *,
    edge_types: Sequence[Any],
    node_attribute: str = "atomic_num",
    edge_attribute: str = "bond_type",
) -> TypedInvariant:
    """Extract indexed signatures without discarding node correspondence."""

    if graph.is_directed() or graph.is_multigraph() or nx.number_of_selfloops(graph):
        raise ValueError("Typed invariants require a simple undirected graph.")
    normalized = nx.convert_node_labels_to_integers(
        nx.Graph(graph), first_label=0, ordering="sorted"
    )
    edge_types = tuple(edge_types)
    edge_index = {value: index for index, value in enumerate(edge_types)}
    counts = np.zeros((normalized.number_of_nodes(), len(edge_types)), dtype=np.int64)
    for u, v, data in normalized.edges(data=True):
        if edge_attribute not in data:
            raise KeyError(f"Edge {(u, v)!r} is missing {edge_attribute!r}.")
        edge_type = data[edge_attribute]
        if edge_type not in edge_index:
            raise ValueError(f"Unsupported edge category {edge_type!r}.")
        column = edge_index[edge_type]
        counts[int(u), column] += 1
        counts[int(v), column] += 1

    signatures: list[TypedDegreeSignature] = []
    for node, data in normalized.nodes(data=True):
        if node_attribute not in data:
            raise KeyError(f"Node {node!r} is missing {node_attribute!r}.")
        signatures.append(
            TypedDegreeSignature(
                node_type=data[node_attribute],
                edge_degrees=tuple(int(value) for value in counts[int(node)]),
            )
        )
    return TypedInvariant(
        signatures=tuple(signatures),
        edge_types=edge_types,
        node_attribute=node_attribute,
        edge_attribute=edge_attribute,
    )


def typed_invariant_histogram(
    invariant: TypedInvariant,
) -> dict[TypedDegreeSignature, int]:
    out: dict[TypedDegreeSignature, int] = {}
    for signature in invariant.signatures:
        out[signature] = out.get(signature, 0) + 1
    return out


def typed_invariant_errors(
    invariant: TypedInvariant,
    *,
    require_connected: bool = True,
    max_ordinary_degree: int | None = None,
    max_weighted_valence: dict[Any, float] | None = None,
    endpoint_compatible: Callable[[Any, Any, Any], bool] | None = None,
) -> list[str]:
    """Return inexpensive necessary feasibility violations.

    Simultaneous typed realization is checked by the exact constructor.  These
    filters deliberately remain necessary rather than pretending that a
    per-category graphicality test is sufficient.
    """

    errors: list[str] = []
    n = invariant.num_nodes
    degrees = [signature.degree for signature in invariant.signatures]
    if n == 0:
        return ["the invariant contains no nodes"]
    for node, signature in enumerate(invariant.signatures):
        if signature.degree >= n:
            errors.append(f"node {node}: ordinary degree must be below {n}")
        if max_ordinary_degree is not None and signature.degree > int(
            max_ordinary_degree
        ):
            errors.append(
                f"node {node}: degree {signature.degree} exceeds "
                f"{int(max_ordinary_degree)}"
            )
        if require_connected and n > 1 and signature.degree == 0:
            errors.append(f"node {node}: zero degree cannot be connected")
        if max_weighted_valence is not None:
            maximum = max_weighted_valence.get(signature.node_type)
            if maximum is None:
                errors.append(
                    f"node {node}: no valence rule for {signature.node_type!r}"
                )
            else:
                try:
                    used = signature.weighted_degree(invariant.edge_types)
                except (TypeError, ValueError):
                    used = float("inf")
                if used > float(maximum) + 1.0e-8:
                    errors.append(
                        f"node {node}: weighted degree {used:g} exceeds "
                        f"{float(maximum):g}"
                    )

    for column, edge_type in enumerate(invariant.edge_types):
        total = sum(
            signature.edge_degrees[column] for signature in invariant.signatures
        )
        if total % 2:
            errors.append(f"edge type {edge_type!r}: incidence total is odd")
        if endpoint_compatible is not None:
            for node, signature in enumerate(invariant.signatures):
                required = signature.edge_degrees[column]
                possible = sum(
                    other != node
                    and invariant.signatures[other].edge_degrees[column] > 0
                    and endpoint_compatible(
                        signature.node_type,
                        invariant.signatures[other].node_type,
                        edge_type,
                    )
                    for other in range(n)
                )
                if required > possible:
                    errors.append(
                        f"node {node}: type {edge_type!r} demand {required} "
                        f"exceeds {possible} compatible endpoints"
                    )

    if sum(degrees) % 2:
        errors.append("aggregate degree sum is odd")
    elif not nx.is_graphical(sorted(degrees, reverse=True), method="eg"):
        errors.append("aggregate ordinary degree sequence is not graphical")
    if require_connected and n > 1 and sum(degrees) < 2 * (n - 1):
        errors.append("aggregate degrees cannot support a connected realization")
    return errors


def typed_invariant_matches_graph(
    graph: nx.Graph,
    invariant: TypedInvariant,
) -> bool:
    try:
        observed = extract_typed_invariant(
            graph,
            edge_types=invariant.edge_types,
            node_attribute=invariant.node_attribute,
            edge_attribute=invariant.edge_attribute,
        )
    except (KeyError, TypeError, ValueError):
        return False
    return observed.signatures == invariant.signatures


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
            probabilities = np.exp(row - np.max(row))
            probabilities /= probabilities.sum()
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

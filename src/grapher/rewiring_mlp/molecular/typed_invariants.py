from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import networkx as nx
import numpy as np

from grapher.rewiring_mlp.molecular.constraints import bond_order


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


@dataclass(frozen=True)
class AttributedRewiringInvariant:
    """Hard invariant for bond-reassigning attributed rewiring.

    Unlike :class:`TypedInvariant`, this invariant deliberately does not fix the
    per-node degree in every edge category.  A valid attributed double-edge
    swap preserves the indexed ordinary degree and node category of every node
    while reassigning the two removed edge categories to the two inserted
    edges.  The global edge-category histogram is therefore preserved exactly,
    but typed degrees and per-node weighted valence may change.
    """

    node_types: tuple[Any, ...]
    degrees: tuple[int, ...]
    edge_types: tuple[Any, ...]
    edge_counts: tuple[int, ...]
    node_attribute: str = "atomic_num"
    edge_attribute: str = "bond_type"

    def __post_init__(self) -> None:
        if len(self.node_types) != len(self.degrees):
            raise ValueError("node_types and degrees must have equal length.")
        if len(self.edge_types) != len(self.edge_counts):
            raise ValueError("edge_types and edge_counts must have equal length.")
        if len(set(self.edge_types)) != len(self.edge_types):
            raise ValueError("edge_types must be unique.")
        if any(int(value) < 0 for value in self.degrees):
            raise ValueError("Ordinary degrees must be non-negative.")
        if any(int(value) < 0 for value in self.edge_counts):
            raise ValueError("Edge-category counts must be non-negative.")

    @property
    def num_nodes(self) -> int:
        return len(self.node_types)

    @property
    def edge_count_map(self) -> dict[Any, int]:
        return {
            edge_type: int(count)
            for edge_type, count in zip(self.edge_types, self.edge_counts)
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_types": list(self.node_types),
            "degrees": [int(value) for value in self.degrees],
            "edge_types": list(self.edge_types),
            "edge_counts": [int(value) for value in self.edge_counts],
            "node_attribute": self.node_attribute,
            "edge_attribute": self.edge_attribute,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AttributedRewiringInvariant":
        return cls(
            node_types=tuple(data.get("node_types", [])),
            degrees=tuple(int(value) for value in data.get("degrees", [])),
            edge_types=tuple(data.get("edge_types", [])),
            edge_counts=tuple(int(value) for value in data.get("edge_counts", [])),
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


def extract_attributed_rewiring_invariant(
    graph: nx.Graph,
    *,
    edge_types: Sequence[Any],
    node_attribute: str = "atomic_num",
    edge_attribute: str = "bond_type",
) -> AttributedRewiringInvariant:
    """Extract the invariant preserved by bond-reassigning edge swaps.

    The returned object fixes indexed node categories and ordinary degrees plus
    the global count of every configured edge category.  It intentionally does
    not fix per-node typed degrees.
    """

    if graph.is_directed() or graph.is_multigraph() or nx.number_of_selfloops(graph):
        raise ValueError("Attributed rewiring invariants require a simple undirected graph.")
    normalized = nx.convert_node_labels_to_integers(
        nx.Graph(graph), first_label=0, ordering="sorted"
    )
    edge_types = tuple(edge_types)
    edge_index = {value: index for index, value in enumerate(edge_types)}
    edge_counts = [0 for _ in edge_types]
    for u, v, data in normalized.edges(data=True):
        if edge_attribute not in data:
            raise KeyError(f"Edge {(u, v)!r} is missing {edge_attribute!r}.")
        edge_type = data[edge_attribute]
        if edge_type not in edge_index:
            raise ValueError(f"Unsupported edge category {edge_type!r}.")
        edge_counts[edge_index[edge_type]] += 1

    node_types: list[Any] = []
    degrees: list[int] = []
    for node, data in normalized.nodes(data=True):
        if node_attribute not in data:
            raise KeyError(f"Node {node!r} is missing {node_attribute!r}.")
        node_types.append(data[node_attribute])
        degrees.append(int(normalized.degree(node)))
    return AttributedRewiringInvariant(
        node_types=tuple(node_types),
        degrees=tuple(degrees),
        edge_types=edge_types,
        edge_counts=tuple(edge_counts),
        node_attribute=node_attribute,
        edge_attribute=edge_attribute,
    )


def attributed_rewiring_invariant_matches_graph(
    graph: nx.Graph,
    invariant: AttributedRewiringInvariant,
) -> bool:
    try:
        observed = extract_attributed_rewiring_invariant(
            graph,
            edge_types=invariant.edge_types,
            node_attribute=invariant.node_attribute,
            edge_attribute=invariant.edge_attribute,
        )
    except (KeyError, TypeError, ValueError):
        return False
    return (
        observed.node_types == invariant.node_types
        and observed.degrees == invariant.degrees
        and observed.edge_counts == invariant.edge_counts
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


# One-release compatibility bridge for callers that previously imported the
# typed DH-VAE from this neutral invariant module. The model package is loaded
# only when one of these legacy names is explicitly requested, so GraphER's
# invariant checks remain independent of Torch and the DH-VAE baseline.
_LEGACY_TYPED_VAE_EXPORTS = frozenset(
    {
        "TYPED_CHECKPOINT_FORMAT",
        "TypedSignatureVocabulary",
        "TypedSignatureVectorizer",
        "TypedSignatureHistogramVAE",
        "typed_signature_vae_loss",
        "build_typed_signature_vae",
        "save_typed_signature_checkpoint",
        "load_typed_signature_checkpoint",
    }
)


def __getattr__(name: str) -> Any:
    if name not in _LEGACY_TYPED_VAE_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from grapher.models.dhvae_hh import typed_degree_vae

    return getattr(typed_degree_vae, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LEGACY_TYPED_VAE_EXPORTS))

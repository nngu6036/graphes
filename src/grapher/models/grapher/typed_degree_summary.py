"""Training-only typed-degree histogram vocabularies for attributed GDSM models.

A typed-degree signature records one node category together with the number of
incident edges in every configured edge category.  Histograms over these
signatures are permutation invariant and provide a compact radius-one
attribute summary complementary to typed graphlets.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

import networkx as nx
import numpy as np

from grapher.rewiring_mlp.attributed.data import GraphCategoryVocabulary


TYPED_DEGREE_OVERFLOW_KEY = "__overflow__"


@dataclass(frozen=True, order=True)
class TypedDegreeKey:
    """One categorical node type and its incident counts by edge type."""

    node_category: int
    edge_degrees: tuple[int, ...]

    def __post_init__(self) -> None:
        if int(self.node_category) < 0:
            raise ValueError("node_category must be non-negative")
        if any(int(value) < 0 for value in self.edge_degrees):
            raise ValueError("typed edge degrees must be non-negative")

    @property
    def degree(self) -> int:
        return int(sum(int(value) for value in self.edge_degrees))

    def to_dict(self, vocabulary: GraphCategoryVocabulary | None = None) -> dict[str, Any]:
        result: dict[str, Any] = {
            "node_category_index": int(self.node_category),
            "edge_degrees": [int(value) for value in self.edge_degrees],
        }
        if vocabulary is not None:
            result["node_value"] = vocabulary.node_value(self.node_category)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TypedDegreeKey":
        return cls(
            node_category=int(data["node_category_index"]),
            edge_degrees=tuple(int(value) for value in data.get("edge_degrees", [])),
        )


def typed_degree_keys_for_graph(
    graph: nx.Graph,
    vocabulary: GraphCategoryVocabulary,
) -> tuple[TypedDegreeKey, ...]:
    """Return one indexed typed-degree key per node in sorted node order."""

    if graph.is_directed() or graph.is_multigraph() or nx.number_of_selfloops(graph):
        raise ValueError("Typed-degree summaries require simple undirected graphs")
    normalized = nx.convert_node_labels_to_integers(
        nx.Graph(graph), first_label=0, ordering="sorted"
    )
    n = normalized.number_of_nodes()
    if n < 1:
        raise ValueError("Typed-degree summaries require non-empty graphs")
    edge_counts = np.zeros((n, len(vocabulary.edge_values)), dtype=np.int64)
    for u, v, data in normalized.edges(data=True):
        edge_category = int(vocabulary.edge_index(data) - 1)
        edge_counts[int(u), edge_category] += 1
        edge_counts[int(v), edge_category] += 1
    result: list[TypedDegreeKey] = []
    for node, data in normalized.nodes(data=True):
        result.append(
            TypedDegreeKey(
                node_category=int(vocabulary.node_index(data)),
                edge_degrees=tuple(int(value) for value in edge_counts[int(node)]),
            )
        )
    return tuple(result)


@dataclass(frozen=True)
class TypedDegreeHistogramBasis:
    """Training-only vocabulary of typed-degree signatures plus overflow."""

    signatures: tuple[TypedDegreeKey, ...]
    node_values: tuple[Any, ...]
    edge_values: tuple[Any, ...]
    training_counts: tuple[int, ...]
    overflow_count: int = 0
    overflow_key: str = TYPED_DEGREE_OVERFLOW_KEY

    def __post_init__(self) -> None:
        if len(self.signatures) != len(self.training_counts):
            raise ValueError("signatures and training_counts must have equal length")
        if len(set(self.signatures)) != len(self.signatures):
            raise ValueError("typed-degree signatures must be unique")
        width = len(self.edge_values)
        if any(len(key.edge_degrees) != width for key in self.signatures):
            raise ValueError("Every typed-degree signature needs one count per edge type")
        if any(key.node_category >= len(self.node_values) for key in self.signatures):
            raise ValueError("Typed-degree signature has an unknown node category")
        if any(int(value) < 0 for value in self.training_counts):
            raise ValueError("training_counts must be non-negative")
        if int(self.overflow_count) < 0:
            raise ValueError("overflow_count must be non-negative")

    @property
    def width(self) -> int:
        return len(self.signatures) + 1

    @property
    def overflow_index(self) -> int:
        return len(self.signatures)

    @property
    def total_training_nodes(self) -> int:
        return int(sum(self.training_counts) + int(self.overflow_count))

    @classmethod
    def fit_from_graphs(
        cls,
        graphs: Sequence[nx.Graph],
        vocabulary: GraphCategoryVocabulary,
        *,
        min_count: int = 1,
        max_signatures: int | None = None,
    ) -> "TypedDegreeHistogramBasis":
        min_count = int(min_count)
        if min_count < 1:
            raise ValueError("min_count must be >= 1")
        if max_signatures is not None and int(max_signatures) < 1:
            raise ValueError("max_signatures must be positive or null")
        counts: Counter[TypedDegreeKey] = Counter()
        for graph in graphs:
            counts.update(typed_degree_keys_for_graph(graph, vocabulary))
        eligible = [key for key, count in counts.items() if int(count) >= min_count]
        eligible.sort(
            key=lambda key: (
                -int(counts[key]),
                int(key.node_category),
                tuple(int(value) for value in key.edge_degrees),
            )
        )
        if max_signatures is not None:
            eligible = eligible[: int(max_signatures)]
        selected = tuple(eligible)
        selected_set = set(selected)
        overflow_count = sum(
            int(count) for key, count in counts.items() if key not in selected_set
        )
        return cls(
            signatures=selected,
            node_values=tuple(vocabulary.node_values),
            edge_values=tuple(vocabulary.edge_values),
            training_counts=tuple(int(counts[key]) for key in selected),
            overflow_count=int(overflow_count),
        )

    @classmethod
    def disabled(cls, vocabulary: GraphCategoryVocabulary) -> "TypedDegreeHistogramBasis":
        return cls(
            signatures=tuple(),
            node_values=tuple(vocabulary.node_values),
            edge_values=tuple(vocabulary.edge_values),
            training_counts=tuple(),
            overflow_count=0,
        )

    def validate_vocabulary(self, vocabulary: GraphCategoryVocabulary) -> None:
        if tuple(vocabulary.node_values) != tuple(self.node_values):
            raise ValueError("Typed-degree basis node vocabulary mismatch")
        if tuple(vocabulary.edge_values) != tuple(self.edge_values):
            raise ValueError("Typed-degree basis edge vocabulary mismatch")

    def histogram_for_graph(
        self,
        graph: nx.Graph,
        vocabulary: GraphCategoryVocabulary,
    ) -> np.ndarray:
        self.validate_vocabulary(vocabulary)
        keys = typed_degree_keys_for_graph(graph, vocabulary)
        index = {key: position for position, key in enumerate(self.signatures)}
        histogram = np.zeros(self.width, dtype=np.float64)
        for key in keys:
            histogram[index.get(key, self.overflow_index)] += 1.0
        histogram /= float(len(keys))
        return histogram.astype(np.float32)

    def node_marginal_from_histogram(self, histogram: np.ndarray) -> np.ndarray:
        values = np.asarray(histogram, dtype=np.float64).reshape(-1)
        if values.size != self.width:
            raise ValueError("Typed-degree histogram has an incompatible width")
        result = np.zeros(len(self.node_values), dtype=np.float64)
        for index, key in enumerate(self.signatures):
            result[int(key.node_category)] += float(values[index])
        return result

    def edge_incidence_from_histogram(self, histogram: np.ndarray) -> np.ndarray:
        values = np.asarray(histogram, dtype=np.float64).reshape(-1)
        if values.size != self.width:
            raise ValueError("Typed-degree histogram has an incompatible width")
        result = np.zeros(len(self.edge_values), dtype=np.float64)
        for index, key in enumerate(self.signatures):
            result += float(values[index]) * np.asarray(key.edge_degrees, dtype=np.float64)
        return result

    def to_dict(self) -> dict[str, Any]:
        vocabulary = GraphCategoryVocabulary(
            node_values=tuple(self.node_values),
            edge_values=tuple(self.edge_values),
        )
        return {
            "format": "typed_degree_histogram_basis_v1",
            "signatures": [key.to_dict(vocabulary) for key in self.signatures],
            "node_values": list(self.node_values),
            "edge_values": list(self.edge_values),
            "training_counts": [int(value) for value in self.training_counts],
            "overflow_count": int(self.overflow_count),
            "overflow_index": int(self.overflow_index),
            "overflow_key": self.overflow_key,
            "width": int(self.width),
            "total_training_nodes": int(self.total_training_nodes),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TypedDegreeHistogramBasis":
        if data.get("format", "typed_degree_histogram_basis_v1") != "typed_degree_histogram_basis_v1":
            raise ValueError("Unsupported typed-degree histogram basis format")
        result = cls(
            signatures=tuple(
                TypedDegreeKey.from_dict(item) for item in data.get("signatures", [])
            ),
            node_values=tuple(data.get("node_values", [])),
            edge_values=tuple(data.get("edge_values", [])),
            training_counts=tuple(int(value) for value in data.get("training_counts", [])),
            overflow_count=int(data.get("overflow_count", 0)),
            overflow_key=str(data.get("overflow_key", TYPED_DEGREE_OVERFLOW_KEY)),
        )
        expected_width = data.get("width", result.width)
        if int(expected_width) != result.width:
            raise ValueError("Serialized typed-degree basis width is inconsistent")
        return result

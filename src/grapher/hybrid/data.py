from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Any, Sequence

import networkx as nx
import numpy as np
import torch

from grapher.construction.coarse import repair_connectivity_degree_preserving
from grapher.construction.typed import construct_typed_graph
from grapher.molecular.typed_invariants import (
    extract_typed_invariant,
    typed_invariant_matches_graph,
)
from grapher.properties.summary import SummaryConfig
from grapher.refinement.rewiring import (
    Action,
    candidate_actions_from_edge_pair,
    is_valid_action,
    make_action,
    sample_valid_double_edge_swaps,
)
from grapher.utils.motifs import (
    attributed_graphlet_count_dict,
    graphlet_count_dict,
    normalize_count_dict,
    topology_graphlet_keys_by_size,
)

GRAPHLET_OVERFLOW_KEY = "__overflow__"


def _normalized_graph(graph: nx.Graph) -> nx.Graph:
    """Return a simple integer-labelled graph without changing node order."""

    if graph.is_directed() or graph.is_multigraph():
        raise ValueError("Hybrid GraphER requires a simple undirected graph.")
    out = nx.convert_node_labels_to_integers(
        nx.Graph(graph),
        first_label=0,
        ordering="sorted",
        label_attribute="_original_node",
    )
    if nx.number_of_selfloops(out):
        raise ValueError("Hybrid GraphER does not support self-loops.")
    return out


@dataclass(frozen=True)
class GraphCategoryVocabulary:
    """Mapping between graph attributes and categorical tensor indices.

    Edge index zero is always reserved for ``no edge``.  For a topology-only
    graph the vocabulary is one node category and one present-edge category.
    """

    node_values: tuple[Any, ...] = (0,)
    edge_values: tuple[Any, ...] = (1,)
    node_attribute: str | None = None
    edge_attribute: str | None = None

    @property
    def num_node_categories(self) -> int:
        return len(self.node_values)

    @property
    def num_edge_categories(self) -> int:
        return len(self.edge_values) + 1

    @classmethod
    def topology_only(cls) -> "GraphCategoryVocabulary":
        return cls()

    @classmethod
    def from_graphs(
        cls,
        graphs: Sequence[nx.Graph],
        config: dict[str, Any] | None = None,
    ) -> "GraphCategoryVocabulary":
        cfg = config or {}
        node_attribute = cfg.get("node_attribute")
        edge_attribute = cfg.get("edge_attribute")
        configured_nodes = cfg.get("node_categories")
        configured_edges = cfg.get("edge_categories")

        if not node_attribute:
            node_values: tuple[Any, ...] = tuple(configured_nodes or [0])
        else:
            values = (
                list(configured_nodes)
                if configured_nodes is not None
                else [
                    data[node_attribute]
                    for graph in graphs
                    for _, data in graph.nodes(data=True)
                    if node_attribute in data
                ]
            )
            node_values = tuple(sorted(set(values), key=lambda value: repr(value)))

        if not edge_attribute:
            edge_values: tuple[Any, ...] = tuple(configured_edges or [1])
        else:
            values = (
                list(configured_edges)
                if configured_edges is not None
                else [
                    data[edge_attribute]
                    for graph in graphs
                    for _, _, data in graph.edges(data=True)
                    if edge_attribute in data
                ]
            )
            edge_values = tuple(sorted(set(values), key=lambda value: repr(value)))

        if not node_values:
            raise ValueError("At least one node category is required.")
        if not edge_values:
            raise ValueError("At least one present-edge category is required.")
        return cls(
            node_values=node_values,
            edge_values=edge_values,
            node_attribute=str(node_attribute) if node_attribute else None,
            edge_attribute=str(edge_attribute) if edge_attribute else None,
        )

    def node_index(self, data: dict[str, Any]) -> int:
        if self.node_attribute and self.node_attribute not in data:
            raise KeyError(f"Missing node category attribute {self.node_attribute!r}.")
        value = (
            data[self.node_attribute] if self.node_attribute else self.node_values[0]
        )
        try:
            return self.node_values.index(value)
        except ValueError as exc:
            raise ValueError(f"Unknown node category {value!r}.") from exc

    def edge_index(self, data: dict[str, Any]) -> int:
        if self.edge_attribute and self.edge_attribute not in data:
            raise KeyError(f"Missing edge category attribute {self.edge_attribute!r}.")
        value = (
            data[self.edge_attribute] if self.edge_attribute else self.edge_values[0]
        )
        try:
            return self.edge_values.index(value) + 1
        except ValueError as exc:
            raise ValueError(f"Unknown edge category {value!r}.") from exc

    def node_value(self, index: int) -> Any:
        return self.node_values[int(index)]

    def edge_value(self, index: int) -> Any:
        index = int(index)
        if index <= 0:
            raise ValueError("Edge index zero denotes no edge.")
        return self.edge_values[index - 1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_values": list(self.node_values),
            "edge_values": list(self.edge_values),
            "node_attribute": self.node_attribute,
            "edge_attribute": self.edge_attribute,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GraphCategoryVocabulary":
        return cls(
            node_values=tuple(data.get("node_values", [0])),
            edge_values=tuple(data.get("edge_values", [1])),
            node_attribute=data.get("node_attribute"),
            edge_attribute=data.get("edge_attribute"),
        )


@dataclass(frozen=True)
class GraphletBasis:
    keys_by_k: dict[str, tuple[str, ...]]
    connected_only: bool = True
    attributed: bool = False
    node_attribute: str | None = None
    edge_attribute: str | None = None
    overflow_key: str | None = None
    attributed_backend: str = "auto"

    @classmethod
    def from_config(
        cls,
        config: SummaryConfig | dict[str, Any],
    ) -> "GraphletBasis":
        cfg = (
            config
            if isinstance(config, SummaryConfig)
            else SummaryConfig.from_dict(config or {})
        )
        keys = topology_graphlet_keys_by_size(
            cfg.graphlet_k_min,
            cfg.graphlet_k_max,
            connected_only=cfg.graphlet_connected_only,
        )
        return cls(
            keys_by_k={key: tuple(value) for key, value in keys.items()},
            connected_only=cfg.graphlet_connected_only,
        )

    @classmethod
    def fit_from_graphs(
        cls,
        graphs: Sequence[nx.Graph],
        config: SummaryConfig | dict[str, Any],
        *,
        vocabulary: GraphCategoryVocabulary | None = None,
        attributed: bool | None = None,
        seed: int = 0,
    ) -> "GraphletBasis":
        """Build a training-only vocabulary with one unseen overflow class."""

        cfg = (
            config
            if isinstance(config, SummaryConfig)
            else SummaryConfig.from_dict(config or {})
        )
        raw = config if isinstance(config, dict) else {}
        if attributed is None:
            attributed = bool(
                raw.get("attributed", raw.get("graphlet_attributed", False))
                or (
                    vocabulary is not None
                    and vocabulary.node_attribute is not None
                    and vocabulary.edge_attribute is not None
                )
            )
        node_attribute = (
            vocabulary.node_attribute if vocabulary is not None else None
        ) or raw.get("node_attribute")
        edge_attribute = (
            vocabulary.edge_attribute if vocabulary is not None else None
        ) or raw.get("edge_attribute")
        if attributed and (not node_attribute or not edge_attribute):
            raise ValueError(
                "Attributed graphlets require node_attribute and edge_attribute."
            )
        if not attributed:
            # Topology-only graphlets have a complete, finite atlas basis.  The
            # train-only vocabulary plus overflow rule is specifically needed
            # for attributed graphlets, whose labelled state space is sparse.
            return cls.from_config(cfg)

        keys_by_k: dict[str, tuple[str, ...]] = {}
        rng = np.random.default_rng(int(seed))
        for k in range(cfg.graphlet_k_min, cfg.graphlet_k_max + 1):
            keys: set[str] = set()
            for graph in graphs:
                if attributed:
                    counts = attributed_graphlet_count_dict(
                        graph,
                        k,
                        node_label_attr=str(node_attribute),
                        edge_label_attr=str(edge_attribute),
                        connected_only=cfg.graphlet_connected_only,
                        num_samples=cfg.graphlet_num_samples,
                        rng=rng,
                        backend=str(raw.get("attributed_backend", "auto")),
                    )
                else:
                    counts = graphlet_count_dict(
                        graph,
                        k,
                        connected_only=cfg.graphlet_connected_only,
                        num_samples=cfg.graphlet_num_samples,
                        rng=rng,
                    )
                keys.update(str(key) for key in counts)
            ordered = sorted(keys)
            ordered.append(GRAPHLET_OVERFLOW_KEY)
            keys_by_k[str(k)] = tuple(ordered)
        return cls(
            keys_by_k=keys_by_k,
            connected_only=cfg.graphlet_connected_only,
            attributed=bool(attributed),
            node_attribute=str(node_attribute) if node_attribute else None,
            edge_attribute=str(edge_attribute) if edge_attribute else None,
            overflow_key=GRAPHLET_OVERFLOW_KEY,
            attributed_backend=str(raw.get("attributed_backend", "auto")),
        )

    @property
    def sizes(self) -> tuple[str, ...]:
        return tuple(sorted(self.keys_by_k, key=int))

    @property
    def width(self) -> int:
        return sum(len(self.keys_by_k[k]) for k in self.sizes)

    @property
    def slices(self) -> tuple[tuple[int, int], ...]:
        out: list[tuple[int, int]] = []
        start = 0
        for k in self.sizes:
            stop = start + len(self.keys_by_k[k])
            out.append((start, stop))
            start = stop
        return tuple(out)

    def flatten_history(self, history: dict[str, Any]) -> np.ndarray:
        values: list[float] = []
        for k in self.sizes:
            block = history.get(k, {}) or {}
            known = set(self.keys_by_k[k])
            overflow = sum(
                float(value) for key, value in block.items() if key not in known
            )
            for key in self.keys_by_k[k]:
                value = float(block.get(key, 0.0))
                if self.overflow_key is not None and key == self.overflow_key:
                    value += overflow
                values.append(value)
        return np.asarray(values, dtype=np.float32)

    def statistics_for_graph(
        self,
        graph: nx.Graph,
        config: SummaryConfig | dict[str, Any],
        *,
        rng: np.random.Generator | None = None,
    ) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
        """Extract graphlet frequencies in this fixed model vocabulary."""

        cfg = (
            config
            if isinstance(config, SummaryConfig)
            else SummaryConfig.from_dict(config or {})
        )
        history: dict[str, dict[str, float]] = {}
        connected_mass: dict[str, float] = {}
        generator = rng if rng is not None else np.random.default_rng(0)
        n = graph.number_of_nodes()
        for key in self.sizes:
            k = int(key)
            if self.attributed:
                counts = attributed_graphlet_count_dict(
                    graph,
                    k,
                    node_label_attr=str(self.node_attribute),
                    edge_label_attr=str(self.edge_attribute),
                    connected_only=self.connected_only,
                    num_samples=cfg.graphlet_num_samples,
                    rng=generator,
                    backend=self.attributed_backend,
                )
            else:
                counts = graphlet_count_dict(
                    graph,
                    k,
                    connected_only=self.connected_only,
                    num_samples=cfg.graphlet_num_samples,
                    rng=generator,
                )
            normalized = normalize_count_dict(counts)
            flattened = self.flatten_history({key: normalized})
            start, stop = self.slices[self.sizes.index(key)]
            block = flattened[start:stop]
            history[key] = {
                graphlet_key: float(value)
                for graphlet_key, value in zip(self.keys_by_k[key], block)
            }
            total_subsets = comb(n, k) if n >= k else 0
            sampled = (
                min(total_subsets, int(cfg.graphlet_num_samples))
                if cfg.graphlet_num_samples is not None
                and int(cfg.graphlet_num_samples) > 0
                else total_subsets
            )
            connected_mass[key] = (
                float(sum(counts.values()) / sampled)
                if sampled > 0 and self.connected_only
                else float(sampled > 0)
            )
        return history, connected_mass

    def unflatten_history(self, values: Sequence[float]) -> dict[str, dict[str, float]]:
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        if array.size != self.width:
            raise ValueError(
                f"Expected {self.width} graphlet values, received {array.size}."
            )
        out: dict[str, dict[str, float]] = {}
        for k, (start, stop) in zip(self.sizes, self.slices):
            block = np.maximum(array[start:stop], 0.0)
            total = float(block.sum())
            if total > 0.0:
                block = block / total
            out[k] = {key: float(value) for key, value in zip(self.keys_by_k[k], block)}
        return out

    def flatten_mass(self, mass: dict[str, Any]) -> np.ndarray:
        return np.asarray(
            [float((mass or {}).get(k, 0.0)) for k in self.sizes],
            dtype=np.float32,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "keys_by_k": {k: list(v) for k, v in self.keys_by_k.items()},
            "connected_only": self.connected_only,
            "attributed": self.attributed,
            "node_attribute": self.node_attribute,
            "edge_attribute": self.edge_attribute,
            "overflow_key": self.overflow_key,
            "attributed_backend": self.attributed_backend,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GraphletBasis":
        return cls(
            keys_by_k={
                str(k): tuple(str(key) for key in keys)
                for k, keys in (data.get("keys_by_k", {}) or {}).items()
            },
            connected_only=bool(data.get("connected_only", True)),
            attributed=bool(data.get("attributed", False)),
            node_attribute=data.get("node_attribute"),
            edge_attribute=data.get("edge_attribute"),
            overflow_key=data.get("overflow_key"),
            attributed_backend=str(data.get("attributed_backend", "auto")),
        )


@dataclass
class HybridEndpointExample:
    current_graph: nx.Graph
    target_graph: nx.Graph
    time: float
    graphlet_target: np.ndarray
    graphlet_mass_target: np.ndarray
    trajectory_id: int = -1
    step: int = -1
    teacher_actions: tuple[Action, ...] = ()
    teacher_distribution: np.ndarray | None = None
    teacher_selected_index: int = -1


@dataclass
class HybridEndpointBatch:
    current_node_labels: torch.Tensor
    current_edge_labels: torch.Tensor
    target_node_labels: torch.Tensor
    target_edge_labels: torch.Tensor
    node_mask: torch.Tensor
    pair_mask: torch.Tensor
    degrees: torch.Tensor
    typed_degrees: torch.Tensor
    graph_size: torch.Tensor
    time: torch.Tensor
    graphlet_target: torch.Tensor
    graphlet_mass_target: torch.Tensor

    def to(self, device: torch.device | str) -> "HybridEndpointBatch":
        return HybridEndpointBatch(
            **{key: value.to(device) for key, value in self.__dict__.items()}
        )


class EndpointTrajectoryIterableDataset(torch.utils.data.IterableDataset):
    """Build endpoint trajectories lazily with memory bounded by one graph."""

    def __init__(
        self,
        graphs: Sequence[nx.Graph],
        *,
        summary_config: SummaryConfig | dict[str, Any],
        graphlet_basis: GraphletBasis,
        vocabulary: GraphCategoryVocabulary | None = None,
        trajectory_config: dict[str, Any] | None = None,
        seed: int = 0,
        shuffle_graphs: bool = True,
    ) -> None:
        super().__init__()
        self.graphs = tuple(graphs)
        self.summary_config = summary_config
        self.graphlet_basis = graphlet_basis
        self.vocabulary = vocabulary
        self.trajectory_config = dict(trajectory_config or {})
        self.seed = int(seed)
        self.shuffle_graphs = bool(shuffle_graphs)
        self.epoch = 0
        self.last_diagnostics: list[dict[str, Any]] = []

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    @property
    def estimated_examples(self) -> int:
        return (
            len(self.graphs)
            * max(int(self.trajectory_config.get("states_per_graph", 8)), 1)
            * max(int(self.trajectory_config.get("paths_per_graph", 1)), 1)
        )

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        worker_id = worker.id if worker is not None else 0
        worker_count = worker.num_workers if worker is not None else 1
        indices = np.arange(len(self.graphs), dtype=np.int64)
        generator = np.random.default_rng(self.seed + 1_000_003 * self.epoch)
        if self.shuffle_graphs:
            generator.shuffle(indices)
        indices = indices[worker_id::worker_count]
        if worker is None:
            self.last_diagnostics = []
        for position, graph_index in enumerate(indices):
            per_graph_seed = (
                self.seed
                + 1_000_003 * self.epoch
                + 10_007 * int(graph_index)
                + position
            )
            examples, diagnostics = build_endpoint_examples(
                [self.graphs[int(graph_index)]],
                summary_config=self.summary_config,
                graphlet_basis=self.graphlet_basis,
                vocabulary=self.vocabulary,
                trajectory_config=self.trajectory_config,
                seed=per_graph_seed,
            )
            if worker is None:
                self.last_diagnostics.append(diagnostics)
            yield from examples


def graph_to_categorical_arrays(
    graph: nx.Graph,
    vocabulary: GraphCategoryVocabulary,
) -> tuple[np.ndarray, np.ndarray]:
    graph = _normalized_graph(graph)
    n = graph.number_of_nodes()
    nodes = np.zeros(n, dtype=np.int64)
    edges = np.zeros((n, n), dtype=np.int64)
    for node, data in graph.nodes(data=True):
        nodes[int(node)] = vocabulary.node_index(data)
    for u, v, data in graph.edges(data=True):
        category = vocabulary.edge_index(data)
        edges[int(u), int(v)] = category
        edges[int(v), int(u)] = category
    return nodes, edges


def collate_endpoint_examples(
    examples: Sequence[HybridEndpointExample],
    vocabulary: GraphCategoryVocabulary,
) -> HybridEndpointBatch:
    if not examples:
        raise ValueError("Cannot collate an empty endpoint batch.")
    max_nodes = max(example.current_graph.number_of_nodes() for example in examples)
    batch_size = len(examples)
    graphlet_width = int(examples[0].graphlet_target.size)
    mass_width = int(examples[0].graphlet_mass_target.size)

    current_nodes = np.zeros((batch_size, max_nodes), dtype=np.int64)
    current_edges = np.zeros((batch_size, max_nodes, max_nodes), dtype=np.int64)
    target_nodes = np.zeros((batch_size, max_nodes), dtype=np.int64)
    target_edges = np.zeros((batch_size, max_nodes, max_nodes), dtype=np.int64)
    node_mask = np.zeros((batch_size, max_nodes), dtype=np.bool_)
    pair_mask = np.zeros((batch_size, max_nodes, max_nodes), dtype=np.bool_)
    degrees = np.zeros((batch_size, max_nodes), dtype=np.float32)
    typed_degrees = np.zeros(
        (batch_size, max_nodes, vocabulary.num_edge_categories - 1),
        dtype=np.float32,
    )
    graph_sizes = np.zeros(batch_size, dtype=np.float32)
    times = np.zeros(batch_size, dtype=np.float32)
    graphlets = np.zeros((batch_size, graphlet_width), dtype=np.float32)
    masses = np.zeros((batch_size, mass_width), dtype=np.float32)

    for index, example in enumerate(examples):
        current = _normalized_graph(example.current_graph)
        target = _normalized_graph(example.target_graph)
        n = current.number_of_nodes()
        if target.number_of_nodes() != n:
            raise ValueError("Current and target graphs must have the same nodes.")
        current_degree = [int(current.degree(node)) for node in range(n)]
        target_degree = [int(target.degree(node)) for node in range(n)]
        if current_degree != target_degree:
            raise ValueError(
                "Endpoint training pairs must preserve each labelled node degree."
            )
        cn, ce = graph_to_categorical_arrays(current, vocabulary)
        tn, te = graph_to_categorical_arrays(target, vocabulary)
        if vocabulary.node_attribute and not np.array_equal(cn, tn):
            raise ValueError(
                "Invariant node categories must agree in current and target graphs."
            )
        current_typed = np.stack(
            [
                (ce == category).sum(axis=1)
                for category in range(1, vocabulary.num_edge_categories)
            ],
            axis=-1,
        ).astype(np.float32)
        target_typed = np.stack(
            [
                (te == category).sum(axis=1)
                for category in range(1, vocabulary.num_edge_categories)
            ],
            axis=-1,
        ).astype(np.float32)
        if vocabulary.edge_attribute and not np.array_equal(
            current_typed, target_typed
        ):
            raise ValueError(
                "Endpoint training pairs must preserve every indexed typed degree."
            )
        current_nodes[index, :n] = cn
        current_edges[index, :n, :n] = ce
        target_nodes[index, :n] = tn
        target_edges[index, :n, :n] = te
        node_mask[index, :n] = True
        pair_mask[index, :n, :n] = True
        np.fill_diagonal(pair_mask[index], False)
        degrees[index, :n] = np.asarray(current_degree, dtype=np.float32) / max(
            n - 1, 1
        )
        typed_degrees[index, :n] = current_typed
        graph_sizes[index] = float(n)
        times[index] = float(np.clip(example.time, 0.0, 1.0))
        graphlets[index] = np.asarray(example.graphlet_target, dtype=np.float32)
        masses[index] = np.asarray(
            example.graphlet_mass_target,
            dtype=np.float32,
        )

    return HybridEndpointBatch(
        current_node_labels=torch.from_numpy(current_nodes),
        current_edge_labels=torch.from_numpy(current_edges),
        target_node_labels=torch.from_numpy(target_nodes),
        target_edge_labels=torch.from_numpy(target_edges),
        node_mask=torch.from_numpy(node_mask),
        pair_mask=torch.from_numpy(pair_mask),
        degrees=torch.from_numpy(degrees),
        typed_degrees=torch.from_numpy(typed_degrees),
        graph_size=torch.from_numpy(graph_sizes),
        time=torch.from_numpy(times),
        graphlet_target=torch.from_numpy(graphlets),
        graphlet_mass_target=torch.from_numpy(masses),
    )


def aligned_havel_hakimi_source(
    target_graph: nx.Graph,
    *,
    ensure_connected: bool = True,
    max_repair_trials: int = 10000,
    rng: np.random.Generator | None = None,
    vocabulary: GraphCategoryVocabulary | None = None,
    typed_constructor_config: dict[str, Any] | None = None,
) -> tuple[nx.Graph, nx.Graph]:
    """Construct an HH source with a per-node correspondence to the target.

    The legacy constructor sorts the degree multiset and may independently
    relabel the source.  That is correct for summary-only refinement but makes
    endpoint adjacency supervision undefined.  Here, the degree at target node
    ``i`` is supplied at position ``i`` to Havel-Hakimi.
    """

    target = _normalized_graph(target_graph)
    generator = rng if rng is not None else np.random.default_rng(0)
    if (
        vocabulary is not None
        and vocabulary.node_attribute is not None
        and vocabulary.edge_attribute is not None
    ):
        invariant = extract_typed_invariant(
            target,
            edge_types=vocabulary.edge_values,
            node_attribute=vocabulary.node_attribute,
            edge_attribute=vocabulary.edge_attribute,
        )
        constructor_cfg = dict(typed_constructor_config or {})
        constructor_cfg.update(
            {
                "ensure_connected": ensure_connected,
                "randomize_assignment": False,
                "max_backtracks": int(
                    constructor_cfg.get("max_backtracks", max_repair_trials)
                ),
            }
        )
        source, diagnostics = construct_typed_graph(
            invariant,
            constructor_cfg,
            generator,
        )
        source.graph.update(target.graph)
        source.graph["constructor"] = "aligned_typed_backtracking"
        source.graph["constructor_diagnostics"] = diagnostics
        return source, target
    degrees = [int(target.degree(node)) for node in range(target.number_of_nodes())]
    if not nx.is_graphical(degrees, method="eg"):
        raise ValueError("Target degree sequence is not graphical.")
    source = nx.havel_hakimi_graph(degrees)
    source = nx.convert_node_labels_to_integers(
        nx.Graph(source),
        first_label=0,
        ordering="sorted",
    )
    if [int(source.degree(i)) for i in source.nodes()] != degrees:
        raise RuntimeError("Havel-Hakimi did not preserve labelled-node degrees.")

    if (
        ensure_connected
        and source.number_of_nodes() > 1
        and not nx.is_connected(source)
    ):
        source = repair_connectivity_degree_preserving(
            source,
            generator,
            max_trials=max_repair_trials,
        )
    if (
        ensure_connected
        and source.number_of_nodes() > 1
        and not nx.is_connected(source)
    ):
        raise RuntimeError("Could not construct a connected aligned HH source.")
    for node, data in target.nodes(data=True):
        source.nodes[int(node)].update(dict(data))
    source.graph.update(target.graph)
    source.graph["constructor"] = "aligned_havel_hakimi"
    return source, target


def _edge_set(graph: nx.Graph) -> set[tuple[int, int]]:
    return {(min(int(u), int(v)), max(int(u), int(v))) for u, v in graph.edges()}


def _edge_disagreement(
    graph: nx.Graph,
    target: nx.Graph,
    vocabulary: GraphCategoryVocabulary | None = None,
) -> float:
    if vocabulary is None or vocabulary.edge_attribute is None:
        return float(len(_edge_set(graph).symmetric_difference(_edge_set(target))))

    def states(value: nx.Graph) -> set[tuple[int, int, Any]]:
        return {
            (
                min(int(u), int(v)),
                max(int(u), int(v)),
                data[vocabulary.edge_attribute],
            )
            for u, v, data in value.edges(data=True)
        }

    return float(len(states(graph).symmetric_difference(states(target))))


def _target_aware_candidates(
    graph: nx.Graph,
    target: nx.Graph,
    *,
    budget: int,
    rng: np.random.Generator,
    preserve_connectivity: bool,
    vocabulary: GraphCategoryVocabulary | None = None,
) -> list[Action]:
    current_edges = _edge_set(graph)
    target_edges = _edge_set(target)
    wrong = list(current_edges - target_edges)
    missing = target_edges - current_edges
    rng.shuffle(wrong)
    out: list[Action] = []
    seen: set[Action] = set()
    if len(wrong) >= 2:
        attempts = 0
        max_attempts = max(100, int(budget) * 50)
        while len(out) < int(budget) and attempts < max_attempts:
            attempts += 1
            indices = rng.choice(len(wrong), size=2, replace=False)
            for action in candidate_actions_from_edge_pair(
                wrong[int(indices[0])],
                wrong[int(indices[1])],
            ):
                if action in seen:
                    continue
                if vocabulary is not None and vocabulary.edge_attribute is not None:
                    removed_types = {
                        graph.edges[u, v].get(vocabulary.edge_attribute)
                        for u, v in action[0]
                    }
                    if None in removed_types or len(removed_types) != 1:
                        continue
                _, added = action
                if not set(added).intersection(missing):
                    continue
                if is_valid_action(
                    graph,
                    action,
                    preserve_connectivity=preserve_connectivity,
                ):
                    out.append(action)
                    seen.add(action)
                    if len(out) >= int(budget):
                        break
    if len(out) < int(budget):
        random_actions = sample_valid_double_edge_swaps(
            graph,
            max(int(budget) - len(out), 1)
            * (8 if vocabulary is not None and vocabulary.edge_attribute else 1),
            rng,
            preserve_connectivity=preserve_connectivity,
        )
        for action in random_actions:
            if vocabulary is not None and vocabulary.edge_attribute is not None:
                removed_types = {
                    graph.edges[u, v].get(vocabulary.edge_attribute)
                    for u, v in action[0]
                }
                if None in removed_types or len(removed_types) != 1:
                    continue
            if action not in seen:
                out.append(action)
                seen.add(action)
    return out[: int(budget)]


def _apply_action_in_place(
    graph: nx.Graph,
    action: Action,
    vocabulary: GraphCategoryVocabulary | None = None,
) -> None:
    removed, added = action
    edge_attributes: dict[str, Any] = {}
    if vocabulary is not None and vocabulary.edge_attribute is not None:
        categories = {
            graph.edges[u, v].get(vocabulary.edge_attribute) for u, v in removed
        }
        if None in categories or len(categories) != 1:
            raise ValueError("Typed teacher swaps require one shared edge category.")
        edge_attributes[vocabulary.edge_attribute] = next(iter(categories))
    for u, v in removed:
        graph.remove_edge(u, v)
    for u, v in added:
        graph.add_edge(u, v, **edge_attributes)


def _teacher_energy(
    graph: nx.Graph,
    target: nx.Graph,
    *,
    vocabulary: GraphCategoryVocabulary | None,
    graphlet_basis: GraphletBasis | None,
    summary_config: SummaryConfig | dict[str, Any] | None,
    target_graphlets: dict[str, dict[str, float]] | None,
    pair_weight: float,
    graphlet_weight: float,
) -> float:
    possible_pairs = max(graph.number_of_nodes() * (graph.number_of_nodes() - 1), 1)
    energy = (
        float(pair_weight)
        * _edge_disagreement(graph, target, vocabulary)
        / possible_pairs
    )
    if (
        graphlet_weight != 0.0
        and graphlet_basis is not None
        and summary_config is not None
        and target_graphlets is not None
    ):
        history, _ = graphlet_basis.statistics_for_graph(graph, summary_config)
        current = graphlet_basis.flatten_history(history).astype(np.float64)
        target_values = graphlet_basis.flatten_history(target_graphlets).astype(
            np.float64
        )
        distances = [
            float(np.linalg.norm(current[start:stop] - target_values[start:stop]))
            for start, stop in graphlet_basis.slices
            if stop > start
        ]
        energy += float(graphlet_weight) * (
            float(np.mean(distances)) if distances else 0.0
        )
    return float(energy)


def build_aligned_teacher_states(
    target_graph: nx.Graph,
    *,
    steps: int,
    candidate_budget: int,
    preserve_connectivity: bool,
    rng: np.random.Generator,
    ensure_connected_source: bool = True,
    max_repair_trials: int = 10000,
    vocabulary: GraphCategoryVocabulary | None = None,
    typed_constructor_config: dict[str, Any] | None = None,
    source_randomization_steps: int = 0,
    graphlet_basis: GraphletBasis | None = None,
    summary_config: SummaryConfig | dict[str, Any] | None = None,
    teacher_mode: str = "hard",
    teacher_temperature: float = 1.0,
    teacher_top_k: int = 0,
    teacher_sample_actions: bool = False,
    teacher_pair_weight: float = 1.0,
    teacher_graphlet_weight: float = 1.0,
    teacher_min_improvement: float = 0.0,
) -> tuple[list[nx.Graph], nx.Graph, dict[str, Any]]:
    """Build endpoint-guided intermediate graphs in one fixed degree fibre."""

    source, target = aligned_havel_hakimi_source(
        target_graph,
        ensure_connected=ensure_connected_source,
        max_repair_trials=max_repair_trials,
        rng=rng,
        vocabulary=vocabulary,
        typed_constructor_config=typed_constructor_config,
    )
    generic_randomization_steps = (
        0
        if vocabulary is not None and vocabulary.edge_attribute is not None
        else max(int(source_randomization_steps), 0)
    )
    for _ in range(generic_randomization_steps):
        candidates = sample_valid_double_edge_swaps(
            source,
            1,
            rng,
            preserve_connectivity=preserve_connectivity,
        )
        if not candidates:
            break
        _apply_action_in_place(source, candidates[0], vocabulary)
    graph = source.copy()
    states = [graph.copy()]
    indexed_invariant = (
        extract_typed_invariant(
            graph,
            edge_types=vocabulary.edge_values,
            node_attribute=str(vocabulary.node_attribute),
            edge_attribute=str(vocabulary.edge_attribute),
        )
        if vocabulary is not None
        and vocabulary.node_attribute is not None
        and vocabulary.edge_attribute is not None
        else None
    )
    initial = _edge_disagreement(graph, target, vocabulary)
    teacher_mode = str(teacher_mode).lower()
    if teacher_mode not in {"hard", "soft"}:
        raise ValueError("teacher_mode must be hard or soft.")
    target_graphlets = None
    if graphlet_basis is not None and summary_config is not None:
        target_graphlets, _ = graphlet_basis.statistics_for_graph(
            target,
            summary_config,
        )
    accepted = 0
    decisions: list[dict[str, Any]] = []
    stop_reason = "max_steps"
    for step in range(int(steps)):
        if _edge_disagreement(graph, target, vocabulary) == 0.0:
            stop_reason = "target_reached"
            break
        candidates = _target_aware_candidates(
            graph,
            target,
            budget=int(candidate_budget),
            rng=rng,
            preserve_connectivity=preserve_connectivity,
            vocabulary=vocabulary,
        )
        if not candidates:
            decisions.append(
                {
                    "step": step,
                    "actions": [],
                    "improvements": [],
                    "distribution": [1.0],
                    "selected_index": 0,
                    "stop_index": 0,
                }
            )
            stop_reason = "no_candidates"
            break
        before_energy = _teacher_energy(
            graph,
            target,
            vocabulary=vocabulary,
            graphlet_basis=graphlet_basis,
            summary_config=summary_config,
            target_graphlets=target_graphlets,
            pair_weight=teacher_pair_weight,
            graphlet_weight=teacher_graphlet_weight,
        )
        improvements: list[float] = []
        for action in candidates:
            candidate = graph.copy()
            _apply_action_in_place(candidate, action, vocabulary)
            improvements.append(
                before_energy
                - _teacher_energy(
                    candidate,
                    target,
                    vocabulary=vocabulary,
                    graphlet_basis=graphlet_basis,
                    summary_config=summary_config,
                    target_graphlets=target_graphlets,
                    pair_weight=teacher_pair_weight,
                    graphlet_weight=teacher_graphlet_weight,
                )
            )
        improving = [
            index
            for index, value in enumerate(improvements)
            if value > float(teacher_min_improvement)
        ]
        if teacher_top_k > 0 and len(improving) > int(teacher_top_k):
            improving = sorted(
                improving,
                key=lambda index: improvements[index],
                reverse=True,
            )[: int(teacher_top_k)]
        stop_index = len(candidates)
        distribution = np.zeros(stop_index + 1, dtype=np.float64)
        if not improving:
            distribution[stop_index] = 1.0
        elif teacher_mode == "hard":
            selected = max(improving, key=lambda index: improvements[index])
            distribution[selected] = 1.0
        else:
            support = improving + [stop_index]
            logits = np.asarray(
                [improvements[index] for index in improving] + [0.0],
                dtype=np.float64,
            ) / max(float(teacher_temperature), 1.0e-12)
            logits -= float(np.max(logits))
            probabilities = np.exp(logits)
            probabilities /= float(probabilities.sum())
            distribution[np.asarray(support, dtype=np.int64)] = probabilities
        chosen_index = (
            int(rng.choice(len(distribution), p=distribution))
            if teacher_sample_actions
            else int(np.argmax(distribution))
        )
        decisions.append(
            {
                "step": step,
                "actions": [
                    [[list(edge) for edge in removed], [list(edge) for edge in added]]
                    for removed, added in candidates
                ],
                "improvements": [float(value) for value in improvements],
                "distribution": distribution.tolist(),
                "selected_index": chosen_index,
                "stop_index": stop_index,
            }
        )
        if chosen_index == stop_index:
            stop_reason = "teacher_stop"
            break
        _apply_action_in_place(graph, candidates[chosen_index], vocabulary)
        if indexed_invariant is not None:
            if not typed_invariant_matches_graph(graph, indexed_invariant):
                raise AssertionError("A teacher action changed the typed invariant.")
        states.append(graph.copy())
        accepted += 1

    reached = _edge_disagreement(graph, target, vocabulary) == 0.0
    report = {
        "initial_edge_disagreement": float(initial),
        "final_teacher_edge_disagreement": float(
            _edge_disagreement(graph, target, vocabulary)
        ),
        "accepted_teacher_steps": int(accepted),
        "reached_target": bool(reached),
        "teacher_stop_reason": stop_reason,
        "teacher_stop_selected": bool(stop_reason == "teacher_stop"),
        "teacher_decisions": decisions,
        "mean_valid_candidates": float(
            np.mean([len(value["actions"]) for value in decisions])
        )
        if decisions
        else 0.0,
    }
    return states, target, report


def _shared_permutation(
    current: nx.Graph,
    target: nx.Graph,
    rng: np.random.Generator,
) -> tuple[nx.Graph, nx.Graph]:
    left, right, _ = _shared_permutation_with_mapping(current, target, rng)
    return left, right


def _shared_permutation_with_mapping(
    current: nx.Graph,
    target: nx.Graph,
    rng: np.random.Generator,
) -> tuple[nx.Graph, nx.Graph, dict[int, int]]:
    n = current.number_of_nodes()
    permutation = rng.permutation(n).tolist()
    mapping = {node: int(permutation[node]) for node in range(n)}
    left = nx.relabel_nodes(current, mapping, copy=True)
    right = nx.relabel_nodes(target, mapping, copy=True)
    left = nx.convert_node_labels_to_integers(left, ordering="sorted")
    right = nx.convert_node_labels_to_integers(right, ordering="sorted")
    return left, right, mapping


def _permuted_action(action: Action, mapping: dict[int, int]) -> Action:
    removed, added = action
    return make_action(
        [(mapping[int(u)], mapping[int(v)]) for u, v in removed],
        [(mapping[int(u)], mapping[int(v)]) for u, v in added],
    )


def build_endpoint_examples(
    graphs: Sequence[nx.Graph],
    *,
    summary_config: SummaryConfig | dict[str, Any],
    graphlet_basis: GraphletBasis,
    vocabulary: GraphCategoryVocabulary | None = None,
    trajectory_config: dict[str, Any] | None = None,
    seed: int = 0,
) -> tuple[list[HybridEndpointExample], dict[str, Any]]:
    """Create ``(G_t, G_1, H(G_1))`` examples from aligned teacher paths."""

    cfg = trajectory_config or {}
    summary_cfg = (
        summary_config
        if isinstance(summary_config, SummaryConfig)
        else SummaryConfig.from_dict(summary_config or {})
    )
    rng = np.random.default_rng(int(seed))
    examples: list[HybridEndpointExample] = []
    reports: list[dict[str, Any]] = []
    trajectory_id = 0
    paths_per_graph = max(int(cfg.get("paths_per_graph", 1)), 1)
    for raw_target in graphs:
        for _path in range(paths_per_graph):
            states, target, report = build_aligned_teacher_states(
                raw_target,
                steps=int(cfg.get("steps", 32)),
                candidate_budget=int(cfg.get("candidate_budget", 64)),
                preserve_connectivity=bool(cfg.get("preserve_connectivity", True)),
                ensure_connected_source=bool(cfg.get("ensure_connected_source", True)),
                max_repair_trials=int(cfg.get("max_repair_trials", 10000)),
                vocabulary=vocabulary,
                typed_constructor_config=cfg.get("typed_constructor", {}) or {},
                source_randomization_steps=int(
                    cfg.get("source_randomization_steps", 0)
                ),
                graphlet_basis=graphlet_basis,
                summary_config=summary_cfg,
                teacher_mode=str(cfg.get("teacher_mode", "hard")),
                teacher_temperature=float(cfg.get("teacher_temperature", 1.0)),
                teacher_top_k=int(cfg.get("teacher_top_k", 0)),
                teacher_sample_actions=bool(cfg.get("teacher_sample_actions", False)),
                teacher_pair_weight=float(cfg.get("teacher_pair_weight", 1.0)),
                teacher_graphlet_weight=float(cfg.get("teacher_graphlet_weight", 1.0)),
                teacher_min_improvement=float(cfg.get("teacher_min_improvement", 0.0)),
                rng=rng,
            )
            graphlet_history, graphlet_connected_mass = (
                graphlet_basis.statistics_for_graph(target, summary_cfg, rng=rng)
            )
            graphlet_target = graphlet_basis.flatten_history(graphlet_history)
            graphlet_mass = graphlet_basis.flatten_mass(graphlet_connected_mass)
            state_count = len(states)
            selected_count = min(
                max(int(cfg.get("states_per_graph", 8)), 1),
                state_count,
            )
            indices = (
                list(range(state_count))
                if selected_count == state_count
                else sorted(
                    {
                        int(round(value))
                        for value in np.linspace(
                            0,
                            state_count - 1,
                            num=selected_count,
                        )
                    }
                )
            )
            denominator = max(int(cfg.get("steps", 32)), 1)
            decisions = report.get("teacher_decisions", [])
            for step in indices:
                current = states[step]
                target_for_example = target
                decision = decisions[step] if step < len(decisions) else None
                teacher_actions: tuple[Action, ...] = ()
                teacher_distribution = None
                teacher_selected_index = -1
                if decision is not None:
                    teacher_actions = tuple(
                        make_action(value[0], value[1]) for value in decision["actions"]
                    )
                    teacher_distribution = np.asarray(
                        decision["distribution"], dtype=np.float32
                    )
                    teacher_selected_index = int(decision["selected_index"])
                if bool(cfg.get("shared_relabel_augmentation", True)):
                    current, target_for_example, mapping = (
                        _shared_permutation_with_mapping(current, target, rng)
                    )
                    teacher_actions = tuple(
                        _permuted_action(action, mapping) for action in teacher_actions
                    )
                examples.append(
                    HybridEndpointExample(
                        current_graph=current,
                        target_graph=target_for_example,
                        time=float(step / denominator),
                        graphlet_target=graphlet_target.copy(),
                        graphlet_mass_target=graphlet_mass.copy(),
                        trajectory_id=trajectory_id,
                        step=step,
                        teacher_actions=teacher_actions,
                        teacher_distribution=teacher_distribution,
                        teacher_selected_index=teacher_selected_index,
                    )
                )
            reports.append(report)
            trajectory_id += 1

    reached = [float(report["reached_target"]) for report in reports]
    diagnostics = {
        "num_graphs": len(graphs),
        "num_paths": len(reports),
        "num_examples": len(examples),
        "teacher_target_reach_rate": float(np.mean(reached)) if reached else 0.0,
        "mean_initial_edge_disagreement": float(
            np.mean([r["initial_edge_disagreement"] for r in reports])
        )
        if reports
        else 0.0,
        "mean_final_teacher_edge_disagreement": float(
            np.mean([r["final_teacher_edge_disagreement"] for r in reports])
        )
        if reports
        else 0.0,
        "mean_accepted_teacher_steps": float(
            np.mean([r["accepted_teacher_steps"] for r in reports])
        )
        if reports
        else 0.0,
        "teacher_stop_rate": float(
            np.mean([bool(r.get("teacher_stop_selected", False)) for r in reports])
        )
        if reports
        else 0.0,
        "mean_valid_candidates": float(
            np.mean([float(r.get("mean_valid_candidates", 0.0)) for r in reports])
        )
        if reports
        else 0.0,
    }
    return examples, diagnostics

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import networkx as nx
import numpy as np
import torch

from grapher.construction.coarse import repair_connectivity_degree_preserving
from grapher.properties.summary import SummaryConfig, extract_summary
from grapher.refinement.rewiring import (
    Action,
    candidate_actions_from_edge_pair,
    is_valid_action,
    sample_valid_double_edge_swaps,
)
from grapher.utils.motifs import topology_graphlet_keys_by_size


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
        value = (
            data.get(self.node_attribute, self.node_values[0])
            if self.node_attribute
            else self.node_values[0]
        )
        try:
            return self.node_values.index(value)
        except ValueError as exc:
            raise ValueError(f"Unknown node category {value!r}.") from exc

    def edge_index(self, data: dict[str, Any]) -> int:
        value = (
            data.get(self.edge_attribute, self.edge_values[0])
            if self.edge_attribute
            else self.edge_values[0]
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
            values.extend(float(block.get(key, 0.0)) for key in self.keys_by_k[k])
        return np.asarray(values, dtype=np.float32)

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
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GraphletBasis":
        return cls(
            keys_by_k={
                str(k): tuple(str(key) for key in keys)
                for k, keys in (data.get("keys_by_k", {}) or {}).items()
            },
            connected_only=bool(data.get("connected_only", True)),
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


@dataclass
class HybridEndpointBatch:
    current_node_labels: torch.Tensor
    current_edge_labels: torch.Tensor
    target_node_labels: torch.Tensor
    target_edge_labels: torch.Tensor
    node_mask: torch.Tensor
    pair_mask: torch.Tensor
    degrees: torch.Tensor
    time: torch.Tensor
    graphlet_target: torch.Tensor
    graphlet_mass_target: torch.Tensor

    def to(self, device: torch.device | str) -> "HybridEndpointBatch":
        return HybridEndpointBatch(
            **{key: value.to(device) for key, value in self.__dict__.items()}
        )


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
) -> tuple[nx.Graph, nx.Graph]:
    """Construct an HH source with a per-node correspondence to the target.

    The legacy constructor sorts the degree multiset and may independently
    relabel the source.  That is correct for summary-only refinement but makes
    endpoint adjacency supervision undefined.  Here, the degree at target node
    ``i`` is supplied at position ``i`` to Havel-Hakimi.
    """

    target = _normalized_graph(target_graph)
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

    generator = rng if rng is not None else np.random.default_rng(0)
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


def _edge_disagreement(graph: nx.Graph, target: nx.Graph) -> float:
    return float(len(_edge_set(graph).symmetric_difference(_edge_set(target))))


def _target_aware_candidates(
    graph: nx.Graph,
    target: nx.Graph,
    *,
    budget: int,
    rng: np.random.Generator,
    preserve_connectivity: bool,
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
            int(budget) - len(out),
            rng,
            preserve_connectivity=preserve_connectivity,
        )
        for action in random_actions:
            if action not in seen:
                out.append(action)
                seen.add(action)
    return out[: int(budget)]


def _apply_action_in_place(graph: nx.Graph, action: Action) -> None:
    removed, added = action
    for u, v in removed:
        graph.remove_edge(u, v)
    for u, v in added:
        graph.add_edge(u, v)


def build_aligned_teacher_states(
    target_graph: nx.Graph,
    *,
    steps: int,
    candidate_budget: int,
    preserve_connectivity: bool,
    rng: np.random.Generator,
    ensure_connected_source: bool = True,
    max_repair_trials: int = 10000,
) -> tuple[list[nx.Graph], nx.Graph, dict[str, Any]]:
    """Build endpoint-guided intermediate graphs in one fixed degree fibre."""

    source, target = aligned_havel_hakimi_source(
        target_graph,
        ensure_connected=ensure_connected_source,
        max_repair_trials=max_repair_trials,
        rng=rng,
    )
    graph = source.copy()
    states = [graph.copy()]
    initial = _edge_disagreement(graph, target)
    accepted = 0
    for _ in range(int(steps)):
        if _edge_set(graph) == _edge_set(target):
            break
        candidates = _target_aware_candidates(
            graph,
            target,
            budget=int(candidate_budget),
            rng=rng,
            preserve_connectivity=preserve_connectivity,
        )
        if not candidates:
            break
        before = _edge_disagreement(graph, target)
        improvements = []
        for action in candidates:
            candidate = graph.copy()
            _apply_action_in_place(candidate, action)
            improvements.append(before - _edge_disagreement(candidate, target))
        best = int(np.argmax(np.asarray(improvements, dtype=np.float64)))
        if improvements[best] <= 0.0:
            break
        _apply_action_in_place(graph, candidates[best])
        states.append(graph.copy())
        accepted += 1

    reached = _edge_set(graph) == _edge_set(target)
    if not reached:
        # The clean endpoint is still a valid t=1 denoising example.  It is
        # intentionally not reported as a teacher transition.
        states.append(target.copy())
    report = {
        "initial_edge_disagreement": float(initial),
        "final_teacher_edge_disagreement": float(_edge_disagreement(graph, target)),
        "accepted_teacher_steps": int(accepted),
        "reached_target": bool(reached),
    }
    return states, target, report


def _shared_permutation(
    current: nx.Graph,
    target: nx.Graph,
    rng: np.random.Generator,
) -> tuple[nx.Graph, nx.Graph]:
    n = current.number_of_nodes()
    permutation = rng.permutation(n).tolist()
    mapping = {node: int(permutation[node]) for node in range(n)}
    left = nx.relabel_nodes(current, mapping, copy=True)
    right = nx.relabel_nodes(target, mapping, copy=True)
    left = nx.convert_node_labels_to_integers(left, ordering="sorted")
    right = nx.convert_node_labels_to_integers(right, ordering="sorted")
    return left, right


def build_endpoint_examples(
    graphs: Sequence[nx.Graph],
    *,
    summary_config: SummaryConfig | dict[str, Any],
    graphlet_basis: GraphletBasis,
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
    for trajectory_id, raw_target in enumerate(graphs):
        states, target, report = build_aligned_teacher_states(
            raw_target,
            steps=int(cfg.get("steps", 32)),
            candidate_budget=int(cfg.get("candidate_budget", 64)),
            preserve_connectivity=bool(cfg.get("preserve_connectivity", True)),
            ensure_connected_source=bool(cfg.get("ensure_connected_source", True)),
            max_repair_trials=int(cfg.get("max_repair_trials", 10000)),
            rng=rng,
        )
        summary = extract_summary(target, summary_cfg)
        graphlet_target = graphlet_basis.flatten_history(
            summary.get("graphlet_history", {}) or {}
        )
        graphlet_mass = graphlet_basis.flatten_mass(
            summary.get("graphlet_connected_mass", {}) or {}
        )
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
        denominator = max(len(states) - 1, 1)
        for step in indices:
            current = states[step]
            target_for_example = target
            if bool(cfg.get("shared_relabel_augmentation", True)):
                current, target_for_example = _shared_permutation(
                    current,
                    target,
                    rng,
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
                )
            )
        reports.append(report)

    reached = [float(report["reached_target"]) for report in reports]
    diagnostics = {
        "num_graphs": len(graphs),
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
    }
    return examples, diagnostics

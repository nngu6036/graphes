from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import networkx as nx
import numpy as np
import torch

from grapher.rewiring_mlp.generic.basis import TopologyGraphletBasis
from grapher.rewiring_mlp.generic.data import (
    TopologyTrainingPair,
    normalize_topology_graph,
)
from grapher.rewiring_mlp.generic.graphlet_diffusion import (
    extract_topology_graphlet_simplex,
    graphlet_simplex_to_clr,
)
from grapher.rewiring_mlp.generic.spectral_data import (
    _resolve_spectral_diffusion_endpoints,
)


@dataclass(frozen=True)
class FlowMatchingConfig:
    """Continuous adjacency-space conditional flow matching configuration.

    The default path is deterministic linear interpolation

        P_t = (1-t) A_source + t A_target,
        U_t = A_target - A_source.

    Source and target are first aligned inside equal-degree node classes so the
    indexed degree vector is identical at both endpoints. Consequently every
    linear intermediate state has the same row sums and the conditional target
    velocity lies in the degree-preserving tangent space.
    """

    storage: str = "streaming"
    path: str = "linear"
    samples_per_graph: int = 32
    paths_per_graph: int = 2
    time_sampling: str = "stratified"
    min_progress: float = 0.0
    max_progress: float = 1.0
    require_same_degree_sequence: bool = True
    align_nodes_by_degree: bool = True
    randomize_equal_degree_alignment: bool = True
    joint_random_relabel: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None = None) -> "FlowMatchingConfig":
        values = dict(data or {})
        config = cls(
            storage=str(values.get("storage", "streaming")).lower(),
            path=str(values.get("path", "linear")).lower(),
            samples_per_graph=max(int(values.get("samples_per_graph", 32)), 1),
            paths_per_graph=max(int(values.get("paths_per_graph", 2)), 1),
            time_sampling=str(values.get("time_sampling", "stratified")).lower(),
            min_progress=float(values.get("min_progress", 0.0)),
            max_progress=float(values.get("max_progress", 1.0)),
            require_same_degree_sequence=bool(
                values.get("require_same_degree_sequence", True)
            ),
            align_nodes_by_degree=bool(values.get("align_nodes_by_degree", True)),
            randomize_equal_degree_alignment=bool(
                values.get("randomize_equal_degree_alignment", True)
            ),
            joint_random_relabel=bool(values.get("joint_random_relabel", True)),
        )
        if config.storage not in {"eager", "streaming"}:
            raise ValueError("flow_matching.storage must be eager or streaming.")
        if config.path not in {"linear", "conditional_linear"}:
            raise ValueError("flow_matching.path currently supports only linear.")
        if config.time_sampling not in {"uniform", "stratified", "grid"}:
            raise ValueError(
                "flow_matching.time_sampling must be uniform, stratified, or grid."
            )
        if not (0.0 <= config.min_progress <= config.max_progress <= 1.0):
            raise ValueError(
                "flow_matching progress bounds must satisfy 0 <= min <= max <= 1."
            )
        if not config.require_same_degree_sequence:
            raise ValueError(
                "Degree-preserving flow+graphlet training requires "
                "flow_matching.require_same_degree_sequence: true."
            )
        if not config.align_nodes_by_degree:
            raise ValueError(
                "Degree-preserving edge flow requires flow_matching.align_nodes_by_degree: true."
            )
        return config

    def sample_progresses(
        self,
        count: int,
        *,
        rng: np.random.Generator,
    ) -> np.ndarray:
        count = max(int(count), 1)
        lo = float(self.min_progress)
        hi = float(self.max_progress)
        if hi <= lo:
            return np.full(count, lo, dtype=np.float64)
        if self.time_sampling == "uniform":
            return rng.uniform(lo, hi, size=count).astype(np.float64)
        if self.time_sampling == "grid":
            if count == 1:
                return np.asarray([(lo + hi) * 0.5], dtype=np.float64)
            return np.linspace(lo, hi, num=count, dtype=np.float64)
        # Stratified sampling gives every epoch coverage over the complete path
        # while still changing the exact states between epochs.
        offsets = (np.arange(count, dtype=np.float64) + rng.random(count)) / count
        values = lo + (hi - lo) * offsets
        rng.shuffle(values)
        return values.astype(np.float64)


@dataclass
class TopologyFlowGraphletExample:
    source_graph: nx.Graph
    time: float
    current_edge_probabilities: np.ndarray
    flow_target: np.ndarray
    clean_graphlet_probabilities_target: np.ndarray
    clean_graphlet_logits_target: np.ndarray
    graphlet_coordinate_mask: np.ndarray
    base_generator: str = "target_degree_havel_hakimi"
    source_index: int = -1
    target_index: int = -1
    matching_cost: float = 0.0
    path_id: int = -1
    sample_index: int = -1


@dataclass
class TopologyFlowGraphletBatch:
    source_adjacency: torch.Tensor
    current_edge_probabilities: torch.Tensor
    flow_target: torch.Tensor
    node_mask: torch.Tensor
    pair_mask: torch.Tensor
    degrees: torch.Tensor
    graph_size: torch.Tensor
    time: torch.Tensor
    clean_graphlet_probabilities_target: torch.Tensor
    clean_graphlet_logits_target: torch.Tensor
    graphlet_coordinate_mask: torch.Tensor

    def to(self, device: torch.device | str) -> "TopologyFlowGraphletBatch":
        return TopologyFlowGraphletBatch(
            **{
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in self.__dict__.items()
            }
        )


def collate_flow_graphlet_examples(
    examples: Sequence[TopologyFlowGraphletExample],
) -> TopologyFlowGraphletBatch:
    if not examples:
        raise ValueError("Cannot collate an empty flow+graphlet batch.")
    max_nodes = max(example.source_graph.number_of_nodes() for example in examples)
    batch_size = len(examples)
    graphlet_widths = {
        int(np.asarray(example.clean_graphlet_logits_target).size) for example in examples
    }
    if len(graphlet_widths) != 1:
        raise ValueError("Flow+graphlet examples must share one graphlet target width.")
    graphlet_width = next(iter(graphlet_widths))

    source_adjacency = np.zeros((batch_size, max_nodes, max_nodes), dtype=np.float32)
    current_probability = np.zeros((batch_size, max_nodes, max_nodes), dtype=np.float32)
    flow_target = np.zeros((batch_size, max_nodes, max_nodes), dtype=np.float32)
    node_mask = np.zeros((batch_size, max_nodes), dtype=np.bool_)
    pair_mask = np.zeros((batch_size, max_nodes, max_nodes), dtype=np.bool_)
    degrees = np.zeros((batch_size, max_nodes), dtype=np.float32)
    graph_size = np.zeros(batch_size, dtype=np.float32)
    time = np.zeros(batch_size, dtype=np.float32)
    graphlet_prob = np.zeros((batch_size, graphlet_width), dtype=np.float32)
    graphlet_logits = np.zeros((batch_size, graphlet_width), dtype=np.float32)
    graphlet_mask = np.zeros((batch_size, graphlet_width), dtype=np.bool_)

    for index, example in enumerate(examples):
        source = normalize_topology_graph(example.source_graph)
        n = source.number_of_nodes()
        source_matrix = nx.to_numpy_array(
            source,
            nodelist=list(range(n)),
            dtype=np.float32,
        )
        current = np.asarray(
            example.current_edge_probabilities,
            dtype=np.float32,
        )
        velocity = np.asarray(example.flow_target, dtype=np.float32)
        if current.shape != (n, n) or velocity.shape != (n, n):
            raise ValueError(
                "Flow examples require n x n current probabilities and velocity targets."
            )
        if not np.allclose(current, current.T, atol=1.0e-6):
            raise ValueError("Current edge probabilities must be symmetric.")
        if not np.allclose(velocity, velocity.T, atol=1.0e-6):
            raise ValueError("Flow targets must be symmetric.")
        if not np.allclose(np.diag(current), 0.0, atol=1.0e-6):
            raise ValueError("Current edge probabilities must have zero diagonal.")
        if not np.allclose(np.diag(velocity), 0.0, atol=1.0e-6):
            raise ValueError("Flow targets must have zero diagonal.")

        source_adjacency[index, :n, :n] = source_matrix
        current_probability[index, :n, :n] = current
        flow_target[index, :n, :n] = velocity
        node_mask[index, :n] = True
        pair_mask[index, :n, :n] = True
        np.fill_diagonal(pair_mask[index], False)
        degrees[index, :n] = np.asarray(
            [float(source.degree(node)) / max(n - 1, 1) for node in range(n)],
            dtype=np.float32,
        )
        graph_size[index] = float(n)
        time[index] = float(np.clip(example.time, 0.0, 1.0))

        probability = np.asarray(
            example.clean_graphlet_probabilities_target, dtype=np.float32
        ).reshape(-1)
        logits = np.asarray(
            example.clean_graphlet_logits_target, dtype=np.float32
        ).reshape(-1)
        mask = np.asarray(example.graphlet_coordinate_mask, dtype=np.bool_).reshape(-1)
        if probability.size != graphlet_width or logits.size != graphlet_width or mask.size != graphlet_width:
            raise ValueError("Flow+graphlet target width mismatch during collation.")
        graphlet_prob[index] = probability
        graphlet_logits[index] = logits
        graphlet_mask[index] = mask

    return TopologyFlowGraphletBatch(
        source_adjacency=torch.from_numpy(source_adjacency),
        current_edge_probabilities=torch.from_numpy(current_probability),
        flow_target=torch.from_numpy(flow_target),
        node_mask=torch.from_numpy(node_mask),
        pair_mask=torch.from_numpy(pair_mask),
        degrees=torch.from_numpy(degrees),
        graph_size=torch.from_numpy(graph_size),
        time=torch.from_numpy(time),
        clean_graphlet_probabilities_target=torch.from_numpy(graphlet_prob),
        clean_graphlet_logits_target=torch.from_numpy(graphlet_logits),
        graphlet_coordinate_mask=torch.from_numpy(graphlet_mask),
    )


def _degree_groups(graph: nx.Graph) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = {}
    for node in sorted(graph.nodes()):
        groups.setdefault(int(graph.degree(node)), []).append(int(node))
    return groups


def align_target_to_source_degrees(
    source: nx.Graph,
    target: nx.Graph,
    *,
    rng: np.random.Generator,
    randomize_equal_degree_alignment: bool = True,
) -> nx.Graph:
    """Relabel target so every node index has the source node's degree.

    Generic graph nodes are unlabeled, so pairing nodes inside an equal-degree
    class is arbitrary. Randomizing that tie matching avoids introducing a fixed
    storage-order correspondence while retaining the exact indexed degree vector
    required by degree-tangent flow matching.
    """

    source = normalize_topology_graph(source)
    target = normalize_topology_graph(target)
    source_groups = _degree_groups(source)
    target_groups = _degree_groups(target)
    if set(source_groups) != set(target_groups):
        raise ValueError("Source and target degree supports differ.")
    mapping: dict[int, int] = {}
    for degree in sorted(source_groups):
        source_nodes = list(source_groups[degree])
        target_nodes = list(target_groups[degree])
        if len(source_nodes) != len(target_nodes):
            raise ValueError("Source and target degree multiplicities differ.")
        if randomize_equal_degree_alignment and len(target_nodes) > 1:
            rng.shuffle(target_nodes)
        for source_node, target_node in zip(source_nodes, target_nodes):
            mapping[int(target_node)] = int(source_node)
    aligned = nx.relabel_nodes(target, mapping, copy=True)
    aligned = normalize_topology_graph(aligned)
    for node in range(source.number_of_nodes()):
        if int(source.degree(node)) != int(aligned.degree(node)):
            raise AssertionError("Degree-class alignment failed to match indexed degrees.")
    return aligned


def _joint_random_relabel(
    source: nx.Graph,
    target: nx.Graph,
    *,
    rng: np.random.Generator,
) -> tuple[nx.Graph, nx.Graph]:
    n = source.number_of_nodes()
    if target.number_of_nodes() != n:
        raise ValueError("Joint relabeling requires equal graph size.")
    permutation = rng.permutation(n)
    mapping = {int(node): int(permutation[node]) for node in range(n)}
    source_out = normalize_topology_graph(nx.relabel_nodes(source, mapping, copy=True))
    target_out = normalize_topology_graph(nx.relabel_nodes(target, mapping, copy=True))
    return source_out, target_out


def _resolve_flow_endpoints(
    raw_item: nx.Graph | TopologyTrainingPair,
    *,
    flow_config: FlowMatchingConfig,
    source_config: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[nx.Graph, nx.Graph, dict[str, Any]]:
    source, target, metadata = _resolve_spectral_diffusion_endpoints(
        raw_item,
        source_config=source_config,
        require_same_degree_sequence=flow_config.require_same_degree_sequence,
        rng=rng,
    )
    target = align_target_to_source_degrees(
        source,
        target,
        rng=rng,
        randomize_equal_degree_alignment=flow_config.randomize_equal_degree_alignment,
    )
    if flow_config.joint_random_relabel:
        source, target = _joint_random_relabel(source, target, rng=rng)
    for node in range(source.number_of_nodes()):
        if int(source.degree(node)) != int(target.degree(node)):
            raise AssertionError("Flow endpoints must have identical indexed degrees.")
    return source, target, metadata


def build_flow_graphlet_examples(
    graphs: Sequence[nx.Graph | TopologyTrainingPair],
    *,
    flow_config: dict[str, Any] | None = None,
    source_config: dict[str, Any] | None = None,
    graphlet_basis: TopologyGraphletBasis,
    graphlet_logit_epsilon: float = 1.0e-5,
    seed: int = 0,
) -> tuple[list[TopologyFlowGraphletExample], dict[str, Any]]:
    """Build continuous edge-probability flow-matching states.

    No rewiring trajectory is used. Only source and terminal graphs are
    materialized. Intermediate states are soft adjacency matrices on the linear
    conditional path; graphlet supervision always uses the clean terminal graph.
    """

    cfg = FlowMatchingConfig.from_dict(flow_config)
    source_cfg = dict(source_config or {})
    rng = np.random.default_rng(int(seed))
    examples: list[TopologyFlowGraphletExample] = []
    endpoint_reports: list[dict[str, Any]] = []
    path_id = 0

    for raw_item in graphs:
        source, target, metadata = _resolve_flow_endpoints(
            raw_item,
            flow_config=cfg,
            source_config=source_cfg,
            rng=rng,
        )
        n = source.number_of_nodes()
        source_adjacency = nx.to_numpy_array(
            source, nodelist=list(range(n)), dtype=np.float64
        )
        target_adjacency = nx.to_numpy_array(
            target, nodelist=list(range(n)), dtype=np.float64
        )
        velocity = target_adjacency - source_adjacency
        degree_residual = np.max(np.abs(velocity.sum(axis=1))) if n else 0.0
        if degree_residual > 1.0e-8:
            raise AssertionError(
                "Conditional flow target left the degree-preserving tangent space."
            )
        source_degrees = source_adjacency.sum(axis=1)
        clean_probability, graphlet_mask, _ = extract_topology_graphlet_simplex(
            target,
            graphlet_basis=graphlet_basis,
        )
        clean_logits = graphlet_simplex_to_clr(
            clean_probability,
            graphlet_basis=graphlet_basis,
            epsilon=float(graphlet_logit_epsilon),
            coordinate_mask=graphlet_mask,
        )

        changed_upper = np.triu(np.abs(velocity) > 0.5, k=1)
        endpoint_reports.append(
            {
                **metadata,
                "num_nodes": int(n),
                "num_edges": int(source.number_of_edges()),
                "changed_pairs": int(changed_upper.sum()),
                "changed_pair_fraction": float(
                    changed_upper.sum() / max(n * (n - 1) / 2, 1)
                ),
                "max_target_degree_tangent_residual": float(degree_residual),
            }
        )

        for _path in range(cfg.paths_per_graph):
            progresses = cfg.sample_progresses(cfg.samples_per_graph, rng=rng)
            for sample_index, progress in enumerate(progresses):
                p = float(progress)
                current = (1.0 - p) * source_adjacency + p * target_adjacency
                degree_path_residual = (
                    float(np.max(np.abs(current.sum(axis=1) - source_degrees)))
                    if n
                    else 0.0
                )
                if degree_path_residual > 1.0e-8:
                    raise AssertionError(
                        "Linear adjacency flow failed to preserve indexed degrees."
                    )
                examples.append(
                    TopologyFlowGraphletExample(
                        source_graph=source.copy(),
                        time=p,
                        current_edge_probabilities=current.astype(np.float32),
                        flow_target=velocity.astype(np.float32),
                        clean_graphlet_probabilities_target=clean_probability.astype(
                            np.float32
                        ),
                        clean_graphlet_logits_target=clean_logits.astype(np.float32),
                        graphlet_coordinate_mask=graphlet_mask.astype(np.bool_),
                        base_generator=str(metadata["base_generator"]),
                        source_index=int(metadata["source_index"]),
                        target_index=int(metadata["target_index"]),
                        matching_cost=float(metadata["matching_cost"]),
                        path_id=path_id,
                        sample_index=sample_index,
                    )
                )
            path_id += 1

    diagnostics = {
        "format": "topology_flow_graphlet_training_states_v1",
        "training_state_source": "continuous_edge_probability_flow_matching",
        "rewiring_used_for_training_states": False,
        "num_graphs": len(graphs),
        "num_paths": len(graphs) * cfg.paths_per_graph,
        "num_examples": len(examples),
        "samples_per_graph": cfg.samples_per_graph,
        "paths_per_graph": cfg.paths_per_graph,
        "path": cfg.path,
        "time_sampling": cfg.time_sampling,
        "indexed_degree_alignment": True,
        "mean_changed_pairs": float(
            np.mean([row["changed_pairs"] for row in endpoint_reports])
        )
        if endpoint_reports
        else 0.0,
        "mean_changed_pair_fraction": float(
            np.mean([row["changed_pair_fraction"] for row in endpoint_reports])
        )
        if endpoint_reports
        else 0.0,
        "max_target_degree_tangent_residual": float(
            max(
                (row["max_target_degree_tangent_residual"] for row in endpoint_reports),
                default=0.0,
            )
        ),
        "source_modes": sorted({str(row["source_mode"]) for row in endpoint_reports}),
        "base_generators": sorted(
            {str(row["base_generator"]) for row in endpoint_reports}
        ),
    }
    return examples, diagnostics


class TopologyFlowGraphletIterableDataset(torch.utils.data.IterableDataset):
    """Resample time points and degree-class alignments every epoch."""

    def __init__(
        self,
        graphs: Sequence[nx.Graph | TopologyTrainingPair],
        *,
        flow_config: dict[str, Any] | None = None,
        source_config: dict[str, Any] | None = None,
        graphlet_basis: TopologyGraphletBasis,
        graphlet_logit_epsilon: float = 1.0e-5,
        seed: int = 0,
        shuffle_graphs: bool = True,
    ) -> None:
        super().__init__()
        self.graphs = tuple(graphs)
        self.flow_config = dict(flow_config or {})
        self.source_config = dict(source_config or {})
        self.graphlet_basis = graphlet_basis
        self.graphlet_logit_epsilon = float(graphlet_logit_epsilon)
        self.seed = int(seed)
        self.shuffle_graphs = bool(shuffle_graphs)
        self.epoch = 0
        self.last_diagnostics: list[dict[str, Any]] = []

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    @property
    def estimated_examples(self) -> int:
        cfg = FlowMatchingConfig.from_dict(self.flow_config)
        return len(self.graphs) * cfg.samples_per_graph * cfg.paths_per_graph

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
            examples, diagnostics = build_flow_graphlet_examples(
                [self.graphs[int(graph_index)]],
                flow_config=self.flow_config,
                source_config=self.source_config,
                graphlet_basis=self.graphlet_basis,
                graphlet_logit_epsilon=self.graphlet_logit_epsilon,
                seed=(
                    self.seed
                    + 1_000_003 * self.epoch
                    + 10_007 * int(graph_index)
                    + position
                ),
            )
            if worker is None:
                self.last_diagnostics.append(diagnostics)
            yield from examples

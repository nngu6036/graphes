from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import networkx as nx
import numpy as np
import torch

from grapher.rewiring_mlp.attributed.data import (
    GraphCategoryVocabulary,
    GraphletBasis,
    aligned_havel_hakimi_source,
    graph_to_categorical_arrays,
)
from grapher.rewiring_mlp.attributed.graphlet_diffusion import (
    attributed_graphlet_clr_to_simplex,
    attributed_graphlet_simplex_to_clr,
    extract_attributed_graphlet_simplex,
)
from grapher.rewiring_mlp.attributed.spectral import (
    attributed_laplacian_spectra,
    attributed_spectral_distance,
    attributed_spectral_scales,
    normalize_attributed_graph,
)
from grapher.rewiring_mlp.generic.summary_diffusion import (
    SummaryDiffusionConfig,
    sample_graphlet_clr_bridge_marginal,
    sample_spectral_bridge_marginal,
)
from grapher.rewiring_mlp.molecular.constraints import bond_order
from grapher.rewiring_mlp.molecular.typed_invariants import (
    extract_typed_invariant,
    typed_invariant_matches_graph,
)


@dataclass(frozen=True)
class AttributedTrainingPair:
    source_graph: nx.Graph
    target_graph: nx.Graph
    base_generator: str = "completed_base_output"
    source_index: int = -1
    target_index: int = -1
    matching_cost: float = 0.0


@dataclass
class AttributedSpectralExample:
    """One continuous summary-diffusion sample.

    ``conditioning_graph`` is the fixed typed source graph. Intermediate
    diffusion states live only in continuous spectrum/CLR space and are not
    produced by rewiring or assumed to correspond to a realizable molecule.
    """

    conditioning_graph: nx.Graph
    time: float
    current_spectra: np.ndarray
    source_spectra: np.ndarray
    clean_spectra_target: np.ndarray
    current_graphlet_probabilities: np.ndarray
    source_graphlet_probabilities: np.ndarray
    clean_graphlet_probabilities_target: np.ndarray
    current_graphlet_logits: np.ndarray
    source_graphlet_logits: np.ndarray
    clean_graphlet_logits_target: np.ndarray
    graphlet_coordinate_mask: np.ndarray
    base_generator: str = "target_typed_constructor"
    source_index: int = -1
    target_index: int = -1
    matching_cost: float = 0.0
    path_id: int = -1
    sample_id: int = -1


@dataclass
class AttributedSpectralBatch:
    source_node_labels: torch.Tensor
    source_edge_labels: torch.Tensor
    source_edge_weights: torch.Tensor
    node_mask: torch.Tensor
    pair_mask: torch.Tensor
    degrees: torch.Tensor
    typed_degrees: torch.Tensor
    graph_size: torch.Tensor
    time: torch.Tensor
    current_spectra: torch.Tensor
    source_spectra: torch.Tensor
    clean_spectra_target: torch.Tensor
    spectrum_mask: torch.Tensor
    current_graphlet_probabilities: torch.Tensor
    source_graphlet_probabilities: torch.Tensor
    clean_graphlet_probabilities_target: torch.Tensor
    current_graphlet_logits: torch.Tensor
    source_graphlet_logits: torch.Tensor
    clean_graphlet_logits_target: torch.Tensor
    graphlet_coordinate_mask: torch.Tensor

    def to(self, device: torch.device | str) -> "AttributedSpectralBatch":
        return AttributedSpectralBatch(
            **{
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in self.__dict__.items()
            }
        )


def _typed_degree_array(
    edge_labels: np.ndarray,
    *,
    num_present_edge_categories: int,
) -> np.ndarray:
    return np.stack(
        [
            (edge_labels == category).sum(axis=1)
            for category in range(1, num_present_edge_categories + 1)
        ],
        axis=-1,
    ).astype(np.float32)


def collate_attributed_spectral_examples(
    examples: Sequence[AttributedSpectralExample],
    vocabulary: GraphCategoryVocabulary,
) -> AttributedSpectralBatch:
    if not examples:
        raise ValueError("Cannot collate an empty attributed spectral batch.")
    max_nodes = max(example.conditioning_graph.number_of_nodes() for example in examples)
    batch_size = len(examples)
    graphlet_widths = {
        int(np.asarray(example.current_graphlet_logits).size) for example in examples
    }
    if len(graphlet_widths) != 1:
        raise ValueError("Attributed graphlet examples must share a fixed width.")
    graphlet_width = next(iter(graphlet_widths))

    node_labels = np.zeros((batch_size, max_nodes), dtype=np.int64)
    edge_labels = np.zeros((batch_size, max_nodes, max_nodes), dtype=np.int64)
    edge_weights = np.zeros((batch_size, max_nodes, max_nodes), dtype=np.float32)
    node_mask = np.zeros((batch_size, max_nodes), dtype=np.bool_)
    pair_mask = np.zeros((batch_size, max_nodes, max_nodes), dtype=np.bool_)
    degrees = np.zeros((batch_size, max_nodes), dtype=np.float32)
    typed_degrees = np.zeros(
        (batch_size, max_nodes, vocabulary.num_edge_categories - 1),
        dtype=np.float32,
    )
    graph_size = np.zeros(batch_size, dtype=np.float32)
    time = np.zeros(batch_size, dtype=np.float32)
    current_spectra = np.zeros((batch_size, 2, max_nodes), dtype=np.float32)
    source_spectra = np.zeros((batch_size, 2, max_nodes), dtype=np.float32)
    clean_spectra = np.zeros((batch_size, 2, max_nodes), dtype=np.float32)
    spectrum_mask = np.zeros((batch_size, max_nodes), dtype=np.bool_)

    current_prob = np.zeros((batch_size, graphlet_width), dtype=np.float32)
    source_prob = np.zeros((batch_size, graphlet_width), dtype=np.float32)
    clean_prob = np.zeros((batch_size, graphlet_width), dtype=np.float32)
    current_logits = np.zeros((batch_size, graphlet_width), dtype=np.float32)
    source_logits = np.zeros((batch_size, graphlet_width), dtype=np.float32)
    clean_logits = np.zeros((batch_size, graphlet_width), dtype=np.float32)
    graphlet_mask = np.zeros((batch_size, graphlet_width), dtype=np.bool_)

    for index, example in enumerate(examples):
        graph = normalize_attributed_graph(example.conditioning_graph)
        n = graph.number_of_nodes()
        nodes, edges = graph_to_categorical_arrays(graph, vocabulary)
        node_labels[index, :n] = nodes
        edge_labels[index, :n, :n] = edges
        for u in range(n):
            for v in range(u + 1, n):
                category = int(edges[u, v])
                if category <= 0:
                    continue
                weight = float(bond_order(int(vocabulary.edge_value(category))))
                edge_weights[index, u, v] = edge_weights[index, v, u] = weight
        node_mask[index, :n] = True
        pair_mask[index, :n, :n] = True
        np.fill_diagonal(pair_mask[index], False)
        degree_values = np.asarray([graph.degree(node) for node in range(n)], dtype=np.float32)
        degrees[index, :n] = degree_values / max(n - 1, 1)
        typed_degrees[index, :n] = _typed_degree_array(
            edges,
            num_present_edge_categories=vocabulary.num_edge_categories - 1,
        )
        graph_size[index] = float(n)
        time[index] = float(np.clip(example.time, 0.0, 1.0))

        for values, destination, name in (
            (example.current_spectra, current_spectra, "current"),
            (example.source_spectra, source_spectra, "source"),
            (example.clean_spectra_target, clean_spectra, "clean"),
        ):
            array = np.asarray(values, dtype=np.float32)
            if array.shape != (2, n):
                raise ValueError(
                    f"Attributed {name} spectra must have shape (2, {n}); got {array.shape}."
                )
            destination[index, :, :n] = array
        spectrum_mask[index, :n] = True

        arrays = [
            np.asarray(example.current_graphlet_probabilities, dtype=np.float32).reshape(-1),
            np.asarray(example.source_graphlet_probabilities, dtype=np.float32).reshape(-1),
            np.asarray(example.clean_graphlet_probabilities_target, dtype=np.float32).reshape(-1),
            np.asarray(example.current_graphlet_logits, dtype=np.float32).reshape(-1),
            np.asarray(example.source_graphlet_logits, dtype=np.float32).reshape(-1),
            np.asarray(example.clean_graphlet_logits_target, dtype=np.float32).reshape(-1),
            np.asarray(example.graphlet_coordinate_mask, dtype=np.bool_).reshape(-1),
        ]
        if any(value.size != graphlet_width for value in arrays):
            raise ValueError("Attributed graphlet diffusion width mismatch during collation.")
        current_prob[index], source_prob[index], clean_prob[index] = arrays[:3]
        current_logits[index], source_logits[index], clean_logits[index] = arrays[3:6]
        graphlet_mask[index] = arrays[6]

    return AttributedSpectralBatch(
        source_node_labels=torch.from_numpy(node_labels),
        source_edge_labels=torch.from_numpy(edge_labels),
        source_edge_weights=torch.from_numpy(edge_weights),
        node_mask=torch.from_numpy(node_mask),
        pair_mask=torch.from_numpy(pair_mask),
        degrees=torch.from_numpy(degrees),
        typed_degrees=torch.from_numpy(typed_degrees),
        graph_size=torch.from_numpy(graph_size),
        time=torch.from_numpy(time),
        current_spectra=torch.from_numpy(current_spectra),
        source_spectra=torch.from_numpy(source_spectra),
        clean_spectra_target=torch.from_numpy(clean_spectra),
        spectrum_mask=torch.from_numpy(spectrum_mask),
        current_graphlet_probabilities=torch.from_numpy(current_prob),
        source_graphlet_probabilities=torch.from_numpy(source_prob),
        clean_graphlet_probabilities_target=torch.from_numpy(clean_prob),
        current_graphlet_logits=torch.from_numpy(current_logits),
        source_graphlet_logits=torch.from_numpy(source_logits),
        clean_graphlet_logits_target=torch.from_numpy(clean_logits),
        graphlet_coordinate_mask=torch.from_numpy(graphlet_mask),
    )


def _joint_relabel(
    source: nx.Graph,
    target: nx.Graph,
    *,
    rng: np.random.Generator,
) -> tuple[nx.Graph, nx.Graph]:
    n = source.number_of_nodes()
    permutation = rng.permutation(n)
    mapping = {old: int(permutation[old]) for old in range(n)}
    return (
        nx.relabel_nodes(source, mapping, copy=True),
        nx.relabel_nodes(target, mapping, copy=True),
    )


def resolve_attributed_diffusion_endpoints(
    raw_item: nx.Graph | AttributedTrainingPair,
    *,
    vocabulary: GraphCategoryVocabulary,
    source_config: dict[str, Any] | None,
    rng: np.random.Generator,
) -> tuple[nx.Graph, nx.Graph, dict[str, Any]]:
    cfg = dict(source_config or {})
    if isinstance(raw_item, AttributedTrainingPair):
        source = normalize_attributed_graph(raw_item.source_graph)
        target = normalize_attributed_graph(raw_item.target_graph)
        metadata = {
            "source_mode": "completed_base_output",
            "base_generator": raw_item.base_generator,
            "source_index": raw_item.source_index,
            "target_index": raw_item.target_index,
            "matching_cost": raw_item.matching_cost,
        }
    else:
        target = normalize_attributed_graph(raw_item)
        source, target = aligned_havel_hakimi_source(
            target,
            ensure_connected=bool(cfg.get("ensure_connected_source", True)),
            max_repair_trials=int(cfg.get("max_repair_trials", 50000)),
            rng=rng,
            vocabulary=vocabulary,
            typed_constructor_config=dict(cfg.get("typed_constructor", {}) or {}),
        )
        metadata = {
            "source_mode": "target_typed_constructor",
            "base_generator": "target_typed_constructor",
            "source_index": -1,
            "target_index": -1,
            "matching_cost": 0.0,
        }
    source = normalize_attributed_graph(source)
    target = normalize_attributed_graph(target)
    if source.number_of_nodes() != target.number_of_nodes():
        raise ValueError("Attributed source and clean target must have the same size.")
    if source.number_of_nodes() > 1 and not nx.is_connected(source):
        raise ValueError("Attributed diffusion source must be connected.")
    if target.number_of_nodes() > 1 and not nx.is_connected(target):
        raise ValueError("Attributed diffusion target must be connected.")
    invariant = extract_typed_invariant(
        target,
        edge_types=vocabulary.edge_values,
        node_attribute=str(vocabulary.node_attribute),
        edge_attribute=str(vocabulary.edge_attribute),
    )
    if not typed_invariant_matches_graph(source, invariant):
        raise ValueError(
            "Attributed diffusion source and clean target must have identical indexed typed degrees."
        )
    if bool(cfg.get("shared_relabel_augmentation", False)):
        source, target = _joint_relabel(source, target, rng=rng)
    return source, target, metadata


def build_attributed_spectral_diffusion_examples(
    graphs: Sequence[nx.Graph | AttributedTrainingPair],
    *,
    vocabulary: GraphCategoryVocabulary,
    graphlet_basis: GraphletBasis,
    diffusion_config: dict[str, Any] | None = None,
    source_config: dict[str, Any] | None = None,
    spectral_config: dict[str, Any] | None = None,
    graphlet_logit_epsilon: float = 1.0e-4,
    seed: int = 0,
) -> tuple[list[AttributedSpectralExample], dict[str, Any]]:
    diff_values = dict(diffusion_config or {})
    diff_cfg = SummaryDiffusionConfig.from_dict(diff_values)
    spec_cfg = dict(spectral_config or {})
    rng = np.random.default_rng(int(seed))
    samples_per_graph = max(int(diff_values.get("samples_per_graph", 8)), 1)
    paths_per_graph = max(int(diff_values.get("paths_per_graph", 1)), 1)
    edge_attribute = str(vocabulary.edge_attribute or "bond_type")

    examples: list[AttributedSpectralExample] = []
    reports: list[dict[str, Any]] = []
    spectral_noise: list[float] = []
    graphlet_noise: list[float] = []
    path_id = 0
    for raw_item in graphs:
        source, target, metadata = resolve_attributed_diffusion_endpoints(
            raw_item,
            vocabulary=vocabulary,
            source_config=source_config,
            rng=rng,
        )
        source_spectra = attributed_laplacian_spectra(source, edge_attribute=edge_attribute)
        clean_spectra = attributed_laplacian_spectra(target, edge_attribute=edge_attribute)
        scales = attributed_spectral_scales(
            source,
            mode=str(spec_cfg.get("normalization", "mean_degree")),
            edge_attribute=edge_attribute,
        )
        source_prob, source_mask, _ = extract_attributed_graphlet_simplex(
            source, graphlet_basis=graphlet_basis
        )
        clean_prob, clean_mask, _ = extract_attributed_graphlet_simplex(
            target, graphlet_basis=graphlet_basis
        )
        if not np.array_equal(source_mask, clean_mask):
            raise AssertionError("Equal-size attributed endpoints must share graphlet masks.")
        source_logits = attributed_graphlet_simplex_to_clr(
            source_prob,
            graphlet_basis=graphlet_basis,
            epsilon=graphlet_logit_epsilon,
            coordinate_mask=source_mask,
        )
        clean_logits = attributed_graphlet_simplex_to_clr(
            clean_prob,
            graphlet_basis=graphlet_basis,
            epsilon=graphlet_logit_epsilon,
            coordinate_mask=clean_mask,
        )
        endpoint_distance, channel_distance = attributed_spectral_distance(
            source_spectra,
            clean_spectra,
            scales=scales,
            metric=str(spec_cfg.get("distance", "rmse")),
            channel_weights=spec_cfg.get("channel_weights", [1.0, 1.0]),
            low_frequency_weight=float(spec_cfg.get("low_frequency_weight", 1.0)),
            low_frequency_cutoff=int(spec_cfg.get("low_frequency_cutoff", 0)),
        )
        reports.append(
            {
                **metadata,
                "num_nodes": source.number_of_nodes(),
                "spectral_endpoint_distance": endpoint_distance,
                "topology_endpoint_distance": float(channel_distance[0]),
                "bond_endpoint_distance": float(channel_distance[1]),
            }
        )

        for _path in range(paths_per_graph):
            progresses = diff_cfg.sample_progresses(samples_per_graph, rng=rng)
            for sample_id, progress in enumerate(progresses):
                current_channels: list[np.ndarray] = []
                for channel in range(2):
                    current, diag = sample_spectral_bridge_marginal(
                        source_spectra[channel],
                        clean_spectra[channel],
                        progress=float(progress),
                        sigma=diff_cfg.spectral_sigma,
                        scale=float(scales[channel]),
                        preserve_trace=diff_cfg.preserve_spectral_trace,
                        fix_lambda1=diff_cfg.fix_spectral_lambda1,
                        schedule=diff_cfg,
                        rng=rng,
                    )
                    current_channels.append(current)
                    spectral_noise.append(float(diag["noise_rms"]))
                current_logits, graph_diag = sample_graphlet_clr_bridge_marginal(
                    source_logits,
                    clean_logits,
                    progress=float(progress),
                    sigma=diff_cfg.graphlet_sigma,
                    graphlet_basis=graphlet_basis,  # duck-typed simplex API
                    coordinate_mask=source_mask,
                    schedule=diff_cfg,
                    rng=rng,
                )
                current_prob = attributed_graphlet_clr_to_simplex(
                    current_logits,
                    graphlet_basis=graphlet_basis,
                    coordinate_mask=source_mask,
                )
                graphlet_noise.append(float(graph_diag["noise_rms"]))
                examples.append(
                    AttributedSpectralExample(
                        conditioning_graph=source.copy(),
                        time=float(progress),
                        current_spectra=np.stack(current_channels).astype(np.float32),
                        source_spectra=source_spectra.astype(np.float32),
                        clean_spectra_target=clean_spectra.astype(np.float32),
                        current_graphlet_probabilities=current_prob.astype(np.float32),
                        source_graphlet_probabilities=source_prob.astype(np.float32),
                        clean_graphlet_probabilities_target=clean_prob.astype(np.float32),
                        current_graphlet_logits=current_logits.astype(np.float32),
                        source_graphlet_logits=source_logits.astype(np.float32),
                        clean_graphlet_logits_target=clean_logits.astype(np.float32),
                        graphlet_coordinate_mask=source_mask.astype(np.bool_),
                        base_generator=str(metadata["base_generator"]),
                        source_index=int(metadata["source_index"]),
                        target_index=int(metadata["target_index"]),
                        matching_cost=float(metadata["matching_cost"]),
                        path_id=path_id,
                        sample_id=sample_id,
                    )
                )
            path_id += 1

    diagnostics = {
        "format": "attributed_summary_diffusion_training_states_v1",
        "training_state_source": "continuous_summary_diffusion",
        "rewiring_used_for_training_states": False,
        "spectral_channels": ["topology", "bond_weighted"],
        "num_graphs": len(graphs),
        "num_examples": len(examples),
        "samples_per_graph": samples_per_graph,
        "paths_per_graph": paths_per_graph,
        "mean_spectral_noise_rms": float(np.mean(spectral_noise)) if spectral_noise else 0.0,
        "mean_graphlet_noise_rms": float(np.mean(graphlet_noise)) if graphlet_noise else 0.0,
        "mean_endpoint_spectral_distance": float(
            np.mean([row["spectral_endpoint_distance"] for row in reports])
        ) if reports else 0.0,
    }
    return examples, diagnostics


class AttributedSpectralDiffusionIterableDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        graphs: Sequence[nx.Graph | AttributedTrainingPair],
        *,
        vocabulary: GraphCategoryVocabulary,
        graphlet_basis: GraphletBasis,
        diffusion_config: dict[str, Any] | None = None,
        source_config: dict[str, Any] | None = None,
        spectral_config: dict[str, Any] | None = None,
        graphlet_logit_epsilon: float = 1.0e-4,
        seed: int = 0,
        shuffle_graphs: bool = True,
    ) -> None:
        super().__init__()
        self.graphs = tuple(graphs)
        self.vocabulary = vocabulary
        self.graphlet_basis = graphlet_basis
        self.diffusion_config = dict(diffusion_config or {})
        self.source_config = dict(source_config or {})
        self.spectral_config = dict(spectral_config or {})
        self.graphlet_logit_epsilon = float(graphlet_logit_epsilon)
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
            * max(int(self.diffusion_config.get("samples_per_graph", 8)), 1)
            * max(int(self.diffusion_config.get("paths_per_graph", 1)), 1)
        )

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        worker_id = worker.id if worker is not None else 0
        worker_count = worker.num_workers if worker is not None else 1
        indices = np.arange(len(self.graphs), dtype=np.int64)
        rng = np.random.default_rng(self.seed + 1_000_003 * self.epoch)
        if self.shuffle_graphs:
            rng.shuffle(indices)
        indices = indices[worker_id::worker_count]
        if worker is None:
            self.last_diagnostics = []
        for position, graph_index in enumerate(indices):
            examples, diagnostics = build_attributed_spectral_diffusion_examples(
                [self.graphs[int(graph_index)]],
                vocabulary=self.vocabulary,
                graphlet_basis=self.graphlet_basis,
                diffusion_config=self.diffusion_config,
                source_config=self.source_config,
                spectral_config=self.spectral_config,
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


__all__ = [
    "AttributedSpectralBatch",
    "AttributedSpectralDiffusionIterableDataset",
    "AttributedSpectralExample",
    "AttributedTrainingPair",
    "build_attributed_spectral_diffusion_examples",
    "collate_attributed_spectral_examples",
    "resolve_attributed_diffusion_endpoints",
]

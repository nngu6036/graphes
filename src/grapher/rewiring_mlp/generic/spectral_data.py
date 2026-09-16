from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import networkx as nx
import numpy as np
import torch

from grapher.rewiring_mlp.core.rewiring import Action, make_action
from grapher.rewiring_mlp.generic.basis import TopologyGraphletBasis
from grapher.rewiring_mlp.generic.clustering import (
    clustering_histogram_bins, extract_clustering_histogram, validate_clustering_histogram,
)
from grapher.rewiring_mlp.generic.induced_graphlets import (
    InducedGraphletSpec, extract_histogram as extract_induced_histogram,
    validate_histogram as validate_induced_histogram,
)
from grapher.rewiring_mlp.generic.cycle_graphlets import (
    cycle_graphlet_k, extract_cycle_graphlet_histogram, validate_cycle_graphlet_histogram,
)
from grapher.rewiring_mlp.generic.orbit import (
    extract_orbit_summary, orbit_summary_width, validate_orbit_summary,
)
from grapher.rewiring_mlp.generic.data import (
    TopologyTrainingPair,
    _construct_source_from_degree_sequence,
    _randomly_relabel_topology_graph,
    normalize_topology_graph,
)
from grapher.rewiring_mlp.generic.graphlet_diffusion import (
    extract_topology_graphlet_simplex,
    graphlet_clr_to_simplex,
    graphlet_simplex_to_clr,
)
from grapher.rewiring_mlp.generic.rewiring import (
    propose_valid_topology_swaps,
    topology_state_key,
)
from grapher.rewiring_mlp.generic.summary_diffusion import (
    SummaryDiffusionConfig,
    sample_graphlet_clr_bridge_marginal,
    sample_heat_kernel_bridge_marginal,
    sample_eigenspace_projector_bridge_marginal,
    sample_eigenspace_histogram_bridge_marginal,
    sample_spectral_bridge_marginal,
)
from grapher.rewiring_mlp.generic.heat_kernel import (
    heat_kernel_distance,
    heat_kernel_stack,
    validate_heat_times,
)
from grapher.rewiring_mlp.generic.eigenspace import (
    effective_projector_rank,
    eigenspace_projector_distance,
    laplacian_eigenspace_projector,
    projector_node_signatures,
)
from grapher.rewiring_mlp.generic.spectral_distance_histogram import (
    SpectralDistanceHistogramSpec,
    extract_degree_conditioned_spectral_histogram,
    spectral_histogram_wasserstein,
    validate_histogram as validate_spectral_distance_histogram,
)
from grapher.rewiring_mlp.generic.spectral import (
    degree_spectral_moments,
    laplacian_eigenvalues,
    spectral_distance,
    spectral_scale,
    spectrum_moments,
)


@dataclass
class TopologySpectralExample:
    # ``current_graph`` is the fixed source/conditioning graph for diffusion
    # training.  During generation it is likewise the initial HH/base graph;
    # the continuous/current summary state is carried separately below.
    current_graph: nx.Graph
    time: float
    clean_spectrum_target: np.ndarray
    current_heat_kernel: np.ndarray | None = None
    source_heat_kernel: np.ndarray | None = None
    clean_heat_kernel_target: np.ndarray | None = None
    current_projector: np.ndarray | None = None
    source_projector: np.ndarray | None = None
    clean_projector_target: np.ndarray | None = None
    current_eigenspace_histogram: np.ndarray | None = None
    source_eigenspace_histogram: np.ndarray | None = None
    clean_eigenspace_histogram_target: np.ndarray | None = None
    eigenspace_histogram_block_mask: np.ndarray | None = None
    eigenspace_histogram_block_weights: np.ndarray | None = None
    # Optional clean structural summary target. For the minimal spectral debug
    # model this is the graph-average local clustering coefficient in [0, 1].
    # It is predicted as an auxiliary x0 target but is NOT itself diffused.
    clean_clustering_coefficient_target: float | None = None
    current_spectrum: np.ndarray | None = None
    source_spectrum: np.ndarray | None = None
    # Optional generic soft binary-edge bridge over {no-edge, edge}.
    current_edge_logits: np.ndarray | None = None
    source_edge_logits: np.ndarray | None = None
    clean_edge_logits_target: np.ndarray | None = None
    clean_edge_labels_target: np.ndarray | None = None
    # Optional graphlet-logit diffusion supervision. Each graphlet order is a
    # probability simplex over connected graphlet classes plus one disconnected
    # subset bin; CLR coordinates are the Euclidean diffusion variables.
    current_graphlet_probabilities: np.ndarray | None = None
    source_graphlet_probabilities: np.ndarray | None = None
    clean_graphlet_probabilities_target: np.ndarray | None = None
    current_graphlet_logits: np.ndarray | None = None
    source_graphlet_logits: np.ndarray | None = None
    clean_graphlet_logits_target: np.ndarray | None = None
    graphlet_coordinate_mask: np.ndarray | None = None
    base_generator: str = "target_degree_havel_hakimi"
    source_index: int = -1
    target_index: int = -1
    matching_cost: float = 0.0
    trajectory_id: int = -1
    step: int = -1
    teacher_actions: tuple[Action, ...] = ()
    teacher_distribution: np.ndarray | None = None
    teacher_selected_index: int = -1
    clean_clustering_histogram_target: np.ndarray | None = None
    # Optional evaluator-compatible mean per-node ORCA orbit vector (0--14).
    # Like clustering summaries, it is a clean x0 target and is not diffused.
    clean_orbit_summary_target: np.ndarray | None = None
    clean_cycle_graphlet_histogram_target: np.ndarray | None = None
    clean_induced_graphlet_histogram_target: np.ndarray | None = None


@dataclass
class TopologySpectralBatch:
    adjacency: torch.Tensor
    node_mask: torch.Tensor
    pair_mask: torch.Tensor
    degrees: torch.Tensor
    graph_size: torch.Tensor
    time: torch.Tensor
    current_spectrum: torch.Tensor
    source_spectrum: torch.Tensor
    clean_spectrum_target: torch.Tensor
    spectrum_mask: torch.Tensor
    current_heat_kernel: torch.Tensor | None = None
    source_heat_kernel: torch.Tensor | None = None
    clean_heat_kernel_target: torch.Tensor | None = None
    current_projector: torch.Tensor | None = None
    source_projector: torch.Tensor | None = None
    clean_projector_target: torch.Tensor | None = None
    current_eigenspace_histogram: torch.Tensor | None = None
    source_eigenspace_histogram: torch.Tensor | None = None
    clean_eigenspace_histogram_target: torch.Tensor | None = None
    eigenspace_histogram_block_mask: torch.Tensor | None = None
    eigenspace_histogram_block_weights: torch.Tensor | None = None
    current_edge_logits: torch.Tensor | None = None
    source_edge_logits: torch.Tensor | None = None
    clean_edge_logits_target: torch.Tensor | None = None
    clean_edge_labels_target: torch.Tensor | None = None
    clean_clustering_coefficient_target: torch.Tensor | None = None
    current_graphlet_probabilities: torch.Tensor | None = None
    source_graphlet_probabilities: torch.Tensor | None = None
    clean_graphlet_probabilities_target: torch.Tensor | None = None
    current_graphlet_logits: torch.Tensor | None = None
    source_graphlet_logits: torch.Tensor | None = None
    clean_graphlet_logits_target: torch.Tensor | None = None
    graphlet_coordinate_mask: torch.Tensor | None = None

    clean_clustering_histogram_target: torch.Tensor | None = None
    clean_orbit_summary_target: torch.Tensor | None = None
    clean_cycle_graphlet_histogram_target: torch.Tensor | None = None
    clean_induced_graphlet_histogram_target: torch.Tensor | None = None

    def to(self, device: torch.device | str) -> "TopologySpectralBatch":
        return TopologySpectralBatch(
            **{
                key: (value.to(device) if isinstance(value, torch.Tensor) else value)
                for key, value in self.__dict__.items()
            }
        )


def collate_spectral_examples(
    examples: Sequence[TopologySpectralExample],
) -> TopologySpectralBatch:
    """Pad graph states and variable-length eigenvalue targets for one batch."""

    if not examples:
        raise ValueError("Cannot collate an empty spectral batch.")
    max_nodes = max(example.current_graph.number_of_nodes() for example in examples)
    batch_size = len(examples)

    adjacency = np.zeros((batch_size, max_nodes, max_nodes), dtype=np.bool_)
    node_mask = np.zeros((batch_size, max_nodes), dtype=np.bool_)
    pair_mask = np.zeros((batch_size, max_nodes, max_nodes), dtype=np.bool_)
    degrees = np.zeros((batch_size, max_nodes), dtype=np.float32)
    graph_sizes = np.zeros(batch_size, dtype=np.float32)
    times = np.zeros(batch_size, dtype=np.float32)
    current_spectra = np.zeros((batch_size, max_nodes), dtype=np.float32)
    source_spectra = np.zeros((batch_size, max_nodes), dtype=np.float32)
    clean_spectra = np.zeros((batch_size, max_nodes), dtype=np.float32)
    spectrum_mask = np.zeros((batch_size, max_nodes), dtype=np.bool_)

    heat_widths = {
        int(np.asarray(example.current_heat_kernel).shape[-1])
        for example in examples
        if example.current_heat_kernel is not None
    }
    if len(heat_widths) > 1:
        raise ValueError("Heat-kernel examples in one batch must share one scale count.")
    heat_width = next(iter(heat_widths), 0)
    heat_enabled = heat_width > 0
    if heat_enabled and any(
        example.current_heat_kernel is None
        or example.source_heat_kernel is None
        or example.clean_heat_kernel_target is None
        for example in examples
    ):
        raise ValueError("Cannot mix examples with and without heat-kernel diffusion states.")
    current_heat_kernel = (
        np.zeros((batch_size, max_nodes, max_nodes, heat_width), dtype=np.float32)
        if heat_enabled else None
    )
    source_heat_kernel = (
        np.zeros((batch_size, max_nodes, max_nodes, heat_width), dtype=np.float32)
        if heat_enabled else None
    )
    clean_heat_kernel = (
        np.zeros((batch_size, max_nodes, max_nodes, heat_width), dtype=np.float32)
        if heat_enabled else None
    )

    projector_enabled = any(example.current_projector is not None for example in examples)
    if projector_enabled and any(
        example.current_projector is None
        or example.source_projector is None
        or example.clean_projector_target is None
        for example in examples
    ):
        raise ValueError("Cannot mix examples with and without eigenspace-projector diffusion states.")
    current_projector = (
        np.zeros((batch_size, max_nodes, max_nodes), dtype=np.float32)
        if projector_enabled else None
    )
    source_projector = (
        np.zeros((batch_size, max_nodes, max_nodes), dtype=np.float32)
        if projector_enabled else None
    )
    clean_projector = (
        np.zeros((batch_size, max_nodes, max_nodes), dtype=np.float32)
        if projector_enabled else None
    )

    eig_hist_widths = {
        int(np.asarray(example.current_eigenspace_histogram).size)
        for example in examples
        if example.current_eigenspace_histogram is not None
    }
    if len(eig_hist_widths) > 1:
        raise ValueError("Eigenspace-histogram examples in one batch must share one width.")
    eig_hist_width = next(iter(eig_hist_widths), 0)
    eig_hist_enabled = eig_hist_width > 0
    if eig_hist_enabled and any(
        example.current_eigenspace_histogram is None
        or example.source_eigenspace_histogram is None
        or example.clean_eigenspace_histogram_target is None
        or example.eigenspace_histogram_block_mask is None
        or example.eigenspace_histogram_block_weights is None
        for example in examples
    ):
        raise ValueError("Cannot mix examples with and without eigenspace-histogram diffusion states.")
    eig_hist_blocks = (
        int(np.asarray(examples[0].eigenspace_histogram_block_mask).size)
        if eig_hist_enabled else 0
    )
    current_eigenspace_histogram = (
        np.zeros((batch_size, eig_hist_width), dtype=np.float32) if eig_hist_enabled else None
    )
    source_eigenspace_histogram = (
        np.zeros((batch_size, eig_hist_width), dtype=np.float32) if eig_hist_enabled else None
    )
    clean_eigenspace_histogram = (
        np.zeros((batch_size, eig_hist_width), dtype=np.float32) if eig_hist_enabled else None
    )
    eigenspace_histogram_block_mask = (
        np.zeros((batch_size, eig_hist_blocks), dtype=np.bool_) if eig_hist_enabled else None
    )
    eigenspace_histogram_block_weights = (
        np.zeros((batch_size, eig_hist_blocks), dtype=np.float32) if eig_hist_enabled else None
    )

    edge_enabled = any(example.current_edge_logits is not None for example in examples)
    if edge_enabled and any(
        example.current_edge_logits is None or example.source_edge_logits is None
        or example.clean_edge_logits_target is None or example.clean_edge_labels_target is None
        for example in examples
    ):
        raise ValueError("Cannot mix examples with and without generic edge-diffusion states.")
    current_edge_logits = np.zeros((batch_size,max_nodes,max_nodes,2),dtype=np.float32) if edge_enabled else None
    source_edge_logits = np.zeros((batch_size,max_nodes,max_nodes,2),dtype=np.float32) if edge_enabled else None
    clean_edge_logits = np.zeros((batch_size,max_nodes,max_nodes,2),dtype=np.float32) if edge_enabled else None
    clean_edge_labels = np.zeros((batch_size,max_nodes,max_nodes),dtype=np.int64) if edge_enabled else None

    clustering_enabled = any(
        example.clean_clustering_coefficient_target is not None for example in examples
    )
    if clustering_enabled and any(
        example.clean_clustering_coefficient_target is None for example in examples
    ):
        raise ValueError(
            "Cannot mix spectral examples with and without clean clustering targets."
        )
    clean_clustering = (
        np.zeros(batch_size, dtype=np.float32) if clustering_enabled else None
    )

    histogram_targets = [example.clean_clustering_histogram_target for example in examples]
    clean_clustering_histogram = None
    if any(value is not None for value in histogram_targets):
        if any(value is None for value in histogram_targets):
            raise ValueError("Cannot mix examples with and without clean clustering histogram targets.")
        histograms = [validate_clustering_histogram(value) for value in histogram_targets]
        if len({value.size for value in histograms}) != 1:
            raise ValueError("Clustering histogram targets in a batch must share one bin count.")
        clean_clustering_histogram = np.stack(histograms).astype(np.float32)

    orbit_targets = [example.clean_orbit_summary_target for example in examples]
    clean_orbit_summary = None
    if any(value is not None for value in orbit_targets):
        if any(value is None for value in orbit_targets):
            raise ValueError("Cannot mix examples with and without clean orbit-summary targets.")
        orbit_vectors = [validate_orbit_summary(value) for value in orbit_targets]
        clean_orbit_summary = np.stack(orbit_vectors).astype(np.float32)

    cycle_targets = [example.clean_cycle_graphlet_histogram_target for example in examples]
    clean_cycle_histogram = None
    if any(value is not None for value in cycle_targets):
        if any(value is None for value in cycle_targets):
            raise ValueError("Cannot mix examples with and without cycle graphlet histogram targets.")
        clean_cycle_histogram = np.stack([
            validate_cycle_graphlet_histogram(value) for value in cycle_targets
        ]).astype(np.float32)

    induced_targets = [example.clean_induced_graphlet_histogram_target for example in examples]
    clean_induced_histogram = None
    if any(value is not None for value in induced_targets):
        if any(value is None for value in induced_targets):
            raise ValueError("Cannot mix examples with and without induced graphlet targets.")
        arrays=[np.asarray(value,dtype=np.float64).reshape(-1) for value in induced_targets]
        if len({arr.size for arr in arrays}) != 1 or any((not np.isfinite(arr).all()) or np.any(arr < -1e-7) for arr in arrays):
            raise ValueError("Induced graphlet targets must share one finite nonnegative width.")
        clean_induced_histogram = np.stack(arrays).astype(np.float32)

    graphlet_widths = {
        int(np.asarray(example.current_graphlet_logits).size)
        for example in examples
        if example.current_graphlet_logits is not None
    }
    if len(graphlet_widths) > 1:
        raise ValueError("Graphlet-logit examples in one batch must share a fixed width.")
    graphlet_width = next(iter(graphlet_widths), 0)
    graphlet_enabled = graphlet_width > 0
    if graphlet_enabled and any(example.current_graphlet_logits is None for example in examples):
        raise ValueError("Cannot mix spectral-only and spectral+graphlet examples in one batch.")
    current_graphlet_probabilities = (
        np.zeros((batch_size, graphlet_width), dtype=np.float32)
        if graphlet_enabled else None
    )
    source_graphlet_probabilities = (
        np.zeros((batch_size, graphlet_width), dtype=np.float32)
        if graphlet_enabled else None
    )
    clean_graphlet_probabilities = (
        np.zeros((batch_size, graphlet_width), dtype=np.float32)
        if graphlet_enabled else None
    )
    current_graphlet_logits = (
        np.zeros((batch_size, graphlet_width), dtype=np.float32)
        if graphlet_enabled else None
    )
    source_graphlet_logits = (
        np.zeros((batch_size, graphlet_width), dtype=np.float32)
        if graphlet_enabled else None
    )
    clean_graphlet_logits = (
        np.zeros((batch_size, graphlet_width), dtype=np.float32)
        if graphlet_enabled else None
    )
    graphlet_coordinate_mask = (
        np.zeros((batch_size, graphlet_width), dtype=np.bool_)
        if graphlet_enabled else None
    )

    for index, example in enumerate(examples):
        graph = normalize_topology_graph(example.current_graph)
        n = graph.number_of_nodes()
        matrix = nx.to_numpy_array(
            graph,
            nodelist=list(range(n)),
            dtype=np.float32,
        )
        adjacency[index, :n, :n] = matrix > 0.0
        node_mask[index, :n] = True
        pair_mask[index, :n, :n] = True
        np.fill_diagonal(pair_mask[index], False)
        degrees[index, :n] = np.asarray(
            [float(graph.degree(node)) / max(n - 1, 1) for node in range(n)],
            dtype=np.float32,
        )
        graph_sizes[index] = float(n)
        times[index] = float(np.clip(example.time, 0.0, 1.0))
        # Most diffusion examples already carry both current and source
        # spectra.  Avoid recomputing the fixed conditioning-graph spectrum in
        # every collation call; fall back to an eigensolve only for legacy
        # examples that omit one of those fields.
        graph_spectrum = None
        if example.current_spectrum is None or example.source_spectrum is None:
            graph_spectrum = laplacian_eigenvalues(graph).astype(np.float32)
        current = (
            graph_spectrum
            if example.current_spectrum is None
            else np.asarray(example.current_spectrum, dtype=np.float32).reshape(-1)
        )
        source_spectrum = (
            graph_spectrum
            if example.source_spectrum is None
            else np.asarray(example.source_spectrum, dtype=np.float32).reshape(-1)
        )
        target = np.asarray(
            example.clean_spectrum_target,
            dtype=np.float32,
        ).reshape(-1)
        if current.size != n or source_spectrum.size != n or target.size != n:
            raise ValueError(
                "Spectral examples must contain exactly one eigenvalue per node: "
                f"current={current.size}, source={source_spectrum.size}, target={target.size}, n={n}."
            )
        current_spectra[index, :n] = current
        source_spectra[index, :n] = source_spectrum
        clean_spectra[index, :n] = target
        spectrum_mask[index, :n] = True
        if heat_enabled:
            assert current_heat_kernel is not None
            assert source_heat_kernel is not None
            assert clean_heat_kernel is not None
            heat_arrays = [
                np.asarray(example.current_heat_kernel, dtype=np.float32),
                np.asarray(example.source_heat_kernel, dtype=np.float32),
                np.asarray(example.clean_heat_kernel_target, dtype=np.float32),
            ]
            if any(array.shape != (n, n, heat_width) for array in heat_arrays):
                raise ValueError(
                    "Heat-kernel bridge tensors must have shape [n,n,num_scales]."
                )
            current_heat_kernel[index, :n, :n] = heat_arrays[0]
            source_heat_kernel[index, :n, :n] = heat_arrays[1]
            clean_heat_kernel[index, :n, :n] = heat_arrays[2]
        if projector_enabled:
            assert current_projector is not None
            assert source_projector is not None
            assert clean_projector is not None
            projector_arrays = [
                np.asarray(example.current_projector, dtype=np.float32),
                np.asarray(example.source_projector, dtype=np.float32),
                np.asarray(example.clean_projector_target, dtype=np.float32),
            ]
            if any(array.shape != (n, n) for array in projector_arrays):
                raise ValueError("Eigenspace-projector bridge tensors must have shape [n,n].")
            current_projector[index, :n, :n] = projector_arrays[0]
            source_projector[index, :n, :n] = projector_arrays[1]
            clean_projector[index, :n, :n] = projector_arrays[2]
        if eig_hist_enabled:
            assert current_eigenspace_histogram is not None
            assert source_eigenspace_histogram is not None
            assert clean_eigenspace_histogram is not None
            assert eigenspace_histogram_block_mask is not None
            assert eigenspace_histogram_block_weights is not None
            arrays = [
                np.asarray(example.current_eigenspace_histogram, dtype=np.float32).reshape(-1),
                np.asarray(example.source_eigenspace_histogram, dtype=np.float32).reshape(-1),
                np.asarray(example.clean_eigenspace_histogram_target, dtype=np.float32).reshape(-1),
            ]
            mask_value = np.asarray(example.eigenspace_histogram_block_mask, dtype=np.bool_).reshape(-1)
            weights_value = np.asarray(example.eigenspace_histogram_block_weights, dtype=np.float32).reshape(-1)
            if any(array.size != eig_hist_width for array in arrays):
                raise ValueError("Eigenspace-histogram width mismatch during collation.")
            if mask_value.size != eig_hist_blocks or weights_value.size != eig_hist_blocks:
                raise ValueError("Eigenspace-histogram block metadata width mismatch during collation.")
            current_eigenspace_histogram[index] = arrays[0]
            source_eigenspace_histogram[index] = arrays[1]
            clean_eigenspace_histogram[index] = arrays[2]
            eigenspace_histogram_block_mask[index] = mask_value
            eigenspace_histogram_block_weights[index] = weights_value
        if edge_enabled:
            arrays = [
                np.asarray(example.current_edge_logits,dtype=np.float32),
                np.asarray(example.source_edge_logits,dtype=np.float32),
                np.asarray(example.clean_edge_logits_target,dtype=np.float32),
            ]
            labels = np.asarray(example.clean_edge_labels_target,dtype=np.int64)
            if any(a.shape != (n,n,2) for a in arrays) or labels.shape != (n,n):
                raise ValueError("Generic edge bridge tensors must have shape [n,n,2] and labels [n,n].")
            current_edge_logits[index,:n,:n]=arrays[0]
            source_edge_logits[index,:n,:n]=arrays[1]
            clean_edge_logits[index,:n,:n]=arrays[2]
            clean_edge_labels[index,:n,:n]=labels
        if clean_clustering is not None:
            value = float(example.clean_clustering_coefficient_target)
            if not np.isfinite(value) or value < -1.0e-8 or value > 1.0 + 1.0e-8:
                raise ValueError(
                    "clean_clustering_coefficient_target must be finite and in [0, 1]."
                )
            clean_clustering[index] = float(np.clip(value, 0.0, 1.0))
        if graphlet_enabled:
            assert current_graphlet_probabilities is not None
            assert source_graphlet_probabilities is not None
            assert clean_graphlet_probabilities is not None
            assert current_graphlet_logits is not None
            assert source_graphlet_logits is not None
            assert clean_graphlet_logits is not None
            assert graphlet_coordinate_mask is not None
            source_prob_value = (
                example.current_graphlet_probabilities
                if example.source_graphlet_probabilities is None
                else example.source_graphlet_probabilities
            )
            source_logit_value = (
                example.current_graphlet_logits
                if example.source_graphlet_logits is None
                else example.source_graphlet_logits
            )
            arrays = [
                np.asarray(example.current_graphlet_probabilities, dtype=np.float32).reshape(-1),
                np.asarray(source_prob_value, dtype=np.float32).reshape(-1),
                np.asarray(example.clean_graphlet_probabilities_target, dtype=np.float32).reshape(-1),
                np.asarray(example.current_graphlet_logits, dtype=np.float32).reshape(-1),
                np.asarray(source_logit_value, dtype=np.float32).reshape(-1),
                np.asarray(example.clean_graphlet_logits_target, dtype=np.float32).reshape(-1),
                np.asarray(example.graphlet_coordinate_mask, dtype=np.bool_).reshape(-1),
            ]
            if any(array.size != graphlet_width for array in arrays):
                raise ValueError("Graphlet-logit target width mismatch during collation.")
            current_graphlet_probabilities[index] = arrays[0]
            source_graphlet_probabilities[index] = arrays[1]
            clean_graphlet_probabilities[index] = arrays[2]
            current_graphlet_logits[index] = arrays[3]
            source_graphlet_logits[index] = arrays[4]
            clean_graphlet_logits[index] = arrays[5]
            graphlet_coordinate_mask[index] = arrays[6]

    return TopologySpectralBatch(
        adjacency=torch.from_numpy(adjacency),
        node_mask=torch.from_numpy(node_mask),
        pair_mask=torch.from_numpy(pair_mask),
        degrees=torch.from_numpy(degrees),
        graph_size=torch.from_numpy(graph_sizes),
        time=torch.from_numpy(times),
        current_spectrum=torch.from_numpy(current_spectra),
        source_spectrum=torch.from_numpy(source_spectra),
        clean_spectrum_target=torch.from_numpy(clean_spectra),
        spectrum_mask=torch.from_numpy(spectrum_mask),
        current_heat_kernel=(
            torch.from_numpy(current_heat_kernel) if current_heat_kernel is not None else None
        ),
        source_heat_kernel=(
            torch.from_numpy(source_heat_kernel) if source_heat_kernel is not None else None
        ),
        clean_heat_kernel_target=(
            torch.from_numpy(clean_heat_kernel) if clean_heat_kernel is not None else None
        ),
        current_projector=(
            torch.from_numpy(current_projector) if current_projector is not None else None
        ),
        source_projector=(
            torch.from_numpy(source_projector) if source_projector is not None else None
        ),
        clean_projector_target=(
            torch.from_numpy(clean_projector) if clean_projector is not None else None
        ),
        current_eigenspace_histogram=(
            torch.from_numpy(current_eigenspace_histogram)
            if current_eigenspace_histogram is not None else None
        ),
        source_eigenspace_histogram=(
            torch.from_numpy(source_eigenspace_histogram)
            if source_eigenspace_histogram is not None else None
        ),
        clean_eigenspace_histogram_target=(
            torch.from_numpy(clean_eigenspace_histogram)
            if clean_eigenspace_histogram is not None else None
        ),
        eigenspace_histogram_block_mask=(
            torch.from_numpy(eigenspace_histogram_block_mask)
            if eigenspace_histogram_block_mask is not None else None
        ),
        eigenspace_histogram_block_weights=(
            torch.from_numpy(eigenspace_histogram_block_weights)
            if eigenspace_histogram_block_weights is not None else None
        ),
        current_edge_logits=(torch.from_numpy(current_edge_logits) if current_edge_logits is not None else None),
        source_edge_logits=(torch.from_numpy(source_edge_logits) if source_edge_logits is not None else None),
        clean_edge_logits_target=(torch.from_numpy(clean_edge_logits) if clean_edge_logits is not None else None),
        clean_edge_labels_target=(torch.from_numpy(clean_edge_labels) if clean_edge_labels is not None else None),
        clean_clustering_histogram_target=(
            torch.from_numpy(clean_clustering_histogram)
            if clean_clustering_histogram is not None else None
        ),
        clean_orbit_summary_target=(
            torch.from_numpy(clean_orbit_summary)
            if clean_orbit_summary is not None else None
        ),
        clean_induced_graphlet_histogram_target=(
            torch.from_numpy(clean_induced_histogram) if clean_induced_histogram is not None else None
        ),
        clean_cycle_graphlet_histogram_target=(
            torch.from_numpy(clean_cycle_histogram)
            if clean_cycle_histogram is not None else None
        ),
        clean_clustering_coefficient_target=(
            torch.from_numpy(clean_clustering) if clean_clustering is not None else None
        ),
        current_graphlet_probabilities=(
            torch.from_numpy(current_graphlet_probabilities)
            if current_graphlet_probabilities is not None else None
        ),
        source_graphlet_probabilities=(
            torch.from_numpy(source_graphlet_probabilities)
            if source_graphlet_probabilities is not None else None
        ),
        clean_graphlet_probabilities_target=(
            torch.from_numpy(clean_graphlet_probabilities)
            if clean_graphlet_probabilities is not None else None
        ),
        current_graphlet_logits=(
            torch.from_numpy(current_graphlet_logits)
            if current_graphlet_logits is not None else None
        ),
        source_graphlet_logits=(
            torch.from_numpy(source_graphlet_logits)
            if source_graphlet_logits is not None else None
        ),
        clean_graphlet_logits_target=(
            torch.from_numpy(clean_graphlet_logits)
            if clean_graphlet_logits is not None else None
        ),
        graphlet_coordinate_mask=(
            torch.from_numpy(graphlet_coordinate_mask)
            if graphlet_coordinate_mask is not None else None
        ),
    )


def _sorted_degree_sequence(graph: nx.Graph) -> list[int]:
    return sorted((int(degree) for _, degree in graph.degree()), reverse=True)


def assert_same_degree_fibre(source: nx.Graph, target: nx.Graph) -> None:
    """Require exact sorted degree equality for a clean spectral endpoint.

    Rewiring preserves all node degrees, so a terminal spectrum from a graph in a
    different degree fibre violates not only ``sum(lambda)=2m`` but also the
    fixed second Laplacian moment ``sum(lambda^2)=sum(d^2)+2m``.
    """

    left = _sorted_degree_sequence(source)
    right = _sorted_degree_sequence(target)
    if left != right:
        raise ValueError(
            "Spectral-guided training requires source and target graphs to have "
            "the same degree sequence. A clean target spectrum from another "
            "degree fibre is unreachable by degree-preserving rewiring."
        )


def build_spectral_teacher_states(
    degree_sequence: Sequence[int] | None = None,
    *,
    source_graph: nx.Graph | None = None,
    target_graph: nx.Graph,
    steps: int,
    proposal_budget: int,
    valid_candidate_budget: int,
    preserve_connectivity: bool,
    ensure_connected_source: bool,
    max_repair_trials: int,
    random_relabel_source: bool,
    source_randomization_steps: int,
    teacher_mode: str,
    teacher_temperature: float,
    teacher_top_k: int,
    teacher_sample_actions: bool,
    teacher_min_improvement: float = 0.0,
    target_tolerance: float = 0.0,
    teacher_allow_sideways: bool = False,
    teacher_max_consecutive_sideways: int = 0,
    teacher_sideways_tolerance: float = 0.0,
    teacher_tabu: bool = True,
    distance_metric: str = "rmse",
    distance_normalization: str = "mean_degree",
    low_frequency_weight: float = 1.0,
    low_frequency_cutoff: int = 0,
    require_same_degree_sequence: bool = True,
    rng: np.random.Generator,
) -> tuple[list[nx.Graph], dict[str, Any]]:
    """Build actual graph states using an oracle clean-spectrum teacher.

    The teacher is used only to create state coverage for x0 prediction.  It
    chooses valid degree-preserving swaps that move the actual graph spectrum
    toward the clean target spectrum.  Generation later converts the predicted
    clean spectrum into a scheduled *next* target before projecting by rewiring.
    """

    if not preserve_connectivity or not ensure_connected_source:
        raise ValueError(
            "Spectral teacher trajectories require a connected source and "
            "connectivity-preserving swaps."
        )
    if int(steps) < 0:
        raise ValueError("Spectral teacher steps must be nonnegative.")
    if int(teacher_top_k) < 0:
        raise ValueError("teacher_top_k must be nonnegative.")
    if not np.isfinite(teacher_temperature) or float(teacher_temperature) <= 0.0:
        raise ValueError("teacher_temperature must be finite and positive.")
    if not np.isfinite(teacher_min_improvement) or teacher_min_improvement < 0.0:
        raise ValueError("teacher_min_improvement must be finite and nonnegative.")
    if not np.isfinite(target_tolerance) or target_tolerance < 0.0:
        raise ValueError("target_tolerance must be finite and nonnegative.")
    if not np.isfinite(low_frequency_weight) or low_frequency_weight <= 0.0:
        raise ValueError("low_frequency_weight must be finite and positive.")
    if int(teacher_max_consecutive_sideways) < 0:
        raise ValueError("teacher_max_consecutive_sideways must be nonnegative.")
    if (
        not np.isfinite(teacher_sideways_tolerance)
        or teacher_sideways_tolerance < 0.0
    ):
        raise ValueError(
            "teacher_sideways_tolerance must be finite and nonnegative."
        )

    target = normalize_topology_graph(target_graph)
    if target.number_of_nodes() > 1 and not nx.is_connected(target):
        raise ValueError("Spectral training targets must be connected graphs.")

    if source_graph is not None:
        source = normalize_topology_graph(source_graph)
        source_mode = "completed_base_output"
        if int(source_randomization_steps) != 0:
            raise ValueError(
                "source_randomization_steps must be 0 for completed base outputs."
            )
        if random_relabel_source:
            source = _randomly_relabel_topology_graph(source, rng=rng)
    else:
        if degree_sequence is None:
            raise ValueError(
                "Spectral teacher training requires source_graph or degree_sequence."
            )
        source_mode = "target_degree_havel_hakimi"
        source = _construct_source_from_degree_sequence(
            degree_sequence,
            ensure_connected=ensure_connected_source,
            max_repair_trials=max_repair_trials,
            random_relabel=random_relabel_source,
            source_randomization_steps=source_randomization_steps,
            rng=rng,
        )
    if source.number_of_nodes() != target.number_of_nodes():
        raise ValueError("Spectral source and target must have identical graph size.")
    if source.number_of_nodes() > 1 and not nx.is_connected(source):
        raise ValueError("Spectral teacher training requires a connected source graph.")
    if require_same_degree_sequence:
        assert_same_degree_fibre(source, target)

    target_spectrum = laplacian_eigenvalues(target)
    target_first, target_second = spectrum_moments(target_spectrum)
    source_fixed_first, source_fixed_second = degree_spectral_moments(source)
    moment_residual = {
        "trace": float(target_first - source_fixed_first),
        "second": float(target_second - source_fixed_second),
    }
    if require_same_degree_sequence and (
        abs(moment_residual["trace"]) > 1.0e-7
        or abs(moment_residual["second"]) > 1.0e-6
    ):
        raise AssertionError(
            "Target Laplacian moments disagree with the source degree invariant."
        )

    scale = spectral_scale(source, mode=distance_normalization)

    def distance(graph: nx.Graph) -> float:
        return spectral_distance(
            laplacian_eigenvalues(graph),
            target_spectrum,
            metric=distance_metric,
            scale=scale,
            low_frequency_weight=low_frequency_weight,
            low_frequency_cutoff=low_frequency_cutoff,
        )

    initial_distance = distance(source)
    current = source.copy()
    states = [current.copy()]
    decisions: list[dict[str, Any]] = []
    accepted = 0
    stop_reason = "max_steps"
    mode = str(teacher_mode).lower()
    if mode not in {"hard", "soft"}:
        raise ValueError("teacher_mode must be 'hard' or 'soft'.")

    # F1: greedy one-swap descent on the spectral distance reaches a local
    # minimum after a handful of moves.  A tabu set plus bounded sideways moves
    # lets the teacher traverse flat regions instead of terminating there.
    visited: set[bytes] = {topology_state_key(current)}
    consecutive_sideways = 0
    sideways_accepted = 0
    # Sideways moves can end on a worse state than the best one seen, so the
    # best state is tracked explicitly and reported alongside the final one.
    best_distance = initial_distance
    best_state_index = 0

    for step in range(max(int(steps), 0)):
        current_spectrum = laplacian_eigenvalues(current)
        current_distance = spectral_distance(
            current_spectrum,
            target_spectrum,
            metric=distance_metric,
            scale=scale,
            low_frequency_weight=low_frequency_weight,
            low_frequency_cutoff=low_frequency_cutoff,
        )
        if current_distance <= float(target_tolerance):
            decisions.append(
                {
                    "step": step,
                    "actions": [],
                    "improvements": [],
                    "distribution": [1.0],
                    "selected_index": 0,
                    "stop_index": 0,
                    "current_spectral_discrepancy": current_distance,
                }
            )
            stop_reason = "target_spectral_tolerance"
            break

        candidates, candidate_graphs, proposal_diagnostics = (
            propose_valid_topology_swaps(
                current,
                proposal_budget=int(proposal_budget),
                valid_candidate_budget=int(valid_candidate_budget),
                preserve_connectivity=bool(preserve_connectivity),
                rng=rng,
                excluded_states=visited if teacher_tabu else None,
            )
        )
        candidate_distances: list[float] = []
        improvements: list[float] = []
        for action in candidates:
            candidate_distance = spectral_distance(
                laplacian_eigenvalues(candidate_graphs[action]),
                target_spectrum,
                metric=distance_metric,
                scale=scale,
                low_frequency_weight=low_frequency_weight,
                low_frequency_cutoff=low_frequency_cutoff,
            )
            candidate_distances.append(candidate_distance)
            improvements.append(current_distance - candidate_distance)

        improving = [
            index
            for index, value in enumerate(improvements)
            if value > float(teacher_min_improvement)
        ]
        if int(teacher_top_k) > 0 and len(improving) > int(teacher_top_k):
            ranked = sorted(
                improving,
                key=lambda index: improvements[index],
                reverse=True,
            )
            cutoff = improvements[ranked[int(teacher_top_k) - 1]]
            improving = [
                index
                for index in improving
                if improvements[index] >= cutoff - 1.0e-12
            ]

        stop_index = len(candidates)
        distribution = np.zeros(stop_index + 1, dtype=np.float64)
        move_kind = "improving"
        if not improving:
            # F1: no strictly improving swap.  Rather than terminating at the
            # local minimum, optionally take the least-harmful valid move so
            # the walk can cross a plateau.  `sideways_tolerance` is expressed
            # in the same normalized units as the spectral distance.
            sideways = [
                index
                for index, value in enumerate(improvements)
                if value >= -float(teacher_sideways_tolerance)
            ]
            can_step_sideways = (
                bool(teacher_allow_sideways)
                and bool(sideways)
                and consecutive_sideways < int(teacher_max_consecutive_sideways)
            )
            if can_step_sideways:
                best_sideways = max(
                    sideways, key=lambda index: improvements[index]
                )
                distribution[best_sideways] = 1.0
                move_kind = "sideways"
            else:
                distribution[stop_index] = 1.0
        elif mode == "hard":
            best = max(improvements[index] for index in improving)
            maxima = [
                index
                for index in improving
                if abs(improvements[index] - best) <= 1.0e-12
            ]
            distribution[int(rng.choice(maxima))] = 1.0
        else:
            logits = np.asarray(
                [improvements[index] for index in improving],
                dtype=np.float64,
            ) / float(teacher_temperature)
            logits -= float(np.max(logits))
            probabilities = np.exp(logits)
            probabilities /= float(probabilities.sum())
            distribution[np.asarray(improving, dtype=np.int64)] = probabilities

        if teacher_sample_actions:
            selected_index = int(rng.choice(len(distribution), p=distribution))
        else:
            maxima = np.flatnonzero(
                np.isclose(distribution, distribution.max(), atol=1.0e-12)
            )
            selected_index = int(rng.choice(maxima))

        decisions.append(
            {
                "step": step,
                "actions": [
                    [[list(edge) for edge in removed], [list(edge) for edge in added]]
                    for removed, added in candidates
                ],
                "improvements": [float(value) for value in improvements],
                "candidate_spectral_discrepancies": [
                    float(value) for value in candidate_distances
                ],
                "distribution": distribution.tolist(),
                "selected_index": selected_index,
                "stop_index": stop_index,
                "move_kind": move_kind,
                "current_spectral_discrepancy": current_distance,
                **proposal_diagnostics,
            }
        )
        if selected_index == stop_index:
            stop_reason = (
                "no_improving_spectral_swap"
                if not teacher_allow_sideways
                else "no_improving_or_sideways_spectral_swap"
            )
            break
        current = candidate_graphs[candidates[selected_index]]
        states.append(current.copy())
        accepted += 1
        visited.add(topology_state_key(current))
        if move_kind == "sideways":
            consecutive_sideways += 1
            sideways_accepted += 1
        else:
            consecutive_sideways = 0
        step_distance = current_distance - improvements[selected_index]
        if step_distance < best_distance:
            best_distance = float(step_distance)
            best_state_index = len(states) - 1

    final_distance = distance(current)
    report = {
        "source_mode": source_mode,
        "initial_spectral_discrepancy": float(initial_distance),
        "final_teacher_spectral_discrepancy": float(final_distance),
        "teacher_spectral_reduction": float(initial_distance - final_distance),
        "accepted_teacher_steps": int(accepted),
        "accepted_sideways_steps": int(sideways_accepted),
        "best_teacher_spectral_discrepancy": float(best_distance),
        "best_teacher_state_index": int(best_state_index),
        "teacher_stop_reason": stop_reason,
        "teacher_stop_selected": stop_reason != "max_steps",
        "teacher_decisions": decisions,
        "mean_valid_candidates": (
            float(
                np.mean(
                    [
                        row["num_valid_candidates"]
                        for row in decisions
                        if "num_valid_candidates" in row
                    ]
                )
            )
            if any("num_valid_candidates" in row for row in decisions)
            else 0.0
        ),
        "spectral_target_moment_residual": moment_residual,
        "spectral_distance_metric": str(distance_metric),
        "spectral_distance_normalization": str(distance_normalization),
    }
    return states, report


def build_spectral_examples(
    graphs: Sequence[nx.Graph | TopologyTrainingPair],
    *,
    trajectory_config: dict[str, Any] | None = None,
    spectral_config: dict[str, Any] | None = None,
    graphlet_basis: TopologyGraphletBasis | None = None,
    graphlet_logit_epsilon: float = 1.0e-5,
    seed: int = 0,
) -> tuple[list[TopologySpectralExample], dict[str, Any]]:
    """Create variable-length clean-spectrum supervision from actual graph states."""

    cfg = dict(trajectory_config or {})
    spec_cfg = dict(spectral_config or {})
    rng = np.random.default_rng(int(seed))
    examples: list[TopologySpectralExample] = []
    reports: list[dict[str, Any]] = []
    trajectory_id = 0
    paths_per_graph = max(int(cfg.get("paths_per_graph", 1)), 1)
    valid_budget = int(
        cfg.get("valid_candidate_budget", cfg.get("candidate_budget", 64))
    )
    proposal_budget = int(
        cfg.get(
            "proposal_budget",
            valid_budget if valid_budget < 0 else max(valid_budget, 1) * 4,
        )
    )
    require_same_degree_sequence = bool(
        spec_cfg.get("require_same_degree_sequence", True)
    )

    for raw_item in graphs:
        if isinstance(raw_item, TopologyTrainingPair):
            source = normalize_topology_graph(raw_item.source_graph)
            target = normalize_topology_graph(raw_item.target_graph)
            base_generator = str(raw_item.base_generator)
            source_index = int(raw_item.source_index)
            target_index = int(raw_item.target_index)
            matching_cost = float(raw_item.matching_cost)
            if source.number_of_nodes() != target.number_of_nodes():
                raise ValueError(
                    "Completed spectral source/target pairs must have identical size."
                )
            if require_same_degree_sequence:
                assert_same_degree_fibre(source, target)
        else:
            source = None
            target = normalize_topology_graph(raw_item)
            base_generator = "target_degree_havel_hakimi"
            source_index = -1
            target_index = -1
            matching_cost = 0.0

        target_spectrum = laplacian_eigenvalues(target)
        target_graphlet_probabilities: np.ndarray | None = None
        target_graphlet_logits: np.ndarray | None = None
        target_graphlet_mask: np.ndarray | None = None
        if graphlet_basis is not None:
            target_graphlet_probabilities, target_graphlet_mask, _ = (
                extract_topology_graphlet_simplex(
                    target,
                    graphlet_basis=graphlet_basis,
                )
            )
            target_graphlet_logits = graphlet_simplex_to_clr(
                target_graphlet_probabilities,
                graphlet_basis=graphlet_basis,
                epsilon=float(graphlet_logit_epsilon),
                coordinate_mask=target_graphlet_mask,
            )
        degree_sequence = [int(target.degree(node)) for node in target.nodes()]
        for _path in range(paths_per_graph):
            states, report = build_spectral_teacher_states(
                degree_sequence if source is None else None,
                source_graph=source,
                target_graph=target,
                steps=int(cfg.get("steps", 32)),
                proposal_budget=proposal_budget,
                valid_candidate_budget=valid_budget,
                preserve_connectivity=bool(cfg.get("preserve_connectivity", True)),
                ensure_connected_source=bool(cfg.get("ensure_connected_source", True)),
                max_repair_trials=int(cfg.get("max_repair_trials", 10000)),
                random_relabel_source=bool(cfg.get("random_relabel_source", True)),
                source_randomization_steps=int(cfg.get("source_randomization_steps", 0)),
                teacher_mode=str(cfg.get("teacher_mode", "hard")),
                teacher_temperature=float(cfg.get("teacher_temperature", 1.0)),
                teacher_top_k=int(cfg.get("teacher_top_k", 0)),
                teacher_sample_actions=bool(cfg.get("teacher_sample_actions", False)),
                teacher_min_improvement=float(cfg.get("teacher_min_improvement", 0.0)),
                target_tolerance=float(cfg.get("target_tolerance", 0.0)),
                teacher_allow_sideways=bool(
                    cfg.get("teacher_allow_sideways", False)
                ),
                teacher_max_consecutive_sideways=int(
                    cfg.get("teacher_max_consecutive_sideways", 0)
                ),
                teacher_sideways_tolerance=float(
                    cfg.get("teacher_sideways_tolerance", 0.0)
                ),
                teacher_tabu=bool(cfg.get("teacher_tabu", True)),
                distance_metric=str(spec_cfg.get("distance", "rmse")),
                distance_normalization=str(
                    spec_cfg.get("normalization", "mean_degree")
                ),
                low_frequency_weight=float(
                    spec_cfg.get("low_frequency_weight", 1.0)
                ),
                low_frequency_cutoff=int(
                    spec_cfg.get("low_frequency_cutoff", 0)
                ),
                require_same_degree_sequence=require_same_degree_sequence,
                rng=rng,
            )
            selected_count = min(
                max(int(cfg.get("states_per_graph", 8)), 1),
                len(states),
            )
            indices = (
                list(range(len(states)))
                if selected_count == len(states)
                else sorted(
                    {
                        int(round(value))
                        for value in np.linspace(
                            0,
                            len(states) - 1,
                            num=selected_count,
                        )
                    }
                )
            )
            horizon = max(int(cfg.get("steps", 32)), 1)
            decisions = report.get("teacher_decisions", [])
            for step in indices:
                decision = decisions[step] if step < len(decisions) else None
                actions: tuple[Action, ...] = ()
                distribution = None
                selected_index = -1
                if decision is not None:
                    actions = tuple(
                        make_action(value[0], value[1])
                        for value in decision.get("actions", [])
                    )
                    distribution = np.asarray(
                        decision.get("distribution", [1.0]),
                        dtype=np.float32,
                    )
                    selected_index = int(decision.get("selected_index", -1))
                current_graphlet_probabilities = None
                current_graphlet_logits = None
                current_graphlet_mask = None
                if graphlet_basis is not None:
                    current_graphlet_probabilities, current_graphlet_mask, _ = (
                        extract_topology_graphlet_simplex(
                            states[step],
                            graphlet_basis=graphlet_basis,
                        )
                    )
                    current_graphlet_logits = graphlet_simplex_to_clr(
                        current_graphlet_probabilities,
                        graphlet_basis=graphlet_basis,
                        epsilon=float(graphlet_logit_epsilon),
                        coordinate_mask=current_graphlet_mask,
                    )
                    if target_graphlet_mask is None or not np.array_equal(
                        current_graphlet_mask, target_graphlet_mask
                    ):
                        raise AssertionError(
                            "Source and clean graphlet coordinate masks must agree for equal-size graphs."
                        )
                examples.append(
                    TopologySpectralExample(
                        current_graph=states[step],
                        time=float(step / horizon),
                        clean_spectrum_target=target_spectrum.astype(np.float32).copy(),
                        current_graphlet_probabilities=(
                            None if current_graphlet_probabilities is None
                            else current_graphlet_probabilities.astype(np.float32).copy()
                        ),
                        source_graphlet_probabilities=(
                            None if current_graphlet_probabilities is None
                            else current_graphlet_probabilities.astype(np.float32).copy()
                        ),
                        clean_graphlet_probabilities_target=(
                            None if target_graphlet_probabilities is None
                            else target_graphlet_probabilities.astype(np.float32).copy()
                        ),
                        current_graphlet_logits=(
                            None if current_graphlet_logits is None
                            else current_graphlet_logits.astype(np.float32).copy()
                        ),
                        source_graphlet_logits=(
                            None if current_graphlet_logits is None
                            else current_graphlet_logits.astype(np.float32).copy()
                        ),
                        clean_graphlet_logits_target=(
                            None if target_graphlet_logits is None
                            else target_graphlet_logits.astype(np.float32).copy()
                        ),
                        graphlet_coordinate_mask=(
                            None if current_graphlet_mask is None
                            else current_graphlet_mask.astype(np.bool_).copy()
                        ),
                        base_generator=base_generator,
                        source_index=source_index,
                        target_index=target_index,
                        matching_cost=matching_cost,
                        trajectory_id=trajectory_id,
                        step=step,
                        teacher_actions=actions,
                        teacher_distribution=distribution,
                        teacher_selected_index=selected_index,
                    )
                )
            report = {
                **report,
                "base_generator": base_generator,
                "source_index": source_index,
                "target_index": target_index,
                "matching_cost": matching_cost,
            }
            reports.append(report)
            trajectory_id += 1

    diagnostics = {
        "num_graphs": len(graphs),
        "num_paths": len(reports),
        "num_examples": len(examples),
        "source_modes": sorted({str(r.get("source_mode")) for r in reports}),
        "base_generators": sorted({str(r.get("base_generator")) for r in reports}),
        "mean_matching_cost": (
            float(np.mean([r["matching_cost"] for r in reports])) if reports else 0.0
        ),
        "mean_initial_spectral_discrepancy": (
            float(np.mean([r["initial_spectral_discrepancy"] for r in reports]))
            if reports
            else 0.0
        ),
        "mean_final_teacher_spectral_discrepancy": (
            float(
                np.mean([r["final_teacher_spectral_discrepancy"] for r in reports])
            )
            if reports
            else 0.0
        ),
        "mean_accepted_teacher_steps": (
            float(np.mean([r["accepted_teacher_steps"] for r in reports]))
            if reports
            else 0.0
        ),
        "teacher_stop_rate": (
            float(np.mean([bool(r["teacher_stop_selected"]) for r in reports]))
            if reports
            else 0.0
        ),
        "mean_valid_candidates": (
            float(np.mean([r["mean_valid_candidates"] for r in reports]))
            if reports
            else 0.0
        ),
    }
    return examples, diagnostics


class TopologySpectralTrajectoryIterableDataset(torch.utils.data.IterableDataset):
    """Generate clean-spectrum teacher examples lazily."""

    def __init__(
        self,
        graphs: Sequence[nx.Graph | TopologyTrainingPair],
        *,
        trajectory_config: dict[str, Any] | None = None,
        spectral_config: dict[str, Any] | None = None,
        graphlet_basis: TopologyGraphletBasis | None = None,
        graphlet_logit_epsilon: float = 1.0e-5,
        seed: int = 0,
        shuffle_graphs: bool = True,
    ) -> None:
        super().__init__()
        self.graphs = tuple(graphs)
        self.trajectory_config = dict(trajectory_config or {})
        self.spectral_config = dict(spectral_config or {})
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
            examples, diagnostics = build_spectral_examples(
                [self.graphs[int(graph_index)]],
                trajectory_config=self.trajectory_config,
                spectral_config=self.spectral_config,
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


def _resolve_spectral_diffusion_endpoints(
    raw_item: nx.Graph | TopologyTrainingPair,
    *,
    source_config: dict[str, Any],
    require_same_degree_sequence: bool,
    rng: np.random.Generator,
) -> tuple[nx.Graph, nx.Graph, dict[str, Any]]:
    """Resolve a fixed source graph and clean target for summary diffusion.

    Unlike the legacy teacher-trajectory builder, this helper never rewires the
    source.  It only constructs/loads the two bridge endpoints.
    """

    if isinstance(raw_item, TopologyTrainingPair):
        source = normalize_topology_graph(raw_item.source_graph)
        target = normalize_topology_graph(raw_item.target_graph)
        metadata = {
            "source_mode": "completed_base_output",
            "base_generator": str(raw_item.base_generator),
            "source_index": int(raw_item.source_index),
            "target_index": int(raw_item.target_index),
            "matching_cost": float(raw_item.matching_cost),
        }
    else:
        target = normalize_topology_graph(raw_item)
        degree_sequence = [int(target.degree(node)) for node in target.nodes()]
        source = _construct_source_from_degree_sequence(
            degree_sequence,
            ensure_connected=bool(source_config.get("ensure_connected_source", True)),
            max_repair_trials=int(source_config.get("max_repair_trials", 10000)),
            random_relabel=bool(source_config.get("random_relabel_source", True)),
            source_randomization_steps=int(source_config.get("source_randomization_steps", 0)),
            rng=rng,
        )
        metadata = {
            "source_mode": "target_degree_havel_hakimi",
            "base_generator": "target_degree_havel_hakimi",
            "source_index": -1,
            "target_index": -1,
            "matching_cost": 0.0,
        }

    if source.number_of_nodes() != target.number_of_nodes():
        raise ValueError("Summary-diffusion source and target must have identical graph size.")
    if source.number_of_nodes() > 1 and not nx.is_connected(source):
        raise ValueError("Summary-diffusion source graph must be connected.")
    if target.number_of_nodes() > 1 and not nx.is_connected(target):
        raise ValueError("Summary-diffusion clean target must be connected.")
    if require_same_degree_sequence:
        assert_same_degree_fibre(source, target)
    return source, target, metadata


def _align_target_to_source_indexed_degrees(source: nx.Graph, target: nx.Graph, rng: np.random.Generator) -> nx.Graph:
    """Relabel target so each node index has the source node's degree.

    Pairwise edge diffusion needs a node-aligned clean endpoint. GraphER swaps
    preserve indexed degrees, so alignment is restricted to equal-degree groups
    and never changes either graph's degree multiset or any invariant summary.
    """
    source = normalize_topology_graph(source)
    target = normalize_topology_graph(target)
    if sorted(dict(source.degree()).values()) != sorted(dict(target.degree()).values()):
        raise ValueError("Edge diffusion requires source/target in the same degree fibre.")
    mapping = {}
    degrees = sorted(set(dict(source.degree()).values()))
    for degree in degrees:
        src = [u for u,d in source.degree() if d == degree]
        tgt = [u for u,d in target.degree() if d == degree]
        if len(src) != len(tgt):
            raise AssertionError("Degree-group cardinality mismatch during endpoint alignment.")
        tgt = list(np.asarray(tgt)[rng.permutation(len(tgt))]) if len(tgt)>1 else tgt
        mapping.update({int(v):int(u) for u,v in zip(sorted(src),tgt)})
    aligned = normalize_topology_graph(nx.relabel_nodes(target,mapping,copy=True))
    if [aligned.degree(i) for i in range(len(aligned))] != [source.degree(i) for i in range(len(source))]:
        raise AssertionError("Indexed degree alignment failed.")
    return aligned


def _align_target_to_source_projector_structure(
    source: nx.Graph,
    target: nx.Graph,
    *,
    projector_rank: int,
) -> tuple[nx.Graph, float]:
    """Deterministically align equal-degree nodes using local/projector signatures.

    A projector is node-indexed, so the random equal-degree permutation used by
    the edge-only path would create arbitrary eigenspace supervision.  This
    matching keeps the hard indexed-degree constraint and uses a Hungarian
    assignment inside each degree group.  The signatures themselves are
    invariant to the labels of the *other* vertices.
    """

    from scipy.optimize import linear_sum_assignment

    source = normalize_topology_graph(source)
    target = normalize_topology_graph(target)
    source_degrees = dict(source.degree())
    target_degrees = dict(target.degree())
    if sorted(source_degrees.values()) != sorted(target_degrees.values()):
        raise ValueError("Projector alignment requires source/target in the same degree fibre.")
    source_p = laplacian_eigenspace_projector(source, rank=projector_rank)
    target_p = laplacian_eigenspace_projector(target, rank=projector_rank)
    source_sig = projector_node_signatures(source, source_p)
    target_sig = projector_node_signatures(target, target_p)
    mapping: dict[int, int] = {}
    total_cost = 0.0
    for degree in sorted(set(source_degrees.values())):
        src = sorted(node for node, value in source_degrees.items() if value == degree)
        tgt = sorted(node for node, value in target_degrees.items() if value == degree)
        if len(src) != len(tgt):
            raise AssertionError("Degree-group cardinality mismatch during projector alignment.")
        if len(src) == 1:
            mapping[int(tgt[0])] = int(src[0])
            continue
        a = source_sig[np.asarray(src, dtype=np.int64)]
        b = target_sig[np.asarray(tgt, dtype=np.int64)]
        cost = np.square(a[:, None, :] - b[None, :, :]).mean(axis=-1)
        # Stable tie breaking without changing the meaningful cost at printed precision.
        tie = np.arange(cost.size, dtype=np.float64).reshape(cost.shape) * 1.0e-12
        rows, cols = linear_sum_assignment(cost + tie)
        for row, col in zip(rows.tolist(), cols.tolist()):
            mapping[int(tgt[col])] = int(src[row])
            total_cost += float(cost[row, col])
    aligned = normalize_topology_graph(nx.relabel_nodes(target, mapping, copy=True))
    if [aligned.degree(i) for i in range(len(aligned))] != [source.degree(i) for i in range(len(source))]:
        raise AssertionError("Projector structural alignment failed to preserve indexed degrees.")
    return aligned, total_cost / max(len(source), 1)


def _binary_edge_logits(graph: nx.Graph, smoothing: float) -> tuple[np.ndarray,np.ndarray]:
    if not 0.0 < float(smoothing) < 0.5:
        raise ValueError("generic edge_diffusion.smoothing must be in (0,0.5).")
    n=len(graph); labels=nx.to_numpy_array(graph,nodelist=list(range(n)),weight=None,dtype=np.int64)
    labels=(labels>0).astype(np.int64); np.fill_diagonal(labels,0)
    probs=np.full((n,n,2),float(smoothing),dtype=np.float64)
    for r in (0,1): probs[...,r]=np.where(labels==r,1.0-float(smoothing),float(smoothing))
    logits=np.log(np.maximum(probs,1e-12)); logits-=logits.mean(axis=-1,keepdims=True)
    mask=~np.eye(n,dtype=bool); logits*=mask[...,None]
    return logits.astype(np.float32),labels


def _sample_binary_edge_bridge(source:np.ndarray,target:np.ndarray,progress:float,sigma:float,rng:np.random.Generator)->tuple[np.ndarray,float]:
    if not np.isfinite(sigma) or sigma<0: raise ValueError("generic edge_diffusion.sigma must be finite and nonnegative.")
    t=float(np.clip(progress,0.0,1.0)); mean=(1-t)*source+t*target
    noise=rng.normal(size=source.shape); noise=0.5*(noise+noise.transpose(1,0,2)); noise-=noise.mean(axis=-1,keepdims=True)
    n=source.shape[0]; noise*=~np.eye(n,dtype=bool)[...,None]
    std=float(sigma)*np.sqrt(max(t*(1-t),0.0)); state=mean+std*noise
    state=0.5*(state+state.transpose(1,0,2));state-=state.mean(axis=-1,keepdims=True);state*=~np.eye(n,dtype=bool)[...,None]
    return state.astype(np.float32),float(np.sqrt(np.mean((std*noise)**2)))


@dataclass(frozen=True)
class TopologySpectralDiffusionEndpoint:
    source: nx.Graph
    target: nx.Graph
    metadata: dict[str, Any]
    source_spectrum: np.ndarray
    clean_spectrum: np.ndarray
    spectral_scale: float
    source_heat_kernel: np.ndarray | None = None
    clean_heat_kernel: np.ndarray | None = None
    heat_kernel_times: tuple[float, ...] = ()
    source_projector: np.ndarray | None = None
    clean_projector: np.ndarray | None = None
    projector_rank: int = 0
    projector_alignment_cost: float = 0.0
    source_eigenspace_histogram: np.ndarray | None = None
    clean_eigenspace_histogram: np.ndarray | None = None
    eigenspace_histogram_block_mask: np.ndarray | None = None
    eigenspace_histogram_block_weights: np.ndarray | None = None
    eigenspace_histogram_spec: dict[str, Any] | None = None
    source_edge_logits: np.ndarray | None = None
    clean_edge_logits: np.ndarray | None = None
    clean_edge_labels: np.ndarray | None = None
    source_graphlet_probabilities: np.ndarray | None = None
    clean_graphlet_probabilities: np.ndarray | None = None
    source_graphlet_logits: np.ndarray | None = None
    clean_graphlet_logits: np.ndarray | None = None
    graphlet_coordinate_mask: np.ndarray | None = None
    clean_clustering_coefficient: float = 0.0
    spectral_endpoint_distance: float = 0.0
    clean_clustering_histogram: np.ndarray | None = None
    clean_orbit_summary: np.ndarray | None = None
    clean_cycle_graphlet_histogram: np.ndarray | None = None
    clean_induced_graphlet_histogram: np.ndarray | None = None


def _prepare_spectral_diffusion_endpoint(
    raw_item: nx.Graph | TopologyTrainingPair,
    *,
    source_config: dict[str, Any],
    spectral_config: dict[str, Any],
    graphlet_basis: TopologyGraphletBasis | None,
    graphlet_logit_epsilon: float,
    require_same_degree_sequence: bool,
    rng: np.random.Generator,
    structure_summary_config: dict[str, Any] | None = None,
    edge_diffusion_config: dict[str, Any] | None = None,
) -> TopologySpectralDiffusionEndpoint:
    source, target, metadata = _resolve_spectral_diffusion_endpoints(
        raw_item,
        source_config=source_config,
        require_same_degree_sequence=require_same_degree_sequence,
        rng=rng,
    )
    representation = str(spectral_config.get("representation", "eigenvalues")).lower()
    if representation in {"heat", "heatkernel", "heat-kernel"}:
        representation = "heat_kernel"
    if representation in {"eigenspace", "lambda_projector", "eigenvalues_projector", "lambda+p", "lambda_p"}:
        representation = "lambda_projector"
    if representation in {
        "eigenspace_histogram", "lambda_eigenspace_histogram",
        "degree_spectral_histogram", "spectral_distance_histogram",
    }:
        representation = "lambda_eigenspace_histogram"
    if representation not in {"eigenvalues", "heat_kernel", "lambda_projector", "lambda_eigenspace_histogram"}:
        raise ValueError(
            "spectral_prediction.representation must be eigenvalues, heat_kernel, "
            "lambda_projector, or lambda_eigenspace_histogram."
        )
    heat_times = validate_heat_times(spectral_config.get("heat_kernel_times", [0.25, 1.0, 4.0]))
    heat_normalization = str(spectral_config.get("heat_kernel_normalization", spectral_config.get("normalization", "mean_degree")))
    projector_rank = int(spectral_config.get("projector_rank", 4))
    if projector_rank < 1:
        raise ValueError("spectral_prediction.projector_rank must be positive.")
    eig_hist_spec = None
    if representation == "lambda_eigenspace_histogram":
        if "eigenspace_histogram_degree_max" not in spectral_config:
            raise ValueError(
                "lambda_eigenspace_histogram requires spectral_prediction.eigenspace_histogram_degree_max "
                "so all variable-size graphs share one fixed histogram vocabulary."
            )
        eig_hist_spec = SpectralDistanceHistogramSpec.from_config(spectral_config)
    edge_cfg=dict(edge_diffusion_config or {})
    edge_enabled=bool(edge_cfg.get("enabled",False))
    projector_alignment_cost = 0.0
    if representation == "lambda_projector":
        target, projector_alignment_cost = _align_target_to_source_projector_structure(
            source, target, projector_rank=projector_rank
        )
    elif edge_enabled or representation == "heat_kernel":
        target=_align_target_to_source_indexed_degrees(source,target,rng)
    source_spectrum = laplacian_eigenvalues(source)
    clean_spectrum = laplacian_eigenvalues(target)
    source_heat_kernel = clean_heat_kernel = None
    if representation == "heat_kernel":
        source_heat_kernel = heat_kernel_stack(
            source, times=heat_times, normalization=heat_normalization
        )
        clean_heat_kernel = heat_kernel_stack(
            target, times=heat_times, normalization=heat_normalization
        )
    source_projector = clean_projector = None
    if representation == "lambda_projector":
        source_projector = laplacian_eigenspace_projector(source, rank=projector_rank)
        clean_projector = laplacian_eigenspace_projector(target, rank=projector_rank)
    source_eigenspace_histogram = clean_eigenspace_histogram = None
    eigenspace_histogram_block_mask = eigenspace_histogram_block_weights = None
    if representation == "lambda_eigenspace_histogram":
        assert eig_hist_spec is not None
        source_eigenspace_histogram, source_block_mask, source_block_weights = (
            extract_degree_conditioned_spectral_histogram(source, eig_hist_spec)
        )
        clean_eigenspace_histogram, clean_block_mask, clean_block_weights = (
            extract_degree_conditioned_spectral_histogram(target, eig_hist_spec)
        )
        # Source and clean graphs lie in the same degree fibre, so the degree-pair
        # histogram support and weights are exact invariants of the pair.
        if not np.array_equal(source_block_mask, clean_block_mask):
            raise AssertionError("Same-degree source/target graphs must share eigenspace histogram blocks.")
        if not np.allclose(source_block_weights, clean_block_weights, rtol=0.0, atol=1.0e-12):
            raise AssertionError("Same-degree source/target graphs must share eigenspace histogram block weights.")
        eigenspace_histogram_block_mask = source_block_mask
        eigenspace_histogram_block_weights = source_block_weights
    source_edge_logits=clean_edge_logits=clean_edge_labels=None
    if edge_enabled:
        smoothing=float(edge_cfg.get("smoothing",0.01))
        source_edge_logits,_=_binary_edge_logits(source,smoothing)
        clean_edge_logits,clean_edge_labels=_binary_edge_logits(target,smoothing)
    scale = spectral_scale(
        source,
        mode=str(spectral_config.get("normalization", "mean_degree")),
    )

    source_prob = source_logits = clean_prob = clean_logits = graphlet_mask = None
    if graphlet_basis is not None:
        source_prob, source_mask, _ = extract_topology_graphlet_simplex(
            source,
            graphlet_basis=graphlet_basis,
        )
        clean_prob, clean_mask, _ = extract_topology_graphlet_simplex(
            target,
            graphlet_basis=graphlet_basis,
        )
        if not np.array_equal(source_mask, clean_mask):
            raise AssertionError(
                "Equal-size source and clean graphs must share graphlet coordinate masks."
            )
        graphlet_mask = source_mask
        source_logits = graphlet_simplex_to_clr(
            source_prob,
            graphlet_basis=graphlet_basis,
            epsilon=float(graphlet_logit_epsilon),
            coordinate_mask=graphlet_mask,
        )
        clean_logits = graphlet_simplex_to_clr(
            clean_prob,
            graphlet_basis=graphlet_basis,
            epsilon=float(graphlet_logit_epsilon),
            coordinate_mask=graphlet_mask,
        )

    histogram_bins = clustering_histogram_bins(structure_summary_config)
    orbit_width = orbit_summary_width(structure_summary_config)
    cycle_k = cycle_graphlet_k(structure_summary_config)
    induced_spec = InducedGraphletSpec.from_config(structure_summary_config)
    return TopologySpectralDiffusionEndpoint(
        source=source,
        target=target,
        metadata=dict(metadata),
        source_spectrum=source_spectrum,
        clean_spectrum=clean_spectrum,
        spectral_scale=float(scale),
        source_heat_kernel=source_heat_kernel,
        clean_heat_kernel=clean_heat_kernel,
        heat_kernel_times=(heat_times if representation == "heat_kernel" else ()),
        source_projector=source_projector,
        clean_projector=clean_projector,
        projector_rank=(effective_projector_rank(source.number_of_nodes(), projector_rank) if representation == "lambda_projector" else 0),
        projector_alignment_cost=float(projector_alignment_cost),
        source_eigenspace_histogram=(
            None if source_eigenspace_histogram is None else source_eigenspace_histogram.astype(np.float32)
        ),
        clean_eigenspace_histogram=(
            None if clean_eigenspace_histogram is None else clean_eigenspace_histogram.astype(np.float32)
        ),
        eigenspace_histogram_block_mask=(
            None if eigenspace_histogram_block_mask is None else eigenspace_histogram_block_mask.astype(np.bool_)
        ),
        eigenspace_histogram_block_weights=(
            None if eigenspace_histogram_block_weights is None else eigenspace_histogram_block_weights.astype(np.float32)
        ),
        eigenspace_histogram_spec=(None if eig_hist_spec is None else eig_hist_spec.metadata()),
        source_edge_logits=source_edge_logits,
        clean_edge_logits=clean_edge_logits,
        clean_edge_labels=clean_edge_labels,
        source_graphlet_probabilities=source_prob,
        clean_graphlet_probabilities=clean_prob,
        source_graphlet_logits=source_logits,
        clean_graphlet_logits=clean_logits,
        graphlet_coordinate_mask=graphlet_mask,
        clean_clustering_coefficient=float(nx.average_clustering(target)),
        clean_clustering_histogram=(
            extract_clustering_histogram(target, histogram_bins)
            if histogram_bins is not None else None
        ),
        clean_orbit_summary=(
            extract_orbit_summary(target) if orbit_width is not None else None
        ),
        clean_induced_graphlet_histogram=(
            extract_induced_histogram(target, induced_spec) if induced_spec is not None else None
        ),
        clean_cycle_graphlet_histogram=(
            extract_cycle_graphlet_histogram(target, k=cycle_k) if cycle_k is not None else None
        ),
        spectral_endpoint_distance=(
            heat_kernel_distance(
                source_heat_kernel, clean_heat_kernel,
                metric=str(spectral_config.get("distance", "rmse")),
            )
            if representation == "heat_kernel"
            else (
                float(spectral_config.get("lambda_weight", 1.0)) * spectral_distance(
                    source_spectrum, clean_spectrum,
                    metric=str(spectral_config.get("distance", "rmse")), scale=scale,
                    low_frequency_weight=float(spectral_config.get("low_frequency_weight", 1.0)),
                    low_frequency_cutoff=int(spectral_config.get("low_frequency_cutoff", 0)),
                )
                + float(spectral_config.get("projector_weight", 1.0)) * eigenspace_projector_distance(
                    source_projector, clean_projector, rank=projector_rank,
                    metric=str(spectral_config.get("projector_distance", "chordal")),
                )
                if representation == "lambda_projector"
                else (
                    float(spectral_config.get("lambda_weight", 1.0)) * spectral_distance(
                        source_spectrum, clean_spectrum,
                        metric=str(spectral_config.get("distance", "rmse")), scale=scale,
                        low_frequency_weight=float(spectral_config.get("low_frequency_weight", 1.0)),
                        low_frequency_cutoff=int(spectral_config.get("low_frequency_cutoff", 0)),
                    )
                    + float(spectral_config.get("eigenspace_histogram_weight", 1.0)) * spectral_histogram_wasserstein(
                        source_eigenspace_histogram, clean_eigenspace_histogram, eig_hist_spec,
                        block_weights=eigenspace_histogram_block_weights,
                    )
                    if representation == "lambda_eigenspace_histogram"
                    else spectral_distance(
                        source_spectrum, clean_spectrum,
                        metric=str(spectral_config.get("distance", "rmse")), scale=scale,
                        low_frequency_weight=float(spectral_config.get("low_frequency_weight", 1.0)),
                        low_frequency_cutoff=int(spectral_config.get("low_frequency_cutoff", 0)),
                    )
                )
            )
        ),
    )


def _sample_spectral_diffusion_endpoint_examples(
    endpoint: TopologySpectralDiffusionEndpoint,
    *,
    diffusion_config: dict[str, Any],
    graphlet_basis: TopologyGraphletBasis | None,
    seed: int,
) -> tuple[list[TopologySpectralExample], dict[str, Any]]:
    diff_values = dict(diffusion_config or {})
    diff_cfg = SummaryDiffusionConfig.from_dict(diff_values)
    rng = np.random.default_rng(int(seed))
    samples_per_graph = max(int(diff_values.get("samples_per_graph", 32)), 1)
    paths_per_graph = max(int(diff_values.get("paths_per_graph", 1)), 1)
    spectral_noise_rms: list[float] = []
    heat_kernel_noise_rms: list[float] = []
    projector_noise_rms: list[float] = []
    eigenspace_histogram_noise_rms: list[float] = []
    graphlet_noise_rms: list[float] = []
    edge_noise_rms: list[float] = []
    examples: list[TopologySpectralExample] = []

    for path in range(paths_per_graph):
        progresses = diff_cfg.sample_progresses(samples_per_graph, rng=rng)
        for local_index, progress in enumerate(progresses):
            current_heat_kernel = None
            current_projector = None
            current_eigenspace_histogram = None
            if endpoint.source_heat_kernel is not None:
                assert endpoint.clean_heat_kernel is not None
                current_heat_kernel, heat_diag = sample_heat_kernel_bridge_marginal(
                    endpoint.source_heat_kernel,
                    endpoint.clean_heat_kernel,
                    progress=float(progress),
                    sigma=diff_cfg.heat_kernel_sigma,
                    schedule=diff_cfg,
                    rng=rng,
                )
                heat_kernel_noise_rms.append(float(heat_diag["noise_rms"]))
                # Legacy heat-kernel mode carries the continuous spectral state
                # entirely in H_tau.
                current_spectrum = endpoint.source_spectrum.copy()
                spec_diag = {"noise_rms": 0.0}
            else:
                # Eigenvalue diffusion remains active for both eigenvalue-only
                # and lambda+projector representations.
                current_spectrum, spec_diag = sample_spectral_bridge_marginal(
                    endpoint.source_spectrum,
                    endpoint.clean_spectrum,
                    progress=float(progress),
                    sigma=diff_cfg.spectral_sigma,
                    scale=endpoint.spectral_scale,
                    preserve_trace=diff_cfg.preserve_spectral_trace,
                    fix_lambda1=diff_cfg.fix_spectral_lambda1,
                    schedule=diff_cfg,
                    rng=rng,
                )
                spectral_noise_rms.append(float(spec_diag["noise_rms"]))
                if endpoint.source_projector is not None:
                    assert endpoint.clean_projector is not None
                    current_projector, projector_diag = sample_eigenspace_projector_bridge_marginal(
                        endpoint.source_projector, endpoint.clean_projector,
                        progress=float(progress), sigma=diff_cfg.projector_sigma,
                        schedule=diff_cfg, rng=rng,
                    )
                    projector_noise_rms.append(float(projector_diag["noise_rms"]))
                if endpoint.source_eigenspace_histogram is not None:
                    assert endpoint.clean_eigenspace_histogram is not None
                    assert endpoint.eigenspace_histogram_block_mask is not None
                    assert endpoint.eigenspace_histogram_spec is not None
                    bins = int(endpoint.eigenspace_histogram_spec["bins"])
                    current_eigenspace_histogram, eig_hist_diag = sample_eigenspace_histogram_bridge_marginal(
                        endpoint.source_eigenspace_histogram,
                        endpoint.clean_eigenspace_histogram,
                        progress=float(progress),
                        sigma=diff_cfg.eigenspace_histogram_sigma,
                        block_mask=endpoint.eigenspace_histogram_block_mask,
                        bins=bins,
                        schedule=diff_cfg,
                        rng=rng,
                    )
                    eigenspace_histogram_noise_rms.append(float(eig_hist_diag["noise_rms"]))

            current_edge_logits=None
            if endpoint.source_edge_logits is not None:
                current_edge_logits,edge_rms=_sample_binary_edge_bridge(
                    endpoint.source_edge_logits,endpoint.clean_edge_logits,float(progress),
                    float(diff_values.get("edge_sigma",1.0)),rng)
                edge_noise_rms.append(edge_rms)
            current_prob = current_logits = None
            if graphlet_basis is not None:
                assert endpoint.source_graphlet_logits is not None
                assert endpoint.clean_graphlet_logits is not None
                assert endpoint.graphlet_coordinate_mask is not None
                current_logits, graph_diag = sample_graphlet_clr_bridge_marginal(
                    endpoint.source_graphlet_logits,
                    endpoint.clean_graphlet_logits,
                    progress=float(progress),
                    sigma=diff_cfg.graphlet_sigma,
                    graphlet_basis=graphlet_basis,
                    coordinate_mask=endpoint.graphlet_coordinate_mask,
                    schedule=diff_cfg,
                    rng=rng,
                )
                current_prob = graphlet_clr_to_simplex(
                    current_logits,
                    graphlet_basis=graphlet_basis,
                    coordinate_mask=endpoint.graphlet_coordinate_mask,
                )
                graphlet_noise_rms.append(float(graph_diag["noise_rms"]))

            examples.append(
                TopologySpectralExample(
                    current_graph=endpoint.source,
                    time=float(progress),
                    current_spectrum=current_spectrum.astype(np.float32),
                    source_spectrum=endpoint.source_spectrum.astype(np.float32),
                    clean_spectrum_target=endpoint.clean_spectrum.astype(np.float32),
                    current_heat_kernel=(
                        None if current_heat_kernel is None else current_heat_kernel.astype(np.float32)
                    ),
                    source_heat_kernel=(
                        None if endpoint.source_heat_kernel is None else endpoint.source_heat_kernel.astype(np.float32)
                    ),
                    clean_heat_kernel_target=(
                        None if endpoint.clean_heat_kernel is None else endpoint.clean_heat_kernel.astype(np.float32)
                    ),
                    current_projector=(
                        None if current_projector is None else current_projector.astype(np.float32)
                    ),
                    source_projector=(
                        None if endpoint.source_projector is None else endpoint.source_projector.astype(np.float32)
                    ),
                    clean_projector_target=(
                        None if endpoint.clean_projector is None else endpoint.clean_projector.astype(np.float32)
                    ),
                    current_eigenspace_histogram=(
                        None if current_eigenspace_histogram is None else current_eigenspace_histogram.astype(np.float32)
                    ),
                    source_eigenspace_histogram=(
                        None if endpoint.source_eigenspace_histogram is None else endpoint.source_eigenspace_histogram.astype(np.float32)
                    ),
                    clean_eigenspace_histogram_target=(
                        None if endpoint.clean_eigenspace_histogram is None else endpoint.clean_eigenspace_histogram.astype(np.float32)
                    ),
                    eigenspace_histogram_block_mask=(
                        None if endpoint.eigenspace_histogram_block_mask is None else endpoint.eigenspace_histogram_block_mask.astype(np.bool_)
                    ),
                    eigenspace_histogram_block_weights=(
                        None if endpoint.eigenspace_histogram_block_weights is None else endpoint.eigenspace_histogram_block_weights.astype(np.float32)
                    ),
                    current_edge_logits=current_edge_logits,
                    source_edge_logits=(None if endpoint.source_edge_logits is None else endpoint.source_edge_logits.astype(np.float32)),
                    clean_edge_logits_target=(None if endpoint.clean_edge_logits is None else endpoint.clean_edge_logits.astype(np.float32)),
                    clean_edge_labels_target=(None if endpoint.clean_edge_labels is None else endpoint.clean_edge_labels.astype(np.int64)),
                    clean_clustering_histogram_target=(
                        None if endpoint.clean_clustering_histogram is None
                        else endpoint.clean_clustering_histogram.astype(np.float32)
                    ),
                    clean_orbit_summary_target=(
                        None if endpoint.clean_orbit_summary is None
                        else endpoint.clean_orbit_summary.astype(np.float32)
                    ),
                    clean_induced_graphlet_histogram_target=(
                        None if endpoint.clean_induced_graphlet_histogram is None
                        else endpoint.clean_induced_graphlet_histogram.astype(np.float32)
                    ),
                    clean_cycle_graphlet_histogram_target=(
                        None if endpoint.clean_cycle_graphlet_histogram is None
                        else endpoint.clean_cycle_graphlet_histogram.astype(np.float32)
                    ),
                    clean_clustering_coefficient_target=float(
                        endpoint.clean_clustering_coefficient
                    ),
                    current_graphlet_probabilities=(
                        None if current_prob is None else current_prob.astype(np.float32)
                    ),
                    source_graphlet_probabilities=(
                        None
                        if endpoint.source_graphlet_probabilities is None
                        else endpoint.source_graphlet_probabilities.astype(np.float32)
                    ),
                    clean_graphlet_probabilities_target=(
                        None
                        if endpoint.clean_graphlet_probabilities is None
                        else endpoint.clean_graphlet_probabilities.astype(np.float32)
                    ),
                    current_graphlet_logits=(
                        None if current_logits is None else current_logits.astype(np.float32)
                    ),
                    source_graphlet_logits=(
                        None
                        if endpoint.source_graphlet_logits is None
                        else endpoint.source_graphlet_logits.astype(np.float32)
                    ),
                    clean_graphlet_logits_target=(
                        None
                        if endpoint.clean_graphlet_logits is None
                        else endpoint.clean_graphlet_logits.astype(np.float32)
                    ),
                    graphlet_coordinate_mask=(
                        None
                        if endpoint.graphlet_coordinate_mask is None
                        else endpoint.graphlet_coordinate_mask.astype(np.bool_)
                    ),
                    base_generator=str(endpoint.metadata["base_generator"]),
                    source_index=int(endpoint.metadata["source_index"]),
                    target_index=int(endpoint.metadata["target_index"]),
                    matching_cost=float(endpoint.metadata["matching_cost"]),
                    trajectory_id=path,
                    step=local_index,
                )
            )

    diagnostics = {
        "format": "summary_diffusion_training_states_v1",
        "training_state_source": "continuous_summary_diffusion",
        "rewiring_used_for_training_states": False,
        "num_graphs": 1,
        "num_paths": paths_per_graph,
        "num_examples": len(examples),
        "samples_per_graph": samples_per_graph,
        "paths_per_graph": paths_per_graph,
        "bridge": diff_cfg.bridge,
        "graphlet_bridge": diff_cfg.resolved_graphlet_bridge,
        "schedule": diff_cfg.schedule,
        "spectral_sigma": diff_cfg.spectral_sigma,
        "heat_kernel_sigma": (diff_cfg.heat_kernel_sigma if endpoint.source_heat_kernel is not None else None),
        "projector_sigma": (diff_cfg.projector_sigma if endpoint.source_projector is not None else None),
        "eigenspace_histogram_sigma": (
            diff_cfg.eigenspace_histogram_sigma if endpoint.source_eigenspace_histogram is not None else None
        ),
        "graphlet_sigma": diff_cfg.graphlet_sigma if graphlet_basis is not None else None,
        "preserve_spectral_trace": diff_cfg.preserve_spectral_trace,
        "fix_spectral_lambda1": diff_cfg.fix_spectral_lambda1,
        "mean_spectral_noise_rms": float(np.mean(spectral_noise_rms)) if spectral_noise_rms else 0.0,
        "mean_heat_kernel_noise_rms": float(np.mean(heat_kernel_noise_rms)) if heat_kernel_noise_rms else 0.0,
        "mean_projector_noise_rms": float(np.mean(projector_noise_rms)) if projector_noise_rms else 0.0,
        "mean_eigenspace_histogram_noise_rms": (
            float(np.mean(eigenspace_histogram_noise_rms)) if eigenspace_histogram_noise_rms else 0.0
        ),
        "projector_alignment_cost": float(endpoint.projector_alignment_cost),
        "projector_rank": int(endpoint.projector_rank),
        "spectral_representation": (
            "heat_kernel" if endpoint.source_heat_kernel is not None
            else (
                "lambda_projector" if endpoint.source_projector is not None
                else (
                    "lambda_eigenspace_histogram"
                    if endpoint.source_eigenspace_histogram is not None else "eigenvalues"
                )
            )
        ),
        "mean_graphlet_noise_rms": float(np.mean(graphlet_noise_rms)) if graphlet_noise_rms else 0.0,
        "mean_edge_noise_rms": float(np.mean(edge_noise_rms)) if edge_noise_rms else 0.0,
        "edge_diffusion_enabled": endpoint.source_edge_logits is not None,
        "mean_endpoint_spectral_discrepancy": float(endpoint.spectral_endpoint_distance),
        "source_modes": [str(endpoint.metadata["source_mode"])],
        "base_generators": [str(endpoint.metadata["base_generator"])],
    }
    return examples, diagnostics


def build_spectral_diffusion_examples(
    graphs: Sequence[nx.Graph | TopologyTrainingPair],
    *,
    diffusion_config: dict[str, Any] | None = None,
    source_config: dict[str, Any] | None = None,
    spectral_config: dict[str, Any] | None = None,
    graphlet_basis: TopologyGraphletBasis | None = None,
    graphlet_logit_epsilon: float = 1.0e-5,
    seed: int = 0,
    structure_summary_config: dict[str, Any] | None = None,
    edge_diffusion_config: dict[str, Any] | None = None,
) -> tuple[list[TopologySpectralExample], dict[str, Any]]:
    """Sample eager continuous summary-diffusion states.

    The eager and streaming paths intentionally share the same endpoint and
    bridge helpers. This keeps heat-kernel and legacy eigenvalue diffusion
    bit-for-bit consistent at the level of their stochastic state definition.
    """

    diff_values = dict(diffusion_config or {})
    source_cfg = dict(source_config or {})
    spec_cfg = dict(spectral_config or {})
    require_same_degree_sequence = bool(spec_cfg.get("require_same_degree_sequence", True))
    rng = np.random.default_rng(int(seed))
    examples: list[TopologySpectralExample] = []
    reports: list[dict[str, Any]] = []
    for graph_index, raw_item in enumerate(graphs):
        endpoint = _prepare_spectral_diffusion_endpoint(
            raw_item,
            source_config=source_cfg,
            spectral_config=spec_cfg,
            graphlet_basis=graphlet_basis,
            graphlet_logit_epsilon=graphlet_logit_epsilon,
            require_same_degree_sequence=require_same_degree_sequence,
            rng=rng,
            structure_summary_config=structure_summary_config,
            edge_diffusion_config=edge_diffusion_config,
        )
        block, report = _sample_spectral_diffusion_endpoint_examples(
            endpoint,
            diffusion_config=diff_values,
            graphlet_basis=graphlet_basis,
            seed=int(seed) + 10_007 * int(graph_index),
        )
        examples.extend(block)
        reports.append(report)

    diff_cfg = SummaryDiffusionConfig.from_dict(diff_values)
    representation = str(spec_cfg.get("representation", "eigenvalues")).lower()
    if representation in {"heat", "heatkernel", "heat-kernel"}:
        representation = "heat_kernel"
    if representation in {"eigenspace", "lambda_projector", "eigenvalues_projector", "lambda+p", "lambda_p"}:
        representation = "lambda_projector"
    if representation in {
        "eigenspace_histogram", "lambda_eigenspace_histogram",
        "degree_spectral_histogram", "spectral_distance_histogram",
    }:
        representation = "lambda_eigenspace_histogram"
    diagnostics = {
        "format": "summary_diffusion_training_states_v2",
        "training_state_source": "continuous_summary_diffusion",
        "rewiring_used_for_training_states": False,
        "spectral_representation": representation,
        "num_graphs": len(graphs),
        "num_paths": sum(int(row.get("num_paths", 0)) for row in reports),
        "num_examples": len(examples),
        "samples_per_graph": max(int(diff_values.get("samples_per_graph", 32)), 1),
        "paths_per_graph": max(int(diff_values.get("paths_per_graph", 1)), 1),
        "bridge": diff_cfg.bridge,
        "graphlet_bridge": diff_cfg.resolved_graphlet_bridge,
        "schedule": diff_cfg.schedule,
        "spectral_sigma": diff_cfg.spectral_sigma,
        "heat_kernel_sigma": diff_cfg.heat_kernel_sigma if representation == "heat_kernel" else None,
        "projector_sigma": diff_cfg.projector_sigma if representation == "lambda_projector" else None,
        "eigenspace_histogram_sigma": (
            diff_cfg.eigenspace_histogram_sigma if representation == "lambda_eigenspace_histogram" else None
        ),
        "graphlet_sigma": diff_cfg.graphlet_sigma if graphlet_basis is not None else None,
        "preserve_spectral_trace": diff_cfg.preserve_spectral_trace,
        "fix_spectral_lambda1": diff_cfg.fix_spectral_lambda1,
        "mean_spectral_noise_rms": float(np.mean([row.get("mean_spectral_noise_rms", 0.0) for row in reports])) if reports else 0.0,
        "mean_heat_kernel_noise_rms": float(np.mean([row.get("mean_heat_kernel_noise_rms", 0.0) for row in reports])) if reports else 0.0,
        "mean_projector_noise_rms": float(np.mean([row.get("mean_projector_noise_rms", 0.0) for row in reports])) if reports else 0.0,
        "mean_projector_alignment_cost": float(np.mean([row.get("projector_alignment_cost", 0.0) for row in reports])) if reports else 0.0,
        "mean_eigenspace_histogram_noise_rms": float(np.mean([row.get("mean_eigenspace_histogram_noise_rms", 0.0) for row in reports])) if reports else 0.0,
        "mean_graphlet_noise_rms": float(np.mean([row.get("mean_graphlet_noise_rms", 0.0) for row in reports])) if reports else 0.0,
        "mean_edge_noise_rms": float(np.mean([row.get("mean_edge_noise_rms", 0.0) for row in reports])) if reports else 0.0,
        "edge_diffusion_enabled": bool((edge_diffusion_config or {}).get("enabled", False)),
        "mean_endpoint_spectral_discrepancy": float(np.mean([row.get("mean_endpoint_spectral_discrepancy", 0.0) for row in reports])) if reports else 0.0,
        "source_modes": sorted({mode for row in reports for mode in row.get("source_modes", [])}),
        "base_generators": sorted({mode for row in reports for mode in row.get("base_generators", [])}),
    }
    return examples, diagnostics


class TopologySpectralDiffusionIterableDataset(torch.utils.data.IterableDataset):
    """Resample stochastic continuous summary-diffusion states every epoch."""

    def __init__(
        self,
        graphs: Sequence[nx.Graph | TopologyTrainingPair],
        *,
        diffusion_config: dict[str, Any] | None = None,
        source_config: dict[str, Any] | None = None,
        spectral_config: dict[str, Any] | None = None,
        graphlet_basis: TopologyGraphletBasis | None = None,
        graphlet_logit_epsilon: float = 1.0e-5,
        seed: int = 0,
        shuffle_graphs: bool = True,
        structure_summary_config: dict[str, Any] | None = None,
        edge_diffusion_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.graphs = tuple(graphs)
        self.diffusion_config = dict(diffusion_config or {})
        self.source_config = dict(source_config or {})
        self.spectral_config = dict(spectral_config or {})
        self.structure_summary_config = dict(structure_summary_config or {})
        self.edge_diffusion_config = dict(edge_diffusion_config or {})
        clustering_histogram_bins(self.structure_summary_config)
        orbit_summary_width(self.structure_summary_config)
        cycle_graphlet_k(self.structure_summary_config)
        InducedGraphletSpec.from_config(self.structure_summary_config)
        self.graphlet_basis = graphlet_basis
        self.graphlet_logit_epsilon = float(graphlet_logit_epsilon)
        self.seed = int(seed)
        self.shuffle_graphs = bool(shuffle_graphs)
        self.epoch = 0
        self.last_diagnostics: list[dict[str, Any]] = []
        requested_cache = bool(self.diffusion_config.get("cache_endpoints", True))
        # Structural source randomization intentionally changes the HH
        # realization, so only cache when source_randomization_steps=0.  A
        # random relabel alone is harmless because all endpoint summaries are
        # permutation invariant and the graph encoder pools equivariantly.
        self.cache_endpoints = requested_cache and int(
            self.source_config.get("source_randomization_steps", 0)
        ) == 0
        self._endpoint_cache: tuple[TopologySpectralDiffusionEndpoint, ...] | None = None
        if self.cache_endpoints:
            require_same_degree_sequence = bool(
                self.spectral_config.get("require_same_degree_sequence", True)
            )
            prepared: list[TopologySpectralDiffusionEndpoint] = []
            for graph_index, raw_item in enumerate(self.graphs):
                prepared.append(
                    _prepare_spectral_diffusion_endpoint(
                        raw_item,
                        source_config=self.source_config,
                        structure_summary_config=self.structure_summary_config,
                        spectral_config=self.spectral_config,
                        graphlet_basis=self.graphlet_basis,
                        graphlet_logit_epsilon=self.graphlet_logit_epsilon,
                        require_same_degree_sequence=require_same_degree_sequence,
                        rng=np.random.default_rng(self.seed + 10_007 * graph_index),
                        edge_diffusion_config=self.edge_diffusion_config,
                    )
                )
            self._endpoint_cache = tuple(prepared)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    @property
    def estimated_examples(self) -> int:
        return (
            len(self.graphs)
            * max(int(self.diffusion_config.get("samples_per_graph", 32)), 1)
            * max(int(self.diffusion_config.get("paths_per_graph", 1)), 1)
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
            sample_seed = (
                self.seed
                + 1_000_003 * self.epoch
                + 10_007 * int(graph_index)
                + position
            )
            if self._endpoint_cache is not None:
                examples, diagnostics = _sample_spectral_diffusion_endpoint_examples(
                    self._endpoint_cache[int(graph_index)],
                    diffusion_config={**self.diffusion_config, "edge_sigma": float(self.edge_diffusion_config.get("sigma",1.0))},
                    graphlet_basis=self.graphlet_basis,
                    seed=sample_seed,
                )
                diagnostics = dict(diagnostics)
                diagnostics["endpoint_cache"] = True
            else:
                examples, diagnostics = build_spectral_diffusion_examples(
                    [self.graphs[int(graph_index)]],
                    diffusion_config=self.diffusion_config,
                    source_config=self.source_config,
                    structure_summary_config=self.structure_summary_config,
                    spectral_config=self.spectral_config,
                    graphlet_basis=self.graphlet_basis,
                    graphlet_logit_epsilon=self.graphlet_logit_epsilon,
                    seed=sample_seed,
                    edge_diffusion_config=self.edge_diffusion_config,
                )
                diagnostics = dict(diagnostics)
                diagnostics["endpoint_cache"] = False
            if worker is None:
                self.last_diagnostics.append(diagnostics)
            yield from examples

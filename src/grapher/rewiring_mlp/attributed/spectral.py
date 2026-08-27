from __future__ import annotations

from typing import Any, Mapping, Sequence

import networkx as nx
import numpy as np

from grapher.rewiring_mlp.generic.spectral import spectral_distance
from grapher.rewiring_mlp.molecular.constraints import bond_order

SPECTRAL_CHANNELS = ("topology", "bond_weighted")


def normalize_attributed_graph(graph: nx.Graph) -> nx.Graph:
    """Return a simple integer-labelled graph while preserving attributes."""

    if graph.is_directed() or graph.is_multigraph():
        raise ValueError("Attributed spectral GraphER requires a simple undirected graph.")
    normalized = nx.convert_node_labels_to_integers(
        nx.Graph(graph),
        first_label=0,
        ordering="sorted",
        label_attribute="_original_node",
    )
    if nx.number_of_selfloops(normalized):
        raise ValueError("Attributed spectral GraphER does not support self-loops.")
    return normalized


def _laplacian_eigenvalues_from_weight_matrix(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise ValueError("Expected a square symmetric weight matrix.")
    if weights.size == 0:
        return np.zeros(0, dtype=np.float64)
    weights = 0.5 * (weights + weights.T)
    np.fill_diagonal(weights, 0.0)
    degrees = weights.sum(axis=1)
    laplacian = np.diag(degrees) - weights
    values = np.linalg.eigvalsh(laplacian)
    values[np.abs(values) < 1.0e-10] = 0.0
    return np.sort(values.astype(np.float64))


def attributed_laplacian_spectra(
    graph: nx.Graph,
    *,
    edge_attribute: str = "bond_type",
) -> np.ndarray:
    """Return topology and bond-order-weighted Laplacian spectra.

    Output shape is ``[2, n]`` with channels ``topology`` and
    ``bond_weighted``.  Strict same-bond-type rewiring preserves the trace of
    both channels: ordinary degree fixes the first and typed degree fixes the
    second.
    """

    graph = normalize_attributed_graph(graph)
    n = graph.number_of_nodes()
    topology = np.zeros((n, n), dtype=np.float64)
    weighted = np.zeros((n, n), dtype=np.float64)
    for u, v, data in graph.edges(data=True):
        topology[int(u), int(v)] = topology[int(v), int(u)] = 1.0
        if edge_attribute not in data:
            raise KeyError(f"Edge {(u, v)!r} is missing {edge_attribute!r}.")
        weight = float(bond_order(int(data[edge_attribute])))
        weighted[int(u), int(v)] = weighted[int(v), int(u)] = weight
    return np.stack(
        [
            _laplacian_eigenvalues_from_weight_matrix(topology),
            _laplacian_eigenvalues_from_weight_matrix(weighted),
        ],
        axis=0,
    )


def attributed_spectral_scales(
    graph: nx.Graph,
    *,
    mode: str = "mean_degree",
    edge_attribute: str = "bond_type",
) -> np.ndarray:
    graph = normalize_attributed_graph(graph)
    spectra = attributed_laplacian_spectra(graph, edge_attribute=edge_attribute)
    traces = spectra.sum(axis=1)
    name = str(mode).lower()
    if name in {"none", "raw"}:
        return np.ones(2, dtype=np.float64)
    if name in {"trace", "degree_sum"}:
        return np.maximum(traces, 1.0)
    if name not in {"mean_degree", "average_degree", "avg_degree"}:
        raise ValueError("spectral normalization must be mean_degree, trace, or none.")
    return np.maximum(traces / max(graph.number_of_nodes(), 1), 1.0e-8)


def attributed_spectral_distance(
    left: Sequence[Sequence[float]] | np.ndarray,
    right: Sequence[Sequence[float]] | np.ndarray,
    *,
    scales: Sequence[float] | np.ndarray,
    metric: str = "rmse",
    channel_weights: Sequence[float] | np.ndarray = (1.0, 1.0),
    low_frequency_weight: float = 1.0,
    low_frequency_cutoff: int = 0,
) -> tuple[float, np.ndarray]:
    """Weighted dual-channel spectral distance and per-channel values."""

    left_values = np.asarray(left, dtype=np.float64)
    right_values = np.asarray(right, dtype=np.float64)
    if left_values.shape != right_values.shape or left_values.ndim != 2:
        raise ValueError("Attributed spectra must have matching [channels, n] shape.")
    if left_values.shape[0] != 2:
        raise ValueError("Attributed spectral GraphER expects two spectral channels.")
    scales_array = np.asarray(scales, dtype=np.float64).reshape(-1)
    if isinstance(channel_weights, Mapping):
        channel_weights = (
            channel_weights.get("topology", 1.0),
            channel_weights.get(
                "bond_weighted", channel_weights.get("bond", 1.0)
            ),
        )
    weights = np.asarray(channel_weights, dtype=np.float64).reshape(-1)
    if scales_array.size != 2 or weights.size != 2:
        raise ValueError("Expected two spectral scales and two channel weights.")
    if np.any(weights < 0.0) or not np.isfinite(weights).all() or float(weights.sum()) <= 0.0:
        raise ValueError("Spectral channel weights must be finite, nonnegative, and nonzero.")
    values = np.asarray(
        [
            spectral_distance(
                left_values[channel],
                right_values[channel],
                metric=metric,
                scale=float(scales_array[channel]),
                low_frequency_weight=low_frequency_weight,
                low_frequency_cutoff=low_frequency_cutoff,
            )
            for channel in range(2)
        ],
        dtype=np.float64,
    )
    return float(np.dot(weights, values) / float(weights.sum())), values


def attributed_spectrum_moments(spectra: np.ndarray) -> dict[str, list[float]]:
    values = np.asarray(spectra, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != 2:
        raise ValueError("Expected spectra with shape [2, n].")
    return {
        "trace": [float(block.sum()) for block in values],
        "second_moment": [float(np.square(block).sum()) for block in values],
    }

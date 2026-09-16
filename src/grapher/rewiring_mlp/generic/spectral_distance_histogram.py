"""Permutation-invariant low-frequency eigenspace summaries.

The raw Laplacian eigenspace is node indexed.  This module removes that
correspondence dependence by converting the first ``rank`` non-trivial
Laplacian eigenvectors into pairwise spectral-embedding distances and then
aggregating those distances inside unordered endpoint-degree blocks.

For ``U_k = [u_2, ..., u_{k+1}]`` and ``P_k = U_k U_k^T``, the squared
spectral distance between nodes i,j is

    d_ij^2 = ||U_k[i]-U_k[j]||^2
           = P_ii + P_jj - 2 P_ij.

Distances are normalized by the graph-size/rank-dependent RMS scale
``sqrt(2 k / (n-1))``.  If the requested rank cuts through a repeated
eigenvalue block, the complete block is included and ``k`` is the resulting
effective rank.  Histograms are therefore comparable across graph
sizes while remaining invariant to node permutations, eigenvector signs and
orthogonal rotations inside the selected eigenspace.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

import networkx as nx
import numpy as np


@dataclass(frozen=True)
class SpectralDistanceHistogramSpec:
    rank: int = 4
    bins: int = 16
    degree_max: int = 19
    max_normalized_distance: float = 3.0

    def __post_init__(self) -> None:
        if isinstance(self.rank, bool) or int(self.rank) != self.rank or int(self.rank) < 1:
            raise ValueError("spectral-distance histogram rank must be an integer >= 1.")
        if isinstance(self.bins, bool) or int(self.bins) != self.bins or int(self.bins) < 2:
            raise ValueError("spectral-distance histogram bins must be an integer >= 2.")
        if isinstance(self.degree_max, bool) or int(self.degree_max) != self.degree_max or int(self.degree_max) < 0:
            raise ValueError("spectral-distance histogram degree_max must be a nonnegative integer.")
        if not np.isfinite(self.max_normalized_distance) or float(self.max_normalized_distance) <= 0.0:
            raise ValueError("spectral-distance histogram max_normalized_distance must be finite and positive.")

    @property
    def degree_pairs(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (a, b)
            for a in range(int(self.degree_max) + 1)
            for b in range(a, int(self.degree_max) + 1)
        )

    @property
    def num_blocks(self) -> int:
        d = int(self.degree_max) + 1
        return d * (d + 1) // 2

    @property
    def width(self) -> int:
        return self.num_blocks * int(self.bins)

    @property
    def bin_width(self) -> float:
        return float(self.max_normalized_distance) / float(self.bins)

    def metadata(self) -> dict[str, Any]:
        payload = {
            "version": "degree_conditioned_spectral_distance_histogram_v2",
            "rank": int(self.rank),
            "bins": int(self.bins),
            "degree_max": int(self.degree_max),
            "max_normalized_distance": float(self.max_normalized_distance),
            "normalization": "expected_pair_rms_sqrt_2k_over_n_minus_1",
            "distance": "euclidean_low_frequency_embedding",
            "rank_policy": "requested_rank_plus_complete_cutoff_degenerate_block",
            "degeneracy_tolerance": "numpy_isclose_rtol_1e-8_atol_1e-10",
            "block_conditioning": "unordered_endpoint_degree_pair",
            "num_blocks": int(self.num_blocks),
            "width": int(self.width),
        }
        payload["fingerprint"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return payload

    @classmethod
    def from_config(cls, data: dict[str, Any] | None) -> "SpectralDistanceHistogramSpec":
        values = dict(data or {})
        return cls(
            rank=int(values.get("eigenspace_rank", values.get("projector_rank", 4))),
            bins=int(values.get("eigenspace_histogram_bins", 16)),
            degree_max=int(values.get("eigenspace_histogram_degree_max", 19)),
            max_normalized_distance=float(values.get("eigenspace_histogram_max_distance", 3.0)),
        )


def effective_rank(num_nodes: int, rank: int) -> int:
    n = int(num_nodes)
    if n <= 1:
        return 0
    return min(int(rank), n - 1)


def _laplacian_embedding(graph: nx.Graph, rank: int) -> tuple[np.ndarray, int]:
    n = graph.number_of_nodes()
    requested = effective_rank(n, rank)
    if n == 0 or requested == 0:
        return np.zeros((n, 0), dtype=np.float64), requested
    nodes = list(range(n))
    adjacency = nx.to_numpy_array(graph, nodelist=nodes, weight=None, dtype=np.float64)
    degree = np.diag(adjacency.sum(axis=1))
    laplacian = degree - adjacency
    values, vectors = np.linalg.eigh(laplacian)
    # Column zero is the constant mode for the connected graphs used by GraphER.
    # A fixed-rank cut can otherwise split an exactly repeated eigenvalue block,
    # making the selected subspace depend on the arbitrary basis returned by the
    # eigensolver.  Include the *entire* degenerate block at the cutoff so the
    # resulting subspace geometry is invariant to rotations inside it.
    end = requested + 1  # exclusive column index after requested nontrivial modes
    cutoff = float(values[end - 1])
    while end < n and np.isclose(values[end], cutoff, rtol=1.0e-8, atol=1.0e-10):
        end += 1
    effective = end - 1
    return np.asarray(vectors[:, 1:end], dtype=np.float64), effective


def normalized_spectral_distance_matrix(graph: nx.Graph, *, rank: int) -> np.ndarray:
    """Return normalized all-pairs distances in the low-frequency embedding."""
    n = graph.number_of_nodes()
    embedding, k = _laplacian_embedding(graph, rank)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)
    if k == 0:
        return np.zeros((n, n), dtype=np.float64)
    squared_norm = np.sum(embedding * embedding, axis=1)
    squared = squared_norm[:, None] + squared_norm[None, :] - 2.0 * (embedding @ embedding.T)
    squared = np.maximum(squared, 0.0)
    distances = np.sqrt(squared)
    # Identity for an orthonormal k-dimensional non-trivial Laplacian embedding:
    # sum_{i<j} ||z_i-z_j||^2 = n k, hence E[d^2]=2k/(n-1).
    scale = math.sqrt(2.0 * float(k) / float(max(n - 1, 1)))
    if scale > 0.0:
        distances /= scale
    np.fill_diagonal(distances, 0.0)
    return distances


def _pair_to_block(spec: SpectralDistanceHistogramSpec) -> dict[tuple[int, int], int]:
    return {pair: index for index, pair in enumerate(spec.degree_pairs)}


def degree_pair_weights(graph: nx.Graph, spec: SpectralDistanceHistogramSpec) -> np.ndarray:
    """Fraction of unordered node pairs in every degree block."""
    counts = np.zeros(spec.num_blocks, dtype=np.float64)
    mapping = _pair_to_block(spec)
    degrees = [int(graph.degree(i)) for i in range(graph.number_of_nodes())]
    for degree in degrees:
        if degree < 0 or degree > spec.degree_max:
            raise ValueError(
                f"Graph degree {degree} exceeds eigenspace_histogram_degree_max={spec.degree_max}."
            )
    for i in range(len(degrees)):
        for j in range(i + 1, len(degrees)):
            a, b = sorted((degrees[i], degrees[j]))
            counts[mapping[(a, b)]] += 1.0
    total = counts.sum()
    return counts / total if total > 0.0 else counts


def extract_degree_conditioned_spectral_histogram(
    graph: nx.Graph,
    spec: SpectralDistanceHistogramSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return flattened block histograms, active-block mask, and pair weights.

    Each nonempty degree-pair block is normalized independently to a probability
    simplex.  ``block_weights`` stores its fraction of all unordered node pairs;
    this is fixed by the degree sequence and is used to aggregate W1 distances.
    """
    n = graph.number_of_nodes()
    histogram = np.zeros((spec.num_blocks, spec.bins), dtype=np.float64)
    counts = np.zeros(spec.num_blocks, dtype=np.float64)
    mapping = _pair_to_block(spec)
    if n <= 1:
        return histogram.reshape(-1), counts.astype(bool), counts

    distances = normalized_spectral_distance_matrix(graph, rank=spec.rank)
    degrees = [int(graph.degree(i)) for i in range(n)]
    for degree in degrees:
        if degree < 0 or degree > spec.degree_max:
            raise ValueError(
                f"Graph degree {degree} exceeds eigenspace_histogram_degree_max={spec.degree_max}."
            )
    max_distance = float(spec.max_normalized_distance)
    bins = int(spec.bins)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = sorted((degrees[i], degrees[j]))
            block = mapping[(a, b)]
            value = float(np.clip(distances[i, j], 0.0, np.nextafter(max_distance, 0.0)))
            bin_index = min(int(value / max_distance * bins), bins - 1)
            histogram[block, bin_index] += 1.0
            counts[block] += 1.0
    active = counts > 0.0
    histogram[active] /= counts[active, None]
    total = counts.sum()
    weights = counts / total if total > 0.0 else counts
    return histogram.reshape(-1), active, weights


def validate_histogram(
    histogram: np.ndarray,
    spec: SpectralDistanceHistogramSpec,
    *,
    block_mask: np.ndarray | None = None,
    atol: float = 1.0e-5,
) -> np.ndarray:
    values = np.asarray(histogram, dtype=np.float64)
    if values.shape != (spec.width,):
        raise ValueError(f"Spectral-distance histogram must have width {spec.width}, got {values.shape}.")
    if not np.isfinite(values).all() or np.any(values < -atol):
        raise ValueError("Spectral-distance histogram contains invalid values.")
    blocks = values.reshape(spec.num_blocks, spec.bins)
    if block_mask is not None:
        mask = np.asarray(block_mask, dtype=bool)
        if mask.shape != (spec.num_blocks,):
            raise ValueError("Spectral-distance histogram block mask has the wrong shape.")
    else:
        mask = blocks.sum(axis=1) > atol
    if np.any(np.abs(blocks[mask].sum(axis=1) - 1.0) > atol):
        raise ValueError("Every active spectral-distance histogram block must sum to one.")
    if np.any(np.abs(blocks[~mask]) > atol):
        raise ValueError("Inactive spectral-distance histogram blocks must be zero.")
    return values


def spectral_histogram_wasserstein(
    first: np.ndarray,
    second: np.ndarray,
    spec: SpectralDistanceHistogramSpec,
    *,
    block_weights: np.ndarray,
) -> float:
    """Degree-pair-weighted one-dimensional W1 over normalized-distance bins."""
    a = np.asarray(first, dtype=np.float64).reshape(spec.num_blocks, spec.bins)
    b = np.asarray(second, dtype=np.float64).reshape(spec.num_blocks, spec.bins)
    weights = np.asarray(block_weights, dtype=np.float64)
    if weights.shape != (spec.num_blocks,):
        raise ValueError("Spectral-distance histogram block weights have the wrong shape.")
    active = weights > 0.0
    if not np.any(active):
        return 0.0
    # For equal-width 1-D bins, integral |CDF_a-CDF_b| dx.
    cdf_delta = np.abs(np.cumsum(a, axis=1) - np.cumsum(b, axis=1))
    per_block = cdf_delta.sum(axis=1) * spec.bin_width
    return float(np.sum(weights[active] * per_block[active]) / max(weights[active].sum(), 1.0e-12))


def histogram_rmse(first: np.ndarray, second: np.ndarray, spec: SpectralDistanceHistogramSpec) -> float:
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if a.shape != (spec.width,) or b.shape != (spec.width,):
        raise ValueError("Spectral-distance histogram RMSE received an invalid width.")
    return float(np.sqrt(np.mean(np.square(a - b))))

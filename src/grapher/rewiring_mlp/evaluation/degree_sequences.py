from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from grapher.rewiring_mlp.evaluation.metrics import mmd_gaussian_emd


def _normalise(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    total = float(values.sum())
    if total <= 0.0:
        return np.zeros_like(values, dtype=np.float64)
    return values / total


def degree_histogram_matrix(
    sequences: Sequence[Sequence[int]],
    *,
    max_degree: int | None = None,
) -> np.ndarray:
    sequences = list(sequences)
    if max_degree is None:
        max_degree = max(
            (max((int(d) for d in sequence), default=0) for sequence in sequences),
            default=0,
        )
    width = max(int(max_degree) + 1, 1)
    rows = np.zeros((len(sequences), width), dtype=np.float64)
    for row, sequence in zip(rows, sequences):
        for degree in sequence:
            degree = int(degree)
            if 0 <= degree < width:
                row[degree] += 1.0
        total = float(row.sum())
        if total > 0.0:
            row /= total
    return rows


def aggregate_degree_distribution(
    sequences: Sequence[Sequence[int]],
    *,
    max_degree: int,
) -> np.ndarray:
    matrix = degree_histogram_matrix(sequences, max_degree=max_degree)
    if matrix.shape[0] == 0:
        return np.zeros(max(int(max_degree) + 1, 1), dtype=np.float64)
    counts = np.zeros(matrix.shape[1], dtype=np.float64)
    for sequence in sequences:
        for degree in sequence:
            degree = int(degree)
            if 0 <= degree < counts.size:
                counts[degree] += 1.0
    return _normalise(counts)


def kl_reference_to_candidate(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    epsilon: float = 1.0e-8,
) -> float:
    """Return KL(P_reference || P_candidate) with additive smoothing."""

    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    candidate = np.asarray(candidate, dtype=np.float64).reshape(-1)
    width = max(reference.size, candidate.size, 1)
    p = np.full(width, float(epsilon), dtype=np.float64)
    q = np.full(width, float(epsilon), dtype=np.float64)
    p[: reference.size] += reference
    q[: candidate.size] += candidate
    p = _normalise(p)
    q = _normalise(q)
    return float(np.sum(p * np.log(p / q)))


def jensen_shannon_divergence(
    left: np.ndarray,
    right: np.ndarray,
    *,
    epsilon: float = 1.0e-8,
) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    width = max(left.size, right.size, 1)
    p = np.full(width, float(epsilon), dtype=np.float64)
    q = np.full(width, float(epsilon), dtype=np.float64)
    p[: left.size] += left
    q[: right.size] += right
    p = _normalise(p)
    q = _normalise(q)
    middle = 0.5 * (p + q)
    return float(
        0.5 * np.sum(p * np.log(p / middle)) + 0.5 * np.sum(q * np.log(q / middle))
    )


def discrete_total_variation(
    reference_values: Sequence[int],
    candidate_values: Sequence[int],
) -> float:
    values = list(reference_values) + list(candidate_values)
    width = max(values, default=0) + 1
    p = np.bincount(
        np.asarray(reference_values, dtype=np.int64), minlength=width
    ).astype(np.float64)
    q = np.bincount(
        np.asarray(candidate_values, dtype=np.int64), minlength=width
    ).astype(np.float64)
    return float(0.5 * np.abs(_normalise(p) - _normalise(q)).sum())


def median_rbf_bandwidth(*matrices: np.ndarray) -> float:
    nonempty = [
        np.asarray(matrix, dtype=np.float64)
        for matrix in matrices
        if np.asarray(matrix).size
    ]
    if not nonempty:
        return 1.0
    values = np.vstack(nonempty)
    if values.shape[0] < 2:
        return 1.0
    squared = np.sum(
        (values[:, None, :] - values[None, :, :]) ** 2,
        axis=-1,
    )
    upper = squared[np.triu_indices(values.shape[0], k=1)]
    upper = upper[upper > 0.0]
    if upper.size == 0:
        return 1.0
    return float(np.sqrt(0.5 * np.median(upper))) or 1.0


def median_emd_bandwidth(*matrices: np.ndarray) -> float:
    """Return one median Earth-Mover bandwidth shared by all comparisons."""

    nonempty = [
        np.asarray(matrix, dtype=np.float64)
        for matrix in matrices
        if np.asarray(matrix).size
    ]
    if not nonempty:
        return 1.0
    values = np.vstack(nonempty)
    if values.shape[0] < 2:
        return 1.0
    distances = np.sum(
        np.abs(
            np.cumsum(
                values[:, None, :] - values[None, :, :],
                axis=-1,
            )
        ),
        axis=-1,
    )
    upper = distances[np.triu_indices(values.shape[0], k=1)]
    upper = upper[upper > 0.0]
    return float(np.median(upper)) if upper.size else 1.0


def evaluate_degree_sequence_sets(
    reference: Sequence[Sequence[int]],
    candidate: Sequence[Sequence[int]],
    *,
    train: Sequence[Sequence[int]] | None = None,
    degree_mmd_sigma: float | None = None,
) -> dict[str, Any]:
    reference = [sorted([int(d) for d in seq], reverse=True) for seq in reference]
    candidate = [sorted([int(d) for d in seq], reverse=True) for seq in candidate]
    train = (
        [sorted([int(d) for d in seq], reverse=True) for seq in train]
        if train is not None
        else None
    )

    all_sequences = reference + candidate + (train or [])
    max_degree = max(
        (max(sequence, default=0) for sequence in all_sequences),
        default=0,
    )
    ref_hist = degree_histogram_matrix(reference, max_degree=max_degree)
    candidate_hist = degree_histogram_matrix(candidate, max_degree=max_degree)
    train_hist = (
        degree_histogram_matrix(train, max_degree=max_degree)
        if train is not None
        else None
    )

    if degree_mmd_sigma is None:
        degree_mmd_sigma = median_emd_bandwidth(
            ref_hist,
            train_hist if train_hist is not None else candidate_hist,
        )

    ref_marginal = aggregate_degree_distribution(reference, max_degree=max_degree)
    candidate_marginal = aggregate_degree_distribution(candidate, max_degree=max_degree)
    node_reference = [len(sequence) for sequence in reference]
    node_candidate = [len(sequence) for sequence in candidate]
    edge_reference = [sum(sequence) // 2 for sequence in reference]
    edge_candidate = [sum(sequence) // 2 for sequence in candidate]

    def _mean_degree(sequence: Sequence[int]) -> float:
        return float(sum(sequence) / len(sequence)) if sequence else 0.0

    def _degree_variance(sequence: Sequence[int]) -> float:
        return float(np.var(np.asarray(sequence, dtype=np.float64))) if sequence else 0.0

    def _degree_second_moment(sequence: Sequence[int]) -> float:
        if not sequence:
            return 0.0
        values = np.asarray(sequence, dtype=np.float64)
        return float(np.mean(values * values))

    def _wedge_count(sequence: Sequence[int]) -> int:
        return int(sum(int(d) * (int(d) - 1) // 2 for d in sequence))

    mean_degree_reference = [_mean_degree(sequence) for sequence in reference]
    mean_degree_candidate = [_mean_degree(sequence) for sequence in candidate]
    degree_variance_reference = [_degree_variance(sequence) for sequence in reference]
    degree_variance_candidate = [_degree_variance(sequence) for sequence in candidate]
    second_moment_reference = [_degree_second_moment(sequence) for sequence in reference]
    second_moment_candidate = [_degree_second_moment(sequence) for sequence in candidate]
    max_degree_reference = [max(sequence, default=0) for sequence in reference]
    max_degree_candidate = [max(sequence, default=0) for sequence in candidate]
    wedge_reference = [_wedge_count(sequence) for sequence in reference]
    wedge_candidate = [_wedge_count(sequence) for sequence in candidate]

    candidate_keys = [tuple(sequence) for sequence in candidate]
    train_keys = set(tuple(sequence) for sequence in (train or []))
    reference_keys = set(tuple(sequence) for sequence in reference)

    metrics: dict[str, Any] = {
        "num_candidate_sequences": len(candidate),
        "num_reference_sequences": len(reference),
        # Historical GraphRNN/GDSS/HOG-Diff generic-graph degree MMD.
        # Keep this fixed at sigma=1.0 so the DH-VAE diagnostic is directly
        # comparable to evaluate_graph_generation_report.py.
        "degree_histogram_mmd_graphrnn": mmd_gaussian_emd(
            ref_hist, candidate_hist, sigma=1.0
        ),
        # Adaptive-bandwidth diagnostic retained for backward compatibility
        # and degree-prior model selection.
        "degree_histogram_mmd": mmd_gaussian_emd(
            ref_hist, candidate_hist, sigma=degree_mmd_sigma
        ),
        "degree_marginal_kl_reference_to_candidate": (
            kl_reference_to_candidate(ref_marginal, candidate_marginal)
        ),
        "degree_marginal_js": jensen_shannon_divergence(
            ref_marginal, candidate_marginal
        ),
        "node_count_total_variation": discrete_total_variation(
            node_reference, node_candidate
        ),
        "edge_count_total_variation": discrete_total_variation(
            edge_reference, edge_candidate
        ),
        "sequence_uniqueness_rate": (
            len(set(candidate_keys)) / max(len(candidate_keys), 1)
        ),
        "reference_sequence_coverage_rate": (
            len(reference_keys.intersection(candidate_keys))
            / max(len(reference_keys), 1)
        ),
        "degree_mmd_sigma": float(degree_mmd_sigma),
        "candidate_num_nodes_mean": float(np.mean(node_candidate))
        if node_candidate
        else 0.0,
        "candidate_num_nodes_std": float(np.std(node_candidate))
        if node_candidate
        else 0.0,
        "reference_num_nodes_mean": float(np.mean(node_reference))
        if node_reference
        else 0.0,
        "reference_num_nodes_std": float(np.std(node_reference))
        if node_reference
        else 0.0,
        "candidate_num_edges_mean": float(np.mean(edge_candidate))
        if edge_candidate
        else 0.0,
        "candidate_num_edges_std": float(np.std(edge_candidate))
        if edge_candidate
        else 0.0,
        "reference_num_edges_mean": float(np.mean(edge_reference))
        if edge_reference
        else 0.0,
        "reference_num_edges_std": float(np.std(edge_reference))
        if edge_reference
        else 0.0,
        "candidate_mean_degree_mean": float(np.mean(mean_degree_candidate))
        if mean_degree_candidate
        else 0.0,
        "reference_mean_degree_mean": float(np.mean(mean_degree_reference))
        if mean_degree_reference
        else 0.0,
        "mean_degree_mean_abs_error": abs(
            float(np.mean(mean_degree_candidate)) - float(np.mean(mean_degree_reference))
        )
        if mean_degree_candidate and mean_degree_reference
        else 0.0,
        "candidate_degree_variance_mean": float(np.mean(degree_variance_candidate))
        if degree_variance_candidate
        else 0.0,
        "reference_degree_variance_mean": float(np.mean(degree_variance_reference))
        if degree_variance_reference
        else 0.0,
        "degree_variance_mean_abs_error": abs(
            float(np.mean(degree_variance_candidate))
            - float(np.mean(degree_variance_reference))
        )
        if degree_variance_candidate and degree_variance_reference
        else 0.0,
        "candidate_degree_second_moment_mean": float(np.mean(second_moment_candidate))
        if second_moment_candidate
        else 0.0,
        "reference_degree_second_moment_mean": float(np.mean(second_moment_reference))
        if second_moment_reference
        else 0.0,
        "degree_second_moment_mean_abs_error": abs(
            float(np.mean(second_moment_candidate)) - float(np.mean(second_moment_reference))
        )
        if second_moment_candidate and second_moment_reference
        else 0.0,
        "max_degree_total_variation": discrete_total_variation(
            max_degree_reference, max_degree_candidate
        ),
        "wedge_count_total_variation": discrete_total_variation(
            wedge_reference, wedge_candidate
        ),
        "candidate_wedge_count_mean": float(np.mean(wedge_candidate))
        if wedge_candidate
        else 0.0,
        "reference_wedge_count_mean": float(np.mean(wedge_reference))
        if wedge_reference
        else 0.0,
    }
    if train is not None:
        metrics["sequence_novelty_rate"] = (
            float(np.mean([key not in train_keys for key in candidate_keys]))
            if candidate_keys
            else 0.0
        )
    return metrics

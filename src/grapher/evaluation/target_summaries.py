from __future__ import annotations

from typing import Any, Sequence

import numpy as np

EPS = 1.0e-12


def _pad_rows(rows: Sequence[np.ndarray], width: int | None = None) -> np.ndarray:
    arrays = [np.asarray(row, dtype=np.float64).reshape(-1) for row in rows]
    if width is None:
        width = max((row.size for row in arrays), default=1)
    width = max(int(width), 1)
    out = np.zeros((len(arrays), width), dtype=np.float64)
    for index, row in enumerate(arrays):
        out[index, : min(width, row.size)] = row[:width]
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    totals = values.sum(axis=1, keepdims=True)
    return np.divide(
        values,
        totals,
        out=np.zeros_like(values),
        where=totals > 0.0,
    )


def median_rbf_bandwidth(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    pooled = np.vstack([x, y])
    if pooled.shape[0] < 2:
        return 1.0
    squared = np.sum(
        (pooled[:, None, :] - pooled[None, :, :]) ** 2,
        axis=-1,
    )
    distances = squared[np.triu_indices(pooled.shape[0], k=1)]
    distances = distances[distances > 0.0]
    if distances.size == 0:
        return 1.0
    sigma = float(np.sqrt(0.5 * np.median(distances)))
    return sigma if np.isfinite(sigma) and sigma > 0.0 else 1.0


def mmd_rbf_fixed(
    x: np.ndarray,
    y: np.ndarray,
    *,
    sigma: float,
) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape[0] == 0 or y.shape[0] == 0:
        return float("nan")
    gamma = 1.0 / (2.0 * float(sigma) ** 2)

    def kernel(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        squared = np.sum(
            (left[:, None, :] - right[None, :, :]) ** 2,
            axis=-1,
        )
        return np.exp(-gamma * squared)

    return float(
        kernel(x, x).mean()
        + kernel(y, y).mean()
        - 2.0 * kernel(x, y).mean()
    )


def _component_arrays(
    summaries: Sequence[dict[str, Any]],
    vectorizer: Any,
) -> dict[str, np.ndarray]:
    summaries = list(summaries)
    clustering = _normalize_rows(
        _pad_rows(
            [
                np.asarray(item.get("clustering_hist", []), dtype=np.float64)
                for item in summaries
            ],
            vectorizer.clustering_bins,
        )
    )
    spectral = _normalize_rows(
        _pad_rows(
            [
                np.asarray(item.get("spectral_hist", []), dtype=np.float64)
                for item in summaries
            ],
            vectorizer.spectral_bins,
        )
    )

    motif = _pad_rows(
        [
            np.log1p(
                np.maximum(
                    np.asarray(item.get("motif_proxy", []), dtype=np.float64),
                    0.0,
                )
            )
            for item in summaries
        ],
        vectorizer.motif_dim,
    )
    if vectorizer.motif_dim:
        motif = motif / np.maximum(
            np.asarray(vectorizer.motif_scale, dtype=np.float64)[None, :],
            1.0,
        )

    orbit = _pad_rows(
        [
            np.log1p(
                np.maximum(
                    np.asarray(item.get("orbit_count", []), dtype=np.float64),
                    0.0,
                )
            )
            for item in summaries
        ],
        vectorizer.orbit_dim,
    )
    if vectorizer.orbit_dim:
        orbit = orbit / np.maximum(
            np.asarray(vectorizer.orbit_scale, dtype=np.float64)[None, :],
            1.0,
        )

    graphlet = _pad_rows(
        [vectorizer.graphlet_to_vector(item) for item in summaries],
        vectorizer.graphlet_dim,
    )
    for graphlet_slice in vectorizer.graphlet_slices().values():
        if graphlet_slice.stop > graphlet_slice.start:
            graphlet[:, graphlet_slice] = _normalize_rows(
                graphlet[:, graphlet_slice]
            )
    graphlet_sizes = sorted(
        (getattr(vectorizer, "graphlet_keys_by_k", {}) or {}).keys(),
        key=int,
    )
    connected_mass = np.asarray(
        [
            [
                float(
                    (item.get("graphlet_connected_mass", {}) or {}).get(k, 0.0)
                )
                for k in graphlet_sizes
            ]
            for item in summaries
        ],
        dtype=np.float64,
    ).reshape(len(summaries), len(graphlet_sizes))

    triangle_scale = max(float(vectorizer.scalar_scale[1]), 1.0)
    triangle = np.asarray(
        [
            [
                max(float(item.get("triangle_count_norm", 0.0)), 0.0)
                / triangle_scale
            ]
            for item in summaries
        ],
        dtype=np.float64,
    )
    return {
        "clustering": clustering,
        "spectral": spectral,
        "motif": motif,
        "orbit": orbit,
        "graphlet": graphlet,
        "connected_mass": connected_mass,
        "triangle": triangle,
    }


def active_component_names(
    vectorizer: Any,
    loss_weights: dict[str, Any] | None = None,
) -> list[str]:
    weights = dict(loss_weights or {})
    candidates = [
        ("clustering", int(vectorizer.clustering_bins)),
        ("spectral", int(vectorizer.spectral_bins)),
        ("motif", int(vectorizer.motif_dim)),
        ("orbit", int(vectorizer.orbit_dim)),
        ("graphlet", int(vectorizer.graphlet_dim)),
        (
            "connected_mass",
            len(getattr(vectorizer, "graphlet_keys_by_k", {}) or {}),
        ),
        ("triangle", 1),
    ]
    active = []
    for name, width in candidates:
        if name == "triangle":
            weight = weights.get("triangle", weights.get("scalar", 1.0))
        elif name == "connected_mass":
            weight = weights.get("connected_mass", 0.0)
        else:
            weight = weights.get(name, 1.0)
        if width > 0 and float(weight) > 0.0:
            active.append(name)
    return active


def structural_matrix(
    summaries: Sequence[dict[str, Any]],
    vectorizer: Any,
    *,
    component_names: Sequence[str],
    loss_weights: dict[str, Any] | None = None,
) -> np.ndarray:
    arrays = _component_arrays(summaries, vectorizer)
    weights = dict(loss_weights or {})
    blocks: list[np.ndarray] = []
    for name in component_names:
        block = arrays[name]
        if name == "triangle":
            raw_weight = weights.get("triangle", weights.get("scalar", 1.0))
        elif name == "connected_mass":
            raw_weight = weights.get("connected_mass", 0.0)
        else:
            raw_weight = weights.get(name, 1.0)
        weight = max(float(raw_weight), 0.0)
        # sqrt(weight) makes squared Euclidean distances consistent with the
        # weighted reconstruction objective.
        blocks.append(block * np.sqrt(weight) / np.sqrt(max(block.shape[1], 1)))
    if not blocks:
        return np.zeros((len(list(summaries)), 1), dtype=np.float64)
    return np.concatenate(blocks, axis=1)


def fit_mmd_bandwidths(
    reference: Sequence[dict[str, Any]],
    baseline: Sequence[dict[str, Any]],
    vectorizer: Any,
    *,
    component_names: Sequence[str],
    loss_weights: dict[str, Any] | None = None,
) -> dict[str, float]:
    reference_components = _component_arrays(reference, vectorizer)
    baseline_components = _component_arrays(baseline, vectorizer)
    bandwidths = {
        name: median_rbf_bandwidth(
            reference_components[name],
            baseline_components[name],
        )
        for name in component_names
    }
    bandwidths["structural"] = median_rbf_bandwidth(
        structural_matrix(
            reference,
            vectorizer,
            component_names=component_names,
            loss_weights=loss_weights,
        ),
        structural_matrix(
            baseline,
            vectorizer,
            component_names=component_names,
            loss_weights=loss_weights,
        ),
    )
    return bandwidths


def evaluate_summary_sets(
    reference: Sequence[dict[str, Any]],
    candidate: Sequence[dict[str, Any]],
    vectorizer: Any,
    *,
    component_names: Sequence[str],
    bandwidths: dict[str, float],
    loss_weights: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reference = list(reference)
    candidate = list(candidate)
    reference_components = _component_arrays(reference, vectorizer)
    candidate_components = _component_arrays(candidate, vectorizer)
    component_mmd = {
        name: mmd_rbf_fixed(
            reference_components[name],
            candidate_components[name],
            sigma=float(bandwidths[name]),
        )
        for name in component_names
    }
    structural_mmd = mmd_rbf_fixed(
        structural_matrix(
            reference,
            vectorizer,
            component_names=component_names,
            loss_weights=loss_weights,
        ),
        structural_matrix(
            candidate,
            vectorizer,
            component_names=component_names,
            loss_weights=loss_weights,
        ),
        sigma=float(bandwidths["structural"]),
    )
    return {
        "num_reference": len(reference),
        "num_candidate": len(candidate),
        "structural_mmd": structural_mmd,
        "component_mmd": component_mmd,
    }


def graphlet_bin_errors(
    reference: Sequence[dict[str, Any]],
    candidate: Sequence[dict[str, Any]],
    vectorizer: Any,
) -> list[dict[str, Any]]:
    """Report marginal errors for every canonical graphlet coordinate."""

    if int(vectorizer.graphlet_dim) <= 0:
        return []
    reference_values = _component_arrays(reference, vectorizer)["graphlet"]
    candidate_values = _component_arrays(candidate, vectorizer)["graphlet"]
    rows: list[dict[str, Any]] = []
    position = 0
    keys_by_k = getattr(vectorizer, "graphlet_keys_by_k", {}) or {}
    for k in sorted(keys_by_k, key=lambda value: int(value)):
        for key in keys_by_k[k]:
            ref = reference_values[:, position]
            pred = candidate_values[:, position]
            rows.append(
                {
                    "k": int(k),
                    "canonical_key": str(key),
                    "reference_mean": float(ref.mean()),
                    "candidate_mean": float(pred.mean()),
                    "mean_absolute_error": float(abs(pred.mean() - ref.mean())),
                    "rmse": float(
                        np.sqrt(
                            np.mean(
                                (
                                    np.sort(pred)
                                    - np.interp(
                                        np.linspace(0.0, 1.0, pred.size),
                                        np.linspace(0.0, 1.0, ref.size),
                                        np.sort(ref),
                                    )
                                )
                                ** 2
                            )
                        )
                    ),
                }
            )
            position += 1
    return sorted(rows, key=lambda item: item["mean_absolute_error"], reverse=True)


def paired_summary_errors(
    targets: Sequence[dict[str, Any]],
    predictions: Sequence[dict[str, Any]],
    vectorizer: Any,
    *,
    component_names: Sequence[str],
    loss_weights: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if len(targets) != len(predictions):
        raise ValueError("Paired summary evaluation requires equal list lengths.")
    target_components = _component_arrays(targets, vectorizer)
    prediction_components = _component_arrays(predictions, vectorizer)
    component_rmse: dict[str, float] = {}
    component_mae: dict[str, float] = {}
    for name in component_names:
        difference = prediction_components[name] - target_components[name]
        component_rmse[name] = float(np.sqrt(np.mean(difference**2)))
        component_mae[name] = float(np.mean(np.abs(difference)))

    target_structural = structural_matrix(
        targets,
        vectorizer,
        component_names=component_names,
        loss_weights=loss_weights,
    )
    prediction_structural = structural_matrix(
        predictions,
        vectorizer,
        component_names=component_names,
        loss_weights=loss_weights,
    )
    difference = prediction_structural - target_structural
    return {
        "structural_rmse": float(np.sqrt(np.mean(difference**2))),
        "structural_mae": float(np.mean(np.abs(difference))),
        "component_rmse": component_rmse,
        "component_mae": component_mae,
    }


def conditional_sample_metrics(
    targets: Sequence[dict[str, Any]],
    samples_by_condition: Sequence[Sequence[dict[str, Any]]],
    vectorizer: Any,
    *,
    component_names: Sequence[str],
    loss_weights: dict[str, Any] | None = None,
) -> dict[str, float]:
    if len(targets) != len(samples_by_condition):
        raise ValueError("Expected one sample group per target condition.")

    energy_scores: list[float] = []
    conditional_mean_errors: list[float] = []
    within_condition_diversities: list[float] = []
    for target, samples in zip(targets, samples_by_condition):
        samples = list(samples)
        if not samples:
            continue
        target_vector = structural_matrix(
            [target],
            vectorizer,
            component_names=component_names,
            loss_weights=loss_weights,
        )[0]
        sample_vectors = structural_matrix(
            samples,
            vectorizer,
            component_names=component_names,
            loss_weights=loss_weights,
        )
        target_distances = np.linalg.norm(
            sample_vectors - target_vector[None, :],
            axis=1,
        )
        pairwise = np.linalg.norm(
            sample_vectors[:, None, :] - sample_vectors[None, :, :],
            axis=-1,
        )
        energy_scores.append(
            float(target_distances.mean() - 0.5 * pairwise.mean())
        )
        conditional_mean_errors.append(
            float(np.linalg.norm(sample_vectors.mean(axis=0) - target_vector))
        )
        if sample_vectors.shape[0] > 1:
            upper = pairwise[np.triu_indices(sample_vectors.shape[0], k=1)]
            within_condition_diversities.append(float(upper.mean()))
        else:
            within_condition_diversities.append(0.0)

    return {
        "conditional_energy_score": float(np.mean(energy_scores))
        if energy_scores
        else float("nan"),
        "conditional_mean_l2": float(np.mean(conditional_mean_errors))
        if conditional_mean_errors
        else float("nan"),
        "within_condition_diversity": float(
            np.mean(within_condition_diversities)
        )
        if within_condition_diversities
        else float("nan"),
    }


def degree_condition_match_rate(
    targets: Sequence[dict[str, Any]],
    predictions: Sequence[dict[str, Any]],
) -> float:
    if not targets:
        return 0.0
    if len(targets) != len(predictions):
        raise ValueError("Invariant evaluation requires equal list lengths.")
    matches = []
    for target, prediction in zip(targets, predictions):
        matches.append(
            int(prediction.get("num_nodes", -1))
            == int(target.get("num_nodes", -2))
            and int(prediction.get("num_edges", -1))
            == int(target.get("num_edges", -2))
            and list(prediction.get("degree_sequence", []))
            == list(target.get("degree_sequence", []))
        )
    return float(np.mean(matches))

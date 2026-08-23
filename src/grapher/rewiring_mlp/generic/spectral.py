from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import networkx as nx
import numpy as np


def laplacian_eigenvalues(graph: nx.Graph) -> np.ndarray:
    """Return the sorted combinatorial-Laplacian eigenvalues of ``graph``.

    The topology pipeline uses simple undirected graphs.  Small negative values
    caused by floating-point eigensolvers are clipped to zero.  No eigenvectors
    are computed: spectral guidance intentionally operates only on eigenvalues.
    """

    if graph.is_directed() or graph.is_multigraph():
        raise ValueError("Spectral GraphER requires a simple undirected graph.")
    if nx.number_of_selfloops(graph):
        raise ValueError("Spectral GraphER does not support self-loops.")
    nodes = sorted(graph.nodes())
    n = len(nodes)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    adjacency = nx.to_numpy_array(graph, nodelist=nodes, dtype=np.float64)
    degree = adjacency.sum(axis=1)
    laplacian = np.diag(degree) - adjacency
    values = np.linalg.eigvalsh(laplacian)
    values = np.sort(np.asarray(values, dtype=np.float64))
    # The combinatorial Laplacian is PSD.  Numerical noise around zero should
    # not be interpreted as a meaningful negative eigenvalue.
    values[np.abs(values) < 1.0e-10] = 0.0
    values = np.maximum(values, 0.0)
    return values


def degree_spectral_moments(graph: nx.Graph) -> tuple[float, float]:
    """Return the first two Laplacian spectral moments fixed by degrees.

    For a simple undirected graph,
      sum_i lambda_i   = 2m,
      sum_i lambda_i^2 = sum_v d_v^2 + 2m.

    These diagnostics make degree/spectrum incompatibilities visible when a
    training source and target do not belong to the same degree fibre.
    """

    degrees = np.asarray([float(degree) for _, degree in graph.degree()], dtype=np.float64)
    first = float(degrees.sum())
    second = float(np.square(degrees).sum() + degrees.sum())
    return first, second


def spectrum_moments(spectrum: Sequence[float] | np.ndarray) -> tuple[float, float]:
    values = np.asarray(spectrum, dtype=np.float64).reshape(-1)
    return float(values.sum()), float(np.square(values).sum())


def spectral_scale(graph: nx.Graph, *, mode: str = "mean_degree") -> float:
    """Return a stable scale for comparing spectra across graph sizes."""

    normalized = str(mode).lower()
    n = graph.number_of_nodes()
    trace = 2.0 * float(graph.number_of_edges())
    if normalized in {"none", "raw"}:
        return 1.0
    if normalized in {"trace", "degree_sum"}:
        return max(trace, 1.0)
    if normalized in {"mean_degree", "average_degree", "avg_degree"}:
        return max(trace / max(float(n), 1.0), 1.0e-8)
    raise ValueError(
        "spectral normalization must be one of: mean_degree, trace, none."
    )


def spectral_distance(
    spectrum: Sequence[float] | np.ndarray,
    target: Sequence[float] | np.ndarray,
    *,
    metric: str = "rmse",
    scale: float = 1.0,
    low_frequency_weight: float = 1.0,
    low_frequency_cutoff: int = 0,
) -> float:
    """Distance between equally sized ordered eigenvalue vectors.

    ``low_frequency_cutoff`` counts eigenvalues after lambda_1=0.  A weight
    larger than one can emphasize global/low-frequency structure without
    changing the dimensionality of the Spectral Transformer target.
    """

    left = np.asarray(spectrum, dtype=np.float64).reshape(-1)
    right = np.asarray(target, dtype=np.float64).reshape(-1)
    if left.shape != right.shape:
        raise ValueError(
            f"Spectrum shapes must match for scoring: {left.shape} != {right.shape}."
        )
    if left.size == 0:
        return 0.0
    safe_scale = max(float(scale), 1.0e-12)
    delta = (left - right) / safe_scale
    weights = np.ones_like(delta)
    cutoff = min(max(int(low_frequency_cutoff), 0), max(left.size - 1, 0))
    if cutoff > 0 and float(low_frequency_weight) != 1.0:
        weights[1 : 1 + cutoff] = float(low_frequency_weight)
    metric_name = str(metric).lower()
    if metric_name in {"mae", "l1"}:
        return float(np.sum(weights * np.abs(delta)) / np.sum(weights))
    if metric_name in {"mse", "l2_squared"}:
        return float(np.sum(weights * np.square(delta)) / np.sum(weights))
    if metric_name in {"rmse", "l2"}:
        return float(np.sqrt(np.sum(weights * np.square(delta)) / np.sum(weights)))
    raise ValueError("spectral distance must be rmse, mse, or mae.")


@dataclass(frozen=True)
class SpectralBridgeSchedule:
    """Deterministic x0-prediction scheduler for discrete spectral projection.

    The predictor estimates the final clean spectrum.  This scheduler converts
    that clean prediction into a *next* spectral target.  A minimum clean mix
    is useful because one edge swap is a finite discrete move and may overshoot
    an infinitesimal continuous bridge step.
    """

    schedule: str = "linear"
    min_clean_mix: float = 0.15
    max_clean_mix: float = 1.0
    power: float = 2.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None = None) -> "SpectralBridgeSchedule":
        values = dict(data or {})
        schedule = str(values.get("schedule", "linear")).lower()
        if schedule in {"cos", "cosine_bridge"}:
            schedule = "cosine"
        if schedule in {"poly", "polynomial"}:
            schedule = "power"
        result = cls(
            schedule=schedule,
            min_clean_mix=float(values.get("min_clean_mix", 0.15)),
            max_clean_mix=float(values.get("max_clean_mix", 1.0)),
            power=float(values.get("power", 2.0)),
        )
        if result.schedule not in {"linear", "cosine", "power"}:
            raise ValueError(
                "spectral_guidance.schedule must be linear, cosine, or power."
            )
        if not np.isfinite(result.min_clean_mix) or not 0.0 <= result.min_clean_mix <= 1.0:
            raise ValueError("spectral_guidance.min_clean_mix must lie in [0, 1].")
        if not np.isfinite(result.max_clean_mix) or not 0.0 < result.max_clean_mix <= 1.0:
            raise ValueError("spectral_guidance.max_clean_mix must lie in (0, 1].")
        if result.min_clean_mix > result.max_clean_mix:
            raise ValueError(
                "spectral_guidance.min_clean_mix cannot exceed max_clean_mix."
            )
        if not np.isfinite(result.power) or result.power <= 0.0:
            raise ValueError("spectral_guidance.power must be finite and positive.")
        return result

    def alpha(self, progress: float) -> float:
        p = float(np.clip(progress, 0.0, 1.0))
        if self.schedule == "linear":
            return p
        if self.schedule == "cosine":
            return float(np.sin(0.5 * np.pi * p) ** 2)
        if self.schedule == "power":
            return float(p ** self.power)
        raise AssertionError("Unexpected spectral bridge schedule.")

    def clean_mix_for_step(self, *, accepted_step: int, total_steps: int) -> float:
        if int(total_steps) <= 0:
            return 1.0
        current_progress = float(np.clip(accepted_step / total_steps, 0.0, 1.0))
        next_progress = float(
            np.clip((accepted_step + 1) / total_steps, 0.0, 1.0)
        )
        current_alpha = self.alpha(current_progress)
        next_alpha = self.alpha(next_progress)
        if current_alpha >= 1.0 - 1.0e-12:
            base_mix = 1.0
        else:
            # If current spectrum exactly followed the bridge, this convex mix
            # would land on the next continuous bridge state.
            base_mix = (next_alpha - current_alpha) / max(1.0 - current_alpha, 1.0e-12)
        return float(
            np.clip(base_mix, self.min_clean_mix, self.max_clean_mix)
        )

    @staticmethod
    def target(
        current: Sequence[float] | np.ndarray,
        clean: Sequence[float] | np.ndarray,
        clean_mix: float,
    ) -> np.ndarray:
        current_values = np.asarray(current, dtype=np.float64).reshape(-1)
        clean_values = np.asarray(clean, dtype=np.float64).reshape(-1)
        if current_values.shape != clean_values.shape:
            raise ValueError("Current and clean spectra must have identical shape.")
        beta = float(np.clip(clean_mix, 0.0, 1.0))
        return (1.0 - beta) * current_values + beta * clean_values

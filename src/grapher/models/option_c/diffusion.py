"""Scalar bond codec and mixed node/weighted-adjacency forward/reverse kernels."""
from __future__ import annotations

import numpy as np
import torch

# Reuse mathematical categorical kernels only, not the old model or runner.
from grapher.models.gdsm_simple.categorical.noise import (
    MarginalNoise, cosine_alpha_bar, draw_categories, pair_mask,
)


def symmetrize(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (value + value.transpose(1, 2)) * .5 * pair_mask(mask)


def symmetric_noise(mask: torch.Tensor, generator: torch.Generator, *, dtype=torch.float32) -> torch.Tensor:
    """Each active unordered pair has independent N(0,1) noise, not variance 1/2."""
    b, n = mask.shape
    ij = torch.triu_indices(n, n, 1, device=mask.device)
    value = torch.zeros(b, n, n, dtype=dtype, device=mask.device)
    draws = torch.randn(b, ij.shape[1], dtype=dtype, device=mask.device, generator=generator)
    value[:, ij[0], ij[1]] = draws
    value[:, ij[1], ij[0]] = draws
    return value * pair_mask(mask)


def q_sample(clean, noise, alpha_bar, t, mask):
    a = alpha_bar[t, None, None]
    return symmetrize(a.sqrt() * clean + (1 - a).sqrt() * noise, mask)


def reverse_weighted(current, clean, alpha_bar, t, u, mask, generator, *, sampler="ddpm"):
    """Arbitrary-step Gaussian posterior / deterministic DDIM with x0 prediction.

    Supports exact alpha_bar[T]=0 without dividing by its square root. No graph
    thresholding is applied to intermediate states. DDPM uses the conditional
    posterior variance (zero at u=0); DDIM retains the inferred noise residual.
    """
    if (u < 0).any() or (t >= len(alpha_bar)).any() or (u >= t).any():
        raise ValueError("Reverse steps require 0 <= u < t <= T")
    at, au = alpha_bar[t, None, None], alpha_bar[u, None, None]
    if sampler == "ddim":
        residual = (current - at.sqrt() * clean) / (1 - at).sqrt().clamp_min(1e-12)
        value = au.sqrt() * clean + (1 - au).sqrt() * residual
    elif sampler == "ddpm":
        ratio = at / au  # au>0 because u<T
        denom = (1 - at).clamp_min(1e-12)
        c0 = au.sqrt() * (1 - ratio) / denom
        ct = ratio.sqrt() * (1 - au) / denom
        variance = ((1 - au) * (1 - ratio) / denom).clamp_min(0)
        value = c0 * clean + ct * current
        if bool((u > 0).any()):
            value = value + variance.sqrt() * symmetric_noise(mask, generator, dtype=current.dtype)
    else:
        raise ValueError("Unknown weighted-adjacency sampler")
    return symmetrize(value, mask)


def sampling_grid(total_steps: int, steps: int) -> list[int]:
    if not 1 <= steps <= total_steps:
        raise ValueError("sampling.steps must be between 1 and diffusion.steps")
    grid = np.rint(np.linspace(total_steps, 0, steps + 1)).astype(int).tolist()
    if any(a <= b for a, b in zip(grid, grid[1:])):
        raise AssertionError("Non-decreasing reverse-time grid")
    return grid


class WeightedEdgeCodec:
    """Weights are explicit physical numbers, not categorical tensor indices."""
    def __init__(self, vocabulary, representation: dict):
        mapping = {str(k): float(v) for k, v in representation["weights"].items()}
        self.scale = float(representation["scale"])
        self.values = np.array([0.] + [mapping[str(v)] for v in vocabulary.edge_values], np.float64)
        if self.scale <= 0 or not np.isfinite(self.values).all() or len(set(self.values)) != len(self.values):
            raise ValueError("Invalid weight representation")
        self.sort_indices = np.argsort(self.values)
        self.sorted_weights = self.values[self.sort_indices]
        self.thresholds = .5 * (self.sorted_weights[:-1] + self.sorted_weights[1:])

    @property
    def max_scaled(self) -> float:
        return float(self.values.max() / self.scale)

    def encode(self, categories: np.ndarray) -> np.ndarray:
        return (self.values[categories] / self.scale).astype(np.float32)

    def decode(self, scaled_matrix: np.ndarray) -> np.ndarray:
        """Midpoint thresholds; exact midpoint goes to the higher physical weight."""
        value = np.asarray(scaled_matrix, dtype=np.float64)
        if value.ndim != 2 or value.shape[0] != value.shape[1] or not np.isfinite(value).all():
            raise ValueError("Decode needs a finite square matrix")
        physical = .5 * (value + value.T) * self.scale
        index = np.searchsorted(self.thresholds, physical, side="right")
        edges = self.sort_indices[index].astype(np.int16)
        np.fill_diagonal(edges, 0)
        return edges

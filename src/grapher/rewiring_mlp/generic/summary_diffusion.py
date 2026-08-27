from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from grapher.rewiring_mlp.generic.basis import TopologyGraphletBasis


@dataclass(frozen=True)
class SummaryDiffusionConfig:
    """Endpoint-conditioned stochastic bridge used for predictor training.

    Progress is parameterized from 0 (the HH/base source summary) to 1 (the
    clean data summary).  Training samples continuous bridge states directly;
    no intermediate graph and no rewiring trajectory is constructed.

    The implementation uses Brownian-bridge marginals

        x_s = (1-a_s) x_source + a_s x_clean
              + sigma * sqrt(a_s (1-a_s)) * eps,

    where ``a_s`` is a configurable monotone schedule.  Spectral noise can be
    projected to keep lambda_1=0 and the trace fixed.  Graphlet CLR noise is
    centered independently inside every k-block so the CLR gauge is retained.
    """

    schedule: str = "cosine"
    power: float = 2.0
    spectral_sigma: float = 0.20
    graphlet_sigma: float = 0.35
    time_sampling: str = "stratified"
    preserve_spectral_trace: bool = True
    fix_spectral_lambda1: bool = True
    min_progress: float = 0.0
    max_progress: float = 1.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None = None) -> "SummaryDiffusionConfig":
        values = dict(data or {})
        schedule = str(values.get("schedule", "cosine")).lower()
        if schedule in {"cos", "cosine_bridge"}:
            schedule = "cosine"
        if schedule in {"poly", "polynomial"}:
            schedule = "power"
        result = cls(
            schedule=schedule,
            power=float(values.get("power", 2.0)),
            spectral_sigma=float(values.get("spectral_sigma", 0.20)),
            graphlet_sigma=float(values.get("graphlet_sigma", 0.35)),
            time_sampling=str(values.get("time_sampling", "stratified")).lower(),
            preserve_spectral_trace=bool(values.get("preserve_spectral_trace", True)),
            fix_spectral_lambda1=bool(values.get("fix_spectral_lambda1", True)),
            min_progress=float(values.get("min_progress", 0.0)),
            max_progress=float(values.get("max_progress", 1.0)),
        )
        if result.schedule not in {"linear", "cosine", "power"}:
            raise ValueError("summary_diffusion.schedule must be linear, cosine, or power.")
        if result.time_sampling not in {"uniform", "stratified"}:
            raise ValueError("summary_diffusion.time_sampling must be uniform or stratified.")
        if not np.isfinite(result.power) or result.power <= 0.0:
            raise ValueError("summary_diffusion.power must be finite and positive.")
        if not np.isfinite(result.spectral_sigma) or result.spectral_sigma < 0.0:
            raise ValueError("summary_diffusion.spectral_sigma must be finite and nonnegative.")
        if not np.isfinite(result.graphlet_sigma) or result.graphlet_sigma < 0.0:
            raise ValueError("summary_diffusion.graphlet_sigma must be finite and nonnegative.")
        if not (0.0 <= result.min_progress < result.max_progress <= 1.0):
            raise ValueError(
                "summary_diffusion progress bounds must satisfy "
                "0 <= min_progress < max_progress <= 1."
            )
        return result

    def alpha(self, progress: float) -> float:
        p = float(np.clip(progress, 0.0, 1.0))
        if self.schedule == "linear":
            return p
        if self.schedule == "cosine":
            return float(0.5 - 0.5 * np.cos(np.pi * p))
        if self.schedule == "power":
            return float(p ** self.power)
        raise AssertionError("Unexpected summary diffusion schedule.")

    def sample_progresses(
        self,
        count: int,
        *,
        rng: np.random.Generator,
    ) -> np.ndarray:
        count = max(int(count), 1)
        low = float(self.min_progress)
        high = float(self.max_progress)
        if self.time_sampling == "uniform":
            return rng.uniform(low, high, size=count).astype(np.float64)
        # Stratification prevents an eager cache from accidentally leaving large
        # unsupervised time intervals while retaining stochasticity inside each bin.
        edges = np.linspace(low, high, count + 1, dtype=np.float64)
        u = rng.random(count)
        return edges[:-1] + u * (edges[1:] - edges[:-1])


def _unit_rms(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    rms = float(np.sqrt(np.mean(np.square(values)))) if values.size else 0.0
    if rms <= 1.0e-12:
        return values
    return values / rms


def sample_spectral_bridge_marginal(
    source: Sequence[float] | np.ndarray,
    clean: Sequence[float] | np.ndarray,
    *,
    progress: float,
    sigma: float,
    scale: float,
    preserve_trace: bool,
    fix_lambda1: bool,
    schedule: SummaryDiffusionConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, float]]:
    """Sample one continuous Brownian-bridge spectral state.

    The sampled vector is intentionally *not* projected to a realizable graph
    spectrum.  Only inexpensive invariant-compatible noise projections are
    applied when requested; ordering and higher spectral moments may be violated
    because the diffusion state lives in continuous summary space.
    """

    source_values = np.asarray(source, dtype=np.float64).reshape(-1)
    clean_values = np.asarray(clean, dtype=np.float64).reshape(-1)
    if source_values.shape != clean_values.shape:
        raise ValueError("Spectral bridge endpoints must have identical shape.")
    a = schedule.alpha(progress)
    mean = (1.0 - a) * source_values + a * clean_values
    variance_factor = max(a * (1.0 - a), 0.0)
    std = float(sigma) * max(float(scale), 1.0e-12) * float(np.sqrt(variance_factor))
    noise = rng.normal(size=source_values.shape).astype(np.float64)
    if noise.size:
        if fix_lambda1:
            noise[0] = 0.0
        start = 1 if fix_lambda1 and noise.size > 1 else 0
        if preserve_trace and noise.size - start > 0:
            active = noise[start:]
            active = active - float(active.mean())
            noise[start:] = _unit_rms(active)
        elif noise.size - start > 0:
            noise[start:] = _unit_rms(noise[start:])
    state = mean + std * noise
    if fix_lambda1 and state.size:
        state[0] = 0.0
    return state, {
        "progress": float(progress),
        "alpha": float(a),
        "noise_std": float(std),
        "noise_rms": float(np.sqrt(np.mean(np.square(std * noise)))) if noise.size else 0.0,
    }


def sample_graphlet_clr_bridge_marginal(
    source_logits: Sequence[float] | np.ndarray,
    clean_logits: Sequence[float] | np.ndarray,
    *,
    progress: float,
    sigma: float,
    graphlet_basis: TopologyGraphletBasis,
    coordinate_mask: Sequence[bool] | np.ndarray,
    schedule: SummaryDiffusionConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, float]]:
    """Sample one continuous graphlet bridge state in blockwise CLR space."""

    source = np.asarray(source_logits, dtype=np.float64).reshape(-1)
    clean = np.asarray(clean_logits, dtype=np.float64).reshape(-1)
    mask = np.asarray(coordinate_mask, dtype=np.bool_).reshape(-1)
    if source.shape != clean.shape or source.shape != mask.shape:
        raise ValueError("Graphlet CLR bridge endpoint/mask shapes must agree.")
    if source.size != graphlet_basis.simplex_width:
        raise ValueError("Graphlet CLR bridge width does not match the graphlet basis.")
    a = schedule.alpha(progress)
    mean = (1.0 - a) * source + a * clean
    std = float(sigma) * float(np.sqrt(max(a * (1.0 - a), 0.0)))
    noise = np.zeros_like(source)
    for start, stop in graphlet_basis.simplex_slices:
        block_mask = mask[start:stop]
        if not np.any(block_mask):
            continue
        block = rng.normal(size=stop - start).astype(np.float64)
        block = block - float(block.mean())
        block = _unit_rms(block)
        noise[start:stop] = block
    state = mean + std * noise
    state[~mask] = 0.0
    # Numerical recentering preserves the identifiable CLR gauge exactly.
    for start, stop in graphlet_basis.simplex_slices:
        block_mask = mask[start:stop]
        if np.any(block_mask):
            state[start:stop] -= float(state[start:stop].mean())
    return state, {
        "progress": float(progress),
        "alpha": float(a),
        "noise_std": float(std),
        "noise_rms": float(np.sqrt(np.mean(np.square(std * noise[mask])))) if np.any(mask) else 0.0,
    }

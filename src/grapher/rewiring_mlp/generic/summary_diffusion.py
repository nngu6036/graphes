from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Sequence

import numpy as np

from grapher.rewiring_mlp.generic.basis import TopologyGraphletBasis


@lru_cache(maxsize=32)
def _hogdiff_ou_bridge_arrays(
    n_steps: int,
    ou_schedule: str,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute HOG-Diff/GOUB endpoint-marginal coefficients.

    HOG-Diff constructs these tensors once inside ``OUBridge``.  GraphER's
    streaming dataset can request millions of bridge states, so rebuilding an
    N-step schedule per sample would be unnecessarily expensive.
    """

    n_steps = int(n_steps)
    if ou_schedule == "constant":
        thetas = np.ones(n_steps + 1, dtype=np.float64)
    elif ou_schedule == "linear":
        scale = 1000.0 / float(n_steps + 1)
        thetas = np.linspace(
            scale * 1.0e-4,
            scale * 2.0e-2,
            n_steps + 1,
            dtype=np.float64,
        )
    elif ou_schedule == "cosine":
        s = 0.008
        total = n_steps + 2
        x = np.linspace(0.0, float(total), total + 1, dtype=np.float64)
        f_t = np.cos(((x / float(total)) + s) / (1.0 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = f_t / f_t[0]
        thetas = 1.0 - alphas_cumprod[1:-1]
    else:
        raise ValueError(f"Unknown OU schedule: {ou_schedule!r}")

    theta_bar = np.cumsum(thetas) - thetas[0]
    terminal = float(theta_bar[-1])
    if terminal <= 0.0:
        raise ValueError("Degenerate OU bridge theta schedule.")
    dt = -float(np.log(float(eps))) / terminal
    sigma_bar = np.sqrt(np.maximum(1.0 - np.exp(-2.0 * theta_bar * dt), 0.0))
    sigma_bar_t_T = np.sqrt(
        np.maximum(1.0 - np.exp(-2.0 * (terminal - theta_bar) * dt), 0.0)
    )
    terminal_variance = max(float(sigma_bar[-1] ** 2), 1.0e-15)
    clean_coeff = (
        np.exp(-theta_bar * dt) * np.square(sigma_bar_t_T) / terminal_variance
    )
    source_coeff = (
        (1.0 - np.exp(-theta_bar * dt)) * np.square(sigma_bar_t_T)
        + np.exp(-2.0 * (terminal - theta_bar) * dt) * np.square(sigma_bar)
    ) / terminal_variance
    sigma_prime = sigma_bar * sigma_bar_t_T / max(float(sigma_bar[-1]), 1.0e-15)
    for array in (clean_coeff, source_coeff, sigma_prime):
        array.setflags(write=False)
    return clean_coeff, source_coeff, sigma_prime


@dataclass(frozen=True)
class SummaryDiffusionConfig:
    """Endpoint-conditioned stochastic bridge used for predictor training.

    Progress is parameterized from 0 (the HH/base source summary) to 1 (the
    clean data summary).  Training samples continuous bridge states directly;
    no intermediate graph and no rewiring trajectory is constructed.

    Two endpoint-conditioned marginals are supported:

    ``brownian`` (the original GraphER path) uses

        x_s = (1-a_s) x_source + a_s x_clean
              + sigma * sqrt(a_s (1-a_s)) * eps.

    ``ou_bridge`` ports the GOUB/OU-bridge marginal used by HOG-Diff's second
    stage.  It has zero variance at both endpoints and a mean-reverting,
    time-inhomogeneous variance profile rather than the symmetric Brownian
    profile. Spectral noise can still be projected to keep lambda_1=0 and the
    trace fixed; graphlet CLR noise is centered blockwise.
    """

    bridge: str = "brownian"
    # Optional separate marginal for graphlet CLR coordinates. ``inherit``
    # preserves the pre-existing joint-summary behavior; HOG-inspired spectral
    # ablations can set this to ``brownian`` to isolate the OU change to spectra.
    graphlet_bridge: str = "inherit"
    schedule: str = "cosine"
    power: float = 2.0
    spectral_sigma: float = 0.20
    graphlet_sigma: float = 0.35
    time_sampling: str = "stratified"
    preserve_spectral_trace: bool = True
    fix_spectral_lambda1: bool = True
    min_progress: float = 0.0
    max_progress: float = 1.0
    ou_num_scales: int = 800
    ou_schedule: str = "linear"
    ou_eps: float = 0.005

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None = None) -> "SummaryDiffusionConfig":
        values = dict(data or {})
        bridge = str(values.get("bridge", values.get("bridge_type", "brownian"))).lower()
        if bridge in {"brownian_endpoint_conditioned", "brownian_bridge", "bb"}:
            bridge = "brownian"
        if bridge in {"ou", "oubridge", "goub", "hogdiff_ou"}:
            bridge = "ou_bridge"
        graphlet_bridge = str(values.get("graphlet_bridge", "inherit")).lower()
        if graphlet_bridge in {"brownian_endpoint_conditioned", "brownian_bridge", "bb"}:
            graphlet_bridge = "brownian"
        if graphlet_bridge in {"ou", "oubridge", "goub", "hogdiff_ou"}:
            graphlet_bridge = "ou_bridge"
        schedule = str(values.get("schedule", "cosine")).lower()
        if schedule in {"cos", "cosine_bridge"}:
            schedule = "cosine"
        if schedule in {"poly", "polynomial"}:
            schedule = "power"
        result = cls(
            bridge=bridge,
            graphlet_bridge=graphlet_bridge,
            schedule=schedule,
            power=float(values.get("power", 2.0)),
            spectral_sigma=float(values.get("spectral_sigma", 0.20)),
            graphlet_sigma=float(values.get("graphlet_sigma", 0.35)),
            time_sampling=str(values.get("time_sampling", "stratified")).lower(),
            preserve_spectral_trace=bool(values.get("preserve_spectral_trace", True)),
            fix_spectral_lambda1=bool(values.get("fix_spectral_lambda1", True)),
            min_progress=float(values.get("min_progress", 0.0)),
            max_progress=float(values.get("max_progress", 1.0)),
            ou_num_scales=int(values.get("ou_num_scales", values.get("num_scales", 800))),
            ou_schedule=str(values.get("ou_schedule", "linear")).lower(),
            ou_eps=float(values.get("ou_eps", 0.005)),
        )
        if result.bridge not in {"brownian", "ou_bridge"}:
            raise ValueError("summary_diffusion.bridge must be brownian or ou_bridge.")
        if result.graphlet_bridge not in {"inherit", "brownian", "ou_bridge"}:
            raise ValueError(
                "summary_diffusion.graphlet_bridge must be inherit, brownian, or ou_bridge."
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
        if result.ou_num_scales <= 1:
            raise ValueError("summary_diffusion.ou_num_scales must be greater than one.")
        if result.ou_schedule not in {"constant", "linear", "cosine"}:
            raise ValueError(
                "summary_diffusion.ou_schedule must be constant, linear, or cosine."
            )
        if not np.isfinite(result.ou_eps) or not 0.0 < result.ou_eps < 1.0:
            raise ValueError("summary_diffusion.ou_eps must lie in (0, 1).")
        return result

    @property
    def resolved_graphlet_bridge(self) -> str:
        return self.bridge if self.graphlet_bridge == "inherit" else self.graphlet_bridge

    def alpha(self, progress: float) -> float:
        p = float(np.clip(progress, 0.0, 1.0))
        if self.schedule == "linear":
            return p
        if self.schedule == "cosine":
            return float(0.5 - 0.5 * np.cos(np.pi * p))
        if self.schedule == "power":
            return float(p ** self.power)
        raise AssertionError("Unexpected summary diffusion schedule.")

    def ou_bridge_coefficients(self, progress: float) -> tuple[float, float, float]:
        """Return (clean coefficient, source coefficient, unit noise std).

        This is the HOG-Diff ``OUBridge.marginal_prob`` schedule expressed in
        GraphER's progress convention: progress=0 is the source/HH endpoint and
        progress=1 is the clean data endpoint. HOG-Diff indexes the opposite
        direction, so ``t = (1-progress) * N`` here.
        """

        n_steps = int(self.ou_num_scales)
        clean_coeff, source_coeff, sigma_prime = _hogdiff_ou_bridge_arrays(
            n_steps, self.ou_schedule, float(self.ou_eps)
        )
        t = float(np.clip((1.0 - float(progress)) * n_steps, 0.0, float(n_steps)))
        grid = np.arange(n_steps + 1, dtype=np.float64)
        m = float(np.interp(t, grid, clean_coeff))
        n = float(np.interp(t, grid, source_coeff))
        std = float(np.interp(t, grid, sigma_prime))
        # The closed-form bridge is affine in the two endpoints. Tiny floating
        # errors can move m+n a few ulps away from one.
        coeff_sum = m + n
        if coeff_sum > 1.0e-15:
            m /= coeff_sum
            n /= coeff_sum
        return m, n, max(std, 0.0)

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
    """Sample one continuous endpoint-conditioned spectral bridge state.

    The sampled vector is intentionally *not* projected to a realizable graph
    spectrum.  Only inexpensive invariant-compatible noise projections are
    applied when requested; ordering and higher spectral moments may be violated
    because the diffusion state lives in continuous summary space.
    """

    source_values = np.asarray(source, dtype=np.float64).reshape(-1)
    clean_values = np.asarray(clean, dtype=np.float64).reshape(-1)
    if source_values.shape != clean_values.shape:
        raise ValueError("Spectral bridge endpoints must have identical shape.")
    if schedule.bridge == "ou_bridge":
        clean_coeff, source_coeff, unit_std = schedule.ou_bridge_coefficients(progress)
        mean = source_coeff * source_values + clean_coeff * clean_values
        std = float(sigma) * max(float(scale), 1.0e-12) * float(unit_std)
        a = clean_coeff
    else:
        a = schedule.alpha(progress)
        source_coeff = 1.0 - a
        clean_coeff = a
        mean = source_coeff * source_values + clean_coeff * clean_values
        variance_factor = max(a * (1.0 - a), 0.0)
        unit_std = float(np.sqrt(variance_factor))
        std = float(sigma) * max(float(scale), 1.0e-12) * unit_std
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
        "clean_coefficient": float(clean_coeff),
        "source_coefficient": float(source_coeff),
        "unit_noise_std": float(unit_std),
        "bridge": str(schedule.bridge),
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
    bridge_kind = schedule.resolved_graphlet_bridge
    if bridge_kind == "ou_bridge":
        clean_coeff, source_coeff, unit_std = schedule.ou_bridge_coefficients(progress)
        mean = source_coeff * source + clean_coeff * clean
        std = float(sigma) * float(unit_std)
        a = clean_coeff
    else:
        a = schedule.alpha(progress)
        source_coeff = 1.0 - a
        clean_coeff = a
        mean = source_coeff * source + clean_coeff * clean
        unit_std = float(np.sqrt(max(a * (1.0 - a), 0.0)))
        std = float(sigma) * unit_std
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
        "clean_coefficient": float(clean_coeff),
        "source_coefficient": float(source_coeff),
        "unit_noise_std": float(unit_std),
        "bridge": str(bridge_kind),
        "noise_std": float(std),
        "noise_rms": float(np.sqrt(np.mean(np.square(std * noise[mask])))) if np.any(mask) else 0.0,
    }

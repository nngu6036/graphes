"""Small project-owned spectral diffusion model used as a GSDM reference.

The model intentionally implements only the core experiment:
  * diffuse sorted adjacency eigenvalues;
  * keep / sample an empirical eigenvector basis from the training split;
  * reconstruct A = U diag(lambda) U^T;
  * threshold the reconstructed adjacency.

It is deliberately smaller than the released GSDM architecture so GraphER
components can be introduced one at a time around a stable reference.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class DiffusionSchedule:
    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_bar: torch.Tensor


def make_schedule(*, steps: int, beta_start: float, beta_end: float, device=None) -> DiffusionSchedule:
    if steps < 2:
        raise ValueError("diffusion steps must be >= 2")
    if not (0.0 < beta_start < beta_end < 1.0):
        raise ValueError("expected 0 < beta_start < beta_end < 1")
    betas = torch.linspace(beta_start, beta_end, steps, dtype=torch.float32, device=device)
    alphas = 1.0 - betas
    return DiffusionSchedule(betas, alphas, torch.cumprod(alphas, dim=0))


def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard sinusoidal embedding for integer diffusion indices."""
    if dim < 4:
        raise ValueError("time embedding dimension must be >= 4")
    half = dim // 2
    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, dtype=torch.float32, device=t.device)
        / max(half - 1, 1)
    )
    angles = t.float().unsqueeze(-1) * frequencies.unsqueeze(0)
    value = torch.cat((angles.sin(), angles.cos()), dim=-1)
    if value.size(-1) < dim:
        value = torch.nn.functional.pad(value, (0, dim - value.size(-1)))
    return value


class EigenvalueDenoiser(nn.Module):
    """Masked Transformer epsilon-predictor over a variable-length spectrum."""

    def __init__(
        self,
        *,
        max_nodes: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        ff_dim: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if max_nodes < 2:
            raise ValueError("max_nodes must be >= 2")
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.max_nodes = int(max_nodes)
        self.hidden_dim = int(hidden_dim)
        # noisy eigenvalue, normalized rank, normalized graph size
        self.token_input = nn.Linear(3, hidden_dim)
        self.position = nn.Parameter(torch.zeros(max_nodes, hidden_dim))
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))
        nn.init.normal_(self.position, std=0.02)

    def forward(
        self,
        noisy: torch.Tensor,
        timestep: torch.Tensor,
        mask: torch.Tensor,
        num_nodes: torch.Tensor,
        diffusion_steps: int,
    ) -> torch.Tensor:
        if noisy.ndim != 2 or mask.shape != noisy.shape:
            raise ValueError("noisy/mask must have shape [B, N]")
        b, nmax = noisy.shape
        if nmax != self.max_nodes:
            raise ValueError(f"expected max_nodes={self.max_nodes}, got {nmax}")
        rank = torch.arange(nmax, device=noisy.device, dtype=noisy.dtype)
        rank = rank.unsqueeze(0).expand(b, -1) / max(nmax - 1, 1)
        size = num_nodes.to(noisy.dtype).unsqueeze(-1).expand(-1, nmax) / float(nmax)
        h = self.token_input(torch.stack((noisy, rank, size), dim=-1))
        h = h + self.position.unsqueeze(0)
        time = sinusoidal_time_embedding(
            timestep.to(noisy.device) / max(float(diffusion_steps - 1), 1.0),
            self.hidden_dim,
        )
        h = h + self.time_mlp(time).unsqueeze(1)
        h = self.encoder(h, src_key_padding_mask=~mask.bool())
        eps = self.output(h).squeeze(-1)
        return eps * mask.to(eps.dtype)


def q_sample(
    clean: torch.Tensor,
    timestep: torch.Tensor,
    noise: torch.Tensor,
    schedule: DiffusionSchedule,
) -> torch.Tensor:
    ab = schedule.alpha_bar[timestep].unsqueeze(-1)
    return ab.sqrt() * clean + (1.0 - ab).sqrt() * noise


@torch.no_grad()
def ddim_sample(
    model: EigenvalueDenoiser,
    *,
    mask: torch.Tensor,
    num_nodes: torch.Tensor,
    schedule: DiffusionSchedule,
    sample_steps: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Deterministic DDIM (eta=0) over the masked eigenvalue sequence."""
    device = mask.device
    b, nmax = mask.shape
    x = torch.randn((b, nmax), device=device, generator=generator) * mask
    total = int(schedule.alpha_bar.numel())
    steps = max(2, min(int(sample_steps), total))
    indices = torch.linspace(total - 1, 0, steps, device=device).round().long()
    indices = torch.unique_consecutive(indices)
    if indices[-1].item() != 0:
        indices = torch.cat((indices, torch.zeros(1, dtype=torch.long, device=device)))
    for position, t_scalar in enumerate(indices):
        t = torch.full((b,), int(t_scalar.item()), dtype=torch.long, device=device)
        eps = model(x, t, mask, num_nodes, total)
        ab_t = schedule.alpha_bar[t_scalar]
        x0 = (x - (1.0 - ab_t).sqrt() * eps) / ab_t.sqrt().clamp_min(1e-8)
        x0 = x0 * mask
        if position + 1 == len(indices):
            x = x0
            break
        prev = indices[position + 1]
        ab_prev = schedule.alpha_bar[prev]
        x = (ab_prev.sqrt() * x0 + (1.0 - ab_prev).sqrt() * eps) * mask
    return x


def reconstruct_soft_adjacency(eigenvectors: torch.Tensor, eigenvalues: torch.Tensor) -> torch.Tensor:
    """Reconstruct a symmetric matrix U diag(lambda) U^T."""
    return eigenvectors @ torch.diag_embed(eigenvalues) @ eigenvectors.transpose(-1, -2)

"""Degree/basis-conditioned spectral denoiser with clean-structure heads."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from grapher.models.gdsm_simple.model import EigenvalueDenoiser, DiffusionSchedule, sinusoidal_time_embedding


class StructuredEigenvalueDenoiser(EigenvalueDenoiser):
    def __init__(self, *, clustering_bins: int = 100, **kwargs):
        super().__init__(**kwargs)
        hidden = self.hidden_dim
        self.clustering_bins = int(clustering_bins)
        self.condition = nn.Sequential(nn.Linear(4*self.max_nodes, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.summary_trunk = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.SiLU())
        self.clustering_head = nn.Linear(hidden, self.clustering_bins)
        self.orbit_head = nn.Linear(hidden, 4)
        self.graphlet_head = nn.Linear(hidden, 2)

    def forward(self, noisy, timestep, mask, num_nodes, diffusion_steps, condition):
        if noisy.ndim != 2 or noisy.shape != mask.shape or noisy.shape[1] != self.max_nodes:
            raise ValueError("noisy/mask must have shape [B,max_nodes].")
        if condition.shape != (noisy.shape[0], 4*self.max_nodes):
            raise ValueError("condition must have shape [B,4*max_nodes].")
        b, nmax = noisy.shape
        rank = torch.arange(nmax, device=noisy.device, dtype=noisy.dtype).expand(b, -1) / max(nmax-1, 1)
        size = num_nodes.to(noisy.dtype).unsqueeze(1).expand(-1, nmax) / nmax
        h = self.token_input(torch.stack((noisy, rank, size), dim=-1)) + self.position.unsqueeze(0)
        t = sinusoidal_time_embedding(timestep.to(noisy.device) / max(float(diffusion_steps-1), 1.), self.hidden_dim)
        h = h + self.time_mlp(t).unsqueeze(1) + self.condition(condition).unsqueeze(1)
        h = self.encoder(h, src_key_padding_mask=~mask.bool())
        eps = self.output(h).squeeze(-1) * mask
        pooled = (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        pooled = self.summary_trunk(pooled)
        return eps, {
            "clustering_logits": self.clustering_head(pooled),
            "orbit_log_mean": F.softplus(self.orbit_head(pooled)),
            "graphlet_logits": self.graphlet_head(pooled),
        }


def summary_probabilities(prediction: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        "clustering_histogram": prediction["clustering_logits"].softmax(-1),
        "orbit_log_mean": prediction["orbit_log_mean"],
        "graphlet_histogram": prediction["graphlet_logits"].softmax(-1),
    }


def summary_losses(prediction, clustering, orbit, graphlet):
    # KL (not cross entropy) removes target-entropy constants from model selection.
    logc = prediction["clustering_logits"].log_softmax(-1)
    logg = prediction["graphlet_logits"].log_softmax(-1)
    klc = F.kl_div(logc, clustering, reduction="batchmean")
    klg = F.kl_div(logg, graphlet, reduction="batchmean")
    # Empty connected-graphlet targets have no graphlet CE; orbit=0 still trains.
    return {
        "clustering": klc + F.mse_loss(logc.exp().cumsum(-1), clustering.cumsum(-1)),
        "orbit": F.mse_loss(prediction["orbit_log_mean"], orbit),
        "graphlet": klg,
    }


def centered_q_sample(clean, timestep, noise, anchor, schedule: DiffusionSchedule):
    """x_t = s + sqrt(alpha_bar_t)(x_0-s) + sqrt(1-alpha_bar_t) eps."""
    ab = schedule.alpha_bar[timestep].unsqueeze(-1)
    return anchor + ab.sqrt() * (clean-anchor) + (1-ab).sqrt() * noise


def centered_x0(noisy, eps, timestep, anchor, schedule: DiffusionSchedule):
    ab = schedule.alpha_bar[timestep].unsqueeze(-1)
    return anchor + (noisy-anchor-(1-ab).sqrt()*eps) / ab.sqrt().clamp_min(1e-8)


def centered_ddim_step(x0, eps, next_timestep, anchor, schedule: DiffusionSchedule):
    ab = schedule.alpha_bar[next_timestep].unsqueeze(-1)
    return anchor + ab.sqrt()*(x0-anchor) + (1-ab).sqrt()*eps

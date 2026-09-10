"""Continuous endpoint-conditioned diffusion of symmetric categorical edge logits.

Time runs from the typed source (t=0) to the data endpoint (t=1). This is a
Gaussian bridge in centered-logit space, NOT a categorical Markov chain on
chemically valid molecules. Hard graphs are constructed/projected separately.
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def pair_mask(node_mask: torch.Tensor, *, upper: bool = False) -> torch.Tensor:
    if node_mask.ndim != 2 or node_mask.dtype != torch.bool:
        raise ValueError("node_mask must be bool [batch, nodes].")
    mask = node_mask[:, :, None] & node_mask[:, None, :]
    if upper:
        return torch.triu(mask, diagonal=1)
    eye = torch.eye(mask.shape[-1], dtype=torch.bool, device=mask.device)
    return mask & ~eye


def center_edges(values: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
    """Class-centered, symmetric; diagonal and padding are zero (not bond states)."""
    if values.ndim != 4 or values.shape[1:3] != (node_mask.shape[1],) * 2:
        raise ValueError("Expected [batch, nodes, nodes, categories] edge logits.")
    values = (values + values.transpose(1, 2)) * 0.5
    return (values - values.mean(-1, keepdim=True)) * pair_mask(node_mask)[..., None]


def edge_probabilities(logits: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
    """Invalid/diagonal pairs are explicitly no-bond, not uniform categories."""
    probs = logits.softmax(-1)
    absent = torch.zeros_like(probs); absent[..., 0] = 1
    return torch.where(pair_mask(node_mask)[..., None], probs, absent)


def labels_to_logits(labels: torch.Tensor, categories: int, node_mask: torch.Tensor,
                     smoothing: float = 0.01) -> torch.Tensor:
    if categories < 2 or not 0 < smoothing < 1:
        raise ValueError("At least two categories and 0 < smoothing < 1 required.")
    if labels.shape != (node_mask.shape[0], node_mask.shape[1], node_mask.shape[1]):
        raise ValueError("Bad edge-label shape.")
    if not torch.equal(labels, labels.transpose(1, 2)):
        raise ValueError("Undirected edge labels must be symmetric.")
    if torch.any(labels < 0) or torch.any(labels >= categories):
        raise ValueError("Edge label outside category support.")
    p = (1 - smoothing) * F.one_hot(labels.long(), categories).float() + smoothing / categories
    return center_edges(p.log(), node_mask)


def edge_noise(shape: tuple, node_mask: torch.Tensor, *, generator=None,
               dtype=torch.float32) -> torch.Tensor:
    """One independent Gaussian vector per unordered pair, then class centering.

    Mirroring (rather than averaging two noise draws) preserves the intended
    variance. In class space the noise covariance is I - 11'/K.
    """
    raw = torch.randn(shape, device=node_mask.device, dtype=dtype, generator=generator)
    raw = raw * pair_mask(node_mask, upper=True)[..., None]
    raw = raw + raw.transpose(1, 2)
    return center_edges(raw, node_mask)


def bridge_edges(source: torch.Tensor, target: torch.Tensor, t: torch.Tensor,
                 node_mask: torch.Tensor, sigma: float = 1.0, *, generator=None) -> torch.Tensor:
    if not math.isfinite(sigma) or sigma < 0:
        raise ValueError("edge sigma must be finite and nonnegative.")
    if source.shape != target.shape or t.shape != (source.shape[0],):
        raise ValueError("Bridge endpoint/time shapes disagree.")
    if torch.any((t < 0) | (t > 1)):
        raise ValueError("Bridge time outside [0,1].")
    a = t[:, None, None, None]
    noise = edge_noise(tuple(source.shape), node_mask, generator=generator, dtype=source.dtype)
    return center_edges((1-a)*source + a*target + sigma*(a*(1-a)).sqrt()*noise, node_mask)


def advance_edges(state: torch.Tensor, endpoint: torch.Tensor, t: float, s: float,
                  node_mask: torch.Tensor, sigma: float, *, generator=None) -> torch.Tensor:
    """Exact bridge conditional if `endpoint` is known; plug-in learned sampler otherwise.

    E[Z_s|Z_t,Z_1] = Z_t + (s-t)/(1-t) (Z_1-Z_t)
    Var = sigma^2 (s-t)(1-s)/(1-t), in the symmetric centered subspace.
    This is a finite-step endpoint-estimate sampler, not an exact conditional
    data-distribution sampler for a finite trained model/step count.
    """
    if not 0 <= t < s <= 1 or not math.isfinite(sigma) or sigma < 0:
        raise ValueError("Expected 0 <= t < s <= 1 and nonnegative sigma.")
    a = (s-t)/(1-t)
    sd = sigma * math.sqrt((s-t)*(1-s)/(1-t))
    noise = edge_noise(tuple(state.shape), node_mask, generator=generator, dtype=state.dtype)
    return center_edges(state + a*(endpoint-state) + sd*noise, node_mask)


def spectral_noise(source: torch.Tensor, node_mask: torch.Tensor, *, generator=None) -> torch.Tensor:
    """Noise on nonzero eigenvalue slots, projected to zero sum per channel."""
    mask = node_mask.clone()
    mask[:, 0] = False
    w = mask[:, None, :].to(source.dtype)
    raw = torch.randn(source.shape, device=source.device, dtype=source.dtype, generator=generator)*w
    raw = (raw - raw.sum(-1, keepdim=True)/w.sum(-1, keepdim=True).clamp_min(1))*w
    return raw


def bridge_spectra(source: torch.Tensor, target: torch.Tensor, t: torch.Tensor,
                   node_mask: torch.Tensor, sigma: float, *, generator=None) -> torch.Tensor:
    # No sorting/clipping: that would change the Gaussian bridge law. These are
    # noisy continuous coordinates, not claimed to be actual graph eigenvalues.
    a = t[:, None, None]
    return (1-a)*source + a*target + sigma*(a*(1-a)).sqrt()*spectral_noise(
        source, node_mask, generator=generator)


def project_spectra(values: torch.Tensor, trace: torch.Tensor,
                    node_mask: torch.Tensor) -> torch.Tensor:
    """Positive ordered endpoint spectra with lambda1=0 and prescribed trace.

    Trace and lambda1 constraints are necessary, not sufficient, for a graph
    spectrum. Never used to turn the soft bond matrix directly into a molecule.
    """
    mask = node_mask.clone(); mask[:, 0] = False
    positive = F.softplus(values) * mask[:, None, :]
    positive = positive / positive.sum(-1, keepdim=True).clamp_min(1e-12) * trace[..., None]
    # Valid slots sorted first; padding at the end, not mixed into small eigenvalues.
    sorted_values = positive.masked_fill(~node_mask[:, None, :], float('inf')).sort(-1).values
    return sorted_values.masked_fill(~node_mask[:, None, :], 0)

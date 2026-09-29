"""Marginal categorical diffusion, indexed 0 (clean) through T (marginal).

Original implementation of Q(a)=a I+(1-a) 1 m^T and the clean-endpoint
posterior mixture. Both node and edge chains use marginal, NOT absorbing, noise.
The exact finite-T terminal marginal avoids a train/sample endpoint mismatch.
"""
from __future__ import annotations

import math
import torch
from torch.nn import functional as F


def cosine_alpha_bar(steps: int, *, device=None) -> torch.Tensor:
    if steps < 2:
        raise ValueError("diffusion.steps must be >=2")
    t = torch.arange(steps + 1, dtype=torch.float64, device=device) / steps
    a = torch.cos((t + .008) / 1.008 * math.pi / 2).square()
    a /= a[0].clone()
    a[0], a[-1] = 1., 0.
    return a.float()


def _expand(a: torch.Tensor, ndim: int) -> torch.Tensor:
    return a.reshape(a.shape + (1,) * (ndim - a.ndim))


class MarginalNoise:
    def __init__(self, marginal, alpha_bar: torch.Tensor):
        m = torch.as_tensor(marginal, dtype=alpha_bar.dtype, device=alpha_bar.device)
        if m.ndim != 1 or not torch.isfinite(m).all() or (m <= 0).any():
            raise ValueError("All marginal entries must be finite and positive; use train-only pseudocounts.")
        if alpha_bar.ndim != 1 or len(alpha_bar) < 3 or not torch.isfinite(alpha_bar).all():
            raise ValueError("Invalid cumulative noise schedule")
        if alpha_bar[0] != 1 or alpha_bar[-1] != 0 or not torch.all(alpha_bar[:-1] > alpha_bar[1:]):
            raise ValueError("Schedule must decrease strictly from 1 at 0 to 0 at T")
        self.marginal, self.alpha_bar = m / m.sum(), alpha_bar

    def matrix(self, retention):
        a = torch.as_tensor(retention, device=self.marginal.device, dtype=self.marginal.dtype)
        k = len(self.marginal)
        return a[..., None, None] * torch.eye(k, device=a.device) + (1-a[..., None, None]) * self.marginal

    def forward_probs(self, clean: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        one = F.one_hot(clean.long(), len(self.marginal)).to(self.marginal.dtype)
        a = _expand(self.alpha_bar[t], one.ndim)
        return a * one + (1-a) * self.marginal

    def reverse_probs(self, clean_probs: torch.Tensor, current: torch.Tensor,
                      t: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Exact mixture sum_k p0[k] q(y_s=j | y_t=i,y0=k), including skipped steps.

        This O(K) expression is algebraically identical to enumerating the KxK
        posterior table; no clean argmax shortcut or repeated one-step kernel.
        """
        if clean_probs.shape[:-1] != current.shape or clean_probs.shape[-1] != len(self.marginal):
            raise ValueError("Current labels and clean probabilities have incompatible shapes")
        if (s < 0).any() or (t >= len(self.alpha_bar)).any() or (s >= t).any():
            raise ValueError("Reverse steps require 0 <= s < t <= T")
        if not torch.isfinite(clean_probs).all() or (clean_probs < 0).any() or (clean_probs.sum(-1) <= 0).any():
            raise ValueError("Invalid clean-category probabilities")
        p = clean_probs / clean_probs.sum(-1, keepdim=True)
        one = F.one_hot(current.long(), len(self.marginal)).to(p.dtype)
        at = _expand(self.alpha_bar[t], p.ndim)
        ass = _expand(self.alpha_bar[s], p.ndim)
        mi = self.marginal[current.long()].unsqueeze(-1)
        denom = at * one + (1-at) * mi
        weights = p / denom.clamp_min(torch.finfo(p.dtype).tiny)
        prior = ass * weights + (1-ass) * self.marginal * weights.sum(-1, keepdim=True)
        likelihood = (at/ass) * one + (1-at/ass) * mi
        result = prior * likelihood
        return result / result.sum(-1, keepdim=True).clamp_min(torch.finfo(p.dtype).tiny)


def pair_mask(mask: torch.Tensor) -> torch.Tensor:
    n = mask.shape[1]
    return mask[:, :, None] & mask[:, None, :] & ~torch.eye(n, dtype=torch.bool, device=mask.device)[None]


def draw_categories(prob: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    return torch.multinomial(prob.reshape(-1, prob.shape[-1]), 1, generator=generator).reshape(prob.shape[:-1])


def draw_graph(node_probs, edge_probs, mask, generator):
    """Sample each unordered pair exactly once. Retain isolates; no hidden repair."""
    nodes = draw_categories(node_probs, generator).masked_fill(~mask, 0)
    n = mask.shape[1]
    ij = torch.triu_indices(n, n, 1, device=mask.device)
    edges = torch.zeros((len(mask), n, n), dtype=torch.long, device=mask.device)
    if ij.shape[1]:
        draws = draw_categories(edge_probs[:, ij[0], ij[1]], generator)
        edges[:, ij[0], ij[1]] = draws
        edges[:, ij[1], ij[0]] = draws
    return nodes, edges.masked_fill(~pair_mask(mask), 0)


def spectral_q_sample(clean, noise, anchor, alpha_bar, t, mask):
    a = alpha_bar[t, None]
    return (anchor + a.sqrt()*(clean-anchor) + (1-a).sqrt()*noise) * mask


def spectral_reverse(current, clean_prediction, anchor, alpha_bar, t, s, mask):
    """Source-centred DDIM with x0 prediction, safe even at exact zero terminal SNR.

    Keep the inferred residual noise when a clean prediction is changed by
    guidance. Never overwrite the noisy state with an unnoised graph spectrum.
    """
    at, ass = alpha_bar[t, None], alpha_bar[s, None]
    eps = (current-anchor-at.sqrt()*(clean_prediction-anchor))/(1-at).sqrt().clamp_min(1e-12)
    return (anchor + ass.sqrt()*(clean_prediction-anchor) + (1-ass).sqrt()*eps) * mask


def draw_bonds(prob, support, mask, generator):
    """Draw labels 1..K only for existing unordered edges; zero is a sentinel.

    `prob` has K REAL bond categories, not K+1 categories including no-edge.
    No multinomial is evaluated on nonedges, padding or the diagonal.
    """
    if support.dtype != torch.bool or support.shape != (len(mask), mask.shape[1], mask.shape[1]):
        raise ValueError('Bond support must be a boolean [B,N,N] tensor')
    if not torch.equal(support, support.transpose(1, 2)) or (support & ~pair_mask(mask)).any():
        raise ValueError('Bond support must be symmetric, loop-free and unpadded')
    if prob.shape[:-1] != support.shape or prob.shape[-1] < 1:
        raise ValueError('Expected K positive bond categories on [B,N,N,K]')
    out = torch.zeros_like(support, dtype=torch.long)
    active = torch.nonzero(torch.triu(support, diagonal=1), as_tuple=True)
    if len(active[0]):
        if prob.shape[-1] == 1:
            draws = torch.ones(len(active[0]), dtype=torch.long, device=out.device)
        else:
            selected = prob[active]
            if not torch.isfinite(selected).all() or (selected < 0).any() or (selected.sum(-1) <= 0).any():
                raise ValueError('Invalid probabilities on existing bonds')
            draws = draw_categories(selected, generator) + 1
        out[active] = draws
        out[active[0], active[2], active[1]] = draws
    if not torch.equal(out > 0, support):
        raise AssertionError('Bond sampling changed the authoritative topology')
    return out


def bond_forward_probs(noise, clean_edges, t):
    """Noise real bonds; pairs absent from the clean graph have no bond target."""
    q = noise.forward_probs((clean_edges - 1).clamp_min(0), t)
    return torch.where((clean_edges > 0)[..., None], q, noise.marginal)


def bond_reverse_probs(noise, clean_probs, current_edges, t, s):
    """Posterior for persistent bonds; fresh q_s mixture for newly born bonds.

    There is no y_t for a pair absent from the current topology. A new edge
    therefore cannot reuse no-edge as a bond observation or call it a posterior.
    At s=0 its birth distribution is simply the predicted clean bond law.
    The exact posterior claim applies ONLY to the fixed-support bond chain.
    """
    posterior = noise.reverse_probs(clean_probs, (current_edges - 1).clamp_min(0), t, s)
    retention = _expand(noise.alpha_bar[s], clean_probs.ndim)
    births = retention * clean_probs + (1 - retention) * noise.marginal
    return torch.where((current_edges > 0)[..., None], posterior, births)

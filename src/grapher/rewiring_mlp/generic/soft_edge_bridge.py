"""Symmetric binary-edge logit bridge used by generic GraphER.

The stochastic edge state is a two-category centered logit tensor over
{no-edge, edge}.  It is independent from the Laplacian-eigenvalue bridge but
shares the same diffusion progress and predictor.
"""
from __future__ import annotations

import math
import torch


def pair_mask(node_mask: torch.Tensor, *, upper: bool = False) -> torch.Tensor:
    mask = node_mask.bool()
    out = mask[:, :, None] & mask[:, None, :]
    n = mask.shape[1]
    diagonal = torch.eye(n, dtype=torch.bool, device=mask.device)[None]
    out = out & ~diagonal
    if upper:
        out = out & torch.triu(torch.ones((n,n),dtype=torch.bool,device=mask.device),diagonal=1)[None]
    return out


def center_edges(values: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
    values = 0.5 * (values + values.transpose(1,2))
    values = values - values.mean(dim=-1,keepdim=True)
    return values * pair_mask(node_mask)[...,None].to(values.dtype)


def edge_noise(reference: torch.Tensor, node_mask: torch.Tensor, *, generator=None) -> torch.Tensor:
    noise = torch.randn(reference.shape, dtype=reference.dtype, device=reference.device, generator=generator)
    noise = 0.5 * (noise + noise.transpose(1,2))
    noise = noise - noise.mean(dim=-1,keepdim=True)
    return noise * pair_mask(node_mask)[...,None].to(noise.dtype)


def labels_to_logits(labels: torch.Tensor, node_mask: torch.Tensor, smoothing: float = 0.01) -> torch.Tensor:
    if not 0.0 < float(smoothing) < 0.5:
        raise ValueError("edge smoothing must be in (0,0.5).")
    labels = labels.long()
    probs = torch.full((*labels.shape,2), float(smoothing), dtype=torch.float32, device=labels.device)
    probs.scatter_(-1, labels[...,None], 1.0-float(smoothing))
    return center_edges(probs.clamp_min(1e-12).log(), node_mask)


def edge_probabilities(logits: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(center_edges(logits,node_mask),dim=-1)
    mask = pair_mask(node_mask)[...,None].to(probs.dtype)
    # Padded/diagonal locations are never scored; keep deterministic no-edge.
    fallback = torch.zeros_like(probs); fallback[...,0]=1.0
    return probs*mask + fallback*(1-mask)


def bridge_edges(source: torch.Tensor, target: torch.Tensor, progress: torch.Tensor,
                 node_mask: torch.Tensor, sigma: float, *, generator=None) -> torch.Tensor:
    if sigma < 0 or not math.isfinite(float(sigma)):
        raise ValueError("edge bridge sigma must be finite and nonnegative.")
    t = progress.to(source.dtype).view(-1,1,1,1)
    mean = (1-t)*source + t*target
    std = float(sigma)*torch.sqrt((t*(1-t)).clamp_min(0))
    return center_edges(mean + std*edge_noise(source,node_mask,generator=generator), node_mask)


def advance_edges(current: torch.Tensor, clean_prediction: torch.Tensor, t: float, s: float,
                  node_mask: torch.Tensor, sigma: float, *, generator=None) -> torch.Tensor:
    if not (0.0 <= float(t) <= float(s) <= 1.0) or t >= 1.0:
        raise ValueError("Require 0 <= t <= s <= 1 and t < 1 for a bridge transition.")
    if s == t:
        return center_edges(current,node_mask)
    alpha=(s-t)/(1-t)
    variance=float(sigma)**2*(s-t)*(1-s)/(1-t)
    result=current + alpha*(clean_prediction-current)
    if variance>0:
        result=result+math.sqrt(variance)*edge_noise(current,node_mask,generator=generator)
    return center_edges(result,node_mask)

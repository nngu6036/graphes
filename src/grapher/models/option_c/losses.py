"""Clean-target losses; the spectrum is computed from the predicted adjacency."""
from __future__ import annotations

import torch
from torch.nn import functional as F
from .diffusion import pair_mask


def weighted_spectral_loss(predicted, clean_spectrum, mask, *, solver_device="cpu"):
    """Mean per-graph MSE of lambda(W / physical_scale) / sqrt(n).

    Bucket by active node count BEFORE diagonalizing: padding zeros must not be
    mixed into sorted spectra. The CPU transfer, float64 eigvalsh and transfer
    back remain differentiable. No eigenvectors, jitter, detach, or hard graph
    reconstruction enters the loss. Skip this call entirely when its weight=0.
    """
    total = predicted.sum() * 0.
    sizes = mask.sum(1)
    for size in torch.unique(sizes).tolist():
        indices = torch.nonzero(sizes == size, as_tuple=False).flatten()
        block = predicted[indices, :size, :size]
        where = torch.device("cpu") if solver_device == "cpu" else predicted.device
        values = torch.linalg.eigvalsh(block.to(device=where, dtype=torch.float64)) / size**.5
        target = clean_spectrum[indices, :size].to(device=where, dtype=torch.float64)
        error = (values-target).square().mean(-1).sum()
        total = total + error.to(device=predicted.device, dtype=predicted.dtype)
    return total / len(predicted)


def joint_loss(pred, target, basis, weights, spectral_cfg):
    mask = target["mask"]
    upper = pair_mask(mask).triu(1)
    ce = F.cross_entropy(pred["node_logits"].transpose(1, 2), target["x"], reduction="none")
    node = ((ce * mask).sum(1) / mask.sum(1).clamp_min(1)).mean()
    squared = (pred["clean_adjacency"]-target["w"]).square()
    adjacency = ((squared*upper).sum((1, 2)) / upper.sum((1, 2)).clamp_min(1)).mean()
    spectral = (weighted_spectral_loss(pred["clean_adjacency"], target["spectrum"], mask,
                                      solver_device=spectral_cfg["solver_device"])
                if weights["spectral"] > 0 else pred["clean_adjacency"].sum()*0.)
    gh = pred["graphlet_logits"].sum()*0.
    gm = pred["graphlet_mass"].sum()*0.
    by_order = {}
    for i, (k, wk) in enumerate(zip(basis.orders, basis.size_weights)):
        sl = basis.slices[k]
        available = target["graphlet_order_mask"][:, i]
        has_mass = available & (target["mass"][:, i] > 0)
        logh = pred["graphlet_logits"][:, sl].log_softmax(-1)
        h_loss = (F.kl_div(logh[has_mass], target["histogram"][has_mass, sl], reduction="batchmean")
                  if has_mass.any() else logh.sum()*0.)
        m_loss = (F.mse_loss(pred["graphlet_mass"][available, i], target["mass"][available, i])
                  if available.any() else pred["graphlet_mass"][:, i].sum()*0.)
        gh, gm = gh+float(wk)*h_loss, gm+float(wk)*m_loss
        by_order[f"graphlet_{k}"] = h_loss
        by_order[f"mass_{k}"] = m_loss
    logc = pred["clustering_logits"].log_softmax(-1)
    clustering = (F.kl_div(logc, target["clustering"], reduction="batchmean")
                  + F.mse_loss(logc.exp().cumsum(-1), target["clustering"].cumsum(-1)))
    parts = {"node": node, "adjacency": adjacency, "spectral": spectral, "graphlet": gh, "mass": gm,
             "clustering": clustering, "orbit": F.mse_loss(pred["orbit_log_mean"], target["orbit"])}
    total = sum(float(weights[k])*v for k, v in parts.items())
    return total, {**parts, **by_order}

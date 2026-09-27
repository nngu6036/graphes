"""Clean-target losses for Option C.

Schema v1 keeps the original weighted-adjacency eigenvalue auxiliary loss.
Schema v2 adds two topology-aware consistency terms on the SAME predicted
continuous weighted adjacency:

* a soft edge-presence map around the physical no-edge/edge threshold;
* ordinary-degree consistency from that soft topology; and
* normalized-Laplacian eigenvalue consistency from that soft topology.

No hard threshold, categorical edge head, sampled eigenbasis, or detached graph
construction is used during training.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F
from .diffusion import pair_mask


def weighted_spectral_loss(predicted, clean_spectrum, mask, *, solver_device="cpu"):
    """Schema-v1 mean per-graph MSE of lambda(W_scaled) / sqrt(n)."""
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


def soft_presence(predicted, mask, consistency_cfg, *, edge_scale: float):
    """Differentiable approximation of the final edge-present threshold.

    ``predicted`` is in the scaled Option-C weight space. Configuration values
    are declared in PHYSICAL edge units so the same 0.5 threshold has identical
    meaning for binary graphs (scale=1) and molecular bond weights (scale=3).
    """
    threshold = float(consistency_cfg["soft_threshold_physical"]) / float(edge_scale)
    temperature = float(consistency_cfg["temperature_physical"]) / float(edge_scale)
    if temperature <= 0:
        raise ValueError("soft-threshold temperature must be positive")
    return torch.sigmoid((predicted-threshold) / temperature) * pair_mask(mask)


def soft_degree_consistency_loss(predicted, clean_weighted, mask, consistency_cfg, *, edge_scale: float):
    """MSE between normalized soft ordinary degrees and clean ordinary degrees.

    Degrees are divided by n-1 per graph (with the n=1 denominator clamped to
    one), so the loss has a comparable scale across graph sizes.
    """
    soft = soft_presence(predicted, mask, consistency_cfg, edge_scale=edge_scale)
    clean_topology = (clean_weighted > 0).to(predicted.dtype) * pair_mask(mask)
    pred_degree = soft.sum(-1)
    clean_degree = clean_topology.sum(-1)
    normalizer = (mask.sum(1).to(predicted.dtype)-1).clamp_min(1)[:, None]
    squared = ((pred_degree-clean_degree) / normalizer).square() * mask
    return (squared.sum(1) / mask.sum(1).clamp_min(1)).mean()


def _normalized_laplacian(block: torch.Tensor, eps: float) -> torch.Tensor:
    """Evaluator-compatible normalized Laplacian for dense symmetric adjacency.

    The repository evaluator uses diag(1[d_i>0]) - D^-1/2 A D^-1/2 so isolates
    contribute zero eigenvalues. We use the same convention. For the soft graph,
    all n>1 active nodes normally have positive degree, while the branch also
    handles n=1 and numerical underflow exactly.
    """
    degrees = block.sum(-1)
    positive = degrees > float(eps)
    inv_sqrt = torch.where(positive, degrees.clamp_min(float(eps)).rsqrt(), torch.zeros_like(degrees))
    diagonal = torch.diag_embed(positive.to(block.dtype))
    return diagonal - inv_sqrt[..., :, None] * block * inv_sqrt[..., None, :]


def normalized_laplacian_spectral_loss(predicted, clean_spectrum, mask, consistency_cfg, *,
                                        edge_scale: float, solver_device="cpu"):
    """MSE of sorted normalized-Laplacian eigenvalues from soft edge presence.

    The target is computed from the CLEAN DISCRETE topology using the exact same
    isolate convention as the generic evaluator. The prediction uses the
    differentiable sigmoid topology; gradients therefore flow through eigvalsh,
    normalized-Laplacian construction, soft thresholding, and the adjacency head.
    """
    soft = soft_presence(predicted, mask, consistency_cfg, edge_scale=edge_scale)
    total = predicted.sum() * 0.
    sizes = mask.sum(1)
    eps = float(consistency_cfg["normalized_laplacian_epsilon"])
    for size in torch.unique(sizes).tolist():
        indices = torch.nonzero(sizes == size, as_tuple=False).flatten()
        block = soft[indices, :size, :size]
        where = torch.device("cpu") if solver_device == "cpu" else predicted.device
        block64 = block.to(device=where, dtype=torch.float64)
        lap = _normalized_laplacian(block64, eps)
        values = torch.linalg.eigvalsh(lap)
        target = clean_spectrum[indices, :size].to(device=where, dtype=torch.float64)
        error = (values-target).square().mean(-1).sum()
        total = total + error.to(device=predicted.device, dtype=predicted.dtype)
    return total / len(predicted)


def joint_loss(pred, target, basis, weights, spectral_cfg, *, schema_version=1,
               consistency_cfg=None, edge_scale=1.0):
    mask = target["mask"]
    upper = pair_mask(mask).triu(1)
    ce = F.cross_entropy(pred["node_logits"].transpose(1, 2), target["x"], reduction="none")
    node = ((ce * mask).sum(1) / mask.sum(1).clamp_min(1)).mean()
    squared = (pred["clean_adjacency"]-target["w"]).square()
    adjacency = ((squared*upper).sum((1, 2)) / upper.sum((1, 2)).clamp_min(1)).mean()

    if int(schema_version) >= 2:
        if consistency_cfg is None:
            raise ValueError("schema v2 requires consistency_cfg")
        degree = (soft_degree_consistency_loss(pred["clean_adjacency"], target["w"], mask,
                                               consistency_cfg, edge_scale=edge_scale)
                  if weights["degree"] > 0 else pred["clean_adjacency"].sum()*0.)
        spectral = (normalized_laplacian_spectral_loss(
                        pred["clean_adjacency"], target["spectrum"], mask, consistency_cfg,
                        edge_scale=edge_scale, solver_device=spectral_cfg["solver_device"])
                    if weights["spectral"] > 0 else pred["clean_adjacency"].sum()*0.)
    else:
        degree = None
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
    if degree is not None:
        parts["degree"] = degree
    total = sum(float(weights[k])*v for k, v in parts.items())
    return total, {**parts, **by_order}

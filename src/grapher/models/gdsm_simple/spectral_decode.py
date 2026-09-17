"""Option-A decoding and basis-aware feedback, without degree projection.

The denoiser's coordinates are normalized adjacency eigenvalues. The sampled
orthonormal basis is fixed. A rewired graph need not share that basis, so feeding
back only its sorted eigenvalues would not encode the edge changes in U.
Instead project the *swap delta* into the fixed basis and keep the spectral
coefficient ordering using an isotonic least-squares projection.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import torch


def decode_spectral_batch(x0: torch.Tensor, bases: torch.Tensor,
                          mask: torch.Tensor, sizes: torch.Tensor,
                          threshold: float = .5) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode every clean estimate into an undirected binary adjacency.

    Zero padding must not be sorted into real negative/positive eigenvalues.
    Returns binary adjacency, sorted normalized coefficients, and the permutation
    mapping the sorted coefficients back to their original model coordinates.
    This does not fix degrees, edge counts, connectivity, or remove isolates.
    """
    if x0.ndim != 2 or mask.shape != x0.shape or bases.shape != (*x0.shape, x0.shape[1]):
        raise ValueError("Expected x0/mask [B,N] and padded bases [B,N,N].")
    if sizes.shape != (x0.shape[0],) or not torch.equal(mask.sum(1), sizes):
        raise ValueError("sizes must agree with the active-node mask.")
    if not torch.isfinite(x0).all() or not torch.isfinite(bases).all():
        raise FloatingPointError("Nonfinite spectral decode input.")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and in [0,1].")
    values, order = x0.masked_fill(~mask.bool(), float("inf")).sort(dim=-1, stable=True)
    values = values.masked_fill(~mask.bool(), 0.)
    scales = sizes.to(x0.dtype).sqrt().unsqueeze(1)
    soft = torch.bmm(bases * (values * scales).unsqueeze(1), bases.transpose(1, 2))
    soft = .5 * (soft + soft.transpose(1, 2))
    active = mask.unsqueeze(1) & mask.unsqueeze(2)
    diagonal = torch.eye(x0.shape[1], dtype=torch.bool, device=x0.device).unsqueeze(0)
    binary = (soft > threshold) & active & ~diagonal
    return binary, values, order


def isotonic_nondecreasing(values: np.ndarray) -> np.ndarray:
    """Equal-weight L2 projection onto ordered coordinates (pool adjacent violators)."""
    y = np.asarray(values, dtype=np.float64)
    if y.ndim != 1 or not np.isfinite(y).all():
        raise ValueError("Expected a finite 1D coefficient vector.")
    means, counts = [], []
    for value in y:
        means.append(float(value)); counts.append(1)
        while len(means) > 1 and means[-2] > means[-1]:
            count = counts[-2] + counts[-1]
            mean = (means[-2]*counts[-2] + means[-1]*counts[-1])/count
            means[-2:] = [mean]; counts[-2:] = [count]
    return np.repeat(np.asarray(means, dtype=np.float64), counts)


def project_rewiring_feedback(values: np.ndarray, basis: np.ndarray,
                              before: nx.Graph, after: nx.Graph,
                              weight: float) -> tuple[np.ndarray, dict[str, float | bool | str]]:
    """Feed back the part of an accepted edge delta representable in fixed U.

    z_new = Iso(z + rho diag(U^T (A_after-A_before) U) / sqrt(n)).
    This is a heuristic sampling intervention, not exact graph re-encoding.
    Isotonic projection preserves the association between ordered coefficients
    and the fixed eigenbasis columns; sorting the updated coefficients alone
    would instead permute that association.
    """
    z = np.asarray(values, dtype=np.float64)
    u = np.asarray(basis, dtype=np.float64)
    n = len(z)
    if n < 2 or u.shape != (n, n) or not np.isfinite(z).all() or not np.isfinite(u).all():
        raise ValueError("Expected finite coefficients and a matching square basis, n>=2.")
    if not np.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("Feedback weight must be in [0,1].")
    if np.any(np.diff(z) < -1e-10):
        raise ValueError("Feedback coefficients must be in nondecreasing order.")
    if set(before) != set(range(n)) or set(after) != set(range(n)):
        raise ValueError("Feedback graphs must share the indexed nodes 0..n-1.")
    delta = (nx.to_numpy_array(after, nodelist=range(n), dtype=np.float64) -
             nx.to_numpy_array(before, nodelist=range(n), dtype=np.float64))
    projected = np.sum(u * (delta @ u), axis=0)
    raw = z + weight * projected / np.sqrt(n)
    updated = isotonic_nondecreasing(raw)
    residual = delta - (u * projected[None, :]) @ u.T
    norm = float(np.linalg.norm(delta))
    effective = float(np.linalg.norm(updated-z))
    return updated, {
        "mode": "fixed_basis_delta", "weight": float(weight),
        "edge_delta_frobenius": norm,
        "basis_projection_relative_residual": float(np.linalg.norm(residual)/max(norm, 1e-12)),
        "ordering_projection_l2": float(np.linalg.norm(updated-raw)),
        "normalized_coefficient_delta_l2": effective,
        "representable_feedback_nonzero": bool(effective > 1e-12),
    }

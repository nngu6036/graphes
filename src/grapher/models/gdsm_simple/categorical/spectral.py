"""Binary topology spectra and dynamically recomputed, degeneracy-safe proposals."""
from __future__ import annotations
import numpy as np
import torch


def degree_anchor(basis, degrees, ridge=.001, diagonal_weight=1.):
    """Fast ridge seed, NOT an exact degree-constrained graph reconstruction.

    Fit row sums/zero diagonal/zero trace in a sampled training eigenbasis, then
    sort and rescale to the prior's second moment. Projection can increase the
    residual. No degree, trace-moment or basis constraint is applied to generation.
    """
    u, d = np.asarray(basis, dtype=np.float64), np.asarray(degrees, dtype=np.float64)
    n = len(d)
    if n < 1 or u.shape != (n, n) or not np.isfinite(d).all() or np.any(d < 0) or np.any(d >= n):
        raise ValueError("Invalid basis/degree prior")
    if not np.allclose(u.T@u, np.eye(n), atol=1e-5):
        raise ValueError("Prior eigenbasis must be orthonormal")
    if n == 1 or d.sum() == 0:
        return np.zeros(n, np.float32), {"row_sum_rmse": 0., "diagonal_rmse": 0.}
    b, c = u * (u.T@np.ones(n))[None], u*u
    h = b.T@b + diagonal_weight*c.T@c + ridge*np.eye(n)
    kkt = np.block([[h, np.ones((n,1))], [np.ones((1,n)), np.zeros((1,1))]])
    values = np.linalg.solve(kkt, np.r_[b.T@d, 0.])[:n]
    if np.linalg.norm(values) < 1e-10:
        values = np.linspace(-1., 1., n)
    values = np.sort(values - values.mean())
    values *= np.sqrt(d.sum()) / max(np.linalg.norm(values), 1e-12)
    return (values/np.sqrt(n)).astype(np.float32), {
        "row_sum_rmse": float(np.sqrt(np.mean((b@values-d)**2))),
        "diagonal_rmse": float(np.sqrt(np.mean((c@values)**2))),
    }


@torch.no_grad()
def eigenpairs(edges: torch.Tensor, mask: torch.Tensor):
    """Eigendecompose each active binary adjacency, grouping graphs by size.

    Padding zeros must not be sorted among active eigenvalues. The returned
    eigenvalues are normalized by sqrt(n); vectors reconstruct binary adjacency.
    """
    b, n = mask.shape
    dtype = torch.float32
    vals = torch.zeros((b,n), dtype=dtype, device=edges.device)
    vecs = torch.zeros((b,n,n), dtype=dtype, device=edges.device)
    sizes = mask.sum(1)
    for size in sizes.unique().tolist():
        count = int(size)
        ids = torch.where(sizes == count)[0]
        a = (edges[ids,:count,:count] > 0).to(dtype)
        v, u = torch.linalg.eigh(a)
        vals[ids,:count] = v / count**.5
        vecs[ids,:count,:count] = u
    return vals, vecs


def spectral_proposal(clean_values, current_values, current_vectors, mask, tolerance=1e-6):
    """U diag(z0) U^T using the CURRENT graph's basis, not an initial basis.

    Equal current eigenvalues have no canonical eigenvector orientation. Average
    predicted coefficients within such eigenspaces, making this proposal
    independent of sign/rotation choices (including the edgeless graph).
    Gradients still propagate to the predicted coefficients.
    """
    n = mask.shape[1]
    delta = (current_values[:,1:] - current_values[:,:-1]).abs()
    scale = 1 + current_values.abs().amax(1, keepdim=True)
    split = (delta > tolerance*scale) | ~mask[:,1:] | ~mask[:,:-1]
    groups = torch.cat((torch.zeros_like(mask[:,:1], dtype=torch.long), split.long().cumsum(1)), 1)
    totals = torch.zeros_like(clean_values).scatter_add(1, groups, clean_values*mask)
    counts = torch.zeros_like(clean_values).scatter_add(1, groups, mask.to(clean_values.dtype))
    coefficients = (totals/counts.clamp_min(1)).gather(1, groups) * mask
    values = coefficients * mask.sum(1).to(clean_values.dtype).sqrt()[:,None]
    scores = (current_vectors * values[:,None,:]) @ current_vectors.transpose(1,2)
    active = mask[:,:,None] & mask[:,None,:] & ~torch.eye(n, dtype=torch.bool, device=mask.device)[None]
    return (.5*(scores + scores.transpose(1,2))).masked_fill(~active, 0)

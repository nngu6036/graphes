"""Fixed training-eigenvector proposals for the degree-constrained sampler.

This is a generation-only intervention on the existing joint denoiser. It does
not add a topology head, change its parameter shapes, or assert that an exact
binary fixed-degree graph shares the donor eigenbasis. The continuous proposal
and the discrete feasible graph are deliberately different objects.
"""
from __future__ import annotations

import hashlib
import numpy as np
import torch

from .noise import pair_mask


def basis_digest(basis: np.ndarray) -> str:
    """Content hash with explicit shape and a platform-independent float format."""
    u = np.ascontiguousarray(basis, dtype='<f4')
    h = hashlib.sha256()
    h.update(np.asarray(u.shape, dtype='<i8').tobytes())
    h.update(u.tobytes())
    return h.hexdigest()


def sample_training_basis(bank: dict, n: int, rng: np.random.Generator):
    """Uniform same-size draw from the checkpoint's training-only reservoir.

    Missing sizes fail explicitly. No validation/test basis, Haar basis,
    truncation, resizing, or current-graph basis is an admissible substitute.
    Rows are retained in the donor's stored order; the sampled degree multiset
    is assigned to these indexed rows by the surrounding sampler.
    """
    if not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError('Training eigenbasis size must be a positive integer')
    choices = bank.get(n, ())
    if len(choices) == 0:
        supported = sorted(int(k) for k, v in bank.items() if len(v))
        raise ValueError(
            f'No training eigenbasis for sampled size n={n}; supported sizes are {supported}. '
            'training_bank never falls back to Haar, validation/test, or current-graph bases. '
            'Use a size prior supported by the checkpoint training bank.'
        )
    index = int(rng.integers(len(choices)))
    u = np.array(choices[index], dtype=np.float32, copy=True)
    if u.shape != (n, n) or not np.isfinite(u).all():
        raise ValueError(f'Invalid training eigenbasis at n={n}, index={index}')
    ud = u.astype(np.float64)
    error = float(np.max(np.abs(ud.T @ ud - np.eye(n))))
    if error > 2e-5:
        raise ValueError(f'Training eigenbasis is not orthonormal (error={error:.3g})')
    return u, index, {
        'source': 'checkpoint_training_bank',
        'selection': 'uniform_same_size_reservoir',
        'num_nodes': n,
        'bank_index_within_size': index,
        'bank_size_for_n': len(choices),
        'row_order': 'stored_donor_order',
        'basis_sha256': basis_digest(u),
        'orthogonality_max_error': error,
        'fixed_for_entire_trajectory': True,
        'fallback': False,
    }


def fixed_basis_proposal(clean_values: torch.Tensor, basis: torch.Tensor,
                         mask: torch.Tensor) -> torch.Tensor:
    """Return masked S = U diag(sqrt(n) * z_hat) U^T, in saved donor columns.

    The denoiser emits ascending, normalized adjacency eigenvalues. U was saved
    with ascending eigenvalue columns, so no independent sort/rotation of U is
    performed. Repeated donor eigenspaces use the saved orientation, rather
    than averaging coefficients according to an unrelated current graph.

    Symmetry is enforced numerically; diagonal and padding do not score edges.
    Orthonormality is checked once when sampling, not in every neural step.
    """
    if mask.dtype != torch.bool or clean_values.ndim != 2 or clean_values.shape != mask.shape:
        raise ValueError('Expected clean eigenvalues/mask [B,N] and a boolean mask')
    if basis.shape != (*mask.shape, mask.shape[1]):
        raise ValueError('Expected padded fixed eigenbasis [B,N,N]')
    if basis.device != clean_values.device or mask.device != clean_values.device:
        raise ValueError('Basis, eigenvalues and mask must be on the same device')
    sizes = mask.sum(1)
    expected = torch.arange(mask.shape[1], device=mask.device)[None] < sizes[:, None]
    if not bool((sizes > 0).all()) or not torch.equal(mask, expected):
        raise ValueError('Fixed-basis decoding requires nonempty prefix-contiguous masks')
    if not bool(torch.isfinite(clean_values[mask]).all() & torch.isfinite(basis).all()):
        raise FloatingPointError('Nonfinite fixed-basis spectral input')
    values = clean_values.masked_fill(~mask, 0.) * sizes.to(clean_values.dtype).sqrt()[:, None]
    u = basis.to(dtype=clean_values.dtype)
    scores = (u * values[:, None, :]) @ u.transpose(1, 2)
    return (.5 * (scores + scores.transpose(1, 2))).masked_fill(~pair_mask(mask), 0.)

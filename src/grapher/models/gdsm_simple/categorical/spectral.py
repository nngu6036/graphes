"""Binary topology spectra and dynamically recomputed, degeneracy-safe proposals."""
from __future__ import annotations
import os
import warnings
from collections import Counter

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


# Runtime-only policy: no architecture/config/checkpoint changes are required.
# Do not perturb the adjacency to split repeated eigenvalues: their eigenspaces
# are intentional and spectral_proposal handles the corresponding ambiguity.
_EIGH_TOLERANCE = 1e-8
_EIGH_CPU_GROUPS: set[tuple[str, int]] = set()
_EIGH_COUNTS: Counter = Counter()


def eigh_diagnostics() -> dict:
    """Return process-local solver counters (not graph-quality measurements)."""
    return {
        "backend_policy": _eigh_backend(),
        "solve_dtype": "float64",
        "output_dtype": "float32",
        "quality_tolerance": _EIGH_TOLERANCE,
        "counts": dict(_EIGH_COUNTS),
        "cached_cpu_groups": [
            {"device": device, "nodes": n} for device, n in sorted(_EIGH_CPU_GROUPS)
        ],
    }


def _eigh_backend() -> str:
    value = os.environ.get("GDSM_EIGH_BACKEND", "auto").strip().lower()
    if value not in {"auto", "cpu"}:
        raise ValueError("GDSM_EIGH_BACKEND must be 'auto' or 'cpu'")
    return value


def _recoverable_eigh_error(error: RuntimeError) -> bool:
    """Retry solver failures only; never disguise OOM or a broken CUDA context."""
    text = str(error).lower()
    if any(token in text for token in (
        "out of memory", "alloc_failed", "illegal memory", "device-side assert",
        "device side assert", "launch failure", "launch timed out",
    )):
        return False
    error_type = getattr(torch.linalg, "LinAlgError", ())
    return isinstance(error, error_type) or any(token in text for token in (
        "linalg.eigh", "linalg_eigh", "failed to converge",
        "cusolver_status_execution_failed", "cusolver_status_internal_error",
    ))


def _torch_pairs_valid(a: torch.Tensor, v: torch.Tensor, u: torch.Tensor) -> bool:
    if v.shape != a.shape[:-1] or u.shape != a.shape:
        return False
    if not bool(torch.isfinite(v).all() & torch.isfinite(u).all()):
        return False
    residual = torch.linalg.vector_norm(a @ u - u * v[:, None, :], dim=(-2, -1))
    scale = torch.linalg.vector_norm(a, dim=(-2, -1)).clamp_min(1.)
    identity = torch.eye(a.shape[-1], dtype=a.dtype, device=a.device)
    orthogonal = (u.transpose(-2, -1) @ u - identity).abs().amax(dim=(-2, -1))
    return bool(
        (residual <= _EIGH_TOLERANCE * scale).all()
        & (orthogonal <= _EIGH_TOLERANCE).all()
        & (v[:, 1:] >= v[:, :-1]).all()
    )


def _numpy_pairs_valid(a: np.ndarray, v: np.ndarray, u: np.ndarray) -> np.ndarray:
    """A per-matrix check permits retaining good results from a partial batch."""
    if v.shape != a.shape[:-1] or u.shape != a.shape:
        return np.zeros(len(a), dtype=bool)
    with np.errstate(invalid="ignore", over="ignore"):
        residual = np.linalg.norm(a @ u - u * v[:, None, :], axis=(-2, -1))
        scale = np.maximum(np.linalg.norm(a, axis=(-2, -1)), 1.)
        orthogonal = np.max(np.abs(u.swapaxes(-2, -1) @ u - np.eye(a.shape[-1])), axis=(-2, -1))
    return (
        np.isfinite(v).all(axis=1) & np.isfinite(u).all(axis=(1, 2))
        & (residual <= _EIGH_TOLERANCE * scale)
        & (orthogonal <= _EIGH_TOLERANCE)
        & (np.diff(v, axis=1) >= 0).all(axis=1)
    )


def _cpu_eigh(a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve the ORIGINAL matrices in CPU float64, using independent LAPACK calls.

    NumPy's batch/individual divide-and-conquer path is tried first. An uncommon
    failure is retried with SciPy's symmetric QR driver ('ev'), not with jitter.
    Both backends must pass finite, residual, orthogonality and ordering checks.
    """
    array = a.detach().to(device="cpu", dtype=torch.float64).contiguous().numpy()
    _EIGH_COUNTS["cpu_batch_calls"] += 1
    try:
        values, vectors = np.linalg.eigh(array)
        valid = _numpy_pairs_valid(array, values, vectors)
        if values.shape != array.shape[:-1] or vectors.shape != array.shape:
            raise np.linalg.LinAlgError("Invalid batch eigenpair shapes")
    except np.linalg.LinAlgError:
        values = np.zeros(array.shape[:-1], dtype=np.float64)
        vectors = np.zeros_like(array)
        valid = np.zeros(len(array), dtype=bool)
    for index in np.flatnonzero(~valid):
        reason = "Eigenpair quality check failed"
        _EIGH_COUNTS["cpu_individual_calls"] += 1
        try:
            v, u = np.linalg.eigh(array[index])
            if not _numpy_pairs_valid(array[index:index+1], v[None], u[None])[0]:
                raise np.linalg.LinAlgError(reason)
        except np.linalg.LinAlgError as error:
            reason = str(error)
            _EIGH_COUNTS["cpu_qr_calls"] += 1
            try:
                from scipy.linalg import eigh as scipy_eigh
                v, u = scipy_eigh(array[index], driver="ev", check_finite=True)
                if not _numpy_pairs_valid(array[index:index+1], v[None], u[None])[0]:
                    raise np.linalg.LinAlgError("CPU QR eigenpair quality check failed")
            except (ImportError, np.linalg.LinAlgError, ValueError) as final_error:
                raise RuntimeError(
                    f"GDSM eigenpairs: CPU float64 solvers failed for matrix {index} "
                    f"of size {array.shape[-1]} (original device {a.device}). "
                    f"NumPy: {reason}; QR: {final_error}. No graph was changed or skipped."
                ) from final_error
        values[index], vectors[index] = v, u
    return (
        torch.from_numpy(np.ascontiguousarray(values)).to(a.device),
        torch.from_numpy(np.ascontiguousarray(vectors)).to(a.device),
    )


def _stable_eigh(a: torch.Tensor, backend: str) -> tuple[torch.Tensor, torch.Tensor]:
    key = (str(a.device), a.shape[-1])
    if backend == "cpu" or key in _EIGH_CPU_GROUPS:
        _EIGH_COUNTS["cpu_routed_batches"] += 1
        return _cpu_eigh(a)
    _EIGH_COUNTS["torch_float64_calls"] += 1
    try:
        values, vectors = torch.linalg.eigh(a)
        if _torch_pairs_valid(a, values, vectors):
            return values, vectors
        reason = "nonfinite, inaccurate, nonorthogonal or unordered eigenpairs"
    except RuntimeError as error:
        if not _recoverable_eigh_error(error):
            raise
        reason = str(error)
    # Cache only AFTER a successful fallback. Avoid triggering the same failing
    # CUDA batch solver on every diffusion step / optimizer batch.
    result = _cpu_eigh(a)
    _EIGH_COUNTS["fallback_batches"] += 1
    _EIGH_CPU_GROUPS.add(key)
    warnings.warn(
        f"GDSM eigenpairs: {a.device} float64 eigh failed for n={a.shape[-1]} "
        f"({reason[:300]}). Recovered with CPU float64 LAPACK; subsequent n={a.shape[-1]} "
        f"groups on {a.device} will use CPU in this process. "
        "Graph matrices are unchanged; eigenpairs return to the original device.",
        RuntimeWarning, stacklevel=3,
    )
    return result


@torch.no_grad()
def eigenpairs(edges: torch.Tensor, mask: torch.Tensor):
    """Recompute the current graph's eigenpairs without perturbing its topology.

    Solve active binary adjacency blocks in float64 (including on CUDA). A
    convergence/quality failure uses CPU float64 LAPACK. Set GDSM_EIGH_BACKEND=cpu
    to bypass CUDA eigh entirely, while leaving the neural network on the GPU.
    Edgeless graphs use an exact zero spectrum/identity basis. Padding never
    enters a graph's spectrum. Outputs keep the existing float32/device contract;
    values are normalized by sqrt(n). Only this discrete feature computation is
    under no_grad; gradients through the predicted spectral proposal are intact.
    """
    backend = _eigh_backend()
    if mask.ndim != 2 or mask.dtype != torch.bool:
        raise ValueError("GDSM eigenpairs expects a boolean node mask [B,N]")
    b, n = mask.shape
    if edges.shape != (b, n, n) or edges.device != mask.device:
        raise ValueError("GDSM eigenpairs expects edges [B,N,N] and mask on the same device")
    if edges.is_complex():
        raise ValueError("GDSM eigenpairs expects real categorical edge labels")
    if edges.is_floating_point():
        if not bool(torch.isfinite(edges).all()):
            raise ValueError("GDSM eigenpairs: nonfinite categorical labels before binarization")
        if not bool((edges == edges.round()).all()):
            raise ValueError("GDSM eigenpairs: edge categories must be integers, not scores")
    if bool((edges < 0).any()):
        raise ValueError("GDSM eigenpairs: edge categories must be nonnegative (0 = no edge)")
    sizes = mask.sum(1)
    if bool((sizes <= 0).any()) or not torch.equal(
        mask, torch.arange(n, device=mask.device)[None, :] < sizes[:, None]
    ):
        raise ValueError("GDSM eigenpairs requires nonempty graphs and prefix-contiguous node masks")
    vals = torch.zeros((b, n), dtype=torch.float32, device=edges.device)
    vecs = torch.zeros((b, n, n), dtype=torch.float32, device=edges.device)
    for count in sizes.unique().tolist():
        count = int(count)
        ids = torch.where(sizes == count)[0]
        labels = edges[ids, :count, :count]
        if not torch.equal(labels, labels.transpose(-2, -1)) or bool(labels.diagonal(dim1=-2, dim2=-1).any()):
            raise ValueError("GDSM eigenpairs: active edges must be symmetric and have zero diagonal")
        a = (labels > 0).to(dtype=torch.float64).contiguous()
        v = torch.zeros((len(ids), count), dtype=torch.float64, device=edges.device)
        u = torch.eye(count, dtype=torch.float64, device=edges.device).expand(len(ids), -1, -1).clone()
        nonzero = a.bool().any(dim=-1).any(dim=-1)
        _EIGH_COUNTS["edgeless_graphs"] += int((~nonzero).sum())
        if bool(nonzero.any()):
            v[nonzero], u[nonzero] = _stable_eigh(a[nonzero], backend)
        vals[ids, :count] = (v / count**.5).float()
        vecs[ids, :count, :count] = u.float()
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

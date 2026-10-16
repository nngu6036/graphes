"""Generation-only perturbations of sampled Laplacian eigenbases.

The GraphES topology generator samples a full Laplacian eigenbasis from a
training-only reservoir.  This module provides a controlled way to move away
from the exact donor basis without changing the generated eigenvalues.

Only the non-constant Laplacian modes are rotated.  The first eigenvector
``1/sqrt(n)`` is left untouched, so ``L 1 = 0`` remains exact up to floating
point error.  Rotations are products of small Givens rotations and are
therefore orthogonal by construction.

Two generation-time stages are supported:

``reverse``
    Perturb the sampled basis before reverse diffusion and keep the perturbed
    basis fixed for the complete reverse trajectory.

``decode``
    Run reverse diffusion in the original training basis, then perturb the
    basis only for the final continuous-matrix reconstruction and threshold.

The perturbation has its own RNG stream so changing its strength does not
consume donor-selection or diffusion noise randomness.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch


DEFAULT_SEED_OFFSET = 730_001


def default_config() -> dict[str, Any]:
    return {
        "enabled": False,
        "stage": "decode",
        "method": "givens",
        "pairing": "adjacent",
        "sigma": 0.0,
        "num_rotations": 1,
        "max_abs_angle": None,
        "seed_offset": DEFAULT_SEED_OFFSET,
        "preserve_zero_mode": True,
    }


def resolve_config(sample_cfg: Mapping[str, Any] | None) -> dict[str, Any]:
    result = default_config()
    if not sample_cfg:
        return result
    raw = sample_cfg.get("eigenbasis_perturbation", {}) or {}
    if not isinstance(raw, Mapping):
        raise TypeError("sample.eigenbasis_perturbation must be a mapping")
    result.update(dict(raw))
    stage = str(result.get("stage", "decode")).strip().lower()
    aliases = {
        "beginning": "reverse",
        "initial": "reverse",
        "trajectory": "reverse",
        "reverse_process": "reverse",
        "final": "decode",
        "final_decode": "decode",
        "decoding": "decode",
    }
    result["stage"] = aliases.get(stage, stage)
    result["method"] = str(result.get("method", "givens")).strip().lower()
    result["pairing"] = str(result.get("pairing", "adjacent")).strip().lower()
    result["enabled"] = bool(result.get("enabled", False))
    result["sigma"] = float(result.get("sigma", 0.0))
    result["num_rotations"] = int(result.get("num_rotations", 1))
    max_abs = result.get("max_abs_angle", None)
    result["max_abs_angle"] = None if max_abs in {None, "", "none", "None"} else float(max_abs)
    result["seed_offset"] = int(result.get("seed_offset", DEFAULT_SEED_OFFSET))
    result["preserve_zero_mode"] = bool(result.get("preserve_zero_mode", True))
    validate_config(result)
    return result


def validate_config(config: Mapping[str, Any]) -> None:
    unknown = sorted(
        set(config)
        - {
            "enabled",
            "stage",
            "method",
            "pairing",
            "sigma",
            "num_rotations",
            "max_abs_angle",
            "seed_offset",
            "preserve_zero_mode",
        }
    )
    if unknown:
        raise ValueError(f"Unknown eigenbasis perturbation settings: {unknown}")
    if str(config.get("stage", "decode")) not in {"reverse", "decode"}:
        raise ValueError("eigenbasis perturbation stage must be 'reverse' or 'decode'")
    if str(config.get("method", "givens")) != "givens":
        raise ValueError("eigenbasis perturbation currently supports method=givens only")
    if str(config.get("pairing", "adjacent")) not in {"adjacent", "random"}:
        raise ValueError("eigenbasis perturbation pairing must be adjacent or random")
    sigma = float(config.get("sigma", 0.0))
    if not math.isfinite(sigma) or sigma < 0.0:
        raise ValueError("eigenbasis perturbation sigma must be finite and non-negative")
    rotations = int(config.get("num_rotations", 1))
    if rotations < 0:
        raise ValueError("eigenbasis perturbation num_rotations must be non-negative")
    max_abs = config.get("max_abs_angle", None)
    if max_abs is not None:
        max_abs = float(max_abs)
        if not math.isfinite(max_abs) or max_abs <= 0.0:
            raise ValueError("eigenbasis perturbation max_abs_angle must be positive when set")
    if not bool(config.get("preserve_zero_mode", True)):
        raise ValueError(
            "Laplacian eigenbasis perturbation must preserve the constant zero mode"
        )


def enabled(config: Mapping[str, Any]) -> bool:
    return bool(config.get("enabled", False)) and float(config.get("sigma", 0.0)) > 0.0 and int(
        config.get("num_rotations", 0)
    ) > 0


def _orthogonality_error(matrix: torch.Tensor) -> float:
    n = matrix.shape[0]
    ident = torch.eye(n, dtype=matrix.dtype, device=matrix.device)
    return float(torch.linalg.matrix_norm(matrix.transpose(0, 1) @ matrix - ident, ord="fro").item())


def _zero_mode_error(matrix: torch.Tensor, n: int) -> float:
    if n <= 0:
        return 0.0
    target = torch.full((n,), 1.0 / math.sqrt(float(n)), dtype=matrix.dtype, device=matrix.device)
    first = matrix[:n, 0]
    # Eigenvectors are sign-indeterminate.  The training Laplacian convention
    # normally fixes +1/sqrt(n), but use sign-invariant comparison for auditing.
    return float(min(torch.linalg.vector_norm(first - target), torch.linalg.vector_norm(first + target)).item())


def perturb_laplacian_basis(
    basis: torch.Tensor,
    num_nodes: torch.Tensor,
    *,
    config: Mapping[str, Any],
    rng: np.random.Generator,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Apply independent small Givens rotations to a padded Laplacian basis.

    Args:
        basis: ``[B,N,N]`` orthonormal matrices.  For each row the active
            Laplacian basis occupies the leading ``n x n`` block and padding is
            an identity completion.
        num_nodes: active node counts ``[B]``.
        config: resolved perturbation configuration.
        rng: dedicated NumPy generator.

    Returns:
        Perturbed basis and one diagnostic record per graph.
    """
    if basis.ndim != 3 or basis.shape[1] != basis.shape[2]:
        raise ValueError("basis must have shape [B,N,N]")
    if num_nodes.ndim != 1 or num_nodes.shape[0] != basis.shape[0]:
        raise ValueError("num_nodes must have shape [B]")
    validate_config(config)

    out = basis.clone()
    sigma = float(config.get("sigma", 0.0))
    requested_rotations = int(config.get("num_rotations", 0))
    pairing = str(config.get("pairing", "adjacent"))
    max_abs = config.get("max_abs_angle", None)
    active = enabled(config)
    rows: list[dict[str, Any]] = []

    for row in range(out.shape[0]):
        n = int(num_nodes[row].item())
        if n < 1 or n > out.shape[1]:
            raise ValueError(f"invalid active node count n={n}")
        before = basis[row].clone()
        rotations: list[dict[str, Any]] = []

        # Columns 1,...,n-1 are the nonconstant Laplacian modes.  At least two
        # such modes are required for a nontrivial rotation.
        if active and n >= 3:
            for _ in range(requested_rotations):
                if pairing == "adjacent":
                    left = int(rng.integers(1, n - 1))
                    right = left + 1
                else:
                    pair = rng.choice(np.arange(1, n, dtype=np.int64), size=2, replace=False)
                    left, right = int(pair[0]), int(pair[1])
                theta = float(rng.normal(0.0, sigma))
                if max_abs is not None:
                    theta = float(np.clip(theta, -float(max_abs), float(max_abs)))
                c = math.cos(theta)
                s = math.sin(theta)
                col_left = out[row, :, left].clone()
                col_right = out[row, :, right].clone()
                out[row, :, left] = c * col_left + s * col_right
                out[row, :, right] = -s * col_left + c * col_right
                rotations.append({"left_rank": left, "right_rank": right, "angle": theta})

        after = out[row]
        # The Givens updates are orthogonal in exact arithmetic.  These audits
        # catch accidental rotation of the zero mode or padding corruption.
        ortho_before = _orthogonality_error(before)
        ortho_after = _orthogonality_error(after)
        zero_before = _zero_mode_error(before, n)
        zero_after = _zero_mode_error(after, n)
        delta = float(torch.linalg.matrix_norm(after - before, ord="fro").item())
        active_delta = float(
            torch.linalg.matrix_norm(after[:n, :n] - before[:n, :n], ord="fro").item()
        )
        angles = [float(item["angle"]) for item in rotations]
        rows.append(
            {
                "enabled": bool(active),
                "num_nodes": n,
                "requested_rotations": requested_rotations,
                "applied_rotations": len(rotations),
                "pairing": pairing,
                "sigma": sigma,
                "frobenius_basis_change": delta,
                "active_frobenius_basis_change": active_delta,
                "orthogonality_error_before": ortho_before,
                "orthogonality_error_after": ortho_after,
                "zero_mode_error_before": zero_before,
                "zero_mode_error_after": zero_after,
                "angle_abs_mean": float(np.mean(np.abs(angles))) if angles else 0.0,
                "angle_abs_max": float(np.max(np.abs(angles))) if angles else 0.0,
                "rotations": rotations,
            }
        )

    return out, rows


def aggregate_diagnostics(config: Mapping[str, Any], rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    def mean(key: str) -> float | None:
        values = [float(row[key]) for row in rows if key in row]
        return float(np.mean(values)) if values else None

    return {
        "enabled": bool(enabled(config)),
        "stage": str(config.get("stage", "decode")),
        "method": str(config.get("method", "givens")),
        "pairing": str(config.get("pairing", "adjacent")),
        "sigma": float(config.get("sigma", 0.0)),
        "num_rotations": int(config.get("num_rotations", 0)),
        "max_abs_angle": config.get("max_abs_angle", None),
        "num_graphs": len(rows),
        "mean_applied_rotations": mean("applied_rotations"),
        "mean_active_frobenius_basis_change": mean("active_frobenius_basis_change"),
        "mean_abs_angle": mean("angle_abs_mean"),
        "max_abs_angle_realized": (
            max((float(row.get("angle_abs_max", 0.0)) for row in rows), default=0.0)
            if rows else None
        ),
        "max_orthogonality_error_after": (
            max((float(row.get("orthogonality_error_after", 0.0)) for row in rows), default=0.0)
            if rows else None
        ),
        "max_zero_mode_error_after": (
            max((float(row.get("zero_mode_error_after", 0.0)) for row in rows), default=0.0)
            if rows else None
        ),
        "config": dict(config),
    }

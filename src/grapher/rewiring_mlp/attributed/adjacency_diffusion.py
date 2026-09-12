"""One categorical adjacency state, deterministic signed spectral views.

There is no eigenvector decoder and no independent spectral stochastic process.
Every spectrum is computed from the same symmetric, masked pair probabilities.
Supports binary edges (one present category) as well as multiple bond categories.
The molecular pipeline continues to use its existing typed prior and hard swaps.
"""
from __future__ import annotations

from contextlib import nullcontext
import math
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from .soft_edge_bridge import edge_probabilities, pair_mask

LEGACY_MODE = "laplacian_independent"
ADJACENCY_MODE = "adjacency_derived"
ADJACENCY_TYPE = "joint_typed_adjacency"
LEGACY_TYPE = "joint_typed_soft_edge"
VIEWS = ("topology", "bond_weighted", "per_bond")
NORMALIZATIONS = ("size_bound", "none")
VERSION = "symmetric_categorical_adjacency_derived_spectra_v1"


def resolve_mode(config: Mapping[str, Any]) -> str:
    """Require an explicit opt-in; never reinterpret an old checkpoint as adjacency."""
    predictor = config.get("attributed_predictor", {}).get("type", LEGACY_TYPE)
    mode = config.get("edge_diffusion", {}).get("spectral_mode", LEGACY_MODE)
    if predictor not in (LEGACY_TYPE, ADJACENCY_TYPE):
        raise ValueError(f"Unsupported attributed predictor type: {predictor!r}.")
    expected = ADJACENCY_MODE if predictor == ADJACENCY_TYPE else LEGACY_MODE
    if mode != expected:
        raise ValueError(f"{predictor} requires edge_diffusion.spectral_mode={expected!r}; "
                         "new adjacency checkpoints are not Laplacian checkpoints.")
    return mode


def validate_views(views: Sequence[str]) -> tuple[str, ...]:
    if isinstance(views, str) or not views:
        raise ValueError("adjacency_spectrum.views must be a nonempty list.")
    result = tuple(views)
    if len(set(result)) != len(result) or any(v not in VIEWS for v in result):
        raise ValueError(f"Use unique adjacency views from {VIEWS}.")
    return result


def validate_weights(weights: Sequence[float]) -> tuple[float, ...]:
    if isinstance(weights, (str, bytes)) or not weights:
        raise ValueError("At least one positive present-edge weight is required.")
    if any(isinstance(w, bool) for w in weights):
        raise ValueError("Boolean bond weights are not allowed.")
    result = tuple(float(w) for w in weights)
    if any(not math.isfinite(w) or w <= 0 for w in result):
        raise ValueError("Bond weights must be finite and strictly positive.")
    return result


def settings(config: Mapping[str, Any], edge_types: Sequence[Any]) -> dict[str, Any]:
    """Resolve explicit physical weights in category order, not category IDs."""
    values = config.get("adjacency_spectrum", {}) or {}
    unknown = set(values) - {"views", "bond_weights", "normalization"}
    if unknown:
        raise ValueError(f"Unknown adjacency_spectrum settings: {sorted(unknown)}")
    views = validate_views(values.get("views", ["topology", "bond_weighted"]))
    normal = values.get("normalization", "size_bound")
    if normal not in NORMALIZATIONS:
        raise ValueError(f"adjacency_spectrum.normalization must be one of {NORMALIZATIONS}.")
    raw = values.get("bond_weights")
    if raw is None:
        raw = config.get("typed_signature", {}).get("bond_orders")
    if raw is None:
        if "bond_weighted" in views:
            raise ValueError("Specify adjacency_spectrum.bond_weights (or typed_signature.bond_orders). "
                             "Bond category IDs are not implicitly interpreted as weights.")
        weights = [1.0] * len(edge_types)
    else:
        if not isinstance(raw, Mapping):
            raise ValueError("bond_weights must map each present edge category to its weight.")
        by_key = {str(k): v for k, v in raw.items()}
        if len(by_key) != len(raw) or set(by_key) != {str(r) for r in edge_types}:
            raise ValueError("bond_weights keys must match present-edge categories exactly; no-bond is implicit.")
        weights = [by_key[str(r)] for r in edge_types]
    weights = validate_weights(weights)
    if len(weights) != len(edge_types):
        raise ValueError("Bond weight/category count mismatch.")
    return {"views": list(views), "bond_weights": list(weights), "normalization": normal}


def view_names(views: Sequence[str], edge_types: Sequence[Any]) -> list[str]:
    names = []
    for view in validate_views(views):
        names.extend([f"bond_type_{r}" for r in edge_types] if view == "per_bond" else [view])
    return names


def adjacency_views(probabilities: torch.Tensor, node_mask: torch.Tensor,
                    bond_weights: Sequence[float], views: Sequence[str]) -> torch.Tensor:
    """Return [B,C,N,N] zero-diagonal symmetric expected-adjacency views.

    Input class 0 is no-edge. Invalid pairs are removed before taking any view.
    No hard category sampling, clipping, or valence repair is performed here.
    """
    views = validate_views(views)
    weights = validate_weights(bond_weights)
    b, n = node_mask.shape
    if probabilities.ndim != 4 or probabilities.shape != (b, n, n, len(weights) + 1):
        raise ValueError("Expected probabilities [B,N,N,R+1] in the checkpoint's category order.")
    pm = pair_mask(node_mask)
    p = (probabilities + probabilities.transpose(1, 2)) * 0.5
    p = p * pm.unsqueeze(-1)
    present = p[..., 1:]
    channels = []
    for view in views:
        if view == "topology":
            channels.append(present.sum(-1))
        elif view == "bond_weighted":
            w = present.new_tensor(weights)
            channels.append((present * w).sum(-1))
        else:
            channels.extend(present.unbind(-1))
    return torch.stack(channels, dim=1)


def masked_eigenvalues(matrices: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
    """Signed ascending eigenvalues of each VALID submatrix, padded on the right.

    Decomposing padded matrices would mix padded zeros into the signed spectrum.
    Gather valid nodes first. No positivity projection and no fixed zero mode.
    Eigenvectors are not exposed. Preserve float64; promote fp16/bf16 to fp32.
    """
    if node_mask.ndim != 2 or node_mask.dtype != torch.bool:
        raise ValueError("node_mask must be bool [B,N].")
    b, n = node_mask.shape
    if matrices.ndim != 4 or matrices.shape[0] != b or matrices.shape[2:] != (n, n):
        raise ValueError("Adjacency views must have shape [B,C,N,N].")
    if not matrices.is_floating_point() or not torch.isfinite(matrices).all():
        raise ValueError("Adjacency views must be finite floating point matrices.")
    # A compatible context for CPU/CUDA mixed precision; the eigensolver itself
    # always sees float32 or float64. No random perturbation of repeated modes.
    context = (torch.autocast(device_type=matrices.device.type, enabled=False)
               if matrices.device.type in ("cpu", "cuda") else nullcontext())
    with context:
        work = matrices if matrices.dtype in (torch.float32, torch.float64) else matrices.float()
        rows = []
        for index in range(b):
            valid = node_mask[index].nonzero(as_tuple=False).flatten()
            m = valid.numel()
            if m:
                block = work[index].index_select(-2, valid).index_select(-1, valid)
                eigenvalues = torch.linalg.eigvalsh((block + block.transpose(-1, -2)) * 0.5)
                rows.append(F.pad(eigenvalues, (0, n - m)))
            else:
                rows.append(work[index].sum((-1, -2))[:, None].expand(-1, n) * 0)
        return torch.stack(rows)


def spectral_features(probabilities: torch.Tensor, node_mask: torch.Tensor,
                      bond_weights: Sequence[float], views: Sequence[str],
                      normalization: str = "size_bound") -> dict[str, torch.Tensor]:
    """Derive matrices, eigenvalues and normalized features from ONE state.

    size_bound divides each view by (n-1) times its maximum edge weight. The
    bound depends only on graph size and the configured coding, not on a target.
    It does not enforce hard degrees on the soft adjacency.
    """
    views = validate_views(views)
    weights = validate_weights(bond_weights)
    if normalization not in NORMALIZATIONS:
        raise ValueError(f"Unknown adjacency normalization: {normalization!r}.")
    matrices = adjacency_views(probabilities, node_mask, weights, views)
    spectra = masked_eigenvalues(matrices, node_mask)
    bounds = []
    for view in views:
        if view == "per_bond":
            bounds.extend([1.0] * len(weights))
        else:
            bounds.append(max(weights) if view == "bond_weighted" else 1.0)
    sizes = node_mask.sum(-1)
    scales = spectra.new_ones((len(sizes), len(bounds)))
    if normalization == "size_bound":
        scales = (sizes - 1).clamp_min(1).to(spectra)[:, None] * spectra.new_tensor(bounds)[None]
    spectral_mask = torch.arange(node_mask.shape[1], device=node_mask.device)[None] < sizes[:, None]
    return {"matrices": matrices, "spectra": spectra, "normalized": spectra / scales[..., None],
            "scale": scales, "mask": spectral_mask}


def features_from_logits(logits: torch.Tensor, node_mask: torch.Tensor,
                         bond_weights: Sequence[float], views: Sequence[str],
                         normalization: str = "size_bound") -> dict[str, torch.Tensor]:
    return spectral_features(edge_probabilities(logits, node_mask), node_mask,
                             bond_weights, views, normalization)


def state_metadata(model) -> dict[str, Any]:
    if model.spectral_mode != ADJACENCY_MODE:
        return {"mode": LEGACY_MODE, "diffused_state": "edge_logits_and_optional_laplacian_spectrum",
                "independent_spectral_diffusion": bool(model.spectral_enabled)}
    return {"version": VERSION, "mode": ADJACENCY_MODE,
            "diffused_state": "symmetric_centered_categorical_adjacency_logits",
            "independent_spectral_diffusion": False, "eigenvectors_generated": False,
            "spectral_feature_source": "same_soft_adjacency_probabilities",
            "spectral_features_enabled": bool(model.spectral_enabled),
            "spectral_outputs_enabled": bool(model.adjacency_output_spectra),
            "views": list(model.adjacency_views),
            "view_names": view_names(model.adjacency_views, model.edge_types),
            "edge_categories": list(model.edge_types), "no_edge_category_index": 0,
            "bond_weights": list(model.adjacency_bond_weights),
            "normalization": model.adjacency_normalization,
            "endpoint_convention": "same_smoothed_one_hot_categories_as_logit_loss"}


def validate_model_config(model, config: Mapping[str, Any]) -> None:
    """Check semantics at sampling/diagnostics, even when tensor sizes coincide."""
    mode = resolve_mode(config)
    if mode != model.spectral_mode:
        raise ValueError("Diffusion mode differs from checkpoint; train a matching adjacency model.")
    diff = config.get("edge_diffusion", {})
    if bool(diff.get("spectral_enabled", True)) != model.spectral_enabled:
        raise ValueError("spectral_enabled differs from checkpoint; this changes the denoiser architecture.")
    if not math.isclose(float(diff.get("smoothing", 0.01)), model.smoothing, rel_tol=0, abs_tol=1e-12):
        raise ValueError("Endpoint smoothing differs from the trained checkpoint.")
    if mode == ADJACENCY_MODE:
        resolved = settings(config, model.edge_types)
        expected = {"views": list(model.adjacency_views), "bond_weights": list(model.adjacency_bond_weights),
                    "normalization": model.adjacency_normalization}
        if resolved != expected:
            raise ValueError("Adjacency views/weights/normalization differ from the checkpoint.")
        if float(diff.get("spectral_sigma", 0.0)) != 0:
            raise ValueError("Unified adjacency has no independent spectral noise; spectral_sigma must be 0.")
        if float(config.get("attributed_predictor", {}).get("loss_weights", {}).get("spectrum", 0)) != 0:
            raise ValueError("Use adjacency_spectrum loss, not the old Laplacian spectrum loss.")

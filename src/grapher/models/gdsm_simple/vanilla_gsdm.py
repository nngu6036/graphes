"""Self-contained vanilla GSDM path for ``gdsm_simple``.

This module implements the generic-graph spectral diffusion core used in
"Fast Graph Generation via Spectral Diffusion":

* degree one-hot node features are diffused with a VP SDE;
* adjacency eigenvalues are diffused in the fixed eigenspace of each graph;
* the eigenvalue score network is conditioned on the continuous reconstructed
  adjacency ``U diag(lambda_t) U^T`` and noisy node features;
* generation samples a training-set adjacency/eigenbasis, runs the coupled
  predictor-corrector reverse process, reconstructs the continuous adjacency,
  and thresholds it once at the end.

The base ``vanilla_gsdm`` variant has no degree prior.  The controlled
``vanilla_gsdm_dhvae`` ablation co-trains an auxiliary DH-VAE on the same
training split and samples one degree sequence per generated graph, but those
sequences are *not* used by the spectral sampler or the thresholded topology.

The ``vanilla_laplacian_gsdm`` Stage-2 ablation changes only the spectral
operator: it diffuses the nonzero eigenvalue coordinates of the combinatorial
Laplacian ``L=D-A`` in a fixed training-graph Laplacian eigenbasis while the
trivial zero mode remains fixed.

The ``vanilla_laplacian_loggap`` ablation diffuses training-standardized
logarithms of successive nontrivial Laplacian eigenvalue gaps without any
structural auxiliary head.  The ``vanilla_laplacian_loggap_graphlet`` Stage-2b
variant adds a k=3,4,5 connected graphlet-summary head that shares the spectral
denoiser encoder and is trained jointly as an auxiliary objective.  The same
shared encoder can additionally predict a node clustering-coefficient histogram
and a 15-role ORCA orbit-count histogram (with log-total magnitude).  That joint
variant can optionally enrich the current noisy graph state with random-walk
arrival node features and truncated shortest-path pair features, and can use
either the released-style DenseGCN backbone or a dense PPGN pair-tensor
backbone.  Finally,
``vanilla_laplacian_loggap_graphlet_refine`` keeps that joint training objective
and uses the final auxiliary graphlet prediction as a fixed post-generation
target for degree-preserving double-edge-swap refinement.

None of these variants uses Havel-Hakimi construction, degree projection,
categorical edge prediction, graphlet guidance, rewiring, or post-hoc repair.
"""
from __future__ import annotations

import copy
from contextlib import nullcontext
import hashlib
import json
import math
import pickle
import random
import shutil
import tempfile
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from grapher.models.dhvae_hh.degree_sampler import DegreeVAESampler
from grapher.models.dhvae_hh.degree_vae import (
    DegreeHistogramVAE,
    DegreeVectorizer,
    build_degree_vae,
    degree_vae_loss,
)

from grapher.models.artifacts import ArtifactLayout
from grapher.models.base import GenerateRequest, GenerationArtifacts, TrainRequest, TrainingArtifacts
from grapher.models.errors import ArtifactCollisionError
from grapher.utils.networkx_pickle import load_trusted_networkx_pickle


CHECKPOINT_FORMAT_V1 = "gdsm_simple_vanilla_gsdm_checkpoint_v1"
CHECKPOINT_FORMAT_V2 = "gdsm_simple_vanilla_gsdm_checkpoint_v2"
CHECKPOINT_FORMAT = "gdsm_simple_vanilla_gsdm_checkpoint_v3"
SUPPORTED_CHECKPOINT_FORMATS = {CHECKPOINT_FORMAT_V1, CHECKPOINT_FORMAT_V2, CHECKPOINT_FORMAT}
TRAINING_FORMAT = "grapher_gdsm_simple_vanilla_gsdm_training_v1"
GENERATION_FORMAT = "grapher_gdsm_simple_vanilla_gsdm_generation_v1"


ADJACENCY_VARIANTS = {"vanilla_gsdm", "vanilla", "gsdm"}
DHVAE_VARIANTS = {"vanilla_gsdm_dhvae", "vanilla_gsdm_plus_dhvae"}
LAPLACIAN_VARIANTS = {
    "vanilla_laplacian_gsdm",
    "laplacian_gsdm",
    "gsdm_laplacian",
}
LAPLACIAN_LOGGAP_VARIANTS = {
    "vanilla_laplacian_loggap",
    "laplacian_loggap",
    "gsdm_laplacian_loggap",
}
LAPLACIAN_LOGGAP_GRAPHLET_VARIANTS = {
    "vanilla_laplacian_loggap_graphlet",
    "laplacian_loggap_graphlet",
    "gsdm_laplacian_loggap_graphlet",
}
LAPLACIAN_LOGGAP_GRAPHLET_REFINEMENT_VARIANTS = {
    "vanilla_laplacian_loggap_graphlet_refine",
    "laplacian_loggap_graphlet_refine",
    "gsdm_laplacian_loggap_graphlet_refine",
}
LAPLACIAN_LOGGAP_GRAPHLET_AUX_VARIANTS = (
    LAPLACIAN_LOGGAP_GRAPHLET_VARIANTS
    | LAPLACIAN_LOGGAP_GRAPHLET_REFINEMENT_VARIANTS
)
LAPLACIAN_LOGGAP_ALL_VARIANTS = (
    LAPLACIAN_LOGGAP_VARIANTS | LAPLACIAN_LOGGAP_GRAPHLET_AUX_VARIANTS
)
GRAPHLET_REFINEMENT_VARIANTS = {
    "vanilla_gsdm_graphlet_refine",
    "vanilla_gsdm_graphlet",
    "gsdm_graphlet_refine",
}


def spectral_operator_for_variant(variant: str) -> str:
    value = str(variant).lower()
    if value in LAPLACIAN_VARIANTS | LAPLACIAN_LOGGAP_ALL_VARIANTS:
        return "combinatorial_laplacian"
    if value in ADJACENCY_VARIANTS | DHVAE_VARIANTS | GRAPHLET_REFINEMENT_VARIANTS:
        return "adjacency"
    raise ValueError(f"Unknown vanilla GSDM variant: {variant!r}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(runtime: Mapping[str, Any]) -> torch.device:
    raw = str(runtime.get("device", "auto")).lower()
    if raw in {"gpu", "cuda"}:
        raw = "cuda"
    if raw == "auto":
        raw = "cuda" if torch.cuda.is_available() else "cpu"
    if raw.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("vanilla GSDM requested CUDA but torch.cuda.is_available() is false")
    return torch.device(raw)


def _graphs(path: Path) -> list[nx.Graph]:
    value = load_trusted_networkx_pickle(path)
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"Expected a non-empty graph list: {path}")
    result: list[nx.Graph] = []
    for index, graph in enumerate(value):
        if not isinstance(graph, nx.Graph):
            raise TypeError(f"Graph {index} in {path} is not networkx.Graph")
        if graph.is_directed() or graph.is_multigraph():
            raise ValueError("vanilla GSDM supports simple undirected graphs only")
        g = nx.convert_node_labels_to_integers(graph, ordering="sorted")
        if g.number_of_nodes() < 2:
            raise ValueError("vanilla GSDM requires at least two nodes")
        if any(g.degree(v) == 0 for v in g.nodes()):
            raise ValueError(
                "Vanilla GSDM infers node support from adjacency; isolated training nodes are unsupported"
            )
        result.append(g)
    return result


def default_vanilla_options() -> dict[str, Any]:
    """Defaults matching the released Community-small GSDM profile where possible."""
    return {
        "train": {
            "epochs": 200,
            "batch_size": 128,
            "lr": 1.0e-2,
            "weight_decay": 1.0e-4,
            "grad_norm": 1.0,
            "lr_schedule": True,
            "lr_decay": 0.999,
            "ema": 0.999,
            "validation_every": 1,
            "log_every": 10,
        },
        "model": {
            "max_nodes": None,
            "max_feat_num": None,
            "hidden_dim": 32,
            "depth": 3,
            # The node-feature score keeps the released DenseGCN path.  The
            # joint log-gap spectrum/graphlet score can independently choose
            # a DenseGCN or a compact PPGN pair-tensor backbone.
            "backbone": "dense_gcn",
            "ppgn_hidden_dim": 64,
            "ppgn_depth": 4,
            "ppgn_residual_scale": 0.1,
            "ppgn_norm_eps": 1.0e-5,
            "ppgn_input_clip": 10.0,
        },
        "structural_features": {
            "enabled": False,
            "binarize_threshold": 0.5,
            "random_walk": {
                "enabled": False,
                "steps": 4,
            },
            "shortest_path": {
                "enabled": False,
                "max_distance": 5,
            },
        },
        "sde": {
            "x": {"type": "vp", "beta_min": 0.1, "beta_max": 1.0, "num_scales": 1000},
            "spectrum": {"type": "vp", "beta_min": 0.1, "beta_max": 1.0, "num_scales": 1000},
            "eps": 1.0e-5,
            "eigen_mask": "official_extremes",
            "spectral_parameterization": "direct_eigenvalues",
            "log_gap_epsilon": 1.0e-6,
            "log_gap_min_std": 1.0e-3,
            "log_gap_exp_clip": 20.0,
        },
        "sample": {
            "predictor": "euler",
            "corrector": "langevin",
            "snr": 0.05,
            "scale_eps": 0.7,
            "n_steps": 1,
            "noise_removal": True,
            "probability_flow": False,
            "eps": 1.0e-4,
            "threshold": 0.5,
            "use_ema": False,
        },
        "graphlet_refinement": {"enabled": False},
        "graphlet_summary": {"enabled": False},  # legacy graphlet-only alias
        "structure_summary": {"enabled": False},
        "structure_score": {
            "enabled": False,
            "loss_weight": 0.05,
            "teacher_clip_norm": 5.0,
            "orbit_log_total_sigma": 0.25,
            "guidance": {
                "enabled": False,
                "scale": 0.05,
                "start_time": 0.50,
                "schedule": "late_linear",
                "target_source": "denoised_summary",
                "score_clip_norm": 5.0,
            },
        },
        "degree_prior": {
            "enabled": False,
            "batch_size": 64,
            "latent_dim": 32,
            "hidden_dim": 128,
            "size_condition_dim": 16,
            "edge_condition_dim": 16,
            "use_edge_count_conditioning": True,
            "prior_condition_on_edges": True,
            "prior_type": "conditional_gmm",
            "prior_components": 4,
            "prior_hidden_dim": 128,
            "prior_logvar_min": -6.0,
            "prior_logvar_max": 4.0,
            "num_layers": 2,
            "dropout": 0.0,
            "learning_rate": 1.0e-3,
            "weight_decay": 1.0e-5,
            "kl_loss_weight": 0.005,
            "kl_warmup_epochs": 50,
            "node_count_loss_weight": 1.0,
            "edge_count_loss_weight": 2.0,
            "degree_histogram_loss_weight": 5.0,
            "degree_moment_loss_weight": 0.25,
            "prior_distribution_loss_weight": 1.0,
            "prior_distribution_kernel_sigma": 0.2,
            "aggregate_prior_moment_loss_weight": 0.05,
            "require_connected": True,
            "sample_num_nodes": "empirical",
            "sample_num_edges": "model",
            "exact_degree_sum_conditioning": True,
            "max_resample": 500,
            "model_resample_attempts": 32,
            "parity_conditioned": False,
            "max_parity_resample": 1,
            "postprocess_policy": "reject_only",
            "fallback": "error",
        },
        "generation_batch_size": 128,
        "runtime": {"device": "auto"},
    }


def _resolved_structure_summary_config(options: Mapping[str, Any]) -> dict[str, Any]:
    """Return a canonical joint structural-summary configuration.

    ``graphlet_summary`` is kept as a backwards-compatible graphlet-only alias.
    New experiments should use ``structure_summary`` so graphlet, clustering,
    and orbit targets can share the same spectrum/PPGN encoder.
    """
    raw_structure = dict(options.get("structure_summary", {}) or {})
    raw_graphlet = dict(options.get("graphlet_summary", {}) or {})
    if bool(raw_structure.get("enabled", False)):
        if bool(raw_graphlet.get("enabled", False)):
            raise ValueError(
                "Use either structure_summary or legacy graphlet_summary, not both."
            )
        graphlet = dict(raw_structure.get("graphlet", {}) or {})
        clustering = dict(raw_structure.get("clustering", {}) or {})
        orbit = dict(raw_structure.get("orbit", {}) or {})
        return {
            "enabled": True,
            "loss_weight": float(raw_structure.get("loss_weight", 0.10)),
            "graphlet": {
                "enabled": bool(graphlet.get("enabled", True)),
                "orders": list(graphlet.get("orders", [3, 4, 5])),
                "histogram_weight": float(graphlet.get("histogram_weight", 1.0)),
                "mass_weight": float(graphlet.get("mass_weight", 0.25)),
            },
            "clustering": {
                "enabled": bool(clustering.get("enabled", False)),
                "bins": int(clustering.get("bins", 100)),
                "histogram_weight": float(clustering.get("histogram_weight", 0.25)),
                "cdf_weight": float(clustering.get("cdf_weight", 1.0)),
            },
            "orbit": {
                "enabled": bool(orbit.get("enabled", False)),
                "width": int(orbit.get("width", 15)),
                "histogram_weight": float(orbit.get("histogram_weight", 0.25)),
                "log_total_weight": float(orbit.get("log_total_weight", 0.10)),
            },
        }
    if bool(raw_graphlet.get("enabled", False)):
        return {
            "enabled": True,
            "loss_weight": float(raw_graphlet.get("loss_weight", 0.10)),
            "graphlet": {
                "enabled": True,
                "orders": list(raw_graphlet.get("orders", [3, 4, 5])),
                "histogram_weight": float(raw_graphlet.get("histogram_weight", 1.0)),
                "mass_weight": float(raw_graphlet.get("mass_weight", 0.25)),
            },
            "clustering": {
                "enabled": False,
                "bins": 100,
                "histogram_weight": 0.0,
                "cdf_weight": 0.0,
            },
            "orbit": {
                "enabled": False,
                "width": 15,
                "histogram_weight": 0.0,
                "log_total_weight": 0.0,
            },
        }
    return {
        "enabled": False,
        "loss_weight": 0.0,
        "graphlet": {"enabled": False, "orders": [3, 4, 5], "histogram_weight": 0.0, "mass_weight": 0.0},
        "clustering": {"enabled": False, "bins": 100, "histogram_weight": 0.0, "cdf_weight": 0.0},
        "orbit": {"enabled": False, "width": 15, "histogram_weight": 0.0, "log_total_weight": 0.0},
    }


def _resolved_structure_score_config(options: Mapping[str, Any]) -> dict[str, Any]:
    """Canonical configuration for the learned structural log-score head.

    The head approximates the conditional structural guidance score

        grad_{z_t} log p(S_0 | X_t, z_t),

    in standardized Laplacian log-gap coordinates.  It is supervised by an
    autodifferentiated teacher score from the primary structural-summary
    likelihood and can be injected into the reverse spectral score at sampling.
    """
    raw = dict(options.get("structure_score", {}) or {})
    guidance = dict(raw.get("guidance", {}) or {})
    return {
        "enabled": bool(raw.get("enabled", False)),
        "loss_weight": float(raw.get("loss_weight", 0.05)),
        "teacher_clip_norm": float(raw.get("teacher_clip_norm", 5.0)),
        "orbit_log_total_sigma": float(raw.get("orbit_log_total_sigma", 0.25)),
        "guidance": {
            "enabled": bool(guidance.get("enabled", False)),
            "scale": float(guidance.get("scale", 0.05)),
            "start_time": float(guidance.get("start_time", 0.50)),
            "schedule": str(guidance.get("schedule", "late_linear")).lower(),
            "target_source": str(guidance.get("target_source", "denoised_summary")).lower(),
            "score_clip_norm": float(guidance.get("score_clip_norm", 5.0)),
        },
    }


def _structure_summary_vector_dim(
    graphlet_slices: tuple[tuple[int, int], ...],
    *,
    clustering_bins: int,
    orbit_width: int,
) -> int:
    width = int(graphlet_slices[-1][1]) + len(graphlet_slices)
    if int(clustering_bins) > 0:
        width += int(clustering_bins)
    if int(orbit_width) > 0:
        width += int(orbit_width) + 1
    return width


def _pack_structure_summary_vector(
    graphlet_histogram: torch.Tensor,
    graphlet_mass: torch.Tensor,
    clustering_histogram: torch.Tensor | None,
    orbit_histogram: torch.Tensor | None,
    orbit_log_total: torch.Tensor | None,
) -> torch.Tensor:
    parts = [graphlet_histogram, graphlet_mass]
    if clustering_histogram is not None:
        parts.append(clustering_histogram)
    if orbit_histogram is not None:
        parts.append(orbit_histogram)
    if orbit_log_total is not None:
        parts.append(orbit_log_total)
    return torch.cat(parts, dim=-1)


def _clip_vector_norm(value: torch.Tensor, max_norm: float) -> torch.Tensor:
    if max_norm <= 0.0:
        return value
    flat = value.reshape(value.size(0), -1)
    norm = flat.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
    scale = torch.clamp(float(max_norm) / norm, max=1.0)
    return value * scale.view(value.size(0), *([1] * (value.ndim - 1)))


def _structure_guidance_scale(t: torch.Tensor, cfg: Mapping[str, Any]) -> torch.Tensor:
    guidance = dict(cfg.get("guidance", {}) or {})
    if not bool(cfg.get("enabled", False)) or not bool(guidance.get("enabled", False)):
        return torch.zeros_like(t)
    scale = float(guidance.get("scale", 0.05))
    start = float(guidance.get("start_time", 0.50))
    schedule = str(guidance.get("schedule", "late_linear")).lower()
    if start <= 0.0:
        return torch.zeros_like(t)
    active = (t <= start).to(t.dtype)
    if schedule == "constant":
        return active * scale
    if schedule == "late_linear":
        ramp = (1.0 - t / start).clamp(0.0, 1.0)
        return ramp * scale
    raise ValueError(f"Unknown structure-score guidance schedule: {schedule!r}")


def validate_options(options: Mapping[str, Any]) -> None:
    allowed = {
        "variant", "train", "model", "sde", "sample", "degree_prior", "graphlet_refinement", "graphlet_summary", "structure_summary", "structure_score", "structural_features", "generation_batch_size",
        "runtime", "extensions", "comparison_reference", "training_estimates", "diffusion",
    }
    unknown = sorted(set(options) - allowed)
    if unknown:
        raise ValueError(f"Unknown vanilla GSDM options: {unknown}")
    variant = str(options.get("variant", "")).lower()
    allowed_variants = (
        ADJACENCY_VARIANTS
        | DHVAE_VARIANTS
        | LAPLACIAN_VARIANTS
        | LAPLACIAN_LOGGAP_ALL_VARIANTS
        | GRAPHLET_REFINEMENT_VARIANTS
    )
    if variant not in allowed_variants:
        raise ValueError(
            "vanilla GSDM pipeline requires vanilla_gsdm, vanilla_gsdm_dhvae, "
            "vanilla_laplacian_gsdm, or vanilla_gsdm_graphlet_refine"
        )
    degree_prior = options.get("degree_prior", {}) or {}
    prior_enabled = bool(degree_prior.get("enabled", False))
    if variant in DHVAE_VARIANTS and not prior_enabled:
        raise ValueError("vanilla_gsdm_dhvae requires degree_prior.enabled=true")
    if variant in ADJACENCY_VARIANTS and prior_enabled:
        raise ValueError("Use variant: vanilla_gsdm_dhvae when degree_prior.enabled=true")
    if variant in (LAPLACIAN_VARIANTS | LAPLACIAN_LOGGAP_ALL_VARIANTS) and prior_enabled:
        raise ValueError(
            "Stage-2 vanilla_laplacian_gsdm is a clean spectral-operator ablation and "
            "does not enable the auxiliary DH-VAE"
        )
    summary_cfg = _resolved_structure_summary_config(options)
    summary_enabled = bool(summary_cfg.get("enabled", False))
    if variant in LAPLACIAN_LOGGAP_GRAPHLET_AUX_VARIANTS:
        if not summary_enabled:
            raise ValueError(
                "Laplacian log-gap graphlet variants require structure_summary.enabled=true "
                "or legacy graphlet_summary.enabled=true"
            )
        graphlet_cfg = dict(summary_cfg.get("graphlet", {}) or {})
        if not bool(graphlet_cfg.get("enabled", False)):
            raise ValueError("The selected joint structural model requires graphlet.enabled=true")
        if list(graphlet_cfg.get("orders", [3, 4, 5])) != [3, 4, 5]:
            raise ValueError("log-gap structural ablation is fixed to graphlet orders 3,4,5")
        if float(summary_cfg.get("loss_weight", 0.0)) < 0.0:
            raise ValueError("structure-summary loss_weight must be nonnegative")
        clustering_cfg = dict(summary_cfg.get("clustering", {}) or {})
        if bool(clustering_cfg.get("enabled", False)):
            bins = int(clustering_cfg.get("bins", 100))
            if bins < 2:
                raise ValueError("structure_summary.clustering.bins must be >= 2")
        orbit_cfg = dict(summary_cfg.get("orbit", {}) or {})
        if bool(orbit_cfg.get("enabled", False)) and int(orbit_cfg.get("width", 15)) != 15:
            raise ValueError(
                "structure_summary.orbit.width must be 15 (standard ORCA node orbits 0..14)"
            )
        for section, keys in (
            (graphlet_cfg, ("histogram_weight", "mass_weight")),
            (clustering_cfg, ("histogram_weight", "cdf_weight")),
            (orbit_cfg, ("histogram_weight", "log_total_weight")),
        ):
            for key in keys:
                if float(section.get(key, 0.0)) < 0.0:
                    raise ValueError(f"structure-summary {key} must be nonnegative")
    elif summary_enabled:
        raise ValueError(
            "structure_summary/graphlet_summary can only be enabled for a Laplacian "
            "log-gap joint structural variant"
        )

    score_cfg = _resolved_structure_score_config(options)
    if bool(score_cfg.get("enabled", False)):
        if variant not in LAPLACIAN_LOGGAP_GRAPHLET_AUX_VARIANTS:
            raise ValueError("structure_score requires a joint Laplacian log-gap structural variant")
        if not summary_enabled:
            raise ValueError("structure_score requires structure_summary.enabled=true")
        if float(score_cfg.get("loss_weight", 0.0)) < 0.0:
            raise ValueError("structure_score.loss_weight must be nonnegative")
        if float(score_cfg.get("teacher_clip_norm", 0.0)) <= 0.0:
            raise ValueError("structure_score.teacher_clip_norm must be positive")
        if float(score_cfg.get("orbit_log_total_sigma", 0.0)) <= 0.0:
            raise ValueError("structure_score.orbit_log_total_sigma must be positive")
        guidance = dict(score_cfg.get("guidance", {}) or {})
        if float(guidance.get("scale", 0.0)) < 0.0:
            raise ValueError("structure_score.guidance.scale must be nonnegative")
        start_time = float(guidance.get("start_time", 0.5))
        if not (0.0 < start_time <= 1.0):
            raise ValueError("structure_score.guidance.start_time must be in (0,1]")
        if str(guidance.get("schedule", "late_linear")) not in {"late_linear", "constant"}:
            raise ValueError("structure_score.guidance.schedule must be late_linear or constant")
        if str(guidance.get("target_source", "denoised_summary")) != "denoised_summary":
            raise ValueError("Only structure_score.guidance.target_source=denoised_summary is supported")
        if float(guidance.get("score_clip_norm", 0.0)) <= 0.0:
            raise ValueError("structure_score.guidance.score_clip_norm must be positive")

    model_cfg = options.get("model", {}) or {}
    legacy_backbone = str(model_cfg.get("backbone", "dense_gcn")).lower()
    node_backbone = str(model_cfg.get("node_backbone", legacy_backbone)).lower()
    spectrum_backbone = str(model_cfg.get("spectrum_backbone", legacy_backbone)).lower()
    for key, backbone in (("node_backbone", node_backbone), ("spectrum_backbone", spectrum_backbone)):
        if backbone not in {"dense_gcn", "ppgn"}:
            raise ValueError(f"model.{key} must be dense_gcn or ppgn")
    if bool(score_cfg.get("enabled", False)) and spectrum_backbone != "ppgn":
        raise ValueError("structure_score currently requires model.spectrum_backbone=ppgn")
    if (node_backbone == "ppgn" or spectrum_backbone == "ppgn") and variant not in LAPLACIAN_LOGGAP_GRAPHLET_AUX_VARIANTS:
        raise ValueError(
            "PPGN backbones are currently implemented for the joint "
            "Laplacian log-gap graphlet variants only"
        )
    if int(model_cfg.get("ppgn_hidden_dim", 64)) <= 0:
        raise ValueError("model.ppgn_hidden_dim must be positive")
    if int(model_cfg.get("ppgn_depth", 4)) <= 0:
        raise ValueError("model.ppgn_depth must be positive")
    residual_scale = float(model_cfg.get("ppgn_residual_scale", 0.1))
    if not math.isfinite(residual_scale) or residual_scale < 0.0 or residual_scale > 1.0:
        raise ValueError("model.ppgn_residual_scale must be finite and in [0,1]")
    norm_eps = float(model_cfg.get("ppgn_norm_eps", 1.0e-5))
    if not math.isfinite(norm_eps) or norm_eps <= 0.0:
        raise ValueError("model.ppgn_norm_eps must be positive and finite")
    input_clip = float(model_cfg.get("ppgn_input_clip", 10.0))
    if not math.isfinite(input_clip) or input_clip <= 0.0:
        raise ValueError("model.ppgn_input_clip must be positive and finite")

    structural_cfg = options.get("structural_features", {}) or {}
    structural_enabled = bool(structural_cfg.get("enabled", False))
    if structural_enabled and variant not in LAPLACIAN_LOGGAP_GRAPHLET_AUX_VARIANTS:
        raise ValueError(
            "structural_features.enabled=true is currently implemented for the joint "
            "Laplacian log-gap graphlet variants only"
        )
    if structural_enabled:
        threshold = float(structural_cfg.get("binarize_threshold", 0.5))
        if not math.isfinite(threshold):
            raise ValueError("structural_features.binarize_threshold must be finite")
        rw_cfg = structural_cfg.get("random_walk", {}) or {}
        if bool(rw_cfg.get("enabled", False)) and int(rw_cfg.get("steps", 0)) <= 0:
            raise ValueError("structural_features.random_walk.steps must be positive")
        sp_cfg = structural_cfg.get("shortest_path", {}) or {}
        if bool(sp_cfg.get("enabled", False)) and int(sp_cfg.get("max_distance", 0)) <= 0:
            raise ValueError("structural_features.shortest_path.max_distance must be positive")

    graphlet_cfg = options.get("graphlet_refinement", {}) or {}
    graphlet_enabled = bool(graphlet_cfg.get("enabled", False))
    if variant in GRAPHLET_REFINEMENT_VARIANTS:
        if prior_enabled:
            raise ValueError("Stage-3 graphlet refinement does not use the DH-VAE degree prior")
        if not graphlet_enabled:
            raise ValueError("vanilla_gsdm_graphlet_refine requires graphlet_refinement.enabled=true")
        from grapher.models.gdsm_simple.graphlet_stage3 import validate_graphlet_refinement_options
        validate_graphlet_refinement_options(graphlet_cfg)
    elif variant in LAPLACIAN_LOGGAP_GRAPHLET_REFINEMENT_VARIANTS:
        if not graphlet_enabled:
            raise ValueError(
                "vanilla_laplacian_loggap_graphlet_refine requires graphlet_refinement.enabled=true"
            )
        from grapher.models.gdsm_simple.graphlet_stage3 import (
            validate_fixed_target_graphlet_refinement_options,
        )
        validate_fixed_target_graphlet_refinement_options(graphlet_cfg)
    elif graphlet_enabled:
        raise ValueError("graphlet_refinement.enabled=true requires a graphlet-refinement variant")
    if prior_enabled:
        if int(degree_prior.get("batch_size", 0)) <= 0:
            raise ValueError("degree_prior.batch_size must be positive")
        if str(degree_prior.get("sample_num_nodes", "empirical")).lower() not in {"empirical", "model"}:
            raise ValueError("degree_prior.sample_num_nodes must be empirical or model")
    extensions = options.get("extensions", {}) or {}
    enabled = []
    for key, value in extensions.items():
        if key == "structural_summary":
            if str(value).lower() != "none":
                enabled.append(key)
        elif isinstance(value, Mapping):
            # Nested option blocks are harmless only when their parent extension is disabled.
            continue
        elif bool(value):
            enabled.append(key)
    if enabled:
        raise ValueError(
            "The vanilla_gsdm baseline intentionally disables GraphES extensions; "
            f"enabled: {sorted(enabled)}"
        )
    sde = options.get("sde", {})
    for key in ("x", "spectrum"):
        cfg = sde.get(key, {})
        if str(cfg.get("type", "vp")).lower() != "vp":
            raise ValueError("The first vanilla implementation supports the released VP-SDE path only")
        if int(cfg.get("num_scales", 0)) < 2:
            raise ValueError("sde num_scales must be >= 2")
        b0, b1 = float(cfg.get("beta_min", 0.0)), float(cfg.get("beta_max", 0.0))
        if not (0.0 < b0 <= b1):
            raise ValueError("expected 0 < beta_min <= beta_max")
    eigen_mask = str(sde.get("eigen_mask", "official_extremes")).lower()
    operator = spectral_operator_for_variant(variant)
    if operator == "combinatorial_laplacian":
        if eigen_mask not in {"laplacian_nonzero_prefix"}:
            raise ValueError(
                "Laplacian GSDM variants require sde.eigen_mask=laplacian_nonzero_prefix"
            )
        parameterization = str(sde.get("spectral_parameterization", "direct_eigenvalues")).lower()
        if variant in LAPLACIAN_LOGGAP_ALL_VARIANTS:
            if parameterization != "laplacian_log_gap":
                raise ValueError(
                    "Laplacian log-gap variants require sde.spectral_parameterization=laplacian_log_gap"
                )
            if float(sde.get("log_gap_epsilon", 0.0)) <= 0.0:
                raise ValueError("sde.log_gap_epsilon must be positive")
            if float(sde.get("log_gap_min_std", 0.0)) <= 0.0:
                raise ValueError("sde.log_gap_min_std must be positive")
        elif parameterization not in {"direct_eigenvalues", "direct", "eigenvalues"}:
            raise ValueError("vanilla_laplacian_gsdm uses direct eigenvalue diffusion")
    elif eigen_mask not in {"official_extremes", "all_padded_eigenvalues"}:
        raise ValueError(
            "adjacency vanilla GSDM requires sde.eigen_mask=official_extremes "
            "(or all_padded_eigenvalues for debugging)"
        )
    if int(options.get("generation_batch_size", 0)) <= 0:
        raise ValueError("generation_batch_size must be positive")
    sample = options.get("sample", {})
    if str(sample.get("predictor", "euler")).lower() != "euler":
        raise ValueError("vanilla GSDM currently supports predictor=euler only")
    if str(sample.get("corrector", "langevin")).lower() not in {"langevin", "none"}:
        raise ValueError("vanilla GSDM supports corrector=langevin or none")
    if bool(sample.get("probability_flow", False)):
        raise ValueError("probability_flow=true is not implemented in this vanilla path")


class DenseGCNConv(nn.Module):
    """Dense GCN layer matching the released GSDM normalization convention."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, adj: torch.Tensor, flags: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 3 or adj.ndim != 3:
            raise ValueError("DenseGCNConv expects x=[B,N,F], adj=[B,N,N]")
        a = adj.clone()
        n = a.size(1)
        idx = torch.arange(n, device=a.device)
        a[:, idx, idx] = 1.0
        out = x @ self.weight
        deg_inv_sqrt = a.sum(dim=-1).clamp(min=1.0).pow(-0.5)
        a = deg_inv_sqrt.unsqueeze(-1) * a * deg_inv_sqrt.unsqueeze(-2)
        out = a @ out + self.bias
        if flags is not None:
            out = out * flags.unsqueeze(-1).to(out.dtype)
        return out


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, layers: int) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("MLP layers must be >= 1")
        if layers == 1:
            self.layers = nn.ModuleList([nn.Linear(input_dim, output_dim)])
        else:
            values = [nn.Linear(input_dim, hidden_dim)]
            values.extend(nn.Linear(hidden_dim, hidden_dim) for _ in range(layers - 2))
            values.append(nn.Linear(hidden_dim, output_dim))
            self.layers = nn.ModuleList(values)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = F.elu(layer(x))
        return self.layers[-1](x)


def _binarize_structural_graph(
    adj: torch.Tensor,
    flags: torch.Tensor,
    *,
    threshold: float,
) -> torch.Tensor:
    """Binarize a continuous graph state for structural feature extraction.

    This branch is deliberately detached from the diffusion state.  The
    random-walk / shortest-path descriptors are auxiliary conditioning
    features, not a differentiable replacement for the carried continuous
    Laplacian-derived adjacency state.
    """
    if adj.ndim != 3 or flags.ndim != 2:
        raise ValueError("structural graph binarization expects adj=[B,N,N], flags=[B,N]")
    binary = (adj.detach() > float(threshold))
    binary = binary | binary.transpose(-1, -2)
    n = binary.size(-1)
    idx = torch.arange(n, device=binary.device)
    binary[:, idx, idx] = False
    pair_mask = flags.bool().unsqueeze(-1) & flags.bool().unsqueeze(-2)
    return (binary & pair_mask).to(adj.dtype)


def random_walk_landing_features(
    binary_adj: torch.Tensor,
    flags: torch.Tensor,
    *,
    steps: int,
) -> torch.Tensor:
    """Return per-node random-walk landing/return probabilities.

    For the row-stochastic transition matrix ``P``, node ``i`` receives
    ``[(P^1)_ii, ..., (P^steps)_ii]``.  These diagonal landing probabilities
    are permutation-equivariant node descriptors and match the ``rw_landing``
    feature naming used by the released HOG-Diff score network.
    """
    if steps <= 0:
        return binary_adj.new_zeros((*flags.shape, 0))
    _b, n, _ = binary_adj.shape
    active = flags > 0
    degree = binary_adj.sum(dim=-1)
    transition = binary_adj / degree.clamp_min(1.0).unsqueeze(-1)
    # Keep an isolated active node at itself so each active row remains a valid
    # stochastic row.  This matters for very noisy intermediate states.
    isolated = active & (degree <= 0)
    if bool(isolated.any()):
        idx = torch.arange(n, device=binary_adj.device)
        transition[:, idx, idx] = torch.where(
            isolated,
            torch.ones_like(degree),
            transition[:, idx, idx],
        )
    values: list[torch.Tensor] = []
    power = transition
    for _ in range(int(steps)):
        landing = torch.diagonal(power, dim1=-2, dim2=-1)
        values.append(landing * flags.to(landing.dtype))
        power = torch.bmm(power, transition)
    return torch.stack(values, dim=-1)


def random_walk_arrival_features(
    binary_adj: torch.Tensor,
    flags: torch.Tensor,
    *,
    steps: int,
) -> torch.Tensor:
    """Backward-compatible alias for the HOG-Diff-style landing features."""
    return random_walk_landing_features(binary_adj, flags, steps=steps)


def truncated_shortest_path_features(
    binary_adj: torch.Tensor,
    flags: torch.Tensor,
    *,
    max_distance: int,
) -> torch.Tensor:
    """One-hot truncated shortest-path pair features.

    Buckets are ``0,1,...,max_distance`` plus a final bucket for distances
    larger than ``max_distance`` or disconnected active node pairs.  Padding
    pairs are masked to all-zero vectors.
    """
    if max_distance <= 0:
        raise ValueError("max_distance must be positive")
    b, n, _ = binary_adj.shape
    dtype = binary_adj.dtype
    device = binary_adj.device
    active = flags.bool()
    pair_mask = active.unsqueeze(-1) & active.unsqueeze(-2)
    overflow = int(max_distance) + 1
    bucket = torch.full((b, n, n), overflow, dtype=torch.long, device=device)
    idx = torch.arange(n, device=device)
    bucket[:, idx, idx] = torch.where(
        active,
        torch.zeros_like(flags, dtype=torch.long),
        torch.full_like(flags, overflow, dtype=torch.long),
    )
    assigned = torch.eye(n, dtype=torch.bool, device=device).unsqueeze(0) & pair_mask
    # We only need distances up to max_distance, so repeated adjacency powers
    # are cheaper than a full Floyd-Warshall pass and map well to batched GPU
    # matrix multiplication.  A^d_ij>0 iff a length-d walk exists.
    power = torch.eye(n, dtype=dtype, device=device).unsqueeze(0).repeat(b, 1, 1)
    for distance in range(1, int(max_distance) + 1):
        power = torch.bmm(power, binary_adj)
        reachable = (power > 0) & pair_mask
        newly = reachable & ~assigned
        bucket = torch.where(newly, torch.full_like(bucket, distance), bucket)
        assigned = assigned | reachable
    features = F.one_hot(bucket, num_classes=overflow + 1).to(dtype)
    return features * pair_mask.unsqueeze(-1).to(dtype)


def _structural_features_from_state(
    adj: torch.Tensor,
    flags: torch.Tensor,
    cfg: Mapping[str, Any] | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(binary_adj, random_walk_node, shortest_path_edge)``."""
    cfg = dict(cfg or {})
    threshold = float(cfg.get("binarize_threshold", 0.5))
    binary = _binarize_structural_graph(adj, flags, threshold=threshold)
    rw_cfg = dict(cfg.get("random_walk", {}) or {})
    sp_cfg = dict(cfg.get("shortest_path", {}) or {})
    if bool(cfg.get("enabled", False)) and bool(rw_cfg.get("enabled", False)):
        rw = random_walk_landing_features(
            binary,
            flags,
            steps=int(rw_cfg.get("steps", 4)),
        )
    else:
        rw = adj.new_zeros((*flags.shape, 0))
    if bool(cfg.get("enabled", False)) and bool(sp_cfg.get("enabled", False)):
        sp = truncated_shortest_path_features(
            binary,
            flags,
            max_distance=int(sp_cfg.get("max_distance", 5)),
        )
    else:
        sp = adj.new_zeros((*adj.shape, 0))
    return binary, rw, sp


class DenseStructuralGCNConv(DenseGCNConv):
    """Dense GCN with an additive shortest-path relation context.

    The normal GCN message remains unchanged.  When pair features are present,
    a learned relation embedding is mean-aggregated over active destination
    nodes and added to the local GCN update.  This lets the DenseGCN ablation
    consume true pair/edge features without turning it into a PPGN.
    """

    def __init__(self, in_channels: int, out_channels: int, edge_dim: int = 0) -> None:
        super().__init__(in_channels, out_channels)
        self.edge_dim = int(edge_dim)
        self.edge_message = (
            nn.Linear(self.edge_dim, out_channels, bias=False)
            if self.edge_dim > 0 else None
        )

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        flags: torch.Tensor | None = None,
        edge_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = super().forward(x, adj, flags)
        if self.edge_message is None or edge_features is None or edge_features.size(-1) == 0:
            return out
        if flags is None:
            pair_mask = torch.ones_like(adj)
            denom = torch.full(
                (adj.size(0), adj.size(1), 1),
                float(adj.size(1)),
                dtype=adj.dtype,
                device=adj.device,
            )
        else:
            pair_mask = flags.to(adj.dtype).unsqueeze(-1) * flags.to(adj.dtype).unsqueeze(-2)
            denom = pair_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        relation = self.edge_message(edge_features)
        relation = (relation * pair_mask.unsqueeze(-1)).sum(dim=2) / denom
        out = out + relation
        if flags is not None:
            out = out * flags.unsqueeze(-1).to(out.dtype)
        return out


class MaskedPairInstanceNorm(nn.Module):
    """Channel-wise instance normalization over active pair positions only."""

    def __init__(self, hidden_dim: int, eps: float = 1.0e-5) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.bias = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, h: torch.Tensor, pair_mask: torch.Tensor) -> torch.Tensor:
        mask = pair_mask.unsqueeze(-1).to(h.dtype)
        count = mask.sum(dim=(1, 2), keepdim=True).clamp_min(1.0)
        mean = (h * mask).sum(dim=(1, 2), keepdim=True) / count
        centered = (h - mean) * mask
        var = centered.square().sum(dim=(1, 2), keepdim=True) / count
        out = centered / torch.sqrt(var + self.eps)
        out = out * self.weight.view(1, 1, 1, -1) + self.bias.view(1, 1, 1, -1)
        return out * mask


class PPGNLayer(nn.Module):
    """Numerically stable masked PPGN block.

    Linear-layer biases are explicitly re-masked before the matrix product so
    padded pairs cannot leak into active nodes.  A masked instance normalization
    follows the residual update, mirroring the normalization used by SPECTRE's
    PPGN implementation and preventing repeated pair-matrix products from
    growing without bound.
    """

    def __init__(
        self,
        hidden_dim: int,
        *,
        residual_scale: float = 0.1,
        norm_eps: float = 1.0e-5,
    ) -> None:
        super().__init__()
        self.left = nn.Linear(hidden_dim, hidden_dim)
        self.right = nn.Linear(hidden_dim, hidden_dim)
        self.skip = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(2 * hidden_dim, hidden_dim)
        self.norm = MaskedPairInstanceNorm(hidden_dim, eps=norm_eps)
        self.residual_scale = float(residual_scale)

    def forward(
        self,
        h: torch.Tensor,
        pair_mask: torch.Tensor,
        flags: torch.Tensor,
    ) -> torch.Tensor:
        mask = pair_mask.unsqueeze(-1).to(h.dtype)
        h = h * mask
        left = F.elu(self.left(h)) * mask
        right = F.elu(self.right(h)) * mask
        product = torch.einsum("bikc,bkjc->bijc", left, right)
        scale = flags.sum(dim=-1).clamp_min(1.0).sqrt().view(-1, 1, 1, 1)
        product = (product / scale) * mask
        skip = F.elu(self.skip(h)) * mask
        updated = F.elu(self.out(torch.cat((skip, product), dim=-1))) * mask
        updated = updated + self.residual_scale * h
        updated = self.norm(updated, pair_mask)
        if not torch.isfinite(updated).all():
            raise FloatingPointError("Non-finite activation in PPGNLayer")
        return updated


class GSDMSpectrumGraphletPPGNScore(nn.Module):
    """PPGN alternative for the joint log-gap spectrum/graphlet score model."""

    def __init__(
        self,
        *,
        max_feat_num: int,
        max_nodes: int,
        hidden_dim: int,
        depth: int,
        graphlet_slices: tuple[tuple[int, int], ...],
        structural_features: Mapping[str, Any] | None = None,
        clustering_bins: int = 0,
        orbit_width: int = 0,
        structure_score_config: Mapping[str, Any] | None = None,
        ppgn_hidden_dim: int = 64,
        ppgn_depth: int = 4,
        ppgn_residual_scale: float = 0.1,
        ppgn_norm_eps: float = 1.0e-5,
        ppgn_input_clip: float = 10.0,
    ) -> None:
        super().__init__()
        del hidden_dim, depth  # DenseGCN-only width/depth; retained in model config for parity.
        self.max_feat_num = int(max_feat_num)
        self.max_nodes = int(max_nodes)
        self.graphlet_slices = tuple((int(a), int(b)) for a, b in graphlet_slices)
        if not self.graphlet_slices:
            raise ValueError("graphlet_slices must be non-empty")
        self.structural_features = copy.deepcopy(dict(structural_features or {}))
        rw_cfg = dict(self.structural_features.get("random_walk", {}) or {})
        sp_cfg = dict(self.structural_features.get("shortest_path", {}) or {})
        active = bool(self.structural_features.get("enabled", False))
        self.rw_steps = int(rw_cfg.get("steps", 4)) if active and bool(rw_cfg.get("enabled", False)) else 0
        self.sp_dim = (
            int(sp_cfg.get("max_distance", 5)) + 2
            if active and bool(sp_cfg.get("enabled", False)) else 0
        )
        node_dim = self.max_feat_num + self.rw_steps
        # Continuous adjacency score, binarized adjacency, optional shortest
        # path one-hot relation, and node features placed on the diagonal.
        pair_input_dim = 2 + self.sp_dim + node_dim
        hdim = int(ppgn_hidden_dim)
        self.pair_input = nn.Linear(pair_input_dim, hdim)
        self.pair_input_norm = nn.LayerNorm(hdim)
        self.ppgn_input_clip = float(ppgn_input_clip)
        self.ppgn_layers = nn.ModuleList(
            PPGNLayer(
                hdim,
                residual_scale=float(ppgn_residual_scale),
                norm_eps=float(ppgn_norm_eps),
            )
            for _ in range(int(ppgn_depth))
        )
        shared_dim = 2 * hdim + self.max_nodes
        self.spectrum_final = MLP(shared_dim, 2 * max(self.max_nodes, hdim), self.max_nodes, 2)
        width = self.graphlet_slices[-1][1]
        self.graphlet_logits = MLP(shared_dim, 2 * shared_dim, width, 2)
        self.graphlet_mass_logits = MLP(shared_dim, 2 * shared_dim, len(self.graphlet_slices), 2)
        self.clustering_bins = int(clustering_bins)
        self.orbit_width = int(orbit_width)
        self.clustering_logits = (
            MLP(shared_dim, 2 * shared_dim, self.clustering_bins, 2)
            if self.clustering_bins > 0 else None
        )
        self.orbit_histogram_logits = (
            MLP(shared_dim, 2 * shared_dim, self.orbit_width, 2)
            if self.orbit_width > 0 else None
        )
        self.orbit_log_total_raw = (
            MLP(shared_dim, 2 * shared_dim, 1, 2)
            if self.orbit_width > 0 else None
        )
        self.structure_score_config = copy.deepcopy(dict(structure_score_config or {}))
        self.structure_score_enabled = bool(self.structure_score_config.get("enabled", False))
        self.structure_target_dim = _structure_summary_vector_dim(
            self.graphlet_slices,
            clustering_bins=self.clustering_bins,
            orbit_width=self.orbit_width,
        )
        self.structure_score_head = (
            MLP(
                shared_dim + self.structure_target_dim + 1,
                2 * shared_dim,
                self.max_nodes,
                2,
            )
            if self.structure_score_enabled else None
        )

    def _encode(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        flags: torch.Tensor,
        eigenvalues: torch.Tensor,
    ) -> torch.Tensor:
        binary, rw, sp = _structural_features_from_state(adj, flags, self.structural_features)
        node = torch.cat((x, rw), dim=-1) if rw.size(-1) else x
        b, n, _ = x.shape
        diag_node = x.new_zeros((b, n, n, node.size(-1)))
        idx = torch.arange(n, device=x.device)
        diag_node[:, idx, idx, :] = node
        bounded_adj = torch.tanh(torch.nan_to_num(
            adj, nan=0.0, posinf=self.ppgn_input_clip, neginf=-self.ppgn_input_clip
        ).clamp(-self.ppgn_input_clip, self.ppgn_input_clip))
        parts = [bounded_adj.unsqueeze(-1), binary.unsqueeze(-1)]
        if sp.size(-1):
            parts.append(sp)
        parts.append(diag_node)
        pair = torch.cat(parts, dim=-1)
        pair_mask = flags.to(pair.dtype).unsqueeze(-1) * flags.to(pair.dtype).unsqueeze(-2)
        h = self.pair_input_norm(self.pair_input(pair))
        h = F.elu(h) * pair_mask.unsqueeze(-1)
        for layer in self.ppgn_layers:
            h = layer(h, pair_mask, flags)
        diag = h[:, idx, idx, :]
        count = flags.sum(dim=-1, keepdim=True).clamp_min(1.0).to(h.dtype)
        diag_pool = (diag * flags.unsqueeze(-1).to(h.dtype)).sum(dim=1) / count
        diag_mask = torch.eye(n, dtype=h.dtype, device=h.device).unsqueeze(0)
        off_mask = pair_mask * (1.0 - diag_mask)
        off_count = off_mask.sum(dim=(1, 2), keepdim=False).unsqueeze(-1).clamp_min(1.0)
        off_pool = (h * off_mask.unsqueeze(-1)).sum(dim=(1, 2)) / off_count
        return torch.cat((diag_pool, off_pool, eigenvalues), dim=-1)

    def _heads_from_shared(self, shared: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = {
            "spectrum": self.spectrum_final(shared),
            "graphlet_logits": self.graphlet_logits(shared),
            "graphlet_mass_logits": self.graphlet_mass_logits(shared),
        }
        if self.clustering_logits is not None:
            outputs["clustering_logits"] = self.clustering_logits(shared)
        if self.orbit_histogram_logits is not None and self.orbit_log_total_raw is not None:
            outputs["orbit_histogram_logits"] = self.orbit_histogram_logits(shared)
            outputs["orbit_log_total_raw"] = self.orbit_log_total_raw(shared)
        return outputs

    def structure_vector_from_outputs(
        self, outputs: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        means = self.structure_means_from_outputs(outputs)
        return _pack_structure_summary_vector(
            means["graphlet_histogram"],
            means["graphlet_mass"],
            means.get("clustering_histogram"),
            means.get("orbit_histogram"),
            means.get("orbit_log_total"),
        )

    def structure_score_from_shared(
        self,
        shared: torch.Tensor,
        target_summary: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        if self.structure_score_head is None:
            raise RuntimeError("Structural score head is disabled")
        if target_summary.ndim != 2 or target_summary.size(1) != self.structure_target_dim:
            raise ValueError(
                f"Expected structural target width {self.structure_target_dim}, "
                f"got {tuple(target_summary.shape)}"
            )
        if time.ndim != 1 or time.size(0) != shared.size(0):
            raise ValueError("Structural score time must be a batch vector")
        value = self.structure_score_head(
            torch.cat((shared, target_summary, time.unsqueeze(-1)), dim=-1)
        )
        if not torch.isfinite(value).all():
            raise FloatingPointError("Non-finite prediction from structural score head")
        return value

    def forward_all(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        flags: torch.Tensor,
        eigenvectors: torch.Tensor,
        eigenvalues: torch.Tensor,
        *,
        time: torch.Tensor | None = None,
        structure_target: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del eigenvectors
        shared = self._encode(x, adj, flags, eigenvalues)
        outputs = self._heads_from_shared(shared)
        if structure_target is not None:
            if time is None:
                raise ValueError("time is required when requesting a structural score prediction")
            outputs["structure_score"] = self.structure_score_from_shared(
                shared, structure_target, time
            )
        for name, value in outputs.items():
            if not torch.isfinite(value).all():
                raise FloatingPointError(f"Non-finite {name} from PPGN spectrum/graphlet score")
        return outputs

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        flags: torch.Tensor,
        eigenvectors: torch.Tensor,
        eigenvalues: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_all(x, adj, flags, eigenvectors, eigenvalues)["spectrum"]

    def graphlet_means_from_outputs(
        self,
        outputs: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = outputs["graphlet_logits"]
        hist = torch.zeros_like(logits)
        for start, stop in self.graphlet_slices:
            hist[:, start:stop] = torch.softmax(logits[:, start:stop], dim=-1)
        mass = torch.sigmoid(outputs["graphlet_mass_logits"])
        if not torch.isfinite(hist).all() or not torch.isfinite(mass).all():
            raise FloatingPointError("Non-finite graphlet summary from PPGN auxiliary head")
        return hist, mass

    def structure_means_from_outputs(
        self,
        outputs: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        graphlet_hist, graphlet_mass = self.graphlet_means_from_outputs(outputs)
        result = {
            "graphlet_histogram": graphlet_hist,
            "graphlet_mass": graphlet_mass,
        }
        if "clustering_logits" in outputs:
            result["clustering_histogram"] = torch.softmax(outputs["clustering_logits"], dim=-1)
        if "orbit_histogram_logits" in outputs and "orbit_log_total_raw" in outputs:
            orbit_hist = torch.softmax(outputs["orbit_histogram_logits"], dim=-1)
            orbit_log_total = F.softplus(outputs["orbit_log_total_raw"])
            result["orbit_histogram"] = orbit_hist
            result["orbit_log_total"] = orbit_log_total
            result["orbit_mean_counts"] = orbit_hist * torch.expm1(orbit_log_total)
        for name, value in result.items():
            if not torch.isfinite(value).all():
                raise FloatingPointError(f"Non-finite {name} from PPGN structural summary")
        return result


def _is_joint_graphlet_score(model: nn.Module) -> bool:
    return (
        hasattr(model, "forward_all")
        and hasattr(model, "graphlet_means_from_outputs")
        and hasattr(model, "graphlet_slices")
    )


class GSDMNodeScore(nn.Module):
    """Noise predictor for the auxiliary degree-feature diffusion."""

    def __init__(
        self,
        *,
        max_feat_num: int,
        hidden_dim: int,
        depth: int,
        structural_features: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.max_feat_num = int(max_feat_num)
        self.depth = int(depth)
        self.structural_features = copy.deepcopy(dict(structural_features or {}))
        rw_cfg = dict(self.structural_features.get("random_walk", {}) or {})
        sp_cfg = dict(self.structural_features.get("shortest_path", {}) or {})
        active = bool(self.structural_features.get("enabled", False))
        self.rw_steps = int(rw_cfg.get("steps", 4)) if active and bool(rw_cfg.get("enabled", False)) else 0
        self.sp_dim = (
            int(sp_cfg.get("max_distance", 5)) + 2
            if active and bool(sp_cfg.get("enabled", False)) else 0
        )
        node_input_dim = self.max_feat_num + self.rw_steps
        layers = []
        for i in range(depth):
            layer_cls = DenseStructuralGCNConv if (self.rw_steps or self.sp_dim) else DenseGCNConv
            if layer_cls is DenseStructuralGCNConv:
                layers.append(
                    DenseStructuralGCNConv(
                        node_input_dim if i == 0 else hidden_dim,
                        hidden_dim,
                        edge_dim=self.sp_dim,
                    )
                )
            else:
                layers.append(DenseGCNConv(max_feat_num if i == 0 else hidden_dim, hidden_dim))
        self.layers = nn.ModuleList(layers)
        fdim = node_input_dim + depth * hidden_dim
        self.final = MLP(fdim, 2 * fdim, max_feat_num, 3)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        flags: torch.Tensor,
        eigenvectors: torch.Tensor,
        eigenvalues: torch.Tensor,
    ) -> torch.Tensor:
        del eigenvectors, eigenvalues
        if self.rw_steps or self.sp_dim:
            _, rw, sp = _structural_features_from_state(adj, flags, self.structural_features)
            h = torch.cat((x, rw), dim=-1) if rw.size(-1) else x
        else:
            sp = x.new_zeros((*adj.shape, 0))
            h = x
        values = [h]
        for layer in self.layers:
            if isinstance(layer, DenseStructuralGCNConv):
                h = torch.tanh(layer(h, adj, flags, sp if sp.size(-1) else None))
            else:
                h = torch.tanh(layer(h, adj))
            values.append(h)
        out = self.final(torch.cat(values, dim=-1))
        return out * flags.unsqueeze(-1).to(out.dtype)


class GSDMNodePPGNScore(nn.Module):
    """PPGN node-noise predictor for the optional all-PPGN backbone."""

    def __init__(
        self,
        *,
        max_feat_num: int,
        max_nodes: int,
        structural_features: Mapping[str, Any] | None = None,
        ppgn_hidden_dim: int = 64,
        ppgn_depth: int = 4,
        ppgn_residual_scale: float = 0.1,
        ppgn_norm_eps: float = 1.0e-5,
        ppgn_input_clip: float = 10.0,
    ) -> None:
        super().__init__()
        self.max_feat_num = int(max_feat_num)
        self.max_nodes = int(max_nodes)
        self.structural_features = copy.deepcopy(dict(structural_features or {}))
        rw_cfg = dict(self.structural_features.get("random_walk", {}) or {})
        sp_cfg = dict(self.structural_features.get("shortest_path", {}) or {})
        active = bool(self.structural_features.get("enabled", False))
        self.rw_steps = int(rw_cfg.get("steps", 4)) if active and bool(rw_cfg.get("enabled", False)) else 0
        self.sp_dim = (
            int(sp_cfg.get("max_distance", 5)) + 2
            if active and bool(sp_cfg.get("enabled", False)) else 0
        )
        node_dim = self.max_feat_num + self.rw_steps
        pair_input_dim = 2 + self.sp_dim + node_dim
        hdim = int(ppgn_hidden_dim)
        self.pair_input = nn.Linear(pair_input_dim, hdim)
        self.pair_input_norm = nn.LayerNorm(hdim)
        self.ppgn_input_clip = float(ppgn_input_clip)
        self.ppgn_layers = nn.ModuleList(
            PPGNLayer(
                hdim,
                residual_scale=float(ppgn_residual_scale),
                norm_eps=float(ppgn_norm_eps),
            )
            for _ in range(int(ppgn_depth))
        )
        self.node_readout = MLP(hdim, 2 * hdim, self.max_feat_num, 2)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        flags: torch.Tensor,
        eigenvectors: torch.Tensor,
        eigenvalues: torch.Tensor,
    ) -> torch.Tensor:
        del eigenvectors, eigenvalues
        binary, rw, sp = _structural_features_from_state(adj, flags, self.structural_features)
        node = torch.cat((x, rw), dim=-1) if rw.size(-1) else x
        b, n, _ = x.shape
        diag_node = x.new_zeros((b, n, n, node.size(-1)))
        idx = torch.arange(n, device=x.device)
        diag_node[:, idx, idx, :] = node
        bounded_adj = torch.tanh(torch.nan_to_num(
            adj, nan=0.0, posinf=self.ppgn_input_clip, neginf=-self.ppgn_input_clip
        ).clamp(-self.ppgn_input_clip, self.ppgn_input_clip))
        parts = [bounded_adj.unsqueeze(-1), binary.unsqueeze(-1)]
        if sp.size(-1):
            parts.append(sp)
        parts.append(diag_node)
        pair = torch.cat(parts, dim=-1)
        pair_mask = flags.to(pair.dtype).unsqueeze(-1) * flags.to(pair.dtype).unsqueeze(-2)
        h = self.pair_input_norm(self.pair_input(pair))
        h = F.elu(h) * pair_mask.unsqueeze(-1)
        for layer in self.ppgn_layers:
            h = layer(h, pair_mask, flags)
        diag = h[:, idx, idx, :]
        out = self.node_readout(diag)
        return out * flags.unsqueeze(-1).to(out.dtype)


class GSDMSpectrumScore(nn.Module):
    """Noise predictor over the spectral state conditioned on the reconstructed graph state."""

    def __init__(self, *, max_feat_num: int, max_nodes: int, hidden_dim: int, depth: int) -> None:
        super().__init__()
        self.max_feat_num = int(max_feat_num)
        self.max_nodes = int(max_nodes)
        self.depth = int(depth)
        layers = []
        for i in range(depth):
            layers.append(DenseGCNConv(max_feat_num if i == 0 else hidden_dim, hidden_dim))
        self.layers = nn.ModuleList(layers)
        fdim = max_feat_num + depth * hidden_dim
        self.node_final = MLP(fdim, 2 * fdim, max_feat_num, 3)
        self.spectrum_final = MLP(max_feat_num + max_nodes, 2 * max_nodes, max_nodes, 2)

    def _encode(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        flags: torch.Tensor,
        eigenvalues: torch.Tensor,
    ) -> torch.Tensor:
        values = [x]
        h = x
        for layer in self.layers:
            h = torch.tanh(layer(h, adj))
            values.append(h)
        h = self.node_final(torch.cat(values, dim=-1))
        h = h * flags.unsqueeze(-1).to(h.dtype)
        count = flags.sum(dim=1, keepdim=True).clamp_min(1).to(h.dtype)
        pooled = h.sum(dim=1) / count
        return torch.cat((pooled, eigenvalues), dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        flags: torch.Tensor,
        eigenvectors: torch.Tensor,
        eigenvalues: torch.Tensor,
    ) -> torch.Tensor:
        del eigenvectors
        return self.spectrum_final(self._encode(x, adj, flags, eigenvalues))


class GSDMSpectrumGraphletScore(GSDMSpectrumScore):
    """Joint log-gap denoiser and clean graphlet-summary predictor.

    The graphlet heads share the spectral denoiser's graph encoder.  This makes
    graphlet supervision an actual auxiliary objective for the spectral model,
    unlike Stage 3 where the predictor is trained only after GSDM has finished.
    """

    def __init__(
        self,
        *,
        max_feat_num: int,
        max_nodes: int,
        hidden_dim: int,
        depth: int,
        graphlet_slices: tuple[tuple[int, int], ...],
        structural_features: Mapping[str, Any] | None = None,
        clustering_bins: int = 0,
        orbit_width: int = 0,
    ) -> None:
        super().__init__(
            max_feat_num=max_feat_num,
            max_nodes=max_nodes,
            hidden_dim=hidden_dim,
            depth=depth,
        )
        self.structural_features = copy.deepcopy(dict(structural_features or {}))
        rw_cfg = dict(self.structural_features.get("random_walk", {}) or {})
        sp_cfg = dict(self.structural_features.get("shortest_path", {}) or {})
        active = bool(self.structural_features.get("enabled", False))
        self.rw_steps = int(rw_cfg.get("steps", 4)) if active and bool(rw_cfg.get("enabled", False)) else 0
        self.sp_dim = (
            int(sp_cfg.get("max_distance", 5)) + 2
            if active and bool(sp_cfg.get("enabled", False)) else 0
        )
        if self.rw_steps or self.sp_dim:
            node_input_dim = int(max_feat_num) + self.rw_steps
            layers: list[nn.Module] = []
            for i in range(int(depth)):
                layers.append(
                    DenseStructuralGCNConv(
                        node_input_dim if i == 0 else int(hidden_dim),
                        int(hidden_dim),
                        edge_dim=self.sp_dim,
                    )
                )
            self.layers = nn.ModuleList(layers)
            fdim = node_input_dim + int(depth) * int(hidden_dim)
            self.node_final = MLP(fdim, 2 * fdim, int(max_feat_num), 3)
        self.graphlet_slices = tuple((int(a), int(b)) for a, b in graphlet_slices)
        if not self.graphlet_slices:
            raise ValueError("graphlet_slices must be non-empty")
        shared_dim = int(max_feat_num) + int(max_nodes)
        width = self.graphlet_slices[-1][1]
        self.graphlet_logits = MLP(shared_dim, 2 * shared_dim, width, 2)
        self.graphlet_mass_logits = MLP(
            shared_dim,
            2 * shared_dim,
            len(self.graphlet_slices),
            2,
        )
        self.clustering_bins = int(clustering_bins)
        self.orbit_width = int(orbit_width)
        self.clustering_logits = (
            MLP(shared_dim, 2 * shared_dim, self.clustering_bins, 2)
            if self.clustering_bins > 0 else None
        )
        self.orbit_histogram_logits = (
            MLP(shared_dim, 2 * shared_dim, self.orbit_width, 2)
            if self.orbit_width > 0 else None
        )
        self.orbit_log_total_raw = (
            MLP(shared_dim, 2 * shared_dim, 1, 2)
            if self.orbit_width > 0 else None
        )

    def _encode(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        flags: torch.Tensor,
        eigenvalues: torch.Tensor,
    ) -> torch.Tensor:
        if not (self.rw_steps or self.sp_dim):
            return super()._encode(x, adj, flags, eigenvalues)
        _, rw, sp = _structural_features_from_state(adj, flags, self.structural_features)
        h = torch.cat((x, rw), dim=-1) if rw.size(-1) else x
        values = [h]
        for layer in self.layers:
            if isinstance(layer, DenseStructuralGCNConv):
                h = torch.tanh(layer(h, adj, flags, sp if sp.size(-1) else None))
            else:  # defensive fallback for old serialized structures
                h = torch.tanh(layer(h, adj))
            values.append(h)
        h = self.node_final(torch.cat(values, dim=-1))
        h = h * flags.unsqueeze(-1).to(h.dtype)
        count = flags.sum(dim=1, keepdim=True).clamp_min(1).to(h.dtype)
        pooled = h.sum(dim=1) / count
        return torch.cat((pooled, eigenvalues), dim=-1)

    def forward_all(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        flags: torch.Tensor,
        eigenvectors: torch.Tensor,
        eigenvalues: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del eigenvectors
        shared = self._encode(x, adj, flags, eigenvalues)
        outputs = {
            "spectrum": self.spectrum_final(shared),
            "graphlet_logits": self.graphlet_logits(shared),
            "graphlet_mass_logits": self.graphlet_mass_logits(shared),
        }
        if self.clustering_logits is not None:
            outputs["clustering_logits"] = self.clustering_logits(shared)
        if self.orbit_histogram_logits is not None and self.orbit_log_total_raw is not None:
            outputs["orbit_histogram_logits"] = self.orbit_histogram_logits(shared)
            outputs["orbit_log_total_raw"] = self.orbit_log_total_raw(shared)
        return outputs

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        flags: torch.Tensor,
        eigenvectors: torch.Tensor,
        eigenvalues: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_all(x, adj, flags, eigenvectors, eigenvalues)["spectrum"]

    def graphlet_means_from_outputs(
        self,
        outputs: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = outputs["graphlet_logits"]
        hist = torch.zeros_like(logits)
        for start, stop in self.graphlet_slices:
            hist[:, start:stop] = torch.softmax(logits[:, start:stop], dim=-1)
        mass = torch.sigmoid(outputs["graphlet_mass_logits"])
        return hist, mass

    def structure_means_from_outputs(
        self,
        outputs: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        graphlet_hist, graphlet_mass = self.graphlet_means_from_outputs(outputs)
        result = {
            "graphlet_histogram": graphlet_hist,
            "graphlet_mass": graphlet_mass,
        }
        if "clustering_logits" in outputs:
            result["clustering_histogram"] = torch.softmax(outputs["clustering_logits"], dim=-1)
        if "orbit_histogram_logits" in outputs and "orbit_log_total_raw" in outputs:
            orbit_hist = torch.softmax(outputs["orbit_histogram_logits"], dim=-1)
            orbit_log_total = F.softplus(outputs["orbit_log_total_raw"])
            result["orbit_histogram"] = orbit_hist
            result["orbit_log_total"] = orbit_log_total
            result["orbit_mean_counts"] = orbit_hist * torch.expm1(orbit_log_total)
        return result


class VPSDE:
    def __init__(self, *, beta_min: float, beta_max: float, num_scales: int, device: torch.device) -> None:
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        self.N = int(num_scales)
        self.T = 1.0
        self.discrete_betas = torch.linspace(
            self.beta_min / self.N, self.beta_max / self.N, self.N,
            dtype=torch.float32, device=device,
        )
        self.alphas = 1.0 - self.discrete_betas

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def marginal_coeff(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        log_mean = -0.25 * t.square() * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        mean_coeff = torch.exp(log_mean)
        std = torch.sqrt((1.0 - torch.exp(2.0 * log_mean)).clamp_min(1.0e-12))
        return mean_coeff, std

    def marginal_x(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        coeff, std = self.marginal_coeff(t)
        return coeff[:, None, None] * x, std

    def marginal_spectrum(self, lam: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        coeff, std = self.marginal_coeff(t)
        return coeff[:, None] * lam, std


def mask_x(x: torch.Tensor, flags: torch.Tensor) -> torch.Tensor:
    return x * flags.unsqueeze(-1).to(x.dtype)


def mask_adj(adj: torch.Tensor, flags: torch.Tensor) -> torch.Tensor:
    f = flags.to(adj.dtype)
    return adj * f.unsqueeze(-1) * f.unsqueeze(-2)


def reconstruct_adjacency(eigenvectors: torch.Tensor, eigenvalues: torch.Tensor) -> torch.Tensor:
    return eigenvectors @ torch.diag_embed(eigenvalues) @ eigenvectors.transpose(-1, -2)


def combinatorial_laplacian(adj: torch.Tensor, flags: torch.Tensor | None = None) -> torch.Tensor:
    """Return ``L=D-A`` for a dense batched adjacency tensor."""
    if adj.ndim != 3:
        raise ValueError("combinatorial_laplacian expects [B,N,N]")
    degrees = adj.sum(dim=-1)
    lap = torch.diag_embed(degrees) - adj
    if flags is not None:
        lap = mask_adj(lap, flags)
    return lap


def laplacian_to_adjacency_scores(lap: torch.Tensor, flags: torch.Tensor) -> torch.Tensor:
    """Map a reconstructed combinatorial-Laplacian state to edge scores.

    For a clean simple graph, ``L_ij=-A_ij`` for ``i != j``.  The diagonal is
    therefore discarded before the graph state is presented to the GNN or
    thresholded at generation time.
    """
    if lap.ndim != 3:
        raise ValueError("laplacian_to_adjacency_scores expects [B,N,N]")
    scores = -0.5 * (lap + lap.transpose(-1, -2))
    idx = torch.arange(scores.size(-1), device=scores.device)
    scores[:, idx, idx] = 0.0
    return mask_adj(scores, flags)


def _laplacian_eigh_padded(
    adj: torch.Tensor, sizes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Eigendecompose active Laplacians without mixing padding zero modes.

    The real ``n`` Laplacian eigenvalues are stored in the prefix ``[:n]``;
    padded coordinates occupy ``[n:]``.  The corresponding full eigenbasis is
    block diagonal with the active Laplacian eigenvectors in the leading block
    and identity vectors for padding coordinates.  This convention avoids the
    ambiguous zero-eigenspace mixing that occurs when ``torch.linalg.eigh`` is
    applied directly to a zero-padded Laplacian.
    """
    if adj.ndim != 3 or sizes.ndim != 1 or adj.size(0) != sizes.numel():
        raise ValueError("invalid padded Laplacian eigendecomposition inputs")
    b, nmax, _ = adj.shape
    values = torch.zeros((b, nmax), dtype=adj.dtype, device=adj.device)
    vectors = torch.eye(nmax, dtype=adj.dtype, device=adj.device).unsqueeze(0).repeat(b, 1, 1)
    for row, n_raw in enumerate(sizes.tolist()):
        n = int(n_raw)
        if n <= 0 or n > nmax:
            raise ValueError(f"invalid active graph size {n} for nmax={nmax}")
        active = adj[row, :n, :n]
        lap = torch.diag(active.sum(dim=-1)) - active
        lam, u = torch.linalg.eigh(lap)
        values[row, :n] = lam
        vectors[row, :n, :n] = u
    return values, vectors



def _fit_laplacian_log_gap_stats(
    eigenvalues: torch.Tensor,
    flags: torch.Tensor,
    *,
    epsilon: float,
    min_std: float,
) -> dict[str, torch.Tensor]:
    """Fit per-coordinate normalization for log successive Laplacian gaps.

    Coordinate ``j`` (j>=1) represents ``log(lambda_j-lambda_{j-1}+epsilon)``
    whenever that coordinate exists for the active graph. Coordinate zero is a
    fixed dummy matching the trivial Laplacian eigenvalue and is never diffused.
    """
    if eigenvalues.shape != flags.shape:
        raise ValueError("log-gap statistics require eigenvalues/flags with matching [B,N]")
    nmax = eigenvalues.size(1)
    mean = torch.zeros(nmax, dtype=eigenvalues.dtype)
    std = torch.ones(nmax, dtype=eigenvalues.dtype)
    counts = torch.zeros(nmax, dtype=torch.long)
    sizes = flags.sum(dim=1).long()
    for j in range(1, nmax):
        valid = sizes > j
        if not bool(valid.any()):
            continue
        gap = (eigenvalues[valid, j] - eigenvalues[valid, j - 1]).clamp_min(0.0)
        values = torch.log(gap + float(epsilon))
        mean[j] = values.mean()
        # population std is stable even if only one graph has this coordinate
        sigma = values.std(unbiased=False) if values.numel() > 1 else torch.tensor(0.0, dtype=values.dtype)
        std[j] = sigma.clamp_min(float(min_std))
        counts[j] = int(values.numel())
    return {"mean": mean, "std": std, "count": counts}


def _laplacian_eigenvalues_to_log_gap_state(
    eigenvalues: torch.Tensor,
    flags: torch.Tensor,
    *,
    mean: torch.Tensor,
    std: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    if eigenvalues.shape != flags.shape:
        raise ValueError("log-gap transform expects eigenvalues/flags with matching [B,N]")
    mean = mean.to(device=eigenvalues.device, dtype=eigenvalues.dtype)
    std = std.to(device=eigenvalues.device, dtype=eigenvalues.dtype)
    state = torch.zeros_like(eigenvalues)
    gaps = (eigenvalues[:, 1:] - eigenvalues[:, :-1]).clamp_min(0.0)
    raw = torch.log(gaps + float(epsilon))
    state[:, 1:] = (raw - mean[1:]) / std[1:].clamp_min(1.0e-12)
    mask = eigen_mask_from_flags(flags, "laplacian_nonzero_prefix")
    return state * mask


def _log_gap_state_to_laplacian_eigenvalues(
    state: torch.Tensor,
    flags: torch.Tensor,
    *,
    mean: torch.Tensor,
    std: torch.Tensor,
    epsilon: float,
    exp_clip: float,
) -> torch.Tensor:
    """Invert normalized log gaps with a hard positivity/order guarantee."""
    if state.shape != flags.shape:
        raise ValueError("inverse log-gap transform expects state/flags with matching [B,N]")
    mean = mean.to(device=state.device, dtype=state.dtype)
    std = std.to(device=state.device, dtype=state.dtype)
    raw = state[:, 1:] * std[1:] + mean[1:]
    raw = torch.clamp(raw, min=-float(exp_clip), max=float(exp_clip))
    gaps = (torch.exp(raw) - float(epsilon)).clamp_min(0.0)
    sizes = flags.sum(dim=1).long()
    eig = torch.zeros_like(state)
    if state.size(1) > 1:
        eig[:, 1:] = torch.cumsum(gaps, dim=1)
    active = torch.arange(state.size(1), device=state.device).unsqueeze(0) < sizes.unsqueeze(1)
    return eig * active.to(eig.dtype)


def _spectral_state_to_operator_eigenvalues(
    state: torch.Tensor,
    flags: torch.Tensor,
    *,
    spectral_operator: str,
    spectral_transform: Mapping[str, Any] | None,
) -> torch.Tensor:
    mode = str((spectral_transform or {}).get("kind", "direct_eigenvalues")).lower()
    if mode in {"direct_eigenvalues", "direct", "eigenvalues"}:
        return state
    if mode == "laplacian_log_gap":
        if spectral_operator != "combinatorial_laplacian":
            raise ValueError("laplacian_log_gap is only valid for the combinatorial Laplacian")
        return _log_gap_state_to_laplacian_eigenvalues(
            state,
            flags,
            mean=torch.as_tensor((spectral_transform or {})["mean"]),
            std=torch.as_tensor((spectral_transform or {})["std"]),
            epsilon=float((spectral_transform or {}).get("epsilon", 1.0e-6)),
            exp_clip=float((spectral_transform or {}).get("exp_clip", 20.0)),
        )
    raise ValueError(f"Unknown spectral parameterization: {mode!r}")


def _operator_eigenvalues_to_spectral_state(
    eigenvalues: torch.Tensor,
    flags: torch.Tensor,
    *,
    spectral_operator: str,
    spectral_transform: Mapping[str, Any] | None,
) -> torch.Tensor:
    mode = str((spectral_transform or {}).get("kind", "direct_eigenvalues")).lower()
    if mode in {"direct_eigenvalues", "direct", "eigenvalues"}:
        return eigenvalues
    if mode == "laplacian_log_gap":
        if spectral_operator != "combinatorial_laplacian":
            raise ValueError("laplacian_log_gap is only valid for the combinatorial Laplacian")
        return _laplacian_eigenvalues_to_log_gap_state(
            eigenvalues,
            flags,
            mean=torch.as_tensor((spectral_transform or {})["mean"]),
            std=torch.as_tensor((spectral_transform or {})["std"]),
            epsilon=float((spectral_transform or {}).get("epsilon", 1.0e-6)),
        )
    raise ValueError(f"Unknown spectral parameterization: {mode!r}")


def _operator_to_adjacency_state(
    operator_matrix: torch.Tensor,
    flags: torch.Tensor,
    spectral_operator: str,
) -> torch.Tensor:
    if spectral_operator == "adjacency":
        return mask_adj(operator_matrix, flags)
    if spectral_operator == "combinatorial_laplacian":
        return laplacian_to_adjacency_scores(operator_matrix, flags)
    raise ValueError(f"Unknown spectral operator {spectral_operator!r}")


def eigen_mask_from_flags(flags: torch.Tensor, mode: str = "official_extremes") -> torch.Tensor:
    mode = str(mode).lower()
    b, nmax = flags.shape
    mask = torch.zeros((b, nmax), dtype=torch.float32, device=flags.device)
    if mode == "official_extremes":
        for i, n in enumerate(flags.sum(dim=1).tolist()):
            n = int(n)
            left = n // 2
            right = n - left
            if left:
                mask[i, :left] = 1.0
            if right:
                mask[i, -right:] = 1.0
        return mask
    if mode == "all_padded_eigenvalues":
        return torch.ones_like(mask)
    if mode == "active_prefix":
        for i, n in enumerate(flags.sum(dim=1).tolist()):
            mask[i, : int(n)] = 1.0
        return mask
    if mode == "laplacian_nonzero_prefix":
        for i, n in enumerate(flags.sum(dim=1).tolist()):
            n = int(n)
            if n > 1:
                mask[i, 1:n] = 1.0
        return mask
    raise ValueError(f"Unknown vanilla GSDM eigen_mask mode: {mode}")


def degree_features(adj: torch.Tensor, flags: torch.Tensor, max_feat_num: int) -> torch.Tensor:
    degrees = adj.sum(dim=-1).round().long()
    valid = flags.bool()
    if valid.any():
        minimum = int(degrees[valid].min().item())
        maximum = int(degrees[valid].max().item())
        if minimum < 0 or maximum >= int(max_feat_num):
            raise ValueError(
                f"Node degree outside degree-one-hot support [0,{int(max_feat_num)-1}]: "
                f"min={minimum}, max={maximum}"
            )
    x = F.one_hot(degrees.clamp_min(0), num_classes=max_feat_num).to(torch.float32)
    return mask_x(x, flags)


def _padded_dataset(
    graphs: list[nx.Graph], *, max_nodes: int, max_feat_num: int,
    spectral_operator: str = "adjacency",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    adjacencies = []
    flags = []
    sizes = []
    for graph in graphs:
        n = graph.number_of_nodes()
        if n > max_nodes:
            raise ValueError(f"Graph size {n} exceeds model.max_nodes={max_nodes}")
        a = np.zeros((max_nodes, max_nodes), dtype=np.float32)
        a[:n, :n] = nx.to_numpy_array(graph, nodelist=range(n), dtype=np.float32)
        adjacencies.append(a)
        f = np.zeros(max_nodes, dtype=np.float32)
        f[:n] = 1.0
        flags.append(f)
        sizes.append(n)
    adj = torch.tensor(np.stack(adjacencies), dtype=torch.float32)
    flg = torch.tensor(np.stack(flags), dtype=torch.float32)
    size = torch.tensor(sizes, dtype=torch.long)
    x = degree_features(adj, flg, max_feat_num)
    if spectral_operator == "adjacency":
        lam, u = torch.linalg.eigh(adj)
    elif spectral_operator == "combinatorial_laplacian":
        lam, u = _laplacian_eigh_padded(adj, size)
    else:
        raise ValueError(f"Unknown spectral operator {spectral_operator!r}")
    return x, adj, flg, size, u, lam



def _joint_structure_targets(
    graphs: list[nx.Graph],
    structure_cfg: Mapping[str, Any],
) -> tuple[tuple[torch.Tensor, ...], dict[str, Any]]:
    """Extract clean structural targets for the joint auxiliary heads.

    The orbit representation is a 15-bin histogram over the standard ORCA
    node-orbit roles (0..14), built from the evaluator-compatible mean per-node
    orbit count vector.  A separate ``log1p(total orbit mass)`` target retains
    magnitude information that the normalized histogram would otherwise lose.
    """
    from dataclasses import asdict
    from grapher.rewiring_mlp.generic.basis import TopologyGraphletBasis
    from grapher.rewiring_mlp.generic.graphlets import extract_topology_graphlet_target
    from grapher.rewiring_mlp.properties.summary import (
        SummaryConfig,
        clustering_histogram,
        python_orbit_count_vector,
    )

    graphlet_cfg = dict(structure_cfg.get("graphlet", {}) or {})
    clustering_cfg = dict(structure_cfg.get("clustering", {}) or {})
    orbit_cfg = dict(structure_cfg.get("orbit", {}) or {})
    cfg = SummaryConfig.from_dict(
        {
            "clustering_summary": False,
            "spectral_summary": False,
            "motif_proxy": False,
            "orbit_count": False,
            "graphlet_history": True,
            "graphlet_k_min": 3,
            "graphlet_k_max": 5,
            "graphlet_connected_only": True,
            "graphlet_topology_filter": "all",
            "graphlet_backend": "exact",
            "graphlet_num_samples": None,
        }
    )
    basis = TopologyGraphletBasis.from_config(cfg)
    graphlet_targets: list[np.ndarray] = []
    graphlet_masses: list[np.ndarray] = []
    clustering_targets: list[np.ndarray] = []
    orbit_histograms: list[np.ndarray] = []
    orbit_log_totals: list[np.ndarray] = []
    clustering_bins = int(clustering_cfg.get("bins", 100))
    orbit_width = int(orbit_cfg.get("width", 15))

    for graph in graphs:
        g = nx.convert_node_labels_to_integers(graph, ordering="sorted")
        target, mass = extract_topology_graphlet_target(
            g,
            graphlet_basis=basis,
            summary_config=cfg,
        )
        graphlet_targets.append(target.astype(np.float32))
        graphlet_masses.append(mass.astype(np.float32))

        if bool(clustering_cfg.get("enabled", False)):
            clustering_targets.append(
                clustering_histogram(g, bins=clustering_bins).astype(np.float32)
            )
        else:
            clustering_targets.append(np.zeros(clustering_bins, dtype=np.float32))

        if bool(orbit_cfg.get("enabled", False)):
            counts = np.maximum(
                np.asarray(python_orbit_count_vector(g), dtype=np.float64).reshape(-1),
                0.0,
            )
            if counts.size != orbit_width:
                raise ValueError(
                    f"Expected orbit target width {orbit_width}, got {counts.size}."
                )
            total = float(counts.sum())
            hist = counts / total if total > 0.0 else np.zeros_like(counts)
            orbit_histograms.append(hist.astype(np.float32))
            orbit_log_totals.append(np.asarray([np.log1p(total)], dtype=np.float32))
        else:
            orbit_histograms.append(np.zeros(orbit_width, dtype=np.float32))
            orbit_log_totals.append(np.zeros(1, dtype=np.float32))

    meta = {
        "graphlet_slices": [list(pair) for pair in basis.slices],
        "graphlet_basis": basis.to_dict(),
        "summary_config": asdict(cfg),
        "width": int(basis.width),
        "orders": list(graphlet_cfg.get("orders", [3, 4, 5])),
        "clustering_enabled": bool(clustering_cfg.get("enabled", False)),
        "clustering_bins": clustering_bins,
        "clustering_representation": "normalized_node_clustering_coefficient_histogram_[0,1]",
        "orbit_enabled": bool(orbit_cfg.get("enabled", False)),
        "orbit_width": orbit_width,
        "orbit_representation": "normalized_histogram_over_mean_per_node_ORCA_0_14_plus_log1p_total",
    }
    tensors = (
        torch.tensor(np.stack(graphlet_targets), dtype=torch.float32),
        torch.tensor(np.stack(graphlet_masses), dtype=torch.float32),
        torch.tensor(np.stack(clustering_targets), dtype=torch.float32),
        torch.tensor(np.stack(orbit_histograms), dtype=torch.float32),
        torch.tensor(np.stack(orbit_log_totals), dtype=torch.float32),
    )
    return tensors, meta


def _joint_graphlet_targets(
    graphs: list[nx.Graph],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Backwards-compatible graphlet-only target helper used by older tests."""
    cfg = {
        "enabled": True,
        "graphlet": {"enabled": True, "orders": [3, 4, 5]},
        "clustering": {"enabled": False, "bins": 100},
        "orbit": {"enabled": False, "width": 15},
    }
    tensors, meta = _joint_structure_targets(graphs, cfg)
    return tensors[0], tensors[1], meta


def _structure_auxiliary_loss(
    model: nn.Module,
    outputs: Mapping[str, torch.Tensor],
    graphlet_target: torch.Tensor,
    graphlet_mass_target: torch.Tensor,
    clustering_target: torch.Tensor | None,
    orbit_histogram_target: torch.Tensor | None,
    orbit_log_total_target: torch.Tensor | None,
    *,
    structure_cfg: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    if not hasattr(model, "graphlet_slices"):
        raise TypeError("joint structural model is missing graphlet_slices")
    graphlet_cfg = dict(structure_cfg.get("graphlet", {}) or {})
    clustering_cfg = dict(structure_cfg.get("clustering", {}) or {})
    orbit_cfg = dict(structure_cfg.get("orbit", {}) or {})

    logits = outputs["graphlet_logits"]
    losses = []
    mean_abs = []
    for start, stop in model.graphlet_slices:
        block_target = graphlet_target[:, start:stop]
        valid = block_target.sum(dim=-1) > 0.0
        if bool(valid.any()):
            logp = torch.log_softmax(logits[valid, start:stop], dim=-1)
            losses.append(-(block_target[valid] * logp).sum(dim=-1).mean())
            pred = torch.softmax(logits[valid, start:stop], dim=-1)
            mean_abs.append((pred - block_target[valid]).abs().mean())
    graphlet_hist_loss = torch.stack(losses).mean() if losses else logits.sum() * 0.0
    graphlet_mass_loss = F.binary_cross_entropy_with_logits(
        outputs["graphlet_mass_logits"], graphlet_mass_target
    )
    total = (
        float(graphlet_cfg.get("histogram_weight", 1.0)) * graphlet_hist_loss
        + float(graphlet_cfg.get("mass_weight", 0.25)) * graphlet_mass_loss
    )
    graphlet_hist_mae = (
        torch.stack(mean_abs).mean() if mean_abs else graphlet_hist_loss.detach() * 0.0
    )
    metrics: dict[str, float] = {
        "graphlet_histogram_loss": float(graphlet_hist_loss.detach().item()),
        "graphlet_mass_loss": float(graphlet_mass_loss.detach().item()),
        "graphlet_histogram_mae": float(graphlet_hist_mae.detach().item()),
    }

    if bool(clustering_cfg.get("enabled", False)):
        if clustering_target is None or "clustering_logits" not in outputs:
            raise RuntimeError("Clustering summary enabled but target/head is unavailable")
        logp = torch.log_softmax(outputs["clustering_logits"], dim=-1)
        clustering_ce = -(clustering_target * logp).sum(dim=-1).mean()
        clustering_pred = torch.softmax(outputs["clustering_logits"], dim=-1)
        # CDF discrepancy is the equal-bin 1-D Wasserstein distance up to a
        # constant factor and aligns the auxiliary target with clustering MMD.
        clustering_cdf = torch.abs(
            torch.cumsum(clustering_pred - clustering_target, dim=-1)[..., :-1]
        ).mean()
        clustering_loss = clustering_ce + float(clustering_cfg.get("cdf_weight", 1.0)) * clustering_cdf
        total = total + float(clustering_cfg.get("histogram_weight", 0.25)) * clustering_loss
        metrics.update(
            {
                "clustering_histogram_loss": float(clustering_ce.detach().item()),
                "clustering_cdf_mae": float(clustering_cdf.detach().item()),
                "clustering_histogram_mae": float(
                    (clustering_pred - clustering_target).abs().mean().detach().item()
                ),
            }
        )

    if bool(orbit_cfg.get("enabled", False)):
        if (
            orbit_histogram_target is None
            or orbit_log_total_target is None
            or "orbit_histogram_logits" not in outputs
            or "orbit_log_total_raw" not in outputs
        ):
            raise RuntimeError("Orbit summary enabled but target/head is unavailable")
        valid = orbit_histogram_target.sum(dim=-1) > 0.0
        if bool(valid.any()):
            orbit_logp = torch.log_softmax(outputs["orbit_histogram_logits"][valid], dim=-1)
            orbit_hist_loss = -(
                orbit_histogram_target[valid] * orbit_logp
            ).sum(dim=-1).mean()
            orbit_pred = torch.softmax(outputs["orbit_histogram_logits"], dim=-1)
            orbit_hist_mae = (
                orbit_pred[valid] - orbit_histogram_target[valid]
            ).abs().mean()
        else:
            orbit_hist_loss = outputs["orbit_histogram_logits"].sum() * 0.0
            orbit_hist_mae = orbit_hist_loss.detach() * 0.0
        orbit_log_total_pred = F.softplus(outputs["orbit_log_total_raw"])
        orbit_total_loss = F.mse_loss(orbit_log_total_pred, orbit_log_total_target)
        total = (
            total
            + float(orbit_cfg.get("histogram_weight", 0.25)) * orbit_hist_loss
            + float(orbit_cfg.get("log_total_weight", 0.10)) * orbit_total_loss
        )
        metrics.update(
            {
                "orbit_histogram_loss": float(orbit_hist_loss.detach().item()),
                "orbit_histogram_mae": float(orbit_hist_mae.detach().item()),
                "orbit_log_total_rmse": float(
                    torch.sqrt(orbit_total_loss.detach().clamp_min(0.0)).item()
                ),
            }
        )
    return total, metrics


def _structure_log_likelihood_per_graph(
    model: nn.Module,
    outputs: Mapping[str, torch.Tensor],
    graphlet_target: torch.Tensor,
    graphlet_mass_target: torch.Tensor,
    clustering_target: torch.Tensor | None,
    orbit_histogram_target: torch.Tensor | None,
    orbit_log_total_target: torch.Tensor | None,
    *,
    structure_cfg: Mapping[str, Any],
    orbit_log_total_sigma: float,
) -> torch.Tensor:
    """Proper log-likelihood terms used to define structural classifier guidance.

    CDF/Wasserstein-style auxiliary penalties are intentionally excluded because
    they are not log densities.  The resulting scalar per graph defines the
    teacher score ``grad_z log p(S_0 | X_t, z_t)`` used to supervise the second
    structural-score head.
    """
    graphlet_cfg = dict(structure_cfg.get("graphlet", {}) or {})
    clustering_cfg = dict(structure_cfg.get("clustering", {}) or {})
    orbit_cfg = dict(structure_cfg.get("orbit", {}) or {})
    b = graphlet_target.size(0)
    total = graphlet_target.new_zeros((b,))

    hist_terms = []
    for start, stop in model.graphlet_slices:
        block_target = graphlet_target[:, start:stop]
        valid = block_target.sum(dim=-1) > 0.0
        block = graphlet_target.new_zeros((b,))
        if bool(valid.any()):
            logp = torch.log_softmax(outputs["graphlet_logits"][:, start:stop], dim=-1)
            block[valid] = (block_target[valid] * logp[valid]).sum(dim=-1)
        hist_terms.append(block)
    if hist_terms:
        hist_ll = torch.stack(hist_terms, dim=-1).mean(dim=-1)
        total = total + float(graphlet_cfg.get("histogram_weight", 1.0)) * hist_ll

    mass_ll = -F.binary_cross_entropy_with_logits(
        outputs["graphlet_mass_logits"],
        graphlet_mass_target,
        reduction="none",
    ).mean(dim=-1)
    total = total + float(graphlet_cfg.get("mass_weight", 0.25)) * mass_ll

    if bool(clustering_cfg.get("enabled", False)):
        if clustering_target is None or "clustering_logits" not in outputs:
            raise RuntimeError("Clustering likelihood requested without target/head")
        cluster_ll = (
            clustering_target * torch.log_softmax(outputs["clustering_logits"], dim=-1)
        ).sum(dim=-1)
        total = total + float(clustering_cfg.get("histogram_weight", 0.25)) * cluster_ll

    if bool(orbit_cfg.get("enabled", False)):
        if (
            orbit_histogram_target is None
            or orbit_log_total_target is None
            or "orbit_histogram_logits" not in outputs
            or "orbit_log_total_raw" not in outputs
        ):
            raise RuntimeError("Orbit likelihood requested without target/head")
        valid = orbit_histogram_target.sum(dim=-1) > 0.0
        orbit_hist_ll = orbit_histogram_target.new_zeros((b,))
        if bool(valid.any()):
            orbit_logp = torch.log_softmax(outputs["orbit_histogram_logits"], dim=-1)
            orbit_hist_ll[valid] = (
                orbit_histogram_target[valid] * orbit_logp[valid]
            ).sum(dim=-1)
        total = total + float(orbit_cfg.get("histogram_weight", 0.25)) * orbit_hist_ll

        sigma = max(float(orbit_log_total_sigma), 1.0e-6)
        orbit_total_pred = F.softplus(outputs["orbit_log_total_raw"])
        residual = (orbit_total_pred - orbit_log_total_target) / sigma
        orbit_total_ll = -0.5 * residual.square().mean(dim=-1)
        total = total + float(orbit_cfg.get("log_total_weight", 0.10)) * orbit_total_ll

    return total


def _clean_structure_target_vector(
    graphlet_target: torch.Tensor,
    graphlet_mass_target: torch.Tensor,
    clustering_target: torch.Tensor | None,
    orbit_histogram_target: torch.Tensor | None,
    orbit_log_total_target: torch.Tensor | None,
    *,
    structure_cfg: Mapping[str, Any],
) -> torch.Tensor:
    clustering_cfg = dict(structure_cfg.get("clustering", {}) or {})
    orbit_cfg = dict(structure_cfg.get("orbit", {}) or {})
    return _pack_structure_summary_vector(
        graphlet_target,
        graphlet_mass_target,
        clustering_target if bool(clustering_cfg.get("enabled", False)) else None,
        orbit_histogram_target if bool(orbit_cfg.get("enabled", False)) else None,
        orbit_log_total_target if bool(orbit_cfg.get("enabled", False)) else None,
    )


def _graphlet_auxiliary_loss(
    model: nn.Module,
    outputs: Mapping[str, torch.Tensor],
    target: torch.Tensor,
    mass_target: torch.Tensor,
    *,
    histogram_weight: float,
    mass_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Legacy wrapper around the generalized structural-summary objective."""
    return _structure_auxiliary_loss(
        model,
        outputs,
        target,
        mass_target,
        None,
        None,
        None,
        structure_cfg={
            "graphlet": {
                "enabled": True,
                "histogram_weight": float(histogram_weight),
                "mass_weight": float(mass_weight),
            },
            "clustering": {"enabled": False},
            "orbit": {"enabled": False},
        },
    )


def _loss_batch(
    model_x: GSDMNodeScore,
    model_lam: GSDMSpectrumScore,
    batch: tuple[torch.Tensor, ...],
    *,
    sde_x: VPSDE,
    sde_lam: VPSDE,
    eps: float,
    eigen_mask_mode: str,
    spectral_operator: str = "adjacency",
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    x0, adj0, flags, _, u, lam0 = batch
    b = x0.size(0)
    if generator is None:
        t = torch.rand(b, device=x0.device) * (1.0 - eps) + eps
        z_x = torch.randn_like(x0)
        z_lam = torch.randn_like(lam0)
    else:
        t = torch.rand((b,), device=x0.device, generator=generator) * (1.0 - eps) + eps
        z_x = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device, generator=generator)
        z_lam = torch.randn(lam0.shape, dtype=lam0.dtype, device=lam0.device, generator=generator)
    z_x = mask_x(z_x, flags)
    e_mask = eigen_mask_from_flags(flags, eigen_mask_mode)

    mean_x, std_x = sde_x.marginal_x(x0, t)
    xt = mask_x(mean_x + std_x[:, None, None] * z_x, flags)
    mean_lam, std_lam = sde_lam.marginal_spectrum(lam0, t)
    lam_t = (mean_lam * e_mask + std_lam[:, None] * z_lam) * e_mask
    operator_t = reconstruct_adjacency(u, lam_t)
    adj_t = _operator_to_adjacency_state(operator_t, flags, spectral_operator)

    pred_x = model_x(xt, adj_t, flags, u, lam_t)
    pred_lam = model_lam(xt, adj_t, flags, u, lam_t)

    # The released score wrapper converts raw network output to score=-eps/std,
    # making its DSM objective exactly an epsilon-prediction objective.
    loss_x = 0.5 * (pred_x - z_x).square().reshape(b, -1).sum(dim=-1).mean()
    if spectral_operator == "combinatorial_laplacian":
        # Laplacian spectra use an explicit active-prefix padding convention.
        # Padded coordinates are not stochastic variables and therefore must
        # not contribute irreducible noise-prediction loss.
        spectral_error = (pred_lam - z_lam) * e_mask
    else:
        # Preserve the released adjacency-GSDM training objective bit-for-bit.
        spectral_error = pred_lam - z_lam
    loss_lam = 0.5 * spectral_error.square().reshape(b, -1).sum(dim=-1).mean()
    return loss_x, loss_lam


def _loss_batch_loggap(
    model_x: GSDMNodeScore,
    model_lam: GSDMSpectrumScore,
    batch: tuple[torch.Tensor, ...],
    *,
    sde_x: VPSDE,
    sde_lam: VPSDE,
    eps: float,
    eigen_mask_mode: str,
    spectral_transform: Mapping[str, Any],
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """VP epsilon-prediction loss in standardized Laplacian log-gap space.

    This is the controlled no-auxiliary-head counterpart of
    ``_loss_batch_loggap_graphlet``.  The GNN still sees the decoded continuous
    Laplacian-derived adjacency state, but the spectrum network predicts noise
    only for the standardized log-gap state.
    """

    if len(batch) != 6:
        raise ValueError("log-gap-only training expects 6 tensors per batch")
    x0, _adj0, flags, _sizes, u, lam0 = batch
    b = x0.size(0)
    state0 = _operator_eigenvalues_to_spectral_state(
        lam0,
        flags,
        spectral_operator="combinatorial_laplacian",
        spectral_transform=spectral_transform,
    )
    if generator is None:
        t = torch.rand(b, device=x0.device) * (1.0 - eps) + eps
        z_x = torch.randn_like(x0)
        z_state = torch.randn_like(state0)
    else:
        t = torch.rand((b,), device=x0.device, generator=generator) * (1.0 - eps) + eps
        z_x = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device, generator=generator)
        z_state = torch.randn(
            state0.shape,
            dtype=state0.dtype,
            device=state0.device,
            generator=generator,
        )
    z_x = mask_x(z_x, flags)
    e_mask = eigen_mask_from_flags(flags, eigen_mask_mode)
    z_state = z_state * e_mask

    mean_x, std_x = sde_x.marginal_x(x0, t)
    xt = mask_x(mean_x + std_x[:, None, None] * z_x, flags)
    mean_state, std_state = sde_lam.marginal_spectrum(state0, t)
    state_t = (mean_state * e_mask + std_state[:, None] * z_state) * e_mask
    lam_t = _spectral_state_to_operator_eigenvalues(
        state_t,
        flags,
        spectral_operator="combinatorial_laplacian",
        spectral_transform=spectral_transform,
    )
    operator_t = reconstruct_adjacency(u, lam_t)
    adj_t = _operator_to_adjacency_state(
        operator_t,
        flags,
        "combinatorial_laplacian",
    )

    pred_x = model_x(xt, adj_t, flags, u, state_t)
    pred_state = model_lam(xt, adj_t, flags, u, state_t)
    loss_x = 0.5 * (pred_x - z_x).square().reshape(b, -1).sum(dim=-1).mean()
    loss_state = 0.5 * (
        (((pred_state - z_state) * e_mask).square().reshape(b, -1).sum(dim=-1))
    ).mean()
    return loss_x, loss_state


def _loss_batch_loggap_graphlet(
    model_x: GSDMNodeScore,
    model_lam: nn.Module,
    batch: tuple[torch.Tensor, ...],
    *,
    sde_x: VPSDE,
    sde_lam: VPSDE,
    eps: float,
    eigen_mask_mode: str,
    spectral_transform: Mapping[str, Any],
    graphlet_cfg: Mapping[str, Any],
    structure_score_cfg: Mapping[str, Any] | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    if len(batch) == 8:
        # Backwards compatibility for graphlet-only checkpoints/tests.
        x0, _adj0, flags, _sizes, u, lam0, graphlet_target, graphlet_mass = batch
        clustering_target = orbit_histogram_target = orbit_log_total_target = None
    elif len(batch) == 11:
        (
            x0, _adj0, flags, _sizes, u, lam0,
            graphlet_target, graphlet_mass, clustering_target,
            orbit_histogram_target, orbit_log_total_target,
        ) = batch
    else:
        raise ValueError(
            "joint log-gap structural training expects 8 legacy or 11 structural tensors per batch"
        )

    score_cfg = dict(structure_score_cfg or {})
    score_enabled = bool(score_cfg.get("enabled", False))
    # Validation is normally wrapped in torch.no_grad().  The structural-score
    # teacher itself is a derivative with respect to the noisy log-gap state,
    # so locally re-enable autograd when the second head is active.
    context = torch.enable_grad() if score_enabled else nullcontext()
    with context:
        b = x0.size(0)
        state0 = _operator_eigenvalues_to_spectral_state(
            lam0,
            flags,
            spectral_operator="combinatorial_laplacian",
            spectral_transform=spectral_transform,
        )
        if generator is None:
            t = torch.rand(b, device=x0.device) * (1.0 - eps) + eps
            z_x = torch.randn_like(x0)
            z_state = torch.randn_like(state0)
        else:
            t = torch.rand((b,), device=x0.device, generator=generator) * (1.0 - eps) + eps
            z_x = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device, generator=generator)
            z_state = torch.randn(
                state0.shape,
                dtype=state0.dtype,
                device=state0.device,
                generator=generator,
            )
        z_x = mask_x(z_x, flags)
        e_mask = eigen_mask_from_flags(flags, eigen_mask_mode)
        z_state = z_state * e_mask

        mean_x, std_x = sde_x.marginal_x(x0, t)
        xt = mask_x(mean_x + std_x[:, None, None] * z_x, flags)
        mean_state, std_state = sde_lam.marginal_spectrum(state0, t)
        state_t = (mean_state * e_mask + std_state[:, None] * z_state) * e_mask
        if score_enabled:
            state_t = state_t.detach().requires_grad_(True)
        lam_t = _spectral_state_to_operator_eigenvalues(
            state_t,
            flags,
            spectral_operator="combinatorial_laplacian",
            spectral_transform=spectral_transform,
        )
        operator_t = reconstruct_adjacency(u, lam_t)
        adj_t = _operator_to_adjacency_state(operator_t, flags, "combinatorial_laplacian")

        pred_x = model_x(xt, adj_t, flags, u, state_t)
        if not _is_joint_graphlet_score(model_lam):
            raise TypeError("joint log-gap graphlet training requires a graphlet-capable spectrum model")

        clean_target_vector = _clean_structure_target_vector(
            graphlet_target,
            graphlet_mass,
            clustering_target,
            orbit_histogram_target,
            orbit_log_total_target,
            structure_cfg=graphlet_cfg,
        )
        if score_enabled:
            outputs = model_lam.forward_all(
                xt,
                adj_t,
                flags,
                u,
                state_t,
                time=t,
                structure_target=clean_target_vector,
            )
        else:
            outputs = model_lam.forward_all(xt, adj_t, flags, u, state_t)
        pred_state = outputs["spectrum"]
        loss_x = 0.5 * (pred_x - z_x).square().reshape(b, -1).sum(dim=-1).mean()
        loss_state = 0.5 * (
            (((pred_state - z_state) * e_mask).square().reshape(b, -1).sum(dim=-1))
        ).mean()
        loss_graphlet, metrics = _structure_auxiliary_loss(
            model_lam,
            outputs,
            graphlet_target,
            graphlet_mass,
            clustering_target,
            orbit_histogram_target,
            orbit_log_total_target,
            structure_cfg=graphlet_cfg,
        )

        loss_structure_score = loss_state.detach() * 0.0
        if score_enabled:
            if "structure_score" not in outputs:
                raise RuntimeError("Structural score is enabled but the model did not return structure_score")
            structural_logp = _structure_log_likelihood_per_graph(
                model_lam,
                outputs,
                graphlet_target,
                graphlet_mass,
                clustering_target,
                orbit_histogram_target,
                orbit_log_total_target,
                structure_cfg=graphlet_cfg,
                orbit_log_total_sigma=float(score_cfg.get("orbit_log_total_sigma", 0.25)),
            )
            teacher_score = torch.autograd.grad(
                structural_logp.sum(),
                state_t,
                retain_graph=True,
                create_graph=False,
                allow_unused=False,
            )[0].detach()
            teacher_score = teacher_score * e_mask
            teacher_score = _clip_vector_norm(
                teacher_score,
                float(score_cfg.get("teacher_clip_norm", 5.0)),
            )
            pred_structure_score = outputs["structure_score"] * e_mask
            loss_structure_score = 0.5 * (
                (pred_structure_score - teacher_score).square().reshape(b, -1).sum(dim=-1)
            ).mean()
            with torch.no_grad():
                teacher_norm = teacher_score.reshape(b, -1).norm(dim=-1)
                pred_norm = pred_structure_score.reshape(b, -1).norm(dim=-1)
                dot = (teacher_score * pred_structure_score).reshape(b, -1).sum(dim=-1)
                cosine = dot / (teacher_norm * pred_norm).clamp_min(1.0e-12)
                metrics.update(
                    {
                        "structure_score_loss": float(loss_structure_score.detach().item()),
                        "structure_score_teacher_norm": float(teacher_norm.mean().item()),
                        "structure_score_pred_norm": float(pred_norm.mean().item()),
                        "structure_score_cosine": float(cosine.mean().item()),
                        "structure_log_likelihood": float(structural_logp.detach().mean().item()),
                    }
                )

    return loss_x, loss_state, loss_graphlet, loss_structure_score, metrics


class _EMA:
    def __init__(self, module: nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.shadow = {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}

    @torch.no_grad()
    def update(self, module: nn.Module) -> None:
        for key, value in module.state_dict().items():
            current = value.detach().cpu()
            if current.dtype.is_floating_point:
                self.shadow[key].mul_(self.decay).add_(current, alpha=1.0 - self.decay)
            else:
                self.shadow[key].copy_(current)


def _model_config(options: Mapping[str, Any], max_nodes: int) -> dict[str, Any]:
    raw = dict(options["model"])
    max_feat_num = raw.get("max_feat_num")
    if max_feat_num is None:
        max_feat_num = max_nodes
    legacy_backbone = str(raw.get("backbone", "dense_gcn")).lower()
    node_backbone = str(raw.get("node_backbone", legacy_backbone)).lower()
    spectrum_backbone = str(raw.get("spectrum_backbone", legacy_backbone)).lower()
    return {
        "max_nodes": int(max_nodes),
        "max_feat_num": int(max_feat_num),
        "hidden_dim": int(raw.get("hidden_dim", 32)),
        "depth": int(raw.get("depth", 3)),
        # Retain `backbone` in checkpoints for backwards compatibility.
        "backbone": legacy_backbone,
        "node_backbone": node_backbone,
        "spectrum_backbone": spectrum_backbone,
        "ppgn_hidden_dim": int(raw.get("ppgn_hidden_dim", 64)),
        "ppgn_depth": int(raw.get("ppgn_depth", 4)),
        "ppgn_residual_scale": float(raw.get("ppgn_residual_scale", 0.1)),
        "ppgn_norm_eps": float(raw.get("ppgn_norm_eps", 1.0e-5)),
        "ppgn_input_clip": float(raw.get("ppgn_input_clip", 10.0)),
        "structural_features": copy.deepcopy(dict(options.get("structural_features", {}) or {})),
        "structure_score": copy.deepcopy(_resolved_structure_score_config(options)),
    }


def _build_models(
    cfg: Mapping[str, Any],
    device: torch.device,
    *,
    graphlet_slices: tuple[tuple[int, int], ...] | None = None,
    summary_meta: Mapping[str, Any] | None = None,
) -> tuple[nn.Module, nn.Module]:
    legacy_backbone = str(cfg.get("backbone", "dense_gcn")).lower()
    node_backbone = str(cfg.get("node_backbone", legacy_backbone)).lower()
    spectrum_backbone = str(cfg.get("spectrum_backbone", legacy_backbone)).lower()
    structural_features = copy.deepcopy(dict(cfg.get("structural_features", {}) or {}))
    structure_score_config = copy.deepcopy(dict(cfg.get("structure_score", {}) or {}))
    ppgn_kwargs = {
        "ppgn_hidden_dim": int(cfg.get("ppgn_hidden_dim", 64)),
        "ppgn_depth": int(cfg.get("ppgn_depth", 4)),
        "ppgn_residual_scale": float(cfg.get("ppgn_residual_scale", 0.1)),
        "ppgn_norm_eps": float(cfg.get("ppgn_norm_eps", 1.0e-5)),
        "ppgn_input_clip": float(cfg.get("ppgn_input_clip", 10.0)),
    }
    if node_backbone == "ppgn":
        mx = GSDMNodePPGNScore(
            max_feat_num=int(cfg["max_feat_num"]),
            max_nodes=int(cfg["max_nodes"]),
            structural_features=structural_features,
            **ppgn_kwargs,
        ).to(device)
    elif node_backbone == "dense_gcn":
        mx = GSDMNodeScore(
            max_feat_num=int(cfg["max_feat_num"]),
            hidden_dim=int(cfg["hidden_dim"]),
            depth=int(cfg["depth"]),
            structural_features=structural_features,
        ).to(device)
    else:
        raise ValueError(f"Unsupported node_backbone={node_backbone!r}")

    spectrum_kwargs = {
        "max_feat_num": int(cfg["max_feat_num"]),
        "max_nodes": int(cfg["max_nodes"]),
        "hidden_dim": int(cfg["hidden_dim"]),
        "depth": int(cfg["depth"]),
    }
    if graphlet_slices:
        summary_meta = dict(summary_meta or {})
        summary_head_kwargs = {
            "clustering_bins": (
                int(summary_meta.get("clustering_bins", 0))
                if bool(summary_meta.get("clustering_enabled", False)) else 0
            ),
            "orbit_width": (
                int(summary_meta.get("orbit_width", 0))
                if bool(summary_meta.get("orbit_enabled", False)) else 0
            ),
        }
        if spectrum_backbone == "ppgn":
            ml = GSDMSpectrumGraphletPPGNScore(
                **spectrum_kwargs,
                graphlet_slices=graphlet_slices,
                structural_features=structural_features,
                structure_score_config=structure_score_config,
                **summary_head_kwargs,
                **ppgn_kwargs,
            ).to(device)
        elif spectrum_backbone == "dense_gcn":
            ml = GSDMSpectrumGraphletScore(
                **spectrum_kwargs,
                graphlet_slices=graphlet_slices,
                structural_features=structural_features,
                **summary_head_kwargs,
            ).to(device)
        else:
            raise ValueError(f"Unsupported spectrum_backbone={spectrum_backbone!r}")
    else:
        if spectrum_backbone != "dense_gcn":
            raise ValueError("PPGN spectrum backbone currently requires the joint graphlet head")
        ml = GSDMSpectrumScore(**spectrum_kwargs).to(device)
    return mx, ml


def _make_sdes(options: Mapping[str, Any], device: torch.device) -> tuple[VPSDE, VPSDE]:
    sx = options["sde"]["x"]
    sl = options["sde"]["spectrum"]
    return (
        VPSDE(beta_min=float(sx["beta_min"]), beta_max=float(sx["beta_max"]), num_scales=int(sx["num_scales"]), device=device),
        VPSDE(beta_min=float(sl["beta_min"]), beta_max=float(sl["beta_max"]), num_scales=int(sl["num_scales"]), device=device),
    )



def _degree_targets_to_tensors(
    targets: Mapping[str, np.ndarray], device: torch.device,
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, value in targets.items():
        tensor = torch.as_tensor(value, device=device)
        if key in {"num_nodes", "num_nodes_count", "num_edges_count"}:
            tensor = tensor.long()
        else:
            tensor = tensor.float()
        out[key] = tensor
    return out


def _build_degree_prior_training(
    train_graphs: list[nx.Graph],
    cfg: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Build the auxiliary DH-VAE used by the Stage-1 ablation.

    It is deliberately independent of the GSDM networks.  "Joint" here means
    co-trained in the same managed training run/epoch schedule and saved in the
    same checkpoint, not a shared latent or cross-conditioned objective yet.
    """

    vectorizer = DegreeVectorizer.fit(
        train_graphs,
        max_degree=None,
        require_connected=bool(cfg.get("require_connected", True)),
    )
    x_np, targets_np = vectorizer.to_training_arrays(train_graphs)
    inputs = torch.as_tensor(x_np, dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(inputs, torch.arange(inputs.shape[0])),
        batch_size=int(cfg.get("batch_size", 64)),
        shuffle=True,
        drop_last=False,
    )
    targets = _degree_targets_to_tensors(targets_np, device)
    model = build_degree_vae(
        vectorizer,
        latent_dim=int(cfg.get("latent_dim", 32)),
        hidden_dim=int(cfg.get("hidden_dim", 128)),
        size_condition_dim=int(cfg.get("size_condition_dim", 16)),
        edge_condition_dim=int(cfg.get("edge_condition_dim", 16)),
        use_edge_count_conditioning=bool(cfg.get("use_edge_count_conditioning", True)),
        prior_condition_on_edges=bool(cfg.get("prior_condition_on_edges", True)),
        prior_type=str(cfg.get("prior_type", "conditional_gmm")),
        prior_components=int(cfg.get("prior_components", 4)),
        prior_hidden_dim=int(cfg.get("prior_hidden_dim", cfg.get("hidden_dim", 128))),
        prior_logvar_min=float(cfg.get("prior_logvar_min", -6.0)),
        prior_logvar_max=float(cfg.get("prior_logvar_max", 4.0)),
        num_layers=int(cfg.get("num_layers", 2)),
        dropout=float(cfg.get("dropout", 0.0)),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg.get("learning_rate", 1.0e-3)),
        weight_decay=float(cfg.get("weight_decay", 1.0e-5)),
    )
    weights = {
        "num_nodes": float(cfg.get("node_count_loss_weight", 1.0)),
        "num_edges": float(cfg.get("edge_count_loss_weight", 2.0)),
        "degree": float(cfg.get("degree_histogram_loss_weight", 5.0)),
        "degree_moment": float(cfg.get("degree_moment_loss_weight", 0.25)),
        "aggregate_prior_moment": float(cfg.get("aggregate_prior_moment_loss_weight", 0.05)),
        "prior_distribution": float(cfg.get("prior_distribution_loss_weight", 1.0)),
    }
    return {
        "model": model,
        "vectorizer": vectorizer,
        "loader": loader,
        "targets": targets,
        "optimizer": optimizer,
        "weights": weights,
    }


def _train_degree_prior_epoch(
    bundle: Mapping[str, Any],
    cfg: Mapping[str, Any],
    *,
    epoch: int,
    device: torch.device,
) -> dict[str, float]:
    model: DegreeHistogramVAE = bundle["model"]
    loader: DataLoader = bundle["loader"]
    all_targets: Mapping[str, torch.Tensor] = bundle["targets"]
    optimizer: torch.optim.Optimizer = bundle["optimizer"]
    weights: Mapping[str, float] = bundle["weights"]

    model.train()
    rows: dict[str, list[float]] = {}
    beta = float(cfg.get("kl_loss_weight", 0.005))
    warmup = int(cfg.get("kl_warmup_epochs", 0))
    effective_beta = beta * (min(float(epoch) / warmup, 1.0) if warmup else 1.0)
    for batch_inputs, batch_indices in loader:
        batch_inputs = batch_inputs.to(device)
        indices = batch_indices.to(device)
        targets = {key: value[indices] for key, value in all_targets.items()}
        outputs, mu, logvar = model(
            batch_inputs,
            targets["num_nodes_count"],
            targets.get("num_edges_count"),
        )
        prior_outputs = None
        if float(weights.get("prior_distribution", 0.0)) > 0.0:
            prior_z = model.sample_prior(
                targets["num_nodes_count"],
                edge_counts=targets.get("num_edges_count"),
                prior_mode="model",
            )
            prior_outputs = model.decode(
                prior_z,
                targets["num_nodes_count"],
                targets.get("num_edges_count"),
            )
        loss, metrics = degree_vae_loss(
            outputs,
            targets,
            mu,
            logvar,
            beta=effective_beta,
            weights=dict(weights),
            prior_outputs=prior_outputs,
            prior_distribution_sigma=float(cfg.get("prior_distribution_kernel_sigma", 0.2)),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        for key, value in metrics.items():
            rows.setdefault(key, []).append(float(value))
    result = {key: float(np.mean(values)) for key, values in rows.items()}
    result["beta"] = float(effective_beta)
    return result


def _degree_prior_payload(bundle: Mapping[str, Any], cfg: Mapping[str, Any]) -> dict[str, Any]:
    model: DegreeHistogramVAE = bundle["model"]
    vectorizer: DegreeVectorizer = bundle["vectorizer"]
    return {
        "enabled": True,
        "role": "auxiliary_unconstrained_degree_sequence_prior",
        "used_for_gsdm_generation": False,
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "model_config": model.model_config(),
        "vectorizer": copy.deepcopy(vectorizer.__dict__),
        "sampling_config": {
            key: copy.deepcopy(cfg.get(key))
            for key in (
                "sample_num_nodes",
                "sample_num_edges",
                "exact_degree_sum_conditioning",
                "max_resample",
                "model_resample_attempts",
                "parity_conditioned",
                "max_parity_resample",
                "fallback",
                "postprocess_policy",
            )
        },
    }


def _load_embedded_degree_prior(
    state: Mapping[str, Any], device: torch.device,
) -> tuple[DegreeHistogramVAE, DegreeVectorizer, Mapping[str, Any]]:
    payload = state.get("degree_prior")
    if not isinstance(payload, Mapping) or not bool(payload.get("enabled", False)):
        raise RuntimeError("Checkpoint does not contain the jointly trained auxiliary DH-VAE")
    model_cfg = dict(payload["model_config"])
    architecture_version = int(model_cfg.pop("architecture_version", 4))
    if architecture_version != 4:
        raise RuntimeError(
            f"Expected embedded DH-VAE architecture version 4, found {architecture_version}"
        )
    vectorizer = DegreeVectorizer(**dict(payload["vectorizer"]))
    model = DegreeHistogramVAE(**model_cfg).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, vectorizer, payload


def training_artifacts(wrapper, request: TrainRequest) -> TrainingArtifacts:
    layout = request.run.layout
    return TrainingArtifacts(
        run_dir=layout.run_dir,
        checkpoint_path=layout.checkpoints_dir / "gdsm_simple.pt",
        manifest_path=layout.training_manifest_path,
        log_path=layout.training_log_path,
    )


def train(wrapper, request: TrainRequest, options: Mapping[str, Any]) -> TrainingArtifacts:
    validate_options(options)
    if request.run.dataset_id in {"qm9", "zinc", "attributed"}:
        raise ValueError("The first vanilla_gsdm implementation is generic-graph only")
    layout = request.run.layout
    artifacts = training_artifacts(wrapper, request)
    fingerprint = request.dataset.fingerprint()
    if layout.training_manifest_path.is_file() and not request.overwrite:
        old = json.loads(layout.training_manifest_path.read_text(encoding="utf-8"))
        if (
            old.get("dataset", {}).get("fingerprint") == fingerprint
            and old.get("options") == _jsonable(options)
            and artifacts.checkpoint_path.is_file()
        ):
            return artifacts
        raise ArtifactCollisionError("Existing vanilla GSDM run differs; choose a new run-id or --overwrite")

    ArtifactLayout.require_available(layout.train_dir, overwrite=request.overwrite)
    _seed_everything(request.run.train_seed)
    device = _resolve_device(options.get("runtime", {}))
    variant = str(options.get("variant", "vanilla_gsdm")).lower()
    spectral_operator = spectral_operator_for_variant(variant)
    degree_cfg = dict(options.get("degree_prior", {}) or {})
    degree_prior_enabled = bool(degree_cfg.get("enabled", False))
    train_graphs = _graphs(request.dataset.split_paths["train"])
    val_graphs = _graphs(request.dataset.split_paths["val"])
    if spectral_operator == "combinatorial_laplacian":
        disconnected = [
            (split, index)
            for split, graphs in (("train", train_graphs), ("val", val_graphs))
            for index, graph in enumerate(graphs)
            if not nx.is_connected(graph)
        ]
        if disconnected:
            split, index = disconnected[0]
            raise ValueError(
                "vanilla_laplacian_gsdm Stage-2 currently requires connected "
                f"graphs so the fixed zero Laplacian mode is unique; found disconnected {split}[{index}]"
            )
    configured_max = options["model"].get("max_nodes")
    max_nodes = int(configured_max or max(g.number_of_nodes() for g in train_graphs))
    if max(g.number_of_nodes() for g in val_graphs) > max_nodes:
        raise ValueError("Validation graph exceeds model.max_nodes")
    model_cfg = _model_config(options, max_nodes)
    train_data = _padded_dataset(
        train_graphs,
        max_nodes=max_nodes,
        max_feat_num=model_cfg["max_feat_num"],
        spectral_operator=spectral_operator,
    )
    val_data = _padded_dataset(
        val_graphs,
        max_nodes=max_nodes,
        max_feat_num=model_cfg["max_feat_num"],
        spectral_operator=spectral_operator,
    )
    graphlet_meta: dict[str, Any] | None = None
    spectral_transform: dict[str, Any] = {"kind": "direct_eigenvalues"}
    joint_graphlet_cfg = _resolved_structure_summary_config(options)
    structure_score_cfg = _resolved_structure_score_config(options)
    if variant in LAPLACIAN_LOGGAP_ALL_VARIANTS:
        sde_cfg = dict(options.get("sde", {}) or {})
        stats = _fit_laplacian_log_gap_stats(
            train_data[5],
            train_data[2],
            epsilon=float(sde_cfg.get("log_gap_epsilon", 1.0e-6)),
            min_std=float(sde_cfg.get("log_gap_min_std", 1.0e-3)),
        )
        spectral_transform = {
            "kind": "laplacian_log_gap",
            "epsilon": float(sde_cfg.get("log_gap_epsilon", 1.0e-6)),
            "exp_clip": float(sde_cfg.get("log_gap_exp_clip", 20.0)),
            "mean": stats["mean"],
            "std": stats["std"],
            "count": stats["count"],
        }
    if variant in LAPLACIAN_LOGGAP_GRAPHLET_AUX_VARIANTS:
        train_targets, graphlet_meta = _joint_structure_targets(
            train_graphs, joint_graphlet_cfg
        )
        val_targets, val_meta = _joint_structure_targets(
            val_graphs, joint_graphlet_cfg
        )
        if val_meta["graphlet_slices"] != graphlet_meta["graphlet_slices"]:
            raise AssertionError("Train/validation graphlet bases differ")
        for key in ("clustering_bins", "clustering_enabled", "orbit_width", "orbit_enabled"):
            if val_meta.get(key) != graphlet_meta.get(key):
                raise AssertionError(f"Train/validation structural-summary metadata differs for {key}")
        train_data = (*train_data, *train_targets)
        val_data = (*val_data, *val_targets)
    train_loader = DataLoader(
        TensorDataset(*train_data),
        batch_size=int(options["train"]["batch_size"]),
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(*val_data),
        batch_size=int(options["train"]["batch_size"]),
        shuffle=False,
    )
    graphlet_slices = (
        tuple(tuple(int(v) for v in pair) for pair in graphlet_meta["graphlet_slices"])
        if graphlet_meta is not None else None
    )
    model_x, model_lam = _build_models(
        model_cfg,
        device,
        graphlet_slices=graphlet_slices,
        summary_meta=graphlet_meta,
    )
    sde_x, sde_lam = _make_sdes(options, device)
    degree_bundle = None
    degree_fork_devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda" else []
    )
    if degree_prior_enabled:
        # Keep the auxiliary prior's stochasticity from perturbing the vanilla
        # GSDM training trajectory.  This makes Stage 1 a clean observational
        # ablation: the GSDM networks see exactly the same RNG stream as the
        # vanilla run for the same seed/config.
        with torch.random.fork_rng(devices=degree_fork_devices, enabled=True):
            torch.manual_seed(request.run.train_seed + 700001)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(request.run.train_seed + 700001)
            degree_bundle = _build_degree_prior_training(train_graphs, degree_cfg, device)
    train_cfg = options["train"]
    node_lr = float(train_cfg.get("node_lr", train_cfg["lr"]))
    spectrum_lr = float(train_cfg.get("spectrum_lr", train_cfg["lr"]))
    optimizers = [
        torch.optim.Adam(model_x.parameters(), lr=node_lr, weight_decay=float(train_cfg.get("weight_decay", 0.0))),
        torch.optim.Adam(model_lam.parameters(), lr=spectrum_lr, weight_decay=float(train_cfg.get("weight_decay", 0.0))),
    ]
    schedulers = [
        torch.optim.lr_scheduler.ExponentialLR(opt, gamma=float(train_cfg.get("lr_decay", 1.0)))
        for opt in optimizers
    ]
    ema_x = _EMA(model_x, float(train_cfg.get("ema", 0.999)))
    ema_lam = _EMA(model_lam, float(train_cfg.get("ema", 0.999)))
    epochs = int(train_cfg["epochs"])
    if epochs <= 0:
        raise ValueError("train.epochs must be positive")
    val_every = max(1, int(train_cfg.get("validation_every", 1)))
    log_every = max(1, int(train_cfg.get("log_every", 10)))
    eps = float(options["sde"].get("eps", 1.0e-5))
    eigen_mask_mode = str(options["sde"].get("eigen_mask", "official_extremes"))
    history: list[dict[str, Any]] = []
    start = time.monotonic()
    layout.train_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".gdsm_simple_vanilla_train_", dir=layout.train_dir.parent))
    log_path = staging / "train.log"
    try:
        with log_path.open("w", encoding="utf-8") as log:
            for epoch in range(1, epochs + 1):
                model_x.train(); model_lam.train()
                total_x = total_l = total_g = total_ss = total_n = 0.0
                train_graphlet_metrics: list[dict[str, float]] = []
                for batch_cpu in train_loader:
                    batch = tuple(v.to(device) for v in batch_cpu)
                    for opt in optimizers:
                        opt.zero_grad(set_to_none=True)
                    if variant in LAPLACIAN_LOGGAP_GRAPHLET_AUX_VARIANTS:
                        if not _is_joint_graphlet_score(model_lam):
                            raise AssertionError("log-gap graphlet variant requires the joint spectrum/graphlet model")
                        lx, ll, lg, lss, gmetrics = _loss_batch_loggap_graphlet(
                            model_x, model_lam, batch,
                            sde_x=sde_x, sde_lam=sde_lam, eps=eps,
                            eigen_mask_mode=eigen_mask_mode,
                            spectral_transform=spectral_transform,
                            graphlet_cfg=joint_graphlet_cfg,
                            structure_score_cfg=structure_score_cfg,
                        )
                        loss = (
                            lx + ll
                            + float(joint_graphlet_cfg.get("loss_weight", 0.10)) * lg
                            + float(structure_score_cfg.get("loss_weight", 0.05)) * lss
                        )
                        train_graphlet_metrics.append(gmetrics)
                    elif variant in LAPLACIAN_LOGGAP_VARIANTS:
                        lx, ll = _loss_batch_loggap(
                            model_x,
                            model_lam,
                            batch,
                            sde_x=sde_x,
                            sde_lam=sde_lam,
                            eps=eps,
                            eigen_mask_mode=eigen_mask_mode,
                            spectral_transform=spectral_transform,
                        )
                        lg = lx.detach() * 0.0
                        lss = lx.detach() * 0.0
                        loss = lx + ll
                    else:
                        lx, ll = _loss_batch(
                            model_x, model_lam, batch,
                            sde_x=sde_x, sde_lam=sde_lam, eps=eps,
                            eigen_mask_mode=eigen_mask_mode,
                            spectral_operator=spectral_operator,
                        )
                        lg = lx.detach() * 0.0
                        lss = lx.detach() * 0.0
                        loss = lx + ll
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite vanilla GSDM training loss")
                    loss.backward()
                    grad_x = torch.nn.utils.clip_grad_norm_(model_x.parameters(), float(train_cfg.get("grad_norm", 1.0)))
                    grad_lam = torch.nn.utils.clip_grad_norm_(model_lam.parameters(), float(train_cfg.get("grad_norm", 1.0)))
                    if not torch.isfinite(torch.as_tensor(grad_x)):
                        raise FloatingPointError("Non-finite node-score gradient norm")
                    if not torch.isfinite(torch.as_tensor(grad_lam)):
                        raise FloatingPointError("Non-finite spectrum-score gradient norm")
                    for opt in optimizers:
                        opt.step()
                    for name, module in (("node", model_x), ("spectrum", model_lam)):
                        for parameter in module.parameters():
                            if parameter.dtype.is_floating_point and not torch.isfinite(parameter).all():
                                raise FloatingPointError(f"Non-finite parameter after optimizer step in {name} model")
                    ema_x.update(model_x); ema_lam.update(model_lam)
                    b = batch[0].size(0)
                    total_x += float(lx.item()) * b
                    total_l += float(ll.item()) * b
                    total_g += float(lg.item()) * b
                    total_ss += float(lss.item()) * b
                    total_n += b
                degree_metrics = None
                if degree_bundle is not None:
                    with torch.random.fork_rng(devices=degree_fork_devices, enabled=True):
                        degree_seed = request.run.train_seed + 700001 + epoch
                        torch.manual_seed(degree_seed)
                        if device.type == "cuda":
                            torch.cuda.manual_seed_all(degree_seed)
                        degree_metrics = _train_degree_prior_epoch(
                            degree_bundle, degree_cfg, epoch=epoch, device=device
                        )
                if bool(train_cfg.get("lr_schedule", True)):
                    for scheduler in schedulers:
                        scheduler.step()
                record: dict[str, Any] = {
                    "epoch": epoch,
                    "train_node_loss": total_x / max(total_n, 1),
                    "train_spectrum_loss": total_l / max(total_n, 1),
                    "node_lr": float(optimizers[0].param_groups[0]["lr"]),
                    "spectrum_lr": float(optimizers[1].param_groups[0]["lr"]),
                }
                if variant in LAPLACIAN_LOGGAP_GRAPHLET_AUX_VARIANTS:
                    record["train_graphlet_summary_loss"] = total_g / max(total_n, 1)
                    record["train_structure_summary_loss"] = record["train_graphlet_summary_loss"]
                    record["train_structure_score_loss"] = total_ss / max(total_n, 1)
                    if train_graphlet_metrics:
                        for key in sorted(train_graphlet_metrics[0]):
                            record[f"train_{key}"] = float(np.mean([row[key] for row in train_graphlet_metrics]))
                    record["train_loss"] = (
                        record["train_node_loss"] + record["train_spectrum_loss"]
                        + float(joint_graphlet_cfg.get("loss_weight", 0.10)) * record["train_graphlet_summary_loss"]
                        + float(structure_score_cfg.get("loss_weight", 0.05)) * record["train_structure_score_loss"]
                    )
                else:
                    record["train_loss"] = record["train_node_loss"] + record["train_spectrum_loss"]
                if degree_metrics is not None:
                    record["train_degree_prior_loss"] = float(degree_metrics["loss"])
                    record["train_degree_prior_degree_loss"] = float(degree_metrics["degree_loss"])
                    record["train_degree_prior_kl_loss"] = float(degree_metrics["kl_loss"])
                    record["train_degree_prior_beta"] = float(degree_metrics["beta"])
                    record["train_joint_loss"] = record["train_loss"] + record["train_degree_prior_loss"]
                if epoch == 1 or epoch % val_every == 0 or epoch == epochs:
                    model_x.eval(); model_lam.eval()
                    vx = vl = vg = vss = vn = 0.0
                    val_graphlet_metrics: list[dict[str, float]] = []
                    gen = torch.Generator(device=device).manual_seed(request.run.train_seed + 100003 + epoch)
                    with torch.no_grad():
                        for batch_cpu in val_loader:
                            batch = tuple(v.to(device) for v in batch_cpu)
                            if variant in LAPLACIAN_LOGGAP_GRAPHLET_AUX_VARIANTS:
                                if not _is_joint_graphlet_score(model_lam):
                                    raise AssertionError("log-gap graphlet variant requires the joint spectrum/graphlet model")
                                lx, ll, lg, lss, gmetrics = _loss_batch_loggap_graphlet(
                                    model_x, model_lam, batch,
                                    sde_x=sde_x, sde_lam=sde_lam, eps=eps,
                                    eigen_mask_mode=eigen_mask_mode,
                                    spectral_transform=spectral_transform,
                                    graphlet_cfg=joint_graphlet_cfg,
                                    structure_score_cfg=structure_score_cfg,
                                    generator=gen,
                                )
                                val_graphlet_metrics.append(gmetrics)
                            elif variant in LAPLACIAN_LOGGAP_VARIANTS:
                                lx, ll = _loss_batch_loggap(
                                    model_x,
                                    model_lam,
                                    batch,
                                    sde_x=sde_x,
                                    sde_lam=sde_lam,
                                    eps=eps,
                                    eigen_mask_mode=eigen_mask_mode,
                                    spectral_transform=spectral_transform,
                                    generator=gen,
                                )
                                lg = lx.detach() * 0.0
                                lss = lx.detach() * 0.0
                            else:
                                lx, ll = _loss_batch(
                                    model_x, model_lam, batch,
                                    sde_x=sde_x, sde_lam=sde_lam, eps=eps,
                                    eigen_mask_mode=eigen_mask_mode,
                                    spectral_operator=spectral_operator,
                                    generator=gen,
                                )
                                lg = lx.detach() * 0.0
                                lss = lx.detach() * 0.0
                            b = batch[0].size(0)
                            vx += float(lx.item()) * b
                            vl += float(ll.item()) * b
                            vg += float(lg.item()) * b
                            vss += float(lss.item()) * b
                            vn += b
                    record["val_node_loss"] = vx / max(vn, 1)
                    record["val_spectrum_loss"] = vl / max(vn, 1)
                    if variant in LAPLACIAN_LOGGAP_GRAPHLET_AUX_VARIANTS:
                        record["val_graphlet_summary_loss"] = vg / max(vn, 1)
                        record["val_structure_summary_loss"] = record["val_graphlet_summary_loss"]
                        record["val_structure_score_loss"] = vss / max(vn, 1)
                        if val_graphlet_metrics:
                            for key in sorted(val_graphlet_metrics[0]):
                                record[f"val_{key}"] = float(np.mean([row[key] for row in val_graphlet_metrics]))
                        record["val_loss"] = (
                            record["val_node_loss"] + record["val_spectrum_loss"]
                            + float(joint_graphlet_cfg.get("loss_weight", 0.10)) * record["val_graphlet_summary_loss"]
                            + float(structure_score_cfg.get("loss_weight", 0.05)) * record["val_structure_score_loss"]
                        )
                    else:
                        record["val_loss"] = record["val_node_loss"] + record["val_spectrum_loss"]
                history.append(record)
                if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
                    line = (
                        f"Vanilla-GSDM epoch {epoch}/{epochs} "
                        f"train={record['train_loss']:.6f} "
                        f"node={record['train_node_loss']:.6f} spectrum={record['train_spectrum_loss']:.6f}"
                    )
                    if "train_degree_prior_loss" in record:
                        line += (
                            f" dhvae={record['train_degree_prior_loss']:.6f}"
                            f" dh_degree={record['train_degree_prior_degree_loss']:.6f}"
                        )
                    if "val_loss" in record:
                        line += f" val={record['val_loss']:.6f}"
                    print(line, flush=True)
                    log.write(line + "\n"); log.flush()

        graphlet_payload: dict[str, Any] = {"enabled": False}
        graphlet_report: dict[str, Any] | None = None
        if variant in GRAPHLET_REFINEMENT_VARIANTS:
            from grapher.models.gdsm_simple.graphlet_stage3 import train_graphlet_predictor
            graphlet_cfg = dict(options.get("graphlet_refinement", {}) or {})
            with torch.random.fork_rng(devices=degree_fork_devices, enabled=True):
                predictor_seed = request.run.train_seed + 800001
                torch.manual_seed(predictor_seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(predictor_seed)
                graphlet_payload, graphlet_report = train_graphlet_predictor(
                    train_graphs,
                    val_graphs,
                    config=graphlet_cfg,
                    device=device,
                    seed=request.run.train_seed,
                )
            with log_path.open("a", encoding="utf-8") as log:
                log.write(
                    "Stage3 graphlet predictor "
                    f"best_epoch={graphlet_report['best_epoch']} "
                    f"best_val_loss={graphlet_report['best_val_loss']:.6f} "
                    f"train_examples={graphlet_report['num_train_examples']} "
                    f"val_examples={graphlet_report['num_val_examples']}\n"
                )

        checkpoint_dir = staging / "checkpoints"
        checkpoint_dir.mkdir(parents=True)
        checkpoint_path = checkpoint_dir / "gdsm_simple.pt"
        checkpoint = {
            "format": CHECKPOINT_FORMAT,
            "variant": variant,
            "spectral_operator": spectral_operator,
            "spectral_transform": _jsonable(spectral_transform),
            "structure_score": copy.deepcopy(structure_score_cfg),
            "structure_summary": (
                {
                    "enabled": True,
                    "training_config": copy.deepcopy(joint_graphlet_cfg),
                    **(graphlet_meta or {}),
                }
                if variant in LAPLACIAN_LOGGAP_GRAPHLET_AUX_VARIANTS
                else {"enabled": False}
            ),
            # Backwards-compatible alias consumed by older graphlet-only tools.
            "graphlet_summary": (
                {
                    "enabled": True,
                    "training_config": copy.deepcopy(joint_graphlet_cfg),
                    **(graphlet_meta or {}),
                }
                if variant in LAPLACIAN_LOGGAP_GRAPHLET_AUX_VARIANTS
                else {"enabled": False}
            ),
            "model_x_state": {k: v.detach().cpu() for k, v in model_x.state_dict().items()},
            "model_spectrum_state": {k: v.detach().cpu() for k, v in model_lam.state_dict().items()},
            "ema_x_state": ema_x.shadow,
            "ema_spectrum_state": ema_lam.shadow,
            "model_config": model_cfg,
            "sde": copy.deepcopy(dict(options["sde"])),
            "sample": copy.deepcopy(dict(options["sample"])),
            "max_nodes": max_nodes,
            "basis_adjacencies": train_data[1].numpy(),
            "basis_num_nodes": train_data[3].numpy().astype(np.int64),
            "basis_source": "training_split_only",
            "node_feature_init": (
                "diffused_degree_one_hot_plus_random_walk_arrival_conditioning"
                if bool(model_cfg.get("structural_features", {}).get("enabled", False))
                and bool(model_cfg.get("structural_features", {}).get("random_walk", {}).get("enabled", False))
                else "degree_one_hot"
            ),
            "edge_feature_init": (
                "truncated_shortest_path_one_hot_from_binarized_current_state"
                if bool(model_cfg.get("structural_features", {}).get("enabled", False))
                and bool(model_cfg.get("structural_features", {}).get("shortest_path", {}).get("enabled", False))
                else "none"
            ),
            "history": history,
            "train_seed": request.run.train_seed,
            "degree_prior": (
                _degree_prior_payload(degree_bundle, degree_cfg)
                if degree_bundle is not None
                else {"enabled": False, "used_for_gsdm_generation": False}
            ),
            "graphlet_refinement": graphlet_payload,
        }
        torch.save(checkpoint, checkpoint_path)
        resolved = copy.deepcopy(dict(options))
        resolved["model"] = dict(resolved["model"])
        resolved["model"]["max_nodes"] = max_nodes
        resolved["model"]["max_feat_num"] = model_cfg["max_feat_num"]
        (staging / "resolved_config.yaml").write_text(
            yaml.safe_dump({wrapper.model_id: resolved}, sort_keys=False), encoding="utf-8"
        )
        manifest = {
            "format": TRAINING_FORMAT,
            "model_id": wrapper.model_id,
            "variant": variant,
            "run_id": request.run.run_id,
            "train_seed": request.run.train_seed,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": time.monotonic() - start,
            "dataset": {
                "benchmark_id": request.dataset.benchmark_id,
                "serialized_id": request.dataset.serialized_id,
                "fingerprint": fingerprint,
                "split_sha256": {k: _sha256(v) for k, v in request.dataset.split_paths.items()},
            },
            "options": _jsonable(options),
            "checkpoint": {"path": "checkpoints/gdsm_simple.pt", "sha256": _sha256(checkpoint_path)},
            "checkpoint_selection": {"kind": "final_configured_epoch", "epoch": epochs},
            "reference_contract": {
                "forward_state": (
                    "degree_features_and_normalized_log_successive_laplacian_eigenvalue_gaps"
                    if variant in LAPLACIAN_LOGGAP_ALL_VARIANTS
                    else (
                        "degree_features_and_combinatorial_laplacian_eigenvalues"
                        if spectral_operator == "combinatorial_laplacian"
                        else "degree_features_and_adjacency_eigenvalues"
                    )
                ),
                "spectral_operator": spectral_operator,
                "spectral_parameterization": str(spectral_transform.get("kind", "direct_eigenvalues")),
                "spectral_corruption": (
                    "VP_SDE_on_standardized_log_successive_laplacian_eigenvalue_gaps"
                    if variant in LAPLACIAN_LOGGAP_ALL_VARIANTS
                    else (
                        "VP_SDE_on_laplacian_eigenvalues_in_fixed_training_graph_laplacian_eigenbasis"
                        if spectral_operator == "combinatorial_laplacian"
                        else "VP_SDE_on_eigenvalues_in_fixed_training_graph_eigenbasis"
                    )
                ),
                "denoiser_conditioning": (
                    "offdiagonal_minus_U_diag_lambda_t_Ut_plus_noisy_degree_features"
                    if spectral_operator == "combinatorial_laplacian"
                    else "continuous_U_diag_lambda_t_Ut_plus_noisy_degree_features"
                ),
                "generation_basis": (
                    "uniform_training_laplacian_eigenbasis_joint_with_node_count"
                    if spectral_operator == "combinatorial_laplacian"
                    else "uniform_training_adjacency_eigenbasis_joint_with_node_count"
                ),
                "reverse_sampler": "Euler_Maruyama_plus_optional_Langevin_corrector",
                "structure_summary_training": (
                    {
                        "graphlet": "connected_induced_k3_k4_k5_histograms_plus_connected_subset_mass",
                        "clustering": (
                            "node_clustering_coefficient_histogram"
                            if bool(joint_graphlet_cfg.get("clustering", {}).get("enabled", False))
                            else "disabled"
                        ),
                        "orbit": (
                            "ORCA_0_14_orbit_type_histogram_plus_log1p_total_count"
                            if bool(joint_graphlet_cfg.get("orbit", {}).get("enabled", False))
                            else "disabled"
                        ),
                    }
                    if variant in LAPLACIAN_LOGGAP_GRAPHLET_AUX_VARIANTS else "none"
                ),
                "graphlet_summary_training": (
                    "joint_auxiliary_head_shared_with_spectrum_denoiser_predicting_clean_k3_k4_k5_graphlet_histograms_and_connected_subset_mass"
                    if variant in LAPLACIAN_LOGGAP_GRAPHLET_AUX_VARIANTS else "none"
                ),
                "structure_score_training": (
                    "distilled_grad_log_p_clean_structure_summary_given_noisy_log_gap_state"
                    if bool(structure_score_cfg.get("enabled", False)) else "none"
                ),
                "structure_score_sampling": (
                    "late_reverse_score_guidance_conditioned_on_denoised_predicted_structure_summary"
                    if bool(structure_score_cfg.get("enabled", False))
                    and bool(structure_score_cfg.get("guidance", {}).get("enabled", False))
                    else "none"
                ),
                "discretization": (
                    "single_final_threshold_on_negative_laplacian_offdiagonal"
                    if spectral_operator == "combinatorial_laplacian"
                    else "single_final_threshold"
                ),
                "auxiliary_degree_prior": (
                    "jointly_trained_DH-VAE_independent_of_GSDM"
                    if degree_prior_enabled else "none"
                ),
                "degree_prior_used_for_graph_generation": False,
                "degree_constraint": (
                    "freeze_final_vanilla_gsdm_indexed_degree_vector"
                    if variant in (
                        GRAPHLET_REFINEMENT_VARIANTS
                        | LAPLACIAN_LOGGAP_GRAPHLET_REFINEMENT_VARIANTS
                    ) else False
                ),
                "rewiring": (
                    "post_gsdm_degree_preserving_double_edge_swaps"
                    if variant in (
                        GRAPHLET_REFINEMENT_VARIANTS
                        | LAPLACIAN_LOGGAP_GRAPHLET_REFINEMENT_VARIANTS
                    ) else False
                ),
                "categorical_edge_head": False,
                "structural_guidance": (
                    "joint_auxiliary_graphlet_prediction_plus_post_generation_degree_preserving_rewiring"
                    if variant in LAPLACIAN_LOGGAP_GRAPHLET_REFINEMENT_VARIANTS
                    else (
                    "post_generation_graphlet_guided_rewiring"
                    if variant in GRAPHLET_REFINEMENT_VARIANTS
                    else (
                        "joint_auxiliary_clean_graphlet_summary_prediction_during_log_gap_spectral_training"
                        if variant in LAPLACIAN_LOGGAP_GRAPHLET_AUX_VARIANTS else False
                    )
                    )
                ),
                "graphlet_predictor": graphlet_report,
                "posthoc_repair": False,
            },
            "test_used_for_training": False,
        }
        _write_json(staging / "manifest.json", manifest)
        if layout.train_dir.exists():
            shutil.rmtree(layout.train_dir)
        staging.replace(layout.train_dir)
        _write_json(layout.run_manifest_path, {
            "format": "grapher_baseline_run_v1",
            "model_id": wrapper.model_id,
            "dataset_id": request.run.dataset_id,
            "run_id": request.run.run_id,
            "train_seed": request.run.train_seed,
            "variant": variant,
        })
        return training_artifacts(wrapper, request)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _score_x(
    model: GSDMNodeScore, x: torch.Tensor, adj: torch.Tensor, flags: torch.Tensor,
    t: torch.Tensor, u: torch.Tensor, lam: torch.Tensor, sde: VPSDE,
) -> torch.Tensor:
    raw = model(x, adj, flags, u, lam)
    if not torch.isfinite(raw).all():
        raise FloatingPointError("Non-finite node score during sampling")
    _, std = sde.marginal_coeff(t)
    score = -raw / std[:, None, None].clamp_min(1.0e-12)
    if not torch.isfinite(score).all():
        raise FloatingPointError("Non-finite scaled node score during sampling")
    return score


def _score_lam(
    model: GSDMSpectrumScore,
    x: torch.Tensor,
    adj: torch.Tensor,
    flags: torch.Tensor,
    t: torch.Tensor,
    u: torch.Tensor,
    lam: torch.Tensor,
    sde: VPSDE,
    *,
    spectral_operator: str = "adjacency",
    spectral_transform: Mapping[str, Any] | None = None,
    structure_score_cfg: Mapping[str, Any] | None = None,
) -> torch.Tensor:
    score_cfg = dict(structure_score_cfg or {})
    guidance_scale = _structure_guidance_scale(t, score_cfg)
    guided = bool((guidance_scale > 0).any())

    current_shared = None
    if guided:
        if not isinstance(model, GSDMSpectrumGraphletPPGNScore):
            raise TypeError("Structural-score guidance currently requires the PPGN spectrum model")
        if model.structure_score_head is None:
            raise RuntimeError("Structural-score guidance requested but checkpoint has no score head")
        current_shared = model._encode(x, adj, flags, lam)
        current_outputs = model._heads_from_shared(current_shared)
        raw = current_outputs["spectrum"]
    else:
        raw = model(x, adj, flags, u, lam)

    if not torch.isfinite(raw).all():
        raise FloatingPointError("Non-finite spectrum score during sampling")
    mean_coeff, std = sde.marginal_coeff(t)
    score = -raw / std[:, None].clamp_min(1.0e-12)

    if guided:
        if spectral_operator != "combinatorial_laplacian":
            raise RuntimeError("Structural-score guidance is defined for Laplacian log-gap diffusion")
        guidance = dict(score_cfg.get("guidance", {}) or {})
        if str(guidance.get("target_source", "denoised_summary")) != "denoised_summary":
            raise ValueError("Only denoised_summary structural guidance is implemented")

        e_mask = eigen_mask_from_flags(flags, "laplacian_nonzero_prefix")
        # Epsilon prediction gives the usual x0 estimate in standardized
        # log-gap coordinates.  Its structural summary is the time-dependent
        # clean target presented to the second score head.
        state0_hat = (
            lam - std[:, None] * raw
        ) / mean_coeff[:, None].clamp_min(1.0e-6)
        state0_hat = state0_hat * e_mask
        eig0_hat = _spectral_state_to_operator_eigenvalues(
            state0_hat,
            flags,
            spectral_operator="combinatorial_laplacian",
            spectral_transform=spectral_transform,
        )
        op0_hat = reconstruct_adjacency(u, eig0_hat)
        adj0_hat = _operator_to_adjacency_state(
            op0_hat, flags, "combinatorial_laplacian"
        )
        target_shared = model._encode(x, adj0_hat, flags, state0_hat)
        target_outputs = model._heads_from_shared(target_shared)
        target_summary = model.structure_vector_from_outputs(target_outputs).detach()
        structural_score = model.structure_score_from_shared(
            current_shared,
            target_summary,
            t,
        )
        structural_score = structural_score * e_mask
        structural_score = _clip_vector_norm(
            structural_score,
            float(guidance.get("score_clip_norm", 5.0)),
        )
        score = score + guidance_scale[:, None] * structural_score

    if not torch.isfinite(score).all():
        raise FloatingPointError("Non-finite scaled spectrum score during sampling")
    return score


def _langevin_x(
    model: GSDMNodeScore, x: torch.Tensor, adj: torch.Tensor, flags: torch.Tensor,
    t: torch.Tensor, u: torch.Tensor, lam: torch.Tensor, sde: VPSDE,
    *, snr: float, scale_eps: float, n_steps: int, generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_mean = x
    timestep = (t * (sde.N - 1) / sde.T).long()
    alpha = sde.alphas[timestep]
    for _ in range(n_steps):
        grad = _score_x(model, x, adj, flags, t, u, lam, sde)
        noise = mask_x(torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator), flags)
        grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean().clamp_min(1.0e-12)
        noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean().clamp_min(1.0e-12)
        step = (float(snr) * noise_norm / grad_norm).square() * 2.0 * alpha
        x_mean = x + step[:, None, None] * grad
        x = x_mean + torch.sqrt(2.0 * step)[:, None, None] * noise * float(scale_eps)
        x = mask_x(x, flags)
    return x, x_mean


def _langevin_lambda(
    model: GSDMSpectrumScore, x: torch.Tensor, adj: torch.Tensor, flags: torch.Tensor,
    t: torch.Tensor, u: torch.Tensor, lam: torch.Tensor, sde: VPSDE,
    *, snr: float, scale_eps: float, n_steps: int, eigen_mask: torch.Tensor,
    generator: torch.Generator,
    spectral_operator: str = "adjacency",
    spectral_transform: Mapping[str, Any] | None = None,
    structure_score_cfg: Mapping[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    lam_mean = lam
    timestep = (t * (sde.N - 1) / sde.T).long()
    alpha = sde.alphas[timestep]
    u_t = u.transpose(-1, -2)
    adj_sample = adj
    adj_mean = adj
    for _ in range(n_steps):
        grad = _score_lam(
            model, x, adj_sample, flags, t, u, lam, sde,
            spectral_operator=spectral_operator,
            spectral_transform=spectral_transform,
            structure_score_cfg=structure_score_cfg,
        )
        # The released GSDM sampler draws full eigenvalue noise.  The persistent
        # reverse-state mean is masked *after* each corrector/predictor update.
        noise = torch.randn(lam.shape, device=lam.device, dtype=lam.dtype, generator=generator)
        if spectral_operator == "combinatorial_laplacian":
            grad = grad * eigen_mask
            noise = noise * eigen_mask
        grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean().clamp_min(1.0e-12)
        noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean().clamp_min(1.0e-12)
        step = (float(snr) * noise_norm / grad_norm).square() * 2.0 * alpha
        lam_mean = lam + step[:, None] * grad
        lam_sample = lam_mean + torch.sqrt(2.0 * step)[:, None] * noise * float(scale_eps)
        if spectral_operator == "combinatorial_laplacian":
            lam_mean = lam_mean * eigen_mask
            lam_sample = lam_sample * eigen_mask
        eig_sample = _spectral_state_to_operator_eigenvalues(
            lam_sample, flags, spectral_operator=spectral_operator, spectral_transform=spectral_transform
        )
        eig_mean = _spectral_state_to_operator_eigenvalues(
            lam_mean, flags, spectral_operator=spectral_operator, spectral_transform=spectral_transform
        )
        operator_sample = u @ torch.diag_embed(eig_sample) @ u_t
        operator_mean = u @ torch.diag_embed(eig_mean) @ u_t
        adj_sample = _operator_to_adjacency_state(operator_sample, flags, spectral_operator)
        adj_mean = _operator_to_adjacency_state(operator_mean, flags, spectral_operator)
    return adj_sample, adj_mean, lam_sample, lam_mean


def _euler_x(
    model: GSDMNodeScore, x: torch.Tensor, adj: torch.Tensor, flags: torch.Tensor,
    t: torch.Tensor, u: torch.Tensor, lam: torch.Tensor, sde: VPSDE,
    *, generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    beta = sde.beta(t)
    score = _score_x(model, x, adj, flags, t, u, lam, sde)
    drift = -0.5 * beta[:, None, None] * x - beta[:, None, None] * score
    dt = -1.0 / float(sde.N)
    x_mean = x + drift * dt
    noise = mask_x(torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator), flags)
    x = x_mean + torch.sqrt(beta * (-dt))[:, None, None] * noise
    return mask_x(x, flags), mask_x(x_mean, flags)


def _euler_lambda(
    model: GSDMSpectrumScore, x: torch.Tensor, adj: torch.Tensor, flags: torch.Tensor,
    t: torch.Tensor, u: torch.Tensor, lam: torch.Tensor, sde: VPSDE,
    *, eigen_mask: torch.Tensor, generator: torch.Generator,
    spectral_operator: str = "adjacency",
    spectral_transform: Mapping[str, Any] | None = None,
    structure_score_cfg: Mapping[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    beta = sde.beta(t)
    score = _score_lam(
        model, x, adj, flags, t, u, lam, sde,
        spectral_operator=spectral_operator,
        spectral_transform=spectral_transform,
        structure_score_cfg=structure_score_cfg,
    )
    if spectral_operator == "combinatorial_laplacian":
        score = score * eigen_mask
    drift = -0.5 * beta[:, None] * lam - beta[:, None] * score
    dt = -1.0 / float(sde.N)
    lam_mean = lam + drift * dt
    noise = torch.randn(lam.shape, device=lam.device, dtype=lam.dtype, generator=generator)
    if spectral_operator == "combinatorial_laplacian":
        noise = noise * eigen_mask
    lam_sample = lam_mean + torch.sqrt(beta * (-dt))[:, None] * noise
    if spectral_operator == "combinatorial_laplacian":
        lam_mean = lam_mean * eigen_mask
        lam_sample = lam_sample * eigen_mask
    u_t = u.transpose(-1, -2)
    eig_sample = _spectral_state_to_operator_eigenvalues(
        lam_sample, flags, spectral_operator=spectral_operator, spectral_transform=spectral_transform
    )
    eig_mean = _spectral_state_to_operator_eigenvalues(
        lam_mean, flags, spectral_operator=spectral_operator, spectral_transform=spectral_transform
    )
    operator_sample = u @ torch.diag_embed(eig_sample) @ u_t
    operator_mean = u @ torch.diag_embed(eig_mean) @ u_t
    adj_sample = _operator_to_adjacency_state(operator_sample, flags, spectral_operator)
    adj_mean = _operator_to_adjacency_state(operator_mean, flags, spectral_operator)
    return adj_sample, adj_mean, lam_sample, lam_mean


def sample_batch(
    model_x: GSDMNodeScore,
    model_lam: GSDMSpectrumScore,
    *,
    donor_adjacencies: torch.Tensor,
    donor_sizes: torch.Tensor,
    sde_x: VPSDE,
    sde_lam: VPSDE,
    sample_cfg: Mapping[str, Any],
    eigen_mask_mode: str,
    spectral_operator: str,
    spectral_transform: Mapping[str, Any] | None = None,
    structure_score_cfg: Mapping[str, Any] | None = None,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    donor_adjacencies = donor_adjacencies.to(device)
    donor_sizes = donor_sizes.to(device)
    b, nmax, _ = donor_adjacencies.shape
    flags = (
        torch.arange(nmax, device=device).unsqueeze(0)
        < donor_sizes.unsqueeze(1)
    ).to(torch.float32)
    if spectral_operator == "adjacency":
        _, u = torch.linalg.eigh(donor_adjacencies)
    elif spectral_operator == "combinatorial_laplacian":
        _, u = _laplacian_eigh_padded(donor_adjacencies, donor_sizes)
    else:
        raise ValueError(f"Unknown spectral operator {spectral_operator!r}")
    e_mask = eigen_mask_from_flags(flags, eigen_mask_mode)
    max_feat_num = model_x.max_feat_num
    x = mask_x(
        torch.randn((b, nmax, max_feat_num), device=device, generator=generator), flags
    )
    # Upstream prior_sampling_sym3 is an unmasked standard-normal vector.
    # The eigen-mask is applied to the carried mean after each reverse update.
    lam = torch.randn((b, nmax), device=device, generator=generator)
    if spectral_operator == "combinatorial_laplacian":
        lam = lam * e_mask
    eig = _spectral_state_to_operator_eigenvalues(
        lam, flags, spectral_operator=spectral_operator, spectral_transform=spectral_transform
    )
    operator = reconstruct_adjacency(u, eig)
    adj = _operator_to_adjacency_state(operator, flags, spectral_operator)
    timesteps = torch.linspace(
        sde_lam.T, float(sample_cfg.get("eps", 1.0e-4)), sde_lam.N, device=device
    )
    corrector = str(sample_cfg.get("corrector", "langevin")).lower()
    n_steps = int(sample_cfg.get("n_steps", 1))
    if n_steps < 0:
        raise ValueError("sample.n_steps must be non-negative")
    x_mean = x
    adj_mean = adj
    final_lam_mean = lam
    final_lam_sample = lam
    for t_scalar in timesteps:
        t = torch.full((b,), float(t_scalar.item()), dtype=torch.float32, device=device)
        old_x = x
        if corrector == "langevin" and n_steps:
            x, x_mean = _langevin_x(
                model_x, x, adj, flags, t, u, lam, sde_x,
                snr=float(sample_cfg.get("snr", 0.05)),
                scale_eps=float(sample_cfg.get("scale_eps", 0.7)),
                n_steps=n_steps, generator=generator,
            )
            adj, adj_mean, corr_sample, corr_mean = _langevin_lambda(
                model_lam, old_x, adj, flags, t, u, lam, sde_lam,
                snr=float(sample_cfg.get("snr", 0.05)),
                scale_eps=float(sample_cfg.get("scale_eps", 0.7)),
                n_steps=n_steps, eigen_mask=e_mask, generator=generator,
                spectral_operator=spectral_operator,
                spectral_transform=spectral_transform,
                structure_score_cfg=structure_score_cfg,
            )
            final_lam_sample = corr_sample
            final_lam_mean = corr_mean
            lam = corr_mean * e_mask
        old_x = x
        x, x_mean = _euler_x(model_x, x, adj, flags, t, u, lam, sde_x, generator=generator)
        adj, adj_mean, pred_sample, pred_mean = _euler_lambda(
            model_lam, old_x, adj, flags, t, u, lam, sde_lam,
            eigen_mask=e_mask, generator=generator,
            spectral_operator=spectral_operator,
            spectral_transform=spectral_transform,
            structure_score_cfg=structure_score_cfg,
        )
        final_lam_sample = pred_sample
        final_lam_mean = pred_mean
        lam = pred_mean * e_mask
    if bool(sample_cfg.get("noise_removal", True)):
        eig_mean = _spectral_state_to_operator_eigenvalues(
            final_lam_mean, flags, spectral_operator=spectral_operator, spectral_transform=spectral_transform
        )
        return x_mean, adj_mean, eig_mean, u
    eig_sample = _spectral_state_to_operator_eigenvalues(
        final_lam_sample, flags, spectral_operator=spectral_operator, spectral_transform=spectral_transform
    )
    return x, adj, eig_sample, u


def generate(wrapper, request: GenerateRequest, state: Mapping[str, Any], manifest: Mapping[str, Any], options: Mapping[str, Any]) -> GenerationArtifacts:
    validate_options(options)
    if state.get("format") not in SUPPORTED_CHECKPOINT_FORMATS:
        raise RuntimeError(
            f"Expected one of {sorted(SUPPORTED_CHECKPOINT_FORMATS)}, found {state.get('format')!r}"
        )
    device = _resolve_device(options.get("runtime", {}))
    _seed_everything(request.generation_seed)
    model_cfg = state["model_config"]
    checkpoint_structure_score = copy.deepcopy(
        dict(state.get("structure_score", model_cfg.get("structure_score", {})) or {})
    )
    option_structure_score = _resolved_structure_score_config(options)
    structure_score_cfg = checkpoint_structure_score
    if bool(checkpoint_structure_score.get("enabled", False)):
        # Guidance strength/schedule can be changed at generation time without
        # changing the trained head architecture.
        structure_score_cfg["guidance"] = copy.deepcopy(
            dict(option_structure_score.get("guidance", checkpoint_structure_score.get("guidance", {})) or {})
        )
        structure_score_cfg["enabled"] = True
    structure_summary_payload = state.get("structure_summary", {}) or {}
    if not bool(structure_summary_payload.get("enabled", False)):
        structure_summary_payload = state.get("graphlet_summary", {}) or {}
    graphlet_summary_payload = structure_summary_payload  # backwards-compatible local alias
    graphlet_slices = None
    if bool(structure_summary_payload.get("enabled", False)):
        graphlet_slices = tuple(
            tuple(int(v) for v in pair)
            for pair in structure_summary_payload.get("graphlet_slices", [])
        )
    model_x, model_lam = _build_models(
        model_cfg,
        device,
        graphlet_slices=graphlet_slices,
        summary_meta=structure_summary_payload,
    )
    if bool(options.get("sample", {}).get("use_ema", False)):
        model_x.load_state_dict(state["ema_x_state"])
        model_lam.load_state_dict(state["ema_spectrum_state"])
    else:
        model_x.load_state_dict(state["model_x_state"])
        model_lam.load_state_dict(state["model_spectrum_state"])
    model_x.eval(); model_lam.eval()
    sde_x, sde_lam = _make_sdes({"sde": state["sde"]}, device)
    sample_cfg = copy.deepcopy(dict(state["sample"]))
    sample_cfg.update(dict(options.get("sample", {})))
    if int(sde_x.N) != int(sde_lam.N):
        raise ValueError("Vanilla GSDM sampling requires matching node/spectrum discretization counts")
    basis_adj = torch.tensor(np.asarray(state["basis_adjacencies"]), dtype=torch.float32)
    basis_n = torch.tensor(np.asarray(state["basis_num_nodes"]), dtype=torch.long)
    if len(basis_adj) == 0:
        raise RuntimeError("Checkpoint has an empty training basis bank")
    batch_size = int(options.get("generation_batch_size", 128))
    rng = np.random.default_rng(request.generation_seed)
    generator = torch.Generator(device=device).manual_seed(request.generation_seed)
    eigen_mask_mode = str(state["sde"].get("eigen_mask", "official_extremes"))
    threshold = float(sample_cfg.get("threshold", 0.5))
    variant = str(state.get("variant", options.get("variant", "vanilla_gsdm"))).lower()
    spectral_operator = str(
        state.get("spectral_operator", spectral_operator_for_variant(variant))
    )
    spectral_transform = dict(state.get("spectral_transform", {"kind": "direct_eigenvalues"}) or {})
    degree_payload = state.get("degree_prior", {})
    degree_prior_enabled = bool(
        isinstance(degree_payload, Mapping) and degree_payload.get("enabled", False)
    )
    if variant in {"vanilla_gsdm_dhvae", "vanilla_gsdm_plus_dhvae"} and not degree_prior_enabled:
        raise RuntimeError("vanilla_gsdm_dhvae generation requires an embedded jointly trained DH-VAE")
    graphs: list[nx.Graph] = []
    continuous: list[np.ndarray] = []
    continuous_laplacians: list[np.ndarray] = []
    spectra: list[np.ndarray] = []
    basis_indices: list[int] = []
    joint_graphlet_predictions: list[dict[str, Any]] = []
    joint_structure_predictions: list[dict[str, Any]] = []
    start = time.monotonic()
    with torch.no_grad():
        while len(graphs) < request.num_graphs:
            b = min(batch_size, request.num_graphs - len(graphs))
            indices = rng.integers(0, len(basis_adj), size=b)
            donor_adj = basis_adj[indices]
            donor_n = basis_n[indices]
            sample_x, soft, lam, sampled_u = sample_batch(
                model_x, model_lam,
                donor_adjacencies=donor_adj,
                donor_sizes=donor_n,
                sde_x=sde_x, sde_lam=sde_lam,
                sample_cfg=sample_cfg,
                eigen_mask_mode=eigen_mask_mode,
                spectral_operator=spectral_operator,
                spectral_transform=spectral_transform,
                structure_score_cfg=structure_score_cfg,
                device=device, generator=generator,
            )
            soft = 0.5 * (soft + soft.transpose(-1, -2))
            operator_batch = reconstruct_adjacency(sampled_u, lam)
            batch_structure_summary = None
            if variant in LAPLACIAN_LOGGAP_GRAPHLET_AUX_VARIANTS:
                if not _is_joint_graphlet_score(model_lam):
                    raise AssertionError("log-gap graphlet generation requires joint summary head")
                nmax = soft.size(1)
                flags = (
                    torch.arange(nmax, device=device).unsqueeze(0)
                    < donor_n.to(device).unsqueeze(1)
                ).to(torch.float32)
                state_for_summary = _operator_eigenvalues_to_spectral_state(
                    lam,
                    flags,
                    spectral_operator="combinatorial_laplacian",
                    spectral_transform=spectral_transform,
                )
                summary_outputs = model_lam.forward_all(
                    sample_x, soft, flags, sampled_u, state_for_summary
                )
                if hasattr(model_lam, "structure_means_from_outputs"):
                    batch_structure_summary = model_lam.structure_means_from_outputs(summary_outputs)
                else:
                    gh, gm = model_lam.graphlet_means_from_outputs(summary_outputs)
                    batch_structure_summary = {
                        "graphlet_histogram": gh,
                        "graphlet_mass": gm,
                    }
            for row, idx in enumerate(indices.tolist()):
                n = int(donor_n[row].item())
                matrix = soft[row, :n, :n].detach().cpu().numpy().astype(np.float64)
                np.fill_diagonal(matrix, 0.0)
                discrete = matrix > threshold
                np.fill_diagonal(discrete, False)
                graph = nx.from_numpy_array(discrete.astype(np.int8), create_using=nx.Graph)
                graphs.append(graph)
                continuous.append(matrix)
                if spectral_operator == "combinatorial_laplacian":
                    lap = (
                        operator_batch[row, :n, :n]
                        .detach().cpu().numpy().astype(np.float64)
                    )
                    lap = 0.5 * (lap + lap.T)
                    continuous_laplacians.append(lap)
                spectra.append(lam[row, :].detach().cpu().numpy().astype(np.float64))
                basis_indices.append(int(idx))
                if batch_structure_summary is not None:
                    structure_row: dict[str, Any] = {"graph_index": len(graphs) - 1}
                    for key, value in batch_structure_summary.items():
                        structure_row[key] = value[row].detach().cpu().numpy().astype(np.float64)
                    joint_structure_predictions.append(structure_row)
                    joint_graphlet_predictions.append(
                        {
                            "graph_index": structure_row["graph_index"],
                            "graphlet_histogram": structure_row["graphlet_histogram"],
                            "graphlet_mass": structure_row["graphlet_mass"],
                        }
                    )
            label = "Vanilla-Laplacian-GSDM" if spectral_operator == "combinatorial_laplacian" else "Vanilla-GSDM"
            print(f"{label} generated {len(graphs)}/{request.num_graphs}", flush=True)

    vanilla_graphs: list[nx.Graph] = []
    pre_refinement_graphs: list[nx.Graph] = []
    frozen_degree_sequences: list[list[int]] = []
    graphlet_refinement_diagnostics: list[dict[str, Any]] = []
    graphlet_prediction_traces: list[list[dict[str, Any]]] = []
    if variant in GRAPHLET_REFINEMENT_VARIANTS:
        if spectral_operator != "adjacency":
            raise RuntimeError("Stage-3 graphlet refinement must use adjacency-spectrum vanilla GSDM")
        payload = state.get("graphlet_refinement", {})
        if not isinstance(payload, Mapping) or not bool(payload.get("enabled", False)):
            raise RuntimeError("Stage-3 checkpoint is missing its trained graphlet-summary predictor")
        from grapher.models.gdsm_simple.graphlet_stage3 import refine_generated_graphs
        vanilla_graphs = [graph.copy() for graph in graphs]
        pre_refinement_graphs = [graph.copy() for graph in graphs]
        graphs, frozen_degree_sequences, graphlet_refinement_diagnostics, graphlet_prediction_traces = (
            refine_generated_graphs(
                vanilla_graphs,
                payload=payload,
                config=dict(options.get("graphlet_refinement", {}) or {}),
                device=device,
                seed=request.generation_seed,
            )
        )
    elif variant in LAPLACIAN_LOGGAP_GRAPHLET_REFINEMENT_VARIANTS:
        if spectral_operator != "combinatorial_laplacian":
            raise RuntimeError("Log-gap joint-head refinement requires Laplacian spectral generation")
        if len(joint_graphlet_predictions) != len(graphs):
            raise RuntimeError(
                "Log-gap joint-head refinement requires one auxiliary graphlet prediction per generated graph"
            )
        from grapher.models.gdsm_simple.graphlet_stage3 import (
            refine_generated_graphs_with_fixed_predictions,
        )
        pre_refinement_graphs = [graph.copy() for graph in graphs]
        graphs, frozen_degree_sequences, graphlet_refinement_diagnostics, graphlet_prediction_traces = (
            refine_generated_graphs_with_fixed_predictions(
                pre_refinement_graphs,
                joint_graphlet_predictions,
                graphlet_summary_payload=graphlet_summary_payload,
                config=dict(options.get("graphlet_refinement", {}) or {}),
                seed=request.generation_seed,
            )
        )

    degree_sequences: list[list[int]] = []
    degree_summaries: list[dict[str, Any]] = []
    if degree_prior_enabled:
        degree_model, degree_vectorizer, degree_payload = _load_embedded_degree_prior(state, device)
        saved_sampling = dict(degree_payload.get("sampling_config", {}) or {})
        requested_sampling = dict(options.get("degree_prior", {}) or {})
        sampling = {**saved_sampling, **{
            key: requested_sampling[key]
            for key in saved_sampling
            if key in requested_sampling
        }}
        degree_rng = np.random.default_rng(request.generation_seed + 900001)
        # Degree sampling uses its own RNG stream.  It cannot alter the GSDM
        # graph sample because graph generation above is already complete and
        # no sampled degree value is passed back into the spectral model.
        fork_devices = (
            [device.index if device.index is not None else torch.cuda.current_device()]
            if device.type == "cuda" else []
        )
        with torch.random.fork_rng(devices=fork_devices, enabled=True):
            degree_seed = request.generation_seed + 900001
            torch.manual_seed(degree_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(degree_seed)
            sampler = DegreeVAESampler(
                "<embedded>",
                device=str(device),
                deterministic=False,
                seed=degree_seed,
                sample_num_nodes=str(sampling.get("sample_num_nodes", "empirical")),
                sample_num_edges=str(sampling.get("sample_num_edges", "model")),
                exact_degree_sum_conditioning=bool(sampling.get("exact_degree_sum_conditioning", True)),
                max_resample=int(sampling.get("max_resample", 500)),
                model_resample_attempts=int(sampling.get("model_resample_attempts", 32)),
                parity_conditioned=bool(sampling.get("parity_conditioned", False)),
                max_parity_resample=int(sampling.get("max_parity_resample", 1)),
                fallback=str(sampling.get("fallback", "error")),
                postprocess_policy=str(sampling.get("postprocess_policy", "reject_only")),
                model=degree_model,
                vectorizer=degree_vectorizer,
            )
            for index in range(request.num_graphs):
                summary = sampler.sample(degree_rng)
                sequence = [int(v) for v in summary["degree_sequence"]]
                degree_sequences.append(sequence)
                degree_summaries.append(_jsonable(summary))
                if (index + 1) % max(1, min(128, request.num_graphs)) == 0 or index + 1 == request.num_graphs:
                    print(
                        f"Auxiliary DH-VAE sampled {index + 1}/{request.num_graphs} degree sequences",
                        flush=True,
                    )

    layout = request.run.layout
    generation_id = request.resolved_generation_id
    target = layout.generation_dir(generation_id)
    ArtifactLayout.require_available(target, overwrite=request.overwrite)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".gdsm_simple_vanilla_generate_", dir=target.parent))
    try:
        graph_path = staging / "base_graphs.pkl"
        with graph_path.open("wb") as handle:
            pickle.dump(graphs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        vanilla_graph_path = None
        pre_refinement_graph_path = None
        frozen_degree_path = None
        graphlet_diagnostics_path = None
        graphlet_predictions_path = None
        if vanilla_graphs:
            vanilla_graph_path = staging / "vanilla_graphs.pkl"
            with vanilla_graph_path.open("wb") as handle:
                pickle.dump(vanilla_graphs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        if pre_refinement_graphs:
            pre_refinement_graph_path = staging / "pre_refinement_graphs.pkl"
            with pre_refinement_graph_path.open("wb") as handle:
                pickle.dump(pre_refinement_graphs, handle, protocol=pickle.HIGHEST_PROTOCOL)
            frozen_degree_path = staging / "frozen_degree_sequences.pkl"
            with frozen_degree_path.open("wb") as handle:
                pickle.dump(frozen_degree_sequences, handle, protocol=pickle.HIGHEST_PROTOCOL)
            graphlet_diagnostics_path = staging / "graphlet_refinement_diagnostics.pkl"
            with graphlet_diagnostics_path.open("wb") as handle:
                pickle.dump(graphlet_refinement_diagnostics, handle, protocol=pickle.HIGHEST_PROTOCOL)
            graphlet_predictions_path = staging / "graphlet_prediction_traces.pkl"
            with graphlet_predictions_path.open("wb") as handle:
                pickle.dump(graphlet_prediction_traces, handle, protocol=pickle.HIGHEST_PROTOCOL)
        soft_path = staging / "continuous_adjacencies.pkl"
        with soft_path.open("wb") as handle:
            pickle.dump(continuous, handle, protocol=pickle.HIGHEST_PROTOCOL)
        laplacian_path = None
        if continuous_laplacians:
            laplacian_path = staging / "continuous_laplacians.pkl"
            with laplacian_path.open("wb") as handle:
                pickle.dump(continuous_laplacians, handle, protocol=pickle.HIGHEST_PROTOCOL)
        spectrum_path = staging / "sampled_spectra.pkl"
        with spectrum_path.open("wb") as handle:
            pickle.dump(spectra, handle, protocol=pickle.HIGHEST_PROTOCOL)
        index_path = staging / "sampled_basis_indices.pkl"
        with index_path.open("wb") as handle:
            pickle.dump(basis_indices, handle, protocol=pickle.HIGHEST_PROTOCOL)
        joint_graphlet_prediction_path = None
        joint_structure_prediction_path = None
        if joint_graphlet_predictions:
            joint_graphlet_prediction_path = staging / "predicted_graphlet_summaries.pkl"
            with joint_graphlet_prediction_path.open("wb") as handle:
                pickle.dump(joint_graphlet_predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)
        if joint_structure_predictions:
            joint_structure_prediction_path = staging / "predicted_structure_summaries.pkl"
            with joint_structure_prediction_path.open("wb") as handle:
                pickle.dump(joint_structure_predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)
        degree_sequence_path = None
        degree_summary_path = None
        if degree_sequences:
            degree_sequence_path = staging / "sampled_degree_sequences.pkl"
            with degree_sequence_path.open("wb") as handle:
                pickle.dump(degree_sequences, handle, protocol=pickle.HIGHEST_PROTOCOL)
            degree_summary_path = staging / "sampled_degree_summaries.pkl"
            with degree_summary_path.open("wb") as handle:
                pickle.dump(degree_summaries, handle, protocol=pickle.HIGHEST_PROTOCOL)
        graph_hash = _sha256(graph_path)
        connected = [g.number_of_nodes() <= 1 or nx.is_connected(g) for g in graphs]
        degree_sums = [sum(dict(g.degree()).values()) for g in graphs]
        laplacian_diagnostics: dict[str, Any] = {}
        if spectral_operator == "combinatorial_laplacian":
            active_spectra = [
                np.asarray(spectra[i], dtype=np.float64)[: graphs[i].number_of_nodes()]
                for i in range(len(graphs))
            ]
            total_active = sum(values.size for values in active_spectra)
            negative = sum(int(np.count_nonzero(values < -1.0e-6)) for values in active_spectra)
            first_abs = [abs(float(values[0])) for values in active_spectra if values.size]
            row_sum_rmse = [
                float(np.sqrt(np.mean(np.square(lap.sum(axis=1)))))
                for lap in continuous_laplacians
            ]
            monotonic_pairs = sum(max(values.size - 1, 0) for values in active_spectra)
            monotonic_violations = sum(
                int(np.count_nonzero(np.diff(values) < -1.0e-8)) for values in active_spectra
            )
            laplacian_diagnostics = {
                "active_negative_eigenvalue_fraction": (
                    float(negative / total_active) if total_active else 0.0
                ),
                "nondecreasing_violation_fraction": (
                    float(monotonic_violations / monotonic_pairs) if monotonic_pairs else 0.0
                ),
                "first_eigenvalue_abs_mean": float(np.mean(first_abs)) if first_abs else 0.0,
                "reconstructed_laplacian_row_sum_rmse_mean": (
                    float(np.mean(row_sum_rmse)) if row_sum_rmse else 0.0
                ),
                "spectral_parameterization": str(spectral_transform.get("kind", "direct_eigenvalues")),
                "nonnegative_and_nondecreasing_by_construction": (
                    str(spectral_transform.get("kind", "")).lower() == "laplacian_log_gap"
                ),
                "valid_laplacian_projection_applied": False,
            }
        graphlet_aggregate: dict[str, Any] = {}
        if graphlet_refinement_diagnostics:
            accepted = [int(row.get("accepted_steps", 0)) for row in graphlet_refinement_diagnostics]
            calls = [int(row.get("prediction_calls", 0)) for row in graphlet_refinement_diagnostics]
            changed = [bool(row.get("changed", False)) for row in graphlet_refinement_diagnostics]
            preserved = [bool(row.get("degree_preserved", False)) for row in graphlet_refinement_diagnostics]
            skipped = [bool(row.get("skipped", False)) for row in graphlet_refinement_diagnostics]
            graphlet_aggregate = {
                "hard_degree_constraint": True,
                "degree_preservation_rate": float(np.mean(preserved)),
                "changed_rate": float(np.mean(changed)),
                "mean_accepted_steps": float(np.mean(accepted)),
                "mean_prediction_calls": float(np.mean(calls)),
                "disconnected_source_skip_rate": float(np.mean(skipped)),
                "graphlet_orders": [3, 4, 5],
            }
            _write_json(
                staging / "rewiring_diagnostics.json",
                {
                    "aggregate": graphlet_aggregate,
                    "per_graph": graphlet_refinement_diagnostics,
                },
            )

        _write_json(staging / "manifest.json", {
            "format": GENERATION_FORMAT,
            "model_id": wrapper.model_id,
            "variant": variant,
            "run_id": request.run.run_id,
            "generation_id": generation_id,
            "generation_seed": request.generation_seed,
            "num_requested": request.num_graphs,
            "num_generated": len(graphs),
            "duration_seconds": time.monotonic() - start,
            "base_graphs": {"path": "base_graphs.pkl", "sha256": graph_hash, "role": "final_graphs_after_optional_graphlet_refinement"},
            "vanilla_graphs": (
                {"path": "vanilla_graphs.pkl", "sha256": _sha256(vanilla_graph_path), "role": "unmodified_thresholded_vanilla_gsdm_sources"}
                if vanilla_graph_path is not None else None
            ),
            "pre_refinement_graphs": (
                {
                    "path": "pre_refinement_graphs.pkl",
                    "sha256": _sha256(pre_refinement_graph_path),
                    "role": (
                        "unmodified_thresholded_laplacian_loggap_sources"
                        if variant in LAPLACIAN_LOGGAP_GRAPHLET_REFINEMENT_VARIANTS
                        else "unmodified_thresholded_vanilla_gdsm_sources"
                    ),
                }
                if pre_refinement_graph_path is not None else None
            ),
            "frozen_degree_sequences": (
                {"path": "frozen_degree_sequences.pkl", "sha256": _sha256(frozen_degree_path), "role": "indexed_degrees_extracted_from_each_pre_refinement_source"}
                if frozen_degree_path is not None else None
            ),
            "graphlet_refinement_diagnostics": (
                {"path": "graphlet_refinement_diagnostics.pkl", "sha256": _sha256(graphlet_diagnostics_path)}
                if graphlet_diagnostics_path is not None else None
            ),
            "graphlet_prediction_traces": (
                {"path": "graphlet_prediction_traces.pkl", "sha256": _sha256(graphlet_predictions_path)}
                if graphlet_predictions_path is not None else None
            ),
            "rewiring_diagnostics": (
                {"path": "rewiring_diagnostics.json", "sha256": _sha256(staging / "rewiring_diagnostics.json")}
                if graphlet_refinement_diagnostics else None
            ),
            "continuous_adjacencies": {"path": "continuous_adjacencies.pkl", "sha256": _sha256(soft_path), "role": "pre_refinement_continuous_adjacency_scores"},
            "continuous_laplacians": (
                {"path": "continuous_laplacians.pkl", "sha256": _sha256(laplacian_path)}
                if laplacian_path is not None else None
            ),
            "predicted_graphlet_summaries": (
                {"path": "predicted_graphlet_summaries.pkl", "sha256": _sha256(joint_graphlet_prediction_path)}
                if joint_graphlet_prediction_path is not None else None
            ),
            "predicted_structure_summaries": (
                {
                    "path": "predicted_structure_summaries.pkl",
                    "sha256": _sha256(joint_structure_prediction_path),
                    "clustering_enabled": bool(structure_summary_payload.get("clustering_enabled", False)),
                    "orbit_enabled": bool(structure_summary_payload.get("orbit_enabled", False)),
                }
                if joint_structure_prediction_path is not None else None
            ),
            "sampled_spectra": {"path": "sampled_spectra.pkl", "sha256": _sha256(spectrum_path)},
            "sampled_basis_indices": {
                "path": "sampled_basis_indices.pkl", "sha256": _sha256(index_path),
                "role": "uniform indices into training-only adjacency/eigenbasis bank",
            },
            "sampled_degree_sequences": (
                {
                    "path": "sampled_degree_sequences.pkl",
                    "sha256": _sha256(degree_sequence_path),
                    "count": len(degree_sequences),
                    "role": "auxiliary_DH-VAE_samples_not_used_for_graph_generation",
                }
                if degree_sequence_path is not None else None
            ),
            "sampled_degree_summaries": (
                {"path": "sampled_degree_summaries.pkl", "sha256": _sha256(degree_summary_path)}
                if degree_summary_path is not None else None
            ),
            "checkpoint": {"path": str(request.checkpoint_path.resolve()), "sha256": _sha256(request.checkpoint_path)},
            "sampling": {
                "node_count": "jointly_sampled_with_training_eigenbasis",
                "score_backbone": str(model_cfg.get("backbone", "dense_gcn")),
                "node_backbone": str(model_cfg.get("node_backbone", model_cfg.get("backbone", "dense_gcn"))),
                "spectrum_structure_backbone": str(model_cfg.get("spectrum_backbone", model_cfg.get("backbone", "dense_gcn"))),
                "structural_features": _jsonable(model_cfg.get("structural_features", {})),
                "spectral_operator": spectral_operator,
                "spectral_parameterization": str(spectral_transform.get("kind", "direct_eigenvalues")),
                "eigenvectors": (
                    "training_split_empirical_laplacian_eigenbasis"
                    if spectral_operator == "combinatorial_laplacian"
                    else "training_split_empirical"
                ),
                "reverse_process": "coupled_VP_spectral_predictor_corrector",
                "threshold": threshold,
                "threshold_semantics": (
                    "edge iff -L_ij > threshold for i!=j"
                    if spectral_operator == "combinatorial_laplacian"
                    else "edge iff reconstructed_A_ij > threshold"
                ),
                "auxiliary_degree_prior": "DH-VAE" if degree_prior_enabled else "none",
                "degree_sequences_sampled": len(degree_sequences),
                "degree_sequences_used_for_graph_generation": False,
                "posthoc_repair": False,
                "rewiring": (
                    "graphlet_guided_degree_preserving_double_edge_swaps"
                    if variant in (
                        GRAPHLET_REFINEMENT_VARIANTS
                        | LAPLACIAN_LOGGAP_GRAPHLET_REFINEMENT_VARIANTS
                    ) else False
                ),
                "degree_constraint": (
                    "indexed_degree_vector_frozen_from_final_pre_refinement_graph"
                    if variant in (
                        GRAPHLET_REFINEMENT_VARIANTS
                        | LAPLACIAN_LOGGAP_GRAPHLET_REFINEMENT_VARIANTS
                    ) else False
                ),
                "structural_guidance": (
                    "learned_structural_log_score_guidance_in_standardized_log_gap_space"
                    if bool(structure_score_cfg.get("enabled", False))
                    and bool(structure_score_cfg.get("guidance", {}).get("enabled", False))
                    else (
                        "fixed_joint_auxiliary_graphlet_summary_k3_k4_k5"
                        if variant in LAPLACIAN_LOGGAP_GRAPHLET_REFINEMENT_VARIANTS
                        else (
                            "learned_connected_induced_graphlet_summary_k3_k4_k5"
                            if variant in GRAPHLET_REFINEMENT_VARIANTS else False
                        )
                    )
                ),
                "structure_score": _jsonable(structure_score_cfg),
            },
            "diagnostics": {
                "connectedness_rate": float(np.mean(connected)),
                "mean_degree_sum": float(np.mean(degree_sums)),
                **laplacian_diagnostics,
                **graphlet_aggregate,
            },
            "posthoc_repair": False,
            "largest_component_filter": False,
        })
        if target.exists():
            shutil.rmtree(target)
        staging.replace(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return GenerationArtifacts(
        run_dir=layout.run_dir,
        generation_dir=target,
        graphs_path=target / "base_graphs.pkl",
        manifest_path=target / "manifest.json",
        num_requested=request.num_graphs,
        num_generated=len(graphs),
        graphs_sha256=graph_hash,
    )

"""Hierarchical attributed Laplacian log-gap GSDM.

Design contract
---------------
* Topology is generated only by the Laplacian log-gap diffusion branch.
* The topology branch is jointly supervised by topology-only connected induced
  graphlets and ORCA orbit summaries.  Those targets depend on binary topology
  only; atom and bond labels are never used by this branch.
* Node categories and present-edge bond categories are predicted after topology
  by a separate categorical-denoising PPGN branch.
* The edge head has *no no-edge class*: it predicts bond type only for pairs
  whose existence is supplied by the topology branch.
* Node/edge categorical inputs are masked/corrupted one-hot vectors injected
  into the PPGN pair tensor.  This is masked categorical denoising, not a
  categorical flow-matching process.
* Typed connected induced graphlets k=3,4,5 supervise only the attribute branch.
  The topology and attribute PPGN encoders are separate, so typed-graphlet and
  categorical gradients cannot alter the topology score network.
* No post-generation rewiring or repair is performed.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import multiprocessing as mp
import pickle
import random
import shutil
import tempfile
import time
import warnings
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict
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

from grapher.models.artifacts import ArtifactLayout
from grapher.models.base import GenerateRequest, GenerationArtifacts, TrainRequest, TrainingArtifacts
from grapher.models.errors import ArtifactCollisionError
from grapher.properties.summary import SummaryConfig
from grapher.rewiring_mlp.attributed.data import GraphCategoryVocabulary, GraphletBasis
from grapher.rewiring_mlp.generic.basis import TopologyGraphletBasis
from grapher.rewiring_mlp.generic.graphlets import extract_topology_graphlet_target
from grapher.rewiring_mlp.properties.summary import clustering_histogram, python_orbit_count_vector

from .vanilla_gsdm import (
    MLP,
    PPGNLayer,
    GSDMNodeScore,
    _EMA,
    _fit_laplacian_log_gap_stats,
    _jsonable,
    _laplacian_eigh_padded,
    _make_sdes,
    _operator_eigenvalues_to_spectral_state,
    _operator_to_adjacency_state,
    _padded_dataset,
    _resolve_device,
    _seed_everything,
    _sha256,
    _spectral_state_to_operator_eigenvalues,
    _structural_features_from_state,
    _write_json,
    eigen_mask_from_flags,
    mask_x,
    reconstruct_adjacency,
    sample_batch,
)

from .attribute_decoding import (
    sample_categorical_logits as _sample_categorical_logits,
    validate_decode_config, sample_atoms, sample_bonds, decoding_diagnostics,
)

CATEGORICAL_TRAINING_CONTRACT = "undirected_mask_no_visible_target_fallback_v2"

CHECKPOINT_FORMAT = "gdsm_simple_laplacian_loggap_hierarchical_attributed_ppgn_v2"
TRAINING_FORMAT = "grapher_gdsm_simple_laplacian_loggap_hierarchical_attributed_training_v2"
GENERATION_FORMAT = "grapher_gdsm_simple_laplacian_loggap_hierarchical_attributed_generation_v2"
VARIANTS = {
    "vanilla_laplacian_loggap_hierarchical_attributed_ppgn",
    "laplacian_loggap_hierarchical_attributed_ppgn",
    "gsdm_laplacian_loggap_hierarchical_attributed_ppgn",
}

def _attributed_graphs(path: Path) -> list[nx.Graph]:
    """Load attributed molecular graphs, including valid one-node molecules.

    The generic/vanilla GSDM loader rejects graphs with fewer than two nodes
    because its original topology-only pipeline assumes at least one possible
    edge and infers support from adjacency.  Heavy-atom QM9 legitimately
    contains singleton molecules (for example methane after hydrogens are
    removed).  In the attributed Laplacian model a singleton is a well-defined
    degenerate topology: its combinatorial Laplacian has only the fixed zero
    eigenvalue, there are no log-gap coordinates or bond labels to predict, and
    the categorical node head still predicts the atom type.

    We therefore allow n >= 1 here and let the later connectedness check reject
    disconnected multi-node inputs.
    """
    from grapher.utils.networkx_pickle import load_trusted_networkx_pickle

    value = load_trusted_networkx_pickle(path)
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"Expected a non-empty graph list: {path}")
    result: list[nx.Graph] = []
    for index, graph in enumerate(value):
        if not isinstance(graph, nx.Graph):
            raise TypeError(f"Graph {index} in {path} is not networkx.Graph")
        if graph.is_directed() or graph.is_multigraph():
            raise ValueError("Attributed Laplacian GSDM supports simple undirected graphs only")
        g = nx.convert_node_labels_to_integers(graph, ordering="sorted")
        if g.number_of_nodes() < 1:
            raise ValueError(f"Attributed Laplacian GSDM requires at least one node: {path}[{index}]")
        result.append(g)
    return result



def default_options() -> dict[str, Any]:
    return {
        "variant": "vanilla_laplacian_loggap_hierarchical_attributed_ppgn",
        "train": {
            "epochs": 200,
            "batch_size": 128,
            "lr": 1.0e-3,
            "node_lr": 1.0e-2,
            "spectrum_lr": 1.0e-3,
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
            "node_backbone": "dense_gcn",
            "spectrum_backbone": "ppgn",
            "ppgn_hidden_dim": 64,
            "ppgn_depth": 4,
            "ppgn_residual_scale": 0.1,
            "ppgn_norm_eps": 1.0e-5,
            "ppgn_input_clip": 10.0,
        },
        "structural_features": {
            "enabled": True,
            "binarize_threshold": 0.5,
            "random_walk": {"enabled": True, "steps": 4},
            "shortest_path": {"enabled": True, "max_distance": 5},
        },
        "sde": {
            "x": {"type": "vp", "beta_min": 0.1, "beta_max": 1.0, "num_scales": 1000},
            "spectrum": {"type": "vp", "beta_min": 0.1, "beta_max": 1.0, "num_scales": 1000},
            "eps": 1.0e-5,
            "eigen_mask": "laplacian_nonzero_prefix",
            "spectral_parameterization": "laplacian_log_gap",
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
            "separate_attribute_rng": True,
        },
        "topology_summary": {
            "enabled": True,
            "loss_weight": 0.10,
            "graphlet": {
                "enabled": True,
                "orders": [3, 4, 5],
                "histogram_weight": 1.0,
                "mass_weight": 0.25,
                "connected_only": True,
                "topology_filter": "all",
            },
            "clustering": {
                "enabled": False,
                "bins": 100,
                "histogram_weight": 0.25,
                "cdf_weight": 1.0,
            },
            "orbit": {
                "enabled": True,
                "width": 15,
                "histogram_weight": 0.25,
                "log_total_weight": 0.10,
            },
        },
        "attribute_summary": {
            "enabled": True,
            "loss_weight": 0.10,
            "typed_graphlet": {
                "enabled": True,
                "orders": [3, 4, 5],
                "histogram_weight": 1.0,
                "mass_weight": 0.25,
                "connected_only": True,
                "topology_filter": "all",
                "attributed_backend": "python",
                "max_basis_graphs": 20000,
            },
        },
        "attributed": {
            "node_attribute": "atomic_num",
            "node_categories": [6, 7, 8, 9],
            "edge_attribute": "bond_type",
            "edge_categories": [1, 2, 3],
            "node_loss_weight": 1.0,
            "edge_loss_weight": 1.0,
            "corruption": {
                "mask_probability_min": 0.15,
                "mask_probability_max": 0.95,
                "full_mask_probability": 0.25,
            },
            "decode": {
                "node_mode": "argmax",
                "edge_mode": "argmax",
                "node_temperature": 1.0,
                "edge_temperature": 1.0,
                "two_pass": True,
                "constraint_mode": "none",
                "edge_order": "random",
                "infeasible_policy": "retain",
            },
        },
        "generation_batch_size": 128,
        "runtime": {"device": "auto"},
        "preprocess": {
            "num_workers": 0,
            "chunksize": 64,
            "progress_every": 2000,
            "cache_structure_targets": True,
        },
        "extensions": {
            "degree_conditioning": False,
            "hh_initialization": False,
            "degree_preserving_rewiring": False,
            "structural_summary": "none",
        },
    }


def _deep_update(base: dict[str, Any], changes: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in changes.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            base[key] = _deep_update(dict(base[key]), value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def validate_options(options: Mapping[str, Any]) -> None:
    if str(options.get("variant", "")).lower() not in VARIANTS:
        raise ValueError(
            "Hierarchical attributed log-gap pipeline requires "
            "vanilla_laplacian_loggap_hierarchical_attributed_ppgn"
        )
    if bool((options.get("graphlet_refinement", {}) or {}).get("enabled", False)):
        raise ValueError("Hierarchical attributed generation does not use rewiring")
    model = dict(options.get("model", {}) or {})
    if str(model.get("node_backbone", "dense_gcn")).lower() != "dense_gcn":
        raise ValueError("Attributed baseline keeps the node-feature denoiser DenseGCN")
    if str(model.get("spectrum_backbone", "ppgn")).lower() != "ppgn":
        raise ValueError("Attributed baseline requires PPGN for spectrum/structure/attribute heads")
    sde = dict(options.get("sde", {}) or {})
    if str(sde.get("spectral_parameterization", "")).lower() != "laplacian_log_gap":
        raise ValueError("Attributed baseline requires Laplacian log-gap spectral parameterization")
    topology_summary = dict(options.get("topology_summary", {}) or {})
    if not bool(topology_summary.get("enabled", False)):
        raise ValueError("topology_summary.enabled must be true")
    topology_graphlet = dict(topology_summary.get("graphlet", {}) or {})
    if not bool(topology_graphlet.get("enabled", False)):
        raise ValueError("topology_summary.graphlet.enabled must be true")
    if list(topology_graphlet.get("orders", [])) != [3, 4, 5]:
        raise ValueError("Topology graphlets are fixed to orders [3,4,5]")
    if not bool(topology_graphlet.get("connected_only", True)):
        raise ValueError("Topology graphlets must be connected-only")
    orbit = dict(topology_summary.get("orbit", {}) or {})
    if bool(orbit.get("enabled", False)) and int(orbit.get("width", 15)) != 15:
        raise ValueError("Orbit width must be 15")
    attribute_summary = dict(options.get("attribute_summary", {}) or {})
    if not bool(attribute_summary.get("enabled", False)):
        raise ValueError("attribute_summary.enabled must be true")
    typed_graphlet = dict(attribute_summary.get("typed_graphlet", {}) or {})
    if not bool(typed_graphlet.get("enabled", False)):
        raise ValueError("attribute_summary.typed_graphlet.enabled must be true")
    if list(typed_graphlet.get("orders", [])) != [3, 4, 5]:
        raise ValueError("Typed graphlets are fixed to orders [3,4,5]")
    if not bool(typed_graphlet.get("connected_only", True)):
        raise ValueError("Typed graphlets must be connected-only")
    attr = dict(options.get("attributed", {}) or {})
    for key in ("node_attribute", "edge_attribute"):
        if not attr.get(key):
            raise ValueError(f"attributed.{key} is required")
    if not list(attr.get("node_categories", [])):
        raise ValueError("attributed.node_categories must be non-empty")
    if not list(attr.get("edge_categories", [])):
        raise ValueError("attributed.edge_categories must be non-empty")
    decode = dict(attr.get("decode", {}) or {})
    for key in ("node_mode", "edge_mode"):
        mode = str(decode.get(key, "argmax")).lower()
        if mode not in {"argmax", "sample", "categorical", "stochastic"}:
            raise ValueError(f"attributed.decode.{key} must be argmax or sample, found {mode!r}")
    for key in ("node_temperature", "edge_temperature"):
        value = float(decode.get(key, 1.0))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"attributed.decode.{key} must be finite and > 0")
    validate_decode_config(
        decode, attr.get("node_categories", []), attr.get("edge_categories", []),
        node_attribute=attr.get("node_attribute"), edge_attribute=attr.get("edge_attribute"),
    )
    corruption = dict(attr.get("corruption", {}) or {})
    p0 = float(corruption.get("mask_probability_min", 0.15))
    p1 = float(corruption.get("mask_probability_max", 0.95))
    pf = float(corruption.get("full_mask_probability", 0.25))
    if not (0 <= p0 <= p1 <= 1 and 0 <= pf <= 1):
        raise ValueError("categorical corruption probabilities must satisfy 0<=min<=max<=1 and 0<=full<=1")


def _model_config(options: Mapping[str, Any], max_nodes: int, vocab: GraphCategoryVocabulary) -> dict[str, Any]:
    raw = dict(options["model"])
    max_feat_num = int(raw.get("max_feat_num") or max_nodes)
    return {
        "max_nodes": int(max_nodes),
        "max_feat_num": max_feat_num,
        "hidden_dim": int(raw.get("hidden_dim", 32)),
        "depth": int(raw.get("depth", 3)),
        "node_backbone": "dense_gcn",
        "spectrum_backbone": "ppgn",
        "ppgn_hidden_dim": int(raw.get("ppgn_hidden_dim", 64)),
        "ppgn_depth": int(raw.get("ppgn_depth", 4)),
        "ppgn_residual_scale": float(raw.get("ppgn_residual_scale", 0.1)),
        "ppgn_norm_eps": float(raw.get("ppgn_norm_eps", 1e-5)),
        "ppgn_input_clip": float(raw.get("ppgn_input_clip", 10.0)),
        "structural_features": copy.deepcopy(dict(options.get("structural_features", {}) or {})),
        "node_classes": int(vocab.num_node_categories),
        "edge_classes": int(len(vocab.edge_values)),  # real edge types only; no no-edge class
    }


class AttributedSpectrumPPGNScore(nn.Module):
    """Separate topology and attribute PPGN branches.

    The topology branch sees no atom/bond categories and produces the log-gap
    noise prediction plus topology-only graphlet/orbit summaries.  The attribute
    branch sees the fixed topology and masked categorical inputs and produces
    atom logits, present-edge bond logits, and typed graphlet summaries.  The
    branches have disjoint parameters by construction.
    """

    def __init__(
        self,
        *,
        max_feat_num: int,
        max_nodes: int,
        node_classes: int,
        edge_classes: int,
        topology_graphlet_slices: tuple[tuple[int, int], ...],
        typed_graphlet_slices: tuple[tuple[int, int], ...],
        structural_features: Mapping[str, Any] | None,
        topology_clustering_bins: int,
        topology_orbit_width: int,
        ppgn_hidden_dim: int = 64,
        ppgn_depth: int = 4,
        ppgn_residual_scale: float = 0.1,
        ppgn_norm_eps: float = 1e-5,
        ppgn_input_clip: float = 10.0,
        **_: Any,
    ) -> None:
        super().__init__()
        self.max_feat_num = int(max_feat_num)
        self.max_nodes = int(max_nodes)
        self.node_classes = int(node_classes)
        self.edge_classes = int(edge_classes)
        self.topology_graphlet_slices = tuple(
            (int(a), int(b)) for a, b in topology_graphlet_slices
        )
        self.typed_graphlet_slices = tuple(
            (int(a), int(b)) for a, b in typed_graphlet_slices
        )
        if not self.topology_graphlet_slices:
            raise ValueError("topology_graphlet_slices must be non-empty")
        if not self.typed_graphlet_slices:
            raise ValueError("typed_graphlet_slices must be non-empty")

        self.structural_features = copy.deepcopy(dict(structural_features or {}))
        rw_cfg = dict(self.structural_features.get("random_walk", {}) or {})
        sp_cfg = dict(self.structural_features.get("shortest_path", {}) or {})
        active = bool(self.structural_features.get("enabled", False))
        self.rw_steps = (
            int(rw_cfg.get("steps", 4))
            if active and bool(rw_cfg.get("enabled", False)) else 0
        )
        self.sp_dim = (
            int(sp_cfg.get("max_distance", 5)) + 2
            if active and bool(sp_cfg.get("enabled", False)) else 0
        )
        self.node_input_classes = self.node_classes + 1
        self.edge_input_classes = self.edge_classes + 1
        self.ppgn_hidden_dim = int(ppgn_hidden_dim)
        self.ppgn_input_clip = float(ppgn_input_clip)

        topology_node_dim = self.max_feat_num + self.rw_steps
        attribute_node_dim = topology_node_dim + self.node_input_classes
        topology_pair_dim = 2 + self.sp_dim + topology_node_dim
        attribute_pair_dim = 2 + self.sp_dim + attribute_node_dim + self.edge_input_classes
        hdim = self.ppgn_hidden_dim

        self.topology_pair_input = nn.Linear(topology_pair_dim, hdim)
        self.topology_pair_input_norm = nn.LayerNorm(hdim)
        self.topology_ppgn_layers = nn.ModuleList([
            PPGNLayer(
                hdim,
                residual_scale=float(ppgn_residual_scale),
                norm_eps=float(ppgn_norm_eps),
            )
            for _ in range(int(ppgn_depth))
        ])
        self.attribute_pair_input = nn.Linear(attribute_pair_dim, hdim)
        self.attribute_pair_input_norm = nn.LayerNorm(hdim)
        self.attribute_ppgn_layers = nn.ModuleList([
            PPGNLayer(
                hdim,
                residual_scale=float(ppgn_residual_scale),
                norm_eps=float(ppgn_norm_eps),
            )
            for _ in range(int(ppgn_depth))
        ])

        shared_dim = 2 * hdim + self.max_nodes
        self.spectrum_final = MLP(
            shared_dim, 2 * max(hdim, self.max_nodes), self.max_nodes, 2
        )
        topology_width = self.topology_graphlet_slices[-1][1]
        self.topology_graphlet_logits = MLP(
            shared_dim, 2 * shared_dim, topology_width, 2
        )
        self.topology_graphlet_mass_logits = MLP(
            shared_dim, 2 * shared_dim, len(self.topology_graphlet_slices), 2
        )
        self.topology_clustering_bins = int(topology_clustering_bins)
        self.topology_orbit_width = int(topology_orbit_width)
        self.topology_clustering_logits = (
            MLP(shared_dim, 2 * shared_dim, self.topology_clustering_bins, 2)
            if self.topology_clustering_bins > 0 else None
        )
        self.topology_orbit_histogram_logits = (
            MLP(shared_dim, 2 * shared_dim, self.topology_orbit_width, 2)
            if self.topology_orbit_width > 0 else None
        )
        self.topology_orbit_log_total_raw = (
            MLP(shared_dim, 2 * shared_dim, 1, 2)
            if self.topology_orbit_width > 0 else None
        )

        typed_width = self.typed_graphlet_slices[-1][1]
        self.typed_graphlet_logits = MLP(
            shared_dim, 2 * shared_dim, typed_width, 2
        )
        self.typed_graphlet_mass_logits = MLP(
            shared_dim, 2 * shared_dim, len(self.typed_graphlet_slices), 2
        )
        self.node_category_head = MLP(hdim, 2 * hdim, self.node_classes, 2)
        self.edge_category_head = MLP(hdim, 2 * hdim, self.edge_classes, 2)

    def topology_parameters(self):
        modules = [
            self.topology_pair_input,
            self.topology_pair_input_norm,
            self.topology_ppgn_layers,
            self.spectrum_final,
            self.topology_graphlet_logits,
            self.topology_graphlet_mass_logits,
        ]
        if self.topology_clustering_logits is not None:
            modules.append(self.topology_clustering_logits)
        if self.topology_orbit_histogram_logits is not None:
            modules.extend([
                self.topology_orbit_histogram_logits,
                self.topology_orbit_log_total_raw,
            ])
        for module in modules:
            yield from module.parameters()

    def attribute_parameters(self):
        modules = [
            self.attribute_pair_input,
            self.attribute_pair_input_norm,
            self.attribute_ppgn_layers,
            self.typed_graphlet_logits,
            self.typed_graphlet_mass_logits,
            self.node_category_head,
            self.edge_category_head,
        ]
        for module in modules:
            yield from module.parameters()

    def masked_attribute_inputs(
        self,
        flags: torch.Tensor,
        binary: torch.Tensor,
        *,
        node_attr: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, n = flags.shape
        if node_attr is None:
            node_attr = torch.zeros(
                (b, n, self.node_input_classes),
                dtype=flags.dtype,
                device=flags.device,
            )
            node_attr[..., -1] = flags.to(node_attr.dtype)
        if edge_attr is None:
            edge_attr = torch.zeros(
                (b, n, n, self.edge_input_classes),
                dtype=flags.dtype,
                device=flags.device,
            )
            edge_attr[..., -1] = binary.to(edge_attr.dtype)
        return node_attr, edge_attr

    def _bounded_adjacency(self, adj: torch.Tensor) -> torch.Tensor:
        return torch.tanh(
            torch.nan_to_num(
                adj,
                nan=0.0,
                posinf=self.ppgn_input_clip,
                neginf=-self.ppgn_input_clip,
            ).clamp(-self.ppgn_input_clip, self.ppgn_input_clip)
        )

    @staticmethod
    def _diag_pair(node: torch.Tensor) -> torch.Tensor:
        b, n, d = node.shape
        pair = node.new_zeros((b, n, n, d))
        idx = torch.arange(n, device=node.device)
        pair[:, idx, idx, :] = node
        return pair

    @staticmethod
    def _pool_pair(
        h: torch.Tensor,
        flags: torch.Tensor,
        eigenvalues: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, n = flags.shape
        idx = torch.arange(n, device=h.device)
        diag = h[:, idx, idx, :]
        count = flags.sum(dim=-1, keepdim=True).clamp_min(1.0).to(h.dtype)
        diag_pool = (diag * flags.unsqueeze(-1).to(h.dtype)).sum(dim=1) / count
        pair_mask = flags.to(h.dtype).unsqueeze(-1) * flags.to(h.dtype).unsqueeze(-2)
        diag_mask = torch.eye(n, dtype=h.dtype, device=h.device).unsqueeze(0)
        off_mask = pair_mask * (1.0 - diag_mask)
        off_count = off_mask.sum(dim=(1, 2)).unsqueeze(-1).clamp_min(1.0)
        off_pool = (h * off_mask.unsqueeze(-1)).sum(dim=(1, 2)) / off_count
        return torch.cat((diag_pool, off_pool, eigenvalues), dim=-1), pair_mask

    def _run_branch(
        self,
        pair: torch.Tensor,
        flags: torch.Tensor,
        eigenvalues: torch.Tensor,
        *,
        input_layer: nn.Linear,
        input_norm: nn.LayerNorm,
        layers: nn.ModuleList,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pair_mask = flags.to(pair.dtype).unsqueeze(-1) * flags.to(pair.dtype).unsqueeze(-2)
        h = F.elu(input_norm(input_layer(pair))) * pair_mask.unsqueeze(-1)
        for layer in layers:
            h = layer(h, pair_mask, flags)
        shared, _ = self._pool_pair(h, flags, eigenvalues)
        return shared, h

    def _base_parts(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        flags: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        binary, rw, sp = _structural_features_from_state(
            adj, flags, self.structural_features
        )
        node = torch.cat((x, rw), dim=-1) if rw.size(-1) else x
        parts = [self._bounded_adjacency(adj).unsqueeze(-1), binary.unsqueeze(-1)]
        if sp.size(-1):
            parts.append(sp)
        return binary, node, sp, torch.cat(parts, dim=-1)

    def _encode_topology(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        flags: torch.Tensor,
        eigenvalues: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        binary, node, _sp, base = self._base_parts(x, adj, flags)
        pair = torch.cat((base, self._diag_pair(node)), dim=-1)
        shared, pair_h = self._run_branch(
            pair,
            flags,
            eigenvalues,
            input_layer=self.topology_pair_input,
            input_norm=self.topology_pair_input_norm,
            layers=self.topology_ppgn_layers,
        )
        return shared, pair_h, binary

    def _encode_attributes(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        flags: torch.Tensor,
        eigenvalues: torch.Tensor,
        *,
        node_attr: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        binary, node, _sp, base = self._base_parts(x, adj, flags)
        node_attr, edge_attr = self.masked_attribute_inputs(
            flags, binary, node_attr=node_attr, edge_attr=edge_attr
        )
        node = torch.cat((node, node_attr), dim=-1)
        pair = torch.cat((base, self._diag_pair(node), edge_attr), dim=-1)
        shared, pair_h = self._run_branch(
            pair,
            flags,
            eigenvalues,
            input_layer=self.attribute_pair_input,
            input_norm=self.attribute_pair_input_norm,
            layers=self.attribute_ppgn_layers,
        )
        return shared, pair_h, binary

    def forward_topology(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        flags: torch.Tensor,
        eigenvectors: torch.Tensor,
        eigenvalues: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del eigenvectors
        shared, _pair_h, _binary = self._encode_topology(
            x, adj, flags, eigenvalues
        )
        outputs = {
            "spectrum": self.spectrum_final(shared),
            "topology_graphlet_logits": self.topology_graphlet_logits(shared),
            "topology_graphlet_mass_logits": self.topology_graphlet_mass_logits(shared),
        }
        if self.topology_clustering_logits is not None:
            outputs["topology_clustering_logits"] = self.topology_clustering_logits(shared)
        if self.topology_orbit_histogram_logits is not None:
            outputs["topology_orbit_histogram_logits"] = self.topology_orbit_histogram_logits(shared)
            outputs["topology_orbit_log_total_raw"] = self.topology_orbit_log_total_raw(shared)
        for name, value in outputs.items():
            if not torch.isfinite(value).all():
                raise FloatingPointError(f"Non-finite {name} from topology PPGN")
        return outputs

    def forward_attributes(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        flags: torch.Tensor,
        eigenvectors: torch.Tensor,
        eigenvalues: torch.Tensor,
        *,
        node_attr: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del eigenvectors
        shared, pair_h, _binary = self._encode_attributes(
            x,
            adj,
            flags,
            eigenvalues,
            node_attr=node_attr,
            edge_attr=edge_attr,
        )
        n = pair_h.size(1)
        idx = torch.arange(n, device=pair_h.device)
        diag = pair_h[:, idx, idx, :]
        sym_pair = 0.5 * (pair_h + pair_h.transpose(1, 2))
        outputs = {
            "typed_graphlet_logits": self.typed_graphlet_logits(shared),
            "typed_graphlet_mass_logits": self.typed_graphlet_mass_logits(shared),
            "node_logits": self.node_category_head(diag),
            "edge_logits": self.edge_category_head(sym_pair),
        }
        for name, value in outputs.items():
            if not torch.isfinite(value).all():
                raise FloatingPointError(f"Non-finite {name} from attribute PPGN")
        return outputs

    def forward_all(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        flags: torch.Tensor,
        eigenvectors: torch.Tensor,
        eigenvalues: torch.Tensor,
        *,
        node_attr: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        outputs = self.forward_topology(x, adj, flags, eigenvectors, eigenvalues)
        outputs.update(
            self.forward_attributes(
                x,
                adj,
                flags,
                eigenvectors,
                eigenvalues,
                node_attr=node_attr,
                edge_attr=edge_attr,
            )
        )
        return outputs

    def forward(self, x, adj, flags, eigenvectors, eigenvalues):
        return self.forward_topology(
            x, adj, flags, eigenvectors, eigenvalues
        )["spectrum"]

    @staticmethod
    def _block_histogram(
        logits: torch.Tensor,
        slices: tuple[tuple[int, int], ...],
    ) -> torch.Tensor:
        hist = torch.zeros_like(logits)
        for start, stop in slices:
            hist[:, start:stop] = torch.softmax(logits[:, start:stop], dim=-1)
        return hist

    def topology_structure_means_from_outputs(
        self,
        outputs: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        result = {
            "graphlet_histogram": self._block_histogram(
                outputs["topology_graphlet_logits"], self.topology_graphlet_slices
            ),
            "graphlet_mass": torch.sigmoid(outputs["topology_graphlet_mass_logits"]),
        }
        if "topology_clustering_logits" in outputs:
            result["clustering_histogram"] = torch.softmax(
                outputs["topology_clustering_logits"], dim=-1
            )
        if "topology_orbit_histogram_logits" in outputs:
            orbit_hist = torch.softmax(
                outputs["topology_orbit_histogram_logits"], dim=-1
            )
            orbit_log_total = F.softplus(outputs["topology_orbit_log_total_raw"])
            result["orbit_histogram"] = orbit_hist
            result["orbit_log_total"] = orbit_log_total
            result["orbit_mean_counts"] = orbit_hist * torch.expm1(orbit_log_total)
        return result

    def attribute_structure_means_from_outputs(
        self,
        outputs: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        return {
            "typed_graphlet_histogram": self._block_histogram(
                outputs["typed_graphlet_logits"], self.typed_graphlet_slices
            ),
            "typed_graphlet_mass": torch.sigmoid(
                outputs["typed_graphlet_mass_logits"]
            ),
        }

def _attributed_labels(graphs: Sequence[nx.Graph], vocab: GraphCategoryVocabulary, max_nodes: int) -> tuple[torch.Tensor, torch.Tensor]:
    nodes = np.full((len(graphs), max_nodes), -1, dtype=np.int64)
    edges = np.full((len(graphs), max_nodes, max_nodes), -1, dtype=np.int64)
    for gi, graph in enumerate(graphs):
        g = nx.convert_node_labels_to_integers(graph, ordering="sorted")
        n = g.number_of_nodes()
        for i, data in g.nodes(data=True):
            nodes[gi, int(i)] = vocab.node_index(data)
        for u, v, data in g.edges(data=True):
            cls = vocab.edge_index(data) - 1
            edges[gi, int(u), int(v)] = cls
            edges[gi, int(v), int(u)] = cls
    return torch.tensor(nodes, dtype=torch.long), torch.tensor(edges, dtype=torch.long)


def _topology_summary_config(options: Mapping[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(dict(options.get("topology_summary", {}) or {}))


def _attribute_summary_config(options: Mapping[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(dict(options.get("attribute_summary", {}) or {}))


def _topology_summary_spec(options: Mapping[str, Any]) -> SummaryConfig:
    summary = _topology_summary_config(options)
    graphlet = dict(summary.get("graphlet", {}) or {})
    orders = list(graphlet.get("orders", [3, 4, 5]))
    return SummaryConfig.from_dict({
        "clustering_summary": False,
        "spectral_summary": False,
        "motif_proxy": False,
        "orbit_count": False,
        "graphlet_history": True,
        "graphlet_k_min": min(orders),
        "graphlet_k_max": max(orders),
        "graphlet_connected_only": bool(graphlet.get("connected_only", True)),
        "graphlet_topology_filter": str(graphlet.get("topology_filter", "all")),
        "graphlet_backend": "exact",
        "graphlet_num_samples": None,
    })


def _typed_summary_spec(
    options: Mapping[str, Any],
    vocab: GraphCategoryVocabulary,
) -> SummaryConfig:
    summary = _attribute_summary_config(options)
    graphlet = dict(summary.get("typed_graphlet", {}) or {})
    orders = list(graphlet.get("orders", [3, 4, 5]))
    return SummaryConfig.from_dict({
        "clustering_summary": False,
        "spectral_summary": False,
        "motif_proxy": False,
        "orbit_count": False,
        "graphlet_history": True,
        "graphlet_k_min": min(orders),
        "graphlet_k_max": max(orders),
        "graphlet_connected_only": bool(graphlet.get("connected_only", True)),
        "graphlet_topology_filter": str(graphlet.get("topology_filter", "all")),
        "graphlet_backend": "exact",
        "graphlet_num_samples": None,
        "attributed": True,
        "node_attribute": vocab.node_attribute,
        "edge_attribute": vocab.edge_attribute,
        "attributed_backend": str(graphlet.get("attributed_backend", "python")),
    })


_HIERARCHICAL_TARGET_WORKER_STATE: dict[str, Any] = {}


def _init_hierarchical_target_worker(
    topology_basis: TopologyGraphletBasis,
    topology_scfg: SummaryConfig,
    typed_basis: GraphletBasis,
    typed_scfg: SummaryConfig,
    bins: int,
    width: int,
    clustering_enabled: bool,
    orbit_enabled: bool,
    seed: int,
) -> None:
    global _HIERARCHICAL_TARGET_WORKER_STATE
    _HIERARCHICAL_TARGET_WORKER_STATE = {
        "topology_basis": topology_basis,
        "topology_scfg": topology_scfg,
        "typed_basis": typed_basis,
        "typed_scfg": typed_scfg,
        "bins": int(bins),
        "width": int(width),
        "clustering_enabled": bool(clustering_enabled),
        "orbit_enabled": bool(orbit_enabled),
        "seed": int(seed),
    }


def _hierarchical_target_worker(
    item: tuple[int, nx.Graph],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    index, graph = item
    state = _HIERARCHICAL_TARGET_WORKER_STATE
    topology_target, topology_mass = extract_topology_graphlet_target(
        graph,
        graphlet_basis=state["topology_basis"],
        summary_config=state["topology_scfg"],
    )
    rng = np.random.default_rng(int(state["seed"]) + 333 + int(index))
    typed_history, typed_mass = state["typed_basis"].statistics_for_graph(
        graph, state["typed_scfg"], rng=rng
    )
    typed_target = state["typed_basis"].flatten_history(typed_history).astype(np.float32)
    typed_mass_vec = state["typed_basis"].flatten_mass(typed_mass).astype(np.float32)

    bins = int(state["bins"])
    width = int(state["width"])
    if bool(state["clustering_enabled"]):
        clustering = clustering_histogram(graph, bins=bins).astype(np.float32)
    else:
        clustering = np.zeros(bins, dtype=np.float32)
    if bool(state["orbit_enabled"]):
        counts = np.maximum(
            np.asarray(python_orbit_count_vector(graph), dtype=np.float64).reshape(-1),
            0.0,
        )
        if counts.size != width:
            raise ValueError(f"Expected orbit width {width}, got {counts.size}")
        total = float(counts.sum())
        orbit_hist = (
            counts / total if total > 0.0 else np.zeros_like(counts)
        ).astype(np.float32)
        orbit_total = np.asarray([np.log1p(total)], dtype=np.float32)
    else:
        orbit_hist = np.zeros(width, dtype=np.float32)
        orbit_total = np.zeros(1, dtype=np.float32)
    return (
        np.asarray(topology_target, dtype=np.float32),
        np.asarray(topology_mass, dtype=np.float32),
        clustering,
        orbit_hist,
        orbit_total,
        typed_target,
        typed_mass_vec,
    )


def _structure_cache_path(
    layout: ArtifactLayout,
    fingerprint: Any,
    options: Mapping[str, Any],
    vocab: GraphCategoryVocabulary,
    max_nodes: int,
) -> Path:
    payload = {
        "format": "gdsm_hierarchical_attr_structure_cache_v3",
        "dataset_fingerprint": _jsonable(fingerprint),
        "topology_summary": _jsonable(_topology_summary_config(options)),
        "attribute_summary": _jsonable(_attribute_summary_config(options)),
        "attributed": _jsonable(dict(options.get("attributed", {}) or {})),
        "vocabulary": _jsonable(vocab.to_dict()),
        "max_nodes": int(max_nodes),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    key = hashlib.sha256(encoded).hexdigest()[:24]
    return layout.run_dir.parent / "_preprocess_cache" / f"hierarchical_attributed_structure_{key}.pt"


def _torch_load_unrestricted(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _fit_typed_graphlet_basis(
    graphs: Sequence[nx.Graph],
    options: Mapping[str, Any],
    vocab: GraphCategoryVocabulary,
    *,
    seed: int,
) -> tuple[GraphletBasis, SummaryConfig]:
    typed_scfg = _typed_summary_spec(options, vocab)
    graphlet = dict(
        _attribute_summary_config(options).get("typed_graphlet", {}) or {}
    )
    max_basis = graphlet.get("max_basis_graphs", None)
    basis_graphs = list(graphs) if max_basis is None else list(graphs)[: int(max_basis)]
    progress_every = max(
        int((options.get("preprocess", {}) or {}).get("progress_every", 2000)), 1
    )
    last_reported: dict[int, int] = {}

    def _basis_progress(k: int, done: int, total: int, bins_seen: int) -> None:
        if done == total or done % progress_every == 0:
            if last_reported.get(int(k)) != int(done):
                print(
                    f"[hierarchical-preprocess] typed graphlet vocabulary k={k}: "
                    f"{done}/{total} graphs, bins={bins_seen}",
                    flush=True,
                )
                last_reported[int(k)] = int(done)

    raw_cfg = asdict(typed_scfg)
    raw_cfg.update({
        "attributed": True,
        "node_attribute": vocab.node_attribute,
        "edge_attribute": vocab.edge_attribute,
        "attributed_backend": str(graphlet.get("attributed_backend", "python")),
    })
    basis = GraphletBasis.fit_from_graphs(
        basis_graphs,
        raw_cfg,
        vocabulary=vocab,
        attributed=True,
        seed=seed,
        progress_callback=_basis_progress,
    )
    print(
        f"[hierarchical-preprocess] typed graphlet vocabulary complete: "
        f"basis_graphs={len(basis_graphs)}, width={basis.width}",
        flush=True,
    )
    return basis, typed_scfg


def _hierarchical_structure_targets(
    graphs: Sequence[nx.Graph],
    options: Mapping[str, Any],
    vocab: GraphCategoryVocabulary,
    *,
    topology_basis: TopologyGraphletBasis | None = None,
    typed_basis: GraphletBasis | None = None,
    seed: int = 0,
) -> tuple[
    tuple[torch.Tensor, ...],
    TopologyGraphletBasis,
    GraphletBasis,
    dict[str, Any],
    dict[str, Any],
]:
    topology_cfg = _topology_summary_config(options)
    topology_graphlet_cfg = dict(topology_cfg.get("graphlet", {}) or {})
    clustering_cfg = dict(topology_cfg.get("clustering", {}) or {})
    orbit_cfg = dict(topology_cfg.get("orbit", {}) or {})
    topology_scfg = _topology_summary_spec(options)
    if topology_basis is None:
        topology_basis = TopologyGraphletBasis.from_config(topology_scfg)
    if typed_basis is None:
        typed_basis, typed_scfg = _fit_typed_graphlet_basis(
            graphs, options, vocab, seed=seed
        )
    else:
        typed_scfg = _typed_summary_spec(options, vocab)

    pcfg = dict(options.get("preprocess", {}) or {})
    bins = int(clustering_cfg.get("bins", 100))
    width = int(orbit_cfg.get("width", 15))
    workers = max(int(pcfg.get("num_workers", 0)), 0)
    chunksize = max(int(pcfg.get("chunksize", 64)), 1)
    progress_every = max(int(pcfg.get("progress_every", 2000)), 1)
    total_graphs = len(graphs)
    started = time.monotonic()
    print(
        f"[hierarchical-preprocess] extracting topology and typed targets: "
        f"graphs={total_graphs}, workers={workers}",
        flush=True,
    )
    columns: list[list[np.ndarray]] = [[] for _ in range(7)]

    def _consume(result, done: int) -> None:
        for column, value in zip(columns, result):
            column.append(value)
        if done == total_graphs or done % progress_every == 0:
            elapsed = time.monotonic() - started
            rate = done / max(elapsed, 1.0e-9)
            print(
                f"[hierarchical-preprocess] structure targets: {done}/{total_graphs} "
                f"({rate:.1f} graphs/s)",
                flush=True,
            )

    initargs = (
        topology_basis,
        topology_scfg,
        typed_basis,
        typed_scfg,
        bins,
        width,
        bool(clustering_cfg.get("enabled", False)),
        bool(orbit_cfg.get("enabled", False)),
        int(seed),
    )
    if workers > 1 and total_graphs > 1:
        methods = mp.get_all_start_methods()
        ctx = mp.get_context("fork" if "fork" in methods else methods[0])
        with ctx.Pool(
            processes=workers,
            initializer=_init_hierarchical_target_worker,
            initargs=initargs,
        ) as pool:
            for done, result in enumerate(
                pool.imap(
                    _hierarchical_target_worker,
                    enumerate(graphs),
                    chunksize=chunksize,
                ),
                start=1,
            ):
                _consume(result, done)
    else:
        _init_hierarchical_target_worker(*initargs)
        for done, item in enumerate(enumerate(graphs), start=1):
            _consume(_hierarchical_target_worker(item), done)

    tensors = tuple(
        torch.tensor(np.stack(column), dtype=torch.float32) for column in columns
    )
    topology_meta = {
        "graphlet_slices": [list(x) for x in topology_basis.slices],
        "graphlet_basis": topology_basis.to_dict(),
        "summary_config": asdict(topology_scfg),
        "width": int(topology_basis.width),
        "orders": list(topology_graphlet_cfg.get("orders", [3, 4, 5])),
        "topology_only_graphlets": True,
        "clustering_enabled": bool(clustering_cfg.get("enabled", False)),
        "clustering_bins": bins,
        "orbit_enabled": bool(orbit_cfg.get("enabled", False)),
        "orbit_width": width,
    }
    typed_cfg = dict(
        _attribute_summary_config(options).get("typed_graphlet", {}) or {}
    )
    attribute_meta = {
        "graphlet_slices": [list(x) for x in typed_basis.slices],
        "graphlet_basis": typed_basis.to_dict(),
        "summary_config": asdict(typed_scfg),
        "width": int(typed_basis.width),
        "orders": list(typed_cfg.get("orders", [3, 4, 5])),
        "typed_graphlets": True,
        "graphlet_vocabulary": "training_only_plus_overflow",
    }
    return tensors, topology_basis, typed_basis, topology_meta, attribute_meta


def _topology_structure_targets(
    graphs: Sequence[nx.Graph],
    options: Mapping[str, Any],
) -> tuple[tuple[torch.Tensor, ...], TopologyGraphletBasis, dict[str, Any]]:
    """Small public helper used by tests and diagnostics."""
    dummy = GraphCategoryVocabulary.topology_only()
    # Typed extraction is not needed here; compute topology targets directly.
    cfg = _topology_summary_config(options)
    graphlet_cfg = dict(cfg.get("graphlet", {}) or {})
    clustering_cfg = dict(cfg.get("clustering", {}) or {})
    orbit_cfg = dict(cfg.get("orbit", {}) or {})
    scfg = _topology_summary_spec(options)
    basis = TopologyGraphletBasis.from_config(scfg)
    rows = []
    for graph in graphs:
        hist, mass = extract_topology_graphlet_target(
            graph, graphlet_basis=basis, summary_config=scfg
        )
        clustering = (
            clustering_histogram(graph, bins=int(clustering_cfg.get("bins", 100)))
            if bool(clustering_cfg.get("enabled", False))
            else np.zeros(int(clustering_cfg.get("bins", 100)), dtype=np.float32)
        )
        width = int(orbit_cfg.get("width", 15))
        if bool(orbit_cfg.get("enabled", False)):
            counts = np.maximum(np.asarray(python_orbit_count_vector(graph)), 0.0)
            total = float(counts.sum())
            oh = counts / total if total > 0 else np.zeros_like(counts)
            ot = np.asarray([np.log1p(total)], dtype=np.float32)
        else:
            oh = np.zeros(width, dtype=np.float32)
            ot = np.zeros(1, dtype=np.float32)
        rows.append((hist, mass, clustering, oh, ot))
    tensors = tuple(
        torch.tensor(np.stack([row[i] for row in rows]), dtype=torch.float32)
        for i in range(5)
    )
    meta = {
        "graphlet_slices": [list(x) for x in basis.slices],
        "graphlet_basis": basis.to_dict(),
        "summary_config": asdict(scfg),
        "width": basis.width,
        "orders": list(graphlet_cfg.get("orders", [3, 4, 5])),
        "topology_only_graphlets": True,
        "clustering_enabled": bool(clustering_cfg.get("enabled", False)),
        "clustering_bins": int(clustering_cfg.get("bins", 100)),
        "orbit_enabled": bool(orbit_cfg.get("enabled", False)),
        "orbit_width": int(orbit_cfg.get("width", 15)),
    }
    del dummy
    return tensors, basis, meta


def _typed_structure_targets(
    graphs: Sequence[nx.Graph],
    options: Mapping[str, Any],
    vocab: GraphCategoryVocabulary,
    *,
    basis: GraphletBasis | None = None,
    seed: int = 0,
) -> tuple[tuple[torch.Tensor, torch.Tensor], GraphletBasis, dict[str, Any]]:
    """Typed graphlet-only helper retained for focused tests."""
    if basis is None:
        basis, scfg = _fit_typed_graphlet_basis(graphs, options, vocab, seed=seed)
    else:
        scfg = _typed_summary_spec(options, vocab)
    histograms, masses = [], []
    for index, graph in enumerate(graphs):
        history, mass = basis.statistics_for_graph(
            graph, scfg, rng=np.random.default_rng(seed + 333 + index)
        )
        histograms.append(basis.flatten_history(history).astype(np.float32))
        masses.append(basis.flatten_mass(mass).astype(np.float32))
    cfg = dict(_attribute_summary_config(options).get("typed_graphlet", {}) or {})
    meta = {
        "graphlet_slices": [list(x) for x in basis.slices],
        "graphlet_basis": basis.to_dict(),
        "summary_config": asdict(scfg),
        "width": basis.width,
        "orders": list(cfg.get("orders", [3, 4, 5])),
        "typed_graphlets": True,
        "graphlet_vocabulary": "training_only_plus_overflow",
    }
    tensors = (
        torch.tensor(np.stack(histograms), dtype=torch.float32),
        torch.tensor(np.stack(masses), dtype=torch.float32),
    )
    return tensors, basis, meta

def _mask_categorical_inputs(
    node_labels: torch.Tensor,
    edge_labels: torch.Tensor,
    flags: torch.Tensor,
    current_binary: torch.Tensor,
    t: torch.Tensor,
    model: AttributedSpectrumPPGNScore,
    cfg: Mapping[str, Any],
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    b, n = node_labels.shape
    corr = dict(cfg.get("corruption", {}) or {})
    p0 = float(corr.get("mask_probability_min", 0.15))
    p1 = float(corr.get("mask_probability_max", 0.95))
    p = p0 + (p1 - p0) * t.clamp(0,1)
    full = torch.rand((b,), device=flags.device, generator=generator) < float(corr.get("full_mask_probability", 0.25))
    node_valid = flags.bool() & (node_labels >= 0)
    node_mask = (torch.rand((b,n), device=flags.device, generator=generator) < p[:,None]) | full[:,None]
    node_mask &= node_valid
    node_in = torch.zeros((b,n,model.node_input_classes), device=flags.device)
    safe_nodes = node_labels.clamp_min(0)
    node_in[..., :model.node_classes] = F.one_hot(safe_nodes, model.node_classes).float() * node_valid.unsqueeze(-1)
    node_in[node_mask] = 0.0
    node_in[..., -1] = node_mask.float()

    clean_edge = edge_labels >= 0
    active_pair = (current_binary.bool()
                   & flags.bool().unsqueeze(1) & flags.bool().unsqueeze(2)
                   & ~torch.eye(n, device=flags.device, dtype=torch.bool).unsqueeze(0))
    # Categorical edge information is attached only to the *current* topology.
    # Current false-positive edges receive MASK, never a no-edge category.
    known_current = active_pair & clean_edge
    # One Bernoulli draw per unordered edge, mirrored BEFORE encoding.
    # Averaging independently masked categorical vectors leaks half a label.
    i, j = torch.triu_indices(n, n, offset=1, device=flags.device)
    draws = (torch.rand((b, i.numel()), device=flags.device, generator=generator)
             < p[:, None]) | full[:, None]
    edge_mask = torch.zeros((b, n, n), dtype=torch.bool, device=flags.device)
    edge_mask[:, i, j] = draws
    edge_mask[:, j, i] = draws
    edge_mask = (edge_mask & known_current) | (active_pair & ~clean_edge)
    edge_in = torch.zeros((b,n,n,model.edge_input_classes), device=flags.device)
    safe_edges = edge_labels.clamp_min(0)
    edge_in[..., :model.edge_classes] = F.one_hot(safe_edges, model.edge_classes).float() * known_current.unsqueeze(-1)
    edge_in[edge_mask] = 0.0
    edge_in[..., -1] = edge_mask.float()
    return node_in, edge_in, node_mask, edge_mask


def _categorical_denoising_loss(
    out, node_labels, edge_labels, flags, binary, node_mask, edge_mask,
):
    """Supervise only unknown labels. Empty selections give differentiable zero."""
    node_supervised = node_mask & flags.bool() & (node_labels >= 0)
    pairs = flags.bool().unsqueeze(1) & flags.bool().unsqueeze(2)
    upper = torch.triu(torch.ones_like(edge_labels, dtype=torch.bool), diagonal=1)
    # Missing clean edges have zero attribute input and are also legitimate targets.
    edge_supervised = ((edge_labels >= 0) & pairs & upper
                       & (edge_mask | ~binary.bool()))
    def selected_loss(logits, labels, selected):
        if selected.any():
            loss = F.cross_entropy(logits[selected], labels[selected])
            accuracy = (logits[selected].argmax(-1) == labels[selected]).float().mean()
            return loss, float(accuracy.detach())
        return logits.sum() * 0.0, 0.0
    node_ce, node_accuracy = selected_loss(out['node_logits'], node_labels, node_supervised)
    edge_ce, edge_accuracy = selected_loss(out['edge_logits'], edge_labels, edge_supervised)
    return node_ce, edge_ce, {
        'node_accuracy': node_accuracy, 'edge_accuracy': edge_accuracy,
        'node_supervised_count': float(node_supervised.sum()),
        'edge_supervised_count': float(edge_supervised.sum()),
    }


def _graphlet_summary_loss(
    *,
    logits: torch.Tensor,
    mass_logits: torch.Tensor,
    slices: tuple[tuple[int, int], ...],
    histogram_target: torch.Tensor,
    mass_target: torch.Tensor,
    cfg: Mapping[str, Any],
    prefix: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    losses: list[torch.Tensor] = []
    maes: list[torch.Tensor] = []
    for start, stop in slices:
        target = histogram_target[:, start:stop]
        valid = target.sum(dim=-1) > 0.0
        if bool(valid.any()):
            logp = torch.log_softmax(logits[valid, start:stop], dim=-1)
            losses.append(-(target[valid] * logp).sum(dim=-1).mean())
            pred = torch.softmax(logits[valid, start:stop], dim=-1)
            maes.append((pred - target[valid]).abs().mean())
    hist_loss = torch.stack(losses).mean() if losses else logits.sum() * 0.0
    mass_loss = F.binary_cross_entropy_with_logits(mass_logits, mass_target)
    total = (
        float(cfg.get("histogram_weight", 1.0)) * hist_loss
        + float(cfg.get("mass_weight", 0.25)) * mass_loss
    )
    hist_mae = torch.stack(maes).mean() if maes else hist_loss.detach() * 0.0
    return total, {
        f"{prefix}_histogram_loss": float(hist_loss.detach()),
        f"{prefix}_mass_loss": float(mass_loss.detach()),
        f"{prefix}_histogram_mae": float(hist_mae.detach()),
    }


def _topology_structure_loss(
    model: AttributedSpectrumPPGNScore,
    outputs: Mapping[str, torch.Tensor],
    graphlet_target: torch.Tensor,
    mass_target: torch.Tensor,
    clustering_target: torch.Tensor,
    orbit_hist_target: torch.Tensor,
    orbit_total_target: torch.Tensor,
    cfg: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    graphlet_cfg = dict(cfg.get("graphlet", {}) or {})
    clustering_cfg = dict(cfg.get("clustering", {}) or {})
    orbit_cfg = dict(cfg.get("orbit", {}) or {})
    total, metrics = _graphlet_summary_loss(
        logits=outputs["topology_graphlet_logits"],
        mass_logits=outputs["topology_graphlet_mass_logits"],
        slices=model.topology_graphlet_slices,
        histogram_target=graphlet_target,
        mass_target=mass_target,
        cfg=graphlet_cfg,
        prefix="topology_graphlet",
    )
    if bool(clustering_cfg.get("enabled", False)):
        logits = outputs["topology_clustering_logits"]
        logp = torch.log_softmax(logits, dim=-1)
        ce = -(clustering_target * logp).sum(dim=-1).mean()
        pred = torch.softmax(logits, dim=-1)
        cdf = torch.abs(
            torch.cumsum(pred - clustering_target, dim=-1)[..., :-1]
        ).mean()
        total = total + float(clustering_cfg.get("histogram_weight", 0.25)) * (
            ce + float(clustering_cfg.get("cdf_weight", 1.0)) * cdf
        )
        metrics.update({
            "topology_clustering_histogram_loss": float(ce.detach()),
            "topology_clustering_cdf_mae": float(cdf.detach()),
        })
    if bool(orbit_cfg.get("enabled", False)):
        logits = outputs["topology_orbit_histogram_logits"]
        valid = orbit_hist_target.sum(dim=-1) > 0.0
        if bool(valid.any()):
            logp = torch.log_softmax(logits[valid], dim=-1)
            hist_loss = -(orbit_hist_target[valid] * logp).sum(dim=-1).mean()
            hist_mae = (
                torch.softmax(logits[valid], dim=-1) - orbit_hist_target[valid]
            ).abs().mean()
        else:
            hist_loss = logits.sum() * 0.0
            hist_mae = hist_loss.detach() * 0.0
        total_pred = F.softplus(outputs["topology_orbit_log_total_raw"])
        total_loss = F.mse_loss(total_pred, orbit_total_target)
        total = (
            total
            + float(orbit_cfg.get("histogram_weight", 0.25)) * hist_loss
            + float(orbit_cfg.get("log_total_weight", 0.10)) * total_loss
        )
        metrics.update({
            "topology_orbit_histogram_loss": float(hist_loss.detach()),
            "topology_orbit_histogram_mae": float(hist_mae.detach()),
            "topology_orbit_log_total_rmse": float(
                torch.sqrt(total_loss.detach().clamp_min(0.0))
            ),
        })
    return total, metrics


def _typed_graphlet_loss(
    model: AttributedSpectrumPPGNScore,
    outputs: Mapping[str, torch.Tensor],
    graphlet_target: torch.Tensor,
    mass_target: torch.Tensor,
    cfg: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    graphlet_cfg = dict(cfg.get("typed_graphlet", {}) or {})
    return _graphlet_summary_loss(
        logits=outputs["typed_graphlet_logits"],
        mass_logits=outputs["typed_graphlet_mass_logits"],
        slices=model.typed_graphlet_slices,
        histogram_target=graphlet_target,
        mass_target=mass_target,
        cfg=graphlet_cfg,
        prefix="typed_graphlet",
    )


def _loss_batch(
    model_x: GSDMNodeScore,
    model_lam: AttributedSpectrumPPGNScore,
    batch: tuple[torch.Tensor, ...],
    *,
    sde_x,
    sde_lam,
    eps: float,
    eigen_mask_mode: str,
    spectral_transform: Mapping[str, Any],
    topology_summary_cfg: Mapping[str, Any],
    attribute_summary_cfg: Mapping[str, Any],
    attr_cfg: Mapping[str, Any],
    generator: torch.Generator,
) -> tuple[torch.Tensor, dict[str, float]]:
    (
        x0,
        adj0,
        flags,
        _sizes,
        u,
        lam0,
        node_labels,
        edge_labels,
        topology_gh,
        topology_gm,
        topology_ch,
        topology_oh,
        topology_ot,
        typed_gh,
        typed_gm,
    ) = batch
    del adj0
    b = x0.size(0)
    state0 = _operator_eigenvalues_to_spectral_state(
        lam0,
        flags,
        spectral_operator="combinatorial_laplacian",
        spectral_transform=spectral_transform,
    )
    t = torch.rand((b,), device=x0.device, generator=generator) * (1 - eps) + eps
    zx = mask_x(
        torch.randn(x0.shape, device=x0.device, generator=generator), flags
    )
    eigen_mask = eigen_mask_from_flags(flags, eigen_mask_mode)
    zs = torch.randn(
        state0.shape, device=x0.device, generator=generator
    ) * eigen_mask
    mean_x, std_x = sde_x.marginal_x(x0, t)
    xt = mask_x(mean_x + std_x[:, None, None] * zx, flags)
    mean_s, std_s = sde_lam.marginal_spectrum(state0, t)
    st = (mean_s * eigen_mask + std_s[:, None] * zs) * eigen_mask
    lam_t = _spectral_state_to_operator_eigenvalues(
        st,
        flags,
        spectral_operator="combinatorial_laplacian",
        spectral_transform=spectral_transform,
    )
    operator_t = reconstruct_adjacency(u, lam_t)
    adj_t = _operator_to_adjacency_state(
        operator_t, flags, "combinatorial_laplacian"
    )
    pred_x = model_x(xt, adj_t, flags, u, st)
    binary, _, _ = _structural_features_from_state(
        adj_t, flags, model_lam.structural_features
    )

    # Topology branch: categorical labels are structurally absent, not merely
    # hidden.  It receives no atom/bond input channels at all.
    topology_out = model_lam.forward_topology(xt, adj_t, flags, u, st)

    # Attribute branch: condition on the current topology plus partially masked
    # categories.  This branch has disjoint parameters from the topology branch.
    node_in, edge_in, node_mask, edge_mask = _mask_categorical_inputs(
        node_labels,
        edge_labels,
        flags,
        binary,
        t,
        model_lam,
        attr_cfg,
        generator,
    )
    attribute_out = model_lam.forward_attributes(
        xt,
        adj_t,
        flags,
        u,
        st,
        node_attr=node_in,
        edge_attr=edge_in,
    )

    loss_x = 0.5 * (pred_x - zx).square().reshape(b, -1).sum(dim=-1).mean()
    loss_spectrum = 0.5 * (
        ((topology_out["spectrum"] - zs) * eigen_mask)
        .square()
        .reshape(b, -1)
        .sum(dim=-1)
    ).mean()
    topology_loss, metrics = _topology_structure_loss(
        model_lam,
        topology_out,
        topology_gh,
        topology_gm,
        topology_ch,
        topology_oh,
        topology_ot,
        topology_summary_cfg,
    )
    typed_loss, typed_metrics = _typed_graphlet_loss(
        model_lam,
        attribute_out,
        typed_gh,
        typed_gm,
        attribute_summary_cfg,
    )
    metrics.update(typed_metrics)
    node_ce, edge_ce, categorical_metrics = _categorical_denoising_loss(
        attribute_out,
        node_labels,
        edge_labels,
        flags,
        binary,
        node_mask,
        edge_mask,
    )
    metrics.update(categorical_metrics)
    total = (
        loss_x
        + loss_spectrum
        + float(topology_summary_cfg.get("loss_weight", 0.10)) * topology_loss
        + float(attribute_summary_cfg.get("loss_weight", 0.10)) * typed_loss
        + float(attr_cfg.get("node_loss_weight", 1.0)) * node_ce
        + float(attr_cfg.get("edge_loss_weight", 1.0)) * edge_ce
    )
    metrics.update({
        "loss_x": float(loss_x.detach()),
        "loss_spectrum": float(loss_spectrum.detach()),
        "topology_structure_loss": float(topology_loss.detach()),
        "attribute_typed_graphlet_loss": float(typed_loss.detach()),
        "node_ce": float(node_ce.detach()),
        "edge_ce": float(edge_ce.detach()),
    })
    return total, metrics


def _build_models(
    model_cfg: Mapping[str, Any],
    topology_meta: Mapping[str, Any],
    attribute_meta: Mapping[str, Any],
    device: torch.device,
):
    structural_features = copy.deepcopy(
        dict(model_cfg.get("structural_features", {}) or {})
    )
    model_x = GSDMNodeScore(
        max_feat_num=int(model_cfg["max_feat_num"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        depth=int(model_cfg["depth"]),
        structural_features=structural_features,
    ).to(device)
    model_lam = AttributedSpectrumPPGNScore(
        **dict(model_cfg),
        topology_graphlet_slices=tuple(
            tuple(int(v) for v in pair)
            for pair in topology_meta["graphlet_slices"]
        ),
        typed_graphlet_slices=tuple(
            tuple(int(v) for v in pair)
            for pair in attribute_meta["graphlet_slices"]
        ),
        topology_clustering_bins=(
            int(topology_meta["clustering_bins"])
            if topology_meta.get("clustering_enabled") else 0
        ),
        topology_orbit_width=(
            int(topology_meta["orbit_width"])
            if topology_meta.get("orbit_enabled") else 0
        ),
    ).to(device)
    return model_x, model_lam

def _artifacts(wrapper, request: TrainRequest) -> TrainingArtifacts:
    layout=request.run.layout
    return TrainingArtifacts(run_dir=layout.run_dir,checkpoint_path=layout.checkpoints_dir/'gdsm_simple.pt',manifest_path=layout.training_manifest_path,log_path=layout.training_log_path)


def train(wrapper, request: TrainRequest, options: Mapping[str,Any]) -> TrainingArtifacts:
    validate_options(options)
    if request.run.dataset_id not in {'qm9','zinc','attributed'}:
        raise ValueError('Attributed log-gap pipeline is for attributed/molecular datasets')
    layout=request.run.layout; artifacts=_artifacts(wrapper,request); fingerprint=request.dataset.fingerprint()
    if layout.training_manifest_path.is_file() and not request.overwrite:
        old=json.loads(layout.training_manifest_path.read_text())
        if (old.get('categorical_training_contract') == CATEGORICAL_TRAINING_CONTRACT
                and old.get('dataset',{}).get('fingerprint')==fingerprint
                and old.get('options')==_jsonable(options)
                and artifacts.checkpoint_path.is_file()): return artifacts
        raise ArtifactCollisionError('Existing attributed log-gap run differs; choose a new run-id or --overwrite')
    ArtifactLayout.require_available(layout.train_dir,overwrite=request.overwrite)
    _seed_everything(request.run.train_seed); device=_resolve_device(options.get('runtime',{})); started=time.monotonic()
    prep_started=time.monotonic()
    print('[hierarchical-preprocess] loading train/validation molecular graphs...',flush=True)
    train_graphs=_attributed_graphs(request.dataset.split_paths['train']); val_graphs=_attributed_graphs(request.dataset.split_paths['val'])
    print(f'[hierarchical-preprocess] loaded train={len(train_graphs)} val={len(val_graphs)} graphs',flush=True)
    for split,graphs in [('train',train_graphs),('val',val_graphs)]:
        for i,g in enumerate(graphs):
            if not nx.is_connected(g): raise ValueError(f'Attributed Laplacian model requires connected graphs; found {split}[{i}]')
    attr_cfg=dict(options['attributed']); vocab=GraphCategoryVocabulary.from_graphs(train_graphs,attr_cfg)
    # Explicit configured support must match the training vocabulary.
    max_nodes=int(options['model'].get('max_nodes') or max(g.number_of_nodes() for g in train_graphs))
    if max(g.number_of_nodes() for g in val_graphs)>max_nodes: raise ValueError('Validation graph exceeds model.max_nodes')
    mc=_model_config(options,max_nodes,vocab)
    print('[hierarchical-preprocess] building padded Laplacian/eigenbasis tensors...',flush=True)
    stage=time.monotonic()
    train_base=_padded_dataset(train_graphs,max_nodes=max_nodes,max_feat_num=mc['max_feat_num'],spectral_operator='combinatorial_laplacian')
    val_base=_padded_dataset(val_graphs,max_nodes=max_nodes,max_feat_num=mc['max_feat_num'],spectral_operator='combinatorial_laplacian')
    print(f'[hierarchical-preprocess] spectral tensors complete in {time.monotonic()-stage:.1f}s',flush=True)
    train_labels=_attributed_labels(train_graphs,vocab,max_nodes); val_labels=_attributed_labels(val_graphs,vocab,max_nodes)
    pcfg=dict(options.get('preprocess',{}) or {})
    cache_enabled=bool(pcfg.get('cache_structure_targets',True))
    cache_path=_structure_cache_path(layout,fingerprint,options,vocab,max_nodes)
    loaded_cache=False
    if cache_enabled and cache_path.is_file():
        try:
            cached=_torch_load_unrestricted(cache_path)
            if cached.get('format')=='gdsm_hierarchical_attr_structure_cache_v3':
                train_targets=tuple(cached['train_targets'])
                val_targets=tuple(cached['val_targets'])
                topology_basis=cached['topology_basis']
                typed_basis=cached['typed_basis']
                topology_meta=dict(cached['topology_meta'])
                attribute_meta=dict(cached['attribute_meta'])
                val_topology_meta=dict(cached['val_topology_meta'])
                val_attribute_meta=dict(cached['val_attribute_meta'])
                loaded_cache=True
                print(f'[hierarchical-preprocess] loaded cached hierarchical structure targets: {cache_path}',flush=True)
        except Exception as exc:
            print(f'[hierarchical-preprocess] ignoring unreadable cache {cache_path}: {type(exc).__name__}: {exc}',flush=True)
    if not loaded_cache:
        train_targets,topology_basis,typed_basis,topology_meta,attribute_meta=(
            _hierarchical_structure_targets(
                train_graphs,options,vocab,seed=request.run.train_seed
            )
        )
        val_targets,_,_,val_topology_meta,val_attribute_meta=(
            _hierarchical_structure_targets(
                val_graphs,options,vocab,
                topology_basis=topology_basis,
                typed_basis=typed_basis,
                seed=request.run.train_seed+1,
            )
        )
        if cache_enabled:
            cache_path.parent.mkdir(parents=True,exist_ok=True)
            tmp=cache_path.with_suffix(cache_path.suffix+'.tmp')
            torch.save({
                'format':'gdsm_hierarchical_attr_structure_cache_v3',
                'train_targets':tuple(t.cpu() for t in train_targets),
                'val_targets':tuple(t.cpu() for t in val_targets),
                'topology_basis':topology_basis,
                'typed_basis':typed_basis,
                'topology_meta':topology_meta,
                'attribute_meta':attribute_meta,
                'val_topology_meta':val_topology_meta,
                'val_attribute_meta':val_attribute_meta,
            },tmp)
            tmp.replace(cache_path)
            print(f'[hierarchical-preprocess] cached hierarchical structure targets: {cache_path}',flush=True)
    if val_topology_meta['graphlet_slices']!=topology_meta['graphlet_slices']:
        raise AssertionError('Topology graphlet basis mismatch')
    if val_attribute_meta['graphlet_slices']!=attribute_meta['graphlet_slices']:
        raise AssertionError('Typed graphlet basis mismatch')
    print(f'[hierarchical-preprocess] all preprocessing complete in {time.monotonic()-prep_started:.1f}s; starting optimization',flush=True)
    stats=_fit_laplacian_log_gap_stats(train_base[5],train_base[2],epsilon=float(options['sde'].get('log_gap_epsilon',1e-6)),min_std=float(options['sde'].get('log_gap_min_std',1e-3)))
    transform={'kind':'laplacian_log_gap','epsilon':float(options['sde'].get('log_gap_epsilon',1e-6)),'exp_clip':float(options['sde'].get('log_gap_exp_clip',20)),'mean':stats['mean'],'std':stats['std'],'count':stats['count']}
    train_data=(*train_base,*train_labels,*train_targets); val_data=(*val_base,*val_labels,*val_targets)
    train_loader=DataLoader(TensorDataset(*train_data),batch_size=int(options['train']['batch_size']),shuffle=True); val_loader=DataLoader(TensorDataset(*val_data),batch_size=int(options['train']['batch_size']),shuffle=False)
    mx,ml=_build_models(mc,topology_meta,attribute_meta,device); sx,sl=_make_sdes(options,device); tc=dict(options['train'])
    ox=torch.optim.Adam(mx.parameters(),lr=float(tc.get('node_lr',tc['lr'])),weight_decay=float(tc.get('weight_decay',0))); ol=torch.optim.Adam(ml.parameters(),lr=float(tc.get('spectrum_lr',tc['lr'])),weight_decay=float(tc.get('weight_decay',0)))
    sched=[torch.optim.lr_scheduler.ExponentialLR(o,gamma=float(tc.get('lr_decay',1))) for o in (ox,ol)]; ex=_EMA(mx,float(tc.get('ema',.999))); el=_EMA(ml,float(tc.get('ema',.999)))
    history=[]; epochs=int(tc['epochs']); eps=float(options['sde'].get('eps',1e-5)); eigen_mask=str(options['sde'].get('eigen_mask','laplacian_nonzero_prefix'))
    layout.train_dir.parent.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix='.gdsm_attr_loggap_train_',dir=layout.train_dir.parent)); (staging/'checkpoints').mkdir(parents=True,exist_ok=True)
    log_path=staging/'train.log'; gen=torch.Generator(device=device).manual_seed(request.run.train_seed+1729)
    try:
        with log_path.open('w') as log:
            for epoch in range(1,epochs+1):
                mx.train(); ml.train(); sums=defaultdict(float); count=0
                for raw in train_loader:
                    batch=tuple(x.to(device) for x in raw); loss,parts=_loss_batch(
                        mx,ml,batch,sde_x=sx,sde_lam=sl,eps=eps,
                        eigen_mask_mode=eigen_mask,spectral_transform=transform,
                        topology_summary_cfg=_topology_summary_config(options),
                        attribute_summary_cfg=_attribute_summary_config(options),
                        attr_cfg=attr_cfg,generator=gen,
                    )
                    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite attributed training loss')
                    ox.zero_grad(set_to_none=True); ol.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(mx.parameters(),float(tc.get('grad_norm',1))); torch.nn.utils.clip_grad_norm_(ml.parameters(),float(tc.get('grad_norm',1))); ox.step(); ol.step(); ex.update(mx); el.update(ml)
                    sums['loss']+=float(loss.detach())*batch[0].size(0)
                    for k,v in parts.items(): sums[k]+=float(v)*batch[0].size(0)
                    count+=batch[0].size(0)
                row={'epoch':epoch,**{'train_'+k:v/max(count,1) for k,v in sums.items()}}
                if epoch==1 or epoch%int(tc.get('validation_every',1))==0 or epoch==epochs:
                    mx.eval(); ml.eval(); vs=defaultdict(float); vc=0; vg=torch.Generator(device=device).manual_seed(request.run.train_seed+100003+epoch)
                    with torch.no_grad():
                        for raw in val_loader:
                            batch=tuple(x.to(device) for x in raw); loss,parts=_loss_batch(
                                mx,ml,batch,sde_x=sx,sde_lam=sl,eps=eps,
                                eigen_mask_mode=eigen_mask,spectral_transform=transform,
                                topology_summary_cfg=_topology_summary_config(options),
                                attribute_summary_cfg=_attribute_summary_config(options),
                                attr_cfg=attr_cfg,generator=vg,
                            )
                            vs['loss']+=float(loss)*batch[0].size(0)
                            for k,v in parts.items(): vs[k]+=float(v)*batch[0].size(0)
                            vc+=batch[0].size(0)
                    row.update({'val_'+k:v/max(vc,1) for k,v in vs.items()})
                history.append(row)
                if bool(tc.get('lr_schedule',True)):
                    for sc in sched: sc.step()
                if epoch==1 or epoch%int(tc.get('log_every',10))==0 or epoch==epochs:
                    line=f"Hierarchical-attributed-loggap epoch {epoch}/{epochs} train={row['train_loss']:.6f}"+(f" val={row['val_loss']:.6f}" if 'val_loss' in row else '')
                    print(line,flush=True); log.write(line+'\n'); log.flush()
        checkpoint_path=staging/'checkpoints/gdsm_simple.pt'
        reference_contract={
            'factorization':'p(A) p(X,R|A)',
            'topology':'combinatorial_laplacian_log_gap_VP_diffusion',
            'topology_edge_existence':'spectral_decoder_only',
            'topology_auxiliary':'connected_induced_unattributed_graphlets_k3_k4_k5_plus_ORCA_orbits',
            'attribute_auxiliary':'connected_induced_typed_graphlets_k3_k4_k5_training_vocabulary_plus_overflow',
            'gradient_routing':'separate_topology_and_attribute_PPGN_encoders',
            'node_categories':'masked_categorical_denoising_head_no_flow_matching',
            'edge_categories':'masked_categorical_denoising_head_real_edge_types_only_no_no-edge_class',
            'categorical_flow_matching':False,
            'categorical_input':'masked_one_hot_node_and_edge_categories_in_attribute_PPGN',
            'rewiring':False,
            'posthoc_repair':False,
        }
        state={
            'categorical_training_contract':CATEGORICAL_TRAINING_CONTRACT,
            'architecture_contract':'hierarchical_topology_then_attributes_separate_ppgn_v1',
            'format':CHECKPOINT_FORMAT,
            'variant':str(options['variant']),
            'model_x_state':{k:v.detach().cpu() for k,v in mx.state_dict().items()},
            'model_spectrum_state':{k:v.detach().cpu() for k,v in ml.state_dict().items()},
            'ema_x_state':ex.shadow,
            'ema_spectrum_state':el.shadow,
            'model_config':mc,
            'sde':copy.deepcopy(dict(options['sde'])),
            'sample':copy.deepcopy(dict(options['sample'])),
            'spectral_operator':'combinatorial_laplacian',
            'spectral_transform':_jsonable(transform),
            'max_nodes':max_nodes,
            'basis_adjacencies':train_base[1].numpy(),
            'basis_num_nodes':train_base[3].numpy().astype(np.int64),
            'basis_source':'training_split_only',
            'vocabulary':vocab.to_dict(),
            'topology_summary':{
                'enabled':True,
                'training_config':_topology_summary_config(options),
                **topology_meta,
            },
            'attribute_summary':{
                'enabled':True,
                'training_config':_attribute_summary_config(options),
                **attribute_meta,
            },
            'attributed_config':copy.deepcopy(attr_cfg),
            'history':history,
            'train_seed':request.run.train_seed,
            'reference_contract':reference_contract,
        }
        torch.save(state,checkpoint_path)
        resolved=copy.deepcopy(dict(options)); resolved['model']=dict(resolved['model']); resolved['model']['max_nodes']=max_nodes; resolved['model']['max_feat_num']=mc['max_feat_num']; (staging/'resolved_config.yaml').write_text(yaml.safe_dump({wrapper.model_id:resolved},sort_keys=False))
        manifest={
            'categorical_training_contract':CATEGORICAL_TRAINING_CONTRACT,
            'architecture_contract':'hierarchical_topology_then_attributes_separate_ppgn_v1',
            'format':TRAINING_FORMAT,
            'model_id':wrapper.model_id,
            'variant':str(options['variant']),
            'run_id':request.run.run_id,
            'train_seed':request.run.train_seed,
            'created_at':datetime.now(timezone.utc).isoformat(),
            'duration_seconds':time.monotonic()-started,
            'dataset':{
                'benchmark_id':request.dataset.benchmark_id,
                'serialized_id':request.dataset.serialized_id,
                'fingerprint':fingerprint,
                'split_sha256':{k:_sha256(v) for k,v in request.dataset.split_paths.items()},
            },
            'options':_jsonable(options),
            'checkpoint':{'path':'checkpoints/gdsm_simple.pt','sha256':_sha256(checkpoint_path)},
            'checkpoint_selection':{'kind':'final_configured_epoch','epoch':epochs},
            'reference_contract':reference_contract,
            'parameter_counts':{
                'node_diffusion':sum(p.numel() for p in mx.parameters()),
                'topology_branch':sum(p.numel() for p in ml.topology_parameters()),
                'attribute_branch':sum(p.numel() for p in ml.attribute_parameters()),
            },
            'test_used_for_training':False,
        }
        _write_json(staging/'manifest.json',manifest)
        if layout.train_dir.exists(): shutil.rmtree(layout.train_dir)
        staging.replace(layout.train_dir); _write_json(layout.run_manifest_path,{'format':'grapher_baseline_run_v1','model_id':wrapper.model_id,'dataset_id':request.run.dataset_id,'run_id':request.run.run_id,'train_seed':request.run.train_seed})
        return _artifacts(wrapper,request)
    except BaseException:
        shutil.rmtree(staging,ignore_errors=True); raise


def _decode_attributes(
    model: AttributedSpectrumPPGNScore,
    sample_x: torch.Tensor,
    soft: torch.Tensor,
    flags: torch.Tensor,
    u: torch.Tensor,
    state: torch.Tensor,
    discrete: torch.Tensor,
    vocab: GraphCategoryVocabulary,
    cfg: Mapping[str,Any],
    *,
    generator: torch.Generator,
    diagnostics: list[dict[str, Any]] | None = None,
) -> tuple[torch.Tensor,torch.Tensor,dict[str,torch.Tensor]]:
    b,n=flags.shape
    decode=dict(cfg.get('decode',{}) or {})
    validate_decode_config(decode, vocab.node_values, vocab.edge_values,
                           node_attribute=vocab.node_attribute, edge_attribute=vocab.edge_attribute)
    node_mask=torch.zeros((b,n,model.node_input_classes),device=soft.device)
    node_mask[...,-1]=flags
    edge_mask=torch.zeros((b,n,n,model.edge_input_classes),device=soft.device)
    edge_mask[...,-1]=discrete.float()

    # Pass 1: infer atom categories with every categorical state masked.
    out1=model.forward_attributes(
        sample_x,soft,flags,u,state,node_attr=node_mask,edge_attr=edge_mask
    )
    node_idx, atom_masks = sample_atoms(
        out1['node_logits'], discrete, flags, vocab.node_values, decode, generator
    )

    # Pass 2: condition bond logits on the decoded atom categories.  Bond
    # existence remains entirely determined by the spectral topology branch.
    if bool(decode.get('two_pass',True)):
        node_known=torch.zeros_like(node_mask)
        node_known[...,:model.node_classes]=F.one_hot(node_idx,model.node_classes).float()*flags.unsqueeze(-1)
        out=model.forward_attributes(
            sample_x,soft,flags,u,state,node_attr=node_known,edge_attr=edge_mask
        )
    else:
        out=out1

    edge_idx, rows = sample_bonds(
        out['edge_logits'], discrete, flags, node_idx, vocab.node_values,
        vocab.edge_values, decode, generator,
    )
    if diagnostics is not None:
        diagnostics.extend(decoding_diagnostics(
            discrete, flags, node_idx, edge_idx, vocab.node_values, vocab.edge_values,
            atom_masks, rows,
        ))
    return node_idx,edge_idx,out


def _tensor_state_digest(*tensors: torch.Tensor) -> str:
    """Hash complete per-sample decoder inputs, not merely a graph degree sequence."""
    digest = hashlib.sha256()
    for tensor in tensors:
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _generation_rngs(seed: int, device: torch.device, separate: bool = True):
    """Attribute random draws must not advance the topology diffusion stream."""
    topology = torch.Generator(device=device).manual_seed(int(seed))
    attribute_seed = (int(seed) + 104729) % (2**63 - 1)
    attributes = (torch.Generator(device=device).manual_seed(attribute_seed)
                  if separate else topology)
    return np.random.default_rng(seed), topology, attributes, attribute_seed if separate else int(seed)


def generate(
    wrapper, request: GenerateRequest, state: Mapping[str, Any],
    manifest: Mapping[str, Any], options: Mapping[str, Any],
) -> GenerationArtifacts:
    validate_options(options)
    if state.get('format') != CHECKPOINT_FORMAT:
        raise RuntimeError(f"Expected {CHECKPOINT_FORMAT}, found {state.get('format')!r}")
    if state.get('categorical_training_contract') != CATEGORICAL_TRAINING_CONTRACT:
        warnings.warn(
            "This checkpoint predates the leakage-free undirected masking contract. "
            "Decoding remains available for exploratory comparisons; final results "
            "require fresh training with the corrected code and a new run-id.",
            RuntimeWarning, stacklevel=2,
        )
    layout = request.run.layout
    target = layout.generation_dir(request.resolved_generation_id)
    ArtifactLayout.require_available(target, overwrite=request.overwrite)
    if request.num_graphs < 1:
        raise ValueError('num_graphs must be positive')
    batch_size = int(options.get('generation_batch_size', 128))
    if batch_size < 1:
        raise ValueError('generation_batch_size must be positive')
    device = _resolve_device(options.get('runtime', {}))
    _seed_everything(request.generation_seed)
    vocab = GraphCategoryVocabulary.from_dict(dict(state['vocabulary']))
    mx, ml = _build_models(
        dict(state['model_config']),
        dict(state['topology_summary']),
        dict(state['attribute_summary']),
        device,
    )
    use_ema = bool(options.get('sample', {}).get('use_ema', False))
    mx.load_state_dict(state['ema_x_state' if use_ema else 'model_x_state'])
    ml.load_state_dict(state['ema_spectrum_state' if use_ema else 'model_spectrum_state'])
    mx.eval(); ml.eval()
    sx, sl = _make_sdes({'sde': state['sde']}, device)
    sample_cfg = copy.deepcopy(dict(state['sample']))
    sample_cfg.update(dict(options.get('sample', {})))
    separate_rng = bool(sample_cfg.get('separate_attribute_rng', True))
    rng, topology_generator, attribute_generator, attribute_seed = _generation_rngs(
        request.generation_seed, device, separate_rng
    )
    basis_adj = torch.tensor(np.asarray(state['basis_adjacencies']), dtype=torch.float32)
    basis_n = torch.tensor(np.asarray(state['basis_num_nodes']), dtype=torch.long)
    transform = dict(state['spectral_transform'])
    threshold = float(sample_cfg.get('threshold', .5))
    attr_cfg = copy.deepcopy(dict(state['attributed_config']))
    runtime_attr = dict(options.get('attributed', {}) or {})
    if 'decode' in runtime_attr:
        attr_cfg['decode'] = _deep_update(
            dict(attr_cfg.get('decode', {}) or {}), dict(runtime_attr.get('decode', {}) or {})
        )
    decode_cfg = dict(attr_cfg.get('decode', {}) or {})
    validate_decode_config(decode_cfg, vocab.node_values, vocab.edge_values,
                           node_attribute=vocab.node_attribute, edge_attribute=vocab.edge_attribute)
    graphs = []; continuous = []; spectra = []; basis_indices = []
    topology_predictions = []
    typed_predictions = []
    combined_predictions = []
    attr_conf = []; topology_hashes = []
    started = time.monotonic()
    with torch.no_grad():
        while len(graphs) < request.num_graphs:
            b = min(batch_size, request.num_graphs - len(graphs))
            indices = rng.integers(0, len(basis_adj), size=b)
            donor_adj, donor_n = basis_adj[indices], basis_n[indices]
            sample_x, soft, lam, u = sample_batch(
                mx, ml, donor_adjacencies=donor_adj, donor_sizes=donor_n,
                sde_x=sx, sde_lam=sl, sample_cfg=sample_cfg,
                eigen_mask_mode='laplacian_nonzero_prefix',
                spectral_operator='combinatorial_laplacian', spectral_transform=transform,
                device=device, generator=topology_generator,
            )
            soft = 0.5 * (soft + soft.transpose(-1, -2))
            nmax = soft.size(1)
            flags = (torch.arange(nmax, device=device).unsqueeze(0)
                     < donor_n.to(device).unsqueeze(1)).float()
            st = _operator_eigenvalues_to_spectral_state(
                lam, flags, spectral_operator='combinatorial_laplacian', spectral_transform=transform
            )
            eye = torch.eye(nmax, device=device, dtype=torch.bool).unsqueeze(0)
            discrete = ((soft > threshold) & ~eye
                        & flags.bool().unsqueeze(1) & flags.bool().unsqueeze(2))
            discrete = discrete | discrete.transpose(1, 2)
            # Hash BEFORE decoding and check again afterward to detect mutation.
            state_hashes = [_tensor_state_digest(sample_x[r], soft[r], flags[r], u[r], st[r], lam[r], discrete[r])
                            for r in range(b)]
            decode_rows: list[dict[str, Any]] = []
            topology_out = ml.forward_topology(sample_x, soft, flags, u, st)
            node_idx, edge_idx, attribute_out = _decode_attributes(
                ml, sample_x, soft, flags, u, st, discrete, vocab, attr_cfg,
                generator=attribute_generator, diagnostics=decode_rows,
            )
            for r in range(b):
                if state_hashes[r] != _tensor_state_digest(sample_x[r], soft[r], flags[r], u[r], st[r], lam[r], discrete[r]):
                    raise AssertionError('Attribute decoding modified topology-generation inputs')
            topology_summaries = ml.topology_structure_means_from_outputs(topology_out)
            typed_summaries = ml.attribute_structure_means_from_outputs(attribute_out)
            node_prob = torch.softmax(attribute_out['node_logits'], -1).max(-1).values
            edge_prob = torch.softmax(attribute_out['edge_logits'], -1).max(-1).values
            for r, idx in enumerate(indices.tolist()):
                n = int(donor_n[r]); g = nx.Graph()
                for i in range(n):
                    g.add_node(i, **{str(vocab.node_attribute): vocab.node_value(int(node_idx[r, i]))})
                d = discrete[r, :n, :n].cpu().numpy()
                ei = edge_idx[r, :n, :n].cpu().numpy()
                for i, j in zip(*np.nonzero(np.triu(d, 1))):
                    g.add_edge(int(i), int(j), **{str(vocab.edge_attribute): vocab.edge_value(int(ei[i, j]) + 1)})
                if not np.array_equal(nx.to_numpy_array(g, weight=None).astype(bool), d):
                    raise AssertionError('Serialized attributed graph differs from sampled topology')
                row = decode_rows[r]
                row.update({
                    'graph_index': len(graphs), 'num_nodes': n, 'num_edges': g.number_of_edges(),
                    'connected': bool(nx.is_connected(g)),
                    'topology_state_sha256': state_hashes[r],
                    'node_confidence_mean': float(node_prob[r, :n].mean()),
                    'edge_confidence_mean': (float(edge_prob[r, :n, :n][discrete[r, :n, :n]].mean())
                                             if discrete[r, :n, :n].any() else None),
                })
                g.graph['attribute_decoding'] = {
                    'constraint_mode': str(decode_cfg.get('constraint_mode', 'none')),
                    'topology_infeasible': row.get('topology_infeasible'),
                    'capacity_constraints_satisfied': row.get('capacity_constraints_satisfied'),
                    'topology_state_sha256': state_hashes[r],
                    'posthoc_repair': False,
                }
                graphs.append(g)
                matrix = soft[r, :n, :n].cpu().numpy().astype(np.float64)
                np.fill_diagonal(matrix, 0)
                continuous.append(matrix)
                spectra.append(lam[r].cpu().numpy().astype(np.float64))
                basis_indices.append(int(idx)); topology_hashes.append(state_hashes[r]); attr_conf.append(row)
                graph_index = len(graphs) - 1
                topology_row = {
                    'graph_index': graph_index,
                    **{
                        k: v[r].cpu().numpy().astype(np.float64)
                        for k, v in topology_summaries.items()
                    },
                }
                typed_row = {
                    'graph_index': graph_index,
                    **{
                        k: v[r].cpu().numpy().astype(np.float64)
                        for k, v in typed_summaries.items()
                    },
                }
                topology_predictions.append(topology_row)
                typed_predictions.append(typed_row)
                combined_predictions.append({
                    'graph_index': graph_index,
                    **{f'topology_{k}': value for k, value in topology_row.items() if k != 'graph_index'},
                    **{k: value for k, value in typed_row.items() if k != 'graph_index'},
                })
            print(f"Hierarchical-attributed-loggap generated {len(graphs)}/{request.num_graphs}", flush=True)

    aggregate: dict[str, Any] = {
        'num_requested': request.num_graphs, 'num_generated': len(graphs),
        'num_filtered': 0, 'num_topology_resampled': 0,
        'node_confidence_mean': float(np.mean([x['node_confidence_mean'] for x in attr_conf])),
        # Confidence remains unmasked head confidence, not the constrained posterior.
        'confidence_definition': 'untempered_unmasked_head_max_probability',
        **{k: decode_cfg.get(k, v) for k, v in {
            'constraint_mode': 'none', 'node_mode': 'argmax', 'edge_mode': 'argmax',
            'node_temperature': 1.0, 'edge_temperature': 1.0,
            'edge_order': 'random', 'infeasible_policy': 'retain',
        }.items()},
    }
    finite_edge = [x['edge_confidence_mean'] for x in attr_conf if x['edge_confidence_mean'] is not None]
    aggregate['edge_confidence_mean'] = float(np.mean(finite_edge)) if finite_edge else None
    for key in ('topology_infeasible', 'atom_degree_incompatible', 'bond_valence_exceeded',
                'capacity_constraints_satisfied', 'topology_preserved', 'connected'):
        values = [bool(row[key]) for row in attr_conf if key in row]
        aggregate[key + '_count'] = sum(values)
        aggregate[key + '_denominator'] = len(values)
        aggregate[key + '_rate'] = float(np.mean(values)) if values else None
    for key in ('atom_constraint_activations', 'bond_constraint_activations'):
        aggregate[key] = sum(row[key] for row in attr_conf)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.gdsm_attr_loggap_generate_', dir=target.parent))
    try:
        payloads = {
            'base_graphs': graphs,
            'continuous_adjacencies': continuous,
            'sampled_spectra': spectra,
            'sampled_basis_indices': basis_indices,
            'predicted_topology_summaries': topology_predictions,
            'predicted_typed_graphlet_summaries': typed_predictions,
            'predicted_structure_summaries': combined_predictions,
        }
        files = {}
        for name, data in payloads.items():
            path = staging / (name + '.pkl')
            with path.open('wb') as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            files[name] = {'path': path.name, 'sha256': _sha256(path)}
        files['base_graphs']['role'] = 'final_attributed_graphs'
        files['predicted_topology_summaries'].update({
            'topology_only_graphlets': True,
            'orbit_summary': bool(state['topology_summary'].get('orbit_enabled', False)),
        })
        files['predicted_typed_graphlet_summaries']['typed_graphlets'] = True
        files['predicted_structure_summaries']['role'] = 'combined_hierarchical_predictions'
        ap = staging / 'attribute_prediction_diagnostics.json'
        _write_json(ap, {'per_graph': attr_conf, 'aggregate': aggregate})
        hp = staging / 'topology_pairing.json'
        _write_json(hp, {
            'format': 'gdsm_hierarchical_attributed_topology_pairing_v2',
            'hash_inputs': ['sample_x', 'soft_adjacency', 'flags', 'eigenbasis',
                            'spectral_state', 'eigenvalues', 'binary_topology'],
            'per_graph_sha256': topology_hashes,
            'batch_size': batch_size, 'generation_seed': request.generation_seed,
            'checkpoint_sha256': _sha256(request.checkpoint_path),
        })
        files['attribute_prediction_diagnostics'] = {'path': ap.name, 'sha256': _sha256(ap)}
        files['topology_pairing'] = {'path': hp.name, 'sha256': _sha256(hp)}
        _write_json(staging / 'manifest.json', {
            'format': GENERATION_FORMAT, 'model_id': wrapper.model_id, 'variant': state['variant'],
            'run_id': request.run.run_id, 'generation_id': request.resolved_generation_id,
            'generation_seed': request.generation_seed, 'num_requested': request.num_graphs,
            'num_generated': len(graphs), 'num_filtered': 0, 'num_topology_resampled': 0,
            'duration_seconds': time.monotonic() - started, **files,
            'categorical_training_contract': state.get('categorical_training_contract', 'legacy_unverified'),
            'architecture_contract': state.get('architecture_contract'),
            'checkpoint': {'path': str(request.checkpoint_path.resolve()), 'sha256': _sha256(request.checkpoint_path)},
            'sampling': {
                'topology': 'Laplacian_log_gap_reverse_diffusion',
                'topology_auxiliary_supervision': [
                    'topology_only_connected_induced_graphlets_k3_k4_k5',
                    'ORCA_orbit_histogram_and_log_total',
                ],
                'attribute_auxiliary_supervision': 'typed_connected_induced_graphlets_k3_k4_k5',
                'topology_attribute_encoder_sharing': False,
                'node_attributes': ('two_pass_masked_categorical_denoising' if decode_cfg.get('two_pass', True)
                                    else 'one_shot_masked_categorical_denoising'),
                'edge_attributes': 'real_bond_type_head_on_generated_edges_only',
                'edge_head_includes_no_edge': False, 'categorical_flow_matching': False,
                'topology_graphlet_supervision': True,
                'typed_graphlet_supervision': True,
                'node_decode_mode': decode_cfg.get('node_mode', 'argmax'),
                'edge_decode_mode': decode_cfg.get('edge_mode', 'argmax'),
                'node_temperature': float(decode_cfg.get('node_temperature', 1.0)),
                'edge_temperature': float(decode_cfg.get('edge_temperature', 1.0)),
                'constraint_mode': decode_cfg.get('constraint_mode', 'none'),
                'edge_order': decode_cfg.get('edge_order', 'random'),
                'infeasible_policy': decode_cfg.get('infeasible_policy', 'retain'),
                'separate_attribute_rng': separate_rng,
                'topology_seed': request.generation_seed, 'attribute_seed': attribute_seed,
                'generation_batch_size': batch_size,
                'topology_preserved_by_attribute_decoder': True,
                'degree_sequence_prescribed_during_diffusion': False,
                'connectivity_guaranteed': False, 'full_chemical_validity_guaranteed': False,
                'rewiring': False, 'posthoc_repair': False,
            },
        })
        if target.exists():
            shutil.rmtree(target)
        staging.replace(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return GenerationArtifacts(
        run_dir=layout.run_dir, generation_dir=target, graphs_path=target/'base_graphs.pkl',
        manifest_path=target/'manifest.json', num_requested=request.num_graphs,
        num_generated=len(graphs), graphs_sha256=_sha256(target/'base_graphs.pkl'),
    )

"""Attributed Laplacian log-gap GSDM with PPGN typed-structure supervision.

Design contract
---------------
* Topology is generated only by the Laplacian log-gap diffusion branch.
* Node categories and present-edge categories are categorical denoising heads.
* The edge head has *no no-edge class*: it predicts bond type only for pairs
  whose existence is supplied by the topology branch.
* Node/edge categorical inputs are masked/corrupted one-hot vectors injected
  into the PPGN pair tensor.  This is masked categorical denoising, not a
  categorical flow-matching process.
* Typed connected induced graphlets k=3,4,5 are auxiliary targets sharing the
  PPGN encoder with the log-gap spectrum score.  Clustering/orbit summaries
  remain topology summaries.
* No post-generation rewiring or repair is performed.
"""
from __future__ import annotations

import copy
import json
import math
import pickle
import random
import shutil
import tempfile
import time
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

CHECKPOINT_FORMAT = "gdsm_simple_laplacian_loggap_attributed_ppgn_v1"
TRAINING_FORMAT = "grapher_gdsm_simple_laplacian_loggap_attributed_training_v1"
GENERATION_FORMAT = "grapher_gdsm_simple_laplacian_loggap_attributed_generation_v1"
VARIANTS = {
    "vanilla_laplacian_loggap_attributed_ppgn",
    "laplacian_loggap_attributed_ppgn",
    "gsdm_laplacian_loggap_attributed_ppgn",
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
        "variant": "vanilla_laplacian_loggap_attributed_ppgn",
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
        },
        "structure_summary": {
            "enabled": True,
            "loss_weight": 0.10,
            "graphlet": {
                "enabled": True,
                "orders": [3, 4, 5],
                "histogram_weight": 1.0,
                "mass_weight": 0.25,
                "connected_only": True,
                "attributed": True,
                "attributed_backend": "python",
                "max_basis_graphs": 20000,
            },
            "clustering": {
                "enabled": True,
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
                "two_pass": True,
            },
        },
        "generation_batch_size": 128,
        "runtime": {"device": "auto"},
        "graphlet_refinement": {"enabled": False},
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
        raise ValueError("Attributed log-gap pipeline requires vanilla_laplacian_loggap_attributed_ppgn")
    if bool((options.get("graphlet_refinement", {}) or {}).get("enabled", False)):
        raise ValueError("Attributed log-gap baseline intentionally disables rewiring")
    model = dict(options.get("model", {}) or {})
    if str(model.get("node_backbone", "dense_gcn")).lower() != "dense_gcn":
        raise ValueError("Attributed baseline keeps the node-feature denoiser DenseGCN")
    if str(model.get("spectrum_backbone", "ppgn")).lower() != "ppgn":
        raise ValueError("Attributed baseline requires PPGN for spectrum/structure/attribute heads")
    sde = dict(options.get("sde", {}) or {})
    if str(sde.get("spectral_parameterization", "")).lower() != "laplacian_log_gap":
        raise ValueError("Attributed baseline requires Laplacian log-gap spectral parameterization")
    summary = dict(options.get("structure_summary", {}) or {})
    if not bool(summary.get("enabled", False)):
        raise ValueError("Attributed baseline requires structure_summary.enabled=true")
    graphlet = dict(summary.get("graphlet", {}) or {})
    if list(graphlet.get("orders", [])) != [3, 4, 5]:
        raise ValueError("Attributed typed graphlets are fixed to orders [3,4,5]")
    if not bool(graphlet.get("attributed", True)):
        raise ValueError("Attributed baseline requires typed/attributed graphlets")
    orbit = dict(summary.get("orbit", {}) or {})
    if bool(orbit.get("enabled", False)) and int(orbit.get("width", 15)) != 15:
        raise ValueError("Orbit width must be 15")
    attr = dict(options.get("attributed", {}) or {})
    for key in ("node_attribute", "edge_attribute"):
        if not attr.get(key):
            raise ValueError(f"attributed.{key} is required")
    if not list(attr.get("node_categories", [])):
        raise ValueError("attributed.node_categories must be non-empty")
    if not list(attr.get("edge_categories", [])):
        raise ValueError("attributed.edge_categories must be non-empty")
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
    """Shared PPGN for log-gap score, typed summaries, and categorical heads."""

    def __init__(
        self,
        *,
        max_feat_num: int,
        max_nodes: int,
        node_classes: int,
        edge_classes: int,
        graphlet_slices: tuple[tuple[int, int], ...],
        structural_features: Mapping[str, Any] | None,
        clustering_bins: int,
        orbit_width: int,
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
        self.graphlet_slices = tuple((int(a), int(b)) for a, b in graphlet_slices)
        self.structural_features = copy.deepcopy(dict(structural_features or {}))
        rw_cfg = dict(self.structural_features.get("random_walk", {}) or {})
        sp_cfg = dict(self.structural_features.get("shortest_path", {}) or {})
        active = bool(self.structural_features.get("enabled", False))
        self.rw_steps = int(rw_cfg.get("steps", 4)) if active and bool(rw_cfg.get("enabled", False)) else 0
        self.sp_dim = int(sp_cfg.get("max_distance", 5)) + 2 if active and bool(sp_cfg.get("enabled", False)) else 0
        # +1 mask token for nodes and real edge types.  No-edge is represented
        # only by the separate binary topology channel / zero edge attribute vector.
        self.node_input_classes = self.node_classes + 1
        self.edge_input_classes = self.edge_classes + 1
        node_dim = self.max_feat_num + self.rw_steps + self.node_input_classes
        pair_input_dim = 2 + self.sp_dim + node_dim + self.edge_input_classes
        hdim = int(ppgn_hidden_dim)
        self.ppgn_hidden_dim = hdim
        self.pair_input = nn.Linear(pair_input_dim, hdim)
        self.pair_input_norm = nn.LayerNorm(hdim)
        self.ppgn_input_clip = float(ppgn_input_clip)
        self.ppgn_layers = nn.ModuleList([
            PPGNLayer(hdim, residual_scale=float(ppgn_residual_scale), norm_eps=float(ppgn_norm_eps))
            for _ in range(int(ppgn_depth))
        ])
        shared_dim = 2 * hdim + self.max_nodes
        self.spectrum_final = MLP(shared_dim, 2 * max(hdim, self.max_nodes), self.max_nodes, 2)
        width = self.graphlet_slices[-1][1]
        self.graphlet_logits = MLP(shared_dim, 2 * shared_dim, width, 2)
        self.graphlet_mass_logits = MLP(shared_dim, 2 * shared_dim, len(self.graphlet_slices), 2)
        self.clustering_bins = int(clustering_bins)
        self.orbit_width = int(orbit_width)
        self.clustering_logits = MLP(shared_dim, 2 * shared_dim, self.clustering_bins, 2) if self.clustering_bins > 0 else None
        self.orbit_histogram_logits = MLP(shared_dim, 2 * shared_dim, self.orbit_width, 2) if self.orbit_width > 0 else None
        self.orbit_log_total_raw = MLP(shared_dim, 2 * shared_dim, 1, 2) if self.orbit_width > 0 else None
        self.node_category_head = MLP(hdim, 2 * hdim, self.node_classes, 2)
        self.edge_category_head = MLP(hdim, 2 * hdim, self.edge_classes, 2)

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
            node_attr = torch.zeros((b, n, self.node_input_classes), dtype=flags.dtype, device=flags.device)
            node_attr[..., -1] = flags.to(node_attr.dtype)
        if edge_attr is None:
            edge_attr = torch.zeros((b, n, n, self.edge_input_classes), dtype=flags.dtype, device=flags.device)
            edge_attr[..., -1] = binary.to(edge_attr.dtype)
        return node_attr, edge_attr

    def _encode(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        flags: torch.Tensor,
        eigenvalues: torch.Tensor,
        *,
        node_attr: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        binary, rw, sp = _structural_features_from_state(adj, flags, self.structural_features)
        node_attr, edge_attr = self.masked_attribute_inputs(flags, binary, node_attr=node_attr, edge_attr=edge_attr)
        node = torch.cat((x, rw, node_attr), dim=-1) if rw.size(-1) else torch.cat((x, node_attr), dim=-1)
        b, n, _ = x.shape
        diag_node = x.new_zeros((b, n, n, node.size(-1)))
        idx = torch.arange(n, device=x.device)
        diag_node[:, idx, idx, :] = node
        bounded_adj = torch.tanh(torch.nan_to_num(adj, nan=0.0, posinf=self.ppgn_input_clip, neginf=-self.ppgn_input_clip).clamp(-self.ppgn_input_clip, self.ppgn_input_clip))
        parts = [bounded_adj.unsqueeze(-1), binary.unsqueeze(-1)]
        if sp.size(-1):
            parts.append(sp)
        parts.extend([diag_node, edge_attr])
        pair = torch.cat(parts, dim=-1)
        pair_mask = flags.to(pair.dtype).unsqueeze(-1) * flags.to(pair.dtype).unsqueeze(-2)
        h = F.elu(self.pair_input_norm(self.pair_input(pair))) * pair_mask.unsqueeze(-1)
        for layer in self.ppgn_layers:
            h = layer(h, pair_mask, flags)
        diag = h[:, idx, idx, :]
        count = flags.sum(dim=-1, keepdim=True).clamp_min(1.0).to(h.dtype)
        diag_pool = (diag * flags.unsqueeze(-1).to(h.dtype)).sum(dim=1) / count
        diag_mask = torch.eye(n, dtype=h.dtype, device=h.device).unsqueeze(0)
        off_mask = pair_mask * (1.0 - diag_mask)
        off_count = off_mask.sum(dim=(1, 2)).unsqueeze(-1).clamp_min(1.0)
        off_pool = (h * off_mask.unsqueeze(-1)).sum(dim=(1, 2)) / off_count
        shared = torch.cat((diag_pool, off_pool, eigenvalues), dim=-1)
        return shared, h, binary

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
        del eigenvectors
        shared, pair_h, _ = self._encode(x, adj, flags, eigenvalues, node_attr=node_attr, edge_attr=edge_attr)
        n = pair_h.size(1)
        idx = torch.arange(n, device=pair_h.device)
        diag = pair_h[:, idx, idx, :]
        sym_pair = 0.5 * (pair_h + pair_h.transpose(1, 2))
        outputs = {
            "spectrum": self.spectrum_final(shared),
            "graphlet_logits": self.graphlet_logits(shared),
            "graphlet_mass_logits": self.graphlet_mass_logits(shared),
            "node_logits": self.node_category_head(diag),
            "edge_logits": self.edge_category_head(sym_pair),
        }
        if self.clustering_logits is not None:
            outputs["clustering_logits"] = self.clustering_logits(shared)
        if self.orbit_histogram_logits is not None:
            outputs["orbit_histogram_logits"] = self.orbit_histogram_logits(shared)
            outputs["orbit_log_total_raw"] = self.orbit_log_total_raw(shared)
        for name, value in outputs.items():
            if not torch.isfinite(value).all():
                raise FloatingPointError(f"Non-finite {name} from attributed PPGN")
        return outputs

    def forward(self, x, adj, flags, eigenvectors, eigenvalues):
        # Reverse topology sampling has no categorical trajectory.  Unknown
        # attributes are represented by MASK tokens; topology remains solely
        # controlled by the Laplacian branch.
        return self.forward_all(x, adj, flags, eigenvectors, eigenvalues)["spectrum"]

    def structure_means_from_outputs(self, outputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        logits = outputs["graphlet_logits"]
        hist = torch.zeros_like(logits)
        for start, stop in self.graphlet_slices:
            hist[:, start:stop] = torch.softmax(logits[:, start:stop], dim=-1)
        result = {
            "graphlet_histogram": hist,
            "graphlet_mass": torch.sigmoid(outputs["graphlet_mass_logits"]),
        }
        if "clustering_logits" in outputs:
            result["clustering_histogram"] = torch.softmax(outputs["clustering_logits"], dim=-1)
        if "orbit_histogram_logits" in outputs:
            oh = torch.softmax(outputs["orbit_histogram_logits"], dim=-1)
            ot = F.softplus(outputs["orbit_log_total_raw"])
            result["orbit_histogram"] = oh
            result["orbit_log_total"] = ot
            result["orbit_mean_counts"] = oh * torch.expm1(ot)
        return result


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


def _summary_config(options: Mapping[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(dict(options.get("structure_summary", {}) or {}))


def _typed_structure_targets(
    graphs: Sequence[nx.Graph],
    options: Mapping[str, Any],
    vocab: GraphCategoryVocabulary,
    *,
    basis: GraphletBasis | None = None,
    seed: int = 0,
) -> tuple[tuple[torch.Tensor, ...], GraphletBasis, dict[str, Any]]:
    summary = _summary_config(options)
    gcfg = dict(summary.get("graphlet", {}) or {})
    ccfg = dict(summary.get("clustering", {}) or {})
    ocfg = dict(summary.get("orbit", {}) or {})
    scfg_dict = {
        "clustering_summary": False,
        "spectral_summary": False,
        "motif_proxy": False,
        "orbit_count": False,
        "graphlet_history": True,
        "graphlet_k_min": 3,
        "graphlet_k_max": 5,
        "graphlet_connected_only": bool(gcfg.get("connected_only", True)),
        "graphlet_topology_filter": "all",
        "graphlet_backend": "exact",
        "graphlet_num_samples": None,
        "attributed": True,
        "node_attribute": vocab.node_attribute,
        "edge_attribute": vocab.edge_attribute,
        "attributed_backend": str(gcfg.get("attributed_backend", "python")),
    }
    scfg = SummaryConfig.from_dict(scfg_dict)
    if basis is None:
        max_basis = gcfg.get("max_basis_graphs", None)
        basis_graphs = list(graphs) if max_basis is None else list(graphs)[: int(max_basis)]
        basis = GraphletBasis.fit_from_graphs(
            basis_graphs,
            scfg_dict,
            vocabulary=vocab,
            attributed=True,
            seed=seed,
        )
    histograms: list[np.ndarray] = []
    masses: list[np.ndarray] = []
    clusterings: list[np.ndarray] = []
    orbit_hist: list[np.ndarray] = []
    orbit_total: list[np.ndarray] = []
    rng = np.random.default_rng(seed + 333)
    bins = int(ccfg.get("bins", 100))
    width = int(ocfg.get("width", 15))
    for graph in graphs:
        history, mass = basis.statistics_for_graph(graph, scfg, rng=rng)
        histograms.append(basis.flatten_history(history).astype(np.float32))
        masses.append(basis.flatten_mass(mass).astype(np.float32))
        if bool(ccfg.get("enabled", False)):
            clusterings.append(clustering_histogram(graph, bins=bins).astype(np.float32))
        else:
            clusterings.append(np.zeros(bins, np.float32))
        if bool(ocfg.get("enabled", False)):
            counts = np.maximum(np.asarray(python_orbit_count_vector(graph), dtype=np.float64).reshape(-1), 0.0)
            if counts.size != width:
                raise ValueError(f"Expected orbit width {width}, got {counts.size}")
            total = float(counts.sum())
            orbit_hist.append((counts / total if total > 0 else np.zeros_like(counts)).astype(np.float32))
            orbit_total.append(np.asarray([np.log1p(total)], np.float32))
        else:
            orbit_hist.append(np.zeros(width, np.float32)); orbit_total.append(np.zeros(1, np.float32))
    meta = {
        "graphlet_slices": [list(x) for x in basis.slices],
        "graphlet_basis": basis.to_dict(),
        "summary_config": asdict(scfg),
        "width": basis.width,
        "orders": [3,4,5],
        "typed_graphlets": True,
        "graphlet_vocabulary": "training_only_plus_overflow",
        "clustering_enabled": bool(ccfg.get("enabled", False)),
        "clustering_bins": bins,
        "orbit_enabled": bool(ocfg.get("enabled", False)),
        "orbit_width": width,
    }
    tensors = (
        torch.tensor(np.stack(histograms), dtype=torch.float32),
        torch.tensor(np.stack(masses), dtype=torch.float32),
        torch.tensor(np.stack(clusterings), dtype=torch.float32),
        torch.tensor(np.stack(orbit_hist), dtype=torch.float32),
        torch.tensor(np.stack(orbit_total), dtype=torch.float32),
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
    active_pair = current_binary.bool() & (~torch.eye(n, device=flags.device, dtype=torch.bool).unsqueeze(0))
    # Categorical edge information is attached only to the *current* topology.
    # Current false-positive edges receive MASK, never a no-edge category.
    known_current = active_pair & clean_edge
    edge_mask = (torch.rand((b,n,n), device=flags.device, generator=generator) < p[:,None,None]) | full[:,None,None]
    edge_mask = (edge_mask & known_current) | (active_pair & ~clean_edge)
    edge_in = torch.zeros((b,n,n,model.edge_input_classes), device=flags.device)
    safe_edges = edge_labels.clamp_min(0)
    edge_in[..., :model.edge_classes] = F.one_hot(safe_edges, model.edge_classes).float() * known_current.unsqueeze(-1)
    edge_in[edge_mask] = 0.0
    edge_in[..., -1] = edge_mask.float()
    # Symmetric corruption/input.
    edge_in = 0.5 * (edge_in + edge_in.transpose(1,2))
    edge_mask = edge_mask | edge_mask.transpose(1,2)
    return node_in, edge_in, node_mask, edge_mask


def _structure_loss(
    model: AttributedSpectrumPPGNScore,
    outputs: Mapping[str, torch.Tensor],
    graphlet_target: torch.Tensor,
    mass_target: torch.Tensor,
    clustering_target: torch.Tensor,
    orbit_hist_target: torch.Tensor,
    orbit_total_target: torch.Tensor,
    cfg: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str,float]]:
    gcfg = dict(cfg.get("graphlet", {}) or {}); ccfg=dict(cfg.get("clustering",{}) or {}); ocfg=dict(cfg.get("orbit",{}) or {})
    losses=[]; maes=[]
    for start,stop in model.graphlet_slices:
        target=graphlet_target[:,start:stop]; valid=target.sum(-1)>0
        if valid.any():
            logp=torch.log_softmax(outputs['graphlet_logits'][valid,start:stop],-1)
            losses.append(-(target[valid]*logp).sum(-1).mean())
            maes.append((torch.softmax(outputs['graphlet_logits'][valid,start:stop],-1)-target[valid]).abs().mean())
    hist_loss=torch.stack(losses).mean() if losses else outputs['graphlet_logits'].sum()*0
    mass_loss=F.binary_cross_entropy_with_logits(outputs['graphlet_mass_logits'],mass_target)
    total=float(gcfg.get('histogram_weight',1))*hist_loss+float(gcfg.get('mass_weight',.25))*mass_loss
    metrics={'typed_graphlet_histogram_loss':float(hist_loss.detach()),'typed_graphlet_mass_loss':float(mass_loss.detach()),'typed_graphlet_histogram_mae':float((torch.stack(maes).mean().detach() if maes else hist_loss.detach()*0))}
    if bool(ccfg.get('enabled',False)):
        logp=torch.log_softmax(outputs['clustering_logits'],-1); ce=-(clustering_target*logp).sum(-1).mean(); pred=torch.softmax(outputs['clustering_logits'],-1)
        cdf=torch.abs(torch.cumsum(pred-clustering_target,-1)[...,:-1]).mean(); total=total+float(ccfg.get('histogram_weight',.25))*(ce+float(ccfg.get('cdf_weight',1))*cdf)
        metrics.update({'clustering_histogram_loss':float(ce.detach()),'clustering_cdf_mae':float(cdf.detach())})
    if bool(ocfg.get('enabled',False)):
        valid=orbit_hist_target.sum(-1)>0
        if valid.any():
            logp=torch.log_softmax(outputs['orbit_histogram_logits'][valid],-1); oh=-(orbit_hist_target[valid]*logp).sum(-1).mean(); omae=(torch.softmax(outputs['orbit_histogram_logits'][valid],-1)-orbit_hist_target[valid]).abs().mean()
        else:
            oh=outputs['orbit_histogram_logits'].sum()*0; omae=oh.detach()*0
        pred_total=F.softplus(outputs['orbit_log_total_raw']); ot=F.mse_loss(pred_total,orbit_total_target)
        total=total+float(ocfg.get('histogram_weight',.25))*oh+float(ocfg.get('log_total_weight',.1))*ot
        metrics.update({'orbit_histogram_loss':float(oh.detach()),'orbit_histogram_mae':float(omae.detach()),'orbit_log_total_rmse':float(torch.sqrt(ot.detach().clamp_min(0)))})
    return total,metrics


def _loss_batch(
    model_x: GSDMNodeScore,
    model_lam: AttributedSpectrumPPGNScore,
    batch: tuple[torch.Tensor,...],
    *, sde_x, sde_lam, eps: float, eigen_mask_mode: str,
    spectral_transform: Mapping[str,Any], structure_cfg: Mapping[str,Any], attr_cfg: Mapping[str,Any],
    generator: torch.Generator,
) -> tuple[torch.Tensor, dict[str,float]]:
    (x0,adj0,flags,_sizes,u,lam0,node_labels,edge_labels,gh,gm,ch,oh,ot)=batch
    b=x0.size(0)
    state0=_operator_eigenvalues_to_spectral_state(lam0,flags,spectral_operator='combinatorial_laplacian',spectral_transform=spectral_transform)
    t=torch.rand((b,),device=x0.device,generator=generator)*(1-eps)+eps
    zx=mask_x(torch.randn(x0.shape,device=x0.device,generator=generator),flags)
    em=eigen_mask_from_flags(flags,eigen_mask_mode)
    zs=torch.randn(state0.shape,device=x0.device,generator=generator)*em
    mx,sx=sde_x.marginal_x(x0,t); xt=mask_x(mx+sx[:,None,None]*zx,flags)
    ms,ss=sde_lam.marginal_spectrum(state0,t); st=(ms*em+ss[:,None]*zs)*em
    lamt=_spectral_state_to_operator_eigenvalues(st,flags,spectral_operator='combinatorial_laplacian',spectral_transform=spectral_transform)
    op=reconstruct_adjacency(u,lamt); adjt=_operator_to_adjacency_state(op,flags,'combinatorial_laplacian')
    predx=model_x(xt,adjt,flags,u,st)
    binary,_,_=_structural_features_from_state(adjt,flags,model_lam.structural_features)
    # The topology reverse process has no categorical trajectory, so train the
    # Laplacian score under exactly the same condition used at sampling time:
    # active nodes / current edges carry MASK tokens, never clean categories.
    # A second shared-parameter forward pass receives corrupted categorical
    # one-hot inputs for node/edge denoising and typed structural supervision.
    out_topology=model_lam.forward_all(xt,adjt,flags,u,st,node_attr=None,edge_attr=None)
    node_in,edge_in,node_mask,edge_mask=_mask_categorical_inputs(node_labels,edge_labels,flags,binary,t,model_lam,attr_cfg,generator)
    out=model_lam.forward_all(xt,adjt,flags,u,st,node_attr=node_in,edge_attr=edge_in)
    lx=0.5*(predx-zx).square().reshape(b,-1).sum(-1).mean(); ls=0.5*(((out_topology['spectrum']-zs)*em).square().reshape(b,-1).sum(-1)).mean()
    struct,metrics=_structure_loss(model_lam,out,gh,gm,ch,oh,ot,structure_cfg)
    node_valid=flags.bool()&(node_labels>=0); node_supervised=node_mask & node_valid
    if not node_supervised.any(): node_supervised=node_valid
    node_ce=F.cross_entropy(out['node_logits'][node_supervised],node_labels[node_supervised])
    clean_edge=(edge_labels>=0)
    upper=torch.triu(torch.ones_like(clean_edge[0],dtype=torch.bool),diagonal=1).unsqueeze(0)
    # Never supervise the edge head on a pair whose clean bond category is
    # visible verbatim in the PPGN input.  Clean edges that are absent from
    # the noisy/current topology are still supervised: their attribute input
    # is zero (unknown because the topology branch currently says no edge),
    # so there is no label leakage.
    edge_supervised=clean_edge & upper & (edge_mask | ~binary.bool())
    if not edge_supervised.any():
        edge_supervised=clean_edge & upper
    edge_ce=F.cross_entropy(out['edge_logits'][edge_supervised],edge_labels[edge_supervised]) if edge_supervised.any() else out['edge_logits'].sum()*0
    node_acc=(out['node_logits'][node_supervised].argmax(-1)==node_labels[node_supervised]).float().mean()
    edge_acc=(out['edge_logits'][edge_supervised].argmax(-1)==edge_labels[edge_supervised]).float().mean() if edge_supervised.any() else edge_ce.detach()*0
    total=lx+ls+float(structure_cfg.get('loss_weight',.1))*struct+float(attr_cfg.get('node_loss_weight',1))*node_ce+float(attr_cfg.get('edge_loss_weight',1))*edge_ce
    metrics.update({'loss_x':float(lx.detach()),'loss_spectrum':float(ls.detach()),'structure_loss':float(struct.detach()),'node_ce':float(node_ce.detach()),'edge_ce':float(edge_ce.detach()),'node_accuracy':float(node_acc.detach()),'edge_accuracy':float(edge_acc.detach())})
    return total,metrics


def _build_models(model_cfg: Mapping[str,Any], summary_meta: Mapping[str,Any], device: torch.device):
    sf=copy.deepcopy(dict(model_cfg.get('structural_features',{}) or {}))
    mx=GSDMNodeScore(max_feat_num=int(model_cfg['max_feat_num']),hidden_dim=int(model_cfg['hidden_dim']),depth=int(model_cfg['depth']),structural_features=sf).to(device)
    ml=AttributedSpectrumPPGNScore(
        **dict(model_cfg),
        graphlet_slices=tuple(tuple(int(v) for v in x) for x in summary_meta['graphlet_slices']),
        clustering_bins=int(summary_meta['clustering_bins']) if summary_meta.get('clustering_enabled') else 0,
        orbit_width=int(summary_meta['orbit_width']) if summary_meta.get('orbit_enabled') else 0,
    ).to(device)
    return mx,ml


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
        if old.get('dataset',{}).get('fingerprint')==fingerprint and old.get('options')==_jsonable(options) and artifacts.checkpoint_path.is_file(): return artifacts
        raise ArtifactCollisionError('Existing attributed log-gap run differs; choose a new run-id or --overwrite')
    ArtifactLayout.require_available(layout.train_dir,overwrite=request.overwrite)
    _seed_everything(request.run.train_seed); device=_resolve_device(options.get('runtime',{})); started=time.monotonic()
    train_graphs=_attributed_graphs(request.dataset.split_paths['train']); val_graphs=_attributed_graphs(request.dataset.split_paths['val'])
    for split,graphs in [('train',train_graphs),('val',val_graphs)]:
        for i,g in enumerate(graphs):
            if not nx.is_connected(g): raise ValueError(f'Attributed Laplacian model requires connected graphs; found {split}[{i}]')
    attr_cfg=dict(options['attributed']); vocab=GraphCategoryVocabulary.from_graphs(train_graphs,attr_cfg)
    # Explicit configured support must match the training vocabulary.
    max_nodes=int(options['model'].get('max_nodes') or max(g.number_of_nodes() for g in train_graphs))
    if max(g.number_of_nodes() for g in val_graphs)>max_nodes: raise ValueError('Validation graph exceeds model.max_nodes')
    mc=_model_config(options,max_nodes,vocab)
    train_base=_padded_dataset(train_graphs,max_nodes=max_nodes,max_feat_num=mc['max_feat_num'],spectral_operator='combinatorial_laplacian')
    val_base=_padded_dataset(val_graphs,max_nodes=max_nodes,max_feat_num=mc['max_feat_num'],spectral_operator='combinatorial_laplacian')
    train_labels=_attributed_labels(train_graphs,vocab,max_nodes); val_labels=_attributed_labels(val_graphs,vocab,max_nodes)
    train_targets,basis,meta=_typed_structure_targets(train_graphs,options,vocab,seed=request.run.train_seed)
    val_targets,_,val_meta=_typed_structure_targets(val_graphs,options,vocab,basis=basis,seed=request.run.train_seed+1)
    if val_meta['graphlet_slices']!=meta['graphlet_slices']: raise AssertionError('Typed graphlet basis mismatch')
    stats=_fit_laplacian_log_gap_stats(train_base[5],train_base[2],epsilon=float(options['sde'].get('log_gap_epsilon',1e-6)),min_std=float(options['sde'].get('log_gap_min_std',1e-3)))
    transform={'kind':'laplacian_log_gap','epsilon':float(options['sde'].get('log_gap_epsilon',1e-6)),'exp_clip':float(options['sde'].get('log_gap_exp_clip',20)),'mean':stats['mean'],'std':stats['std'],'count':stats['count']}
    train_data=(*train_base,*train_labels,*train_targets); val_data=(*val_base,*val_labels,*val_targets)
    train_loader=DataLoader(TensorDataset(*train_data),batch_size=int(options['train']['batch_size']),shuffle=True); val_loader=DataLoader(TensorDataset(*val_data),batch_size=int(options['train']['batch_size']),shuffle=False)
    mx,ml=_build_models(mc,meta,device); sx,sl=_make_sdes(options,device); tc=dict(options['train'])
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
                    batch=tuple(x.to(device) for x in raw); loss,parts=_loss_batch(mx,ml,batch,sde_x=sx,sde_lam=sl,eps=eps,eigen_mask_mode=eigen_mask,spectral_transform=transform,structure_cfg=_summary_config(options),attr_cfg=attr_cfg,generator=gen)
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
                            batch=tuple(x.to(device) for x in raw); loss,parts=_loss_batch(mx,ml,batch,sde_x=sx,sde_lam=sl,eps=eps,eigen_mask_mode=eigen_mask,spectral_transform=transform,structure_cfg=_summary_config(options),attr_cfg=attr_cfg,generator=vg)
                            vs['loss']+=float(loss)*batch[0].size(0)
                            for k,v in parts.items(): vs[k]+=float(v)*batch[0].size(0)
                            vc+=batch[0].size(0)
                    row.update({'val_'+k:v/max(vc,1) for k,v in vs.items()})
                history.append(row)
                if bool(tc.get('lr_schedule',True)):
                    for sc in sched: sc.step()
                if epoch==1 or epoch%int(tc.get('log_every',10))==0 or epoch==epochs:
                    line=f"Attributed-loggap epoch {epoch}/{epochs} train={row['train_loss']:.6f}"+(f" val={row['val_loss']:.6f}" if 'val_loss' in row else '')
                    print(line,flush=True); log.write(line+'\n'); log.flush()
        checkpoint_path=staging/'checkpoints/gdsm_simple.pt'
        state={
            'format':CHECKPOINT_FORMAT,'variant':str(options['variant']),'model_x_state':{k:v.detach().cpu() for k,v in mx.state_dict().items()},'model_spectrum_state':{k:v.detach().cpu() for k,v in ml.state_dict().items()},'ema_x_state':ex.shadow,'ema_spectrum_state':el.shadow,
            'model_config':mc,'sde':copy.deepcopy(dict(options['sde'])),'sample':copy.deepcopy(dict(options['sample'])),'spectral_operator':'combinatorial_laplacian','spectral_transform':_jsonable(transform),'max_nodes':max_nodes,'basis_adjacencies':train_base[1].numpy(),'basis_num_nodes':train_base[3].numpy().astype(np.int64),'basis_source':'training_split_only','vocabulary':vocab.to_dict(),'structure_summary':{'enabled':True,'training_config':_summary_config(options),**meta},'attributed_config':copy.deepcopy(attr_cfg),'history':history,'train_seed':request.run.train_seed,
        }
        torch.save(state,checkpoint_path)
        resolved=copy.deepcopy(dict(options)); resolved['model']=dict(resolved['model']); resolved['model']['max_nodes']=max_nodes; resolved['model']['max_feat_num']=mc['max_feat_num']; (staging/'resolved_config.yaml').write_text(yaml.safe_dump({wrapper.model_id:resolved},sort_keys=False))
        manifest={'format':TRAINING_FORMAT,'model_id':wrapper.model_id,'variant':str(options['variant']),'run_id':request.run.run_id,'train_seed':request.run.train_seed,'created_at':datetime.now(timezone.utc).isoformat(),'duration_seconds':time.monotonic()-started,'dataset':{'benchmark_id':request.dataset.benchmark_id,'serialized_id':request.dataset.serialized_id,'fingerprint':fingerprint,'split_sha256':{k:_sha256(v) for k,v in request.dataset.split_paths.items()}},'options':_jsonable(options),'checkpoint':{'path':'checkpoints/gdsm_simple.pt','sha256':_sha256(checkpoint_path)},'checkpoint_selection':{'kind':'final_configured_epoch','epoch':epochs},'reference_contract':{'topology':'combinatorial_laplacian_log_gap_VP_diffusion','topology_edge_existence':'spectral_decoder_only','node_categories':'masked_categorical_denoising_head_no_flow_matching','edge_categories':'masked_categorical_denoising_head_real_edge_types_only_no_no-edge_class','categorical_flow_matching':False,'categorical_input':'masked_one_hot_node_and_edge_categories_in_PPGN','typed_graphlets':'connected_induced_attributed_k3_k4_k5_training_vocabulary_plus_overflow','rewiring':False,'posthoc_repair':False},'test_used_for_training':False}
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
) -> tuple[torch.Tensor,torch.Tensor,dict[str,torch.Tensor]]:
    b,n=flags.shape; node_mask=torch.zeros((b,n,model.node_input_classes),device=soft.device); node_mask[...,-1]=flags
    edge_mask=torch.zeros((b,n,n,model.edge_input_classes),device=soft.device); edge_mask[...,-1]=discrete.float()
    out1=model.forward_all(sample_x,soft,flags,u,state,node_attr=node_mask,edge_attr=edge_mask)
    node_idx=out1['node_logits'].argmax(-1)
    if bool(dict(cfg.get('decode',{}) or {}).get('two_pass',True)):
        node_known=torch.zeros_like(node_mask); node_known[...,:model.node_classes]=F.one_hot(node_idx,model.node_classes).float()*flags.unsqueeze(-1)
        out=model.forward_all(sample_x,soft,flags,u,state,node_attr=node_known,edge_attr=edge_mask)
    else: out=out1
    edge_idx=out['edge_logits'].argmax(-1)
    return node_idx,edge_idx,out


def generate(wrapper, request: GenerateRequest, state: Mapping[str,Any], manifest: Mapping[str,Any], options: Mapping[str,Any]) -> GenerationArtifacts:
    validate_options(options)
    if state.get('format')!=CHECKPOINT_FORMAT: raise RuntimeError(f"Expected {CHECKPOINT_FORMAT}, found {state.get('format')!r}")
    device=_resolve_device(options.get('runtime',{})); _seed_everything(request.generation_seed); vocab=GraphCategoryVocabulary.from_dict(dict(state['vocabulary'])); meta=dict(state['structure_summary']); mc=dict(state['model_config']); mx,ml=_build_models(mc,meta,device)
    if bool(options.get('sample',{}).get('use_ema',False)): mx.load_state_dict(state['ema_x_state']); ml.load_state_dict(state['ema_spectrum_state'])
    else: mx.load_state_dict(state['model_x_state']); ml.load_state_dict(state['model_spectrum_state'])
    mx.eval(); ml.eval(); sx,sl=_make_sdes({'sde':state['sde']},device); sample_cfg=copy.deepcopy(dict(state['sample'])); sample_cfg.update(dict(options.get('sample',{}))); basis_adj=torch.tensor(np.asarray(state['basis_adjacencies']),dtype=torch.float32); basis_n=torch.tensor(np.asarray(state['basis_num_nodes']),dtype=torch.long)
    rng=np.random.default_rng(request.generation_seed); generator=torch.Generator(device=device).manual_seed(request.generation_seed); batch_size=int(options.get('generation_batch_size',128)); transform=dict(state['spectral_transform']); threshold=float(sample_cfg.get('threshold',.5)); attr_cfg=dict(state['attributed_config'])
    graphs=[]; continuous=[]; spectra=[]; basis_indices=[]; predictions=[]; attr_conf=[]; started=time.monotonic()
    with torch.no_grad():
        while len(graphs)<request.num_graphs:
            b=min(batch_size,request.num_graphs-len(graphs)); indices=rng.integers(0,len(basis_adj),size=b); donor_adj=basis_adj[indices]; donor_n=basis_n[indices]
            sample_x,soft,lam,u=sample_batch(mx,ml,donor_adjacencies=donor_adj,donor_sizes=donor_n,sde_x=sx,sde_lam=sl,sample_cfg=sample_cfg,eigen_mask_mode='laplacian_nonzero_prefix',spectral_operator='combinatorial_laplacian',spectral_transform=transform,device=device,generator=generator)
            soft=0.5*(soft+soft.transpose(-1,-2)); nmax=soft.size(1); flags=(torch.arange(nmax,device=device).unsqueeze(0)<donor_n.to(device).unsqueeze(1)).float(); st=_operator_eigenvalues_to_spectral_state(lam,flags,spectral_operator='combinatorial_laplacian',spectral_transform=transform)
            discrete=(soft>threshold); eye=torch.eye(nmax,device=device,dtype=torch.bool).unsqueeze(0); discrete=discrete & ~eye & flags.bool().unsqueeze(1)&flags.bool().unsqueeze(2); discrete=discrete|discrete.transpose(1,2)
            node_idx,edge_idx,out=_decode_attributes(ml,sample_x,soft,flags,u,st,discrete,vocab,attr_cfg); summaries=ml.structure_means_from_outputs(out); node_prob=torch.softmax(out['node_logits'],-1).max(-1).values; edge_prob=torch.softmax(out['edge_logits'],-1).max(-1).values
            for r,idx in enumerate(indices.tolist()):
                n=int(donor_n[r]); g=nx.Graph()
                for i in range(n): g.add_node(i,**{str(vocab.node_attribute):vocab.node_value(int(node_idx[r,i]))})
                d=discrete[r,:n,:n].cpu().numpy(); ei=edge_idx[r,:n,:n].cpu().numpy()
                for i,j in zip(*np.nonzero(np.triu(d,1))): g.add_edge(int(i),int(j),**{str(vocab.edge_attribute):vocab.edge_value(int(ei[i,j])+1)})
                graphs.append(g); matrix=soft[r,:n,:n].cpu().numpy().astype(np.float64); np.fill_diagonal(matrix,0); continuous.append(matrix); spectra.append(lam[r].cpu().numpy().astype(np.float64)); basis_indices.append(int(idx)); predictions.append({'graph_index':len(graphs)-1,**{k:v[r].cpu().numpy().astype(np.float64) for k,v in summaries.items()}}); ep=edge_prob[r,:n,:n][discrete[r,:n,:n]].mean().item() if discrete[r,:n,:n].any() else float('nan'); attr_conf.append({'node_confidence_mean':float(node_prob[r,:n].mean()),'edge_confidence_mean':float(ep)})
            print(f"Attributed-Laplacian-loggap generated {len(graphs)}/{request.num_graphs}",flush=True)
    layout=request.run.layout; target=layout.generation_dir(request.resolved_generation_id); ArtifactLayout.require_available(target,overwrite=request.overwrite); target.parent.mkdir(parents=True,exist_ok=True); staging=Path(tempfile.mkdtemp(prefix='.gdsm_attr_loggap_generate_',dir=target.parent))
    try:
        gp=staging/'base_graphs.pkl'; pickle.dump(graphs,gp.open('wb'),protocol=pickle.HIGHEST_PROTOCOL); cp=staging/'continuous_adjacencies.pkl'; pickle.dump(continuous,cp.open('wb'),protocol=pickle.HIGHEST_PROTOCOL); sp=staging/'sampled_spectra.pkl'; pickle.dump(spectra,sp.open('wb'),protocol=pickle.HIGHEST_PROTOCOL); bp=staging/'sampled_basis_indices.pkl'; pickle.dump(basis_indices,bp.open('wb'),protocol=pickle.HIGHEST_PROTOCOL); pp=staging/'predicted_structure_summaries.pkl'; pickle.dump(predictions,pp.open('wb'),protocol=pickle.HIGHEST_PROTOCOL); ap=staging/'attribute_prediction_diagnostics.json'
        finite_edge_conf=[x['edge_confidence_mean'] for x in attr_conf if np.isfinite(x['edge_confidence_mean'])]
        _write_json(ap,{'per_graph':attr_conf,'aggregate':{'node_confidence_mean':float(np.mean([x['node_confidence_mean'] for x in attr_conf])),'edge_confidence_mean':float(np.mean(finite_edge_conf)) if finite_edge_conf else None}})
        _write_json(staging/'manifest.json',{'format':GENERATION_FORMAT,'model_id':wrapper.model_id,'variant':state['variant'],'run_id':request.run.run_id,'generation_id':request.resolved_generation_id,'generation_seed':request.generation_seed,'num_requested':request.num_graphs,'num_generated':len(graphs),'duration_seconds':time.monotonic()-started,'base_graphs':{'path':'base_graphs.pkl','sha256':_sha256(gp),'role':'final_attributed_graphs'},'continuous_adjacencies':{'path':'continuous_adjacencies.pkl','sha256':_sha256(cp)},'sampled_spectra':{'path':'sampled_spectra.pkl','sha256':_sha256(sp)},'predicted_structure_summaries':{'path':'predicted_structure_summaries.pkl','sha256':_sha256(pp),'typed_graphlets':True},'attribute_prediction_diagnostics':{'path':'attribute_prediction_diagnostics.json','sha256':_sha256(ap)},'checkpoint':{'path':str(request.checkpoint_path.resolve()),'sha256':_sha256(request.checkpoint_path)},'sampling':{'topology':'Laplacian_log_gap_reverse_diffusion','node_attributes':'two_pass_masked_categorical_denoising' if bool(dict(attr_cfg.get('decode',{}) or {}).get('two_pass',True)) else 'one_shot_masked_categorical_denoising','edge_attributes':'real_bond_type_head_on_generated_edges_only','edge_head_includes_no_edge':False,'categorical_flow_matching':False,'typed_graphlet_supervision':True,'rewiring':False,'posthoc_repair':False}})
        if target.exists(): shutil.rmtree(target)
        staging.replace(target)
    except BaseException:
        shutil.rmtree(staging,ignore_errors=True); raise
    return GenerationArtifacts(run_dir=layout.run_dir,generation_dir=target,graphs_path=target/'base_graphs.pkl',manifest_path=target/'manifest.json',num_requested=request.num_graphs,num_generated=len(graphs),graphs_sha256=_sha256(target/'base_graphs.pkl'))

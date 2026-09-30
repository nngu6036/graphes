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

No degree prior, Havel-Hakimi constructor, categorical edge head, graphlet
head, rewiring, or post-hoc repair is used in this path.  It is intentionally
kept as the clean baseline on which later GraphES features can be added.
"""
from __future__ import annotations

import copy
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

from grapher.models.artifacts import ArtifactLayout
from grapher.models.base import GenerateRequest, GenerationArtifacts, TrainRequest, TrainingArtifacts
from grapher.models.errors import ArtifactCollisionError
from grapher.utils.networkx_pickle import load_trusted_networkx_pickle


CHECKPOINT_FORMAT = "gdsm_simple_vanilla_gsdm_checkpoint_v1"
TRAINING_FORMAT = "grapher_gdsm_simple_vanilla_gsdm_training_v1"
GENERATION_FORMAT = "grapher_gdsm_simple_vanilla_gsdm_generation_v1"


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
        },
        "sde": {
            "x": {"type": "vp", "beta_min": 0.1, "beta_max": 1.0, "num_scales": 1000},
            "spectrum": {"type": "vp", "beta_min": 0.1, "beta_max": 1.0, "num_scales": 1000},
            "eps": 1.0e-5,
            "eigen_mask": "official_extremes",
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
        "generation_batch_size": 128,
        "runtime": {"device": "auto"},
    }


def validate_options(options: Mapping[str, Any]) -> None:
    allowed = {
        "variant", "train", "model", "sde", "sample", "generation_batch_size",
        "runtime", "extensions", "comparison_reference", "training_estimates", "diffusion",
    }
    unknown = sorted(set(options) - allowed)
    if unknown:
        raise ValueError(f"Unknown vanilla GSDM options: {unknown}")
    if str(options.get("variant", "")).lower() not in {"vanilla_gsdm", "vanilla", "gsdm"}:
        raise ValueError("vanilla GSDM pipeline requires variant: vanilla_gsdm")
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


class GSDMNodeScore(nn.Module):
    """Noise predictor for the auxiliary degree-feature diffusion."""

    def __init__(self, *, max_feat_num: int, hidden_dim: int, depth: int) -> None:
        super().__init__()
        self.max_feat_num = int(max_feat_num)
        self.depth = int(depth)
        layers = []
        for i in range(depth):
            layers.append(DenseGCNConv(max_feat_num if i == 0 else hidden_dim, hidden_dim))
        self.layers = nn.ModuleList(layers)
        fdim = max_feat_num + depth * hidden_dim
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
        values = [x]
        h = x
        for layer in self.layers:
            h = torch.tanh(layer(h, adj))
            values.append(h)
        out = self.final(torch.cat(values, dim=-1))
        return out * flags.unsqueeze(-1).to(out.dtype)


class GSDMSpectrumScore(nn.Module):
    """Noise predictor over adjacency eigenvalues conditioned on the spectral graph state."""

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

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        flags: torch.Tensor,
        eigenvectors: torch.Tensor,
        eigenvalues: torch.Tensor,
    ) -> torch.Tensor:
        del eigenvectors
        values = [x]
        h = x
        for layer in self.layers:
            h = torch.tanh(layer(h, adj))
            values.append(h)
        h = self.node_final(torch.cat(values, dim=-1))
        h = h * flags.unsqueeze(-1).to(h.dtype)
        count = flags.sum(dim=1, keepdim=True).clamp_min(1).to(h.dtype)
        pooled = h.sum(dim=1) / count
        return self.spectrum_final(torch.cat((pooled, eigenvalues), dim=-1))


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
    graphs: list[nx.Graph], *, max_nodes: int, max_feat_num: int
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
    lam, u = torch.linalg.eigh(adj)
    return x, adj, flg, size, u, lam


def _loss_batch(
    model_x: GSDMNodeScore,
    model_lam: GSDMSpectrumScore,
    batch: tuple[torch.Tensor, ...],
    *,
    sde_x: VPSDE,
    sde_lam: VPSDE,
    eps: float,
    eigen_mask_mode: str,
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
    adj_t = mask_adj(reconstruct_adjacency(u, lam_t), flags)

    pred_x = model_x(xt, adj_t, flags, u, lam_t)
    pred_lam = model_lam(xt, adj_t, flags, u, lam_t)

    # The released score wrapper converts raw network output to score=-eps/std,
    # making its DSM objective exactly an epsilon-prediction objective.
    loss_x = 0.5 * (pred_x - z_x).square().reshape(b, -1).sum(dim=-1).mean()
    loss_lam = 0.5 * (pred_lam - z_lam).square().reshape(b, -1).sum(dim=-1).mean()
    return loss_x, loss_lam


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
    return {
        "max_nodes": int(max_nodes),
        "max_feat_num": int(max_feat_num),
        "hidden_dim": int(raw.get("hidden_dim", 32)),
        "depth": int(raw.get("depth", 3)),
    }


def _build_models(cfg: Mapping[str, Any], device: torch.device) -> tuple[GSDMNodeScore, GSDMSpectrumScore]:
    mx = GSDMNodeScore(
        max_feat_num=int(cfg["max_feat_num"]),
        hidden_dim=int(cfg["hidden_dim"]),
        depth=int(cfg["depth"]),
    ).to(device)
    ml = GSDMSpectrumScore(
        max_feat_num=int(cfg["max_feat_num"]),
        max_nodes=int(cfg["max_nodes"]),
        hidden_dim=int(cfg["hidden_dim"]),
        depth=int(cfg["depth"]),
    ).to(device)
    return mx, ml


def _make_sdes(options: Mapping[str, Any], device: torch.device) -> tuple[VPSDE, VPSDE]:
    sx = options["sde"]["x"]
    sl = options["sde"]["spectrum"]
    return (
        VPSDE(beta_min=float(sx["beta_min"]), beta_max=float(sx["beta_max"]), num_scales=int(sx["num_scales"]), device=device),
        VPSDE(beta_min=float(sl["beta_min"]), beta_max=float(sl["beta_max"]), num_scales=int(sl["num_scales"]), device=device),
    )


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
    train_graphs = _graphs(request.dataset.split_paths["train"])
    val_graphs = _graphs(request.dataset.split_paths["val"])
    configured_max = options["model"].get("max_nodes")
    max_nodes = int(configured_max or max(g.number_of_nodes() for g in train_graphs))
    if max(g.number_of_nodes() for g in val_graphs) > max_nodes:
        raise ValueError("Validation graph exceeds model.max_nodes")
    model_cfg = _model_config(options, max_nodes)
    train_data = _padded_dataset(train_graphs, max_nodes=max_nodes, max_feat_num=model_cfg["max_feat_num"])
    val_data = _padded_dataset(val_graphs, max_nodes=max_nodes, max_feat_num=model_cfg["max_feat_num"])
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
    model_x, model_lam = _build_models(model_cfg, device)
    sde_x, sde_lam = _make_sdes(options, device)
    train_cfg = options["train"]
    optimizers = [
        torch.optim.Adam(model_x.parameters(), lr=float(train_cfg["lr"]), weight_decay=float(train_cfg.get("weight_decay", 0.0))),
        torch.optim.Adam(model_lam.parameters(), lr=float(train_cfg["lr"]), weight_decay=float(train_cfg.get("weight_decay", 0.0))),
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
                total_x = total_l = total_n = 0.0
                for batch_cpu in train_loader:
                    batch = tuple(v.to(device) for v in batch_cpu)
                    for opt in optimizers:
                        opt.zero_grad(set_to_none=True)
                    lx, ll = _loss_batch(
                        model_x, model_lam, batch,
                        sde_x=sde_x, sde_lam=sde_lam, eps=eps,
                        eigen_mask_mode=eigen_mask_mode,
                    )
                    loss = lx + ll
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite vanilla GSDM training loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model_x.parameters(), float(train_cfg.get("grad_norm", 1.0)))
                    torch.nn.utils.clip_grad_norm_(model_lam.parameters(), float(train_cfg.get("grad_norm", 1.0)))
                    for opt in optimizers:
                        opt.step()
                    ema_x.update(model_x); ema_lam.update(model_lam)
                    b = batch[0].size(0)
                    total_x += float(lx.item()) * b
                    total_l += float(ll.item()) * b
                    total_n += b
                if bool(train_cfg.get("lr_schedule", True)):
                    for scheduler in schedulers:
                        scheduler.step()
                record: dict[str, Any] = {
                    "epoch": epoch,
                    "train_node_loss": total_x / max(total_n, 1),
                    "train_spectrum_loss": total_l / max(total_n, 1),
                }
                record["train_loss"] = record["train_node_loss"] + record["train_spectrum_loss"]
                if epoch == 1 or epoch % val_every == 0 or epoch == epochs:
                    model_x.eval(); model_lam.eval()
                    vx = vl = vn = 0.0
                    gen = torch.Generator(device=device).manual_seed(request.run.train_seed + 100003 + epoch)
                    with torch.no_grad():
                        for batch_cpu in val_loader:
                            batch = tuple(v.to(device) for v in batch_cpu)
                            lx, ll = _loss_batch(
                                model_x, model_lam, batch,
                                sde_x=sde_x, sde_lam=sde_lam, eps=eps,
                                eigen_mask_mode=eigen_mask_mode, generator=gen,
                            )
                            b = batch[0].size(0)
                            vx += float(lx.item()) * b
                            vl += float(ll.item()) * b
                            vn += b
                    record["val_node_loss"] = vx / max(vn, 1)
                    record["val_spectrum_loss"] = vl / max(vn, 1)
                    record["val_loss"] = record["val_node_loss"] + record["val_spectrum_loss"]
                history.append(record)
                if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
                    line = (
                        f"Vanilla-GSDM epoch {epoch}/{epochs} "
                        f"train={record['train_loss']:.6f} "
                        f"node={record['train_node_loss']:.6f} spectrum={record['train_spectrum_loss']:.6f}"
                    )
                    if "val_loss" in record:
                        line += f" val={record['val_loss']:.6f}"
                    print(line, flush=True)
                    log.write(line + "\n"); log.flush()

        checkpoint_dir = staging / "checkpoints"
        checkpoint_dir.mkdir(parents=True)
        checkpoint_path = checkpoint_dir / "gdsm_simple.pt"
        checkpoint = {
            "format": CHECKPOINT_FORMAT,
            "variant": "vanilla_gsdm",
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
            "node_feature_init": "degree_one_hot",
            "history": history,
            "train_seed": request.run.train_seed,
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
            "variant": "vanilla_gsdm",
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
                "forward_state": "degree_features_and_adjacency_eigenvalues",
                "spectral_corruption": "VP_SDE_on_eigenvalues_in_fixed_training_graph_eigenbasis",
                "denoiser_conditioning": "continuous_U_diag_lambda_t_Ut_plus_noisy_degree_features",
                "generation_basis": "uniform_training_adjacency_eigenbasis_joint_with_node_count",
                "reverse_sampler": "Euler_Maruyama_plus_optional_Langevin_corrector",
                "discretization": "single_final_threshold",
                "degree_constraint": False,
                "rewiring": False,
                "categorical_edge_head": False,
                "structural_guidance": False,
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
            "variant": "vanilla_gsdm",
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
    _, std = sde.marginal_coeff(t)
    return -raw / std[:, None, None].clamp_min(1.0e-12)


def _score_lam(
    model: GSDMSpectrumScore, x: torch.Tensor, adj: torch.Tensor, flags: torch.Tensor,
    t: torch.Tensor, u: torch.Tensor, lam: torch.Tensor, sde: VPSDE,
) -> torch.Tensor:
    raw = model(x, adj, flags, u, lam)
    _, std = sde.marginal_coeff(t)
    return -raw / std[:, None].clamp_min(1.0e-12)


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    lam_mean = lam
    timestep = (t * (sde.N - 1) / sde.T).long()
    alpha = sde.alphas[timestep]
    u_t = u.transpose(-1, -2)
    adj_sample = adj
    adj_mean = adj
    for _ in range(n_steps):
        grad = _score_lam(model, x, adj_sample, flags, t, u, lam, sde)
        # The released GSDM sampler draws full eigenvalue noise.  The persistent
        # reverse-state mean is masked *after* each corrector/predictor update.
        noise = torch.randn(lam.shape, device=lam.device, dtype=lam.dtype, generator=generator)
        grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean().clamp_min(1.0e-12)
        noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean().clamp_min(1.0e-12)
        step = (float(snr) * noise_norm / grad_norm).square() * 2.0 * alpha
        lam_mean = lam + step[:, None] * grad
        lam_sample = lam_mean + torch.sqrt(2.0 * step)[:, None] * noise * float(scale_eps)
        adj_sample = mask_adj(u @ torch.diag_embed(lam_sample) @ u_t, flags)
        adj_mean = mask_adj(u @ torch.diag_embed(lam_mean) @ u_t, flags)
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    beta = sde.beta(t)
    score = _score_lam(model, x, adj, flags, t, u, lam, sde)
    drift = -0.5 * beta[:, None] * lam - beta[:, None] * score
    dt = -1.0 / float(sde.N)
    lam_mean = lam + drift * dt
    noise = torch.randn(lam.shape, device=lam.device, dtype=lam.dtype, generator=generator)
    lam_sample = lam_mean + torch.sqrt(beta * (-dt))[:, None] * noise
    u_t = u.transpose(-1, -2)
    adj_sample = mask_adj(u @ torch.diag_embed(lam_sample) @ u_t, flags)
    adj_mean = mask_adj(u @ torch.diag_embed(lam_mean) @ u_t, flags)
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
    _, u = torch.linalg.eigh(donor_adjacencies)
    e_mask = eigen_mask_from_flags(flags, eigen_mask_mode)
    max_feat_num = model_x.max_feat_num
    x = mask_x(
        torch.randn((b, nmax, max_feat_num), device=device, generator=generator), flags
    )
    # Upstream prior_sampling_sym3 is an unmasked standard-normal vector.
    # The eigen-mask is applied to the carried mean after each reverse update.
    lam = torch.randn((b, nmax), device=device, generator=generator)
    adj = mask_adj(reconstruct_adjacency(u, lam), flags)
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
            )
            final_lam_sample = corr_sample
            final_lam_mean = corr_mean
            lam = corr_mean * e_mask
        old_x = x
        x, x_mean = _euler_x(model_x, x, adj, flags, t, u, lam, sde_x, generator=generator)
        adj, adj_mean, pred_sample, pred_mean = _euler_lambda(
            model_lam, old_x, adj, flags, t, u, lam, sde_lam,
            eigen_mask=e_mask, generator=generator,
        )
        final_lam_sample = pred_sample
        final_lam_mean = pred_mean
        lam = pred_mean * e_mask
    if bool(sample_cfg.get("noise_removal", True)):
        return x_mean, adj_mean, final_lam_mean, u
    return x, adj, final_lam_sample, u


def generate(wrapper, request: GenerateRequest, state: Mapping[str, Any], manifest: Mapping[str, Any], options: Mapping[str, Any]) -> GenerationArtifacts:
    validate_options(options)
    if state.get("format") != CHECKPOINT_FORMAT:
        raise RuntimeError(f"Expected {CHECKPOINT_FORMAT}, found {state.get('format')!r}")
    device = _resolve_device(options.get("runtime", {}))
    _seed_everything(request.generation_seed)
    model_cfg = state["model_config"]
    model_x, model_lam = _build_models(model_cfg, device)
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
    graphs: list[nx.Graph] = []
    continuous: list[np.ndarray] = []
    spectra: list[np.ndarray] = []
    basis_indices: list[int] = []
    start = time.monotonic()
    with torch.no_grad():
        while len(graphs) < request.num_graphs:
            b = min(batch_size, request.num_graphs - len(graphs))
            indices = rng.integers(0, len(basis_adj), size=b)
            donor_adj = basis_adj[indices]
            donor_n = basis_n[indices]
            _, soft, lam, _ = sample_batch(
                model_x, model_lam,
                donor_adjacencies=donor_adj,
                donor_sizes=donor_n,
                sde_x=sde_x, sde_lam=sde_lam,
                sample_cfg=sample_cfg,
                eigen_mask_mode=eigen_mask_mode,
                device=device, generator=generator,
            )
            soft = 0.5 * (soft + soft.transpose(-1, -2))
            for row, idx in enumerate(indices.tolist()):
                n = int(donor_n[row].item())
                matrix = soft[row, :n, :n].detach().cpu().numpy().astype(np.float64)
                np.fill_diagonal(matrix, 0.0)
                discrete = matrix > threshold
                np.fill_diagonal(discrete, False)
                graph = nx.from_numpy_array(discrete.astype(np.int8), create_using=nx.Graph)
                graphs.append(graph)
                continuous.append(matrix)
                spectra.append(lam[row, :].detach().cpu().numpy().astype(np.float64))
                basis_indices.append(int(idx))
            print(f"Vanilla-GSDM generated {len(graphs)}/{request.num_graphs}", flush=True)

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
        soft_path = staging / "continuous_adjacencies.pkl"
        with soft_path.open("wb") as handle:
            pickle.dump(continuous, handle, protocol=pickle.HIGHEST_PROTOCOL)
        spectrum_path = staging / "sampled_spectra.pkl"
        with spectrum_path.open("wb") as handle:
            pickle.dump(spectra, handle, protocol=pickle.HIGHEST_PROTOCOL)
        index_path = staging / "sampled_basis_indices.pkl"
        with index_path.open("wb") as handle:
            pickle.dump(basis_indices, handle, protocol=pickle.HIGHEST_PROTOCOL)
        graph_hash = _sha256(graph_path)
        connected = [g.number_of_nodes() <= 1 or nx.is_connected(g) for g in graphs]
        degree_sums = [sum(dict(g.degree()).values()) for g in graphs]
        _write_json(staging / "manifest.json", {
            "format": GENERATION_FORMAT,
            "model_id": wrapper.model_id,
            "variant": "vanilla_gsdm",
            "run_id": request.run.run_id,
            "generation_id": generation_id,
            "generation_seed": request.generation_seed,
            "num_requested": request.num_graphs,
            "num_generated": len(graphs),
            "duration_seconds": time.monotonic() - start,
            "base_graphs": {"path": "base_graphs.pkl", "sha256": graph_hash},
            "continuous_adjacencies": {"path": "continuous_adjacencies.pkl", "sha256": _sha256(soft_path)},
            "sampled_spectra": {"path": "sampled_spectra.pkl", "sha256": _sha256(spectrum_path)},
            "sampled_basis_indices": {
                "path": "sampled_basis_indices.pkl", "sha256": _sha256(index_path),
                "role": "uniform indices into training-only adjacency/eigenbasis bank",
            },
            "checkpoint": {"path": str(request.checkpoint_path.resolve()), "sha256": _sha256(request.checkpoint_path)},
            "sampling": {
                "node_count": "jointly_sampled_with_training_eigenbasis",
                "eigenvectors": "training_split_empirical",
                "reverse_process": "coupled_VP_spectral_predictor_corrector",
                "threshold": threshold,
                "posthoc_repair": False,
                "rewiring": False,
                "degree_constraint": False,
                "structural_guidance": False,
            },
            "diagnostics": {
                "connectedness_rate": float(np.mean(connected)),
                "mean_degree_sum": float(np.mean(degree_sums)),
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

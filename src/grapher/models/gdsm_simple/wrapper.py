"""Project-owned minimal GSDM reference wrapper.

This is intentionally not a reproduction claim for the released GSDM code.
It provides a compact, inspectable spectral-diffusion reference inside GraphER
so GraphER components can be added incrementally in controlled ablations.
"""
from __future__ import annotations

import copy
import hashlib
import json
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
import yaml
from torch.utils.data import DataLoader, TensorDataset

from grapher.models.artifacts import ArtifactLayout
from grapher.models.base import (
    BaselineCapabilities,
    BaseGeneratorWrapper,
    GenerateRequest,
    GenerationArtifacts,
    TrainRequest,
    TrainingArtifacts,
)
from grapher.models.errors import ArtifactCollisionError
from grapher.models.gdsm_simple.model import (
    EigenvalueDenoiser,
    ddim_sample,
    make_schedule,
    q_sample,
    reconstruct_soft_adjacency,
)
from grapher.models.gdsm_simple.refiner import (
    SpectralRewireConfig,
    adjacency_spectral_rmse,
    initial_lambda_gate,
    refine_graph_toward_adjacency_spectrum,
)
from grapher.utils.networkx_pickle import load_trusted_networkx_pickle


TRAINING_FORMAT = "grapher_gdsm_simple_training_v1"
GENERATION_FORMAT = "grapher_gdsm_simple_generation_v1"


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


def _deep_update(base: dict[str, Any], changes: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in changes.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            base[key] = _deep_update(dict(base[key]), value)
        else:
            base[key] = copy.deepcopy(value)
    return base


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
        raise RuntimeError("gdsm_simple requested CUDA but torch.cuda.is_available() is false")
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
            raise ValueError("gdsm_simple supports simple undirected graphs only")
        result.append(nx.convert_node_labels_to_integers(graph, ordering="sorted"))
    return result


def _spectral_dataset(graphs: list[nx.Graph], max_nodes: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[np.ndarray], list[int]]:
    spectra, masks, sizes, bases, edge_counts = [], [], [], [], []
    for graph in graphs:
        n = graph.number_of_nodes()
        if n < 2 or n > max_nodes:
            raise ValueError(f"Graph size {n} is outside supported range [2, {max_nodes}]")
        adjacency = nx.to_numpy_array(graph, nodelist=range(n), dtype=np.float64)
        values, vectors = np.linalg.eigh(adjacency)
        padded = np.zeros(max_nodes, dtype=np.float32)
        # sqrt(n) keeps adjacency eigenvalues O(1) across small graph sizes.
        padded[:n] = (values / np.sqrt(float(n))).astype(np.float32)
        mask = np.zeros(max_nodes, dtype=np.bool_)
        mask[:n] = True
        spectra.append(padded)
        masks.append(mask)
        sizes.append(n)
        bases.append(vectors.astype(np.float32))
        edge_counts.append(graph.number_of_edges())
    return (
        torch.tensor(np.stack(spectra), dtype=torch.float32),
        torch.tensor(np.stack(masks), dtype=torch.bool),
        torch.tensor(sizes, dtype=torch.long),
        bases,
        edge_counts,
    )


class GDSMSimpleWrapper(BaseGeneratorWrapper):
    model_id = "gdsm_simple"
    display_name = "GSDM-Simple"
    capabilities = BaselineCapabilities(frozenset({"generic"}), "in_process", "ready")
    implementation_note = (
        "Project-owned reference: adjacency-eigenvalue DDPM/DDIM with empirical "
        "training eigenbases, UΛU^T threshold reconstruction, and optional "
        "source-preserving degree-constrained spectral refinement at generation."
    )
    supported_datasets = frozenset({"community_small", "ego_small", "grid"})

    default_options: dict[str, Any] = {
        "train": {
            "epochs": 200,
            "batch_size": 64,
            "lr": 2.0e-4,
            "weight_decay": 1.0e-6,
            "grad_norm": 1.0,
            "validation_every": 5,
            "log_every": 10,
        },
        "model": {
            "max_nodes": None,
            "hidden_dim": 128,
            "num_layers": 4,
            "num_heads": 4,
            "ff_dim": 256,
            "dropout": 0.0,
        },
        "diffusion": {
            "steps": 1000,
            "beta_start": 1.0e-4,
            "beta_end": 2.0e-2,
        },
        "sample": {
            "steps": 100,
            "threshold": 0.5,
            "sort_eigenvalues": True,
        },
        "generation_batch_size": 128,
        "runtime": {"device": "auto"},
        "extensions": {
            "degree_conditioning": False,
            "hh_initialization": False,
            "degree_preserving_rewiring": False,
            "rewiring": {
                "mode": "source_preserving",
                "max_steps": 4,
                "proposal_budget": 256,
                "valid_candidate_budget": 128,
                "min_relative_improvement": 1.0e-6,
                "relative_improvement_epsilon": 1.0e-12,
                "preserve_connectivity_if_source_connected": True,
                "reject_revisited_states": True,
                "lambda_weight": 1.0,
                "projector_weight": 1.0,
                "source_weight": 0.1,
                "projector_rank": 4,
                "projector_normalization_floor": 0.05,
                "projector_relative_worsening_tolerance": 0.10,
                "require_lambda_improvement": True,
                "gate_enabled": True,
                "gate_initial_lambda_quantile": 0.75,
            },
            "structural_summary": "none",
        },
    }

    def _options(self, request: TrainRequest) -> dict[str, Any]:
        request_options = copy.deepcopy(dict(request.options))
        comparison_defaults = request_options.pop("comparison_defaults", {}) or {}
        options = _deep_update(copy.deepcopy(self.default_options), comparison_defaults)
        if request.config_path is not None:
            raw = yaml.safe_load(request.config_path.read_text(encoding="utf-8")) or {}
            selected = raw.get(self.model_id, raw)
            if not isinstance(selected, Mapping):
                raise TypeError("gdsm_simple config section must be a mapping")
            options = _deep_update(options, selected)
        options = _deep_update(options, request_options)
        allowed = {
            "train", "model", "diffusion", "sample", "generation_batch_size",
            "runtime", "extensions", "comparison_reference", "training_estimates",
        }
        unknown = sorted(set(options) - allowed)
        if unknown:
            raise ValueError(f"Unknown gdsm_simple options: {unknown}")
        extensions = options.get("extensions", {}) or {}
        unsupported = [
            key for key in ("degree_conditioning", "hh_initialization")
            if bool(extensions.get(key, False))
        ]
        if str(extensions.get("structural_summary", "none")).lower() != "none":
            unsupported.append("structural_summary")
        if unsupported:
            raise NotImplementedError(
                "This reference intentionally keeps GraphER extensions disabled; "
                f"requested unsupported extensions: {unsupported}. Add them as a new ablation stage."
            )
        estimates = options.get("training_estimates", {}) or {}
        if estimates.get("enabled", False):
            raise ValueError("gdsm_simple does not create corrector-training estimates")
        return options

    def _artifacts(self, request: TrainRequest) -> TrainingArtifacts:
        layout = request.run.layout
        return TrainingArtifacts(
            run_dir=layout.run_dir,
            checkpoint_path=layout.checkpoints_dir / "gdsm_simple.pt",
            manifest_path=layout.training_manifest_path,
            log_path=layout.training_log_path,
        )

    def train(self, request: TrainRequest) -> TrainingArtifacts:
        self.validate_train_request(request)
        if request.run.dataset_id not in self.supported_datasets:
            raise ValueError(f"gdsm_simple supports {sorted(self.supported_datasets)}")
        if request.resume_from is not None:
            raise ValueError("gdsm_simple resume is not implemented")
        options = self._options(request)
        layout = request.run.layout
        artifacts = self._artifacts(request)
        fingerprint = request.dataset.fingerprint()
        if layout.training_manifest_path.is_file() and not request.overwrite:
            old = json.loads(layout.training_manifest_path.read_text(encoding="utf-8"))
            if old.get("dataset", {}).get("fingerprint") == fingerprint and old.get("options") == _jsonable(options) and artifacts.checkpoint_path.is_file():
                return artifacts
            raise ArtifactCollisionError("Existing gdsm_simple run differs; choose a new run-id or --overwrite")
        ArtifactLayout.require_available(layout.train_dir, overwrite=request.overwrite)
        _seed_everything(request.run.train_seed)
        device = _resolve_device(options.get("runtime", {}))
        train_graphs = _graphs(request.dataset.split_paths["train"])
        val_graphs = _graphs(request.dataset.split_paths["val"])
        configured_max = options["model"].get("max_nodes")
        max_nodes = int(configured_max or max(g.number_of_nodes() for g in train_graphs))
        if max(g.number_of_nodes() for g in val_graphs) > max_nodes:
            raise ValueError("Validation graph exceeds model.max_nodes; set an explicit protocol maximum")
        train_x, train_mask, train_n, train_bases, train_edges = _spectral_dataset(train_graphs, max_nodes)
        val_x, val_mask, val_n, _, _ = _spectral_dataset(val_graphs, max_nodes)
        train_cfg = options["train"]
        model_cfg = dict(options["model"])
        model_cfg["max_nodes"] = max_nodes
        model = EigenvalueDenoiser(
            max_nodes=max_nodes,
            hidden_dim=int(model_cfg["hidden_dim"]),
            num_layers=int(model_cfg["num_layers"]),
            num_heads=int(model_cfg["num_heads"]),
            ff_dim=int(model_cfg["ff_dim"]),
            dropout=float(model_cfg.get("dropout", 0.0)),
        ).to(device)
        diffusion_cfg = options["diffusion"]
        schedule = make_schedule(
            steps=int(diffusion_cfg["steps"]),
            beta_start=float(diffusion_cfg["beta_start"]),
            beta_end=float(diffusion_cfg["beta_end"]),
            device=device,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(train_cfg["lr"]),
            weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        )
        batch_size = int(train_cfg["batch_size"])
        train_loader = DataLoader(TensorDataset(train_x, train_mask, train_n), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(val_x, val_mask, val_n), batch_size=batch_size, shuffle=False)
        epochs = int(train_cfg["epochs"])
        val_every = int(train_cfg.get("validation_every", 1))
        history: list[dict[str, Any]] = []
        best_state = None
        best_val = float("inf")
        best_epoch = 0
        start = time.monotonic()
        layout.train_dir.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".gdsm_simple_train_", dir=layout.train_dir.parent))
        log_path = staging / "train.log"
        try:
            with log_path.open("w", encoding="utf-8") as log:
                for epoch in range(1, epochs + 1):
                    model.train()
                    total, count = 0.0, 0
                    for clean, mask, n in train_loader:
                        clean, mask, n = clean.to(device), mask.to(device), n.to(device)
                        t = torch.randint(0, schedule.alpha_bar.numel(), (clean.size(0),), device=device)
                        noise = torch.randn_like(clean) * mask
                        noisy = q_sample(clean, t, noise, schedule) * mask
                        pred = model(noisy, t, mask, n, int(schedule.alpha_bar.numel()))
                        loss = ((pred - noise).square() * mask).sum() / mask.sum().clamp_min(1)
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg.get("grad_norm", 1.0)))
                        optimizer.step()
                        total += float(loss.item()) * clean.size(0)
                        count += clean.size(0)
                    record: dict[str, Any] = {"epoch": epoch, "train_loss": total / max(count, 1)}
                    if epoch == 1 or epoch % val_every == 0 or epoch == epochs:
                        model.eval()
                        vtotal, vcount = 0.0, 0
                        generator = torch.Generator(device=device).manual_seed(request.run.train_seed + 100003)
                        with torch.no_grad():
                            for clean, mask, n in val_loader:
                                clean, mask, n = clean.to(device), mask.to(device), n.to(device)
                                t = torch.randint(0, schedule.alpha_bar.numel(), (clean.size(0),), device=device, generator=generator)
                                noise = torch.randn(clean.shape, device=device, generator=generator) * mask
                                noisy = q_sample(clean, t, noise, schedule) * mask
                                pred = model(noisy, t, mask, n, int(schedule.alpha_bar.numel()))
                                loss = ((pred - noise).square() * mask).sum() / mask.sum().clamp_min(1)
                                vtotal += float(loss.item()) * clean.size(0)
                                vcount += clean.size(0)
                        record["val_loss"] = vtotal / max(vcount, 1)
                        if record["val_loss"] < best_val:
                            best_val = float(record["val_loss"])
                            best_epoch = epoch
                            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    history.append(record)
                    if epoch == 1 or epoch % int(train_cfg.get("log_every", 10)) == 0 or epoch == epochs:
                        line = f"GSDM-Simple epoch {epoch}/{epochs} train={record['train_loss']:.6f}"
                        if "val_loss" in record:
                            line += f" val={record['val_loss']:.6f}"
                        print(line, flush=True)
                        log.write(line + "\n"); log.flush()
            if best_state is None:
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epochs
            checkpoint_dir = staging / "checkpoints"
            checkpoint_dir.mkdir(parents=True)
            checkpoint_path = checkpoint_dir / "gdsm_simple.pt"
            checkpoint = {
                "format": "gdsm_simple_checkpoint_v1",
                "model_state": best_state,
                "model_config": model_cfg,
                "diffusion": dict(diffusion_cfg),
                "sample": dict(options["sample"]),
                "max_nodes": max_nodes,
                "normalization": "adjacency_eigenvalues_div_sqrt_num_nodes",
                "empirical_basis_source": "training_split_only",
                "basis_num_nodes": [int(g.number_of_nodes()) for g in train_graphs],
                "basis_edge_counts": [int(v) for v in train_edges],
                "basis_eigenvectors": train_bases,
                "best_epoch": best_epoch,
                "best_val_loss": best_val,
                "train_seed": request.run.train_seed,
                "history": history,
                "extensions": dict(options.get("extensions", {})),
            }
            torch.save(checkpoint, checkpoint_path)
            resolved = copy.deepcopy(options)
            resolved["model"]["max_nodes"] = max_nodes
            (staging / "resolved_config.yaml").write_text(yaml.safe_dump({self.model_id: resolved}, sort_keys=False), encoding="utf-8")
            manifest = {
                "format": TRAINING_FORMAT,
                "model_id": self.model_id,
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
                "checkpoint_selection": {"kind": "best_validation_epsilon_mse", "epoch": best_epoch, "val_loss": best_val},
                "reference_contract": {
                    "diffused_state": "sorted_adjacency_eigenvalues",
                    "eigenvectors": "sampled_empirically_from_training_split_and_not_diffused",
                    "reconstruction": "U_diag_lambda_Ut_then_fixed_threshold",
                    "generation_extension": (
                        (
                            "source_preserving_spectral_refinement_from_threshold_graph"
                            if str(
                                options.get("extensions", {})
                                .get("rewiring", {})
                                .get("mode", "lambda_only")
                            ).lower() == "source_preserving"
                            else "degree_preserving_spectral_rewiring_from_threshold_graph"
                        )
                        if bool(options.get("extensions", {}).get("degree_preserving_rewiring", False))
                        else "none"
                    ),
                    "grapher_extensions_enabled": bool(
                        options.get("extensions", {}).get("degree_preserving_rewiring", False)
                    ),
                },
                "test_used_for_training": False,
            }
            _write_json(staging / "manifest.json", manifest)
            if layout.train_dir.exists():
                shutil.rmtree(layout.train_dir)
            staging.replace(layout.train_dir)
            _write_json(layout.run_manifest_path, {
                "format": "grapher_baseline_run_v1", "model_id": self.model_id,
                "dataset_id": request.run.dataset_id, "run_id": request.run.run_id,
                "train_seed": request.run.train_seed,
            })
            return self._artifacts(request)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    def generate(self, request: GenerateRequest) -> GenerationArtifacts:
        self.validate_generate_request(request)
        layout = request.run.layout
        if not layout.training_manifest_path.is_file():
            raise FileNotFoundError(f"Missing managed training manifest: {layout.training_manifest_path}")
        manifest = json.loads(layout.training_manifest_path.read_text(encoding="utf-8"))
        if manifest.get("model_id") != self.model_id:
            raise RuntimeError("Checkpoint belongs to a different model")
        if _sha256(request.checkpoint_path) != manifest["checkpoint"]["sha256"]:
            raise RuntimeError("Checkpoint hash differs from managed training manifest")
        state = torch.load(request.checkpoint_path, map_location="cpu", weights_only=False)
        options = copy.deepcopy(manifest["options"])
        # Generation may alter runtime/sampling controls and generation-only
        # rewiring, but never the trained denoiser architecture/diffusion.
        unknown = sorted(
            set(request.options)
            - {"runtime", "generation_batch_size", "sample", "extensions"}
        )
        if unknown:
            raise ValueError(f"Unsupported gdsm_simple generation overrides: {unknown}")
        _deep_update(options, request.options)
        generation_extensions = options.get("extensions", {}) or {}
        generation_unsupported = [
            key for key in ("degree_conditioning", "hh_initialization")
            if bool(generation_extensions.get(key, False))
        ]
        if str(generation_extensions.get("structural_summary", "none")).lower() != "none":
            generation_unsupported.append("structural_summary")
        if generation_unsupported:
            raise NotImplementedError(
                "gdsm_simple generation currently supports only the degree-preserving "
                "spectral-rewiring extension; unsupported extensions: "
                f"{generation_unsupported}"
            )
        device = _resolve_device(options.get("runtime", {}))
        _seed_everything(request.generation_seed)
        model_cfg = state["model_config"]
        model = EigenvalueDenoiser(
            max_nodes=int(model_cfg["max_nodes"]),
            hidden_dim=int(model_cfg["hidden_dim"]),
            num_layers=int(model_cfg["num_layers"]),
            num_heads=int(model_cfg["num_heads"]),
            ff_dim=int(model_cfg["ff_dim"]),
            dropout=float(model_cfg.get("dropout", 0.0)),
        ).to(device)
        model.load_state_dict(state["model_state"])
        model.eval()
        schedule = make_schedule(
            steps=int(state["diffusion"]["steps"]),
            beta_start=float(state["diffusion"]["beta_start"]),
            beta_end=float(state["diffusion"]["beta_end"]),
            device=device,
        )
        bases = state["basis_eigenvectors"]
        basis_n = np.asarray(state["basis_num_nodes"], dtype=np.int64)
        sample_cfg = options.get("sample", state["sample"])
        extensions = options.get("extensions", {}) or {}
        rewiring_enabled = bool(extensions.get("degree_preserving_rewiring", False))
        rewiring_cfg = SpectralRewireConfig.from_mapping(extensions.get("rewiring", {}))
        batch_size = int(options.get("generation_batch_size", 128))
        # Keep basis sampling independent from rewiring proposals so enabling the
        # generation-only refiner does not change the underlying GSDM samples.
        basis_rng = np.random.default_rng(request.generation_seed)
        rewiring_rng = np.random.default_rng(request.generation_seed + 1_000_003)
        generator = torch.Generator(device=device).manual_seed(request.generation_seed)
        graphs: list[nx.Graph] = []
        threshold_graphs: list[nx.Graph] = []
        target_spectra: list[np.ndarray] = []
        sampled_basis_indices: list[int] = []
        sampled_bases: list[np.ndarray] = []
        rewiring_diagnostics: list[dict[str, Any]] = []
        max_nodes = int(state["max_nodes"])
        start = time.monotonic()

        # Stage A: generate the exact S0 threshold samples first.  Refinement is
        # deliberately delayed until the whole batch exists so the optional
        # residual-quantile gate is independent of generation_batch_size.
        while len(threshold_graphs) < request.num_graphs:
            b = min(batch_size, request.num_graphs - len(threshold_graphs))
            # Sampling one training basis index jointly samples graph size and U.
            indices = basis_rng.integers(0, len(bases), size=b)
            sizes = basis_n[indices]
            mask = torch.arange(max_nodes, device=device).unsqueeze(0) < torch.tensor(
                sizes, device=device
            ).unsqueeze(1)
            num_nodes = torch.tensor(sizes, dtype=torch.long, device=device)
            normalized = ddim_sample(
                model,
                mask=mask,
                num_nodes=num_nodes,
                schedule=schedule,
                sample_steps=int(sample_cfg["steps"]),
                generator=generator,
            )
            for row, basis_index in enumerate(indices.tolist()):
                n = int(sizes[row])
                values = normalized[row, :n] * np.sqrt(float(n))
                if bool(sample_cfg.get("sort_eigenvalues", True)):
                    values = torch.sort(values).values
                basis_array = np.asarray(bases[basis_index], dtype=np.float64)
                basis = torch.tensor(
                    basis_array, dtype=torch.float32, device=device
                ).unsqueeze(0)
                soft = reconstruct_soft_adjacency(basis, values.unsqueeze(0))[0]
                soft = 0.5 * (soft + soft.T)
                soft.fill_diagonal_(0.0)
                adjacency = (
                    soft > float(sample_cfg.get("threshold", 0.5))
                ).cpu().numpy()
                graph = nx.from_numpy_array(
                    adjacency.astype(np.int8), create_using=nx.Graph
                )
                target_normalized = np.sort(
                    normalized[row, :n].detach().cpu().numpy().astype(np.float64)
                )
                threshold_graphs.append(graph)
                target_spectra.append(target_normalized.copy())
                sampled_basis_indices.append(int(basis_index))
                sampled_bases.append(basis_array.copy())
            print(
                f"GSDM-Simple threshold-generated {len(threshold_graphs)}/{request.num_graphs}",
                flush=True,
            )

        # Stage B: conservative, source-preserving spectral refinement.  The
        # high-residual gate leaves already-good GSDM samples untouched.
        initial_lambda_errors = [
            adjacency_spectral_rmse(graph, target)
            for graph, target in zip(threshold_graphs, target_spectra)
        ]
        gate_mask, gate_threshold = initial_lambda_gate(
            initial_lambda_errors,
            enabled=bool(rewiring_enabled and rewiring_cfg.gate_enabled),
            quantile=rewiring_cfg.gate_initial_lambda_quantile,
        )

        for index, (source_graph, target_normalized, sampled_basis) in enumerate(
            zip(threshold_graphs, target_spectra, sampled_bases)
        ):
            graph = source_graph.copy()
            if rewiring_enabled:
                graph, diagnostics = refine_graph_toward_adjacency_spectrum(
                    graph,
                    target_normalized,
                    rng=rewiring_rng,
                    config=rewiring_cfg,
                    target_eigenvectors=(
                        sampled_basis
                        if rewiring_cfg.mode == "source_preserving"
                        else None
                    ),
                    gate_passed=bool(gate_mask[index]),
                    gate_threshold=gate_threshold,
                )
            else:
                connected = bool(
                    graph.number_of_nodes() <= 1 or nx.is_connected(graph)
                )
                diagnostics = {
                    "enabled": False,
                    "mode": "disabled",
                    "gate_passed": False,
                    "gate_threshold": None,
                    "initial_error": float(initial_lambda_errors[index]),
                    "final_error": float(initial_lambda_errors[index]),
                    "accepted_steps": 0,
                    "degree_preserved": True,
                    "source_connected": connected,
                    "final_connected": connected,
                    "stop_reason": "disabled",
                }
            rewiring_diagnostics.append(diagnostics)
            graphs.append(graph)
            if (index + 1) % max(1, min(128, request.num_graphs)) == 0 or (
                index + 1 == request.num_graphs
            ):
                print(
                    f"GSDM-Simple refined {index + 1}/{request.num_graphs}",
                    flush=True,
                )
        generation_id = request.resolved_generation_id
        target = layout.generation_dir(generation_id)
        ArtifactLayout.require_available(target, overwrite=request.overwrite)
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".gdsm_simple_generate_", dir=target.parent))
        try:
            graph_path = staging / "base_graphs.pkl"
            with graph_path.open("wb") as handle:
                pickle.dump(graphs, handle, protocol=pickle.HIGHEST_PROTOCOL)
            graph_hash = _sha256(graph_path)

            threshold_path = staging / "threshold_graphs.pkl"
            with threshold_path.open("wb") as handle:
                pickle.dump(threshold_graphs, handle, protocol=pickle.HIGHEST_PROTOCOL)
            target_spectrum_path = staging / "target_adjacency_eigenvalues.pkl"
            with target_spectrum_path.open("wb") as handle:
                pickle.dump(target_spectra, handle, protocol=pickle.HIGHEST_PROTOCOL)
            basis_index_path = staging / "sampled_basis_indices.pkl"
            with basis_index_path.open("wb") as handle:
                pickle.dump(sampled_basis_indices, handle, protocol=pickle.HIGHEST_PROTOCOL)

            initial_errors = [
                float(d["initial_error"]) for d in rewiring_diagnostics if "initial_error" in d
            ]
            final_errors = [
                float(d["final_error"]) for d in rewiring_diagnostics if "final_error" in d
            ]
            accepted_steps = [int(d.get("accepted_steps", 0)) for d in rewiring_diagnostics]
            gate_passed = [bool(d.get("gate_passed", False)) for d in rewiring_diagnostics]
            initial_projector_errors = [
                float(d["initial_projector_error"])
                for d in rewiring_diagnostics
                if "initial_projector_error" in d
            ]
            final_projector_errors = [
                float(d["final_projector_error"])
                for d in rewiring_diagnostics
                if "final_projector_error" in d
            ]
            final_source_distances = [
                float(d["final_source_distance"])
                for d in rewiring_diagnostics
                if "final_source_distance" in d
            ]
            initial_energies = [
                float(d["initial_energy"])
                for d in rewiring_diagnostics
                if "initial_energy" in d
            ]
            final_energies = [
                float(d["final_energy"])
                for d in rewiring_diagnostics
                if "final_energy" in d
            ]
            aggregate_rewiring = {
                "enabled": rewiring_enabled,
                "mode": rewiring_cfg.mode if rewiring_enabled else "disabled",
                "num_graphs": len(graphs),
                "degree_preservation_rate": float(
                    np.mean([bool(d.get("degree_preserved", False)) for d in rewiring_diagnostics])
                ),
                "source_connected_rate": float(
                    np.mean([bool(d.get("source_connected", False)) for d in rewiring_diagnostics])
                ),
                "final_connected_rate": float(
                    np.mean([bool(d.get("final_connected", False)) for d in rewiring_diagnostics])
                ),
                "gate_enabled": bool(rewiring_enabled and rewiring_cfg.gate_enabled),
                "gate_initial_lambda_quantile": (
                    rewiring_cfg.gate_initial_lambda_quantile if rewiring_enabled else None
                ),
                "gate_threshold": gate_threshold if rewiring_enabled else None,
                "gate_pass_rate": float(np.mean(gate_passed)) if gate_passed else 0.0,
                "mean_accepted_steps": float(np.mean(accepted_steps)) if accepted_steps else 0.0,
                "mean_accepted_steps_on_gated": (
                    float(np.mean([v for v, passed in zip(accepted_steps, gate_passed) if passed]))
                    if any(gate_passed) else 0.0
                ),
                "changed_graph_rate": float(np.mean([v > 0 for v in accepted_steps])) if accepted_steps else 0.0,
                "mean_initial_spectral_rmse": float(np.mean(initial_errors)) if initial_errors else None,
                "mean_final_spectral_rmse": float(np.mean(final_errors)) if final_errors else None,
                "mean_absolute_spectral_improvement": (
                    float(np.mean(np.asarray(initial_errors) - np.asarray(final_errors)))
                    if initial_errors else None
                ),
                "mean_initial_projector_error": (
                    float(np.mean(initial_projector_errors))
                    if initial_projector_errors else None
                ),
                "mean_final_projector_error": (
                    float(np.mean(final_projector_errors))
                    if final_projector_errors else None
                ),
                "mean_final_source_edge_distance": (
                    float(np.mean(final_source_distances))
                    if final_source_distances else None
                ),
                "mean_initial_joint_energy": (
                    float(np.mean(initial_energies)) if initial_energies else None
                ),
                "mean_final_joint_energy": (
                    float(np.mean(final_energies)) if final_energies else None
                ),
                "all_accepted_steps_improve_lambda": bool(
                    all(bool(d.get("all_accepted_steps_improve", True)) for d in rewiring_diagnostics)
                ),
                "config": _jsonable(extensions.get("rewiring", {})),
            }
            _write_json(
                staging / "rewiring_diagnostics.json",
                {"aggregate": aggregate_rewiring, "per_graph": rewiring_diagnostics},
            )

            _write_json(staging / "manifest.json", {
                "format": GENERATION_FORMAT,
                "model_id": self.model_id,
                "run_id": request.run.run_id,
                "generation_id": generation_id,
                "generation_seed": request.generation_seed,
                "num_requested": request.num_graphs,
                "num_generated": len(graphs),
                "duration_seconds": time.monotonic() - start,
                "base_graphs": {"path": "base_graphs.pkl", "sha256": graph_hash},
                "threshold_graphs": {
                    "path": "threshold_graphs.pkl",
                    "sha256": _sha256(threshold_path),
                    "role": "pre_rewiring_gdsm_threshold_reconstruction",
                },
                "target_adjacency_eigenvalues": {
                    "path": "target_adjacency_eigenvalues.pkl",
                    "sha256": _sha256(target_spectrum_path),
                    "normalization": "eigenvalues_div_sqrt_num_nodes",
                },
                "sampled_basis_indices": {
                    "path": "sampled_basis_indices.pkl",
                    "sha256": _sha256(basis_index_path),
                    "role": "indices_into_checkpoint_empirical_training_eigenbasis_bank",
                },
                "checkpoint": {"path": str(request.checkpoint_path.resolve()), "sha256": _sha256(request.checkpoint_path)},
                "empirical_prior": {
                    "node_count": "jointly_sampled_with_training_eigenbasis",
                    "eigenvectors": "training_split_empirical",
                    "test_conditioning": False,
                },
                "reconstruction": {
                    "type": (
                        (
                            "spectral_threshold_then_source_preserving_spectral_refinement"
                            if rewiring_cfg.mode == "source_preserving"
                            else "spectral_threshold_then_degree_preserving_spectral_rewiring"
                        )
                        if rewiring_enabled
                        else "spectral_threshold"
                    ),
                    "threshold": float(sample_cfg.get("threshold", 0.5)),
                },
                "spectral_rewiring": aggregate_rewiring,
                "posthoc_repair": rewiring_enabled,
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

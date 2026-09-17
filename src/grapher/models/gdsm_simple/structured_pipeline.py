"""Managed training/generation for the opt-in GSDM-Simple Structure3 stage.

The old v1 spectral-only pipeline remains untouched. Rewiring never runs during
training; clean graph summaries are precomputed, not extracted from Gaussian
states. Generation carries a persistent degree-exact discrete graph through
selected reverse-time events and feeds its spectrum into later DDIM steps.
"""
from __future__ import annotations

import copy
import json
import pickle
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
from grapher.models.base import GenerationArtifacts
from grapher.models.errors import ArtifactCollisionError
from grapher.models.gdsm_simple.degree_initialization import (
    align_basis_rows, conditioning_features, degree_basis_anchor, realize_degrees, validate_degree_bank,
)
from grapher.models.gdsm_simple.model import make_schedule
from grapher.models.gdsm_simple.refiner import normalized_adjacency_eigenvalues
from grapher.models.gdsm_simple.structure3 import (
    graph_summary3, reconcile_summary3, refine_structure3, validate_structure_options,
)
from grapher.models.gdsm_simple.structured_model import (
    StructuredEigenvalueDenoiser, centered_q_sample, centered_x0, centered_ddim_step,
    summary_losses, summary_probabilities,
)
# Imported lazily by wrapper.py, after wrapper helpers have been defined.
from grapher.models.gdsm_simple.wrapper import (
    _graphs, _jsonable, _resolve_device, _seed_everything, _sha256, _spectral_dataset, _write_json,
)

CHECKPOINT_FORMAT = "gdsm_simple_structure3_checkpoint_v2"


def make_model(config, summary, device):
    return StructuredEigenvalueDenoiser(
        max_nodes=int(config["max_nodes"]), hidden_dim=int(config["hidden_dim"]),
        num_layers=int(config["num_layers"]), num_heads=int(config["num_heads"]),
        ff_dim=int(config["ff_dim"]), dropout=float(config.get("dropout", 0.)),
        clustering_bins=int(summary.get("clustering_bins", 100)),
    ).to(device)


def make_anchor(basis, degrees, init):
    if init.get("mode", "degree_basis") == "gaussian":
        return np.zeros(len(degrees), dtype=np.float64), {"mode": "gaussian"}
    anchor, diag = degree_basis_anchor(
        basis, degrees, ridge=float(init.get("ridge", 1e-3)),
        diagonal_weight=float(init.get("diagonal_weight", 1.)), seed_top_k=init.get("seed_top_k"),
    )
    return anchor, {"mode": "degree_basis", **diag}


def make_training_dataset(graphs, bank_graphs, bank_bases, max_nodes, summary_cfg, init, *, seed, pairings):
    """Independent same-size training-bank bases for train AND validation.

    No validation/test eigenbasis enters the empirical bank. A target can pair
    with itself by chance under independent empirical sampling, but is not
    systematically given its own eigenbasis.
    """
    rng = np.random.default_rng(seed)
    clean, masks, sizes, _, _ = _spectral_dataset(graphs, max_nodes)
    bank_sizes = np.array([len(g) for g in bank_graphs])
    basis_degrees = [np.array([g.degree(v) for v in range(len(g))]) for g in bank_graphs]
    rows, cache = [], {}
    bins = int(summary_cfg.get("clustering_bins", 100))
    for i, graph in enumerate(graphs):
        n = len(graph)
        choices = np.flatnonzero(bank_sizes == n)
        if not len(choices):
            raise ValueError(f"No training eigenbasis for n={n}. This empirical-basis stage requires validation sizes in training support.")
        d = np.sort(np.array([v for _, v in graph.degree()], dtype=np.int64))[::-1]
        target = graph_summary3(graph, bins)
        for _ in range(pairings):
            j = int(rng.choice(choices))
            key = (j, tuple(d))
            if key not in cache:
                basis = align_basis_rows(bank_bases[j], basis_degrees[j])
                anchor, _ = make_anchor(basis, d, init)
                cache[key] = (anchor, conditioning_features(basis, d, anchor, max_nodes))
            anchor, condition = cache[key]
            rows.append((clean[i], masks[i], sizes[i], torch.from_numpy(condition),
                         torch.tensor(np.pad(anchor, (0, max_nodes-n)), dtype=torch.float32),
                         torch.tensor(target["clustering_histogram"], dtype=torch.float32),
                         torch.tensor(target["orbit_log_mean"], dtype=torch.float32),
                         torch.tensor(target["graphlet_histogram"], dtype=torch.float32)))
    return TensorDataset(*(torch.stack([row[k] for row in rows]) for k in range(8)))


def batch_loss(model, batch, schedule, weights, device, generator=None):
    clean, mask, n, condition, anchor, clustering, orbit, graphlet = [v.to(device) for v in batch]
    t = torch.randint(0, schedule.alpha_bar.numel(), (len(clean),), device=device, generator=generator)
    noise = torch.randn(clean.shape, device=device, generator=generator) * mask
    noisy = centered_q_sample(clean, t, noise, anchor, schedule) * mask
    eps, prediction = model(noisy, t, mask, n, len(schedule.alpha_bar), condition)
    epsilon_loss = ((eps-noise).square()*mask).sum() / mask.sum().clamp_min(1)
    parts = summary_losses(prediction, clustering, orbit, graphlet)
    loss = epsilon_loss + sum(float(weights.get(k, default))*parts[k] for k, default in (("clustering", 1.), ("orbit", .5), ("graphlet", .5)))
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite joint spectral/summary loss.")
    probs = summary_probabilities(prediction)
    metrics = {"loss": float(loss.detach()), "epsilon_mse": float(epsilon_loss.detach()),
               **{k+"_loss": float(v.detach()) for k, v in parts.items()},
               "clustering_w1": float((probs["clustering_histogram"].cumsum(-1)-clustering.cumsum(-1)).abs().mean().detach()),
               "orbit_log_rmse": float((probs["orbit_log_mean"]-orbit).square().mean().sqrt().detach()),
               "graphlet_tv": float((.5*(probs["graphlet_histogram"]-graphlet).abs().sum(-1)).mean().detach())}
    return loss, metrics, len(clean)


def train_structured(wrapper, request, options):
    validate_structure_options(options)
    layout, artifacts = request.run.layout, wrapper._artifacts(request)
    fingerprint = request.dataset.fingerprint()
    if layout.training_manifest_path.is_file() and not request.overwrite:
        old = json.loads(layout.training_manifest_path.read_text())
        if old.get("dataset", {}).get("fingerprint") == fingerprint and old.get("options") == _jsonable(options) and artifacts.checkpoint_path.is_file():
            return artifacts
        raise ArtifactCollisionError("Existing run differs; use a new Structure3 run-id or --overwrite.")
    ArtifactLayout.require_available(layout.train_dir, overwrite=request.overwrite)
    _seed_everything(request.run.train_seed)
    device = _resolve_device(options.get("runtime", {}))
    train_graphs, val_graphs = _graphs(request.dataset.split_paths["train"]), _graphs(request.dataset.split_paths["val"])
    max_nodes = int(options["model"].get("max_nodes") or max(map(len, train_graphs)))
    train_x, _, _, bases, train_edges = _spectral_dataset(train_graphs, max_nodes)
    del train_x
    ext = options["extensions"]
    summary_cfg, init = ext["structural_summary"], ext.get("initialization", {})
    train_cfg, diffusion_cfg = options["train"], options["diffusion"]
    epochs, batch_size = int(train_cfg["epochs"]), int(train_cfg["batch_size"])
    pairings = int(summary_cfg.get("basis_pairings_per_graph", 4))
    if min(epochs, batch_size, pairings) < 1:
        raise ValueError("epochs, batch_size and basis_pairings_per_graph must be positive.")
    print("GSDM-Simple Structure3: caching exact clean summaries and independent degree/basis anchors", flush=True)
    train_data = make_training_dataset(train_graphs, train_graphs, bases, max_nodes, summary_cfg, init,
                                      seed=request.run.train_seed+13, pairings=pairings)
    val_data = make_training_dataset(val_graphs, train_graphs, bases, max_nodes, summary_cfg, init,
                                    seed=request.run.train_seed+17, pairings=1)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    model_cfg = {**options["model"], "max_nodes": max_nodes}
    model = make_model(model_cfg, summary_cfg, device)
    schedule = make_schedule(steps=int(diffusion_cfg["steps"]), beta_start=float(diffusion_cfg["beta_start"]), beta_end=float(diffusion_cfg["beta_end"]), device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["lr"]), weight_decay=float(train_cfg.get("weight_decay", 0.)))
    weights = summary_cfg.get("loss_weights", {})
    history, best_state, best_val, best_epoch = [], None, float("inf"), 0
    started = time.monotonic()
    layout.train_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".gdsm_structure3_train_", dir=layout.train_dir.parent))
    try:
        with (staging/"train.log").open("w") as log:
            for epoch in range(1, epochs+1):
                model.train()
                totals, count = {}, 0
                for batch in train_loader:
                    optimizer.zero_grad(set_to_none=True)
                    loss, metrics, size = batch_loss(model, batch, schedule, weights, device)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite Structure3 training loss.")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg.get("grad_norm", 1.)))
                    optimizer.step()
                    for k, v in metrics.items(): totals[k] = totals.get(k, 0.) + v*size
                    count += size
                record = {"epoch": epoch, **{"train_"+k: v/count for k, v in totals.items()}}
                if epoch == 1 or epoch == epochs or epoch % max(int(train_cfg.get("validation_every", 5)), 1) == 0:
                    model.eval()
                    totals, count = {}, 0
                    generator = torch.Generator(device=device).manual_seed(request.run.train_seed+100003)
                    with torch.no_grad():
                        for batch in val_loader:
                            _, metrics, size = batch_loss(model, batch, schedule, weights, device, generator)
                            for k, v in metrics.items(): totals[k] = totals.get(k, 0.)+v*size
                            count += size
                    record.update({"val_"+k: v/count for k, v in totals.items()})
                    if record["val_loss"] < best_val:
                        best_val, best_epoch = record["val_loss"], epoch
                        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                history.append(record)
                if epoch == 1 or epoch == epochs or epoch % max(int(train_cfg.get("log_every", 10)), 1) == 0:
                    line = f"GSDM-Simple Structure3 epoch {epoch}/{epochs} train={record['train_loss']:.6f}"
                    if "val_loss" in record:
                        line += f" val={record['val_loss']:.6f} C_W1={record['val_clustering_w1']:.5f} O_log_RMSE={record['val_orbit_log_rmse']:.5f} H3_TV={record['val_graphlet_tv']:.5f}"
                    print(line, flush=True); log.write(line+"\n"); log.flush()
        checkpoint_dir = staging/"checkpoints"
        checkpoint_dir.mkdir()
        path = checkpoint_dir/"gdsm_simple.pt"
        state = {
            "format": CHECKPOINT_FORMAT, "model_state": best_state, "model_config": model_cfg,
            "diffusion": diffusion_cfg, "sample": options["sample"], "max_nodes": max_nodes,
            "summary_config": summary_cfg, "initialization": init, "extensions": ext,
            "normalization": "adjacency_eigenvalues_div_sqrt_num_nodes",
            "empirical_basis_source": "training_split_only",
            "basis_num_nodes": [len(g) for g in train_graphs], "basis_eigenvectors": bases,
            "basis_edge_counts": train_edges,
            "basis_degree_sequences": [[int(g.degree(v)) for v in range(len(g))] for g in train_graphs],
            "best_epoch": best_epoch, "best_val_loss": best_val, "history": history, "train_seed": request.run.train_seed,
        }
        torch.save(state, path)
        resolved = copy.deepcopy(options); resolved["model"]["max_nodes"] = max_nodes
        (staging/"resolved_config.yaml").write_text(yaml.safe_dump({wrapper.model_id: resolved}, sort_keys=False))
        _write_json(staging/"training_metrics.json", {"history": history, "best_epoch": best_epoch, "best_val_loss": best_val})
        manifest = {
            "format": "grapher_gdsm_simple_training_v2", "model_id": wrapper.model_id,
            "run_id": request.run.run_id, "train_seed": request.run.train_seed,
            "created_at": datetime.now(timezone.utc).isoformat(), "duration_seconds": time.monotonic()-started,
            "dataset": {"benchmark_id": request.dataset.benchmark_id, "serialized_id": request.dataset.serialized_id,
                        "fingerprint": fingerprint, "split_sha256": {k: _sha256(v) for k, v in request.dataset.split_paths.items()}},
            "options": _jsonable(options), "checkpoint": {"path": "checkpoints/gdsm_simple.pt", "sha256": _sha256(path)},
            "checkpoint_selection": {"kind": "best_validation_joint_denoising_summary_loss", "epoch": best_epoch, "val_loss": best_val},
            "reference_contract": {"name": "GSDM-Simple-Structure3", "diffused_state": "sorted_adjacency_eigenvalues",
                                   "forward_process": "source_centered_VP" if init.get("mode", "degree_basis") == "degree_basis" else "zero_centered_VP",
                                   "basis_pairing": "independent_same_size_training_basis", "rewiring_during_training": False,
                                   "orbit_columns": [0, 1, 2, 3], "graphlets": ["induced_P3", "triangle"],
                                   "degree_prior_trained_separately": init.get("degree_generator", {}).get("type", "empirical") == "dhvae"},
            "test_used_for_training": False,
        }
        _write_json(staging/"manifest.json", manifest)
        if layout.train_dir.exists(): shutil.rmtree(layout.train_dir)
        staging.replace(layout.train_dir)
        _write_json(layout.run_manifest_path, {"format": "grapher_baseline_run_v1", "model_id": wrapper.model_id,
                                            "dataset_id": request.run.dataset_id, "run_id": request.run.run_id, "train_seed": request.run.train_seed})
        return wrapper._artifacts(request)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _load_degree_sampler(init, state, manifest, device, seed):
    prior = dict(init.get("degree_generator", {}))
    if prior.get("type", "empirical") == "empirical":
        return None, {"type": "empirical_training_degree_sequences", "learned": False}
    from grapher.models.dhvae_hh.degree_vae import load_degree_vae_checkpoint
    from grapher.models.dhvae_hh.degree_sampler import DegreeVAESampler
    path = Path(prior.get("checkpoint_path", ""))
    if not path.is_file():
        raise FileNotFoundError("A learned degree prior was requested. Train scripts/train_degree_generator.py and set initialization.degree_generator.checkpoint_path.")
    # No silent empirical fallback or degree repair in the learned-prior branch.
    if prior.get("fallback", "error") != "error" or prior.get("postprocess_policy", "reject_only") != "reject_only":
        raise ValueError("The learned-degree Structure3 stage requires fallback=error and postprocess_policy=reject_only.")
    degree_model, vectorizer, degree_state = load_degree_vae_checkpoint(path, device=device)
    validate_degree_bank(vectorizer, state["basis_degree_sequences"])
    recorded_hash = degree_state.get("config", {}).get("dataset", {}).get("train_sha256")
    if recorded_hash is not None and recorded_hash != manifest["dataset"]["split_sha256"]["train"]:
        raise ValueError("DH-VAE training split hash differs from the spectral training split.")
    prior.update({"device": str(device), "fallback": "error", "postprocess_policy": "reject_only"})
    sampler = DegreeVAESampler.from_config(prior, seed=seed, model=degree_model, vectorizer=vectorizer)
    return sampler, {"type": "dhvae", "learned": True, "checkpoint_path": str(path.resolve()),
                     "sha256": _sha256(path), "training_degree_multiset_verified": True,
                     "training_split_sha256_verified": recorded_hash is not None,
                     "fallback": "error", "postprocess_policy": "reject_only"}


def _validate_generation_contract(state, options):
    if state.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("Structure3 adds trained conditioning and prediction heads. Retrain with a new run-id; an S0/S1 checkpoint cannot enable this stage.")
    validate_structure_options(options)
    trained_summary, requested_summary = state["summary_config"], options["extensions"]["structural_summary"]
    for key, default in (("graphlet_size", 3), ("clustering_bins", 100), ("orbit_scope", "up_to_3")):
        if trained_summary.get(key, default) != requested_summary.get(key, default):
            raise ValueError(f"Generation cannot change trained structural_summary.{key}.")
    trained, requested = state["initialization"], options["extensions"].get("initialization", {})
    for key, default in (("mode", "degree_basis"), ("ridge", 1e-3), ("diagonal_weight", 1.), ("seed_top_k", None)):
        if trained.get(key, default) != requested.get(key, default):
            raise ValueError(f"Generation cannot change trained initialization.{key}; training corruption must match the sampler.")


@torch.no_grad()
def generate_structured(wrapper, request, state, manifest, options):
    _validate_generation_contract(state, options)
    layout = request.run.layout
    generation_id = request.resolved_generation_id
    target = layout.generation_dir(generation_id)
    ArtifactLayout.require_available(target, overwrite=request.overwrite)
    ext = options["extensions"]
    init, guide = ext.get("initialization", {}), ext.get("structure_guidance", {})
    device = _resolve_device(options.get("runtime", {}))
    _seed_everything(request.generation_seed)
    model = make_model(state["model_config"], state["summary_config"], device)
    model.load_state_dict(state["model_state"]); model.eval()
    dc = state["diffusion"]
    schedule = make_schedule(steps=int(dc["steps"]), beta_start=float(dc["beta_start"]), beta_end=float(dc["beta_end"]), device=device)
    total = len(schedule.alpha_bar)
    sample_cfg = options.get("sample", state["sample"])
    sample_steps = max(2, min(int(sample_cfg["steps"]), total))
    times = torch.linspace(total-1, 0, sample_steps).round().long().unique_consecutive().tolist()
    max_nodes = int(state["max_nodes"])
    batch_size = int(options.get("generation_batch_size", 128))
    if batch_size < 1:
        raise ValueError("generation_batch_size must be positive.")
    basis_rng = np.random.default_rng(request.generation_seed)
    degree_rng = np.random.default_rng(request.generation_seed+1009)
    fit_rng = np.random.default_rng(request.generation_seed+2003)
    rewire_rng = np.random.default_rng(request.generation_seed+1000003)
    generator = torch.Generator(device=device).manual_seed(request.generation_seed)
    degree_sampler, degree_provenance = _load_degree_sampler(init, state, manifest, device, request.generation_seed)
    bank_n = np.asarray(state["basis_num_nodes"], dtype=int)
    bank_d = state["basis_degree_sequences"]
    bank_u = state["basis_eigenvectors"]
    enabled = bool(ext.get("degree_preserving_rewiring", True))
    feedback = float(guide.get("spectrum_feedback", .05)) if enabled else 0.
    start_fraction, every = float(guide.get("start_fraction", .35)), int(guide.get("every", 100))
    connected = bool(init.get("ensure_connected", True))
    outputs, initial_graphs, thresholds, spectra, basis_indices, degree_sequences = [], [], [], [], [], []
    anchors_saved, summaries_saved, diagnostic_rows, intermediate = [], [], [], []
    started = time.monotonic()
    anchor_cache = {}
    while len(outputs) < request.num_graphs:
        b = min(batch_size, request.num_graphs-len(outputs))
        degrees, bases, anchors, conditions, bank_indices, init_diagnostics, degree_diagnostics = [], [], [], [], [], [], []
        for _ in range(b):
            if degree_sampler is None:
                d = np.sort(np.asarray(bank_d[int(degree_rng.integers(len(bank_d)))], dtype=np.int64))[::-1]
                dd = {"source": "empirical_training_split"}
            else:
                sample = degree_sampler.sample(rng=degree_rng)
                d = np.sort(np.asarray(sample["degree_sequence"], dtype=np.int64))[::-1]
                dd = _jsonable(sample.get("sampling_diagnostics", {}))
            n = len(d)
            choices = np.flatnonzero(bank_n == n)
            if n < 2 or n > max_nodes or not len(choices):
                raise ValueError(f"Sampled degree sequence has n={n}, outside the trained eigenbasis support. No silent size fallback is allowed.")
            if not nx.is_graphical(d.tolist()) or (connected and (d.min() < 1 or d.sum() < 2*(n-1))):
                raise ValueError("Degree prior returned a nongraphical or non-connected-feasible sequence.")
            j = int(basis_rng.choice(choices))
            basis = align_basis_rows(bank_u[j], np.asarray(bank_d[j]))
            key = (j, tuple(d))
            if key not in anchor_cache:
                anchor_cache[key] = make_anchor(basis, d, init)
            anchor, ad = anchor_cache[key]
            degrees.append(d); bases.append(basis); anchors.append(anchor)
            conditions.append(conditioning_features(basis, d, anchor, max_nodes))
            bank_indices.append(j); init_diagnostics.append(ad); degree_diagnostics.append(dd)
        sizes = torch.tensor([len(d) for d in degrees], dtype=torch.long, device=device)
        mask = torch.arange(max_nodes, device=device).unsqueeze(0) < sizes.unsqueeze(1)
        anchor_tensor = torch.tensor(np.stack([np.pad(a, (0, max_nodes-len(a))) for a in anchors]), dtype=torch.float32, device=device)
        condition = torch.tensor(np.stack(conditions), dtype=torch.float32, device=device)
        x = (anchor_tensor + torch.randn((b, max_nodes), generator=generator, device=device))*mask
        current, sources, visited = [None]*b, [None]*b, [set() for _ in range(b)]
        events, snapshots, realization = [[] for _ in range(b)], [[] for _ in range(b)], [None]*b
        final_targets, final_raw, final_values = [None]*b, [None]*b, [None]*b
        for position, scalar_t in enumerate(times):
            t = torch.full((b,), scalar_t, dtype=torch.long, device=device)
            eps, prediction = model(x, t, mask, sizes, total, condition)
            x0 = centered_x0(x, eps, t, anchor_tensor, schedule)*mask
            if not torch.isfinite(x0).all():
                raise FloatingPointError(f"Nonfinite spectral prediction at reverse timestep {scalar_t}.")
            last = position == len(times)-1
            due = last or (scalar_t <= start_fraction*(total-1) and position % every == 0)
            if due:
                predicted = {k: v.detach().cpu().numpy() for k, v in summary_probabilities(prediction).items()}
                for row in range(b):
                    n = len(degrees[row])
                    values = np.sort(x0[row, :n].detach().cpu().numpy().astype(np.float64))
                    raw = {k: v[row].astype(np.float64) for k, v in predicted.items()}
                    summary = reconcile_summary3(raw, degrees[row])
                    if current[row] is None:
                        soft = (bases[row] * (values*np.sqrt(n))[None, :]) @ bases[row].T
                        current[row], realization[row] = realize_degrees(degrees[row], soft, ensure_connected=connected, config=guide, rng=fit_rng)
                        sources[row] = current[row].copy()
                    if enabled:
                        current[row], diag = refine_structure3(current[row], values, summary, source=sources[row], config=guide, rng=rewire_rng, visited=visited[row])
                    else:
                        diag = {"accepted_steps": 0, "degree_preserved": True, "all_accepted_steps_improve_energy": True,
                                "all_accepted_steps_improve_structure": True, "enabled": False}
                    diag.update({"timestep": scalar_t, "reverse_step_index": position,
                                 "raw_prediction": _jsonable(raw), "reconciled_target": _jsonable(summary)})
                    apply_feedback = bool(feedback and not last and (
                        diag["accepted_steps"] > 0 or not bool(guide.get("feedback_only_after_accept", True))
                    ))
                    diag["spectrum_feedback_applied"] = apply_feedback
                    events[row].append(diag)
                    if bool(guide.get("save_intermediate_graphs", False)):
                        snapshots[row].append({"timestep": scalar_t, "graph": current[row].copy()})
                    if apply_feedback:
                        discrete_spectrum = torch.tensor(normalized_adjacency_eigenvalues(current[row]), dtype=x0.dtype, device=device)
                        x0[row, :n] = (1-feedback)*x0[row, :n] + feedback*discrete_spectrum
                    final_targets[row], final_raw[row], final_values[row] = summary, raw, values
            if not last:
                previous = torch.full((b,), times[position+1], dtype=torch.long, device=device)
                x = centered_ddim_step(x0, eps, previous, anchor_tensor, schedule)*mask
        from grapher.models.gdsm_simple.structure3 import structure_distances
        for row in range(b):
            g, source, d, values = current[row], sources[row], degrees[row], final_values[row]
            if [g.degree(v) for v in range(len(d))] != d.tolist():
                raise AssertionError("Generation violated sampled indexed degrees.")
            n = len(d)
            soft = (bases[row]*(values*np.sqrt(n))[None, :]) @ bases[row].T
            soft = .5*(soft+soft.T); np.fill_diagonal(soft, 0.)
            binary = (soft > float(sample_cfg.get("threshold", .5))).astype(np.int8)
            np.fill_diagonal(binary, 0)
            thresholds.append(nx.from_numpy_array(binary))
            outputs.append(g); initial_graphs.append(source); spectra.append(values)
            basis_indices.append(bank_indices[row]); degree_sequences.append(d.tolist()); anchors_saved.append(anchors[row])
            summaries_saved.append({"raw": final_raw[row], "reconciled": final_targets[row]})
            bins = int(state["summary_config"].get("clustering_bins", 100))
            diagnostics = {
                "basis_index": bank_indices[row], "initialization": init_diagnostics[row],
                "degree_sampling": degree_diagnostics[row], "realization": realization[row],
                "num_nodes": n, "num_edges": g.number_of_edges(), "degree_preserved": True,
                "initial_connected": nx.is_connected(source), "final_connected": nx.is_connected(g),
                "accepted_steps": sum(e["accepted_steps"] for e in events[row]),
                "num_guidance_events": len(events[row]), "events": events[row],
                "initial_to_final_structure_target": structure_distances(graph_summary3(source, bins), final_targets[row]),
                "final_to_final_structure_target": structure_distances(graph_summary3(g, bins), final_targets[row]),
            }
            diagnostic_rows.append(diagnostics); intermediate.append(snapshots[row])
        print(f"GSDM-Simple Structure3 generated {len(outputs)}/{request.num_graphs}", flush=True)
    aggregate = {
        "enabled": enabled, "mode": "intermediate_structure3", "num_graphs": len(outputs),
        "degree_preservation_rate": float(np.mean([d["degree_preserved"] for d in diagnostic_rows])),
        "initial_connected_rate": float(np.mean([d["initial_connected"] for d in diagnostic_rows])),
        "final_connected_rate": float(np.mean([d["final_connected"] for d in diagnostic_rows])),
        "mean_accepted_steps": float(np.mean([d["accepted_steps"] for d in diagnostic_rows])),
        "mean_guidance_events": float(np.mean([d["num_guidance_events"] for d in diagnostic_rows])),
        "all_accepted_steps_improve_event_energy": all(e["all_accepted_steps_improve_energy"] for d in diagnostic_rows for e in d["events"]),
        "all_accepted_steps_improve_event_structure": all(e["all_accepted_steps_improve_structure"] for d in diagnostic_rows for e in d["events"]),
        "spectrum_feedback": feedback,
        "mean_feedback_events": float(np.mean([sum(e["spectrum_feedback_applied"] for e in d["events"]) for d in diagnostic_rows])),
        "note": "Targets change between DDIM events; energy monotonicity is per frozen-target event, not across the whole trajectory.",
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".gdsm_structure3_generation_", dir=target.parent))
    try:
        objects = {"base_graphs.pkl": outputs, "initial_graphs.pkl": initial_graphs, "threshold_graphs.pkl": thresholds,
                   "target_adjacency_eigenvalues.pkl": spectra, "sampled_basis_indices.pkl": basis_indices,
                   "sampled_degree_sequences.pkl": degree_sequences, "initial_eigenvalue_anchors.pkl": anchors_saved,
                   "predicted_structure_summaries.pkl": summaries_saved}
        if guide.get("save_intermediate_graphs", False): objects["intermediate_graphs.pkl"] = intermediate
        hashes = {}
        for name, value in objects.items():
            with (staging/name).open("wb") as handle: pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
            hashes[name] = _sha256(staging/name)
        _write_json(staging/"rewiring_diagnostics.json", {"aggregate": aggregate, "per_graph": diagnostic_rows})
        _write_json(staging/"manifest.json", {
            "format": "grapher_gdsm_simple_generation_v2", "model_id": wrapper.model_id,
            "variant": "GSDM-Simple-Structure3", "run_id": request.run.run_id, "generation_id": generation_id,
            "generation_seed": request.generation_seed, "num_requested": request.num_graphs, "num_generated": len(outputs),
            "duration_seconds": time.monotonic()-started,
            "base_graphs": {"path": "base_graphs.pkl", "sha256": hashes["base_graphs.pkl"]},
            "initial_graphs": {"path": "initial_graphs.pkl", "role": "degree_exact_realization_at_first_guidance_event"},
            "threshold_graphs": {"path": "threshold_graphs.pkl", "role": "final_spectral_decode_of_this_chain_not_an_independent_S0_baseline"},
            "artifact_sha256": hashes, "checkpoint": {"path": str(request.checkpoint_path.resolve()), "sha256": _sha256(request.checkpoint_path)},
            "dataset": manifest["dataset"], "degree_prior": degree_provenance,
            "empirical_prior": {"eigenvectors": "independent_same_size_training_basis", "test_conditioning": False},
            "initialization": _jsonable(init), "structure_guidance": _jsonable(guide), "structural_rewiring": aggregate,
            "terminal_alpha_bar": float(schedule.alpha_bar[-1]),
            "prior_note": "The terminal Gaussian is the usual small-alpha_bar approximation, centred at the degree/basis anchor.",
            "posthoc_repair": False, "largest_component_filter": False,
            "construction_note": "HH feasibility and connectivity handling occur before intermediate structure-guided swaps; degree-realization fitting is separately logged.",
            "structure_schema": {"graphlet_size": 3, "graphlet_order": ["induced_P3", "triangle"],
                                 "orbit_order": ["edge_endpoint", "P3_endpoint", "P3_centre", "triangle_vertex"],
                                 "orbit_statistic": "log1p(mean_node_orbit_count)",
                                 "count_reconciliation": "necessary_degree_identities_not_a_graph_realizability_guarantee"},
        })
        if target.exists(): shutil.rmtree(target)
        staging.replace(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return GenerationArtifacts(run_dir=layout.run_dir, generation_dir=target, graphs_path=target/"base_graphs.pkl",
                               manifest_path=target/"manifest.json", num_requested=request.num_graphs,
                               num_generated=len(outputs), graphs_sha256=hashes["base_graphs.pkl"])

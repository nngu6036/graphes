"""Graph-balanced training of the joint DH-VAE and spectral-summary model.

One optimizer step uses a minibatch of real graphs. Degree loss is evaluated once
per graph. Bridge-view structural losses are averaged per graph, then across the
minibatch, so more bridge samples do not multiply the degree loss. All structural
labels belong to the exact degree sequence of the corresponding real graph.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any

import networkx as nx
import numpy as np
import torch

from grapher.data.io import load_dataset_splits
from grapher.models.dhvae_hh.degree_vae import (
    DegreeHistogramVAE, DegreeVectorizer, build_degree_vae,
    load_degree_vae_checkpoint, save_degree_vae_checkpoint,
)
from grapher.properties.summary import SummaryConfig
from grapher.rewiring_mlp.generic.joint_degree_model import (
    JointDegreeSpectralPredictor, exact_degree_inputs,
)
from grapher.rewiring_mlp.generic.spectral_data import (
    _prepare_spectral_diffusion_endpoint, _sample_spectral_diffusion_endpoint_examples,
    collate_spectral_examples,
)
from grapher.rewiring_mlp.generic.spectral_model import (
    load_topology_spectral_checkpoint, save_topology_spectral_checkpoint,
)
from grapher.utils.device import resolve_torch_device
from grapher.utils.io import ensure_dir, save_json


def graph_fingerprint(graphs) -> str:
    """Ordered simple-graph identity, not an isomorphism hash."""
    rows = []
    for graph in graphs:
        nodes = list(graph.nodes())
        index = {v: i for i, v in enumerate(nodes)}
        edges = sorted(tuple(sorted((index[u], index[v]))) for u, v in graph.edges())
        rows.append([len(nodes), edges])
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def _finite_nonnegative(value, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return value


def _limit(items, value):
    return list(items) if value is None or int(value) <= 0 else list(items)[:int(value)]


def _degree_edge_support_bound(vectorizer: DegreeVectorizer) -> int:
    """Cover simple graphs within the existing node/degree support, without held-out data."""
    degree_bound = min(vectorizer.max_degree, vectorizer.max_nodes - 1)
    return max(vectorizer.max_edges, vectorizer.max_nodes * degree_bound // 2)


@torch.no_grad()
def _expand_degree_edge_support(
    model: DegreeHistogramVAE, vectorizer: DegreeVectorizer,
) -> None:
    """Expand a warm-start edge head while preserving its learned conditioning.

    Existing edge logits are copied exactly. New classes share at most 1e-6
    initial probability per supported size in eval mode, so expanding support
    does not replace the pretrained sampling distribution with random logits.
    The new classes remain trainable and have finite likelihoods.
    """
    old_max = vectorizer.max_edges
    new_max = _degree_edge_support_bound(vectorizer)
    if model.max_edges != old_max:
        raise ValueError("Warm-start DH-VAE model and vectorizer edge supports disagree.")
    if new_max == old_max:
        return
    if model.use_edge_count_conditioning:
        old_head = model.num_edges_head
        if old_head.out_features != old_max + 1:
            raise ValueError("Warm-start DH-VAE edge head and vectorizer supports disagree.")
        training = model.training
        model.eval()
        sizes = torch.arange(
            vectorizer.min_nodes, vectorizer.max_nodes + 1,
            device=old_head.weight.device,
        )
        log_normalizer = torch.logsumexp(model.edge_count_logits(sizes), dim=-1).min()
        model.train(training)
        head = torch.nn.Linear(old_head.in_features, new_max + 1).to(old_head.weight)
        head.weight[:old_max + 1].copy_(old_head.weight)
        head.bias[:old_max + 1].copy_(old_head.bias)
        head.weight[old_max + 1:].zero_()
        head.bias[old_max + 1:].fill_(
            float(log_normalizer) + math.log(1e-6 / (new_max - old_max))
        )
        head.train(old_head.training)
        model.num_edges_head = head
        # _edge_features uses m/max_edges. Rescale its input weights so
        # explicit (n,m) conditioning is unchanged by the new denominator.
        scale = max(new_max, 1) / max(old_max, 1)
        model.edge_encoder.net[0].weight[:, 0].mul_(scale)
        if model.prior_condition_on_edges:
            model.conditional_prior.net[0].weight[:, 2].mul_(scale)
    model.max_edges = vectorizer.max_edges = new_max
    model.head_dims["num_edges"] = new_max + 1


def build_joint_model(config, train_graphs, *, degree_provenance_graphs=None):
    """Initialize from trusted component checkpoints or explicitly from scratch."""
    joint = dict(config.get("joint_degree", {}) or {})
    prior_cfg = dict(joint.get("degree_model", {}) or {})
    prior_path = joint.get("initialize_degree_checkpoint")
    if prior_path:
        if not Path(prior_path).is_file():
            raise FileNotFoundError(
                f"DH-VAE warm-start checkpoint not found: {prior_path}. "
                "Train the canonical DH-VAE first, or explicitly set joint_degree.initialize_degree_checkpoint=null."
            )
        degree_model, vectorizer, _ = load_degree_vae_checkpoint(prior_path, device="cpu")
        if bool(joint.get("verify_degree_training_sequences", True)):
            sequences = vectorizer.empirical_degree_sequences
            reference = train_graphs if degree_provenance_graphs is None else degree_provenance_graphs
            if sequences is None:
                raise ValueError("DH-VAE lacks degree provenance; cannot verify its training sequences.")
            expected = Counter(tuple(sorted(d for _, d in g.degree())) for g in reference)
            recorded = Counter(tuple(sorted(map(int, d))) for d in sequences)
            if expected != recorded:
                raise ValueError(
                    "Warm-start DH-VAE training degree multisets differ from the current full training split. "
                    "Check dataset provenance; never silently use an old prior. For an intentional transfer only, "
                    "set joint_degree.verify_degree_training_sequences=false."
                )
        initial_max_edges = vectorizer.max_edges
        _expand_degree_edge_support(degree_model, vectorizer)
    else:
        max_degree = joint.get("max_degree")
        if max_degree is None:
            max_degree = max(g.number_of_nodes() for g in train_graphs) - 1
        vectorizer = DegreeVectorizer.fit(
            train_graphs, max_degree=int(max_degree), require_connected=True,
        )
        initial_max_edges = vectorizer.max_edges
        vectorizer.max_edges = _degree_edge_support_bound(vectorizer)
        degree_model = build_degree_vae(vectorizer, **prior_cfg)
    # Always use this experiment's training-split size distribution. No held-out
    # graph may define the prior's empirical support/probabilities.
    vectorizer.empirical_node_counts = [g.number_of_nodes() for g in train_graphs]
    vectorizer.empirical_edge_counts = [g.number_of_edges() for g in train_graphs]
    vectorizer.empirical_degree_sequences = [sorted((d for _, d in g.degree()), reverse=True) for g in train_graphs]

    pcfg = dict(config.get("topology_predictor", {}) or {})
    summaries = dict(config.get("structure_summary_prediction", {}) or {})
    from grapher.rewiring_mlp.generic.spectral_model import TopologySpectralTransformerPredictor
    import inspect
    # Only constructor architecture keys; training and path keys are excluded.
    architecture_names = set(inspect.signature(TopologySpectralTransformerPredictor.__init__).parameters) - {"self"}
    kwargs = {key: value for key, value in pcfg.items() if key in architecture_names}
    kwargs.update(
        predict_clustering_coefficient=bool(summaries.get("clustering_coefficient", False)),
        predict_clustering_histogram=bool(summaries.get("clustering_histogram", False)),
        clustering_histogram_bins=int(summaries.get("clustering_bins", 100)),
        predict_orbit_summary=bool(summaries.get("orbit_summary", False)),
        orbit_summary_width=int(summaries.get("orbit_width", 15)),
        predict_cycle_graphlet_histogram=bool(summaries.get("cycle_graphlet_histogram", False)),
        cycle_graphlet_k=int(summaries.get("cycle_graphlet_k", 3)),
    )
    model = JointDegreeSpectralPredictor(
        joint_degree_config={
            "version": 1,
            "model_config": degree_model.model_config(),
            "vectorizer": deepcopy(vectorizer.__dict__),
            "conditioning_dim": int(joint.get("conditioning_dim", 128)),
            "orbit_consistency": str(joint.get("orbit_consistency", "degree_identities")),
        }, **kwargs,
    )
    model.degree_model.load_state_dict(degree_model.state_dict())
    warm_report = {"degree_checkpoint": str(prior_path) if prior_path else None,
                   "topology_checkpoint": None, "skipped_source_parameters": [],
                   "degree_edge_support": {
                       "initial_max_edges": initial_max_edges,
                       "max_edges": vectorizer.max_edges,
                       "policy": "simple_graph_bound_from_existing_node_and_degree_support",
                       "expanded": vectorizer.max_edges > initial_max_edges,
                       "new_class_initial_probability_mass_bound": (
                           1e-6 if prior_path and degree_model.use_edge_count_conditioning
                           and vectorizer.max_edges > initial_max_edges else None
                       ),
                   }}
    topology_path = joint.get("initialize_topology_checkpoint")
    if topology_path:
        if not Path(topology_path).is_file():
            raise FileNotFoundError(
                f"GraphER warm-start checkpoint not found: {topology_path}. "
                "Set joint_degree.initialize_topology_checkpoint=null for fresh structural training."
            )
        source, _, checkpoint = load_topology_spectral_checkpoint(topology_path, device="cpu")
        if getattr(source, "joint_degree_enabled", False):
            raise ValueError("Initialization expects an ordinary spectral checkpoint, not a joint model/resume checkpoint.")
        for key in architecture_names - {"self"}:
            # Predict-head booleans may differ (the new default disables cycles).
            if key.startswith("predict_") or key in {"cycle_graphlet_k", "clustering_histogram_bins", "orbit_summary_width"}:
                continue
            if source.model_config().get(key) != model.model_config().get(key):
                raise ValueError(f"Warm-start topology architecture mismatch for {key}; use the matching config or disable initialization.")
        if model.predict_clustering_histogram and source.predict_clustering_histogram and model.clustering_histogram_bins != source.clustering_histogram_bins:
            raise ValueError("Warm-start histogram bin count differs.")
        if model.predict_cycle_graphlet_histogram and source.predict_cycle_graphlet_histogram and model.cycle_graphlet_k != source.cycle_graphlet_k:
            raise ValueError("Cannot warm-start a different cycle class despite its identical two-bin tensor shape.")
        destination = model.state_dict()
        weights = {}
        for key, value in source.state_dict().items():
            if key not in destination:
                warm_report["skipped_source_parameters"].append(key)
                continue
            if destination[key].shape != value.shape:
                raise ValueError(f"Warm-start shape mismatch: {key}")
            weights[key] = value
        result = model.load_state_dict(weights, strict=False)
        allowed_missing = ("degree_model.", "degree_conditioner.", "cycle_graphlet_histogram_head.",
                           "clustering_coefficient_head.", "clustering_histogram_head.", "orbit_summary_head.")
        unexpected_missing = [key for key in result.missing_keys if not key.startswith(allowed_missing)]
        if result.unexpected_keys or unexpected_missing:
            raise ValueError(f"Unexpected warm-start state mismatch: {unexpected_missing}, {result.unexpected_keys}")
        warm_report.update(topology_checkpoint=str(topology_path),
                           new_parameters=list(result.missing_keys), source_format=checkpoint["format"])
    return model, warm_report


def _endpoints(graphs, config, seed):
    diffusion = dict(config.get("summary_diffusion", {}) or {})
    constructor = dict(config.get("constructor", {}) or {})
    source_config = dict(
        ensure_connected_source=True,
        random_relabel_source=bool(diffusion.get("random_relabel_source", constructor.get("random_relabel", True))),
        max_repair_trials=int(diffusion.get("max_repair_trials", constructor.get("max_repair_trials", 10000))),
        source_randomization_steps=0,
    )
    return [
        _prepare_spectral_diffusion_endpoint(
            graph, source_config=source_config,
            spectral_config=dict(config.get("spectral_prediction", {}) or {}),
            graphlet_basis=None, graphlet_logit_epsilon=1e-5,
            require_same_degree_sequence=True,
            rng=np.random.default_rng(seed + 10007*i),
            structure_summary_config=config.get("structure_summary_prediction", {}),
        ) for i, graph in enumerate(graphs)
    ]


def run_joint_epoch(model, endpoints, *, config, epoch, seed, device, optimizer=None):
    """Graph-balanced objective, fixed validation randomness, no discrete gradients."""
    train = optimizer is not None
    model.train(train)
    joint = config["joint_degree"]
    structural_weights = dict(config["topology_predictor"].get("loss_weights", {}) or {})
    weights = dict(joint.get("degree_loss_weights", {}) or {})
    beta = float(joint.get("kl_weight", 0.005))
    warmup = int(joint.get("kl_warmup_epochs", 0))
    if train and warmup > 0:
        beta *= min(epoch / warmup, 1.0)
    coefficient = float(joint.get("loss_weight", 0.01))
    batch_size = int(joint.get("graphs_per_batch", 8))
    indices = np.arange(len(endpoints))
    if train:
        np.random.default_rng(seed + epoch*1000003).shuffle(indices)
    rows = []
    effective_epoch = epoch if train else 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for start in range(0, len(indices), batch_size):
            ids = indices[start:start+batch_size]
            views = []
            for index in ids:
                examples, _ = _sample_spectral_diffusion_endpoint_examples(
                    endpoints[int(index)], diffusion_config=config.get("summary_diffusion", {}),
                    graphlet_basis=None,
                    seed=seed + effective_epoch*1000003 + int(index)*10007,
                )
                views.append(examples)
            # One real-degree sample per graph, not one per noisy bridge view.
            degree_batch = collate_spectral_examples([examples[0] for examples in views]).to(device)
            if train:
                optimizer.zero_grad(set_to_none=True)
            degree_loss, degree_metrics = model.degree_loss(
                degree_batch, beta=beta, weights=weights,
                prior_distribution_sigma=float(joint.get("prior_distribution_sigma", 0.20)),
            )
            if not torch.isfinite(degree_loss):
                raise FloatingPointError("Non-finite joint degree objective.")
            if train and degree_loss.requires_grad:
                (coefficient * degree_loss).backward()
            for examples in views:
                batch = collate_spectral_examples(examples).to(device)
                structure_loss, metrics = model.loss(batch, loss_weights=structural_weights)
                if not torch.isfinite(structure_loss):
                    raise FloatingPointError("Non-finite structural objective.")
                if train:
                    (structure_loss / len(ids)).backward()
                rows.append(dict(
                    **metrics,
                    **{f"degree_{key}": value for key, value in degree_metrics.items()},
                    structure_loss=metrics["loss"],
                    joint_loss=metrics["loss"] + coefficient*degree_metrics["loss"],
                ))
            if train:
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(joint.get("gradient_clip", 5.0)))
                if not torch.isfinite(gradient_norm):
                    raise FloatingPointError("Non-finite joint gradient norm.")
                optimizer.step()
    if not rows:
        raise RuntimeError("No joint training/validation graphs were produced.")
    common = set.intersection(*(set(row) for row in rows))
    result = {key: float(np.mean([row[key] for row in rows])) for key in sorted(common)}
    # External spectral diagnostic keeps using model.loss (structural-only).
    result["loss"] = result["joint_loss"]
    result["num_graphs"] = len(endpoints)
    result["num_bridge_views"] = len(endpoints)*int(config["summary_diffusion"].get("samples_per_graph", 32))*int(config["summary_diffusion"].get("paths_per_graph", 1))
    result["degree_loss_evaluations"] = len(endpoints)
    result["num_graph_minibatches"] = math.ceil(len(endpoints) / batch_size)
    result["optimizer_steps"] = result["num_graph_minibatches"] if train else 0
    return result


def train_joint_degree_grapher(config: dict[str, Any], args) -> None:
    """Entry point dispatched by scripts/train_topology_grapher.py."""
    config = deepcopy(config)
    joint = config["joint_degree"]
    pcfg = config.setdefault("topology_predictor", {})
    if str(config.get("pipeline", {}).get("stage", "topology")) != "topology" or config.get("categorical_state") or config.get("molecular_generation"):
        raise ValueError("Joint degree integration currently supports simple unattributed topology graphs only.")
    degree_type = str(config.get("degree_generator", {}).get("type", "degree_histogram_vae"))
    if "typed" in degree_type:
        raise ValueError("Joint ordinary-degree training cannot use a typed-degree prior.")
    if pcfg.get("type", "spectral_transformer") not in {"spectral", "spectral_transformer", "spectrum_transformer"}:
        raise ValueError("Joint degree integration requires the spectral_transformer family.")
    if config.get("source_enrichment", {}).get("enabled", False):
        raise ValueError("Disable source enrichment for the joint degree experiment.")
    if config.get("training_sources", {}).get("mode") != "target_degree_havel_hakimi":
        raise ValueError("Joint structural targets require target_degree_havel_hakimi training sources.")
    if not config.get("constructor", {}).get("ensure_connected", True):
        raise ValueError("Joint degree topology requires connected HH construction.")
    diffusion = config.setdefault("summary_diffusion", {})
    if int(diffusion.get("source_randomization_steps", 0)) != 0:
        raise ValueError("This joint trainer uses fixed HH endpoints: set source_randomization_steps=0.")
    for key in ("samples_per_graph", "paths_per_graph"):
        if int(diffusion.get(key, 1)) < 1:
            raise ValueError(f"summary_diffusion.{key} must be positive.")
    if args.batch_size is not None:
        joint["graphs_per_batch"] = int(args.batch_size)
    if int(joint.get("graphs_per_batch", 8)) < 1:
        raise ValueError("joint_degree.graphs_per_batch must be positive (batch size counts real graphs).")
    for key in ("loss_weight", "kl_weight", "learning_rate", "gradient_clip"):
        if key in joint:
            _finite_nonnegative(joint[key], f"joint_degree.{key}")
    for values in (joint.get("degree_loss_weights", {}), pcfg.get("loss_weights", {})):
        for key, value in values.items():
            _finite_nonnegative(value, f"loss weight {key}")
    if float(joint.get("loss_weight", 0.01)) <= 0:
        raise ValueError("A positive joint_degree.loss_weight is required; freeze the prior for a conditioning-only ablation.")
    seed = int(args.seed if args.seed is not None else config.get("seed", 42))
    config["seed"] = seed
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = resolve_torch_device(args.device or pcfg.get("device", "auto"))
    dataset_cfg = dict(config.get("dataset", {}))
    splits = load_dataset_splits(
        str(dataset_cfg.get("name", "sbm")), root=dataset_cfg.get("root", "outputs/datasets"),
        build_if_missing=bool(dataset_cfg.get("build_if_missing", False)), config_path=dataset_cfg.get("config_path"),
    )
    train_graphs = _limit(splits["train"], args.max_train_graphs if args.max_train_graphs is not None else dataset_cfg.get("max_train_graphs"))
    val_graphs = _limit(splits.get("val", []), args.max_val_graphs if args.max_val_graphs is not None else dataset_cfg.get("max_val_graphs"))
    if not train_graphs or not val_graphs:
        raise ValueError("Joint training requires explicit nonempty train AND validation splits; test is never a fallback.")
    for graph in train_graphs + val_graphs:
        if graph.is_directed() or graph.is_multigraph() or nx.number_of_selfloops(graph) or not graph.number_of_nodes() or not nx.is_connected(graph):
            raise ValueError("Joint training requires nonempty simple connected undirected graphs.")
    epochs = int(args.epochs if args.epochs is not None else pcfg.get("epochs", 200))
    freeze_epochs = int(joint.get("freeze_epochs", 10))
    trainable = bool(joint.get("trainable", True))
    if epochs < 1 or freeze_epochs < 0:
        raise ValueError("Positive epochs and nonnegative freeze_epochs are required.")
    if trainable and freeze_epochs >= epochs:
        raise ValueError("Joint fine-tuning needs epochs > joint_degree.freeze_epochs; use --set joint_degree.freeze_epochs=0 for a short smoke test.")
    config["topology_predictor"]["epochs"] = epochs
    output = ensure_dir(args.output_dir or Path(pcfg["checkpoint_path"]).parent)
    path = Path(output) / "checkpoint.pt"
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite joint checkpoint {path}; use a new output directory.")
    model, warm_report = build_joint_model(config, train_graphs, degree_provenance_graphs=list(splits["train"]))
    model.to(device)
    support = warm_report["degree_edge_support"]
    if support["expanded"]:
        print(f"[GraphER/JointDegree] expanded DH-VAE edge support "
              f"max_edges={support['initial_max_edges']} -> {support['max_edges']} "
              "from existing node/degree limits; empirical prior uses training graphs only.", flush=True)
    print(f"[GraphER/JointDegree] embedded DH-VAE; train={len(train_graphs)} val={len(val_graphs)} "
          f"graphs_per_batch={joint.get('graphs_per_batch',8)}; freeze_epochs={freeze_epochs}; degree_trainable={trainable}", flush=True)
    print("[GraphER/JointDegree] conditioning=actual degree histogram -> posterior mean -> decoder features; "
          "same path at train/inference; HH/swaps are not differentiated.", flush=True)
    print(f"[GraphER/JointDegree] orbit consistency={model.orbit_consistency}; "
          f"selection=val_structure_loss + {joint.get('loss_weight',0.01)} * val_degree_loss", flush=True)
    train_endpoints = _endpoints(train_graphs, config, seed)
    val_endpoints = _endpoints(val_graphs, config, seed + 1)
    # Check held-out sizes/degrees BEFORE training. Never clip unknown degrees.
    for endpoint in train_endpoints + val_endpoints:
        examples, _ = _sample_spectral_diffusion_endpoint_examples(
            endpoint, diffusion_config={**diffusion, "samples_per_graph": 1, "paths_per_graph": 1},
            graphlet_basis=None, seed=seed,
        )
        exact_degree_inputs(collate_spectral_examples(examples).to(device), model.degree_vectorizer)
    degree_parameters = list(model.degree_model.parameters())
    degree_ids = {id(p) for p in degree_parameters}
    structural_parameters = [p for p in model.parameters() if id(p) not in degree_ids]
    optimizer = torch.optim.AdamW([
        {"params": structural_parameters, "lr": float(pcfg.get("learning_rate", 1e-4)), "name": "structure"},
        {"params": degree_parameters, "lr": float(joint.get("learning_rate", 2e-5)), "name": "degree"},
    ], weight_decay=float(pcfg.get("weight_decay", 1e-5)))
    summary_config = SummaryConfig.from_dict(config.get("graphlet_prediction", {}) or {}, train_graphs)
    dataset_fingerprints = {name: graph_fingerprint(list(splits.get(name, []))) for name in ("train", "val", "test")}
    initial_degree_state = {k: v.detach().cpu().clone() for k, v in model.degree_model.state_dict().items()}
    history = []
    best = float("inf"); best_epoch = 0
    progress = max(1, int(pcfg.get("progress_interval", 5)))
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    for epoch in range(1, epochs + 1):
        degree_active = trainable and epoch > freeze_epochs
        model.set_degree_trainable(degree_active)
        train_metrics = run_joint_epoch(model, train_endpoints, config=config, epoch=epoch, seed=seed, device=device, optimizer=optimizer)
        # Validation uses fixed bridge AND latent draws and does not consume the
        # training random stream; changing evaluation frequency cannot change it.
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(seed + 424242)
            val_metrics = run_joint_epoch(model, val_endpoints, config=config, epoch=0, seed=seed+1, device=device)
        row = {
            "epoch": epoch, "degree_trainable": degree_active,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        # A jointly fine-tuned run never silently returns a warm-up-only model.
        eligible = degree_active or not trainable
        if eligible and val_metrics["joint_loss"] < best:
            best = val_metrics["joint_loss"]; best_epoch = epoch
            parameter_delta = math.sqrt(sum(float(((v.detach().cpu()-initial_degree_state[k]).double()**2).sum()) for k, v in model.degree_model.state_dict().items()))
            saved_report = {
                **row, "joint_degree_enabled": True,
                "degree_conditioning": "reencoded_actual_histogram_posterior_mean_and_decoder_hidden",
                "orbit_consistency": model.orbit_consistency,
                "degree_parameter_l2_change_from_initialization": parameter_delta,
                "warm_start": warm_report,
                "dataset_graph_fingerprints": dataset_fingerprints,
                "objective": "graph_balanced_structural_mean_plus_weighted_degree_mean",
            }
            temp = path.with_suffix(".pt.tmp")
            save_topology_spectral_checkpoint(model, temp, summary_config=summary_config, config=config, report=saved_report)
            temp.replace(path)
            # Standalone export for the existing degree evaluator. Generation
            # NEVER loads this file: it uses the embedded degree model above.
            degree_export = Path(output) / "degree_checkpoint.pt"
            temp_degree = degree_export.with_suffix(".pt.tmp")
            save_degree_vae_checkpoint(temp_degree, model.degree_model, model.degree_vectorizer,
                                      config=config, metrics={"joint_epoch": epoch, "joint_checkpoint": str(path), **val_metrics})
            temp_degree.replace(degree_export)
        if epoch == 1 or epoch % progress == 0 or epoch == epochs or epoch == freeze_epochs + 1:
            print(
                f"[GraphER/JointDegree] epoch={epoch}/{epochs} degree_trainable={degree_active} "
                f"val_joint={val_metrics['joint_loss']:.6f} val_degree={val_metrics['degree_loss']:.6f} "
                f"val_structure={val_metrics['structure_loss']:.6f} "
                f"hist_w1={val_metrics.get('clustering_histogram_w1',float('nan')):.6f} "
                f"orbit_log_rmse={val_metrics.get('orbit_summary_log_rmse',float('nan')):.6f} "
                f"identity_max_abs={val_metrics.get('orbit_identity_max_abs',float('nan')):.3e}", flush=True,
            )
        save_json(history, Path(output)/"history.json")
    save_json({
        "joint_degree_enabled": True, "config": config, "config_overrides": args.config_overrides,
        "best_epoch": best_epoch, "best_val_joint_loss": best, "checkpoint": str(path),
        "degree_checkpoint": str(Path(output)/"degree_checkpoint.pt"),
        "warm_start": warm_report, "dataset_graph_fingerprints": dataset_fingerprints,
        "history": history,
    }, Path(output)/"report.json")
    print(f"Saved joint checkpoint: {path}", flush=True)
    print(f"Saved matching degree export: {Path(output)/'degree_checkpoint.pt'}", flush=True)

#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
import random
import threading
import time
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from grapher.data.io import load_dataset_splits
from grapher.properties.summary import SummaryConfig
from grapher.rewiring_mlp.attributed.data import GraphCategoryVocabulary, GraphletBasis
from grapher.rewiring_mlp.attributed.spectral_data import (
    AttributedSpectralDiffusionIterableDataset,
    build_attributed_spectral_diffusion_examples,
    collate_attributed_spectral_examples,
)
from grapher.rewiring_mlp.attributed.spectral_model import (
    ATTRIBUTED_SPECTRAL_GRAPHLET_CHECKPOINT_FORMAT,
    AttributedSpectralGraphletTransformerPredictor,
    save_attributed_spectral_graphlet_checkpoint,
)
from grapher.utils.device import resolve_torch_device
from grapher.utils.io import (
    apply_config_overrides,
    ensure_dir,
    load_yaml,
    save_json,
)


def _limited(values: list[Any], limit: int | None) -> list[Any]:
    return values if limit is None or int(limit) <= 0 else values[: int(limit)]


def _estimated_examples(values: Any) -> int:
    estimate = getattr(values, "estimated_examples", None)
    return int(estimate) if estimate is not None else len(values)


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted(set.intersection(*(set(row) for row in rows)))
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def _format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(float(seconds)):
        return "?"
    value = max(float(seconds), 0.0)
    if value < 60.0:
        return f"{value:.2f}s"
    total = int(value)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    return f"{minutes:d}m{secs:02d}s"


def _progress_line(
    *,
    epoch: int,
    total_epochs: int,
    phase: str,
    status: str,
    completed_batches: int,
    expected_batches: int | None,
    active_batch: int | None,
    completed_examples: int,
    expected_examples: int | None,
    elapsed_seconds: float,
    running_loss: float | None,
    last_data_wait_seconds: float | None = None,
    last_host_step_seconds: float | None = None,
) -> str:
    batch_total = "?" if expected_batches is None else str(expected_batches)
    example_total = "?" if expected_examples is None else str(expected_examples)
    active = "" if active_batch is None else f" active_batch={active_batch}"
    percent = ""
    if expected_examples is not None and expected_examples > 0:
        percent = f" ({100.0 * completed_examples / expected_examples:.1f}%)"
    rate = completed_examples / elapsed_seconds if elapsed_seconds > 0.0 else 0.0
    eta: float | None = None
    if expected_examples is not None and rate > 0.0:
        eta = max(expected_examples - completed_examples, 0) / rate
    loss = "" if running_loss is None else f" running_loss={running_loss:.5f}"
    timings = ""
    if last_data_wait_seconds is not None:
        timings += f" last_data_wait={_format_duration(last_data_wait_seconds)}"
    if last_host_step_seconds is not None:
        timings += f" last_host_step={_format_duration(last_host_step_seconds)}"
    return (
        f"[AttributedTraining] epoch={epoch:04d}/{total_epochs:04d} "
        f"phase={phase} status={status} "
        f"batch={completed_batches}/{batch_total}{active} "
        f"examples={completed_examples}/{example_total}{percent}{loss}{timings} "
        f"rate={rate:.2f} examples/s elapsed={_format_duration(elapsed_seconds)} "
        f"eta={_format_duration(eta)}"
    )


def _heartbeat_loop(
    stop: threading.Event,
    lock: threading.Lock,
    state: dict[str, Any],
    *,
    interval_seconds: float,
    epoch: int,
    total_epochs: int,
    phase: str,
    expected_batches: int | None,
    expected_examples: int | None,
    started: float,
) -> None:
    while not stop.wait(interval_seconds):
        with lock:
            if stop.is_set():
                return
            snapshot = dict(state)
            print(
                _progress_line(
                    epoch=epoch,
                    total_epochs=total_epochs,
                    phase=phase,
                    status=str(snapshot["status"]),
                    completed_batches=int(snapshot["completed_batches"]),
                    expected_batches=expected_batches,
                    active_batch=snapshot.get("active_batch"),
                    completed_examples=int(snapshot["completed_examples"]),
                    expected_examples=expected_examples,
                    elapsed_seconds=time.perf_counter() - started,
                    running_loss=snapshot.get("running_loss"),
                ),
                flush=True,
            )


def _run_epoch(
    model: AttributedSpectralGraphletTransformerPredictor,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    loss_weights: dict[str, float],
    phase: str = "train",
    epoch: int = 1,
    total_epochs: int = 1,
    expected_examples: int | None = None,
    expected_batches: int | None = None,
    batch_progress_interval: int = 0,
    progress_interval_seconds: float = 0.0,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    rows: list[dict[str, float]] = []
    completed_examples = 0
    running_loss_sum = 0.0
    started = time.perf_counter()
    progress_enabled = (
        int(batch_progress_interval) > 0 or float(progress_interval_seconds) > 0.0
    )
    state: dict[str, Any] = {
        "status": "waiting_for_first_batch",
        "active_batch": 1,
        "completed_batches": 0,
        "completed_examples": 0,
        "running_loss": None,
    }
    state_lock = threading.Lock()
    heartbeat_stop = threading.Event()
    heartbeat: threading.Thread | None = None
    if progress_enabled:
        print(
            _progress_line(
                epoch=epoch,
                total_epochs=total_epochs,
                phase=phase,
                status="waiting_for_first_batch",
                completed_batches=0,
                expected_batches=expected_batches,
                active_batch=1,
                completed_examples=0,
                expected_examples=expected_examples,
                elapsed_seconds=0.0,
                running_loss=None,
            ),
            flush=True,
        )
    if progress_interval_seconds > 0.0:
        heartbeat = threading.Thread(
            target=_heartbeat_loop,
            args=(heartbeat_stop, state_lock, state),
            kwargs={
                "interval_seconds": float(progress_interval_seconds),
                "epoch": epoch,
                "total_epochs": total_epochs,
                "phase": phase,
                "expected_batches": expected_batches,
                "expected_examples": expected_examples,
                "started": started,
            },
            name=f"attributed-{phase}-progress",
            daemon=True,
        )
        heartbeat.start()
    context = torch.enable_grad() if training else torch.no_grad()
    try:
        iterator = iter(loader)
        batch_index = 0
        with context:
            while True:
                active_batch = batch_index + 1
                with state_lock:
                    state["status"] = "loading_batch"
                    state["active_batch"] = active_batch
                data_started = time.perf_counter()
                try:
                    batch = next(iterator)
                except StopIteration:
                    break
                data_ready = time.perf_counter()
                with state_lock:
                    state["status"] = (
                        "optimizing_batch" if training else "evaluating_batch"
                    )
                batch_index = active_batch
                batch_examples = int(batch.graph_size.shape[0])
                step_started = data_ready
                batch = batch.to(device)
                loss, metrics = model.loss(batch, loss_weights=loss_weights)
                if training:
                    assert optimizer is not None
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                rows.append(metrics)
                completed_examples += batch_examples
                running_loss_sum += float(metrics["loss"])
                running_loss = running_loss_sum / batch_index
                completed_at = time.perf_counter()
                phase_complete = (
                    expected_examples is not None
                    and completed_examples >= expected_examples
                ) or (
                    expected_batches is not None
                    and batch_index >= expected_batches
                )
                periodic_report = (
                    batch_progress_interval > 0
                    and batch_index % batch_progress_interval == 0
                    and not phase_complete
                )
                first_nonfinal_batch = batch_index == 1 and (
                    expected_batches is None or expected_batches > 1
                )
                with state_lock:
                    state.update(
                        {
                            "status": "batch_complete",
                            "active_batch": None,
                            "completed_batches": batch_index,
                            "completed_examples": completed_examples,
                            "running_loss": running_loss,
                        }
                    )
                    if phase_complete:
                        heartbeat_stop.set()
                    if progress_enabled and (
                        first_nonfinal_batch or periodic_report
                    ):
                        print(
                            _progress_line(
                                epoch=epoch,
                                total_epochs=total_epochs,
                                phase=phase,
                                status="batch_complete",
                                completed_batches=batch_index,
                                expected_batches=expected_batches,
                                active_batch=None,
                                completed_examples=completed_examples,
                                expected_examples=expected_examples,
                                elapsed_seconds=completed_at - started,
                                running_loss=running_loss,
                                last_data_wait_seconds=data_ready - data_started,
                                last_host_step_seconds=completed_at - step_started,
                            ),
                            flush=True,
                        )
    except Exception as exc:
        if progress_enabled:
            with state_lock:
                snapshot = dict(state)
                print(
                    _progress_line(
                        epoch=epoch,
                        total_epochs=total_epochs,
                        phase=phase,
                        status=f"failed:{type(exc).__name__}",
                        completed_batches=int(snapshot["completed_batches"]),
                        expected_batches=expected_batches,
                        active_batch=snapshot.get("active_batch"),
                        completed_examples=int(snapshot["completed_examples"]),
                        expected_examples=expected_examples,
                        elapsed_seconds=time.perf_counter() - started,
                        running_loss=snapshot.get("running_loss"),
                    ),
                    flush=True,
                )
        raise
    finally:
        heartbeat_stop.set()
        if heartbeat is not None:
            heartbeat.join()
    if not rows:
        raise RuntimeError("Attributed summary-diffusion dataset produced no examples.")
    if progress_enabled:
        print(
            _progress_line(
                epoch=epoch,
                total_epochs=total_epochs,
                phase=phase,
                status="complete",
                completed_batches=len(rows),
                expected_batches=expected_batches,
                active_batch=None,
                completed_examples=completed_examples,
                expected_examples=expected_examples,
                elapsed_seconds=time.perf_counter() - started,
                running_loss=running_loss_sum / len(rows),
            ),
            flush=True,
        )
    return _mean_metrics(rows)


def _observed_categories(graphs, node_attribute: str, edge_attribute: str):
    nodes = sorted(
        {
            data[node_attribute]
            for graph in graphs
            for _, data in graph.nodes(data=True)
            if node_attribute in data
        },
        key=repr,
    )
    edges = sorted(
        {
            data[edge_attribute]
            for graph in graphs
            for _, _, data in graph.edges(data=True)
            if edge_attribute in data
        },
        key=repr,
    )
    return nodes, edges


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train attributed GraphER from continuous stochastic topology/bond-spectrum "
            "and attributed-graphlet CLR diffusion states. Rewiring is generation-only."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--batch-progress-interval",
        type=int,
        default=None,
        help="Report intra-epoch progress every N completed batches; 0 disables.",
    )
    parser.add_argument(
        "--progress-interval-seconds",
        type=float,
        default=None,
        help="Emit a heartbeat at this maximum silence interval; 0 disables.",
    )
    parser.add_argument("--max-train-graphs", type=int, default=None)
    parser.add_argument("--max-val-graphs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()

    config = load_yaml(args.config)
    apply_config_overrides(config, args.overrides)
    stage = str((config.get("pipeline", {}) or {}).get("stage", "attributed")).lower()
    if stage not in {"attributed", "attributed_topology", "molecular"}:
        raise ValueError("train_attributed_grapher.py requires pipeline.stage: attributed.")
    seed = int(args.seed if args.seed is not None else config.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset_cfg = dict(config.get("dataset", {}) or {})
    splits = load_dataset_splits(
        str(dataset_cfg.get("name", "qm9_attributed")),
        root=dataset_cfg.get("root", "outputs/datasets"),
        build_if_missing=bool(dataset_cfg.get("build_if_missing", False)),
        config_path=dataset_cfg.get("config_path"),
    )
    train_graphs = _limited(
        list(splits["train"]),
        args.max_train_graphs
        if args.max_train_graphs is not None
        else dataset_cfg.get("max_train_graphs"),
    )
    val_graphs = _limited(
        list(splits.get("val", [])) or list(splits["train"])[:1],
        args.max_val_graphs
        if args.max_val_graphs is not None
        else dataset_cfg.get("max_val_graphs"),
    )
    if not train_graphs or not val_graphs:
        raise ValueError("Attributed training and validation splits must be non-empty.")

    categorical_cfg = dict(config.get("categorical_state", {}) or {})
    vocabulary = GraphCategoryVocabulary.from_graphs(train_graphs, categorical_cfg)
    if not vocabulary.node_attribute or not vocabulary.edge_attribute:
        raise ValueError("Attributed training requires node_attribute and edge_attribute.")
    observed_nodes, observed_edges = _observed_categories(
        train_graphs,
        vocabulary.node_attribute,
        vocabulary.edge_attribute,
    )
    if bool(categorical_cfg.get("require_exact_observed_support", True)):
        if tuple(observed_nodes) != tuple(vocabulary.node_values):
            raise ValueError(
                f"Configured node categories {vocabulary.node_values} disagree with observed training support {tuple(observed_nodes)}."
            )
        if tuple(observed_edges) != tuple(vocabulary.edge_values):
            raise ValueError(
                f"Configured edge categories {vocabulary.edge_values} disagree with observed training support {tuple(observed_edges)}. "
                "For the common QM9 heavy-atom split this is normally (1, 2, 3); "
                "native DiGress/DeFoG reserve an aromatic class but observe zero aromatic mass."
            )

    graphlet_cfg = dict(config.get("graphlet_prediction", {}) or {})
    graphlet_cfg.update(
        {
            "attributed": True,
            "node_attribute": vocabulary.node_attribute,
            "edge_attribute": vocabulary.edge_attribute,
        }
    )
    if str(graphlet_cfg.get("attributed_backend", "python")).lower() != "python":
        raise ValueError(
            "The exact stateful attributed graphlet delta currently requires attributed_backend: python."
        )
    summary_cfg = SummaryConfig.from_dict(graphlet_cfg, train_graphs)
    if summary_cfg.graphlet_k_min < 3:
        raise ValueError("Attributed graphlet diffusion requires graphlet_k_min >= 3.")
    basis_limit = graphlet_cfg.get("max_basis_graphs")
    basis_graphs = _limited(
        train_graphs,
        None if basis_limit in {None, "", "none", "None"} else int(basis_limit),
    )
    basis_started = time.perf_counter()
    graphlet_sampling = (
        "exact"
        if graphlet_cfg.get("graphlet_num_samples") in {None, "", "none", "None"}
        else f"samples={int(graphlet_cfg['graphlet_num_samples'])}"
    )
    print(
        "[AttributedSetup] phase=graphlet_basis status=starting "
        f"graphs={len(basis_graphs)} "
        f"k={summary_cfg.graphlet_k_min}..{summary_cfg.graphlet_k_max} "
        f"backend={graphlet_cfg.get('attributed_backend', 'python')} "
        f"sampling={graphlet_sampling}",
        flush=True,
    )
    graphlet_basis = GraphletBasis.fit_from_graphs(
        basis_graphs,
        graphlet_cfg,
        vocabulary=vocabulary,
        attributed=True,
        seed=seed,
    )
    print(
        "[AttributedSetup] phase=graphlet_basis status=complete "
        f"elapsed={_format_duration(time.perf_counter() - basis_started)} basis="
        + ", ".join(
            f"k={size}:classes={len(graphlet_basis.keys_by_k[size])}+disconnected"
            for size in graphlet_basis.sizes
        ),
        flush=True,
    )

    diffusion_cfg = dict(config.get("summary_diffusion", {}) or {})
    source_cfg = dict(config.get("training_sources", {}) or {})
    spectral_cfg = dict(config.get("spectral_prediction", {}) or {})
    graphlet_diffusion_cfg = dict(config.get("graphlet_diffusion", {}) or {})
    epsilon = float(graphlet_diffusion_cfg.get("logit_epsilon", 1.0e-4))
    storage = str(diffusion_cfg.get("storage", "streaming")).lower()
    if storage not in {"streaming", "eager"}:
        raise ValueError("summary_diffusion.storage must be streaming or eager.")
    print(
        f"Preparing {storage} continuous attributed summary-diffusion samples "
        f"(train={len(train_graphs)}, val={len(val_graphs)})...",
        flush=True,
    )
    print(
        "[GraphER/AttributedDiffusionTraining] source endpoint -> stochastic "
        "continuous topology/bond spectra + attributed graphlet CLR bridge -> clean endpoint; "
        "rewiring is not used to construct training states.",
        flush=True,
    )
    if storage == "streaming":
        train_examples = AttributedSpectralDiffusionIterableDataset(
            train_graphs,
            vocabulary=vocabulary,
            graphlet_basis=graphlet_basis,
            diffusion_config=diffusion_cfg,
            source_config=source_cfg,
            spectral_config=spectral_cfg,
            graphlet_logit_epsilon=epsilon,
            seed=seed,
            shuffle_graphs=True,
            cache_namespace="train",
        )
        val_examples = AttributedSpectralDiffusionIterableDataset(
            val_graphs,
            vocabulary=vocabulary,
            graphlet_basis=graphlet_basis,
            diffusion_config=diffusion_cfg,
            source_config=source_cfg,
            spectral_config=spectral_cfg,
            graphlet_logit_epsilon=epsilon,
            seed=seed + 1,
            shuffle_graphs=False,
            cache_namespace="val",
        )
        train_diffusion_report: dict[str, Any] = {"storage": "streaming"}
        val_diffusion_report: dict[str, Any] = {"storage": "streaming"}
    else:
        train_examples, train_diffusion_report = build_attributed_spectral_diffusion_examples(
            train_graphs,
            vocabulary=vocabulary,
            graphlet_basis=graphlet_basis,
            diffusion_config=diffusion_cfg,
            source_config=source_cfg,
            spectral_config=spectral_cfg,
            graphlet_logit_epsilon=epsilon,
            seed=seed,
        )
        val_examples, val_diffusion_report = build_attributed_spectral_diffusion_examples(
            val_graphs,
            vocabulary=vocabulary,
            graphlet_basis=graphlet_basis,
            diffusion_config=diffusion_cfg,
            source_config=source_cfg,
            spectral_config=spectral_cfg,
            graphlet_logit_epsilon=epsilon,
            seed=seed + 1,
        )

    predictor_cfg = dict(config.get("attributed_predictor", {}) or {})
    device = resolve_torch_device(args.device or predictor_cfg.get("device", "auto"))
    source_enrichment_cfg = dict(config.get("source_enrichment", {}) or {})
    invariant_summary_cfg = dict(source_enrichment_cfg.get("summary_estimator", {}) or {})
    model = AttributedSpectralGraphletTransformerPredictor(
        num_node_categories=vocabulary.num_node_categories,
        num_edge_categories=vocabulary.num_edge_categories,
        graphlet_block_widths=graphlet_basis.simplex_block_widths,
        hidden_dim=int(predictor_cfg.get("hidden_dim", 128)),
        edge_dim=int(predictor_cfg.get("edge_dim", 64)),
        graph_dim=int(predictor_cfg.get("graph_dim", 192)),
        num_layers=int(predictor_cfg.get("num_layers", 4)),
        spectral_dim=int(predictor_cfg.get("spectral_dim", 192)),
        spectral_layers=int(predictor_cfg.get("spectral_layers", 4)),
        spectral_heads=int(predictor_cfg.get("spectral_heads", 6)),
        spectral_ff_dim=int(predictor_cfg.get("spectral_ff_dim", 512)),
        graphlet_dim=int(predictor_cfg.get("graphlet_dim", 384)),
        dropout=float(predictor_cfg.get("dropout", 0.05)),
        graphlet_dropout=float(predictor_cfg.get("graphlet_dropout", 0.05)),
        min_gap=float(predictor_cfg.get("min_gap", 1.0e-6)),
        input_normalization=str(
            predictor_cfg.get(
                "input_normalization", spectral_cfg.get("normalization", "mean_degree")
            )
        ),
        graphlet_logit_epsilon=epsilon,
        invariant_summary_enabled=bool(
            invariant_summary_cfg.get(
                "enabled", source_enrichment_cfg.get("enabled", False)
            )
        ),
        invariant_summary_dim=int(invariant_summary_cfg.get("hidden_dim", 256)),
        invariant_summary_layers=int(invariant_summary_cfg.get("layers", 2)),
        invariant_summary_dropout=float(
            invariant_summary_cfg.get("dropout", predictor_cfg.get("dropout", 0.05))
        ),
    ).to(device)
    batch_size = int(
        args.batch_size if args.batch_size is not None else predictor_cfg.get("batch_size", 32)
    )
    collate = partial(collate_attributed_spectral_examples, vocabulary=vocabulary)
    train_loader = DataLoader(
        train_examples,
        batch_size=batch_size,
        shuffle=storage == "eager",
        num_workers=0,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_examples,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(predictor_cfg.get("learning_rate", 3.0e-4)),
        weight_decay=float(predictor_cfg.get("weight_decay", 1.0e-5)),
    )
    loss_weights = {
        str(key): float(value)
        for key, value in (predictor_cfg.get("loss_weights", {}) or {}).items()
    }
    epochs = int(args.epochs if args.epochs is not None else predictor_cfg.get("epochs", 100))
    progress_interval = max(int(predictor_cfg.get("progress_interval", 5)), 1)
    batch_progress_interval = int(
        args.batch_progress_interval
        if args.batch_progress_interval is not None
        else predictor_cfg.get("batch_progress_interval", 0)
    )
    progress_interval_seconds = float(
        args.progress_interval_seconds
        if args.progress_interval_seconds is not None
        else predictor_cfg.get("progress_interval_seconds", 0.0)
    )
    if batch_progress_interval < 0:
        raise ValueError("batch_progress_interval must be non-negative.")
    if (
        not math.isfinite(progress_interval_seconds)
        or progress_interval_seconds < 0.0
    ):
        raise ValueError("progress_interval_seconds must be finite and non-negative.")
    train_expected_examples = _estimated_examples(train_examples)
    val_expected_examples = _estimated_examples(val_examples)
    train_expected_batches = math.ceil(train_expected_examples / batch_size)
    val_expected_batches = math.ceil(val_expected_examples / batch_size)
    output_dir = ensure_dir(
        args.output_dir
        or Path(predictor_cfg.get("checkpoint_path", "outputs/attributed_grapher/qm9/seed_42/checkpoint.pt")).parent
    )
    checkpoint_path = Path(
        predictor_cfg.get("checkpoint_path", output_dir / "checkpoint.pt")
    )
    if args.output_dir is not None:
        checkpoint_path = Path(args.output_dir) / "checkpoint.pt"
    ensure_dir(checkpoint_path.parent)
    print(
        f"Training attributed predictor device={device} batch_size={batch_size} "
        f"epochs={epochs} checkpoint={checkpoint_path}",
        flush=True,
    )
    print(
        "Attributed training workload: "
        f"train_examples={train_expected_examples} "
        f"train_batches={train_expected_batches} "
        f"val_examples={val_expected_examples} val_batches={val_expected_batches} "
        f"batch_progress_interval={batch_progress_interval} "
        f"heartbeat_seconds={progress_interval_seconds:g} num_workers=0",
        flush=True,
    )
    if storage == "streaming":
        if bool(diffusion_cfg.get("cache_endpoints", False)):
            print(
                "[AttributedTraining] streaming endpoint cache enabled: the first pass "
                "constructs typed sources + dual spectra + exact attributed graphlets; "
                "later epochs reload those fixed endpoints and resample only bridge "
                "time/noise (plus shared relabel augmentation).",
                flush=True,
            )
        else:
            print(
                "[AttributedTraining] status=loading_batch covers CPU typed-endpoint "
                "construction, dual spectra, and exact attributed graphlets before "
                "the batch reaches CUDA.",
                flush=True,
            )

    best_loss = float("inf")
    best_epoch = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        if isinstance(train_examples, AttributedSpectralDiffusionIterableDataset):
            train_examples.set_epoch(epoch - 1)
        if isinstance(val_examples, AttributedSpectralDiffusionIterableDataset):
            val_examples.set_epoch(0)
        train_metrics = _run_epoch(
            model,
            train_loader,
            device=device,
            optimizer=optimizer,
            loss_weights=loss_weights,
            phase="train",
            epoch=epoch,
            total_epochs=epochs,
            expected_examples=train_expected_examples,
            expected_batches=train_expected_batches,
            batch_progress_interval=batch_progress_interval,
            progress_interval_seconds=progress_interval_seconds,
        )
        val_metrics = _run_epoch(
            model,
            val_loader,
            device=device,
            optimizer=None,
            loss_weights=loss_weights,
            phase="val",
            epoch=epoch,
            total_epochs=epochs,
            expected_examples=val_expected_examples,
            expected_batches=val_expected_batches,
            batch_progress_interval=batch_progress_interval,
            progress_interval_seconds=progress_interval_seconds,
        )
        row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)
        if val_metrics["loss"] < best_loss:
            best_loss = float(val_metrics["loss"])
            best_epoch = epoch
            save_attributed_spectral_graphlet_checkpoint(
                model,
                checkpoint_path,
                vocabulary=vocabulary,
                graphlet_basis=graphlet_basis,
                summary_config=summary_cfg,
                config=config,
                report=row,
            )
        if epoch == 1 or epoch % progress_interval == 0 or epoch == epochs:
            print(
                f"epoch={epoch:04d} train={train_metrics['loss']:.5f} val={val_metrics['loss']:.5f} "
                f"top_spec={val_metrics['topology_spectral_nrmse']:.5f} "
                f"bond_spec={val_metrics['bond_spectral_nrmse']:.5f} "
                f"glogit={val_metrics['graphlet_logit_rmse']:.5f} "
                f"gprob={val_metrics['graphlet_probability_mae']:.5f}",
                flush=True,
            )

    report = {
        "format": "attributed_spectral_graphlet_training_v1",
        "checkpoint_format": ATTRIBUTED_SPECTRAL_GRAPHLET_CHECKPOINT_FORMAT,
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "num_train_graphs": len(train_graphs),
        "num_val_graphs": len(val_graphs),
        "vocabulary": vocabulary.to_dict(),
        "observed_node_categories": observed_nodes,
        "observed_edge_categories": observed_edges,
        "graphlet_basis": graphlet_basis.to_dict(),
        "num_graphlet_basis_graphs": len(basis_graphs),
        "training_state_source": "continuous_summary_diffusion",
        "rewiring_used_for_training_states": False,
        "spectral_channels": ["topology", "bond_weighted"],
        "train_diffusion": train_diffusion_report,
        "val_diffusion": val_diffusion_report,
        "history": history,
    }
    save_json(report, output_dir / "training_report.json")
    print(f"Saved best checkpoint: {checkpoint_path}", flush=True)
    print(f"Saved report: {output_dir / 'training_report.json'}", flush=True)


if __name__ == "__main__":
    main()

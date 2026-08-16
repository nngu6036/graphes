#!/usr/bin/env python
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from grapher.data.io import load_dataset_splits
from grapher.rewiring_mlp.properties.summary import SummaryConfig
from grapher.rewiring_mlp.generic.basis import TopologyGraphletBasis
from grapher.rewiring_mlp.generic.data import (
    TopologyTrajectoryIterableDataset,
    build_topology_examples,
    collate_topology_examples,
)
from grapher.rewiring_mlp.generic.model import (
    TopologyGraphletPredictor,
    save_topology_checkpoint,
)
from grapher.utils.device import resolve_torch_device
from grapher.utils.io import ensure_dir, load_yaml, save_json


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
    }


def _limited(values: list[Any], limit: int | None) -> list[Any]:
    if limit is None or int(limit) <= 0:
        return values
    return values[: int(limit)]


def _streaming_teacher_report(
    dataset: TopologyTrajectoryIterableDataset,
) -> dict[str, Any]:
    rows = list(dataset.last_diagnostics)
    report: dict[str, Any] = {
        "storage": "streaming",
        "num_graphs_last_epoch": len(rows),
    }
    if not rows:
        return report
    for key in (
        "num_paths",
        "num_examples",
        "mean_initial_graphlet_discrepancy",
        "mean_final_teacher_graphlet_discrepancy",
        "mean_accepted_teacher_steps",
        "teacher_stop_rate",
        "mean_valid_candidates",
    ):
        values = [float(row[key]) for row in rows if key in row]
        if values:
            report[key] = (
                int(sum(values))
                if key in {"num_paths", "num_examples"}
                else float(np.mean(values))
            )
    return report


def _run_epoch(
    model: TopologyGraphletPredictor,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    loss_weights: dict[str, float],
    target_epsilon: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    rows: list[dict[str, float]] = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            batch = batch.to(device)
            loss, metrics = model.loss(
                batch,
                loss_weights=loss_weights,
                target_epsilon=target_epsilon,
            )
            if training:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            rows.append(metrics)
    if not rows:
        raise RuntimeError("The topology trajectory dataset produced no examples.")
    return _mean_metrics(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train the decoupled GraphER topology graphlet predictor. The "
            "degree prior is trained separately with train_degree_generator.py."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-train-graphs", type=int, default=None)
    parser.add_argument("--max-val-graphs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config = load_yaml(args.config)
    pipeline_stage = str(
        (config.get("pipeline", {}) or {}).get("stage", "topology")
    ).lower()
    if pipeline_stage != "topology":
        raise ValueError("train_topology_grapher.py requires pipeline.stage: topology.")
    if config.get("categorical_state") or config.get("endpoint_predictor"):
        raise ValueError(
            "Topology training cannot define categorical_state or endpoint_predictor. "
            "Use the legacy attributed script until the attribute stage is migrated."
        )

    seed = int(args.seed if args.seed is not None else config.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset_cfg = dict(config.get("dataset", {}) or {})
    splits = load_dataset_splits(
        str(dataset_cfg.get("name", "sbm")),
        root=dataset_cfg.get("root", "outputs/datasets"),
        build_if_missing=bool(dataset_cfg.get("build_if_missing", True)),
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
        raise ValueError("Training and validation graph splits must be non-empty.")

    summary_data = dict(config.get("graphlet_prediction", {}) or {})
    if bool(summary_data.get("attributed", False)):
        raise ValueError("The generic topology stage cannot use attributed graphlets.")
    if str(
        summary_data.get("estimator", "exact_connected_local_delta")
    ).lower() != "exact_connected_local_delta":
        raise ValueError(
            "Topology training requires estimator: exact_connected_local_delta."
        )
    summary_data["graphlet_history"] = True
    summary_cfg = SummaryConfig.from_dict(summary_data, train_graphs)
    if summary_cfg.graphlet_k_min < 3:
        raise ValueError("Topology graphlet prediction requires graphlet_k_min >= 3.")
    graphlet_basis = TopologyGraphletBasis.fit_from_graphs(
        train_graphs,
        summary_data,
        attributed=False,
        seed=seed,
    )

    trajectory_cfg = dict(config.get("topology_trajectory", {}) or {})
    if not bool(trajectory_cfg.get("ensure_connected_source", True)) or not bool(
        trajectory_cfg.get("preserve_connectivity", True)
    ):
        raise ValueError(
            "Topology trajectories require ensure_connected_source: true and "
            "preserve_connectivity: true."
        )
    storage_mode = str(trajectory_cfg.get("storage", "eager")).lower()
    if storage_mode not in {"eager", "streaming"}:
        raise ValueError("topology_trajectory.storage must be eager or streaming.")
    print(
        f"Preparing {storage_mode} topology trajectories "
        f"(train={len(train_graphs)}, val={len(val_graphs)})...",
        flush=True,
    )
    if storage_mode == "streaming":
        train_examples = TopologyTrajectoryIterableDataset(
            train_graphs,
            summary_config=summary_cfg,
            graphlet_basis=graphlet_basis,
            trajectory_config=trajectory_cfg,
            seed=seed,
            shuffle_graphs=True,
        )
        val_examples = TopologyTrajectoryIterableDataset(
            val_graphs,
            summary_config=summary_cfg,
            graphlet_basis=graphlet_basis,
            trajectory_config=trajectory_cfg,
            seed=seed + 1,
            shuffle_graphs=False,
        )
        train_teacher_report: dict[str, Any] = {"storage": "streaming"}
        val_teacher_report: dict[str, Any] = {"storage": "streaming"}
        num_train_examples = train_examples.estimated_examples
        num_val_examples = val_examples.estimated_examples
    else:
        train_examples, train_teacher_report = build_topology_examples(
            train_graphs,
            summary_config=summary_cfg,
            graphlet_basis=graphlet_basis,
            trajectory_config=trajectory_cfg,
            seed=seed,
        )
        val_examples, val_teacher_report = build_topology_examples(
            val_graphs,
            summary_config=summary_cfg,
            graphlet_basis=graphlet_basis,
            trajectory_config=trajectory_cfg,
            seed=seed + 1,
        )
        num_train_examples = len(train_examples)
        num_val_examples = len(val_examples)
    print(
        "Topology examples (estimated for streaming): "
        f"train={num_train_examples} val={num_val_examples}",
        flush=True,
    )

    predictor_cfg = dict(config.get("topology_predictor", {}) or {})
    device = resolve_torch_device(args.device or predictor_cfg.get("device", "auto"))
    model = TopologyGraphletPredictor(
        graphlet_slices=graphlet_basis.slices,
        hidden_dim=int(predictor_cfg.get("hidden_dim", 128)),
        edge_dim=int(predictor_cfg.get("edge_dim", 64)),
        graph_dim=int(predictor_cfg.get("graph_dim", 128)),
        num_layers=int(predictor_cfg.get("num_layers", 4)),
        dropout=float(predictor_cfg.get("dropout", 0.0)),
        min_concentration=float(predictor_cfg.get("min_concentration", 0.05)),
        max_concentration=float(predictor_cfg.get("max_concentration", 50.0)),
    ).to(device)
    batch_size = int(
        args.batch_size
        if args.batch_size is not None
        else predictor_cfg.get("batch_size", 4)
    )
    train_loader = DataLoader(
        train_examples,
        batch_size=batch_size,
        shuffle=storage_mode == "eager",
        num_workers=0,
        collate_fn=collate_topology_examples,
    )
    val_loader = DataLoader(
        val_examples,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_topology_examples,
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
    forbidden_loss_keys = {"node", "edge", "consistency"} & set(loss_weights)
    if forbidden_loss_keys:
        raise ValueError(
            "Topology loss cannot contain endpoint terms: "
            f"{sorted(forbidden_loss_keys)}"
        )
    if not any(
        float(loss_weights.get(key, default)) != 0.0
        for key, default in (
            ("graphlet_mean", 1.0),
            ("graphlet_distribution", 0.1),
            ("graphlet_mass", 0.0),
        )
    ):
        raise ValueError("At least one topology graphlet loss must be active.")
    target_epsilon = float(predictor_cfg.get("target_epsilon", 1.0e-5))

    configured_checkpoint = predictor_cfg.get(
        "checkpoint_path",
        "outputs/topology_grapher/sbm/seed_42/checkpoint.pt",
    )
    if args.output_dir:
        output_dir = ensure_dir(args.output_dir)
        checkpoint_path = output_dir / "checkpoint.pt"
    else:
        checkpoint_path = Path(configured_checkpoint)
        output_dir = ensure_dir(checkpoint_path.parent)
    epochs = int(
        args.epochs if args.epochs is not None else predictor_cfg.get("epochs", 100)
    )
    progress_interval = max(int(predictor_cfg.get("progress_interval", 5)), 1)
    history: list[dict[str, Any]] = []
    best_val = float("inf")
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        if isinstance(train_examples, TopologyTrajectoryIterableDataset):
            train_examples.set_epoch(epoch - 1)
        if isinstance(val_examples, TopologyTrajectoryIterableDataset):
            val_examples.set_epoch(0)
        train_metrics = _run_epoch(
            model,
            train_loader,
            device=device,
            optimizer=optimizer,
            loss_weights=loss_weights,
            target_epsilon=target_epsilon,
        )
        val_metrics = _run_epoch(
            model,
            val_loader,
            device=device,
            optimizer=None,
            loss_weights=loss_weights,
            target_epsilon=target_epsilon,
        )
        row: dict[str, Any] = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)
        if val_metrics["loss"] < best_val:
            best_val = float(val_metrics["loss"])
            best_epoch = epoch
            save_topology_checkpoint(
                model,
                checkpoint_path,
                graphlet_basis=graphlet_basis,
                summary_config=summary_cfg,
                config=config,
                report=row,
            )
        if epoch == 1 or epoch % progress_interval == 0 or epoch == epochs:
            print(
                f"epoch={epoch:04d} "
                f"train={train_metrics['loss']:.5f} "
                f"val={val_metrics['loss']:.5f} "
                f"graphlet_mae={val_metrics['graphlet_mae']:.5f} "
                f"mass_mae={val_metrics['graphlet_mass_mae']:.5f}",
                flush=True,
            )

    report = {
        "format": "topology_graphlet_training_v1",
        "pipeline_mode": "topology",
        "checkpoint_format": "topology_graphlet_predictor_v1",
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "num_train_graphs": len(train_graphs),
        "num_val_graphs": len(val_graphs),
        "num_train_examples": num_train_examples,
        "num_val_examples": num_val_examples,
        "train_teacher": (
            _streaming_teacher_report(train_examples)
            if isinstance(train_examples, TopologyTrajectoryIterableDataset)
            else train_teacher_report
        ),
        "val_teacher": (
            _streaming_teacher_report(val_examples)
            if isinstance(val_examples, TopologyTrajectoryIterableDataset)
            else val_teacher_report
        ),
        "graphlet_basis": graphlet_basis.to_dict(),
        "active_losses": sorted(
            key
            for key, default in (
                ("graphlet_mean", 1.0),
                ("graphlet_distribution", 0.1),
                ("graphlet_mass", 0.0),
            )
            if float(loss_weights.get(key, default)) != 0.0
        ),
        "history": history,
    }
    save_json(report, output_dir / "training_report.json")
    print(f"Saved best topology checkpoint: {checkpoint_path}", flush=True)
    print(f"Best epoch={best_epoch} val_loss={best_val:.6f}", flush=True)


if __name__ == "__main__":
    main()

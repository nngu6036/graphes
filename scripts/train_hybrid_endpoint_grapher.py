#!/usr/bin/env python
from __future__ import annotations

import argparse
import random
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from grapher.data.io import load_dataset_splits
from grapher.hybrid.data import (
    GraphCategoryVocabulary,
    GraphletBasis,
    build_endpoint_examples,
    collate_endpoint_examples,
)
from grapher.hybrid.model import (
    HybridEndpointPredictor,
    save_hybrid_endpoint_checkpoint,
)
from grapher.properties.summary import SummaryConfig
from grapher.utils.device import resolve_torch_device
from grapher.utils.io import ensure_dir, load_yaml, save_json


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
    }


def _run_epoch(
    model: HybridEndpointPredictor,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    loss_weights: dict[str, float],
    edge_class_weights: list[float],
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
                edge_class_weights=edge_class_weights,
                target_epsilon=target_epsilon,
            )
            if training:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            rows.append(metrics)
    return _mean_metrics(rows)


def _limited(values: list[Any], limit: int | None) -> list[Any]:
    if limit is None or int(limit) <= 0:
        return values
    return values[: int(limit)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train the CatFlow-inspired GraphER endpoint categorical and "
            "graphlet-histogram predictor."
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
    if config.get("summary_generator") and not bool(
        config.get("legacy_summary_baseline", False)
    ):
        raise ValueError(
            "The hybrid endpoint route must not define summary_generator. "
            "Keep summary-only generation in a separate baseline config."
        )
    seed = int(args.seed if args.seed is not None else config.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset_cfg = config.get("dataset", {}) or {}
    splits = load_dataset_splits(
        str(dataset_cfg.get("name", "sbm")),
        root=dataset_cfg.get("root", "outputs/datasets"),
        build_if_missing=bool(dataset_cfg.get("build_if_missing", True)),
        config_path=dataset_cfg.get("config_path"),
    )
    predictor_cfg = config.get("endpoint_predictor", {}) or {}
    trajectory_cfg = config.get("endpoint_trajectory", {}) or {}
    max_train = (
        args.max_train_graphs
        if args.max_train_graphs is not None
        else dataset_cfg.get("max_train_graphs")
    )
    max_val = (
        args.max_val_graphs
        if args.max_val_graphs is not None
        else dataset_cfg.get("max_val_graphs")
    )
    train_graphs = _limited(list(splits["train"]), max_train)
    val_graphs = _limited(
        list(splits.get("val", [])) or list(splits["train"])[:1],
        max_val,
    )
    if not train_graphs or not val_graphs:
        raise ValueError("Training and validation graph splits must be non-empty.")

    summary_data = dict(config.get("graphlet_prediction", {}) or {})
    summary_data["graphlet_history"] = True
    summary_cfg = SummaryConfig.from_dict(summary_data, train_graphs)
    if summary_cfg.graphlet_k_min < 3:
        raise ValueError("Hybrid graphlet prediction requires graphlet_k_min >= 3.")
    graphlet_basis = GraphletBasis.from_config(summary_cfg)
    vocabulary = GraphCategoryVocabulary.from_graphs(
        train_graphs,
        config.get("categorical_state", {}) or {},
    )

    print(
        "Building aligned endpoint trajectories "
        f"(train={len(train_graphs)}, val={len(val_graphs)})...",
        flush=True,
    )
    train_examples, train_teacher_report = build_endpoint_examples(
        train_graphs,
        summary_config=summary_cfg,
        graphlet_basis=graphlet_basis,
        trajectory_config=trajectory_cfg,
        seed=seed,
    )
    val_examples, val_teacher_report = build_endpoint_examples(
        val_graphs,
        summary_config=summary_cfg,
        graphlet_basis=graphlet_basis,
        trajectory_config=trajectory_cfg,
        seed=seed + 1,
    )
    print(
        f"Endpoint examples: train={len(train_examples)} "
        f"val={len(val_examples)}",
        flush=True,
    )

    device = resolve_torch_device(args.device or predictor_cfg.get("device", "auto"))
    model = HybridEndpointPredictor(
        num_node_categories=vocabulary.num_node_categories,
        num_edge_categories=vocabulary.num_edge_categories,
        graphlet_slices=graphlet_basis.slices,
        hidden_dim=int(predictor_cfg.get("hidden_dim", 128)),
        edge_dim=int(predictor_cfg.get("edge_dim", 64)),
        graph_dim=int(predictor_cfg.get("graph_dim", 128)),
        num_layers=int(predictor_cfg.get("num_layers", 4)),
        dropout=float(predictor_cfg.get("dropout", 0.0)),
        min_concentration=float(
            predictor_cfg.get("min_concentration", 0.05)
        ),
        max_concentration=float(
            predictor_cfg.get("max_concentration", 50.0)
        ),
    ).to(device)

    batch_size = int(
        args.batch_size
        if args.batch_size is not None
        else predictor_cfg.get("batch_size", 4)
    )
    collate = partial(
        collate_endpoint_examples,
        vocabulary=vocabulary,
    )
    train_loader = DataLoader(
        train_examples,
        batch_size=batch_size,
        shuffle=True,
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
        for key, value in (
            predictor_cfg.get("loss_weights", {}) or {}
        ).items()
    }
    configured_class_weights = predictor_cfg.get("edge_class_weights")
    if configured_class_weights is None:
        edge_class_weights = [
            float(predictor_cfg.get("no_edge_class_weight", 0.25))
        ] + [
            float(predictor_cfg.get("present_edge_class_weight", 1.0))
        ] * (vocabulary.num_edge_categories - 1)
    else:
        edge_class_weights = [float(value) for value in configured_class_weights]
    target_epsilon = float(predictor_cfg.get("target_epsilon", 1.0e-5))

    configured_checkpoint = predictor_cfg.get(
        "checkpoint_path",
        "outputs/hybrid_endpoint/sbm/checkpoint.pt",
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
        train_metrics = _run_epoch(
            model,
            train_loader,
            device=device,
            optimizer=optimizer,
            loss_weights=loss_weights,
            edge_class_weights=edge_class_weights,
            target_epsilon=target_epsilon,
        )
        val_metrics = _run_epoch(
            model,
            val_loader,
            device=device,
            optimizer=None,
            loss_weights=loss_weights,
            edge_class_weights=edge_class_weights,
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
            save_hybrid_endpoint_checkpoint(
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
                f"epoch={epoch:04d} "
                f"train={train_metrics['loss']:.5f} "
                f"val={val_metrics['loss']:.5f} "
                f"edge_acc={val_metrics['edge_accuracy']:.4f} "
                f"present_recall={val_metrics['present_edge_recall']:.4f} "
                f"graphlet_mae={val_metrics['graphlet_mae']:.5f}",
                flush=True,
            )

    report = {
        "format": "hybrid_endpoint_graphlet_training_v1",
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "num_train_graphs": len(train_graphs),
        "num_val_graphs": len(val_graphs),
        "num_train_examples": len(train_examples),
        "num_val_examples": len(val_examples),
        "train_teacher": train_teacher_report,
        "val_teacher": val_teacher_report,
        "vocabulary": vocabulary.to_dict(),
        "graphlet_basis": graphlet_basis.to_dict(),
        "history": history,
    }
    save_json(report, output_dir / "training_report.json")
    print(f"Saved best checkpoint: {checkpoint_path}", flush=True)
    print(
        f"Best epoch={best_epoch} val_loss={best_val:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()

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
from grapher.rewiring_mlp.attributed.data import (
    EndpointTrajectoryIterableDataset,
    GraphCategoryVocabulary,
    GraphletBasis,
    HybridEndpointExample,
    build_endpoint_examples,
    collate_endpoint_examples,
)
from grapher.rewiring_mlp.attributed.model import (
    HybridEndpointPredictor,
    save_hybrid_endpoint_checkpoint,
)
from grapher.rewiring_mlp.attributed.refiner import (
    HybridRefinerConfig,
    predict_hybrid_target,
    score_hybrid_candidates,
)
from grapher.rewiring_mlp.attributed.selector import (
    CANDIDATE_FEATURE_NAMES,
    GRAPH_CONTEXT_FEATURE_NAMES,
    LearnedCandidateSelector,
    build_selector_features,
    save_selector_checkpoint,
)
from grapher.properties.summary import SummaryConfig
from grapher.utils.device import resolve_torch_device
from grapher.utils.io import ensure_dir, load_yaml, save_json


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


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


def _selector_training_record(
    example: HybridEndpointExample,
    *,
    predictor: HybridEndpointPredictor,
    vocabulary: GraphCategoryVocabulary,
    graphlet_basis: GraphletBasis,
    summary_config: SummaryConfig,
    refiner_config: HybridRefinerConfig,
    device: torch.device,
    rng: np.random.Generator,
) -> tuple[Any, torch.Tensor] | None:
    actions = list(example.teacher_actions)
    distribution = example.teacher_distribution
    if distribution is None or len(distribution) != len(actions) + 1:
        return None
    prediction = predict_hybrid_target(
        predictor,
        example.current_graph,
        time=float(example.time),
        vocabulary=vocabulary,
        graphlet_basis=graphlet_basis,
        device=device,
        rng=rng,
        sample_endpoint=False,
        sample_graphlet=False,
        endpoint_temperature=1.0,
    )
    rows = score_hybrid_candidates(
        example.current_graph,
        actions,
        prediction,
        vocabulary=vocabulary,
        graphlet_basis=graphlet_basis,
        summary_config=summary_config,
        config=refiner_config,
    )
    retained = [
        index
        for index, row in enumerate(rows)
        if np.isfinite(float(row["hybrid_score"]))
    ]
    retained_rows = [rows[index] for index in retained]
    target = np.zeros(len(retained_rows) + 1, dtype=np.float32)
    for new_index, old_index in enumerate(retained):
        target[new_index] = float(distribution[old_index])
    target[-1] = float(distribution[-1])
    total = float(target.sum())
    if total <= 0.0:
        target[-1] = 1.0
    else:
        target /= total
    features = build_selector_features(
        example.current_graph,
        [row["action"] for row in retained_rows],
        retained_rows,
        graph_diagnostics={
            "time": float(example.time),
            "remaining_step_fraction": float(max(1.0 - example.time, 0.0)),
            "current_energy": 0.0,
        },
        preserve_connectivity=refiner_config.preserve_connectivity,
        validate_actions=True,
        device=device,
    )
    return features, torch.as_tensor(target, dtype=torch.float32, device=device)


def _run_selector_epoch(
    selector: LearnedCandidateSelector,
    examples: Any,
    *,
    predictor: HybridEndpointPredictor,
    vocabulary: GraphCategoryVocabulary,
    graphlet_basis: GraphletBasis,
    summary_config: SummaryConfig,
    refiner_config: HybridRefinerConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    objective: str,
    max_examples: int | None,
    seed: int,
) -> dict[str, float]:
    training = optimizer is not None
    selector.train(training)
    predictor.eval()
    rng = np.random.default_rng(int(seed))
    losses: list[float] = []
    accuracies: list[float] = []
    stop_targets: list[float] = []
    stop_correct: list[float] = []
    for example in examples:
        record = _selector_training_record(
            example,
            predictor=predictor,
            vocabulary=vocabulary,
            graphlet_basis=graphlet_basis,
            summary_config=summary_config,
            refiner_config=refiner_config,
            device=device,
            rng=rng,
        )
        if record is None:
            continue
        features, teacher = record
        logits = selector(features.candidate_features, features.graph_context)
        loss = selector.distribution_loss(
            logits,
            teacher,
            objective=objective,
        )
        if training:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(selector.parameters(), 5.0)
            optimizer.step()
        prediction_index = int(torch.argmax(logits).item())
        target_index = int(torch.argmax(teacher).item())
        stop_index = teacher.numel() - 1
        losses.append(float(loss.detach().cpu()))
        accuracies.append(float(prediction_index == target_index))
        if target_index == stop_index:
            stop_targets.append(1.0)
            stop_correct.append(float(prediction_index == stop_index))
        if max_examples is not None and len(losses) >= int(max_examples):
            break
    if not losses:
        raise RuntimeError(
            "No cached teacher-action distributions were available for selector "
            "training. Enable endpoint_trajectory teacher decisions."
        )
    return {
        "loss": float(np.mean(losses)),
        "accuracy": float(np.mean(accuracies)),
        "stop_accuracy": float(np.mean(stop_correct)) if stop_targets else float("nan"),
        "num_examples": float(len(losses)),
    }


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
    if str((config.get("pipeline", {}) or {}).get("stage", "endpoint")).lower() == (
        "topology"
    ):
        raise ValueError(
            "Topology configs must be trained with "
            "scripts/train_topology_grapher.py."
        )
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
    vocabulary = GraphCategoryVocabulary.from_graphs(
        train_graphs,
        config.get("categorical_state", {}) or {},
    )
    graphlet_basis = GraphletBasis.fit_from_graphs(
        train_graphs,
        summary_data,
        vocabulary=vocabulary,
        seed=seed,
    )

    storage_mode = str(
        trajectory_cfg.get(
            "storage",
            "streaming" if vocabulary.edge_attribute is not None else "eager",
        )
    ).lower()
    if storage_mode not in {"eager", "streaming"}:
        raise ValueError("endpoint_trajectory.storage must be eager or streaming.")
    print(
        f"Preparing {storage_mode} endpoint trajectories "
        f"(train={len(train_graphs)}, val={len(val_graphs)})...",
        flush=True,
    )
    if storage_mode == "streaming":
        train_examples = EndpointTrajectoryIterableDataset(
            train_graphs,
            summary_config=summary_cfg,
            graphlet_basis=graphlet_basis,
            vocabulary=vocabulary,
            trajectory_config=trajectory_cfg,
            seed=seed,
            shuffle_graphs=True,
        )
        val_examples = EndpointTrajectoryIterableDataset(
            val_graphs,
            summary_config=summary_cfg,
            graphlet_basis=graphlet_basis,
            vocabulary=vocabulary,
            trajectory_config=trajectory_cfg,
            seed=seed + 1,
            shuffle_graphs=False,
        )
        train_teacher_report: dict[str, Any] = {"storage": "streaming"}
        val_teacher_report: dict[str, Any] = {"storage": "streaming"}
        num_train_examples = train_examples.estimated_examples
        num_val_examples = val_examples.estimated_examples
    else:
        train_examples, train_teacher_report = build_endpoint_examples(
            train_graphs,
            summary_config=summary_cfg,
            graphlet_basis=graphlet_basis,
            vocabulary=vocabulary,
            trajectory_config=trajectory_cfg,
            seed=seed,
        )
        val_examples, val_teacher_report = build_endpoint_examples(
            val_graphs,
            summary_config=summary_cfg,
            graphlet_basis=graphlet_basis,
            vocabulary=vocabulary,
            trajectory_config=trajectory_cfg,
            seed=seed + 1,
        )
        num_train_examples = len(train_examples)
        num_val_examples = len(val_examples)
    print(
        f"Endpoint examples (estimated for streaming): "
        f"train={num_train_examples} val={num_val_examples}",
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
        min_concentration=float(predictor_cfg.get("min_concentration", 0.05)),
        max_concentration=float(predictor_cfg.get("max_concentration", 50.0)),
        fixed_node_categories=bool(
            predictor_cfg.get(
                "fixed_node_categories",
                vocabulary.node_attribute is not None,
            )
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
        shuffle=storage_mode == "eager",
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
    configured_class_weights = predictor_cfg.get("edge_class_weights")
    if configured_class_weights is None:
        edge_class_weights = [
            float(predictor_cfg.get("no_edge_class_weight", 0.25))
        ] + [float(predictor_cfg.get("present_edge_class_weight", 1.0))] * (
            vocabulary.num_edge_categories - 1
        )
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
        if isinstance(train_examples, EndpointTrajectoryIterableDataset):
            train_examples.set_epoch(epoch - 1)
        if isinstance(val_examples, EndpointTrajectoryIterableDataset):
            val_examples.set_epoch(0)
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

    selector_report: dict[str, Any] | None = None
    refiner_cfg = HybridRefinerConfig.from_dict(config.get("hybrid_refiner", {}) or {})
    selector_cfg = config.get("candidate_selector", {}) or {}
    train_selector = bool(
        selector_cfg.get(
            "enabled",
            refiner_cfg.mode in {"policy", "hybrid"},
        )
    )
    if train_selector:
        # Selector supervision must use the predictor selected on validation,
        # not whichever predictor happened to remain after the last epoch.
        best_predictor_checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(best_predictor_checkpoint["model_state_dict"])
        model.eval()
        selector = LearnedCandidateSelector(
            candidate_feature_dim=len(CANDIDATE_FEATURE_NAMES),
            graph_context_dim=len(GRAPH_CONTEXT_FEATURE_NAMES),
            hidden_dim=int(selector_cfg.get("hidden_dim", 128)),
            dropout=float(selector_cfg.get("dropout", 0.0)),
        ).to(device)
        selector_optimizer = torch.optim.AdamW(
            selector.parameters(),
            lr=float(selector_cfg.get("learning_rate", 3.0e-4)),
            weight_decay=float(selector_cfg.get("weight_decay", 1.0e-5)),
        )
        selector_epochs = int(selector_cfg.get("epochs", 25))
        selector_objective = str(selector_cfg.get("objective", "cross_entropy"))
        selector_history: list[dict[str, Any]] = []
        selector_best_val = float("inf")
        selector_best_epoch = 0
        selector_path = (
            output_dir / "selector.pt"
            if args.output_dir
            else Path(
                selector_cfg.get(
                    "checkpoint_path",
                    output_dir / "selector.pt",
                )
            )
        )
        ensure_dir(selector_path.parent)
        max_selector_train = selector_cfg.get("max_train_examples_per_epoch")
        max_selector_val = selector_cfg.get("max_val_examples_per_epoch")
        for selector_epoch in range(1, selector_epochs + 1):
            if isinstance(train_examples, EndpointTrajectoryIterableDataset):
                train_examples.set_epoch(epochs + selector_epoch - 1)
            if isinstance(val_examples, EndpointTrajectoryIterableDataset):
                val_examples.set_epoch(0)
            train_selector_metrics = _run_selector_epoch(
                selector,
                train_examples,
                predictor=model,
                vocabulary=vocabulary,
                graphlet_basis=graphlet_basis,
                summary_config=summary_cfg,
                refiner_config=refiner_cfg,
                device=device,
                optimizer=selector_optimizer,
                objective=selector_objective,
                max_examples=(
                    int(max_selector_train) if max_selector_train is not None else None
                ),
                seed=seed + 100_003 * selector_epoch,
            )
            val_selector_metrics = _run_selector_epoch(
                selector,
                val_examples,
                predictor=model,
                vocabulary=vocabulary,
                graphlet_basis=graphlet_basis,
                summary_config=summary_cfg,
                refiner_config=refiner_cfg,
                device=device,
                optimizer=None,
                objective=selector_objective,
                max_examples=(
                    int(max_selector_val) if max_selector_val is not None else None
                ),
                seed=seed + 1,
            )
            selector_row = {
                "epoch": selector_epoch,
                **{
                    f"train_{key}": value
                    for key, value in train_selector_metrics.items()
                },
                **{f"val_{key}": value for key, value in val_selector_metrics.items()},
            }
            selector_history.append(selector_row)
            if val_selector_metrics["loss"] < selector_best_val:
                selector_best_val = float(val_selector_metrics["loss"])
                selector_best_epoch = selector_epoch
                save_selector_checkpoint(
                    selector,
                    selector_path,
                    config={
                        "candidate_selector": selector_cfg,
                        "hybrid_refiner": config.get("hybrid_refiner", {}) or {},
                    },
                    report=selector_row,
                )
            if (
                selector_epoch == 1
                or selector_epoch % progress_interval == 0
                or selector_epoch == selector_epochs
            ):
                print(
                    f"selector_epoch={selector_epoch:04d} "
                    f"train={train_selector_metrics['loss']:.5f} "
                    f"val={val_selector_metrics['loss']:.5f} "
                    f"accuracy={val_selector_metrics['accuracy']:.4f}",
                    flush=True,
                )
        selector_report = {
            "checkpoint": str(selector_path),
            "best_epoch": selector_best_epoch,
            "best_val_loss": selector_best_val,
            "history": selector_history,
        }

    report = {
        "format": "hybrid_endpoint_graphlet_training_v1",
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "num_train_graphs": len(train_graphs),
        "num_val_graphs": len(val_graphs),
        "num_train_examples": num_train_examples,
        "num_val_examples": num_val_examples,
        "train_teacher": train_teacher_report,
        "val_teacher": val_teacher_report,
        "vocabulary": vocabulary.to_dict(),
        "graphlet_basis": graphlet_basis.to_dict(),
        "selector": selector_report,
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

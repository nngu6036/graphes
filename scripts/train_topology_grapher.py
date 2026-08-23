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
from grapher.properties.summary import SummaryConfig
from grapher.rewiring_mlp.generic.basis import TopologyGraphletBasis
from grapher.rewiring_mlp.generic.data import (
    TopologyTrajectoryIterableDataset,
    build_topology_examples,
    collate_topology_examples,
)
from grapher.rewiring_mlp.generic.graphlets import TOPOLOGY_ORBIT_WIDTH
from grapher.rewiring_mlp.generic.model import (
    TOPOLOGY_CHECKPOINT_FORMAT,
    TopologyGraphletPredictor,
    save_topology_checkpoint,
)
from grapher.rewiring_mlp.generic.spectral_data import (
    TopologySpectralTrajectoryIterableDataset,
    build_spectral_examples,
    collate_spectral_examples,
)
from grapher.rewiring_mlp.generic.spectral_model import (
    TOPOLOGY_SPECTRAL_CHECKPOINT_FORMAT,
    TOPOLOGY_SPECTRAL_GRAPHLET_CHECKPOINT_FORMAT,
    TopologySpectralTransformerPredictor,
    TopologySpectralGraphletTransformerPredictor,
    save_topology_spectral_checkpoint,
    save_topology_spectral_graphlet_checkpoint,
)
from grapher.rewiring_mlp.generic.training_sources import (
    build_completed_base_training_pairs,
)
from grapher.utils.device import resolve_torch_device
from grapher.utils.io import ensure_dir, load_yaml, save_json


_SPECTRAL_TYPES = {"spectral", "spectral_transformer", "spectrum_transformer"}
_SPECTRAL_GRAPHLET_TYPES = {
    "spectral_graphlet",
    "spectral_graphlet_transformer",
    "spectral_graphlet_diffusion",
    "dual_diffusion",
}


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted(set.intersection(*(set(row) for row in rows)))
    return {
        key: float(np.mean([float(row[key]) for row in rows]))
        for key in keys
    }


def _limited(values: list[Any], limit: int | None) -> list[Any]:
    if limit is None or int(limit) <= 0:
        return values
    return values[: int(limit)]


def _streaming_teacher_report(dataset: Any) -> dict[str, Any]:
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
        "mean_matching_cost",
        "mean_initial_structural_discrepancy",
        "mean_final_teacher_structural_discrepancy",
        "mean_initial_graphlet_discrepancy",
        "mean_final_teacher_graphlet_discrepancy",
        "mean_initial_spectral_discrepancy",
        "mean_final_teacher_spectral_discrepancy",
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


def _run_structural_epoch(
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


def _run_spectral_epoch(
    model: TopologySpectralTransformerPredictor,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    loss_weights: dict[str, float],
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    rows: list[dict[str, float]] = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            batch = batch.to(device)
            loss, metrics = model.loss(batch, loss_weights=loss_weights)
            if training:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            rows.append(metrics)
    if not rows:
        raise RuntimeError("The spectral topology trajectory produced no examples.")
    return _mean_metrics(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train GraphER topology guidance. Supports structural-summary, "
            "variable-length Spectral Transformer, and joint spectral + "
            "graphlet-logit diffusion predictors."
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
    pipeline_stage = str((config.get("pipeline", {}) or {}).get("stage", "topology")).lower()
    if pipeline_stage != "topology":
        raise ValueError("train_topology_grapher.py requires pipeline.stage: topology.")
    if config.get("categorical_state") or config.get("endpoint_predictor"):
        raise ValueError(
            "Topology training cannot define categorical_state or endpoint_predictor. "
            "Use the legacy attributed script until the attribute stage is migrated."
        )

    predictor_cfg = dict(config.get("topology_predictor", {}) or {})
    predictor_type = str(predictor_cfg.get("type", "structural_summary")).lower()
    spectral_graphlet_mode = predictor_type in _SPECTRAL_GRAPHLET_TYPES
    spectral_mode = predictor_type in _SPECTRAL_TYPES
    spectral_family_mode = spectral_mode or spectral_graphlet_mode
    if not spectral_family_mode and predictor_type not in {
        "structural_summary",
        "graphlet",
        "graphlet_predictor",
        "topology_graphlet",
    }:
        raise ValueError(
            "Unknown topology_predictor.type. Use structural_summary, "
            "spectral_transformer, or spectral_graphlet_transformer."
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

    # Keep SummaryConfig in spectral checkpoints so existing evaluation code can
    # use the same graphlet/orbit settings. In spectral+graphlet mode the fixed
    # connected graphlet basis is also used by the graphlet-logit diffusion head.
    summary_data = dict(config.get("graphlet_prediction", {}) or {})
    if bool(summary_data.get("attributed", False)):
        raise ValueError("The generic topology stage cannot use attributed graphlets.")
    summary_cfg = SummaryConfig.from_dict(summary_data, train_graphs)

    graphlet_basis: TopologyGraphletBasis | None = None
    if not spectral_mode:
        if str(summary_data.get("estimator", "exact_connected_local_delta")).lower() != "exact_connected_local_delta":
            raise ValueError(
                "Topology structural and spectral+graphlet training require estimator: "
                "exact_connected_local_delta."
            )
        summary_data["graphlet_history"] = True
        summary_cfg = SummaryConfig.from_dict(summary_data, train_graphs)
        if summary_cfg.graphlet_k_min < 3:
            raise ValueError("Topology graphlet guidance requires graphlet_k_min >= 3.")
        graphlet_basis = TopologyGraphletBasis.fit_from_graphs(
            train_graphs,
            summary_data,
            attributed=False,
            seed=seed,
        )
        if summary_cfg.orbit_count and not {"3", "4"}.issubset(set(graphlet_basis.sizes)):
            raise ValueError(
                "Orbit supervision/evaluation requires graphlet sizes 3 and 4."
            )

    source_cfg_raw = config.get("training_sources")
    if not isinstance(source_cfg_raw, dict):
        raise ValueError("topology training requires a training_sources mapping.")
    source_mode = str(source_cfg_raw.get("mode", "completed_base_outputs")).lower()
    if source_mode == "completed_base_outputs":
        train_items, val_items, source_report = build_completed_base_training_pairs(
            train_graphs,
            val_graphs,
            config=source_cfg_raw,
            seed=seed,
        )
    elif source_mode in {"target_degree_havel_hakimi", "spectral_havel_hakimi"}:
        if not spectral_family_mode:
            raise ValueError(
                "training_sources.mode: target_degree_havel_hakimi is reserved for "
                "spectral or spectral+graphlet training because the clean target "
                "must lie in the same degree fibre as the HH source."
            )
        train_items = train_graphs
        val_items = val_graphs
        source_report = {
            "format": "target_degree_havel_hakimi_source_v1",
            "mode": "target_degree_havel_hakimi",
            "same_degree_sequence_by_construction": True,
            "description": (
                "Each target graph is paired with a connected Havel-Hakimi "
                "realization of its own degree sequence."
            ),
        }
    elif source_mode == "legacy_havel_hakimi":
        train_items = train_graphs
        val_items = val_graphs
        source_report = {
            "format": "legacy_havel_hakimi_source_v1",
            "mode": source_mode,
            "warning": (
                "Compatibility mode. For new spectral training prefer "
                "target_degree_havel_hakimi."
            ),
        }
    else:
        raise ValueError(
            "training_sources.mode must be completed_base_outputs, "
            "target_degree_havel_hakimi (spectral), or legacy_havel_hakimi."
        )

    trajectory_cfg = dict(config.get("topology_trajectory", {}) or {})
    generation_only_horizon_keys = {
        "prediction_horizon",
        "refresh_prediction_every",
        "refresh_on_plateau",
    }
    misplaced_horizon_keys = generation_only_horizon_keys & set(trajectory_cfg)
    if misplaced_horizon_keys:
        raise ValueError(
            "Adaptive/fixed prediction horizons are generation-only. Remove "
            f"these keys from topology_trajectory: {sorted(misplaced_horizon_keys)}."
        )
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
    if source_mode == "completed_base_outputs" and int(
        trajectory_cfg.get("source_randomization_steps", 0)
    ) != 0:
        raise ValueError(
            "topology_trajectory.source_randomization_steps must be 0 when "
            "training from completed base-generator outputs."
        )

    guidance_name = (
        "spectral_graphlet_transformer" if spectral_graphlet_mode
        else "spectral_transformer" if spectral_mode
        else "structural_summary"
    )
    print(
        f"Preparing {storage_mode} topology trajectories "
        f"(guidance={guidance_name}, train_pairs={len(train_items)}, "
        f"val_pairs={len(val_items)}, source_mode={source_mode})...",
        flush=True,
    )

    spectral_cfg = dict(config.get("spectral_prediction", {}) or {})
    graphlet_diffusion_cfg = dict(config.get("graphlet_diffusion", {}) or {})
    graphlet_logit_epsilon = float(graphlet_diffusion_cfg.get("logit_epsilon", 1.0e-5))
    if spectral_family_mode:
        if spectral_graphlet_mode and graphlet_basis is None:
            raise AssertionError("Spectral+graphlet mode requires a graphlet basis.")
        if storage_mode == "streaming":
            train_examples = TopologySpectralTrajectoryIterableDataset(
                train_items,
                trajectory_config=trajectory_cfg,
                spectral_config=spectral_cfg,
                graphlet_basis=(graphlet_basis if spectral_graphlet_mode else None),
                graphlet_logit_epsilon=graphlet_logit_epsilon,
                seed=seed,
                shuffle_graphs=True,
            )
            val_examples = TopologySpectralTrajectoryIterableDataset(
                val_items,
                trajectory_config=trajectory_cfg,
                spectral_config=spectral_cfg,
                graphlet_basis=(graphlet_basis if spectral_graphlet_mode else None),
                graphlet_logit_epsilon=graphlet_logit_epsilon,
                seed=seed + 1,
                shuffle_graphs=False,
            )
            train_teacher_report: dict[str, Any] = {"storage": "streaming"}
            val_teacher_report: dict[str, Any] = {"storage": "streaming"}
            num_train_examples = train_examples.estimated_examples
            num_val_examples = val_examples.estimated_examples
        else:
            train_examples, train_teacher_report = build_spectral_examples(
                train_items,
                trajectory_config=trajectory_cfg,
                spectral_config=spectral_cfg,
                graphlet_basis=(graphlet_basis if spectral_graphlet_mode else None),
                graphlet_logit_epsilon=graphlet_logit_epsilon,
                seed=seed,
            )
            val_examples, val_teacher_report = build_spectral_examples(
                val_items,
                trajectory_config=trajectory_cfg,
                spectral_config=spectral_cfg,
                graphlet_basis=(graphlet_basis if spectral_graphlet_mode else None),
                graphlet_logit_epsilon=graphlet_logit_epsilon,
                seed=seed + 1,
            )
            num_train_examples = len(train_examples)
            num_val_examples = len(val_examples)
    else:
        assert graphlet_basis is not None
        if storage_mode == "streaming":
            train_examples = TopologyTrajectoryIterableDataset(
                train_items,
                summary_config=summary_cfg,
                graphlet_basis=graphlet_basis,
                trajectory_config=trajectory_cfg,
                seed=seed,
                shuffle_graphs=True,
            )
            val_examples = TopologyTrajectoryIterableDataset(
                val_items,
                summary_config=summary_cfg,
                graphlet_basis=graphlet_basis,
                trajectory_config=trajectory_cfg,
                seed=seed + 1,
                shuffle_graphs=False,
            )
            train_teacher_report = {"storage": "streaming"}
            val_teacher_report = {"storage": "streaming"}
            num_train_examples = train_examples.estimated_examples
            num_val_examples = val_examples.estimated_examples
        else:
            train_examples, train_teacher_report = build_topology_examples(
                train_items,
                summary_config=summary_cfg,
                graphlet_basis=graphlet_basis,
                trajectory_config=trajectory_cfg,
                seed=seed,
            )
            val_examples, val_teacher_report = build_topology_examples(
                val_items,
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

    device = resolve_torch_device(args.device or predictor_cfg.get("device", "auto"))
    if spectral_family_mode:
        common_spectral_kwargs = {
            "hidden_dim": int(predictor_cfg.get("hidden_dim", 128)),
            "edge_dim": int(predictor_cfg.get("edge_dim", 64)),
            "graph_dim": int(predictor_cfg.get("graph_dim", 128)),
            "num_layers": int(predictor_cfg.get("num_layers", 4)),
            "spectral_dim": int(predictor_cfg.get("spectral_dim", 128)),
            "spectral_layers": int(predictor_cfg.get("spectral_layers", 3)),
            "spectral_heads": int(predictor_cfg.get("spectral_heads", 4)),
            "spectral_ff_dim": int(predictor_cfg.get("spectral_ff_dim", 256)),
            "dropout": float(predictor_cfg.get("dropout", 0.0)),
            "min_gap": float(predictor_cfg.get("min_gap", 1.0e-6)),
            "input_normalization": str(
                predictor_cfg.get(
                    "input_normalization",
                    spectral_cfg.get("normalization", "mean_degree"),
                )
            ),
        }
        if spectral_graphlet_mode:
            assert graphlet_basis is not None
            model = TopologySpectralGraphletTransformerPredictor(
                graphlet_block_widths=graphlet_basis.simplex_block_widths,
                graphlet_dim=int(predictor_cfg.get("graphlet_dim", 256)),
                graphlet_dropout=float(
                    predictor_cfg.get("graphlet_dropout", predictor_cfg.get("dropout", 0.05))
                ),
                graphlet_logit_epsilon=graphlet_logit_epsilon,
                **common_spectral_kwargs,
            ).to(device)
        else:
            model = TopologySpectralTransformerPredictor(**common_spectral_kwargs).to(device)
        collate_fn = collate_spectral_examples
    else:
        assert graphlet_basis is not None
        model = TopologyGraphletPredictor(
            graphlet_slices=graphlet_basis.slices,
            clustering_width=(
                int(summary_cfg.clustering_bins) if summary_cfg.clustering_summary else 0
            ),
            orbit_width=(TOPOLOGY_ORBIT_WIDTH if summary_cfg.orbit_count else 0),
            hidden_dim=int(predictor_cfg.get("hidden_dim", 128)),
            edge_dim=int(predictor_cfg.get("edge_dim", 64)),
            graph_dim=int(predictor_cfg.get("graph_dim", 128)),
            num_layers=int(predictor_cfg.get("num_layers", 4)),
            dropout=float(predictor_cfg.get("dropout", 0.0)),
            min_concentration=float(predictor_cfg.get("min_concentration", 0.05)),
            max_concentration=float(predictor_cfg.get("max_concentration", 50.0)),
        ).to(device)
        collate_fn = collate_topology_examples

    batch_size = int(
        args.batch_size if args.batch_size is not None else predictor_cfg.get("batch_size", 4)
    )
    train_loader = DataLoader(
        train_examples,
        batch_size=batch_size,
        shuffle=storage_mode == "eager",
        num_workers=0,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_examples,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
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

    if spectral_family_mode:
        forbidden_loss_keys = {
            "node",
            "edge",
            "consistency",
            "graphlet_mean",
            "graphlet_distribution",
            "clustering_mean",
            "orbit",
        } & set(loss_weights)
        if forbidden_loss_keys:
            raise ValueError(
                "Spectral-family loss cannot contain legacy structural-summary terms: "
                f"{sorted(forbidden_loss_keys)}"
            )
        active_loss_defaults = [
            ("spectrum", 1.0),
            ("moment2", 0.1),
            ("low_frequency", 0.0),
        ]
        if spectral_graphlet_mode:
            active_loss_defaults.extend(
                [("graphlet_logit", 1.0), ("graphlet_probability", 0.25)]
            )
        if not any(
            float(loss_weights.get(key, default)) != 0.0
            for key, default in active_loss_defaults
        ):
            raise ValueError("At least one spectral/graphlet prediction loss must be active.")
        target_epsilon = None
    else:
        forbidden_loss_keys = {"node", "edge", "consistency"} & set(loss_weights)
        if forbidden_loss_keys:
            raise ValueError(
                "Topology loss cannot contain endpoint terms: "
                f"{sorted(forbidden_loss_keys)}"
            )
        active_loss_defaults = [
            ("graphlet_mean", 1.0),
            ("graphlet_distribution", 0.1),
            ("graphlet_mass", 0.0),
        ]
        if summary_cfg.clustering_summary:
            active_loss_defaults.extend(
                [("clustering_mean", 1.0), ("clustering_distribution", 0.1)]
            )
        if summary_cfg.orbit_count:
            active_loss_defaults.append(("orbit", 1.0))
        if not any(
            float(loss_weights.get(key, default)) != 0.0
            for key, default in active_loss_defaults
        ):
            raise ValueError("At least one topology structural loss must be active.")
        target_epsilon = float(predictor_cfg.get("target_epsilon", 1.0e-5))

    configured_checkpoint = predictor_cfg.get(
        "checkpoint_path",
        (
            "outputs/topology_grapher/sbm_spectral_graphlet/seed_42/checkpoint.pt"
            if spectral_graphlet_mode
            else "outputs/topology_grapher/sbm_spectral/seed_42/checkpoint.pt"
            if spectral_mode
            else "outputs/topology_grapher/sbm/seed_42/checkpoint.pt"
        ),
    )
    if args.output_dir:
        output_dir = ensure_dir(args.output_dir)
        checkpoint_path = output_dir / "checkpoint.pt"
    else:
        checkpoint_path = Path(configured_checkpoint)
        output_dir = ensure_dir(checkpoint_path.parent)
    epochs = int(args.epochs if args.epochs is not None else predictor_cfg.get("epochs", 100))
    progress_interval = max(int(predictor_cfg.get("progress_interval", 5)), 1)

    print(
        f"Training predictor={predictor_type} device={device} batch_size={batch_size} "
        f"epochs={epochs} checkpoint={checkpoint_path}",
        flush=True,
    )
    if spectral_family_mode:
        print(
            "Spectral Transformer: predicts all clean Laplacian eigenvalues jointly; "
            "variable graph sizes are handled by padded spectral tokens + mask; "
            "eigenvectors are not predicted.",
            flush=True,
        )
        if spectral_graphlet_mode:
            print(
                "Graphlet-logit diffusion: each k-block is connected graphlet "
                "probabilities + a disconnected bin, transformed to CLR logits; "
                "the shared graph encoder predicts the clean graphlet logits.",
                flush=True,
            )

    history: list[dict[str, Any]] = []
    best_val = float("inf")
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        if isinstance(
            train_examples,
            (TopologyTrajectoryIterableDataset, TopologySpectralTrajectoryIterableDataset),
        ):
            train_examples.set_epoch(epoch - 1)
        if isinstance(
            val_examples,
            (TopologyTrajectoryIterableDataset, TopologySpectralTrajectoryIterableDataset),
        ):
            val_examples.set_epoch(0)

        if spectral_family_mode:
            train_metrics = _run_spectral_epoch(
                model,
                train_loader,
                device=device,
                optimizer=optimizer,
                loss_weights=loss_weights,
            )
            val_metrics = _run_spectral_epoch(
                model,
                val_loader,
                device=device,
                optimizer=None,
                loss_weights=loss_weights,
            )
        else:
            assert target_epsilon is not None
            train_metrics = _run_structural_epoch(
                model,
                train_loader,
                device=device,
                optimizer=optimizer,
                loss_weights=loss_weights,
                target_epsilon=target_epsilon,
            )
            val_metrics = _run_structural_epoch(
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
            if spectral_graphlet_mode:
                assert graphlet_basis is not None
                save_topology_spectral_graphlet_checkpoint(
                    model,
                    checkpoint_path,
                    graphlet_basis=graphlet_basis,
                    summary_config=summary_cfg,
                    config=config,
                    report=row,
                    training_time_horizon=max(int(trajectory_cfg.get("steps", 32)), 1),
                )
            elif spectral_mode:
                save_topology_spectral_checkpoint(
                    model,
                    checkpoint_path,
                    summary_config=summary_cfg,
                    config=config,
                    report=row,
                    # Record the denominator used for the `time` feature so
                    # generation cannot silently rescale it via
                    # `--set topology_refiner.steps=...`.
                    training_time_horizon=max(
                        int(trajectory_cfg.get("steps", 32)), 1
                    ),
                )
            else:
                assert graphlet_basis is not None
                save_topology_checkpoint(
                    model,
                    checkpoint_path,
                    graphlet_basis=graphlet_basis,
                    summary_config=summary_cfg,
                    config=config,
                    report=row,
                )

        if epoch == 1 or epoch % progress_interval == 0 or epoch == epochs:
            if spectral_family_mode:
                extra = (
                    f" graphlet_logit_rmse={val_metrics['graphlet_logit_rmse']:.5f}"
                    f" graphlet_prob_mae={val_metrics['graphlet_probability_mae']:.5f}"
                    if spectral_graphlet_mode else ""
                )
                print(
                    f"epoch={epoch:04d} "
                    f"train={train_metrics['loss']:.5f} "
                    f"val={val_metrics['loss']:.5f} "
                    f"spectral_nrmse={val_metrics['spectral_normalized_rmse']:.5f} "
                    f"spectral_nmae={val_metrics['spectral_normalized_mae']:.5f} "
                    f"spectral_mae={val_metrics['spectral_mae']:.5f} "
                    f"moment2_rel={val_metrics['spectral_moment2_relative_error']:.5f} "
                    f"trace_mae={val_metrics['spectral_trace_mae']:.3e}"
                    f"{extra}",
                    flush=True,
                )
            else:
                print(
                    f"epoch={epoch:04d} "
                    f"train={train_metrics['loss']:.5f} "
                    f"val={val_metrics['loss']:.5f} "
                    f"graphlet_mae={val_metrics['graphlet_mae']:.5f} "
                    f"mass_mae={val_metrics['graphlet_mass_mae']:.5f} "
                    f"clustering_mae={val_metrics.get('clustering_mae', 0.0):.5f} "
                    f"orbit_log_mae={val_metrics.get('orbit_log_mae', 0.0):.5f}",
                    flush=True,
                )

    streaming_types = (
        TopologyTrajectoryIterableDataset,
        TopologySpectralTrajectoryIterableDataset,
    )
    if spectral_family_mode:
        predictor_targets: dict[str, Any] = {
            "clean_laplacian_eigenvalues": True,
            "prediction": "joint_one_shot",
            "variable_length": "spectral_tokens_with_padding_mask",
            "eigenvectors_predicted": False,
            "lambda1_fixed_zero": True,
            "sorted_by_positive_gaps": True,
            "trace_sum_lambda_equals_2m": True,
            "spectral_prediction": spectral_cfg,
            "clean_graphlet_clr_logits": bool(spectral_graphlet_mode),
            "graphlet_simplex_includes_disconnected_bin": bool(spectral_graphlet_mode),
            "graphlet_logit_epsilon": (graphlet_logit_epsilon if spectral_graphlet_mode else None),
        }
        graphlet_basis_report = (
            graphlet_basis.to_dict() if spectral_graphlet_mode and graphlet_basis is not None else None
        )
        if spectral_graphlet_mode:
            report_format = "topology_spectral_graphlet_training_v1"
            checkpoint_format = TOPOLOGY_SPECTRAL_GRAPHLET_CHECKPOINT_FORMAT
        else:
            report_format = "topology_spectral_training_v1"
            checkpoint_format = TOPOLOGY_SPECTRAL_CHECKPOINT_FORMAT
    else:
        assert graphlet_basis is not None
        predictor_targets = {
            "graphlet_histogram": True,
            "graphlet_connected_mass": True,
            "clustering_histogram": bool(summary_cfg.clustering_summary),
            "clustering_bins": (
                int(summary_cfg.clustering_bins) if summary_cfg.clustering_summary else 0
            ),
            "orbit_count": bool(summary_cfg.orbit_count),
            "orbit_width": TOPOLOGY_ORBIT_WIDTH if summary_cfg.orbit_count else 0,
        }
        graphlet_basis_report = graphlet_basis.to_dict()
        report_format = "topology_structural_training_v2"
        checkpoint_format = TOPOLOGY_CHECKPOINT_FORMAT

    report = {
        "format": report_format,
        "pipeline_mode": "topology",
        "guidance_mode": (
            "spectral_graphlet" if spectral_graphlet_mode
            else "spectral" if spectral_mode
            else "structural_summary"
        ),
        "predictor_type": predictor_type,
        "checkpoint_format": checkpoint_format,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "num_train_targets": len(train_graphs),
        "num_val_targets": len(val_graphs),
        "num_train_pairs": len(train_items),
        "num_val_pairs": len(val_items),
        "num_train_examples": num_train_examples,
        "num_val_examples": num_val_examples,
        "train_teacher": (
            _streaming_teacher_report(train_examples)
            if isinstance(train_examples, streaming_types)
            else train_teacher_report
        ),
        "val_teacher": (
            _streaming_teacher_report(val_examples)
            if isinstance(val_examples, streaming_types)
            else val_teacher_report
        ),
        "graphlet_basis": graphlet_basis_report,
        "predictor_targets": predictor_targets,
        "training_sources": source_report,
        "prediction_horizon_training": {
            "enabled": False,
            "reason": (
                "Prediction horizon is generation-only; teacher trajectories "
                "advance one accepted swap per step."
            ),
        },
        "active_losses": sorted(
            key
            for key, default in active_loss_defaults
            if float(loss_weights.get(key, default)) != 0.0
        ),
        "history": history,
    }
    save_json(report, output_dir / "training_report.json")
    print(f"Saved best topology checkpoint: {checkpoint_path}", flush=True)
    print(f"Best epoch={best_epoch} val_loss={best_val:.6f}", flush=True)


if __name__ == "__main__":
    main()

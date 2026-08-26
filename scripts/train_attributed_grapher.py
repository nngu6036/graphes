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


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted(set.intersection(*(set(row) for row in rows)))
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def _run_epoch(
    model: AttributedSpectralGraphletTransformerPredictor,
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
        raise RuntimeError("Attributed summary-diffusion dataset produced no examples.")
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
                "For kekulized QM9 this is normally (1, 2, 3), not aromatic category 4."
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
    graphlet_basis = GraphletBasis.fit_from_graphs(
        basis_graphs,
        graphlet_cfg,
        vocabulary=vocabulary,
        attributed=True,
        seed=seed,
    )
    print(
        "Attributed graphlet basis: "
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
        f"Training attributed predictor device={device} batch_size={batch_size} epochs={epochs} checkpoint={checkpoint_path}",
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
        )
        val_metrics = _run_epoch(
            model,
            val_loader,
            device=device,
            optimizer=None,
            loss_weights=loss_weights,
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

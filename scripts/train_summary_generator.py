#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from grapher.data.io import load_dataset_splits
from grapher.generators.summary_vae import (
    SummaryVectorizer,
    build_conditional_summary_vae,
    build_summary_vae,
    save_summary_vae_checkpoint,
    summary_vae_loss,
)
from grapher.properties.summary import (
    SummaryConfig,
    extract_summary,
    summary_to_jsonable,
)
from grapher.utils.device import resolve_torch_device
from grapher.utils.io import (
    ensure_dir,
    load_yaml,
    require_config,
    require_config_section,
    save_json,
)


def _limit(items: list[Any], limit: int | None) -> list[Any]:
    if limit is None or int(limit) <= 0:
        return items
    return items[: int(limit)]


def _targets_to_tensors(
    targets: dict[str, np.ndarray], device: torch.device
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, value in targets.items():
        tensor = torch.as_tensor(value, device=device)
        if key == "num_nodes":
            tensor = tensor.long()
        else:
            tensor = tensor.float()
        out[key] = tensor
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the learned summary generator (SummaryVAE)."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Experiment YAML used for dataset and summary settings.",
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    generator_cfg = require_config_section(config, "summary_generator")
    seed = int(require_config(config, "seed"))
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = resolve_torch_device(
        require_config(generator_cfg, "device", context="config.summary_generator")
    )
    checkpoint_path = Path(
        require_config(
            generator_cfg, "checkpoint_path", context="config.summary_generator"
        )
    )
    out_dir = ensure_dir(checkpoint_path.parent)
    checkpoint_path = out_dir / checkpoint_path.name
    print(f"Using device: {device}", flush=True)

    dataset_cfg = require_config_section(config, "dataset")
    splits = load_dataset_splits(
        require_config(dataset_cfg, "name", context="config.dataset"),
        root=require_config(dataset_cfg, "root", context="config.dataset"),
        build_if_missing=bool(
            require_config(dataset_cfg, "build_if_missing", context="config.dataset")
        ),
        config_path=require_config(
            dataset_cfg, "config_path", context="config.dataset"
        ),
    )
    max_train = require_config(
        dataset_cfg, "max_train_graphs", context="config.dataset"
    )
    train_graphs = _limit(list(splits["train"]), max_train)
    val_graphs = _limit(list(splits.get("val", [])), max_train)
    summary_cfg = SummaryConfig.from_dict(
        require_config_section(config, "summary"), train_graphs
    )
    summaries = [extract_summary(g, summary_cfg) for g in train_graphs]
    constructor_cfg = require_config_section(config, "constructor")
    vectorizer = SummaryVectorizer.fit(
        summaries,
        summary_cfg,
        require_connected=bool(
            require_config(
                constructor_cfg, "ensure_connected", context="config.constructor"
            )
        ),
    )
    x_np, targets_np = vectorizer.to_training_arrays(summaries)
    conditional_on_degree = bool(generator_cfg.get("conditional_on_degree", False))
    condition_np = vectorizer.to_condition_array(summaries)

    x = torch.as_tensor(x_np, dtype=torch.float32)
    condition = torch.as_tensor(condition_np, dtype=torch.float32)
    dataset = TensorDataset(x, condition, torch.arange(x.shape[0]))
    batch_size = int(
        require_config(generator_cfg, "batch_size", context="config.summary_generator")
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    all_targets = _targets_to_tensors(targets_np, device)
    val_summaries = [extract_summary(g, summary_cfg) for g in val_graphs]
    if val_summaries:
        val_x_np, val_targets_np = vectorizer.to_training_arrays(val_summaries)
        val_condition_np = vectorizer.to_condition_array(val_summaries)
        val_targets = _targets_to_tensors(val_targets_np, device)
    else:
        val_x_np = np.zeros((0, vectorizer.input_dim), dtype=np.float32)
        val_condition_np = np.zeros(
            (0, vectorizer.condition_dim), dtype=np.float32
        )
        val_targets = {}

    model_builder = (
        build_conditional_summary_vae if conditional_on_degree else build_summary_vae
    )
    model = model_builder(
        vectorizer,
        latent_dim=int(
            require_config(
                generator_cfg, "latent_dim", context="config.summary_generator"
            )
        ),
        hidden_dim=int(
            require_config(
                generator_cfg, "hidden_dim", context="config.summary_generator"
            )
        ),
        num_layers=int(
            require_config(
                generator_cfg, "num_layers", context="config.summary_generator"
            )
        ),
        dropout=float(
            require_config(generator_cfg, "dropout", context="config.summary_generator")
        ),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(
            require_config(
                generator_cfg, "learning_rate", context="config.summary_generator"
            )
        ),
        weight_decay=float(
            require_config(
                generator_cfg, "weight_decay", context="config.summary_generator"
            )
        ),
    )

    history: list[dict[str, float]] = []
    epochs = int(
        require_config(generator_cfg, "epochs", context="config.summary_generator")
    )
    beta = float(
        require_config(generator_cfg, "beta", context="config.summary_generator")
    )
    kl_warmup_epochs = max(int(generator_cfg.get("kl_warmup_epochs", 0)), 0)
    loss_weights = dict(
        require_config(
            generator_cfg, "loss_weights", context="config.summary_generator"
        )
        or {}
    )
    if conditional_on_degree:
        # These fields define the condition and are copied exactly at sampling
        # time.  Predicting them again would spend model capacity on invariants
        # that the refiner can never change.
        loss_weights.setdefault("num_nodes", 0.0)
        loss_weights.setdefault("degree", 0.0)
        # Density is completely determined by (n, m), which is part of the
        # condition and is copied exactly during sampling.
        loss_weights.setdefault("density", 0.0)
    graphlet_slices = [
        (int(graphlet_slice.start), int(graphlet_slice.stop))
        for graphlet_slice in vectorizer.graphlet_slices().values()
    ]
    progress_interval = int(
        require_config(
            generator_cfg, "progress_interval", context="config.summary_generator"
        )
    )
    validation_metric = str(generator_cfg.get("validation_metric", "loss"))
    best_metric = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_metrics: dict[str, list[float]] = {}
        effective_beta = (
            beta
            if kl_warmup_epochs <= 0
            else beta * min(float(epoch) / float(kl_warmup_epochs), 1.0)
        )
        for batch_x, batch_condition, batch_idx in loader:
            batch_x = batch_x.to(device)
            batch_condition = batch_condition.to(device)
            batch_targets = {
                key: value[batch_idx.to(device)] for key, value in all_targets.items()
            }
            if conditional_on_degree:
                outputs, mu, logvar = model(batch_x, batch_condition)
            else:
                outputs, mu, logvar = model(batch_x)
            loss, metrics = summary_vae_loss(
                outputs,
                batch_targets,
                mu,
                logvar,
                beta=effective_beta,
                weights=loss_weights,
                graphlet_slices=graphlet_slices,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            for key, value in metrics.items():
                epoch_metrics.setdefault(key, []).append(float(value))
        mean_metrics = {
            key: float(np.mean(values)) for key, values in epoch_metrics.items()
        }
        if val_summaries:
            model.eval()
            val_metrics: dict[str, list[float]] = {}
            with torch.no_grad():
                for start in range(0, len(val_summaries), batch_size):
                    stop = min(start + batch_size, len(val_summaries))
                    batch_x = torch.as_tensor(
                        val_x_np[start:stop],
                        dtype=torch.float32,
                        device=device,
                    )
                    batch_condition = torch.as_tensor(
                        val_condition_np[start:stop],
                        dtype=torch.float32,
                        device=device,
                    )
                    batch_targets = {
                        key: value[start:stop]
                        for key, value in val_targets.items()
                    }
                    if conditional_on_degree:
                        mu, logvar = model.encode(batch_x, batch_condition)
                        outputs = model.decode(mu, batch_condition)
                    else:
                        mu, logvar = model.encode(batch_x)
                        outputs = model.decode(mu)
                    _, metrics = summary_vae_loss(
                        outputs,
                        batch_targets,
                        mu,
                        logvar,
                        beta=effective_beta,
                        weights=loss_weights,
                        graphlet_slices=graphlet_slices,
                    )
                    for key, value in metrics.items():
                        val_metrics.setdefault(key, []).append(float(value))
            for key, values in val_metrics.items():
                mean_metrics[f"val_{key}"] = float(np.mean(values))
            selected = float(
                mean_metrics.get(
                    f"val_{validation_metric}",
                    mean_metrics["val_loss"],
                )
            )
            if selected < best_metric:
                best_metric = selected
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
        mean_metrics["epoch"] = float(epoch)
        mean_metrics["beta"] = float(effective_beta)
        history.append(mean_metrics)
        if (
            epoch == 1
            or (progress_interval > 0 and epoch % progress_interval == 0)
            or epoch == epochs
        ):
            print(
                f"epoch={epoch:04d} loss={mean_metrics['loss']:.4f} "
                f"degree={mean_metrics['degree_loss']:.4f} clustering={mean_metrics['clustering_loss']:.4f} "
                f"spectral={mean_metrics['spectral_loss']:.4f} graphlet={mean_metrics.get('graphlet_loss', 0.0):.4f} "
                f"kl={mean_metrics['kl_loss']:.4f}"
                + (
                    f" val_loss={mean_metrics['val_loss']:.4f} "
                    f"val_graphlet={mean_metrics['val_graphlet_loss']:.4f}"
                    if val_summaries
                    else ""
                ),
                flush=True,
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    save_summary_vae_checkpoint(
        checkpoint_path,
        model,
        vectorizer,
        config={
            "experiment_config": config,
            "conditional_on_degree": conditional_on_degree,
        },
        metrics={
            "history": history,
            "final": history[-1] if history else {},
            "selected_checkpoint": {
                "metric": validation_metric,
                "epoch": best_epoch,
                "value": best_metric,
            },
        },
    )
    vectorizer.save(out_dir / "summary_vectorizer.json")
    save_json(
        {
            "history": history,
            "final": history[-1] if history else {},
            "selected_checkpoint": {
                "metric": validation_metric,
                "epoch": best_epoch,
                "value": best_metric,
            },
        },
        out_dir / "training_metrics.json",
    )
    save_json(
        [summary_to_jsonable(s) for s in summaries[:20]],
        out_dir / "training_summary_examples.json",
    )
    print(f"Saved checkpoint to: {checkpoint_path}")
    print(f"Saved vectorizer to: {out_dir / 'summary_vectorizer.json'}")


if __name__ == "__main__":
    main()

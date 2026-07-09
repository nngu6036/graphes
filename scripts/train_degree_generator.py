#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from grapher.data.io import load_dataset_splits
from grapher.generators.degree_vae import build_degree_vae, degree_vae_loss, save_degree_vae_checkpoint
from grapher.generators.degree_vectorizer import DegreeVectorizer
from grapher.utils.device import resolve_torch_device
from grapher.utils.io import ensure_dir, load_yaml, save_json


def _limit(items: list[Any], limit: int | None) -> list[Any]:
    if limit is None or int(limit) <= 0:
        return items
    return items[: int(limit)]


def _targets_to_tensors(targets: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
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
    parser = argparse.ArgumentParser(description="Train a degree-histogram VAE for the coarse-to-fine generator.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_yaml(args.config)
    degree_cfg = config.get("degree_generator", {}) or {}
    seed = int(degree_cfg.get("seed", config.get("seed", 0)))
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = resolve_torch_device(degree_cfg.get("device", "auto"))
    checkpoint_path = Path(degree_cfg.get("checkpoint_path", "outputs/degree_generators/degree/checkpoint.pt"))
    out_dir = ensure_dir(degree_cfg.get("output_dir", checkpoint_path.parent))
    checkpoint_path = out_dir / checkpoint_path.name
    print(f"Using device: {device}", flush=True)

    dataset_cfg = config.get("dataset", {}) or {}
    splits = load_dataset_splits(
        dataset_cfg.get("name", "sbm"),
        root=dataset_cfg.get("root", "outputs/datasets"),
        build_if_missing=bool(dataset_cfg.get("build_if_missing", True)),
        config_path=dataset_cfg.get("config_path"),
    )
    max_train = dataset_cfg.get("max_train_graphs")
    train_graphs = _limit(list(splits["train"]), max_train)
    require_connected = bool((config.get("constructor", {}) or {}).get("ensure_connected", True))
    max_degree_raw = (config.get("summary", {}) or {}).get("degree_hist_max_degree", "auto")
    max_degree = None if max_degree_raw in {None, "auto"} else int(max_degree_raw)
    vectorizer = DegreeVectorizer.fit(train_graphs, max_degree=max_degree, require_connected=require_connected)
    x_np, targets_np = vectorizer.to_training_arrays(train_graphs)

    x = torch.as_tensor(x_np, dtype=torch.float32)
    dataset = TensorDataset(x, torch.arange(x.shape[0]))
    batch_size = int(degree_cfg.get("batch_size", 32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    all_targets = _targets_to_tensors(targets_np, device)

    model = build_degree_vae(
        vectorizer,
        latent_dim=int(degree_cfg.get("latent_dim", 32)),
        hidden_dim=int(degree_cfg.get("hidden_dim", 128)),
        num_layers=int(degree_cfg.get("num_layers", 2)),
        dropout=float(degree_cfg.get("dropout", 0.0)),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(degree_cfg.get("learning_rate", degree_cfg.get("lr", 2e-3))),
        weight_decay=float(degree_cfg.get("weight_decay", 0.0)),
    )
    weights = {
        "num_nodes": float(degree_cfg.get("node_weight", 1.0)),
        "degree": float(degree_cfg.get("degree_weight", 5.0)),
        "edge_scalar": float(degree_cfg.get("edge_moment_weight", 0.1)),
    }

    history: list[dict[str, float]] = []
    epochs = int(degree_cfg.get("epochs", 300))
    beta = float(degree_cfg.get("beta", 5e-3))
    progress_interval = int(degree_cfg.get("progress_interval", max(1, epochs // 10)))
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_metrics: dict[str, list[float]] = {}
        for batch_x, batch_idx in loader:
            batch_x = batch_x.to(device)
            batch_targets = {key: value[batch_idx.to(device)] for key, value in all_targets.items()}
            outputs, mu, logvar = model(batch_x)
            loss, metrics = degree_vae_loss(outputs, batch_targets, mu, logvar, beta=beta, weights=weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            for key, value in metrics.items():
                epoch_metrics.setdefault(key, []).append(float(value))
        mean_metrics = {key: float(np.mean(values)) for key, values in epoch_metrics.items()}
        mean_metrics["epoch"] = float(epoch)
        history.append(mean_metrics)
        if epoch == 1 or (progress_interval > 0 and epoch % progress_interval == 0) or epoch == epochs:
            print(
                f"epoch={epoch:04d} loss={mean_metrics['loss']:.4f} "
                f"degree={mean_metrics['degree_loss']:.4f} nodes={mean_metrics['num_nodes_loss']:.4f} "
                f"edge={mean_metrics['edge_scalar_loss']:.4f} kl={mean_metrics['kl_loss']:.4f}",
                flush=True,
            )

    save_degree_vae_checkpoint(
        checkpoint_path,
        model,
        vectorizer,
        config={"experiment_config": config},
        metrics={"history": history, "final": history[-1] if history else {}},
    )
    vectorizer.save(out_dir / "degree_vectorizer.json")
    save_json({"history": history, "final": history[-1] if history else {}}, out_dir / "training_metrics.json")
    print(f"Saved checkpoint to: {checkpoint_path}")
    print(f"Saved vectorizer to: {out_dir / 'degree_vectorizer.json'}")


if __name__ == "__main__":
    main()

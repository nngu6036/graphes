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
    parser.add_argument("--output-dir", default="outputs/degree_generators/sbm_report")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--beta", type=float, default=5e-3)
    parser.add_argument("--degree-weight", type=float, default=5.0)
    parser.add_argument("--edge-moment-weight", type=float, default=0.1)
    parser.add_argument("--node-weight", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-train-graphs", type=int, default=None)
    args = parser.parse_args()

    config = load_yaml(args.config)
    seed = int(args.seed if args.seed is not None else config.get("seed", 0))
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = resolve_torch_device(args.device)
    out_dir = ensure_dir(args.output_dir)
    print(f"Using device: {device}", flush=True)

    dataset_cfg = config.get("dataset", {}) or {}
    splits = load_dataset_splits(
        dataset_cfg.get("name", "sbm_spectre"),
        root=dataset_cfg.get("root", "outputs/datasets"),
        build_if_missing=bool(dataset_cfg.get("build_if_missing", True)),
        config_path=dataset_cfg.get("config_path"),
    )
    max_train = args.max_train_graphs if args.max_train_graphs is not None else dataset_cfg.get("max_train_graphs")
    train_graphs = _limit(list(splits["train"]), max_train)
    require_connected = bool((config.get("constructor", {}) or {}).get("ensure_connected", True))
    max_degree_raw = (config.get("summary", {}) or {}).get("degree_hist_max_degree", "auto")
    max_degree = None if max_degree_raw in {None, "auto"} else int(max_degree_raw)
    vectorizer = DegreeVectorizer.fit(train_graphs, max_degree=max_degree, require_connected=require_connected)
    x_np, targets_np = vectorizer.to_training_arrays(train_graphs)

    x = torch.as_tensor(x_np, dtype=torch.float32)
    dataset = TensorDataset(x, torch.arange(x.shape[0]))
    loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=True, drop_last=False)
    all_targets = _targets_to_tensors(targets_np, device)

    model = build_degree_vae(
        vectorizer,
        latent_dim=int(args.latent_dim),
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.num_layers),
        dropout=float(args.dropout),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    weights = {
        "num_nodes": float(args.node_weight),
        "degree": float(args.degree_weight),
        "edge_scalar": float(args.edge_moment_weight),
    }

    history: list[dict[str, float]] = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        epoch_metrics: dict[str, list[float]] = {}
        for batch_x, batch_idx in loader:
            batch_x = batch_x.to(device)
            batch_targets = {key: value[batch_idx.to(device)] for key, value in all_targets.items()}
            outputs, mu, logvar = model(batch_x)
            loss, metrics = degree_vae_loss(outputs, batch_targets, mu, logvar, beta=float(args.beta), weights=weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            for key, value in metrics.items():
                epoch_metrics.setdefault(key, []).append(float(value))
        mean_metrics = {key: float(np.mean(values)) for key, values in epoch_metrics.items()}
        mean_metrics["epoch"] = float(epoch)
        history.append(mean_metrics)
        if epoch == 1 or epoch % max(1, int(args.epochs) // 10) == 0 or epoch == int(args.epochs):
            print(
                f"epoch={epoch:04d} loss={mean_metrics['loss']:.4f} "
                f"degree={mean_metrics['degree_loss']:.4f} nodes={mean_metrics['num_nodes_loss']:.4f} "
                f"edge={mean_metrics['edge_scalar_loss']:.4f} kl={mean_metrics['kl_loss']:.4f}",
                flush=True,
            )

    checkpoint_path = out_dir / "checkpoint.pt"
    save_degree_vae_checkpoint(
        checkpoint_path,
        model,
        vectorizer,
        config={"experiment_config": config, "args": vars(args)},
        metrics={"history": history, "final": history[-1] if history else {}},
    )
    vectorizer.save(out_dir / "degree_vectorizer.json")
    save_json({"history": history, "final": history[-1] if history else {}}, out_dir / "training_metrics.json")
    print(f"Saved checkpoint to: {checkpoint_path}")
    print(f"Saved vectorizer to: {out_dir / 'degree_vectorizer.json'}")


if __name__ == "__main__":
    main()

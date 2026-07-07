#!/usr/bin/env python
from __future__ import annotations

import argparse
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from grapher.data.io import load_dataset_splits
from grapher.molecular.dense_mixture_catflow import (
    DenseMolecularGraphDataset,
    DenseMolecularMixtureCatFlow,
    collate_dense_molecular_graphs,
    node_count_distribution,
    save_dense_mixture_catflow_checkpoint,
)
from grapher.utils.device import resolve_torch_device
from grapher.utils.io import ensure_dir, load_yaml, save_json


def evaluate(model, loader, device, noise_scale: float):
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            loss, stats = model.loss(batch, device=device, noise_scale=noise_scale)
            rows.append(stats)
    if not rows:
        return {"loss": float("nan"), "node_loss": float("nan"), "edge_loss": float("nan")}
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train joint dense QM9 mixture CatFlow baseline.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    flow_cfg = cfg.get("joint_mixture_catflow", {}) or cfg.get("mixture_catflow", {}) or {}
    data_cfg = flow_cfg.get("dataset", {}) or cfg.get("attribute_dataset", {}) or {}
    dataset_name = data_cfg.get("name", "qm9_attributed")
    root = data_cfg.get("root", cfg.get("dataset", {}).get("root", "outputs/datasets"))

    splits = load_dataset_splits(dataset_name, root=root, build_if_missing=False)
    train_graphs = list(splits["train"])
    val_graphs = list(splits.get("val", [])) or train_graphs[: max(1, len(train_graphs) // 10)]
    if data_cfg.get("max_train_graphs"):
        train_graphs = train_graphs[: int(data_cfg["max_train_graphs"])]
    if data_cfg.get("max_val_graphs"):
        val_graphs = val_graphs[: int(data_cfg["max_val_graphs"])]

    device = resolve_torch_device(args.device or flow_cfg.get("device", "auto"))
    batch_size = int(args.batch_size or flow_cfg.get("batch_size", 64))
    epochs = int(args.epochs or flow_cfg.get("epochs", 100))
    noise_scale = float(flow_cfg.get("noise_scale", 1.0))

    model = DenseMolecularMixtureCatFlow(
        hidden_dim=int(flow_cfg.get("hidden_dim", 128)),
        edge_dim=int(flow_cfg.get("edge_dim", 64)),
        num_layers=int(flow_cfg.get("num_layers", 4)),
        dropout=float(flow_cfg.get("dropout", 0.0)),
        num_mixtures=int(flow_cfg.get("num_mixtures", 4)),
        node_count_probs=node_count_distribution(train_graphs),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(flow_cfg.get("learning_rate", 3e-4)),
        weight_decay=float(flow_cfg.get("weight_decay", 1e-5)),
    )
    train_loader = DataLoader(
        DenseMolecularGraphDataset(train_graphs),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_dense_molecular_graphs,
    )
    val_loader = DataLoader(
        DenseMolecularGraphDataset(val_graphs),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_dense_molecular_graphs,
    )

    out_dir = ensure_dir(args.output_dir)
    best_val = float("inf")
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_rows = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss, stats = model.loss(batch, device=device, noise_scale=noise_scale)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_rows.append(stats)
        train_stats = {k: float(np.mean([r[k] for r in train_rows])) for k in train_rows[0]}
        val_stats = evaluate(model, val_loader, device, noise_scale=noise_scale)
        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"val_{k}": v for k, v in val_stats.items()},
        }
        history.append(row)
        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            save_dense_mixture_catflow_checkpoint(model, out_dir / "checkpoint.pt", config=cfg, report=row)
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"epoch={epoch:04d} train_loss={train_stats['loss']:.4f} "
                f"val_loss={val_stats['loss']:.4f} val_node={val_stats['node_loss']:.4f} val_edge={val_stats['edge_loss']:.4f}",
                flush=True,
            )

    report = {
        "best_val_loss": best_val,
        "history": history,
        "num_train": len(train_graphs),
        "num_val": len(val_graphs),
    }
    save_json(report, out_dir / "training_report.json")
    print(f"Saved checkpoint to: {out_dir / 'checkpoint.pt'}")
    print(f"Saved training report to: {out_dir / 'training_report.json'}")


if __name__ == "__main__":
    main()

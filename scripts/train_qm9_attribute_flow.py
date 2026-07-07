#!/usr/bin/env python
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from grapher.data.io import load_dataset_splits
from grapher.molecular.attribute_flow import (
    MolecularGraphDataset,
    TopologyConditionalAttributeFlow,
    collate_molecular_graphs,
    save_attribute_flow_checkpoint,
)
from grapher.utils.device import resolve_torch_device
from grapher.utils.io import ensure_dir, load_yaml, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Train topology-conditioned molecular attribute flow on QM9.")
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

    attr_cfg = cfg.get("attribute_flow", {}) or {}
    data_cfg = attr_cfg.get("dataset", {}) or cfg.get("attribute_dataset", {}) or {}
    dataset_name = data_cfg.get("name", "qm9_attributed")
    root = data_cfg.get("root", cfg.get("dataset", {}).get("root", "outputs/datasets"))
    splits = load_dataset_splits(dataset_name, root=root, build_if_missing=False)
    train_graphs = list(splits["train"])
    val_graphs = list(splits.get("val", [])) or train_graphs[: max(1, len(train_graphs) // 10)]

    max_train = data_cfg.get("max_train_graphs")
    max_val = data_cfg.get("max_val_graphs")
    if max_train:
        train_graphs = train_graphs[: int(max_train)]
    if max_val:
        val_graphs = val_graphs[: int(max_val)]

    device = resolve_torch_device(args.device or attr_cfg.get("device", "auto"))
    batch_size = int(args.batch_size or attr_cfg.get("batch_size", 64))
    epochs = int(args.epochs or attr_cfg.get("epochs", 100))

    model = TopologyConditionalAttributeFlow(
        hidden_dim=int(attr_cfg.get("hidden_dim", 128)),
        edge_dim=int(attr_cfg.get("edge_dim", 64)),
        num_layers=int(attr_cfg.get("num_layers", 4)),
        dropout=float(attr_cfg.get("dropout", 0.0)),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(attr_cfg.get("learning_rate", 3e-4)), weight_decay=float(attr_cfg.get("weight_decay", 1e-5)))

    train_loader = DataLoader(
        MolecularGraphDataset(train_graphs),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_molecular_graphs,
    )
    val_loader = DataLoader(
        MolecularGraphDataset(val_graphs),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_molecular_graphs,
    )

    out_dir = ensure_dir(args.output_dir)
    best_val = float("inf")
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        train_node = []
        train_edge = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss, stats = model.loss(batch, device=device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
            train_node.append(stats["node_loss"])
            train_edge.append(stats["edge_loss"])
        model.eval()
        val_losses = []
        val_node = []
        val_edge = []
        with torch.no_grad():
            for batch in val_loader:
                loss, stats = model.loss(batch, device=device)
                val_losses.append(float(loss.detach().cpu()))
                val_node.append(stats["node_loss"])
                val_edge.append(stats["edge_loss"])
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "train_node_loss": float(np.mean(train_node)),
            "train_edge_loss": float(np.mean(train_edge)),
            "val_loss": float(np.mean(val_losses)),
            "val_node_loss": float(np.mean(val_node)),
            "val_edge_loss": float(np.mean(val_edge)),
        }
        history.append(row)
        if row["val_loss"] < best_val:
            best_val = row["val_loss"]
            save_attribute_flow_checkpoint(model, out_dir / "checkpoint.pt", config=attr_cfg)
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"epoch={epoch:04d} train={row['train_loss']:.4f} val={row['val_loss']:.4f} "
                f"node={row['val_node_loss']:.4f} edge={row['val_edge_loss']:.4f}"
            )
    save_json({"best_val_loss": best_val, "history": history, "num_train": len(train_graphs), "num_val": len(val_graphs)}, out_dir / "training_report.json")
    print(f"Saved attribute-flow checkpoint to: {out_dir / 'checkpoint.pt'}")


if __name__ == "__main__":
    main()

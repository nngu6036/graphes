#!/usr/bin/env python
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from grapher.data.io import load_dataset_splits
from grapher.molecular.attribute_flow import MolecularGraphDataset, collate_molecular_graphs
from grapher.molecular.mixture_catflow import TopologyConditionalMixtureCatFlow, save_mixture_catflow_checkpoint
from grapher.utils.device import resolve_torch_device
from grapher.utils.io import ensure_dir, load_yaml, save_json


def _log(message: str) -> None:
    print(f"[progress] {message}", flush=True)


def evaluate(model, loader, device, *, log_interval: int = 0):
    model.eval()
    rows = []
    started_at = time.perf_counter()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            loss, stats = model.loss(batch, device=device)
            rows.append(stats)
            if log_interval and (batch_idx == 1 or batch_idx % log_interval == 0 or batch_idx == len(loader)):
                elapsed = time.perf_counter() - started_at
                _log(f"validation batch {batch_idx}/{len(loader)} loss={float(loss.detach().cpu().item()):.4f} elapsed={elapsed:.1f}s")
    if not rows:
        return {"loss": float("nan"), "node_loss": float("nan"), "edge_loss": float("nan")}
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train topology-conditioned mixture CatFlow for QM9 attributes.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    started_at = time.perf_counter()
    cfg = load_yaml(args.config)
    flow_cfg = cfg.get("mixture_catflow", {}) or cfg.get("attribute_flow", {}) or {}
    seed = int(flow_cfg.get("seed", cfg.get("seed", 0)))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    checkpoint_path = Path(flow_cfg.get("checkpoint_path", "outputs/attribute_flows/qm9_mixture_catflow/checkpoint.pt"))
    out_dir = ensure_dir(flow_cfg.get("output_dir", checkpoint_path.parent))
    checkpoint_path = out_dir / checkpoint_path.name
    _log(f"starting QM9 mixture CatFlow training seed={seed} output_dir={out_dir}")

    data_cfg = flow_cfg.get("dataset", {}) or cfg.get("attribute_dataset", {}) or {}
    dataset_name = data_cfg.get("name", "qm9_attributed")
    root = data_cfg.get("root", cfg.get("dataset", {}).get("root", "outputs/datasets"))
    _log(f"loading dataset name={dataset_name} root={root}")
    splits = load_dataset_splits(dataset_name, root=root, build_if_missing=False)
    train_graphs = list(splits["train"])
    val_graphs = list(splits.get("val", [])) or train_graphs[: max(1, len(train_graphs) // 10)]
    max_train_graphs = data_cfg.get("max_train_graphs")
    max_val_graphs = data_cfg.get("max_val_graphs")
    if max_train_graphs:
        train_graphs = train_graphs[: int(max_train_graphs)]
    if max_val_graphs:
        val_graphs = val_graphs[: int(max_val_graphs)]
    _log(f"dataset ready train={len(train_graphs)} val={len(val_graphs)} splits={list(splits.keys())}")

    device = resolve_torch_device(flow_cfg.get("device", "auto"))
    batch_size = int(flow_cfg.get("batch_size", 64))
    epochs = int(flow_cfg.get("epochs", 100))
    progress_interval = int(flow_cfg.get("progress_interval", 10))
    batch_log_interval = int(flow_cfg.get("batch_log_interval", 0))
    progress_interval = max(progress_interval, 0)
    batch_log_interval = max(batch_log_interval, 0)
    _log(f"using device={device} epochs={epochs} batch_size={batch_size}")

    model = TopologyConditionalMixtureCatFlow(
        hidden_dim=int(flow_cfg.get("hidden_dim", 128)),
        edge_dim=int(flow_cfg.get("edge_dim", 64)),
        num_layers=int(flow_cfg.get("num_layers", 4)),
        dropout=float(flow_cfg.get("dropout", 0.0)),
        num_mixtures=int(flow_cfg.get("num_mixtures", 4)),
    ).to(device)
    _log(
        f"model hidden_dim={int(flow_cfg.get('hidden_dim', 128))} "
        f"edge_dim={int(flow_cfg.get('edge_dim', 64))} "
        f"layers={int(flow_cfg.get('num_layers', 4))} "
        f"mixtures={int(flow_cfg.get('num_mixtures', 4))}"
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(flow_cfg.get("learning_rate", 3e-4)),
        weight_decay=float(flow_cfg.get("weight_decay", 1e-5)),
    )
    _log(f"optimizer AdamW lr={float(flow_cfg.get('learning_rate', 3e-4))} weight_decay={float(flow_cfg.get('weight_decay', 1e-5))}")
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
    _log(f"loaders ready train_batches={len(train_loader)} val_batches={len(val_loader)}")

    best_val = float("inf")
    history = []
    for epoch in range(1, epochs + 1):
        epoch_started_at = time.perf_counter()
        _log(f"epoch {epoch}/{epochs} starting")
        model.train()
        train_rows = []
        for batch_idx, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            loss, stats = model.loss(batch, device=device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_rows.append(stats)
            if batch_log_interval and (batch_idx == 1 or batch_idx % batch_log_interval == 0 or batch_idx == len(train_loader)):
                elapsed = time.perf_counter() - epoch_started_at
                _log(
                    f"epoch {epoch}/{epochs} train batch {batch_idx}/{len(train_loader)} "
                    f"loss={float(loss.detach().cpu().item()):.4f} elapsed={elapsed:.1f}s"
                )
        train_stats = {k: float(np.mean([r[k] for r in train_rows])) for k in train_rows[0]}
        _log(f"epoch {epoch}/{epochs} evaluating validation set")
        val_stats = evaluate(model, val_loader, device, log_interval=batch_log_interval)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_stats.items()}, **{f"val_{k}": v for k, v in val_stats.items()}}
        history.append(row)
        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            save_mixture_catflow_checkpoint(model, checkpoint_path, config=cfg, report=row)
            _log(f"epoch {epoch}/{epochs} saved new best checkpoint val_loss={best_val:.4f}")
        if epoch == 1 or (progress_interval and epoch % progress_interval == 0) or epoch == epochs:
            epoch_elapsed = time.perf_counter() - epoch_started_at
            total_elapsed = time.perf_counter() - started_at
            _log(
                f"epoch={epoch:04d}/{epochs:04d} train_loss={train_stats['loss']:.4f} "
                f"val_loss={val_stats['loss']:.4f} val_node={val_stats['node_loss']:.4f} "
                f"val_edge={val_stats['edge_loss']:.4f} epoch_time={epoch_elapsed:.1f}s elapsed={total_elapsed:.1f}s"
            )
    report = {"best_val_loss": best_val, "history": history, "num_train": len(train_graphs), "num_val": len(val_graphs)}
    _log(f"saving training report best_val_loss={best_val:.4f}")
    save_json(report, out_dir / "training_report.json")
    print(f"Saved checkpoint to: {checkpoint_path}")
    print(f"Saved training report to: {out_dir / 'training_report.json'}")


if __name__ == "__main__":
    main()

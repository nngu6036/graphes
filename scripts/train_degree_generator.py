#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from grapher.data.io import load_dataset_splits
from grapher.generators.degree_vae import (
    DegreeVectorizer,
    build_degree_vae,
    degree_vae_loss,
    save_degree_vae_checkpoint,
)
from grapher.utils.device import resolve_torch_device
from grapher.utils.io import (
    ensure_dir,
    load_yaml,
    require_config,
    require_config_section,
    save_json,
)

DEFAULT_DEVICE = "auto"
DEFAULT_CHECKPOINT_PATH = Path("outputs/degree_generators/degree/checkpoint.pt")
PROGRESS_INTERVAL = 30


def _targets_to_tensors(
    targets: dict[str, np.ndarray], device: torch.device
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, value in targets.items():
        tensor = torch.as_tensor(value, device=device)
        if key in {"num_nodes", "num_nodes_count"}:
            tensor = tensor.long()
        else:
            tensor = tensor.float()
        out[key] = tensor
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a degree-histogram VAE for the coarse-to-fine generator."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--dataset", default=None, help="Optional dataset-name override."
    )
    parser.add_argument("--root", default=None, help="Optional dataset-root override.")
    args = parser.parse_args()

    config = load_yaml(args.config)
    degree_cfg = require_config_section(config, "degree_generator")
    seed = int(require_config(config, "seed"))
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = resolve_torch_device(degree_cfg.get("device", DEFAULT_DEVICE))
    checkpoint_path = Path(degree_cfg.get("checkpoint_path", DEFAULT_CHECKPOINT_PATH))
    out_dir = ensure_dir(checkpoint_path.parent)
    checkpoint_path = out_dir / checkpoint_path.name
    print(f"Using device: {device}", flush=True)

    dataset_cfg = config.get("dataset", {}) or {}
    dataset_name = str(args.dataset or dataset_cfg.get("name", "sbm"))
    dataset_root = str(args.root or dataset_cfg.get("root", "outputs/datasets"))
    dataset_config_path = Path(
        dataset_cfg.get(
            "config_path",
            Path("configs/datasets") / f"{dataset_name}.yaml",
        )
    )
    if not dataset_config_path.exists():
        raise FileNotFoundError(f"Missing dataset config: {dataset_config_path}")
    splits = load_dataset_splits(
        dataset_name,
        root=dataset_root,
        build_if_missing=bool(dataset_cfg.get("build_if_missing", False)),
        config_path=dataset_config_path,
    )
    train_graphs = list(splits["train"])
    max_train = dataset_cfg.get("max_train_graphs")
    if max_train is not None and int(max_train) > 0:
        train_graphs = train_graphs[: int(max_train)]
    if not train_graphs:
        raise RuntimeError(f"Dataset {dataset_name!r} has an empty training split.")
    constructor_cfg = require_config_section(config, "constructor")
    summary_cfg = require_config_section(config, "summary")
    require_connected = bool(
        require_config(
            constructor_cfg, "ensure_connected", context="config.constructor"
        )
    )
    max_degree_raw = require_config(
        summary_cfg, "degree_hist_max_degree", context="config.summary"
    )
    max_degree = None if max_degree_raw in {None, "auto"} else int(max_degree_raw)
    vectorizer = DegreeVectorizer.fit(
        train_graphs, max_degree=max_degree, require_connected=require_connected
    )
    x_np, targets_np = vectorizer.to_training_arrays(train_graphs)

    x = torch.as_tensor(x_np, dtype=torch.float32)
    dataset = TensorDataset(x, torch.arange(x.shape[0]))
    batch_size = int(
        require_config(degree_cfg, "batch_size", context="config.degree_generator")
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    all_targets = _targets_to_tensors(targets_np, device)

    model = build_degree_vae(
        vectorizer,
        latent_dim=int(
            require_config(degree_cfg, "latent_dim", context="config.degree_generator")
        ),
        hidden_dim=int(
            require_config(degree_cfg, "hidden_dim", context="config.degree_generator")
        ),
        size_condition_dim=int(degree_cfg.get("size_condition_dim", 16)),
        prior_type=str(degree_cfg.get("prior_type", "conditional_gmm")),
        prior_components=int(degree_cfg.get("prior_components", 4)),
        prior_hidden_dim=int(
            degree_cfg.get("prior_hidden_dim", degree_cfg.get("hidden_dim", 128))
        ),
        prior_logvar_min=float(degree_cfg.get("prior_logvar_min", -6.0)),
        prior_logvar_max=float(degree_cfg.get("prior_logvar_max", 4.0)),
        num_layers=int(
            require_config(degree_cfg, "num_layers", context="config.degree_generator")
        ),
        dropout=float(
            require_config(degree_cfg, "dropout", context="config.degree_generator")
        ),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(
            require_config(
                degree_cfg, "learning_rate", context="config.degree_generator"
            )
        ),
        weight_decay=float(
            require_config(
                degree_cfg, "weight_decay", context="config.degree_generator"
            )
        ),
    )
    weights = {
        "num_nodes": float(
            degree_cfg.get(
                "node_count_loss_weight",
                degree_cfg.get("node_weight", 1.0),
            )
        ),
        "degree": float(
            degree_cfg.get(
                "degree_histogram_loss_weight",
                degree_cfg.get("degree_weight", 5.0),
            )
        ),
        "degree_moment": float(
            degree_cfg.get(
                "degree_moment_loss_weight",
                degree_cfg.get("edge_moment_weight", 0.1),
            )
        ),
    }

    history: list[dict[str, float]] = []
    epochs = int(
        require_config(degree_cfg, "epochs", context="config.degree_generator")
    )
    kl_loss_weight = float(
        degree_cfg.get("kl_loss_weight", degree_cfg.get("beta", 0.005))
    )
    kl_warmup_epochs = int(degree_cfg.get("kl_warmup_epochs", 0))
    progress_interval = int(degree_cfg.get("progress_interval", PROGRESS_INTERVAL))
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_metrics: dict[str, list[float]] = {}
        for batch_x, batch_idx in loader:
            batch_x = batch_x.to(device)
            batch_targets = {
                key: value[batch_idx.to(device)] for key, value in all_targets.items()
            }
            outputs, mu, logvar = model(batch_x, batch_targets["num_nodes_count"])
            effective_beta = kl_loss_weight
            if kl_warmup_epochs > 0:
                effective_beta *= min(float(epoch) / kl_warmup_epochs, 1.0)
            loss, metrics = degree_vae_loss(
                outputs,
                batch_targets,
                mu,
                logvar,
                beta=effective_beta,
                weights=weights,
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
        mean_metrics["epoch"] = float(epoch)
        history.append(mean_metrics)
        if (
            epoch == 1
            or (progress_interval > 0 and epoch % progress_interval == 0)
            or epoch == epochs
        ):
            print(
                f"epoch={epoch:04d} loss={mean_metrics['loss']:.4f} "
                f"degree={mean_metrics['degree_loss']:.4f} nodes={mean_metrics['num_nodes_loss']:.4f} "
                f"moment={mean_metrics['degree_moment_loss']:.4f} "
                f"kl={mean_metrics['kl_loss']:.4f} beta={effective_beta:.6f}",
                flush=True,
            )

    save_degree_vae_checkpoint(
        checkpoint_path,
        model,
        vectorizer,
        config={
            "experiment_config": config,
            "dataset": {
                "name": dataset_name,
                "root": dataset_root,
                "config_path": str(dataset_config_path),
                "build_if_missing": bool(dataset_cfg.get("build_if_missing", False)),
                "num_train_graphs": len(train_graphs),
            },
        },
        metrics={"history": history, "final": history[-1] if history else {}},
    )
    vectorizer.save(out_dir / "degree_vectorizer.json")
    save_json(
        {"history": history, "final": history[-1] if history else {}},
        out_dir / "training_metrics.json",
    )
    print(f"Saved checkpoint to: {checkpoint_path}")
    print(f"Saved vectorizer to: {out_dir / 'degree_vectorizer.json'}")


if __name__ == "__main__":
    main()

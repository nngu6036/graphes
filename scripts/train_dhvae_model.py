from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from grapher.evaluation.data_io import load_dataset_splits
from grapher.evaluation.run_utils import make_model_run_config, run_output_dir
from grapher.generation.rewiring import connected_sequence_feasible, degree_sequence
from grapher.models.model_dhvae import DHVAE, encode_degree_sequence
from grapher.registry import available_datasets
from grapher.utils.compute import CudaTrainingDeviceError, PeakMemoryMonitor, compute_report, require_cuda_training_device
from grapher.utils.io import load_yaml, save_json, save_yaml, stable_hash
from grapher.utils.logging import configure_logging, get_logger
from grapher.utils.numerics import assert_model_tensors_finite
from grapher.utils.seed import set_seed

logger = get_logger(__name__)


def _as_int_or_none(value, default: int | None = None) -> int | None:
    if value is None or value == "":
        return default
    return int(value)


def _collect_degree_sequences(graphs: Sequence, *, require_connected_feasible: bool) -> tuple[list[list[int]], dict]:
    sequences: list[list[int]] = []
    skipped: dict[str, int] = {}
    for graph in graphs:
        seq = degree_sequence(graph)
        if require_connected_feasible:
            feasible, reason = connected_sequence_feasible(seq)
            if not feasible:
                skipped[reason] = skipped.get(reason, 0) + 1
                continue
        # DH-VAE explicitly represents degree 0 as histogram bin m_0, so we do
        # not drop zero-degree sequences here.
        sequences.append(seq)
    return sequences, skipped


def _histogram_tensor(sequences: Sequence[Sequence[int]], histogram_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    rows = [encode_degree_sequence(seq, histogram_dim) for seq in sequences]
    if not rows:
        raise ValueError("No degree sequences were available for DH-VAE training after filtering.")
    sizes = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    return torch.stack(rows, dim=0).float(), sizes


def _size_counts(sizes: Sequence[int], max_nodes: int) -> list[int]:
    counts = np.bincount(np.asarray([int(n) for n in sizes], dtype=np.int64), minlength=int(max_nodes) + 1)
    return [int(v) for v in counts[: int(max_nodes) + 1]]


def _num_trainable_parameters(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def train_dhvae(
    *,
    dataset: str,
    model_config: dict,
    dataset_config_path: Path,
    dataset_root: str,
    seed: int,
    run_id: int | None,
    device: str,
    require_connected_feasible: bool = True,
) -> dict:
    """Train the paper-aligned size-conditioned DH-VAE degree prior."""

    device = require_cuda_training_device(device)
    logger.info(
        "Starting DH-VAE training dataset=%s run_id=%s seed=%s device=%s dataset_root=%s dataset_config=%s",
        dataset,
        run_id,
        seed,
        device,
        dataset_root,
        dataset_config_path,
    )
    set_seed(seed, include_torch=True)
    cfg = make_model_run_config(
        model_config,
        dataset=dataset,
        model="dhvae",
        run_id=run_id,
        seed=seed,
        use_run_paths=run_id is not None,
    )
    logger.debug("Resolved DH-VAE config: %s", cfg)
    splits = load_dataset_splits(dataset, output_root=dataset_root, build_if_missing=True, config_path=dataset_config_path)
    logger.info("Loaded dataset splits: %s", {split: len(graphs) for split, graphs in splits.items()})
    train_sequences, skipped = _collect_degree_sequences(splits["train"], require_connected_feasible=require_connected_feasible)
    if not train_sequences:
        raise ValueError(
            "DH-VAE training has no usable degree sequences. "
            f"Skipped counts: {skipped}. Try --allow-disconnected-degree-sequences for diagnostics."
        )
    logger.info(
        "Collected DH-VAE sequences used=%d skipped=%d skipped_by_reason=%s require_connected_feasible=%s",
        len(train_sequences),
        int(sum(skipped.values())),
        skipped,
        require_connected_feasible,
    )

    max_degree = max(max(seq) if seq else 0 for seq in train_sequences)
    max_nodes_observed = max(len(seq) for seq in train_sequences)
    min_nodes_observed = min(len(seq) for seq in train_sequences)

    configured_max_nodes = _as_int_or_none(cfg.get("max_nodes"), None)
    if configured_max_nodes is None:
        # Backward-compatible alias used by older degree-prior configs.
        configured_max_nodes = _as_int_or_none(cfg.get("max_input_dim"), None)

    if configured_max_nodes is not None and int(configured_max_nodes) < int(max_nodes_observed):
        raise ValueError(
            f"Configured max_nodes={int(configured_max_nodes)} is smaller than the largest "
            f"training graph size max_nodes_observed={int(max_nodes_observed)}."
        )

    # DH-VAE represents a graph with at most N nodes by the degree histogram
    # h_D=(m_0,...,m_{N-1}). Therefore the histogram width is not an
    # independent hyperparameter: it is exactly max_nodes.
    max_nodes = int(configured_max_nodes or max_nodes_observed)
    if int(max_degree) >= int(max_nodes):
        raise ValueError(
            f"Observed degree {int(max_degree)} cannot be represented with max_nodes={int(max_nodes)}. "
            "For a simple graph, every degree should be at most n-1; check dataset preprocessing."
        )
    histogram_dim = int(max_nodes)

    histograms, sizes = _histogram_tensor(train_sequences, histogram_dim)
    if torch.any(sizes > max_nodes):
        raise ValueError("Internal error: a training graph size exceeds configured max_nodes.")
    logger.info(
        "DH-VAE tensorized data histograms=%s sizes=%s min_nodes=%d max_nodes_observed=%d max_nodes_model=%d max_degree=%d",
        tuple(histograms.shape),
        tuple(sizes.shape),
        int(min_nodes_observed),
        int(max_nodes_observed),
        int(max_nodes),
        int(max_degree),
    )

    batch_size = int(cfg.get("batch_size", 64))
    loader = DataLoader(TensorDataset(histograms, sizes), batch_size=batch_size, shuffle=True)
    logger.info("DH-VAE DataLoader batches=%d batch_size=%d", len(loader), batch_size)

    model = DHVAE(
        max_nodes=max_nodes,
        histogram_dim=histogram_dim,
        hidden_dim=int(cfg.get("hidden_dim", 128)),
        latent_dim=int(cfg.get("latent_dim", 32)),
        size_embedding_dim=int(cfg.get("size_embedding_dim", 32)),
    )
    size_counts = _size_counts([int(n) for n in sizes.tolist()], max_nodes=max_nodes)
    model.set_size_distribution(torch.tensor(size_counts, dtype=torch.float32), values_are_counts=True)
    model.to(device)
    logger.info(
        "DH-VAE model params=%d hidden_dim=%d latent_dim=%d size_embedding_dim=%d nonzero_size_bins=%s",
        _num_trainable_parameters(model),
        int(cfg.get("hidden_dim", 128)),
        int(cfg.get("latent_dim", 32)),
        int(cfg.get("size_embedding_dim", 32)),
        {i: c for i, c in enumerate(size_counts) if c > 0},
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("learning_rate", 1e-3)), weight_decay=float(cfg.get("weight_decay", 0.0)))
    beta_kl = float(cfg.get("beta_kl", 1e-3))
    epochs = int(cfg.get("epochs", 50))
    logger.info(
        "DH-VAE optimization epochs=%d lr=%g weight_decay=%g beta_kl=%g grad_clip=%g",
        epochs,
        float(cfg.get("learning_rate", 1e-3)),
        float(cfg.get("weight_decay", 0.0)),
        beta_kl,
        float(cfg.get("grad_clip", 5.0)),
    )

    history: list[dict] = []
    start = time.perf_counter()
    with PeakMemoryMonitor() as memory_monitor:
        for epoch in range(1, epochs + 1):
            epoch_start = time.perf_counter()
            model.train()
            epoch_loss = 0.0
            epoch_recon = 0.0
            epoch_kl = 0.0
            n_seen = 0
            for batch_idx, (batch_hist, batch_sizes) in enumerate(loader, start=1):
                batch_hist = batch_hist.to(device)
                batch_sizes = batch_sizes.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss_bundle = model.loss(batch_hist, batch_sizes, beta_kl=beta_kl)
                loss = loss_bundle.loss
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite DH-VAE loss at epoch {epoch}.")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg.get("grad_clip", 5.0)))
                optimizer.step()
                size = int(batch_hist.size(0))
                epoch_loss += float(loss.item()) * size
                epoch_recon += float(loss_bundle.reconstruction_loss.item()) * size
                epoch_kl += float(loss_bundle.kl_loss.item()) * size
                n_seen += size
                if logger.isEnabledFor(10) and (batch_idx == 1 or batch_idx == len(loader)):
                    logger.debug(
                        "DH-VAE epoch=%d batch=%d/%d batch_size=%d size_range=(%d,%d) loss=%.6f nll=%.6f kl=%.6f",
                        epoch,
                        batch_idx,
                        len(loader),
                        size,
                        int(batch_sizes.min().item()),
                        int(batch_sizes.max().item()),
                        float(loss.item()),
                        float(loss_bundle.reconstruction_loss.item()),
                        float(loss_bundle.kl_loss.item()),
                    )
            row = {
                "epoch": epoch,
                "loss": epoch_loss / max(n_seen, 1),
                "multinomial_nll": epoch_recon / max(n_seen, 1),
                "reconstruction_loss": epoch_recon / max(n_seen, 1),
                "kl_loss": epoch_kl / max(n_seen, 1),
                "epoch_seconds": time.perf_counter() - epoch_start,
            }
            history.append(row)
            if epoch == 1 or epoch == epochs or epoch % max(1, epochs // 10) == 0:
                logger.info(
                    "DH-VAE epoch %d/%d loss=%.4f nll=%.4f kl=%.4f seconds=%.2f",
                    epoch,
                    epochs,
                    row["loss"],
                    row["multinomial_nll"],
                    row["kl_loss"],
                    row["epoch_seconds"],
                )
        assert_model_tensors_finite(model, context=f"dhvae/{dataset}")
    elapsed = time.perf_counter() - start

    checkpoint_path = Path(cfg.get("checkpoint_path") or f"outputs/checkpoints/{dataset}/dhvae/dhvae.pt")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    model_params = {
        "architecture": DHVAE.architecture,
        "max_nodes": int(max_nodes),
        "histogram_dim": int(histogram_dim),
        "hidden_dim": int(cfg.get("hidden_dim", 128)),
        "latent_dim": int(cfg.get("latent_dim", 32)),
        "size_embedding_dim": int(cfg.get("size_embedding_dim", 32)),
        # Compatibility keys consumed by older metadata readers.
        "max_input_dim": int(histogram_dim),
        "max_frequency": int(max_nodes + 1),
        "num_nodes": int(max_nodes_observed),
    }
    payload = {
        "model_state_dict": model.state_dict(),
        "model_name": "dhvae",
        "model_params": model_params,
        "dataset": dataset,
        "seed": seed,
        "run_id": run_id,
        "training_config": cfg,
        "training_history": history,
        "degree_sequence_stats": {
            "num_train_graphs": len(splits["train"]),
            "num_used_sequences": len(train_sequences),
            "num_skipped": int(sum(skipped.values())),
            "skipped_by_reason": skipped,
            "min_nodes": int(min(len(seq) for seq in train_sequences)),
            "max_nodes_observed": int(max_nodes_observed),
            "max_nodes_model": int(max_nodes),
            "histogram_dim": int(histogram_dim),
            "max_degree": int(max_degree),
            "size_counts": size_counts,
            "size_distribution": {
                str(i): float(c / max(sum(size_counts), 1)) for i, c in enumerate(size_counts) if c > 0
            },
            "histogram_bins": "degree values 0..histogram_dim-1, padded above n-1",
            "likelihood": "Multinomial(n, pi_theta(k | z, n))",
        },
    }
    torch.save(payload, checkpoint_path)
    logger.info("Wrote DH-VAE checkpoint payload keys=%s", sorted(payload.keys()))

    run_dir = run_output_dir(dataset, "dhvae", run_id=run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(cfg, run_dir / "resolved_model_config.yaml", force=True)
    compute = compute_report(operation="training", runtime_seconds=elapsed, num_graphs=len(train_sequences), memory=memory_monitor.to_dict())
    metadata = {
        "dataset": dataset,
        "model": "dhvae",
        "seed": seed,
        "run_id": run_id,
        "runtime_seconds": elapsed,
        "checkpoint_path": str(checkpoint_path),
        "model_config_hash": stable_hash(cfg),
        "compute": compute,
        "history_tail": history[-5:],
        "degree_sequence_stats": payload["degree_sequence_stats"],
    }
    save_json(metadata, run_dir / "train_metadata.json", force=True)
    logger.info("Saved DH-VAE checkpoint to %s", checkpoint_path)
    logger.info("Saved DH-VAE metadata to %s", run_dir / "train_metadata.json")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the size-conditioned DH-VAE degree-sequence prior.")
    parser.add_argument("--dataset", required=True, choices=available_datasets())
    parser.add_argument("--model-config", type=str, default="configs/models/dhvae.yaml")
    parser.add_argument("--dataset-config", type=str, default=None)
    parser.add_argument("--dataset-root", type=str, default="outputs/datasets")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging.")
    parser.add_argument("--allow-disconnected-degree-sequences", action="store_true", help="Do not require connected-realizable sequences before training.")
    args = parser.parse_args()
    if args.debug:
        configure_logging("DEBUG")
    model_cfg = load_yaml(args.model_config)
    dataset_cfg = Path(args.dataset_config) if args.dataset_config else Path("configs/datasets") / f"{args.dataset}.yaml"
    try:
        train_dhvae(
            dataset=args.dataset,
            model_config=model_cfg,
            dataset_config_path=dataset_cfg,
            dataset_root=args.dataset_root,
            seed=args.seed,
            run_id=args.run_id,
            device=args.device,
            require_connected_feasible=not args.allow_disconnected_degree_sequences,
        )
    except CudaTrainingDeviceError as exc:
        parser.exit(status=2, message=f"error: {exc}\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from grapher.utils.device import resolve_torch_device
from grapher.utils.io import ensure_dir, load_yaml, save_json


def _log(message: str) -> None:
    print(f"[progress] {message}", flush=True)


@dataclass
class SelectorExample:
    features: torch.Tensor
    teacher_probs: torch.Tensor
    deltas: torch.Tensor


class RewiringTeacherDataset(Dataset):
    def __init__(self, examples: list[SelectorExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> SelectorExample:
        return self.examples[idx]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _as_float_array(value: Any) -> np.ndarray:
    if value is None:
        return np.zeros(0, dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    arr[~np.isfinite(arr)] = 0.0
    return arr


def _pad_or_trim(value: Any, width: int) -> np.ndarray:
    arr = _as_float_array(value)
    out = np.zeros(width, dtype=np.float32)
    if arr.size:
        out[: min(width, arr.size)] = arr[:width]
    return out


def _safe_scalar(value: Any, default: float = 0.0) -> float:
    try:
        val = float(value)
        return val if math.isfinite(val) else default
    except Exception:
        return default


def _build_graph(num_nodes: int, edges: list[list[int]]) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(int(num_nodes)))
    graph.add_edges_from((int(u), int(v)) for u, v in edges)
    return graph


def _edge(edge_like: list[int] | tuple[int, int]) -> tuple[int, int]:
    u, v = int(edge_like[0]), int(edge_like[1])
    return (u, v) if u < v else (v, u)


def _common_neighbors(graph: nx.Graph, edge: tuple[int, int]) -> int:
    u, v = edge
    if u not in graph or v not in graph:
        return 0
    return len(set(graph.neighbors(u)).intersection(graph.neighbors(v)))


def _graph_context_features(
    graph: nx.Graph,
    target: dict[str, Any],
    feature_cfg: dict[str, int],
) -> np.ndarray:
    n = max(int(graph.number_of_nodes()), 1)
    m = int(graph.number_of_edges())
    degrees = np.asarray([d for _, d in graph.degree()], dtype=np.float32)
    if degrees.size == 0:
        degrees = np.zeros(1, dtype=np.float32)

    density = float(nx.density(graph)) if n > 1 else 0.0
    triangles = float(sum(nx.triangles(graph).values()) / 3.0) if n else 0.0
    triangle_norm = triangles / max(n, 1)
    transitivity = float(nx.transitivity(graph)) if m > 0 else 0.0
    avg_clustering = float(nx.average_clustering(graph)) if n else 0.0

    target_n = max(_safe_scalar(target.get("num_nodes", n), n), 1.0)
    target_m = _safe_scalar(target.get("num_edges", m), m)
    target_density = _safe_scalar(target.get("density", density), density)
    target_triangle = _safe_scalar(
        target.get("triangle_count_norm", triangle_norm),
        triangle_norm,
    )

    scalar = np.asarray(
        [
            n / 256.0,
            m / max(n * n, 1),
            density,
            float(degrees.mean()) / 256.0,
            float(degrees.std()) / 256.0,
            float(degrees.max()) / 256.0,
            triangle_norm,
            transitivity,
            avg_clustering,
            target_n / 256.0,
            target_m / max(target_n * target_n, 1.0),
            target_density,
            target_triangle,
            target_density - density,
            target_triangle - triangle_norm,
        ],
        dtype=np.float32,
    )

    target_vecs = [
        _pad_or_trim(target.get("degree_hist", []), feature_cfg["degree_width"]),
        _pad_or_trim(target.get("clustering_hist", []), feature_cfg["clustering_width"]),
        _pad_or_trim(target.get("spectral_hist", []), feature_cfg["spectral_width"]),
        _pad_or_trim(target.get("motif_proxy", []), feature_cfg["motif_width"]),
        _pad_or_trim(target.get("orbit_count", []), feature_cfg["orbit_width"]),
    ]

    return np.concatenate([scalar, *target_vecs], axis=0).astype(np.float32)


def _action_local_features(graph: nx.Graph, action_record: dict[str, Any]) -> np.ndarray:
    n = max(int(graph.number_of_nodes()), 1)

    removed = [_edge(e) for e in action_record.get("removed", [])]
    added = [_edge(e) for e in action_record.get("added", [])]

    degree = dict(graph.degree())
    clustering = nx.clustering(graph)

    endpoints: list[int] = []
    for e in removed:
        endpoints.extend([e[0], e[1]])

    endpoints = endpoints[:4]
    while len(endpoints) < 4:
        endpoints.append(0)

    endpoint_degrees = np.asarray(
        [degree.get(v, 0) / max(n - 1, 1) for v in endpoints],
        dtype=np.float32,
    )
    endpoint_clust = np.asarray(
        [clustering.get(v, 0.0) for v in endpoints],
        dtype=np.float32,
    )

    removed_common = np.asarray(
        [_common_neighbors(graph, e) for e in removed],
        dtype=np.float32,
    )

    graph_after_remove = graph.copy()
    for u, v in removed:
        if graph_after_remove.has_edge(u, v):
            graph_after_remove.remove_edge(u, v)

    added_common = np.asarray(
        [_common_neighbors(graph_after_remove, e) for e in added],
        dtype=np.float32,
    )

    rem_sum = float(removed_common.sum())
    add_sum = float(added_common.sum())
    delta_triangles = (add_sum - rem_sum) / max(n, 1)

    add_degree_pairs = []
    for u, v in added:
        add_degree_pairs.append(
            (degree.get(u, 0) + degree.get(v, 0)) / max(2 * (n - 1), 1)
        )
    while len(add_degree_pairs) < 2:
        add_degree_pairs.append(0.0)

    rem_degree_pairs = []
    for u, v in removed:
        rem_degree_pairs.append(
            (degree.get(u, 0) + degree.get(v, 0)) / max(2 * (n - 1), 1)
        )
    while len(rem_degree_pairs) < 2:
        rem_degree_pairs.append(0.0)

    return np.asarray(
        [
            *endpoint_degrees.tolist(),
            *endpoint_clust.tolist(),
            rem_sum / max(n, 1),
            add_sum / max(n, 1),
            float(removed_common.mean()) / max(n, 1) if removed_common.size else 0.0,
            float(added_common.mean()) / max(n, 1) if added_common.size else 0.0,
            delta_triangles,
            *rem_degree_pairs[:2],
            *add_degree_pairs[:2],
        ],
        dtype=np.float32,
    )


def _record_to_example(
    record: dict[str, Any],
    feature_cfg: dict[str, int],
) -> SelectorExample | None:
    actions = record.get("actions", [])
    probs = np.asarray(record.get("teacher_probs", []), dtype=np.float32)

    if not actions or probs.size != len(actions):
        return None
    if probs.sum() <= 0:
        return None

    probs = probs / probs.sum()

    graph = _build_graph(int(record["num_nodes"]), record["edges"])
    target = record.get("target_summary", {}) or {}
    context = _graph_context_features(graph, target, feature_cfg)

    features = []
    deltas = []

    for action in actions:
        local = _action_local_features(graph, action)
        features.append(np.concatenate([context, local], axis=0))
        deltas.append(_safe_scalar(action.get("delta_energy", 0.0), 0.0))

    return SelectorExample(
        features=torch.tensor(np.stack(features, axis=0), dtype=torch.float32),
        teacher_probs=torch.tensor(probs, dtype=torch.float32),
        deltas=torch.tensor(deltas, dtype=torch.float32),
    )


def load_teacher_examples(
    path: Path,
    feature_cfg: dict[str, int],
    max_records: int | None = None,
) -> list[SelectorExample]:
    records = _read_jsonl(path)
    if max_records is not None:
        records = records[: int(max_records)]

    examples: list[SelectorExample] = []
    skipped = 0

    for rec in records:
        ex = _record_to_example(rec, feature_cfg)
        if ex is None:
            skipped += 1
            continue
        examples.append(ex)

    if not examples:
        raise RuntimeError(f"No valid teacher examples loaded from {path}; skipped={skipped}")

    return examples


class CandidateMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        layers: list[nn.Module] = []
        dim = input_dim

        for _ in range(max(int(num_layers) - 1, 1)):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            dim = hidden_dim

        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _batch_loss(
    model: nn.Module,
    batch: list[SelectorExample],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    losses = []
    top1 = []
    mean_pred_delta = []
    entropy = []

    for ex in batch:
        x = ex.features.to(device)
        q = ex.teacher_probs.to(device)
        deltas = ex.deltas.to(device)

        logits = model(x)
        log_probs = torch.log_softmax(logits, dim=0)
        loss = -(q * log_probs).sum()
        losses.append(loss)

        pred_idx = int(torch.argmax(logits).item())
        teacher_idx = int(torch.argmax(q).item())

        top1.append(float(pred_idx == teacher_idx))
        mean_pred_delta.append(float(deltas[pred_idx].detach().cpu().item()))
        entropy.append(
            float((-(torch.softmax(logits, dim=0) * log_probs).sum()).detach().cpu().item())
        )

    return torch.stack(losses).mean(), {
        "top1": float(np.mean(top1)) if top1 else 0.0,
        "mean_predicted_delta": float(np.mean(mean_pred_delta)) if mean_pred_delta else 0.0,
        "pred_entropy": float(np.mean(entropy)) if entropy else 0.0,
    }


def _iter_batches(
    examples: list[SelectorExample],
    batch_size: int,
    rng: random.Random,
    shuffle: bool = True,
):
    indices = list(range(len(examples)))
    if shuffle:
        rng.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        yield [examples[i] for i in indices[start : start + batch_size]]


def evaluate(
    model: nn.Module,
    examples: list[SelectorExample],
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    losses = []
    top1 = []
    mean_pred_delta = []
    pred_entropy = []
    rng = random.Random(0)

    with torch.no_grad():
        for batch in _iter_batches(examples, batch_size, rng, shuffle=False):
            loss, stats = _batch_loss(model, batch, device)
            losses.append(float(loss.cpu().item()))
            top1.append(stats["top1"])
            mean_pred_delta.append(stats["mean_predicted_delta"])
            pred_entropy.append(stats["pred_entropy"])

    return {
        "loss": float(np.mean(losses)),
        "top1": float(np.mean(top1)),
        "mean_predicted_delta": float(np.mean(mean_pred_delta)),
        "pred_entropy": float(np.mean(pred_entropy)),
    }


def train_selector(
    *,
    config: dict[str, Any],
    teacher_dir: Path,
    output_dir: Path,
    epochs: int | None,
    batch_size: int | None,
    seed: int,
    max_train_records: int | None,
    max_val_records: int | None,
    device: str | None = None,
    progress_interval: int | None = None,
    batch_log_interval: int | None = None,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    ensure_dir(output_dir)
    _log(f"starting selector training seed={seed} teacher_dir={teacher_dir} output_dir={output_dir}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    selector_cfg = config.get("selector", {}) or {}
    torch_device = resolve_torch_device(device if device is not None else selector_cfg.get("device", "auto"))
    _log(f"using device={torch_device}")

    feature_cfg = {
        "degree_width": int(selector_cfg.get("degree_width", 64)),
        "clustering_width": int(selector_cfg.get("clustering_width", 20)),
        "spectral_width": int(selector_cfg.get("spectral_width", 20)),
        "motif_width": int(selector_cfg.get("motif_width", 5)),
        "orbit_width": int(selector_cfg.get("orbit_width", 15)),
    }

    train_path = teacher_dir / "train.jsonl"
    val_path = teacher_dir / "val.jsonl"

    if not train_path.exists():
        raise FileNotFoundError(f"Missing teacher train file: {train_path}")

    _log(f"loading train examples from {train_path}")
    train_examples = load_teacher_examples(train_path, feature_cfg, max_train_records)
    _log(f"loaded train examples={len(train_examples)}")

    if val_path.exists() and val_path.stat().st_size > 0:
        _log(f"loading val examples from {val_path}")
        val_examples = load_teacher_examples(val_path, feature_cfg, max_val_records)
        _log(f"loaded val examples={len(val_examples)}")
    else:
        split = max(1, int(0.1 * len(train_examples)))
        val_examples = train_examples[:split]
        train_examples = train_examples[split:]
        _log(f"no val file found; split train examples into train={len(train_examples)} val={len(val_examples)}")

    input_dim = int(train_examples[0].features.shape[-1])

    hidden_dim = int(selector_cfg.get("hidden_dim", 256))
    num_layers = int(selector_cfg.get("num_layers", 3))
    dropout = float(selector_cfg.get("dropout", 0.1))
    lr = float(selector_cfg.get("learning_rate", 3.0e-4))
    weight_decay = float(selector_cfg.get("weight_decay", 1.0e-5))

    epochs = int(epochs if epochs is not None else selector_cfg.get("epochs", 100))
    batch_size = int(batch_size if batch_size is not None else selector_cfg.get("batch_size", 64))
    if progress_interval is None:
        progress_interval = int(selector_cfg.get("progress_interval", 10))
    if batch_log_interval is None:
        batch_log_interval = int(selector_cfg.get("batch_log_interval", 0))
    progress_interval = max(int(progress_interval), 0)
    batch_log_interval = max(int(batch_log_interval), 0)
    batches_per_epoch = math.ceil(len(train_examples) / max(batch_size, 1))
    _log(
        f"model input_dim={input_dim} hidden_dim={hidden_dim} layers={num_layers} dropout={dropout} "
        f"epochs={epochs} batch_size={batch_size} batches_per_epoch={batches_per_epoch}"
    )
    _log(f"optimizer AdamW lr={lr} weight_decay={weight_decay}")

    model = CandidateMLP(
        input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(torch_device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    rng = random.Random(seed)
    best_val = float("inf")
    best_report: dict[str, Any] = {}
    history = []

    for epoch in range(1, epochs + 1):
        epoch_started_at = time.perf_counter()
        _log(f"epoch {epoch}/{epochs} starting")
        model.train()

        train_losses = []
        train_top1 = []
        train_delta = []

        for batch_idx, batch in enumerate(_iter_batches(train_examples, batch_size, rng, shuffle=True), start=1):
            optimizer.zero_grad(set_to_none=True)

            loss, stats = _batch_loss(model, batch, torch_device)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            train_losses.append(float(loss.detach().cpu().item()))
            train_top1.append(stats["top1"])
            train_delta.append(stats["mean_predicted_delta"])

            if batch_log_interval and (batch_idx == 1 or batch_idx % batch_log_interval == 0 or batch_idx == batches_per_epoch):
                _log(
                    f"epoch {epoch}/{epochs} batch {batch_idx}/{batches_per_epoch} "
                    f"loss={float(loss.detach().cpu().item()):.4f} top1={stats['top1']:.3f}"
                )

        _log(f"epoch {epoch}/{epochs} evaluating validation set")
        val = evaluate(model, val_examples, batch_size, torch_device)

        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "train_top1": float(np.mean(train_top1)),
            "train_mean_predicted_delta": float(np.mean(train_delta)),
            "val_loss": val["loss"],
            "val_top1": val["top1"],
            "val_mean_predicted_delta": val["mean_predicted_delta"],
            "val_pred_entropy": val["pred_entropy"],
        }

        history.append(row)

        if val["loss"] < best_val:
            best_val = val["loss"]
            best_report = dict(row)

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": input_dim,
                    "hidden_dim": hidden_dim,
                    "num_layers": num_layers,
                    "dropout": dropout,
                    "feature_cfg": feature_cfg,
                    "selector_cfg": selector_cfg,
                    "seed": seed,
                },
                output_dir / "checkpoint.pt",
            )
            _log(f"epoch {epoch}/{epochs} saved new best checkpoint val_loss={best_val:.4f}")

        epoch_elapsed = time.perf_counter() - epoch_started_at
        should_log_epoch = epoch == 1 or (progress_interval and epoch % progress_interval == 0) or epoch == epochs
        if should_log_epoch:
            elapsed_total = time.perf_counter() - started_at
            _log(
                f"epoch={epoch:04d}/{epochs:04d} "
                f"train_loss={row['train_loss']:.4f} train_top1={row['train_top1']:.3f} "
                f"val_loss={val['loss']:.4f} val_top1={val['top1']:.3f} "
                f"val_delta={val['mean_predicted_delta']:.6f} "
                f"epoch_time={epoch_elapsed:.1f}s elapsed={elapsed_total:.1f}s"
            )

    report = {
        "seed": seed,
        "teacher_dir": str(teacher_dir),
        "output_dir": str(output_dir),
        "num_train_examples": len(train_examples),
        "num_val_examples": len(val_examples),
        "input_dim": input_dim,
        "best_val_loss": best_val,
        "best": best_report,
        "val_loss": best_report.get("val_loss", best_val),
        "val_top1": best_report.get("val_top1"),
        "val_mean_predicted_delta": best_report.get("val_mean_predicted_delta"),
        "history": history,
    }

    _log(f"saving training report best_val_loss={best_val:.4f}")
    save_json(report, output_dir / "training_report.json")

    print(f"Saved selector checkpoint to: {output_dir / 'checkpoint.pt'}")
    print(f"Saved training report to:    {output_dir / 'training_report.json'}")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a learned GraphER rewiring-action selector from an energy-guided teacher cache."
    )

    parser.add_argument("--config", required=True)
    parser.add_argument("--teacher-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None, help="Torch device. Defaults to CUDA when available, otherwise CPU.")
    parser.add_argument("--max-train-records", type=int, default=None)
    parser.add_argument("--max-val-records", type=int, default=None)
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=None,
        help="Print epoch summary every N epochs. Use 0 to keep only stage/checkpoint logs.",
    )
    parser.add_argument(
        "--batch-log-interval",
        type=int,
        default=None,
        help="Print batch progress every N batches. Use 0 to disable batch logs.",
    )

    args = parser.parse_args()

    config = load_yaml(args.config)
    seed = int(args.seed if args.seed is not None else config.get("seed", 0))

    train_selector(
        config=config,
        teacher_dir=Path(args.teacher_dir),
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=seed,
        max_train_records=args.max_train_records,
        max_val_records=args.max_val_records,
        device=args.device,
        progress_interval=args.progress_interval,
        batch_log_interval=args.batch_log_interval,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Generate an exact raw GraphRNN batch and publish a neutral NPZ export."""

from __future__ import annotations

import argparse
import hashlib
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from common import (
    CHECKPOINT_FORMAT,
    atomic_json,
    generate_graphs,
    resolve_device,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphrnn-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--num-graphs", required=True, type=int)
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sample-time", type=int, default=None)
    parser.add_argument("--progress-every-batches", type=int, default=1)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = Path(args.graphrnn_root).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    if not (root / "model.py").is_file():
        raise FileNotFoundError(root / "model.py")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if args.num_graphs <= 0 or args.batch_size <= 0:
        raise ValueError("num-graphs and batch-size must be positive.")
    if args.progress_every_batches <= 0:
        raise ValueError("progress-every-batches must be positive.")

    checkpoint = _torch_load(checkpoint_path)
    if not isinstance(checkpoint, dict) or checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise RuntimeError("Unsupported GraphRNN checkpoint format.")
    config = dict(checkpoint.get("resolved_config", {}))
    if args.sample_time is not None:
        if args.sample_time <= 0:
            raise ValueError("sample-time must be positive.")
        config["sample_time"] = int(args.sample_time)
    device = resolve_device(args.device)
    print(
        "GraphRNN generation worker: "
        f"python={platform.python_version()} torch={torch.__version__} "
        f"device={device}",
        flush=True,
    )
    adjacency, sizes, statistics = generate_graphs(
        graphrnn_root=root,
        checkpoint_path=checkpoint_path,
        config=config,
        num_graphs=args.num_graphs,
        batch_size=args.batch_size,
        seed=args.seed,
        device=device,
        progress_every_batches=args.progress_every_batches,
    )
    _atomic_npz(
        output_path,
        adjacency=adjacency,
        num_nodes=sizes,
        sample_index=np.arange(args.num_graphs, dtype=np.int64),
    )
    output_hash = _sha256(output_path)
    manifest = {
        "format": "grapher_graphrnn_export_v1",
        "num_generated": int(args.num_graphs),
        "generation_seed": int(args.seed),
        "batch_size": int(args.batch_size),
        "sample_order": "graphrnn_worker_index_ascending",
        "max_export_nodes": int(adjacency.shape[1]),
        "node_count": {
            "min": int(sizes.min()) if len(sizes) else None,
            "max": int(sizes.max()) if len(sizes) else None,
            "mean": float(sizes.mean()) if len(sizes) else None,
        },
        "statistics": statistics,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": _sha256(checkpoint_path),
            "epoch": int(checkpoint["epoch"]),
        },
        "output": {"path": output_path.name, "sha256": output_hash},
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "numpy_version": str(np.__version__),
    }
    atomic_json(manifest_path, manifest)
    print(f"GraphRNN neutral export: {output_path}", flush=True)
    print(f"GraphRNN export manifest: {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

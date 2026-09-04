#!/usr/bin/env python3
"""Isolated GDSS training worker over GraphER-neutral split tensors."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_config(path: Path):
    import yaml
    from types import SimpleNamespace

    def convert(value):
        if isinstance(value, dict):
            return SimpleNamespace(**{str(k): convert(v) for k, v in value.items()})
        if isinstance(value, list):
            return [convert(v) for v in value]
        return value

    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise TypeError(f"GDSS resolved config must be a mapping: {path}")
    return convert(value)


def _build_loader(path: Path, config, *, domain: str, shuffle: bool):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    with np.load(path, allow_pickle=False) as payload:
        adjacency_np = np.asarray(payload["adjacency"], dtype=np.float32)
        num_nodes = np.asarray(payload["num_nodes"], dtype=np.int64)
        node_types = np.asarray(payload["node_types"], dtype=np.int64) if "node_types" in payload.files else None

    adjacency = torch.tensor(adjacency_np, dtype=torch.float32)
    batch, max_nodes, _ = adjacency.shape
    max_feat_num = int(config.data.max_feat_num)
    if domain == "generic":
        degrees = adjacency.sum(dim=-1).to(torch.long)
        if degrees.numel() and int(degrees.max()) >= max_feat_num:
            raise ValueError(
                f"GDSS max_feat_num={max_feat_num} cannot encode observed degree {int(degrees.max())}."
            )
        x = F.one_hot(degrees, num_classes=max_feat_num).to(torch.float32)
        active = torch.arange(max_nodes)[None, :] < torch.tensor(num_nodes)[:, None]
        x = x * active[:, :, None].to(torch.float32)
    else:
        if node_types is None:
            raise ValueError(f"Molecular GDSS split is missing node_types: {path}")
        x = torch.zeros((batch, max_nodes, max_feat_num), dtype=torch.float32)
        active = torch.tensor(node_types >= 0)
        if active.any():
            rows, cols = torch.where(active)
            classes = torch.tensor(node_types, dtype=torch.long)[rows, cols]
            if int(classes.max()) >= max_feat_num:
                raise ValueError("GDSS molecular atom class exceeds max_feat_num.")
            x[rows, cols, classes] = 1.0

    dataset = TensorDataset(x, adjacency)
    return DataLoader(
        dataset,
        batch_size=int(config.data.batch_size),
        shuffle=shuffle,
        num_workers=int(getattr(config.data, "num_workers", 0)),
        drop_last=False,
    )



def _install_evaluation_stub() -> None:
    """Bypass GDSS native metric imports during pure train/sample workers.

    ``utils.loader`` imports ``evaluation.mmd`` only so its separate evaluation
    helper can expose Gaussian kernels. GraphER never calls that evaluator from
    these workers, and the attached release otherwise drags in pyemd/dill.
    """
    import types

    module = types.ModuleType("evaluation.mmd")

    def unavailable(*_args, **_kwargs):
        raise RuntimeError("GDSS native MMD evaluation is not invoked by the GraphER wrapper.")

    module.gaussian = unavailable
    module.gaussian_emd = unavailable
    sys.modules["evaluation.mmd"] = module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gdss-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--domain", choices=("generic", "attributed"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    root = args.gdss_root.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    manifest = args.manifest.expanduser().resolve()
    sys.path.insert(0, str(root))

    _install_evaluation_stub()
    import torch
    import trainer as trainer_module

    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("GDSS worker was configured for GPU but CUDA is unavailable.")

    config = _load_config(config_path)
    config.seed = int(args.seed)

    def graph_er_load_data(_config, get_graph_list=False):
        if get_graph_list:
            raise RuntimeError("GraphER GDSS training worker does not expose graph-list loading.")
        return (
            _build_loader(dataset_dir / "train.npz", _config, domain=args.domain, shuffle=True),
            _build_loader(dataset_dir / "val.npz", _config, domain=args.domain, shuffle=True),
        )

    # trainer.py imports load_data into module scope, so patch exactly that
    # boundary while leaving GDSS's models, losses, optimizers, EMA and SDE
    # objectives unchanged.
    trainer_module.load_data = graph_er_load_data

    runtime_dir = checkpoint.parent / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    previous_cwd = Path.cwd()
    try:
        os.chdir(runtime_dir)
        ts = "grapher_final"
        trainer = trainer_module.Trainer(config)
        returned = trainer.train(ts)
        native_checkpoint = runtime_dir / "checkpoints" / str(config.data.data) / f"{returned}.pth"
        if not native_checkpoint.is_file():
            raise RuntimeError(f"GDSS trainer did not write final checkpoint {native_checkpoint}.")
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        temporary = checkpoint.with_name(checkpoint.name + ".tmp")
        shutil.copy2(native_checkpoint, temporary)
        temporary.replace(checkpoint)
    finally:
        os.chdir(previous_cwd)
    shutil.rmtree(runtime_dir, ignore_errors=True)

    if not checkpoint.is_file():
        raise RuntimeError(f"GDSS worker did not publish checkpoint {checkpoint}.")
    _atomic_json(
        manifest,
        {
            "format": "grapher_gdss_training_worker_v1",
            "configured_num_epochs": int(config.train.num_epochs),
            "configured_batch_size": int(config.data.batch_size),
            "optimizer_split": "train",
            "monitor_split": "val",
            "test_used_during_training": False,
            "device": "cuda" if torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" else "cpu",
            "checkpoint": {"path": str(checkpoint), "sha256": _sha256(checkpoint)},
        },
    )
    print(
        f"[GraphER/GDSS] finished training for {int(config.train.num_epochs)} epochs",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

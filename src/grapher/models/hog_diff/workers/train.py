#!/usr/bin/env python
"""Isolated HOG-Diff training worker for one of the two score-model stages."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

try:
    from ._compat import install_torch_functional_alias
except ImportError:  # Direct worker execution uses this file's directory.
    from _compat import install_torch_functional_alias


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hogdiff-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("higher-order", "OU"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--molecular-npz", type=Path, default=None)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--require-cuda", action="store_true")
    return parser


def _load_mode_config(path: Path, *, mode: str):
    import yaml
    from easydict import EasyDict as edict

    raw = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.FullLoader)
    if not isinstance(raw, dict):
        raise TypeError(f"HOG-Diff resolved config must be a mapping: {path}")
    if mode == "higher-order":
        selected = {key: value for key, value in raw.items() if not str(key).startswith("OU")}
    else:
        selected = dict(raw)
        selected.update(
            {
                str(key)[2:]: value
                for key, value in raw.items()
                if str(key).startswith("OU")
            }
        )
    config = edict(selected)
    edge_th = config.model.get("edge_th")
    if isinstance(edge_th, str):
        config.model.edge_th = float(Fraction(edge_th))
    if int(config.sampling.n_steps) <= 0:
        config.sampling.corrector = "None"
    return config


def _materialize_molecular_tensor(npz_path: Path, *, data_root: Path, native_id: str) -> Path:
    import numpy as np
    import torch

    with np.load(npz_path, allow_pickle=False) as payload:
        x = np.asarray(payload["x"], dtype=np.float32)
        adjacency = np.asarray(payload["adjacency"], dtype=np.float32)
    processed = data_root / native_id / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    target = processed / "atom_bond.pt"
    torch.save([torch.from_numpy(x), torch.from_numpy(adjacency)], target)
    return target


def main() -> int:
    args = _parser().parse_args()
    root = args.hogdiff_root.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    manifest = args.manifest.expanduser().resolve()
    if not root.is_dir() or not config_path.is_file():
        raise FileNotFoundError(root if not root.is_dir() else config_path)

    # Prefer a real upstream data.py when present; otherwise Python falls back
    # to this worker directory, which contains only the missing constants.
    sys.path.insert(0, str(root))
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("WANDB_SILENT", "true")

    import torch
    import torch.nn.functional as torch_functional
    import wandb
    from models import layers as hog_layers

    functional_alias_installed = install_torch_functional_alias(
        hog_layers,
        torch_functional,
    )
    from utils import file_utils, loader
    from utils.logger import Logger
    from trainer import Trainer

    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("HOG-Diff worker was configured for GPU but CUDA is unavailable.")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative.")

    config = _load_mode_config(config_path, mode=args.mode)
    config.training.seed = int(args.seed)
    config.eval.seed = int(args.seed)
    config.training.snapshot_sampling = False
    config.data.num_workers = int(args.num_workers)
    config.exp.plot = False
    if args.resume_from is not None:
        resume = args.resume_from.expanduser().resolve()
        if not resume.is_file():
            raise FileNotFoundError(resume)
        config.ckpt_meta = str(resume)
    else:
        config.ckpt_meta = None

    data_root_raw = os.environ.get("DATA_ROOT")
    if not data_root_raw:
        raise RuntimeError("DATA_ROOT must be set by the GraphER HOG-Diff wrapper.")
    materialized = None
    if args.molecular_npz is not None:
        molecular_npz = args.molecular_npz.expanduser().resolve()
        if not molecular_npz.is_file():
            raise FileNotFoundError(molecular_npz)
        materialized = _materialize_molecular_tensor(
            molecular_npz,
            data_root=Path(data_root_raw).expanduser().resolve(),
            native_id=str(config.data.name),
        )

    loader.load_seed(int(args.seed))
    loader.init_exp(config)
    wandb.init(project="GraphER-HOG-Diff", mode="disabled", config={})
    logger = Logger(config, is_train=True, show_exc=True)
    trainer = Trainer(config, logger, mode=args.mode)
    initial_step = int(trainer.initial_step)
    trainer.train()

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    file_utils.save_checkpoint(
        str(checkpoint), trainer.state, trainer.config, trainer.device, args.mode
    )
    wandb.finish()
    if not checkpoint.is_file():
        raise RuntimeError(f"HOG-Diff worker did not write checkpoint {checkpoint}.")

    _atomic_json(
        manifest,
        {
            "format": "grapher_hogdiff_training_stage_v1",
            "mode": args.mode,
            "native_dataset": str(config.data.name),
            "seed": int(args.seed),
            "configured_n_iters": int(config.training.n_iters),
            "configured_batch_size": int(config.training.batch_size),
            "initial_step": initial_step,
            "checkpoint_step": int(trainer.state.get("step", -1)),
            "device": str(trainer.device),
            "checkpoint": {
                "path": str(checkpoint),
                "sha256": _sha256(checkpoint),
            },
            "molecular_tensor": None if materialized is None else str(materialized),
            "compatibility_shim": {
                "missing_data_module": (
                    "workers/data.py when upstream data.py is absent"
                ),
                "models_layers_functional_alias_installed": (
                    functional_alias_installed
                ),
            },
        },
    )
    print(
        f"[GraphER/HOG-Diff] finished {args.mode} stage at step {trainer.state.get('step', -1)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

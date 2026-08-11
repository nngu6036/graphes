#!/usr/bin/env python
"""Run DeFoG sampling in its own interpreter and emit a neutral NPZ export.

This script intentionally imports no GraphER modules.  It is launched by
``grapher.generators.defog`` with a child ``PYTHONPATH`` containing only the
DeFoG source roots, so DeFoG's legacy top-level imports cannot collide with
GraphER packages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import platform
import random
import subprocess
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

EXPORT_FORMAT = "defog_generic_topology_v1"
SUPPORTED_DATASETS = frozenset({"comm20", "planar", "sbm", "tree"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_revision(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
            timeout=5,
            shell=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    revision = result.stdout.strip()
    return revision if result.returncode == 0 and revision else None


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _integer_array(value: Any, *, name: str) -> np.ndarray:
    array = _to_numpy(value)
    if not np.issubdtype(array.dtype, np.number) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite numeric values.")
    rounded = np.rint(array)
    if not np.array_equal(array, rounded):
        raise ValueError(f"{name} must contain integral class labels.")
    return rounded.astype(np.int64, copy=False)


def encode_generic_samples(samples: Sequence[Any]) -> dict[str, np.ndarray]:
    """Validate raw DeFoG ``[X, E]`` samples and encode ragged graph arrays."""

    if not samples:
        raise ValueError("DeFoG returned no samples.")
    node_ptr = [0]
    edge_ptr = [0]
    all_node_labels: list[np.ndarray] = []
    all_endpoints: list[np.ndarray] = []
    all_edge_labels: list[np.ndarray] = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, (list, tuple)) or len(sample) != 2:
            raise TypeError(f"DeFoG sample {index} must be a two-item [X, E] pair.")
        node_labels = _integer_array(sample[0], name=f"sample[{index}].X")
        edge_labels = _integer_array(sample[1], name=f"sample[{index}].E")
        if node_labels.ndim != 1 or node_labels.size <= 0:
            raise ValueError(f"DeFoG sample {index} X must have shape [N], N > 0.")
        n = int(node_labels.size)
        if edge_labels.shape != (n, n):
            raise ValueError(
                f"DeFoG sample {index} E has shape {edge_labels.shape}, "
                f"expected {(n, n)}."
            )
        if np.any(node_labels != 0):
            raise ValueError(
                f"Generic DeFoG sample {index} contains multiple node classes."
            )
        if not np.all(np.isin(edge_labels, (0, 1))):
            raise ValueError(
                f"Generic DeFoG sample {index} contains an edge class outside 0/1."
            )
        if not np.array_equal(edge_labels, edge_labels.T):
            raise ValueError(f"DeFoG sample {index} adjacency is not symmetric.")
        if np.any(np.diag(edge_labels) != 0):
            raise ValueError(f"DeFoG sample {index} contains a self-loop class.")

        endpoints = np.argwhere(np.triu(edge_labels == 1, k=1)).astype(
            np.int64, copy=False
        )
        present_labels = np.ones(endpoints.shape[0], dtype=np.int64)
        all_node_labels.append(node_labels)
        all_endpoints.append(endpoints.reshape(-1, 2))
        all_edge_labels.append(present_labels)
        node_ptr.append(node_ptr[-1] + n)
        edge_ptr.append(edge_ptr[-1] + int(endpoints.shape[0]))

    return {
        "node_ptr": np.asarray(node_ptr, dtype=np.int64),
        "node_labels": np.concatenate(all_node_labels).astype(np.int64, copy=False),
        "edge_ptr": np.asarray(edge_ptr, dtype=np.int64),
        "edge_endpoints": (
            np.concatenate(all_endpoints, axis=0).astype(np.int64, copy=False)
            if any(array.size for array in all_endpoints)
            else np.empty((0, 2), dtype=np.int64)
        ),
        "edge_labels": (
            np.concatenate(all_edge_labels).astype(np.int64, copy=False)
            if any(array.size for array in all_edge_labels)
            else np.empty((0,), dtype=np.int64)
        ),
        "raw_indices": np.arange(len(samples), dtype=np.int64),
    }


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_raw_pickle(path: Path) -> list[Any]:
    # A DeFoG pickle is a Torch/Pickle artifact and can execute code while
    # loading.  The parent accepts it only through an explicitly configured
    # generated_path; callers must treat it like a model checkpoint.
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, (list, tuple)):
        raise TypeError("The DeFoG pickle must contain a sample list.")
    return list(value)


def _compose_config(args: argparse.Namespace) -> Any:
    from hydra import compose, initialize_config_dir
    from omegaconf import open_dict

    with initialize_config_dir(
        version_base="1.3",
        config_dir=str(args.defog_root / "configs"),
    ):
        cfg = compose(
            config_name="config",
            overrides=[
                f"+experiment={args.experiment}",
                f"dataset={args.dataset}",
            ],
        )
    with open_dict(cfg):
        cfg.train.seed = int(args.seed)
        cfg.train.batch_size = int(args.batch_size)
        cfg.train.num_workers = 0
        cfg.general.wandb = "disabled"
        cfg.general.gpus = 1 if str(args.device).startswith("cuda") else 0
        cfg.general.conditional = False
        cfg.sample.sample_steps = int(args.sample_steps)
        cfg.sample.eta = float(args.eta)
        cfg.sample.omega = float(args.omega)
        cfg.sample.time_distortion = str(args.time_distortion)
        cfg.sample.rdb = str(args.rdb)
        cfg.sample.rdb_crit = str(args.rdb_crit)
        cfg.sample.search = False
    return cfg


def _resolve_device(requested: str, torch: Any) -> Any:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"DeFoG device {requested!r} requested without CUDA.")
    return torch.device(requested)


def _generate_samples(args: argparse.Namespace) -> tuple[list[Any], dict[str, str]]:
    import pytorch_lightning as pl
    import torch
    from datasets.spectre_dataset import (
        SpectreDatasetInfos,
        SpectreGraphDataModule,
    )
    from graph_discrete_flow_model import GraphDiscreteFlowModel
    from models.extra_features import DummyExtraFeatures, ExtraFeatures

    cfg = _compose_config(args)
    pl.seed_everything(args.seed, workers=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset_name = str(cfg.dataset.name).lower()
    if dataset_name != args.dataset or dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Resolved DeFoG dataset {dataset_name!r} does not match "
            f"the requested generic dataset {args.dataset!r}."
        )

    datamodule = SpectreGraphDataModule(cfg)
    dataset_infos = SpectreDatasetInfos(datamodule, cfg.dataset)
    extra_features = ExtraFeatures(
        cfg.model.extra_features,
        cfg.model.rrwp_steps,
        dataset_info=dataset_infos,
    )
    domain_features = DummyExtraFeatures()
    dataset_infos.compute_input_output_dims(
        datamodule=datamodule,
        extra_features=extra_features,
        domain_features=domain_features,
    )
    model = GraphDiscreteFlowModel(
        cfg=cfg,
        dataset_infos=dataset_infos,
        train_metrics=None,
        sampling_metrics=None,
        visualization_tools=None,
        extra_features=extra_features,
        domain_features=domain_features,
        test_labels=None,
    )

    try:
        payload = torch.load(
            args.checkpoint,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        payload = torch.load(args.checkpoint, map_location="cpu")
    state_dict = (
        payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    )
    if not isinstance(state_dict, dict):
        raise TypeError("The DeFoG checkpoint has no state_dict mapping.")
    model.load_state_dict(state_dict, strict=True)
    device = _resolve_device(args.device, torch)
    model.to(device)
    model.limit_dist.to_device(device)
    model.eval()

    samples: list[Any] = []
    offset = 0
    while offset < args.num_samples:
        current_batch = min(args.batch_size, args.num_samples - offset)
        batch_samples, _ = model.sample_batch(
            batch_id=offset,
            batch_size=current_batch,
            keep_chain=0,
            number_chain_steps=1,
            save_final=0,
            num_nodes=None,
            save_visualization=False,
        )
        if len(batch_samples) != current_batch:
            raise RuntimeError(
                f"DeFoG returned {len(batch_samples)} samples for a "
                f"batch of {current_batch}."
            )
        samples.extend(batch_samples)
        offset += current_batch
    versions = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "pytorch_lightning": str(pl.__version__),
        "cuda": str(torch.version.cuda),
    }
    return samples, versions


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate/export generic DeFoG samples for GraphER."
    )
    parser.add_argument("--defog-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset", choices=sorted(SUPPORTED_DATASETS), required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--input-pickle", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sample-steps", type=int, default=1000)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--omega", type=float, default=0.0)
    parser.add_argument("--time-distortion", default="identity")
    parser.add_argument("--rdb", default="general")
    parser.add_argument("--rdb-crit", default="dummy")
    return parser


def main() -> None:
    args = _parser().parse_args()
    args.defog_root = args.defog_root.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.manifest = args.manifest.expanduser().resolve()
    if args.num_samples <= 0 or args.batch_size <= 0 or args.sample_steps <= 0:
        raise ValueError("num-samples, batch-size, and sample-steps must be positive.")
    if args.input_pickle is not None and args.checkpoint is not None:
        raise ValueError("Use either --input-pickle or --checkpoint, not both.")
    if args.input_pickle is None and args.checkpoint is None:
        raise ValueError("--checkpoint is required for DeFoG generation.")
    if not (args.defog_root / "src" / "main.py").is_file():
        raise FileNotFoundError(f"Invalid DeFoG root: {args.defog_root}")

    if args.input_pickle is not None:
        args.input_pickle = args.input_pickle.expanduser().resolve()
        samples = _load_raw_pickle(args.input_pickle)
        versions = {
            "python": platform.python_version(),
            "mode": "trusted_pickle_conversion",
        }
        source = {"kind": "trusted_pickle", "path": str(args.input_pickle)}
        checkpoint_info = None
        sampling_info: dict[str, Any] = {
            "status": "unknown_from_pickle",
            "device": None,
        }
    else:
        args.checkpoint = args.checkpoint.expanduser().resolve()
        if not args.checkpoint.is_file():
            raise FileNotFoundError(f"Missing DeFoG checkpoint: {args.checkpoint}")
        samples, versions = _generate_samples(args)
        source = {"kind": "checkpoint_generation"}
        checkpoint_info = {
            "path": str(args.checkpoint),
            "sha256": _sha256(args.checkpoint),
        }
        sampling_info = {
            "batch_size": int(args.batch_size),
            "sample_steps": int(args.sample_steps),
            "eta": float(args.eta),
            "omega": float(args.omega),
            "time_distortion": str(args.time_distortion),
            "rdb": str(args.rdb),
            "rdb_crit": str(args.rdb_crit),
            "device": str(args.device),
        }
    if len(samples) != args.num_samples:
        raise ValueError(
            f"DeFoG supplied {len(samples)} samples, expected {args.num_samples}."
        )

    arrays = encode_generic_samples(samples)
    _atomic_npz(args.output, arrays)
    manifest = {
        "format": EXPORT_FORMAT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "experiment": args.experiment,
        "seed": int(args.seed),
        "requested_samples": int(args.num_samples),
        "exported_samples": len(samples),
        "sampling": sampling_info,
        "source": source,
        "checkpoint": checkpoint_info,
        "defog": {
            "root": str(args.defog_root),
            "revision": _source_revision(args.defog_root),
        },
        "runtime": versions,
        "export": {
            "path": str(args.output),
            "sha256": _sha256(args.output),
            "allow_pickle": False,
        },
    }
    _atomic_json(args.manifest, manifest)
    print(f"Exported {len(samples)} generic DeFoG graphs to {args.output}")


if __name__ == "__main__":
    main()

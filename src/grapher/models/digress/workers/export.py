#!/usr/bin/env python
"""Export an exact DiGress sample batch as neutral numeric arrays."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

from common import (
    atomic_json,
    build_components,
    install_discrete_model_runtime_patches,
    install_upstream_runtime_patches,
    load_json,
    seed_everything,
    status,
)

FORMAT = "grapher_digress_export_v1"
ARRAY_FORMAT = "digress_graph_batch_v1"


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return value


def _nonnegative_int(raw: str) -> int:
    value = int(raw)
    if value < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate an exact neutral DiGress graph batch."
    )
    parser.add_argument("--digress-root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-datadir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--molecular-statistics", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--num-graphs", type=_positive_int, required=True)
    parser.add_argument("--batch-size", type=_positive_int, required=True)
    parser.add_argument("--seed", type=_nonnegative_int, required=True)
    parser.add_argument("--progress-every-batches", type=_positive_int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"), default="auto")
    return parser


def _load_config(path: Path, *, dataset_datadir: Path, dataset: str) -> Any:
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(str(path))
    cfg.dataset.name = str(dataset).lower()
    cfg.dataset.datadir = str(dataset_datadir.resolve())
    cfg.general.wandb = "disabled"
    cfg.general.gpus = 0
    cfg.general.samples_to_generate = 0
    cfg.general.samples_to_save = 0
    cfg.general.chains_to_save = 0
    cfg.general.final_model_samples_to_generate = 0
    cfg.general.final_model_samples_to_save = 0
    cfg.general.final_model_chains_to_save = 0
    OmegaConf.resolve(cfg)
    return cfg


def _device(requested: str) -> Any:
    import torch

    value = str(requested).lower()
    if value == "cpu":
        return torch.device("cpu")
    if value == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("--device gpu requested but CUDA is unavailable.")
        return torch.device("cuda:0")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _load_state(model: Any, checkpoint: Path, device: Any) -> None:
    import torch

    try:
        payload = torch.load(str(checkpoint), map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(str(checkpoint), map_location=device)
    if not isinstance(payload, dict):
        raise TypeError("DiGress checkpoint must contain a mapping.")
    state = payload.get("state_dict", payload)
    if not isinstance(state, dict):
        raise TypeError("DiGress checkpoint has no state_dict mapping.")
    incompatible = model.load_state_dict(state, strict=True)
    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))
    if missing or unexpected:
        raise RuntimeError(
            "DiGress checkpoint/model mismatch: missing=%s, unexpected=%s"
            % (missing, unexpected)
        )


def _pack_samples(samples: Sequence[Any]) -> Tuple[Any, Any, Any, Any, Any]:
    import numpy as np
    import torch

    node_offsets: List[int] = [0]
    node_types: List[int] = []
    edge_offsets: List[int] = [0]
    endpoints: List[Tuple[int, int]] = []
    edge_types: List[int] = []

    for graph_index, sample in enumerate(samples):
        if not isinstance(sample, (list, tuple)) or len(sample) < 2:
            raise TypeError(
                "DiGress sample %d is not [node_types, edge_types]." % graph_index
            )
        x = torch.as_tensor(sample[0]).detach().cpu().long()
        e = torch.as_tensor(sample[1]).detach().cpu().long()
        if x.ndim != 1:
            raise ValueError(
                "DiGress sample %d node tensor has shape %s." % (
                    graph_index,
                    tuple(x.shape),
                )
            )
        n = int(x.shape[0])
        if n <= 0 or e.shape != (n, n):
            raise ValueError(
                "DiGress sample %d has incompatible node/edge shapes %s/%s."
                % (graph_index, tuple(x.shape), tuple(e.shape))
            )
        if not torch.equal(e, e.transpose(0, 1)):
            raise ValueError("DiGress sample %d is not undirected." % graph_index)
        if torch.any(torch.diag(e) != 0):
            raise ValueError("DiGress sample %d contains a self-loop." % graph_index)

        node_types.extend(int(value) for value in x.tolist())
        node_offsets.append(len(node_types))
        for source in range(n):
            for target in range(source + 1, n):
                edge_type = int(e[source, target].item())
                if edge_type <= 0:
                    continue
                endpoints.append((source, target))
                edge_types.append(edge_type)
        edge_offsets.append(len(edge_types))

    endpoint_array = np.asarray(endpoints, dtype=np.int64)
    if endpoint_array.size == 0:
        endpoint_array = np.empty((0, 2), dtype=np.int64)
    else:
        endpoint_array = endpoint_array.reshape((-1, 2))
    return (
        np.asarray(node_offsets, dtype=np.int64),
        np.asarray(node_types, dtype=np.int64),
        np.asarray(edge_offsets, dtype=np.int64),
        endpoint_array,
        np.asarray(edge_types, dtype=np.int64),
    )


def _atomic_npz(
    path: Path,
    *,
    node_offsets: Any,
    node_types: Any,
    edge_offsets: Any,
    edge_endpoints: Any,
    edge_types: Any,
) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            node_offsets=node_offsets,
            node_types=node_types,
            edge_offsets=edge_offsets,
            edge_endpoints=edge_endpoints,
            edge_types=edge_types,
        )
    temporary.replace(path)


def main() -> None:
    args = _parser().parse_args()
    started_at = datetime.now(timezone.utc)
    started = time.monotonic()
    root = args.digress_root.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    dataset_datadir = args.dataset_datadir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    for required in (config_path, checkpoint):
        if not required.is_file():
            raise FileNotFoundError(required)
    if not dataset_datadir.is_dir():
        raise FileNotFoundError(dataset_datadir)

    seed_everything(args.seed)
    cfg = _load_config(
        config_path,
        dataset_datadir=dataset_datadir,
        dataset=args.dataset,
    )
    statistics = None
    if args.molecular_statistics is not None:
        stats_path = args.molecular_statistics.expanduser().resolve()
        if not stats_path.is_file():
            raise FileNotFoundError(stats_path)
        statistics = load_json(stats_path)

    status(
        "Generation worker initializing: "
        "dataset=%s, requested=%d, batch_size=%d, diffusion_steps=%d."
        % (
            args.dataset,
            args.num_graphs,
            args.batch_size,
            int(cfg.model.diffusion_steps),
        )
    )
    install_upstream_runtime_patches()
    _, model_kwargs, _ = build_components(
        cfg, molecular_statistics=statistics
    )
    from diffusion_model_discrete import DiscreteDenoisingDiffusion
    import torch
    import torch_geometric
    import pytorch_lightning as pl

    install_discrete_model_runtime_patches(DiscreteDenoisingDiffusion)
    model = DiscreteDenoisingDiffusion(cfg=cfg, **model_kwargs)
    device = _device(args.device)
    _load_state(model, checkpoint, device)
    model.to(device)
    model.eval()

    samples: List[Any] = []
    total_batches = (args.num_graphs + args.batch_size - 1) // args.batch_size
    with torch.inference_mode():
        for batch_index in range(total_batches):
            count = min(args.batch_size, args.num_graphs - len(samples))
            batch = model.sample_batch(
                batch_id=len(samples),
                batch_size=count,
                keep_chain=0,
                number_chain_steps=1,
                save_final=0,
                num_nodes=None,
            )
            if len(batch) != count:
                raise RuntimeError(
                    "DiGress sample_batch returned %d graphs, expected %d."
                    % (len(batch), count)
                )
            samples.extend(batch)
            completed = batch_index + 1
            if (
                completed == 1
                or completed == total_batches
                or completed % args.progress_every_batches == 0
            ):
                status(
                    "Generation progress: batch=%d/%d, generated=%d/%d."
                    % (completed, total_batches, len(samples), args.num_graphs)
                )
    if len(samples) != args.num_graphs:
        raise RuntimeError(
            "DiGress generation count mismatch: requested %d, got %d."
            % (args.num_graphs, len(samples))
        )

    packed = _pack_samples(samples)
    _atomic_npz(
        output,
        node_offsets=packed[0],
        node_types=packed[1],
        edge_offsets=packed[2],
        edge_endpoints=packed[3],
        edge_types=packed[4],
    )
    finished_at = datetime.now(timezone.utc)
    manifest = {
        "format": FORMAT,
        "array_format": ARRAY_FORMAT,
        "status": "complete",
        "dataset": str(args.dataset).lower(),
        "num_requested": int(args.num_graphs),
        "num_generated": int(len(samples)),
        "generation_seed": int(args.seed),
        "batch_size": int(args.batch_size),
        "diffusion_steps": int(cfg.model.diffusion_steps),
        "device": str(device),
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": _sha256(checkpoint),
        },
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "output": {"path": str(output), "sha256": _sha256(output)},
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": time.monotonic() - started,
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "torch_geometric": str(torch_geometric.__version__),
            "pytorch_lightning": str(pl.__version__),
        },
    }
    atomic_json(manifest_path, manifest)
    status(
        "Generation worker completed: generated=%d, output=%s."
        % (len(samples), output)
    )


if __name__ == "__main__":
    main()

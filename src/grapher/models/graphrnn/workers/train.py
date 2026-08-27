#!/usr/bin/env python
"""Train GraphRNN from a neutral GraphES dataset export."""

from __future__ import annotations

import argparse
import hashlib
import platform
import sys
from pathlib import Path

import numpy as np
import torch

from common import (
    atomic_json,
    load_dataset_arrays,
    load_json,
    resolve_device,
    train_model,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphrnn-root", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume-from", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = Path(args.graphrnn_root).expanduser().resolve()
    dataset_path = Path(args.dataset).expanduser().resolve()
    dataset_manifest_path = Path(args.dataset_manifest).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    for required in (root / "model.py", dataset_path, dataset_manifest_path, config_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    dataset_manifest = load_json(dataset_manifest_path)
    if dataset_manifest.get("format") != "grapher_graphrnn_dataset_v1":
        raise RuntimeError("Unsupported GraphRNN dataset export manifest.")
    expected_hash = str((dataset_manifest.get("output") or {}).get("sha256", ""))
    observed_hash = _sha256(dataset_path)
    if not expected_hash or observed_hash != expected_hash:
        raise RuntimeError("GraphRNN dataset export SHA-256 mismatch.")

    config = load_json(config_path)
    arrays = load_dataset_arrays(dataset_path)
    device = resolve_device(args.device)
    print(
        "GraphRNN compatibility worker: "
        f"python={platform.python_version()} torch={torch.__version__} "
        f"numpy={np.__version__} device={device}",
        flush=True,
    )
    result = train_model(
        graphrnn_root=root,
        dataset=arrays,
        config=config,
        output_dir=output_dir,
        seed=args.seed,
        device=device,
        dataset_sha256=observed_hash,
        resume_from=args.resume_from,
    )
    result.update(
        {
            "dataset_export": {
                "path": str(dataset_path),
                "sha256": observed_hash,
                "manifest": str(dataset_manifest_path),
            },
            "python_executable": str(Path(sys.executable).resolve()),
            "python_version": platform.python_version(),
            "torch_version": str(torch.__version__),
            "numpy_version": str(np.__version__),
        }
    )
    atomic_json(manifest_path, result)
    print(f"GraphRNN training manifest: {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

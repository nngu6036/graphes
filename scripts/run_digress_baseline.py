#!/usr/bin/env python
"""Train DiGress on a prepared GraphER dataset and generate one raw batch."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from grapher.models import (
    DatasetReference,
    GenerateRequest,
    RunSpec,
    TrainRequest,
    create_baseline,
)


@dataclass(frozen=True)
class DatasetProfile:
    serialized_id: str
    native_id: str
    experiment: str


DATASET_PROFILES: dict[str, DatasetProfile] = {
    "community_small": DatasetProfile("sbm", "comm20", "comm20"),
    "ego_small": DatasetProfile("ego_small", "comm20", "comm20"),
    "grid": DatasetProfile("grid", "planar", "planar"),
    "qm9": DatasetProfile("qm9_attributed", "qm9", "qm9_no_h"),
}


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


def _positive_float(raw: str) -> float:
    value = float(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _status(message: str, *, enabled: bool = True) -> None:
    if not enabled:
        return
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(
        f"[run_digress_baseline {timestamp}] {message}",
        file=sys.stderr,
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train the DiGress baseline through its isolated GraphER wrapper, "
            "then generate and serialize an exact raw graph batch."
        )
    )
    parser.add_argument(
        "--dataset", required=True, choices=tuple(DATASET_PROFILES)
    )
    parser.add_argument("--num-samples", required=True, type=_positive_int)
    parser.add_argument(
        "--seed-id", "--seed", dest="seed_id", required=True, type=_nonnegative_int
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("outputs/datasets")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/baselines")
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--generation-id", default=None)
    parser.add_argument("--wrapper-config", type=Path, default=None)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--digress-root", type=Path, default=None)
    parser.add_argument("--digress-python", type=Path, default=None)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help=(
            "Execution device for both training and generation. Use 'gpu' to "
            "require CUDA and fail immediately instead of silently falling "
            "back to CPU."
        ),
    )
    parser.add_argument(
        "--gpu-id",
        type=_nonnegative_int,
        default=None,
        help=(
            "Physical CUDA device index exposed to DiGress when --device gpu "
            "is used. Defaults to 0."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--n-epochs",
        type=_positive_int,
        default=None,
        help=(
            "Total DiGress training horizon. When resuming, this is the total "
            "horizon rather than the number of additional epochs."
        ),
    )
    parser.add_argument("--batch-size", type=_positive_int, default=None)
    parser.add_argument(
        "--generation-batch-size", type=_positive_int, default=None
    )
    parser.add_argument("--num-workers", type=_nonnegative_int, default=None)
    parser.add_argument(
        "--check-val-every-n-epochs", type=_positive_int, default=None
    )
    parser.add_argument("--save-every-n-epochs", type=_positive_int, default=None)
    parser.add_argument(
        "--skip-training-estimates",
        action="store_true",
        help="Do not generate the independent post-training source pool.",
    )
    parser.add_argument(
        "--training-estimate-count", type=_positive_int, default=None
    )
    parser.add_argument(
        "--progress-interval-seconds", type=_positive_float, default=15.0
    )
    parser.add_argument("--epoch-progress-interval", type=_positive_int, default=None)
    parser.add_argument(
        "--generation-progress-every-batches", type=_positive_int, default=1
    )
    parser.add_argument("--no-stream-subprocess-output", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def _configure_environment(args: argparse.Namespace) -> None:
    if args.digress_root is not None:
        os.environ["DIGRESS"] = str(args.digress_root.expanduser().resolve())
    if args.digress_python is not None:
        os.environ["DIGRESS_PYTHON"] = str(
            args.digress_python.expanduser().resolve()
        )


def run_pipeline(args: argparse.Namespace) -> dict[str, object]:
    _configure_environment(args)
    progress_enabled = not args.quiet
    profile = DATASET_PROFILES[args.dataset]
    progress: dict[str, object] = {
        "enabled": progress_enabled,
        "stream_output": (
            progress_enabled and not args.no_stream_subprocess_output
        ),
        "interval_seconds": args.progress_interval_seconds,
        "generation_batch_interval": args.generation_progress_every_batches,
    }
    if args.epoch_progress_interval is not None:
        progress["epoch_interval"] = args.epoch_progress_interval

    training_estimates: dict[str, object] = {
        "enabled": not args.skip_training_estimates
    }
    if args.training_estimate_count is not None:
        training_estimates["num_graphs"] = args.training_estimate_count
    requested_device = str(getattr(args, "device", "auto")).lower()
    gpu_id = getattr(args, "gpu_id", None)
    runtime_options: dict[str, object] = {"progress": progress}
    if requested_device == "gpu":
        resolved_gpu_id = 0 if gpu_id is None else int(gpu_id)
        runtime_options.update(
            {
                "gpus": 1,
                "device": "gpu",
                "cuda_visible_devices": str(resolved_gpu_id),
                "require_cuda": True,
            }
        )
    elif requested_device == "cpu":
        runtime_options.update({"gpus": 0, "device": "cpu"})
    elif gpu_id is not None:
        raise ValueError("--gpu-id requires --device gpu.")

    training_options: dict[str, object] = {
        "experiment": profile.experiment,
        "training_estimates": training_estimates,
        "runtime": runtime_options,
    }
    for key in (
        "n_epochs",
        "batch_size",
        "num_workers",
        "check_val_every_n_epochs",
        "save_every_n_epochs",
        "generation_batch_size",
    ):
        value = getattr(args, key)
        if value is not None:
            training_options[key] = value

    run = RunSpec.for_seed(
        model_id="digress",
        dataset_id=args.dataset,
        seed=args.seed_id,
        run_id=args.run_id,
        output_root=args.output_root,
    )
    dataset = DatasetReference(
        benchmark_id=args.dataset,
        root=args.dataset_root,
        serialized_id=profile.serialized_id,
        native_id=profile.native_id,
    )
    _status(
        "resolved run: "
        f"dataset={args.dataset}, serialized={profile.serialized_id}, "
        f"native={profile.native_id}, experiment={profile.experiment}, "
        f"run_id={run.run_id}, seed={args.seed_id}",
        enabled=progress_enabled,
    )
    dataset.require_prepared()
    wrapper = create_baseline("digress")
    _status(
        "prepared dataset found; starting DiGress training. Durable log will be "
        f"published at {run.layout.training_log_path.resolve()}",
        enabled=progress_enabled,
    )
    training = wrapper.train(
        TrainRequest(
            run=run,
            dataset=dataset,
            config_path=args.wrapper_config,
            options=training_options,
            resume_from=args.resume_from,
            overwrite=args.overwrite,
        )
    )
    _status(
        f"training completed: checkpoint={training.checkpoint_path.resolve()}",
        enabled=progress_enabled,
    )
    generation_options: dict[str, object] = {"runtime": dict(runtime_options)}
    if args.generation_batch_size is not None:
        generation_options["generation_batch_size"] = args.generation_batch_size
    generation = wrapper.generate(
        GenerateRequest(
            run=run,
            checkpoint_path=training.checkpoint_path,
            num_graphs=args.num_samples,
            generation_seed=args.seed_id,
            generation_id=args.generation_id,
            options=generation_options,
            overwrite=args.overwrite,
        )
    )
    if generation.num_generated != args.num_samples:
        raise RuntimeError(
            f"DiGress generated {generation.num_generated}; expected "
            f"{args.num_samples}."
        )
    _status(
        f"generation completed: graphs={generation.graphs_path.resolve()}",
        enabled=progress_enabled,
    )
    return {
        "status": "complete",
        "model": "digress",
        "dataset": args.dataset,
        "serialized_dataset": profile.serialized_id,
        "native_dataset": profile.native_id,
        "experiment": profile.experiment,
        "seed_id": args.seed_id,
        "device": requested_device,
        "gpu_id": (0 if requested_device == "gpu" and gpu_id is None else gpu_id),
        "run_id": run.run_id,
        "generation_id": generation.generation_dir.name,
        "num_samples": generation.num_generated,
        "run_dir": str(training.run_dir.resolve()),
        "checkpoint": str(training.checkpoint_path.resolve()),
        "training_manifest": str(training.manifest_path.resolve()),
        "training_estimated_graphs": (
            str(training.estimated_graphs_path.resolve())
            if training.estimated_graphs_path is not None
            else None
        ),
        "training_ground_truth_graphs": (
            str(training.ground_truth_graphs_path.resolve())
            if training.ground_truth_graphs_path is not None
            else None
        ),
        "training_ground_truth_model_view": (
            str(training.ground_truth_model_view_graphs_path.resolve())
            if training.ground_truth_model_view_graphs_path is not None
            else None
        ),
        "generated_graphs": str(generation.graphs_path.resolve()),
        "generation_manifest": str(generation.manifest_path.resolve()),
        "graphs_sha256": generation.graphs_sha256,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run_pipeline(args)
    except Exception as exc:
        _status(
            f"FAILED: {type(exc).__name__}: {exc}", enabled=not args.quiet
        )
        raise
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

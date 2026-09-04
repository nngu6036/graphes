#!/usr/bin/env python
"""Train HOG-Diff on a prepared GraphER dataset and generate one raw batch."""

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


DATASET_PROFILES: dict[str, DatasetProfile] = {
    "community_small": DatasetProfile(serialized_id="sbm", native_id="community_small"),
    "ego_small": DatasetProfile(serialized_id="ego_small", native_id="ego_small"),
    "qm9": DatasetProfile(serialized_id="qm9_attributed", native_id="qm9"),
    "zinc": DatasetProfile(serialized_id="zinc", native_id="zinc250k"),
}


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return value


def _nonnegative_int(raw: str) -> int:
    value = int(raw)
    if value < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
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
    print(f"[run_hog_diff_baseline {timestamp}] {message}", file=sys.stderr, flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train the attached two-stage HOG-Diff baseline through GraphER's "
            "isolated wrapper and publish an exact raw graph batch."
        )
    )
    parser.add_argument("--dataset", required=True, choices=tuple(DATASET_PROFILES))
    parser.add_argument("--num-samples", required=True, type=_positive_int)
    parser.add_argument("--seed-id", "--seed", dest="seed_id", required=True, type=_nonnegative_int)
    parser.add_argument("--dataset-root", type=Path, default=Path("outputs/datasets"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/baselines"))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--generation-id", default=None)
    parser.add_argument("--wrapper-config", type=Path, default=None)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument(
        "--hogdiff-root",
        type=Path,
        default=None,
        help="HOG-Diff source root; otherwise use HOGDIFF.",
    )
    parser.add_argument(
        "--hogdiff-python",
        type=Path,
        default=None,
        help="Python executable with HOG-Diff dependencies; otherwise use HOGDIFF_PYTHON.",
    )
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument(
        "--ho-iters",
        type=_nonnegative_int,
        default=None,
        help="Override higher-order VPSDE training.n_iters.",
    )
    parser.add_argument(
        "--ou-iters",
        type=_nonnegative_int,
        default=None,
        help="Override conditional OU-bridge training.n_iters.",
    )
    parser.add_argument("--ho-batch-size", type=_positive_int, default=None)
    parser.add_argument("--ou-batch-size", type=_positive_int, default=None)
    parser.add_argument("--num-workers", type=_nonnegative_int, default=None)
    parser.add_argument("--generation-batch-size", type=_positive_int, default=None)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "gpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"),
        default=None,
    )
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--timeout-seconds", type=_positive_float, default=None)
    parser.add_argument("--skip-training-estimates", action="store_true")
    parser.add_argument("--training-estimate-count", type=_positive_int, default=None)
    parser.add_argument("--progress-interval-seconds", type=_positive_float, default=15.0)
    parser.add_argument("--iteration-progress-interval", type=_positive_int, default=None)
    parser.add_argument("--generation-progress-every-batches", type=_positive_int, default=1)
    parser.add_argument("--no-stream-subprocess-output", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def _configure_environment(args: argparse.Namespace) -> None:
    if args.hogdiff_root is not None:
        os.environ["HOGDIFF"] = str(args.hogdiff_root.expanduser().resolve())
    if args.hogdiff_python is not None:
        os.environ["HOGDIFF_PYTHON"] = str(args.hogdiff_python.expanduser().resolve())


def run_pipeline(args: argparse.Namespace) -> dict[str, object]:
    _configure_environment(args)
    progress_enabled = not args.quiet
    profile = DATASET_PROFILES[args.dataset]
    progress: dict[str, object] = {
        "enabled": progress_enabled,
        "stream_output": progress_enabled and not args.no_stream_subprocess_output,
        "interval_seconds": args.progress_interval_seconds,
        "generation_batch_interval": args.generation_progress_every_batches,
    }
    if args.iteration_progress_interval is not None:
        progress["iteration_interval"] = args.iteration_progress_interval

    runtime: dict[str, object] = {"progress": progress}
    for key in ("device", "cuda_visible_devices", "timeout_seconds"):
        value = getattr(args, key)
        if value is not None:
            runtime[key] = value

    training_estimates: dict[str, object] = {"enabled": not args.skip_training_estimates}
    if args.training_estimate_count is not None:
        training_estimates["num_graphs"] = args.training_estimate_count

    options: dict[str, object] = {
        "runtime": runtime,
        "training_estimates": training_estimates,
    }
    if args.num_workers is not None:
        options["num_workers"] = args.num_workers
    if args.generation_batch_size is not None:
        options["generation_batch_size"] = args.generation_batch_size
    higher_order: dict[str, object] = {}
    if args.ho_iters is not None:
        higher_order["n_iters"] = args.ho_iters
    if args.ho_batch_size is not None:
        higher_order["batch_size"] = args.ho_batch_size
    if higher_order:
        options["higher_order"] = higher_order
    ou: dict[str, object] = {}
    if args.ou_iters is not None:
        ou["n_iters"] = args.ou_iters
    if args.ou_batch_size is not None:
        ou["batch_size"] = args.ou_batch_size
    if ou:
        options["ou"] = ou

    run = RunSpec.for_seed(
        model_id="hog_diff",
        dataset_id=args.dataset,
        seed=args.seed_id,
        output_root=args.output_root,
        run_id=args.run_id,
    )
    dataset = DatasetReference(
        benchmark_id=args.dataset,
        root=args.dataset_root,
        serialized_id=profile.serialized_id,
        native_id=profile.native_id,
    )
    _status(
        f"resolved run: dataset={args.dataset}, serialized={profile.serialized_id}, "
        f"native={profile.native_id}, run_id={run.run_id}, seed={args.seed_id}",
        enabled=progress_enabled,
    )
    dataset.require_prepared()
    wrapper = create_baseline("hog_diff")
    _status(
        "prepared dataset found; starting higher-order then OU HOG-Diff training. "
        f"Durable log: {run.layout.training_log_path.resolve()}",
        enabled=progress_enabled,
    )
    training = wrapper.train(
        TrainRequest(
            run=run,
            dataset=dataset,
            config_path=args.wrapper_config,
            options=options,
            resume_from=args.resume_from,
            overwrite=args.overwrite,
        )
    )
    _status(
        f"training completed: checkpoint={training.checkpoint_path.resolve()}",
        enabled=progress_enabled,
    )

    generation_options: dict[str, object] = {"runtime": runtime}
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
            f"HOG-Diff generated {generation.num_generated}; expected {args.num_samples}."
        )
    _status(
        f"generation completed: graphs={generation.graphs_path.resolve()}",
        enabled=progress_enabled,
    )
    return {
        "status": "complete",
        "model": "hog_diff",
        "dataset": args.dataset,
        "serialized_dataset": profile.serialized_id,
        "native_dataset": profile.native_id,
        "seed_id": args.seed_id,
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
        "generated_graphs": str(generation.graphs_path.resolve()),
        "generation_manifest": str(generation.manifest_path.resolve()),
        "graphs_sha256": generation.graphs_sha256,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run_pipeline(args)
    except Exception as exc:
        _status(f"FAILED: {type(exc).__name__}: {exc}", enabled=not args.quiet)
        raise
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Train GraphRNN on a prepared GraphES dataset and generate one raw batch.

The script mirrors ``run_defog_baseline.py`` and deliberately remains a thin
orchestration layer. Dataset conversion, isolated execution of the attached
GraphRNN modules, checkpoint publication, exact-count generation, validation,
hashing, and manifests are handled by ``GraphRNNWrapper``.
"""

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
    """GraphES storage identity for a supported generic benchmark."""

    serialized_id: str
    native_id: str


DATASET_PROFILES: dict[str, DatasetProfile] = {
    "community_small": DatasetProfile(serialized_id="sbm", native_id="community_small"),
    "ego_small": DatasetProfile(serialized_id="ego_small", native_id="ego_small"),
    "grid": DatasetProfile(serialized_id="grid", native_id="grid"),
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
    """Write progress to stderr while preserving JSON-only stdout."""

    if not enabled:
        return
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(
        f"[run_graphrnn_baseline {timestamp}] {message}",
        file=sys.stderr,
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train the GraphRNN baseline through its isolated GraphES wrapper, "
            "then generate and serialize an exact raw graph batch."
        )
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=tuple(DATASET_PROFILES),
        help="Prepared generic GraphES benchmark to use.",
    )
    parser.add_argument(
        "--num-samples",
        required=True,
        type=_positive_int,
        help="Exact number of graphs to generate after training.",
    )
    parser.add_argument(
        "--seed-id",
        "--seed",
        dest="seed_id",
        required=True,
        type=_nonnegative_int,
        help="Seed used for training and the requested generation batch.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("outputs/datasets"),
        help="Root containing prepared train.pkl, val.pkl, and test.pkl files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/baselines"),
        help="Root for managed GraphRNN checkpoints and raw generations.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run identifier; defaults to seed_<seed-id>.",
    )
    parser.add_argument(
        "--generation-id",
        default=None,
        help=(
            "Optional generation identifier; defaults to "
            "seed_<seed-id>_n_<num-samples>."
        ),
    )
    parser.add_argument(
        "--wrapper-config",
        type=Path,
        default=None,
        help="Optional YAML file with a top-level graphrnn section.",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Optional managed GraphRNN checkpoint from which to resume.",
    )
    parser.add_argument(
        "--graphrnn-root",
        type=Path,
        default=None,
        help=(
            "Attached GraphRNN source root; otherwise use the GRAPHRNN "
            "environment variable."
        ),
    )
    parser.add_argument(
        "--graphrnn-python",
        type=Path,
        default=None,
        help=(
            "Python executable containing torch and numpy; otherwise use "
            "GRAPHRNN_PYTHON or the current interpreter."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Explicitly replace matching artifacts. A trained run with existing "
            "generation batches cannot be overwritten."
        ),
    )

    # High-value model/training overrides. All remaining settings stay in YAML.
    parser.add_argument(
        "--variant",
        choices=("GraphRNN_RNN", "GraphRNN_MLP"),
        default=None,
        help="Dependent-output-RNN (paper default) or independent-output MLP.",
    )
    parser.add_argument(
        "--n-epochs",
        "--epochs",
        dest="epochs",
        type=_positive_int,
        default=None,
        help=(
            "Total GraphRNN training horizon. When resuming, this is the total "
            "horizon rather than a number of additional epochs."
        ),
    )
    parser.add_argument("--batch-size", type=_positive_int, default=None)
    parser.add_argument(
        "--batch-ratio",
        type=_positive_int,
        default=None,
        help="Number of sampled mini-batches per GraphRNN epoch.",
    )
    parser.add_argument(
        "--max-prev-node",
        type=_positive_int,
        default=None,
        help="Maximum BFS look-back width used by the adjacency sequence.",
    )
    parser.add_argument("--num-workers", type=_nonnegative_int, default=None)
    parser.add_argument(
        "--generation-batch-size", type=_positive_int, default=None
    )
    parser.add_argument(
        "--sample-time",
        type=_positive_int,
        default=None,
        help="Bernoulli retry count for GraphRNN_MLP sampling.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Runtime device override, for example auto, cpu, cuda, or cuda:0.",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        default=None,
        help="Optional CUDA_VISIBLE_DEVICES value passed to isolated workers.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=_positive_float,
        default=None,
        help="Optional timeout applied separately to training and generation.",
    )
    parser.add_argument(
        "--skip-training-estimates",
        action="store_true",
        help="Do not generate the optional independent post-training source pool.",
    )
    parser.add_argument(
        "--training-estimate-count",
        type=_positive_int,
        default=None,
        help="Size of the optional independent post-training source pool.",
    )

    # Stable progress controls shared with the other baseline runners.
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
    if args.graphrnn_root is not None:
        os.environ["GRAPHRNN"] = str(args.graphrnn_root.expanduser().resolve())
    if args.graphrnn_python is not None:
        os.environ["GRAPHRNN_PYTHON"] = str(
            args.graphrnn_python.expanduser().resolve()
        )


def run_pipeline(args: argparse.Namespace) -> dict[str, object]:
    """Run one managed train-then-generate transaction."""

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

    runtime_options: dict[str, object] = {"progress": progress}
    for key in ("device", "cuda_visible_devices", "timeout_seconds"):
        value = getattr(args, key)
        if value is not None:
            runtime_options[key] = value

    training_options: dict[str, object] = {
        "training_estimates": training_estimates,
        "runtime": runtime_options,
    }
    for key in (
        "variant",
        "epochs",
        "batch_size",
        "batch_ratio",
        "max_prev_node",
        "num_workers",
        "generation_batch_size",
        "sample_time",
    ):
        value = getattr(args, key)
        if value is not None:
            training_options[key] = value

    run = RunSpec.for_seed(
        model_id="graphrnn",
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
        f"run_id={run.run_id}, seed={args.seed_id}",
        enabled=progress_enabled,
    )
    dataset.require_prepared()

    wrapper = create_baseline("graphrnn")
    _status(
        "prepared dataset found; starting isolated GraphRNN training. "
        f"Durable log: {run.layout.training_log_path.resolve()}",
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

    generation_options: dict[str, object] = {"runtime": runtime_options}
    if args.generation_batch_size is not None:
        generation_options["generation_batch_size"] = args.generation_batch_size
    if args.sample_time is not None:
        generation_options["sample_time"] = args.sample_time
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
            f"GraphRNN generated {generation.num_generated}; expected "
            f"{args.num_samples}."
        )
    _status(
        f"generation completed: graphs={generation.graphs_path.resolve()}",
        enabled=progress_enabled,
    )

    return {
        "status": "complete",
        "model": "graphrnn",
        "dataset": args.dataset,
        "serialized_dataset": profile.serialized_id,
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

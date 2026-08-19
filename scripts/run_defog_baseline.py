#!/usr/bin/env python
"""Train DeFoG on a prepared GraphER dataset and generate one raw batch.

The script is intentionally a thin orchestration layer. Dataset conversion,
isolated upstream execution, checkpoint publication, exact-count generation,
validation, hashing, and manifests remain the responsibility of
``DeFoGWrapper``.
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
    """GraphER storage identity and DeFoG-native dataset identity."""

    serialized_id: str
    native_id: str


DATASET_PROFILES: dict[str, DatasetProfile] = {
    "community_small": DatasetProfile(serialized_id="sbm", native_id="comm20"),
    "ego_small": DatasetProfile(serialized_id="ego_small", native_id="comm20"),
    "qm9": DatasetProfile(serialized_id="qm9_attributed", native_id="qm9"),
    "zinc": DatasetProfile(serialized_id="zinc", native_id="zinc"),
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
    """Write human-readable progress to stderr without corrupting JSON stdout."""

    if not enabled:
        return
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[run_defog_baseline {timestamp}] {message}", file=sys.stderr, flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train the DeFoG baseline through its GraphER wrapper, then generate "
            "and serialize an exact raw graph batch."
        )
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=tuple(DATASET_PROFILES),
        help="Prepared GraphER benchmark to use.",
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
        help=(
            "Seed used for both training and this generation batch. The two "
            "roles remain separately recorded in the wrapper manifests."
        ),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("outputs/datasets"),
        help="Root containing the prepared train.pkl, val.pkl, and test.pkl files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/baselines"),
        help="Root for trained checkpoints and generated batches.",
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
        help="Optional YAML file containing DeFoG wrapper options.",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Optional DeFoG checkpoint from which training should resume.",
    )
    parser.add_argument(
        "--defog-root",
        type=Path,
        default=None,
        help=(
            "Optional DeFoG source root; otherwise use the DEFOG "
            "environment variable."
        ),
    )
    parser.add_argument(
        "--defog-python",
        type=Path,
        default=None,
        help=(
            "Optional Python executable for DeFoG; otherwise use "
            "the DEFOG_PYTHON environment variable."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Explicitly allow replacement of matching artifacts. Training "
            "overwrite remains forbidden when the run already has generations."
        ),
    )
    parser.add_argument(
        "--progress-interval-seconds",
        type=_positive_float,
        default=15.0,
        help=(
            "Heartbeat interval while DeFoG subprocesses are running. Progress "
            "is written to stderr; the final JSON summary remains on stdout."
        ),
    )
    parser.add_argument(
        "--epoch-progress-interval",
        type=_positive_int,
        default=None,
        help=(
            "Optional number of training epochs between stable progress lines. "
            "By default the worker chooses an interval that emits about 100 "
            "lines over the configured horizon."
        ),
    )
    parser.add_argument(
        "--generation-progress-every-batches",
        type=_positive_int,
        default=1,
        help="Report generation progress every N completed sampling batches.",
    )
    parser.add_argument(
        "--no-stream-subprocess-output",
        action="store_true",
        help=(
            "Show stage transitions and heartbeats only instead of mirroring "
            "the complete DeFoG subprocess output to the terminal."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output and print only the final JSON summary.",
    )
    return parser


def _configure_upstream_environment(args: argparse.Namespace) -> None:
    if args.defog_root is not None:
        os.environ["DEFOG"] = str(args.defog_root.expanduser().resolve())
    if args.defog_python is not None:
        os.environ["DEFOG_PYTHON"] = str(
            args.defog_python.expanduser().resolve()
        )


def run_pipeline(args: argparse.Namespace) -> dict[str, object]:
    """Run one train-then-generate transaction and return its artifact summary."""

    _configure_upstream_environment(args)
    progress_enabled = not args.quiet
    progress_options: dict[str, object] = {
        "enabled": progress_enabled,
        "stream_output": (
            progress_enabled and not args.no_stream_subprocess_output
        ),
        "interval_seconds": args.progress_interval_seconds,
        "generation_batch_interval": args.generation_progress_every_batches,
    }
    if args.epoch_progress_interval is not None:
        progress_options["epoch_interval"] = args.epoch_progress_interval
    profile = DATASET_PROFILES[args.dataset]
    run = RunSpec.for_seed(
        model_id="defog",
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
        f"native={profile.native_id}, run_id={run.run_id}, seed={args.seed_id}",
        enabled=progress_enabled,
    )
    _status(
        f"checking prepared splits under {dataset.dataset_dir.resolve()}",
        enabled=progress_enabled,
    )
    # Fail before launching the external training process when preparation is
    # incomplete or the selected dataset alias points to the wrong directory.
    dataset.require_prepared()
    _status(
        "prepared dataset found; starting DeFoG training. Durable log will be "
        f"published at {run.layout.training_log_path.resolve()}",
        enabled=progress_enabled,
    )

    wrapper = create_baseline("defog")
    training = wrapper.train(
        TrainRequest(
            run=run,
            dataset=dataset,
            config_path=args.wrapper_config,
            options={
                "training_estimates": {"enabled": True},
                "runtime": {"progress": progress_options},
            },
            resume_from=args.resume_from,
            overwrite=args.overwrite,
        )
    )
    _status(
        "training completed and artifacts were published: "
        f"checkpoint={training.checkpoint_path.resolve()}, "
        f"log={training.log_path.resolve() if training.log_path else None}",
        enabled=progress_enabled,
    )
    _status(
        "starting final raw-batch generation: "
        f"requested={args.num_samples}, seed={args.seed_id}",
        enabled=progress_enabled,
    )
    generation = wrapper.generate(
        GenerateRequest(
            run=run,
            checkpoint_path=training.checkpoint_path,
            num_graphs=args.num_samples,
            generation_seed=args.seed_id,
            generation_id=args.generation_id,
            options={"runtime": {"progress": progress_options}},
            overwrite=args.overwrite,
        )
    )
    if generation.num_generated != args.num_samples:
        raise RuntimeError(
            "DeFoG generation count mismatch after wrapper validation: "
            f"requested {args.num_samples}, received {generation.num_generated}."
        )
    _status(
        "generation completed: "
        f"graphs={generation.graphs_path.resolve()}, "
        f"log={generation.log_path.resolve() if generation.log_path else None}",
        enabled=progress_enabled,
    )

    return {
        "status": "complete",
        "model": "defog",
        "dataset": args.dataset,
        "serialized_dataset": profile.serialized_id,
        "native_dataset": profile.native_id,
        "seed_id": args.seed_id,
        "run_id": run.run_id,
        "generation_id": args.generation_id
        or generation.generation_dir.name,
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
            f"FAILED: {type(exc).__name__}: {exc}",
            enabled=not args.quiet,
        )
        raise
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

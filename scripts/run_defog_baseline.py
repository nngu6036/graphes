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
from dataclasses import dataclass
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
    # Fail before launching the external training process when preparation is
    # incomplete or the selected dataset alias points to the wrong directory.
    dataset.require_prepared()

    wrapper = create_baseline("defog")
    training = wrapper.train(
        TrainRequest(
            run=run,
            dataset=dataset,
            config_path=args.wrapper_config,
            options={"training_estimates": {"enabled": True}},
            resume_from=args.resume_from,
            overwrite=args.overwrite,
        )
    )
    generation = wrapper.generate(
        GenerateRequest(
            run=run,
            checkpoint_path=training.checkpoint_path,
            num_graphs=args.num_samples,
            generation_seed=args.seed_id,
            generation_id=args.generation_id,
            overwrite=args.overwrite,
        )
    )
    if generation.num_generated != args.num_samples:
        raise RuntimeError(
            "DeFoG generation count mismatch after wrapper validation: "
            f"requested {args.num_samples}, received {generation.num_generated}."
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
    summary = run_pipeline(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

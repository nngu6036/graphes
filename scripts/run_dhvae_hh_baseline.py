#!/usr/bin/env python
"""Train DH-VAE+HH and publish one ordered raw generation batch.

The script is a thin command-line layer over :class:`DHVAEHHWrapper`. The
wrapper remains responsible for configuration resolution, checkpoint and
manifest publication, exact-count generation, checksums, and collision-safe
artifact paths.
"""

from __future__ import annotations

import argparse
import json
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
    serialized_id: str
    config_path: Path


DATASET_PROFILES: dict[str, DatasetProfile] = {
    "community_small": DatasetProfile(
        serialized_id="sbm",
        config_path=Path("configs/datasets/community_small.yaml"),
    ),
    "ego_small": DatasetProfile(
        serialized_id="ego_small",
        config_path=Path("configs/datasets/ego_small.yaml"),
    ),
    "grid": DatasetProfile(
        serialized_id="grid",
        config_path=Path("configs/datasets/grid.yaml"),
    ),
    "qm9": DatasetProfile(
        serialized_id="qm9_attributed",
        config_path=Path("configs/datasets/qm9.yaml"),
    ),
    "zinc": DatasetProfile(
        serialized_id="zinc",
        config_path=Path("configs/datasets/zinc.yaml"),
    ),
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
            "Train the DH-VAE invariant prior, construct graphs with randomized "
            "Havel--Hakimi, and save one exact raw batch."
        )
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=tuple(DATASET_PROFILES),
        help="Prepared GraphER benchmark.",
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
        help="Seed used for training and for the requested raw batch.",
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
        help="Root for managed baseline checkpoints and generated batches.",
    )
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=None,
        help=(
            "Optional DH-VAE experiment YAML. If omitted, the wrapper chooses "
            "the project configuration for the selected benchmark."
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device for training and generation, for example auto or cpu.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Training run identifier; defaults to seed_<seed-id>.",
    )
    parser.add_argument(
        "--generation-id",
        default=None,
        help=(
            "Generation identifier; defaults to "
            "seed_<seed-id>_n_<num-samples>."
        ),
    )
    parser.add_argument(
        "--training-estimate-count",
        type=_positive_int,
        default=None,
        help=(
            "Optional size of the unpaired post-training estimate pool. "
            "Defaults to the full generic train split and at most 1024 molecules."
        ),
    )
    parser.add_argument(
        "--disable-training-estimates",
        action="store_true",
        help="Do not sample the optional post-training estimate pool.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Explicitly replace matching artifacts. Training cannot be replaced "
            "after generations have been attached to that run."
        ),
    )
    return parser


def run_pipeline(args: argparse.Namespace) -> dict[str, object]:
    profile = DATASET_PROFILES[args.dataset]
    run = RunSpec.for_seed(
        model_id="dhvae_hh",
        dataset_id=args.dataset,
        seed=args.seed_id,
        run_id=args.run_id,
        output_root=args.output_root,
    )
    dataset = DatasetReference(
        benchmark_id=args.dataset,
        root=args.dataset_root,
        serialized_id=profile.serialized_id,
        config_path=profile.config_path,
    )
    dataset.require_prepared()

    estimate_options: dict[str, object] = {
        "enabled": not args.disable_training_estimates,
    }
    if args.training_estimate_count is not None:
        estimate_options["num_graphs"] = args.training_estimate_count

    wrapper = create_baseline("dhvae_hh")
    training = wrapper.train(
        TrainRequest(
            run=run,
            dataset=dataset,
            config_path=args.experiment_config,
            options={
                "runtime": {"device": args.device},
                "training_estimates": estimate_options,
            },
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
            options={"runtime": {"device": args.device}},
            overwrite=args.overwrite,
        )
    )
    if generation.num_generated != args.num_samples:
        raise RuntimeError(
            "DH-VAE+HH generation count mismatch: "
            f"requested {args.num_samples}, received {generation.num_generated}."
        )

    return {
        "status": "complete",
        "model": "dhvae_hh",
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
    summary = run_pipeline(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

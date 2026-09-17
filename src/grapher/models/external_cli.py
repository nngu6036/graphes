"""Uniform command-line orchestration for GraphER baseline wrappers.

Every ``scripts/run_<model>_baseline.py`` entry point is intentionally a tiny
shim over :func:`main`.  Dataset identity, common/model config precedence,
model-specific CLI compatibility flags, runtime controls, managed checkpoint
reuse, and train/generate orchestration live here so baseline launchers cannot
drift apart.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from grapher.models import (
    DatasetReference,
    GenerateRequest,
    RunSpec,
    TrainRequest,
    create_baseline,
    normalize_baseline_id,
)
from grapher.models.comparison import (
    comparison_request_options,
    resolve_common_config,
    resolve_wrapper_config,
)
from grapher.models.external_codec import PROFILES


@dataclass(frozen=True)
class DatasetCLIProfile:
    serialized_id: str
    native_id: str | None = None
    experiment: str | None = None
    dataset_config: str | None = None


_SOURCE_BACKED = frozenset({"catflow", "gdsm", "edge", "spectre"})

_MODEL_DATASETS: dict[str, dict[str, DatasetCLIProfile]] = {
    "catflow": {
        key: DatasetCLIProfile(value.serialized_id)
        for key, value in PROFILES.items()
    },
    "gdsm": {
        key: DatasetCLIProfile(PROFILES[key].serialized_id)
        for key in ("community_small", "ego_small", "grid")
    },
    "gdsm_simple": {
        key: DatasetCLIProfile(PROFILES[key].serialized_id)
        for key in ("community_small", "ego_small", "grid", "qm9", "zinc")
    },
    "edge": {
        key: DatasetCLIProfile(PROFILES[key].serialized_id)
        for key in ("community_small", "ego_small", "grid")
    },
    "spectre": {
        key: DatasetCLIProfile(PROFILES[key].serialized_id)
        for key in ("community_small", "ego_small", "grid", "qm9")
    },
    "defog": {
        "community_small": DatasetCLIProfile("sbm", "comm20"),
        "ego_small": DatasetCLIProfile("ego_small", "comm20"),
        "qm9": DatasetCLIProfile("qm9_attributed", "qm9"),
        "zinc": DatasetCLIProfile("zinc", "zinc"),
    },
    "digress": {
        "community_small": DatasetCLIProfile("sbm", "comm20", "comm20"),
        "ego_small": DatasetCLIProfile("ego_small", "comm20", "comm20"),
        "grid": DatasetCLIProfile("grid", "planar", "planar"),
        "qm9": DatasetCLIProfile("qm9_attributed", "qm9", "qm9_no_h"),
        "zinc": DatasetCLIProfile("zinc", "zinc", "zinc_no_h"),
    },
    "gdss": {
        "community_small": DatasetCLIProfile("sbm", "community_small"),
        "ego_small": DatasetCLIProfile("ego_small", "ego_small"),
        "grid": DatasetCLIProfile("grid", "grid"),
        "qm9": DatasetCLIProfile("qm9_attributed", "QM9"),
        "zinc": DatasetCLIProfile("zinc", "ZINC250k"),
    },
    "graphrnn": {
        "community_small": DatasetCLIProfile("sbm", "community_small"),
        "ego_small": DatasetCLIProfile("ego_small", "ego_small"),
        "grid": DatasetCLIProfile("grid", "grid"),
    },
    "hog_diff": {
        "community_small": DatasetCLIProfile("sbm", "community_small"),
        "ego_small": DatasetCLIProfile("ego_small", "ego_small"),
        "qm9": DatasetCLIProfile("qm9_attributed", "qm9"),
        "zinc": DatasetCLIProfile("zinc", "zinc250k"),
    },
    "dhvae_hh": {
        "community_small": DatasetCLIProfile(
            "sbm", dataset_config="configs/datasets/community_small.yaml"
        ),
        "ego_small": DatasetCLIProfile(
            "ego_small", dataset_config="configs/datasets/ego_small.yaml"
        ),
        "grid": DatasetCLIProfile(
            "grid", dataset_config="configs/datasets/grid.yaml"
        ),
        "qm9": DatasetCLIProfile(
            "qm9_attributed", dataset_config="configs/datasets/qm9.yaml"
        ),
        "zinc": DatasetCLIProfile(
            "zinc", dataset_config="configs/datasets/zinc.yaml"
        ),
    },
}

_SOURCE_ENVS: dict[str, tuple[str, str]] = {
    "catflow": ("CATFLOW", "CATFLOW_PYTHON"),
    "gdsm": ("GDSM", "GDSM_PYTHON"),
    "edge": ("EDGE", "EDGE_PYTHON"),
    "spectre": ("SPECTRE", "SPECTRE_PYTHON"),
    "defog": ("DEFOG", "DEFOG_PYTHON"),
    "digress": ("DIGRESS", "DIGRESS_PYTHON"),
    "gdss": ("GDSS", "GDSS_PYTHON"),
    "graphrnn": ("GRAPHRNN", "GRAPHRNN_PYTHON"),
    "hog_diff": ("HOGDIFF", "HOGDIFF_PYTHON"),
}

# Preserve the historical runner behaviour.  Source-backed adapters intentionally
# defer to their YAML instead of forcing a training-estimate pool.
_TRAINING_ESTIMATE_DEFAULT: dict[str, bool | None] = {
    "defog": True,
    "digress": True,
    "gdss": True,
    "graphrnn": True,
    "hog_diff": True,
    "dhvae_hh": True,
    "catflow": None,
    "gdsm": None,
    "gdsm_simple": None,
    "edge": None,
    "spectre": None,
}


# Generic user-prepared categorical graphs; never impersonates a benchmark.
_MODEL_DATASETS['gdsm_simple']['attributed'] = DatasetCLIProfile('attributed')


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


def _status(model: str, message: str, *, enabled: bool) -> None:
    if not enabled:
        return
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(
        f"[run_{model}_baseline {timestamp}] {message}",
        file=sys.stderr,
        flush=True,
    )


def _source_flag_names(model: str, suffix: str) -> tuple[str, ...]:
    names = [f"--{model}-{suffix}"]
    compact = model.replace("_", "")
    dashed = model.replace("_", "-")
    for candidate in (f"--{compact}-{suffix}", f"--{dashed}-{suffix}"):
        if candidate not in names:
            names.append(candidate)
    if model == "gdsm":
        for candidate in (f"--gsdm-{suffix}",):
            if candidate not in names:
                names.append(candidate)
    return tuple(names)


def _add_common_arguments(parser: argparse.ArgumentParser, model: str) -> None:
    parser.add_argument(
        "--stage",
        choices=("train", "generate", "all"),
        default="all",
        help="Run training only, generation only, or training followed by generation.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=tuple(sorted(_MODEL_DATASETS[model])),
        help="Prepared GraphER benchmark.",
    )
    parser.add_argument(
        "--seed-id",
        "--seed",
        dest="seed_id",
        type=_nonnegative_int,
        default=42,
        help="Training seed. Defaults to 42.",
    )
    parser.add_argument(
        "--generation-seed",
        type=_nonnegative_int,
        default=None,
        help="Generation seed; defaults to --seed-id.",
    )
    parser.add_argument(
        "--num-samples",
        type=_positive_int,
        default=1024,
        help="Exact number of graphs to generate for generate/all stages.",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--generation-id", default=None)
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("outputs/datasets")
    )
    parser.add_argument("--serialized-dataset", default=None)
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/baselines")
    )
    if model == "dhvae_hh":
        parser.add_argument(
            "--wrapper-config",
            "--experiment-config",
            dest="wrapper_config",
            type=Path,
            default=None,
            help="DH-VAE experiment YAML.",
        )
    else:
        parser.add_argument(
            "--wrapper-config",
            type=Path,
            default=None,
            help=(
                "Model-specific YAML; defaults to "
                f"configs/baselines/{model}_<dataset>.yaml when present."
            ),
        )
        parser.add_argument(
            "--common-config",
            type=Path,
            default=None,
            help=(
                "DeFoG-reference common profile; defaults to "
                "configs/baselines/common_<dataset>.yaml when present."
            ),
        )
        parser.add_argument(
            "--no-common-config",
            action="store_true",
            help="Disable automatic common-profile loading.",
        )

    if model in _SOURCE_ENVS:
        parser.add_argument(
            "--source-root",
            *_source_flag_names(model, "root"),
            dest="source_root",
            type=Path,
            default=None,
            help="Upstream source root; otherwise use the model-specific environment variable.",
        )
        parser.add_argument(
            "--python",
            *_source_flag_names(model, "python"),
            dest="python",
            type=Path,
            default=None,
            help="Upstream Python executable; otherwise use the model-specific environment variable.",
        )

    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Optional training checkpoint for wrappers that support optimizer-state resume.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint for --stage generate; defaults to the managed run checkpoint.",
    )
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument(
        "--device",
        default=None,
        help="auto, cpu, gpu, cuda or cuda:N. Model wrappers perform final validation.",
    )
    parser.add_argument(
        "--gpu-id",
        type=_nonnegative_int,
        default=None,
        help="Compatibility alias for selecting one visible physical GPU.",
    )
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--timeout-seconds", type=_positive_float, default=None)

    parser.add_argument(
        "--epochs",
        "--n-epochs",
        "--num-epochs",
        dest="epochs",
        type=_positive_int,
        default=None,
        help="Model-native epoch horizon where the wrapper has a single epoch-based stage.",
    )
    parser.add_argument("--batch-size", type=_positive_int, default=None)
    parser.add_argument("--generation-batch-size", type=_positive_int, default=None)
    parser.add_argument("--num-workers", type=_nonnegative_int, default=None)
    parser.add_argument("--max-nodes", type=_positive_int, default=None)

    estimates = parser.add_mutually_exclusive_group()
    estimates.add_argument(
        "--training-estimates",
        dest="training_estimates",
        action="store_true",
        help="Enable the independent post-training source pool.",
    )
    estimates.add_argument(
        "--skip-training-estimates",
        "--disable-training-estimates",
        dest="training_estimates",
        action="store_false",
        help="Disable the independent post-training source pool.",
    )
    parser.set_defaults(training_estimates=None)
    parser.add_argument("--training-estimate-count", type=_positive_int, default=None)

    parser.add_argument(
        "--progress-interval-seconds", type=_positive_float, default=15.0
    )
    parser.add_argument("--epoch-progress-interval", type=_positive_int, default=None)
    parser.add_argument(
        "--generation-progress-every-batches", type=_positive_int, default=1
    )
    parser.add_argument("--no-stream-subprocess-output", action="store_true")
    parser.add_argument("--quiet", action="store_true")


def _add_model_arguments(parser: argparse.ArgumentParser, model: str) -> None:
    if model == "digress":
        parser.add_argument("--check-val-every-n-epochs", type=_positive_int)
        parser.add_argument("--save-every-n-epochs", type=_positive_int)
    elif model == "graphrnn":
        parser.add_argument("--variant", default=None)
        parser.add_argument("--batch-ratio", type=_positive_int, default=None)
        parser.add_argument("--max-prev-node", type=_positive_int, default=None)
        parser.add_argument("--sample-time", type=_positive_int, default=None)
    elif model == "hog_diff":
        parser.add_argument("--ho-iters", type=_nonnegative_int, default=None)
        parser.add_argument("--ou-iters", type=_nonnegative_int, default=None)
        parser.add_argument("--ho-batch-size", type=_positive_int, default=None)
        parser.add_argument("--ou-batch-size", type=_positive_int, default=None)
        parser.add_argument("--generation-max-retries", type=_positive_int, default=None)
        parser.add_argument(
            "--iteration-progress-interval", type=_positive_int, default=None
        )
        parser.add_argument(
            "--generation-only",
            action="store_true",
            help="Compatibility alias for --stage generate.",
        )
    elif model == "dhvae_hh":
        parser.add_argument("--max-attempts-per-graph", type=_positive_int, default=None)


def build_parser(model_id: str) -> argparse.ArgumentParser:
    model = normalize_baseline_id(model_id)
    if model not in _MODEL_DATASETS:
        raise KeyError(f"No external CLI profile is registered for {model!r}.")
    wrapper = create_baseline(model)
    parser = argparse.ArgumentParser(
        description=(
            f"Train/generate {wrapper.display_name} through the shared GraphER "
            "baseline orchestration CLI."
        )
    )
    _add_common_arguments(parser, model)
    _add_model_arguments(parser, model)
    return parser


def _configure_source_environment(model: str, args: argparse.Namespace) -> None:
    if model not in _SOURCE_ENVS:
        return
    root_env, python_env = _SOURCE_ENVS[model]
    if args.source_root is not None:
        os.environ[root_env] = str(args.source_root.expanduser().resolve())
    if args.python is not None:
        os.environ[python_env] = str(args.python.expanduser().resolve())


def _runtime_options(model: str, args: argparse.Namespace) -> dict[str, Any]:
    progress: dict[str, Any] = {
        "enabled": not args.quiet,
        "stream_output": not args.quiet and not args.no_stream_subprocess_output,
        "interval_seconds": float(args.progress_interval_seconds),
        "generation_batch_interval": int(args.generation_progress_every_batches),
    }
    if args.epoch_progress_interval is not None:
        progress["epoch_interval"] = int(args.epoch_progress_interval)
    if model == "hog_diff" and args.iteration_progress_interval is not None:
        progress["iteration_interval"] = int(args.iteration_progress_interval)

    runtime: dict[str, Any] = {"progress": progress}
    device = None if args.device is None else str(args.device).lower()

    visible = args.cuda_visible_devices
    if args.gpu_id is not None:
        gpu_visible = str(args.gpu_id)
        if visible is not None and str(visible) != gpu_visible:
            raise ValueError("--gpu-id conflicts with --cuda-visible-devices.")
        visible = gpu_visible
        if device not in (None, "gpu", "cuda"):
            raise ValueError("--gpu-id requires a GPU device.")
        device = "gpu"

    if model in {"defog", "digress"}:
        resolved = device or "auto"
        if resolved in {"gpu", "cuda"}:
            runtime.update(
                {
                    "gpus": 1,
                    "device": "cuda" if model == "defog" else "gpu",
                    "cuda_visible_devices": str(visible or "0"),
                    "require_cuda": True,
                }
            )
        elif resolved == "cpu":
            runtime.update({"gpus": 0, "device": "cpu"})
            if visible is not None:
                raise ValueError("CUDA visibility cannot be set with --device cpu.")
        elif resolved != "auto":
            raise ValueError(
                f"{model} accepts --device auto/cpu/gpu through the shared CLI, got {device!r}."
            )
        elif visible is not None:
            runtime["cuda_visible_devices"] = str(visible)
    else:
        if device is not None:
            runtime["device"] = device
        if visible is not None:
            runtime["cuda_visible_devices"] = str(visible)

    if args.timeout_seconds is not None:
        runtime["timeout_seconds"] = float(args.timeout_seconds)
    return runtime


def _training_estimate_options(model: str, args: argparse.Namespace) -> dict[str, Any] | None:
    enabled = args.training_estimates
    if enabled is None:
        enabled = _TRAINING_ESTIMATE_DEFAULT.get(model)
    if enabled is None and args.training_estimate_count is None:
        return None
    result: dict[str, Any] = {}
    if enabled is not None:
        result["enabled"] = bool(enabled)
    if args.training_estimate_count is not None:
        result["num_graphs"] = int(args.training_estimate_count)
        result.setdefault("enabled", True)
    return result


def _training_options(
    model: str,
    args: argparse.Namespace,
    *,
    common: Any,
    wrapper_config: Path | None,
    profile: DatasetCLIProfile,
) -> dict[str, Any]:
    if model == "dhvae_hh":
        options: dict[str, Any] = {}
    else:
        options = comparison_request_options(
            model, common, model_config=wrapper_config
        )

    runtime = _runtime_options(model, args)
    options["runtime"] = runtime

    estimate_options = _training_estimate_options(model, args)
    if estimate_options is not None:
        options["training_estimates"] = estimate_options

    if model in _SOURCE_ENVS:
        if args.source_root is not None and model in _SOURCE_BACKED:
            options["source_root"] = str(args.source_root.expanduser().resolve())
        if args.python is not None and model in _SOURCE_BACKED:
            options["python"] = str(args.python.expanduser().resolve())

    if model in _SOURCE_BACKED:
        train: dict[str, Any] = {}
        if args.epochs is not None:
            train["epochs"] = int(args.epochs)
        if args.batch_size is not None:
            train["batch_size"] = int(args.batch_size)
        if args.epoch_progress_interval is not None:
            train["log_every"] = int(args.epoch_progress_interval)
        if train:
            options["train"] = train
        if args.generation_batch_size is not None:
            options["generation_batch_size"] = int(args.generation_batch_size)
        if args.max_nodes is not None:
            options["max_nodes"] = int(args.max_nodes)
        if args.num_workers is not None:
            # None of the current source-backed workers consumes this generic
            # field. Refuse to silently pretend otherwise.
            raise ValueError(
                f"--num-workers is not a supported shared override for {model}; "
                "set the model YAML if the upstream implementation exposes it."
            )
        return options

    if args.max_nodes is not None:
        raise ValueError(f"--max-nodes is only supported by source-backed adapters, not {model}.")

    if model == "gdsm_simple":
        train = options.setdefault("train", {})
        if args.epochs is not None:
            train["epochs"] = int(args.epochs)
        if args.batch_size is not None:
            train["batch_size"] = int(args.batch_size)
        if args.num_workers is not None:
            raise ValueError("gdsm_simple is an in-process tensor baseline and does not use DataLoader workers yet")
        if args.generation_batch_size is not None:
            options["generation_batch_size"] = int(args.generation_batch_size)
        return options

    if model == "defog":
        if args.epochs is not None:
            options["n_epochs"] = int(args.epochs)
        if args.batch_size is not None:
            options["batch_size"] = int(args.batch_size)
        if args.num_workers is not None:
            options["num_workers"] = int(args.num_workers)
        if args.generation_batch_size is not None:
            options["generation_batch_size"] = int(args.generation_batch_size)
        return options

    if model == "digress":
        if profile.experiment is not None:
            options["experiment"] = profile.experiment
        for key, value in (
            ("n_epochs", args.epochs),
            ("batch_size", args.batch_size),
            ("num_workers", args.num_workers),
            ("check_val_every_n_epochs", args.check_val_every_n_epochs),
            ("save_every_n_epochs", args.save_every_n_epochs),
            ("generation_batch_size", args.generation_batch_size),
        ):
            if value is not None:
                options[key] = int(value)
        return options

    if model == "gdss":
        if args.epochs is not None:
            options["train"] = {"num_epochs": int(args.epochs)}
        if args.batch_size is not None:
            options["batch_size"] = int(args.batch_size)
        if args.num_workers is not None:
            options["num_workers"] = int(args.num_workers)
        if args.generation_batch_size is not None:
            options["generation_batch_size"] = int(args.generation_batch_size)
        return options

    if model == "graphrnn":
        for key, value in (
            ("variant", args.variant),
            ("epochs", args.epochs),
            ("batch_size", args.batch_size),
            ("batch_ratio", args.batch_ratio),
            ("max_prev_node", args.max_prev_node),
            ("num_workers", args.num_workers),
            ("generation_batch_size", args.generation_batch_size),
            ("sample_time", args.sample_time),
        ):
            if value is not None:
                options[key] = value if key == "variant" else int(value)
        return options

    if model == "hog_diff":
        if args.epochs is not None:
            raise ValueError(
                "HOG-Diff has two native training horizons; use --ho-iters and --ou-iters, not --epochs."
            )
        if args.batch_size is not None:
            raise ValueError(
                "HOG-Diff has two stage-specific batch sizes; use --ho-batch-size and --ou-batch-size."
            )
        if args.num_workers is not None:
            options["num_workers"] = int(args.num_workers)
        if args.generation_batch_size is not None:
            options["generation_batch_size"] = int(args.generation_batch_size)
        if args.generation_max_retries is not None:
            options["generation_max_retries"] = int(args.generation_max_retries)
        higher_order: dict[str, Any] = {}
        if args.ho_iters is not None:
            higher_order["n_iters"] = int(args.ho_iters)
        if args.ho_batch_size is not None:
            higher_order["batch_size"] = int(args.ho_batch_size)
        if higher_order:
            options["higher_order"] = higher_order
        ou: dict[str, Any] = {}
        if args.ou_iters is not None:
            ou["n_iters"] = int(args.ou_iters)
        if args.ou_batch_size is not None:
            ou["batch_size"] = int(args.ou_batch_size)
        if ou:
            options["ou"] = ou
        return options

    if model == "dhvae_hh":
        if any(value is not None for value in (args.epochs, args.batch_size, args.num_workers)):
            raise ValueError(
                "DH-VAE training hyperparameters belong in --experiment-config; "
                "the shared CLI does not reinterpret generic epoch/batch flags."
            )
        # DHVAEHHWrapper accepts only device/timeout in training runtime.
        dh_runtime: dict[str, Any] = {}
        if args.device is not None:
            dh_runtime["device"] = str(args.device)
        if args.timeout_seconds is not None:
            dh_runtime["timeout_seconds"] = float(args.timeout_seconds)
        options["runtime"] = dh_runtime
        return options

    raise KeyError(model)


def _generation_options(
    model: str,
    args: argparse.Namespace,
    *,
    wrapper_config: Path | None = None,
) -> dict[str, Any]:
    runtime = _runtime_options(model, args)

    if model == "dhvae_hh":
        # The DH-VAE generator accepts device only; timeout is a training concern.
        result: dict[str, Any] = {"runtime": {}}
        if args.device is not None:
            result["runtime"]["device"] = str(args.device)
        if args.max_attempts_per_graph is not None:
            result["max_attempts_per_graph"] = int(args.max_attempts_per_graph)
        return result

    result: dict[str, Any] = {}
    # gdsm_simple has generation-only spectral-rewiring controls in its YAML.
    # Reading only generation-safe sections here lets an already-trained S0
    # checkpoint be reused for S1 rewiring without an unnecessary retrain.
    if model == "gdsm_simple" and wrapper_config is not None:
        loaded = yaml.safe_load(wrapper_config.read_text(encoding="utf-8")) or {}
        selected = loaded.get("gdsm_simple", loaded)
        if not isinstance(selected, Mapping):
            raise TypeError("gdsm_simple config section must be a mapping")
        for key in ("sample", "extensions", "generation_batch_size"):
            if key in selected:
                value = selected[key]
                result[key] = (
                    dict(value) if isinstance(value, Mapping) else value
                )

    # Explicit CLI runtime / throughput controls remain highest priority.
    result["runtime"] = runtime
    if args.generation_batch_size is not None:
        result["generation_batch_size"] = int(args.generation_batch_size)
    if model == "graphrnn" and args.sample_time is not None:
        result["sample_time"] = int(args.sample_time)
    if model == "hog_diff" and args.generation_max_retries is not None:
        result["generation_max_retries"] = int(args.generation_max_retries)
    if model in _SOURCE_BACKED:
        if args.source_root is not None:
            result["source_root"] = str(args.source_root.expanduser().resolve())
        if args.python is not None:
            result["python"] = str(args.python.expanduser().resolve())
    return result


def _dataset_reference(
    model: str,
    args: argparse.Namespace,
    profile: DatasetCLIProfile,
) -> DatasetReference:
    config_path = Path(profile.dataset_config) if profile.dataset_config else None
    return DatasetReference(
        benchmark_id=args.dataset,
        root=args.dataset_root,
        serialized_id=args.serialized_dataset or profile.serialized_id,
        native_id=profile.native_id,
        config_path=config_path,
    )


def _managed_checkpoint(run: RunSpec) -> Path:
    manifest_path = run.layout.training_manifest_path
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "No managed training manifest exists for generation: "
            f"{manifest_path}. Train the run first or select the correct --run-id."
        )
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint = raw.get("checkpoint")
    if not isinstance(checkpoint, Mapping) or not checkpoint.get("path"):
        raise RuntimeError(
            f"Managed training manifest has no checkpoint record: {manifest_path}"
        )
    value = Path(str(checkpoint["path"]))
    if not value.is_absolute():
        value = run.layout.train_dir / value
    return value


def _reject_training_overrides_during_generation(model: str, args: argparse.Namespace) -> None:
    fields = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "max_nodes": args.max_nodes,
        "resume_from": args.resume_from,
    }
    if model == "digress":
        fields.update(
            check_val_every_n_epochs=args.check_val_every_n_epochs,
            save_every_n_epochs=args.save_every_n_epochs,
        )
    if model == "graphrnn":
        fields.update(
            variant=args.variant,
            batch_ratio=args.batch_ratio,
            max_prev_node=args.max_prev_node,
        )
    if model == "hog_diff":
        fields.update(
            ho_iters=args.ho_iters,
            ou_iters=args.ou_iters,
            ho_batch_size=args.ho_batch_size,
            ou_batch_size=args.ou_batch_size,
        )
    present = [key for key, value in fields.items() if value is not None]
    if present:
        raise ValueError(
            "Generation reuses the managed trained architecture/config; training-only "
            f"overrides are not allowed with --stage generate: {present}."
        )


def _artifact_summary(training: Any | None) -> dict[str, Any]:
    if training is None:
        return {}
    return {
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
    }


def run(model_id: str, args: argparse.Namespace) -> dict[str, Any]:
    model = normalize_baseline_id(model_id)
    if model not in _MODEL_DATASETS:
        raise KeyError(f"No external CLI profile is registered for {model!r}.")
    if model == "hog_diff" and getattr(args, "generation_only", False):
        if args.stage == "train":
            raise ValueError("--generation-only conflicts with --stage train.")
        args.stage = "generate"

    _configure_source_environment(model, args)
    profile = _MODEL_DATASETS[model][args.dataset]
    progress_enabled = not args.quiet

    if model == "dhvae_hh":
        common = None
        wrapper_config = (
            args.wrapper_config.expanduser().resolve()
            if args.wrapper_config is not None
            else None
        )
        if wrapper_config is not None and not wrapper_config.is_file():
            raise FileNotFoundError(f"Missing DH-VAE experiment config: {wrapper_config}")
    else:
        common = resolve_common_config(
            args.dataset,
            args.common_config,
            disabled=args.no_common_config,
        )
        wrapper_config = resolve_wrapper_config(model, args.dataset, args.wrapper_config)

    run_spec = RunSpec.for_seed(
        model_id=model,
        dataset_id=args.dataset,
        seed=args.seed_id,
        output_root=args.output_root,
        run_id=args.run_id,
    )
    dataset = _dataset_reference(model, args, profile)
    wrapper = create_baseline(model)

    summary: dict[str, Any] = {
        "status": "complete",
        "model": model,
        "dataset": args.dataset,
        "serialized_dataset": dataset.serialized_id,
        "native_dataset": dataset.native_id,
        "experiment": profile.experiment,
        "seed_id": args.seed_id,
        "generation_seed": (
            args.generation_seed if args.generation_seed is not None else args.seed_id
        ),
        "stage": args.stage,
        "run_id": run_spec.run_id,
        "run_dir": str(run_spec.layout.run_dir.resolve()),
        "wrapper_config": str(wrapper_config) if wrapper_config is not None else None,
        "common_config": str(common.path) if common is not None else None,
    }

    checkpoint: Path
    training = None
    if args.stage in {"train", "all"}:
        if args.checkpoint is not None:
            raise ValueError("--checkpoint is only valid with --stage generate.")
        dataset.require_prepared()
        training_options = _training_options(
            model,
            args,
            common=common,
            wrapper_config=wrapper_config,
            profile=profile,
        )
        _status(
            model,
            f"starting training: dataset={args.dataset}, run_id={run_spec.run_id}, seed={args.seed_id}",
            enabled=progress_enabled,
        )
        training = wrapper.train(
            TrainRequest(
                run=run_spec,
                dataset=dataset,
                config_path=wrapper_config,
                options=training_options,
                resume_from=args.resume_from,
                overwrite=args.overwrite,
            )
        )
        checkpoint = training.checkpoint_path
        summary["checkpoint"] = str(checkpoint.resolve())
        summary.update(_artifact_summary(training))
        _status(
            model,
            f"training completed: checkpoint={checkpoint.resolve()}",
            enabled=progress_enabled,
        )
    else:
        _reject_training_overrides_during_generation(model, args)
        checkpoint = (
            args.checkpoint.expanduser().resolve()
            if args.checkpoint is not None
            else _managed_checkpoint(run_spec).resolve()
        )
        summary["checkpoint"] = str(checkpoint)
        if run_spec.layout.training_manifest_path.is_file():
            summary["training_manifest"] = str(
                run_spec.layout.training_manifest_path.resolve()
            )

    if args.stage in {"generate", "all"}:
        generation_seed = (
            args.generation_seed if args.generation_seed is not None else args.seed_id
        )
        generation_options = _generation_options(
            model, args, wrapper_config=wrapper_config
        )
        _status(
            model,
            f"starting generation: requested={args.num_samples}, seed={generation_seed}",
            enabled=progress_enabled,
        )
        generation = wrapper.generate(
            GenerateRequest(
                run=run_spec,
                checkpoint_path=checkpoint,
                num_graphs=args.num_samples,
                generation_seed=generation_seed,
                generation_id=args.generation_id,
                options=generation_options,
                overwrite=args.overwrite,
            )
        )
        if generation.num_generated != args.num_samples:
            raise RuntimeError(
                f"{wrapper.display_name} generated {generation.num_generated}; "
                f"expected {args.num_samples}."
            )
        summary.update(
            generation_id=generation.generation_dir.name,
            num_samples=generation.num_generated,
            generated_graphs=str(generation.graphs_path.resolve()),
            generation_manifest=str(generation.manifest_path.resolve()),
            graphs_sha256=generation.graphs_sha256,
        )
        _status(
            model,
            f"generation completed: graphs={generation.graphs_path.resolve()}",
            enabled=progress_enabled,
        )

    return summary


def main(model_id: str, argv: Sequence[str] | None = None) -> int:
    model = normalize_baseline_id(model_id)
    parser = build_parser(model)
    args = parser.parse_args(argv)
    try:
        summary = run(model, args)
    except Exception as exc:
        _status(
            model,
            f"FAILED: {type(exc).__name__}: {exc}",
            enabled=not getattr(args, "quiet", False),
        )
        raise
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


__all__ = ["build_parser", "main", "run"]

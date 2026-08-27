"""GraphER-facing wrapper implementation for the external DeFoG project.

The wrapper supports generic and molecular graph datasets. DeFoG is invoked in an
isolated subprocess because its source tree uses un-namespaced imports that can
collide with GraphER. Training consumes GraphER's prepared NetworkX splits,
saves the trained model, and emits an independently sampled training-estimate
pool together with the exact ground-truth training batch. Generation delegates
to the validated neutral-export backend.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from grapher.models.artifacts import ArtifactLayout
from grapher.models.base import (
    BaselineCapabilities,
    BaseGeneratorWrapper,
    GenerateRequest,
    GenerationArtifacts,
    TrainRequest,
    TrainingArtifacts,
)
from grapher.models.errors import ArtifactCollisionError
from grapher.utils.subprocess_progress import SubprocessLogReporter

TRAINING_MANIFEST_FORMAT = "grapher_defog_training_v2"
TRAINING_ESTIMATES_MANIFEST_FORMAT = "grapher_defog_training_estimates_v1"
GENERATION_MANIFEST_FORMAT = "grapher_defog_generation_v1"
DEFOG_ROOT_ENV = "DEFOG"
DEFOG_PYTHON_ENV = "DEFOG_PYTHON"
SUPPORTED_GENERIC_DATASETS = frozenset({"comm20", "planar", "sbm", "tree"})
SUPPORTED_MOLECULAR_DATASETS = frozenset({"qm9", "zinc"})
SUPPORTED_NATIVE_DATASETS = SUPPORTED_GENERIC_DATASETS | SUPPORTED_MOLECULAR_DATASETS
_NATIVE_DATASET_BY_BENCHMARK = {
    "community_small": "comm20",
    # The attached source has no Ego-specific Hydra profile. Its generic
    # tensor loader is data-driven, so Ego-small uses the comm20 compatibility
    # profile while the report-facing benchmark identity remains ego_small.
    "ego_small": "comm20",
    "comm20": "comm20",
    "sbm": "sbm",
    "planar": "planar",
    "tree": "tree",
    "qm9": "qm9",
    "qm9_attributed": "qm9",
    "zinc": "zinc",
    "zinc_attributed": "zinc",
}
_DEFAULT_EXPERIMENT_BY_NATIVE_DATASET = {
    "qm9": "qm9_no_h",
    "zinc": "zinc",
}
_SAFE_OVERRIDE = re.compile(r"^[^\x00\r\n]+$")
_PROTECTED_OVERRIDE_KEYS = frozenset(
    {
        "experiment",
        "dataset",
        "dataset.datadir",
        "general.name",
        "general.gpus",
        "general.resume",
        "general.test_only",
        "general.generated_path",
        "hydra.run.dir",
        "train.seed",
        "train.n_epochs",
        "general.check_val_every_n_epochs",
        "general.sample_every_val",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_pickle(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{label} must contain a JSON object: {path}")
    return value


def _verify_prepared_sources_unchanged(
    prepared_splits: Mapping[str, Any],
    split_paths: Mapping[str, Path],
) -> None:
    """Ensure the model input and published target refer to identical bytes."""

    for split, path in split_paths.items():
        record = prepared_splits.get(split)
        if not isinstance(record, Mapping):
            raise RuntimeError(f"Missing prepared split record for {split!r}.")
        source = record.get("source")
        if not isinstance(source, Mapping) or not source.get("sha256"):
            raise RuntimeError(
                f"Prepared split {split!r} has no source SHA-256 provenance."
            )
        expected = str(source["sha256"])
        observed = _sha256(path)
        if observed != expected:
            raise RuntimeError(
                f"GraphER dataset split {split!r} changed while DeFoG was "
                f"training: expected {expected}, observed {observed}."
            )


def _verify_managed_generation_assets(
    layout: ArtifactLayout,
    training_manifest: Mapping[str, Any],
) -> None:
    """Verify the persisted data/config that determine checkpoint sampling."""

    conversion_path = layout.train_dir / "dataset_conversion.json"
    conversion = _read_json_object(
        conversion_path,
        label="DeFoG dataset-conversion manifest",
    )
    splits = conversion.get("splits")
    if not isinstance(splits, Mapping):
        raise RuntimeError("DeFoG conversion manifest has no split records.")
    native_root = layout.native_training_dataset_dir.resolve()
    for split, raw_record in splits.items():
        if not isinstance(raw_record, Mapping):
            raise RuntimeError(f"Invalid DeFoG conversion record for {split!r}.")
        for field in ("output", "model_view"):
            artifact = raw_record.get(field)
            if artifact is None:
                continue
            if not isinstance(artifact, Mapping):
                raise RuntimeError(
                    f"Invalid {field} record for DeFoG split {split!r}."
                )
            relative = Path(str(artifact.get("path", "")))
            if relative.is_absolute():
                raise RuntimeError(
                    f"Managed DeFoG {field} path must be relative: {relative}."
                )
            candidate = (layout.train_dir / relative).resolve()
            try:
                candidate.relative_to(native_root)
            except ValueError as exc:
                raise RuntimeError(
                    f"Managed DeFoG {field} path escapes native_dataset: "
                    f"{candidate}."
                ) from exc
            if not candidate.is_file():
                raise FileNotFoundError(
                    f"Missing persisted DeFoG {field} artifact: {candidate}."
                )
            expected_hash = str(artifact.get("sha256", ""))
            if not expected_hash or _sha256(candidate) != expected_hash:
                raise RuntimeError(
                    f"Persisted DeFoG {field} artifact changed after training: "
                    f"{candidate}."
                )

    resolved_record = training_manifest.get("resolved_config")
    if not isinstance(resolved_record, Mapping):
        raise RuntimeError("DeFoG training manifest has no resolved-config record.")
    expected_config_hash = str(resolved_record.get("sha256", ""))
    if (
        not layout.resolved_training_config_path.is_file()
        or not expected_config_hash
        or _sha256(layout.resolved_training_config_path) != expected_config_hash
    ):
        raise RuntimeError(
            "The persisted DeFoG resolved training configuration changed after "
            "training."
        )
    statistics_record = training_manifest.get("molecular_statistics")
    if statistics_record is not None:
        if not isinstance(statistics_record, Mapping):
            raise RuntimeError("Invalid molecular-statistics training record.")
        relative = Path(str(statistics_record.get("path", "")))
        if relative.is_absolute():
            raise RuntimeError("Managed molecular-statistics path must be relative.")
        statistics_path = (layout.train_dir / relative).resolve()
        try:
            statistics_path.relative_to(layout.train_dir.resolve())
        except ValueError as exc:
            raise RuntimeError(
                "Managed molecular-statistics path escapes the training directory."
            ) from exc
        expected_hash = str(statistics_record.get("sha256", ""))
        if (
            not statistics_path.is_file()
            or not expected_hash
            or _sha256(statistics_path) != expected_hash
        ):
            raise RuntimeError(
                "The persisted DeFoG molecular-statistics record changed after "
                "training."
            )
        statistics = _read_json_object(
            statistics_path,
            label="DeFoG molecular-statistics record",
        )
        if str(statistics.get("distribution_sha256", "")) != str(
            statistics_record.get("distribution_sha256", "")
        ):
            raise RuntimeError(
                "The persisted molecular-prior digest does not match the "
                "training manifest."
            )


def _mapping_option(value: Any, *, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    return dict(value)


def _boolean_option(value: Any, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a YAML/Python boolean.")
    return value


def _runtime_progress_options(runtime: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the optional terminal-progress settings for DeFoG workers."""

    raw = runtime.get("progress", {}) or {}
    if not isinstance(raw, Mapping):
        raise TypeError("runtime.progress must be a mapping.")
    enabled = _boolean_option(
        raw.get("enabled", False),
        name="runtime.progress.enabled",
    )
    stream_output = _boolean_option(
        raw.get("stream_output", enabled),
        name="runtime.progress.stream_output",
    )
    interval_seconds = float(raw.get("interval_seconds", 30.0))
    if interval_seconds <= 0:
        raise ValueError("runtime.progress.interval_seconds must be positive.")
    epoch_interval_value = raw.get("epoch_interval")
    epoch_interval = (
        None if epoch_interval_value is None else int(epoch_interval_value)
    )
    if epoch_interval is not None and epoch_interval <= 0:
        raise ValueError("runtime.progress.epoch_interval must be positive.")
    generation_batch_interval = int(
        raw.get("generation_batch_interval", 1)
    )
    if generation_batch_interval <= 0:
        raise ValueError(
            "runtime.progress.generation_batch_interval must be positive."
        )
    return {
        "enabled": enabled,
        "stream_output": stream_output,
        "interval_seconds": interval_seconds,
        "epoch_interval": epoch_interval,
        "generation_batch_interval": generation_batch_interval,
    }


def _emit_progress(progress: Mapping[str, Any], message: str) -> None:
    if bool(progress.get("enabled")):
        print(f"[GraphER/DeFoG] {message}", file=sys.stderr, flush=True)


def _deep_update(base: dict[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            base[key] = _deep_update(dict(base[key]), value)
        else:
            base[key] = value
    return base


def _load_options(request: TrainRequest) -> dict[str, Any]:
    values: dict[str, Any] = {}
    if request.config_path is not None:
        if not request.config_path.is_file():
            raise FileNotFoundError(f"Missing DeFoG wrapper config: {request.config_path}")
        loaded = yaml.safe_load(request.config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, Mapping):
            raise TypeError("The DeFoG wrapper config must contain a mapping.")
        selected = loaded.get("defog", loaded)
        if not isinstance(selected, Mapping):
            raise TypeError("The defog config section must contain a mapping.")
        values = dict(selected)
    return _deep_update(values, request.options)


def _native_dataset(benchmark_id: str, explicit: Any = None) -> str:
    requested = str(explicit or benchmark_id).lower()
    value = _NATIVE_DATASET_BY_BENCHMARK.get(requested, requested)
    if value not in SUPPORTED_NATIVE_DATASETS:
        raise ValueError(
            f"DeFoGWrapper supports native datasets "
            f"{sorted(SUPPORTED_NATIVE_DATASETS)}; benchmark {benchmark_id!r} "
            "has no declared compatibility profile."
        )
    return value


def _default_experiment(native_dataset: str) -> str:
    return _DEFAULT_EXPERIMENT_BY_NATIVE_DATASET.get(native_dataset, native_dataset)


def _dataset_profile(benchmark_id: str, native_dataset: str) -> dict[str, str]:
    domain = (
        "molecular" if native_dataset in SUPPORTED_MOLECULAR_DATASETS else "generic"
    )
    if benchmark_id == "ego_small" and native_dataset == "comm20":
        role = "generic_loader_compatibility_profile"
    elif benchmark_id != native_dataset:
        role = "declared_native_alias"
    else:
        role = "native"
    if native_dataset == "zinc":
        representation = "kekule_no_aromatic_class"
    elif native_dataset == "qm9":
        representation = "heavy_atom_categorical"
    else:
        representation = "simple_undirected_topology"
    return {
        "domain": domain,
        "profile_role": role,
        "model_representation": representation,
    }


def _prepare_worker_path(native_dataset: str) -> Path:
    filename = (
        "prepare_molecular_dataset.py"
        if native_dataset in SUPPORTED_MOLECULAR_DATASETS
        else "prepare_dataset.py"
    )
    path = Path(__file__).resolve().parent / "workers" / filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing DeFoG dataset worker: {path}")
    return path


def _training_entrypoint(defog_root: Path, native_dataset: str) -> Path:
    if native_dataset not in SUPPORTED_NATIVE_DATASETS:
        raise ValueError(f"Unsupported DeFoG training dataset {native_dataset!r}.")
    if not (defog_root / "src" / "main.py").is_file():
        raise FileNotFoundError(f"Missing upstream DeFoG entrypoint under {defog_root}.")
    path = Path(__file__).resolve().parent / "workers" / "train.py"
    if not path.is_file():
        raise FileNotFoundError(f"Missing DeFoG training entrypoint: {path}")
    return path


def _source_revision(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
            check=False,
            shell=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    revision = result.stdout.strip()
    return revision if result.returncode == 0 and revision else None


def _source_fingerprint(root: Path) -> dict[str, Any]:
    """Hash DeFoG code/config files when an archive has no Git revision."""

    candidates: list[Path] = []
    for directory in (root / "configs", root / "src"):
        if not directory.is_dir():
            continue
        candidates.extend(
            path
            for path in directory.rglob("*")
            if path.is_file()
            and path.suffix.lower() in {".py", ".yaml", ".yml", ".cpp", ".h"}
        )
    digest = hashlib.sha256()
    for path in sorted(candidates, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return {
        "algorithm": "sha256_names_and_contents_v1",
        "sha256": digest.hexdigest(),
        "file_count": len(candidates),
        "scope": "configs/**/*.{yaml,yml} and src/**/*.{py,cpp,h}",
    }


def _source_identity(root: Path) -> dict[str, Any]:
    return {
        "source_root": str(root),
        "revision": _source_revision(root),
        "source_fingerprint": _source_fingerprint(root),
    }


def _verify_source_identity(
    expected: Mapping[str, Any],
    observed: Mapping[str, Any],
) -> None:
    expected_fingerprint = expected.get("source_fingerprint")
    observed_fingerprint = observed.get("source_fingerprint")
    if not isinstance(expected_fingerprint, Mapping) or not isinstance(
        observed_fingerprint, Mapping
    ):
        raise RuntimeError("DeFoG source fingerprint provenance is incomplete.")
    if str(expected_fingerprint.get("sha256", "")) != str(
        observed_fingerprint.get("sha256", "")
    ):
        raise RuntimeError(
            "The external DeFoG source tree differs from the code used for "
            "training. Use a new run_id after changing upstream code."
        )
    expected_revision = expected.get("revision")
    observed_revision = observed.get("revision")
    if (
        expected_revision is not None
        and observed_revision is not None
        and str(expected_revision) != str(observed_revision)
    ):
        raise RuntimeError(
            "The external DeFoG Git revision differs from the training run."
        )


def _python_environment_identity(python_executable: str) -> dict[str, Any]:
    """Fingerprint the isolated interpreter and its installed distributions."""

    completed = subprocess.run(
        [python_executable, "-m", "pip", "freeze", "--all"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=60,
        check=False,
        shell=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Could not fingerprint the DeFoG Python environment: "
            f"{completed.stderr.strip()}"
        )
    packages = sorted(
        line.strip()
        for line in completed.stdout.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    payload = "\n".join(packages).encode("utf-8")
    return {
        "python_executable": str(Path(python_executable).resolve()),
        "pip_freeze_sha256": hashlib.sha256(payload).hexdigest(),
        "distribution_count": len(packages),
        "distributions": packages,
    }


def _verify_python_environment_identity(
    expected: Mapping[str, Any],
    observed: Mapping[str, Any],
) -> None:
    if str(expected.get("python_executable", "")) != str(
        observed.get("python_executable", "")
    ):
        raise RuntimeError(
            "Managed DeFoG generation resolved a different Python interpreter "
            "from training."
        )
    if str(expected.get("pip_freeze_sha256", "")) != str(
        observed.get("pip_freeze_sha256", "")
    ):
        raise RuntimeError(
            "The DeFoG Python environment changed after training. Use a new "
            "run_id or restore the recorded dependencies."
        )


def generate_defog_graphs(*args: Any, **kwargs: Any) -> Any:
    """Lazily delegate to the validated backend.

    Keeping this import inside the call lets the registry inspect the wrapper
    without importing NumPy, NetworkX, Torch, or any upstream dependency.
    """

    from grapher.models.defog.backend import generate_defog_graphs as generate

    return generate(*args, **kwargs)


def _external_environment(
    defog_root: Path,
    *,
    seed: int,
    cuda_visible_devices: str | None = None,
) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join([str(defog_root), str(defog_root / "src")]),
            "PYTHONHASHSEED": str(seed),
            "WANDB_MODE": "disabled",
            "WANDB_DISABLED": "true",
            "HYDRA_FULL_ERROR": "1",
            "MPLBACKEND": "Agg",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "PYTHONUNBUFFERED": "1",
        }
    )
    if cuda_visible_devices is not None:
        environment["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    return environment


def _publish_directory(stage: Path, target: Path, *, overwrite: bool) -> None:
    ArtifactLayout.require_available(target, overwrite=overwrite)
    target.parent.mkdir(parents=True, exist_ok=True)
    backup: Path | None = None
    if target.exists():
        backup = target.with_name(f".{target.name}.backup-{time.time_ns()}")
        target.replace(backup)
    try:
        stage.replace(target)
    except Exception:
        if backup is not None and backup.exists() and not target.exists():
            backup.replace(target)
        raise
    if backup is not None:
        shutil.rmtree(backup)


def _find_checkpoint(native_run: Path) -> Path:
    checkpoints = [path for path in native_run.rglob("*.ckpt") if path.is_file()]
    if not checkpoints:
        raise RuntimeError(
            "DeFoG training finished without a checkpoint. Ensure train.save_model=true "
            "and that the checkpoint interval does not exceed the training horizon."
        )

    def key(path: Path) -> tuple[int, int, int, str]:
        match = re.search(r"epoch[=-](\d+)", path.name)
        epoch = int(match.group(1)) if match else -1
        # The GraphER training shim writes this artifact only after a successful
        # Trainer.fit(), independently of DeFoG's periodic validation cadence.
        is_explicit_final = int(path.name == "grapher_final.ckpt")
        return (is_explicit_final, epoch, path.stat().st_mtime_ns, str(path))

    return max(checkpoints, key=key)


def _checkpoint_epoch(path: Path) -> int | None:
    match = re.search(r"epoch[=-](\d+)", path.name)
    return int(match.group(1)) if match else None


def _log_tail(path: Path, *, lines: int = 200) -> str:
    if not path.is_file():
        return ""
    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(content[-lines:])


def _external_failure_classification(output: str) -> str | None:
    lowered = output.lower()
    if (
        "nvml_error_lib_rm_version_mismatch" in lowered
        or (
            "driver/library version mismatch" in lowered
            and any(token in lowered for token in ("nvml", "nvidia-smi", "nvrm"))
        )
    ):
        return "gpu_driver_library_mismatch"
    if "cuda out of memory" in lowered or "outofmemoryerror" in lowered:
        return "cuda_out_of_memory"
    if "torch.cuda.is_available() == false" in lowered:
        return "cuda_unavailable"
    if "nccl" in lowered:
        return "distributed_runtime_failure"
    return None


def _external_failure_hint(output: str) -> str | None:
    """Explain common isolated-runtime failures without hiding the raw log."""

    classification = _external_failure_classification(output)
    if classification == "gpu_driver_library_mismatch":
        return (
            "Detected an NVIDIA NVML driver/library mismatch. GraphER disables "
            "DDP/NCCL for one-device DeFoG training, but the host driver stack "
            "may still need attention. Run `nvidia-smi`; if it reports the "
            "same mismatch, reboot after the driver update or ask the server "
            "administrator to reconcile the loaded kernel module and user-space "
            "NVIDIA libraries."
        )
    if classification == "distributed_runtime_failure":
        return (
            "Detected an NCCL/distributed-runtime failure. A one-device run "
            "should log `Disabled one-device DDP` before Trainer starts. Confirm "
            "that grapher.models.defog.workers.train is the updated worker "
            "and inspect the saved runtime diagnostics."
        )
    if classification == "cuda_unavailable":
        return (
            "The DeFoG interpreter cannot access CUDA. Verify DEFOG_PYTHON, "
            "CUDA_VISIBLE_DEVICES, the installed PyTorch CUDA build, and the "
            "driver reported by `nvidia-smi`; or explicitly set runtime.gpus: 0."
        )
    if classification == "cuda_out_of_memory":
        return (
            "CUDA ran out of memory. Reduce the DeFoG training batch size or "
            "select a less occupied GPU with runtime.cuda_visible_devices."
        )
    return None


def _debug_environment(environment: Mapping[str, str]) -> dict[str, str]:
    """Return only non-secret runtime selectors that help reproduce failures."""

    keys = (
        "GRAPHER_DEFOG_DATASET",
        "GRAPHER_DEFOG_REQUESTED_GPUS",
        "GRAPHER_DEFOG_DIAGNOSTICS_PATH",
        "GRAPHER_DEFOG_PROGRESS_ENABLED",
        "GRAPHER_DEFOG_PROGRESS_INTERVAL_SECONDS",
        "GRAPHER_DEFOG_EPOCH_PROGRESS_INTERVAL",
        "GRAPHER_DEFOG_GENERATION_PROGRESS_INTERVAL",
        "CUDA_VISIBLE_DEVICES",
        "CUDA_DEVICE_ORDER",
        "NCCL_DEBUG",
        "NCCL_P2P_DISABLE",
        "NCCL_IB_DISABLE",
        "CONDA_DEFAULT_ENV",
        "CONDA_PREFIX",
        "VIRTUAL_ENV",
        "PYTHONPATH",
        "PATH",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "PYTHONHASHSEED",
        "HYDRA_FULL_ERROR",
    )
    return {key: environment[key] for key in keys if key in environment}


def _preserve_training_failure(
    *,
    layout: ArtifactLayout,
    log_path: Path,
    diagnostics_path: Path,
    error: BaseException,
    started_at: str,
    commands: Mapping[str, Sequence[str] | None],
    native_run: Path,
    working_directory: Path,
    environment: Mapping[str, str],
) -> Path:
    """Persist bounded failure evidence outside the disposable staging tree."""

    attempt = f"attempt-{time.time_ns()}"
    target = layout.run_dir / "failures" / attempt
    target.mkdir(parents=True, exist_ok=False)
    if log_path.is_file():
        shutil.copy2(log_path, target / "train.log")
    if diagnostics_path.is_file():
        shutil.copy2(diagnostics_path, target / "runtime_diagnostics.json")
    hydra_source = native_run / ".hydra"
    hydra_files: list[str] = []
    if hydra_source.is_dir():
        hydra_target = target / "hydra"
        hydra_target.mkdir()
        for name in ("config.yaml", "overrides.yaml", "hydra.yaml"):
            source = hydra_source / name
            if source.is_file():
                shutil.copy2(source, hydra_target / name)
                hydra_files.append(f"hydra/{name}")
    _atomic_json(
        target / "failure.json",
        {
            "format": "grapher_defog_training_failure_v1",
            "model_id": "defog",
            "dataset_id": layout.dataset_id,
            "run_id": layout.run_id,
            "started_at": started_at,
            "failed_at": _utc_now(),
            "exception_type": type(error).__name__,
            "exception": str(error),
            "failure_classification": _external_failure_classification(
                str(error)
            ),
            "diagnostic_hint": _external_failure_hint(str(error)),
            "commands": {
                name: list(command) if command is not None else None
                for name, command in commands.items()
            },
            "working_directory": str(working_directory.resolve()),
            "environment": _debug_environment(environment),
            "hydra": hydra_files,
            "log": "train.log" if log_path.is_file() else None,
            "runtime_diagnostics": (
                "runtime_diagnostics.json" if diagnostics_path.is_file() else None
            ),
        },
    )
    return target


class DeFoGWrapper(BaseGeneratorWrapper):
    """Train and sample DeFoG through isolated subprocesses."""

    model_id = "defog"
    display_name = "DeFoG"
    capabilities = BaselineCapabilities(
        domains=frozenset({"generic", "attributed"}),
        isolation="subprocess",
        status="ready",
    )
    implementation_note = (
        "Training and generation support generic graphs plus heavy-atom QM9 "
        "and ZINC molecular graphs through strict attributed adapters."
    )

    def _run_external(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        environment: Mapping[str, str],
        log_path: Path,
        timeout_seconds: float | None,
        label: str,
        progress_enabled: bool = False,
        stream_output: bool = False,
        progress_interval_seconds: float = 30.0,
    ) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(
                f"\n[{label}] started_at={_utc_now()}\n"
                f"[{label}] cwd={cwd.resolve()}\n"
                f"[{label}] argv={json.dumps(list(command))}\n"
                f"[{label}] environment="
                f"{json.dumps(_debug_environment(environment), sort_keys=True)}\n"
            )
            log_file.flush()
            reporter = SubprocessLogReporter(
                label=label,
                log_path=log_path,
                enabled=progress_enabled,
                stream_output=stream_output,
                interval_seconds=progress_interval_seconds,
            )
            reporter.start(start_offset=log_path.stat().st_size)
            try:
                completed = subprocess.run(
                    list(command),
                    cwd=str(cwd),
                    env=dict(environment),
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    timeout=timeout_seconds,
                    check=False,
                    shell=False,
                    start_new_session=True,
                )
            except subprocess.TimeoutExpired as exc:
                log_file.flush()
                reporter.stop(status="timed out")
                tail = _log_tail(log_path)
                raise RuntimeError(
                    f"{label} timed out.\n"
                    f"Working directory: {cwd.resolve()}\n"
                    f"Subprocess log: {log_path.resolve()}\n"
                    f"Command: {json.dumps(list(command))}\n"
                    f"Last subprocess output:\n{tail}"
                ) from exc
            except BaseException:
                log_file.flush()
                reporter.stop(status="failed")
                raise
            else:
                log_file.flush()
                reporter.stop(
                    status=(
                        "completed"
                        if completed.returncode == 0
                        else f"failed with exit code {completed.returncode}"
                    )
                )
        if completed.returncode != 0:
            tail = _log_tail(log_path)
            classification = _external_failure_classification(tail)
            hint = _external_failure_hint(tail)
            raise RuntimeError(
                f"{label} exited with code {completed.returncode}.\n"
                + (
                    f"Failure classification: {classification}\n"
                    if classification
                    else ""
                )
                + f"Working directory: {cwd.resolve()}\n"
                f"Subprocess log: {log_path.resolve()}\n"
                f"Command: {json.dumps(list(command))}\n"
                + (f"Diagnostic hint: {hint}\n" if hint else "")
                + f"Last subprocess output:\n{tail}"
            )

    def train(self, request: TrainRequest) -> TrainingArtifacts:
        self.validate_train_request(request)
        options = _load_options(request)
        native = _native_dataset(
            request.dataset.benchmark_id,
            options.get("native_dataset") or request.dataset.native_id,
        )
        from grapher.models.defog.runtime import (
            resolve_defog_python,
            resolve_defog_root,
        )

        experiment = str(
            options.get("experiment", _default_experiment(native))
        ).lower()
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", experiment):
            raise ValueError("DeFoG experiment must be a safe identifier.")

        source_env = str(options.get("source_env", DEFOG_ROOT_ENV))
        python_env = str(options.get("python_env", DEFOG_PYTHON_ENV))
        defog_root = resolve_defog_root(source_env)
        runtime_cfg = dict(options.get("runtime", {}) or {})
        progress = _runtime_progress_options(runtime_cfg)
        python_executable = resolve_defog_python(
            defog_root=defog_root,
            python_executable=runtime_cfg.get("python_executable"),
            python_env=python_env,
        )
        _emit_progress(
            progress,
            "training preflight: "
            f"benchmark={request.dataset.benchmark_id}, native={native}, "
            f"run_id={request.run.run_id}, seed={request.run.train_seed}, "
            f"defog_root={defog_root}, python={python_executable}",
        )
        source_identity = _source_identity(defog_root)
        python_environment = _python_environment_identity(python_executable)
        timeout_value = runtime_cfg.get("timeout_seconds")
        timeout_seconds = None if timeout_value is None else float(timeout_value)
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("runtime.timeout_seconds must be positive.")
        gpus = int(runtime_cfg.get("gpus", 1))
        if gpus not in {0, 1}:
            raise ValueError(
                "runtime.gpus must be 0 or 1; the attached DeFoG training code "
                "does not implement multi-GPU execution."
            )
        cuda_visible_devices_value = runtime_cfg.get("cuda_visible_devices")
        cuda_visible_devices = (
            None
            if cuda_visible_devices_value is None
            else str(cuda_visible_devices_value)
        )
        if cuda_visible_devices is not None and (
            not cuda_visible_devices
            or any(character in cuda_visible_devices for character in "\x00\r\n")
        ):
            raise ValueError("runtime.cuda_visible_devices is invalid.")
        estimate_options = _mapping_option(
            options.get("training_estimates"),
            name="training_estimates",
        )
        estimates_enabled = _boolean_option(
            estimate_options.get("enabled", True),
            name="training_estimates.enabled",
        )
        required_configs = (
            defog_root / "configs" / "experiment" / f"{experiment}.yaml",
            defog_root / "configs" / "dataset" / f"{native}.yaml",
        )
        missing_configs = [str(path) for path in required_configs if not path.is_file()]
        if missing_configs:
            raise FileNotFoundError(
                "The DeFoG source does not provide the requested configuration: "
                f"{missing_configs}."
            )

        layout = request.run.layout
        if (
            request.overwrite
            and layout.generations_dir.is_dir()
            and any(layout.generations_dir.iterdir())
        ):
            raise ArtifactCollisionError(
                "Cannot overwrite a trained DeFoG run that already has generated "
                "batches. Use a new run_id so existing raw batches remain tied "
                "to their original checkpoint."
            )
        ArtifactLayout.require_available(layout.train_dir, overwrite=request.overwrite)
        staging_root = layout.output_root.expanduser().resolve() / ".staging"
        staging_root.mkdir(parents=True, exist_ok=True)
        workspace = Path(tempfile.mkdtemp(prefix="defog-train-", dir=staging_root))
        stage_train = workspace / "train"
        stage_train.mkdir()
        log_path = stage_train / "train.log"
        runtime_diagnostics_path = stage_train / "runtime_diagnostics.json"
        native_data = workspace / "native_dataset"
        native_run = workspace / "native_run"
        molecular_statistics_capture = workspace / "molecular_statistics.json"
        preparation_manifest = stage_train / "dataset_conversion.json"
        started_at = _utc_now()
        started = time.monotonic()
        prepare_command: list[str] | None = None
        command: list[str] | None = None
        environment: dict[str, str] = {}
        try:
            splits = request.dataset.split_paths
            prepare_command = [
                python_executable,
                str(_prepare_worker_path(native)),
                "--dataset",
                native,
                "--output-root",
                str(native_data),
                "--manifest",
                str(preparation_manifest),
                "--train",
                str(splits["train"].resolve()),
                "--val",
                str(splits["val"].resolve()),
                "--test",
                str(splits["test"].resolve()),
            ]
            environment = _external_environment(
                defog_root,
                seed=request.run.train_seed,
                cuda_visible_devices=cuda_visible_devices,
            )
            environment["GRAPHER_DEFOG_DATASET"] = native
            environment["GRAPHER_DEFOG_REQUESTED_GPUS"] = str(gpus)
            environment["GRAPHER_DEFOG_DIAGNOSTICS_PATH"] = str(
                runtime_diagnostics_path
            )
            environment["GRAPHER_DEFOG_PROGRESS_ENABLED"] = (
                "1" if progress["enabled"] else "0"
            )
            environment["GRAPHER_DEFOG_PROGRESS_INTERVAL_SECONDS"] = str(
                progress["interval_seconds"]
            )
            environment["GRAPHER_DEFOG_GENERATION_PROGRESS_INTERVAL"] = str(
                progress["generation_batch_interval"]
            )
            if progress["epoch_interval"] is not None:
                environment["GRAPHER_DEFOG_EPOCH_PROGRESS_INTERVAL"] = str(
                    progress["epoch_interval"]
                )
            if native in SUPPORTED_MOLECULAR_DATASETS:
                environment["GRAPHER_DEFOG_STATISTICS_PATH"] = str(
                    molecular_statistics_capture
                )
            _emit_progress(
                progress,
                "preparing DeFoG-native dataset tensors from the immutable "
                "GraphER train/validation/test splits",
            )
            self._run_external(
                prepare_command,
                cwd=defog_root / "src",
                environment=environment,
                log_path=log_path,
                timeout_seconds=timeout_seconds,
                label="DeFoG dataset preparation",
                progress_enabled=progress["enabled"],
                stream_output=progress["stream_output"],
                progress_interval_seconds=progress["interval_seconds"],
            )
            if not preparation_manifest.is_file():
                raise RuntimeError("DeFoG dataset worker did not publish its manifest.")
            preparation = _read_json_object(
                preparation_manifest,
                label="DeFoG dataset-conversion manifest",
            )
            prepared_splits = preparation.get("splits")
            if not isinstance(prepared_splits, Mapping):
                raise RuntimeError(
                    "DeFoG dataset-conversion manifest has no split records."
                )
            train_record = prepared_splits.get("train")
            if not isinstance(train_record, Mapping):
                raise RuntimeError(
                    "DeFoG dataset-conversion manifest has no train record."
                )
            train_graph_count = int(train_record.get("graph_count", 0))
            if train_graph_count <= 0:
                raise RuntimeError(
                    "DeFoG dataset conversion reported an empty training split."
                )
            _emit_progress(
                progress,
                f"dataset preparation complete: train_graphs={train_graph_count}",
            )

            upstream_name = f"grapher_{request.run.run_id}"
            command = [
                python_executable,
                str(_training_entrypoint(defog_root, native).resolve()),
                f"+experiment={experiment}",
                f"dataset={native}",
                f"dataset.datadir={native_data.resolve()}",
                f"general.name={upstream_name}",
                "general.wandb=disabled",
                f"general.gpus={gpus}",
                f"train.seed={request.run.train_seed}",
                "train.save_model=true",
                f"hydra.run.dir={native_run.resolve()}",
            ]
            if request.resume_from is not None:
                if not request.resume_from.is_file():
                    raise FileNotFoundError(f"Missing resume checkpoint: {request.resume_from}")
                command.append(f"general.resume={request.resume_from.resolve()}")

            known_overrides = {
                "n_epochs": "train.n_epochs",
                "batch_size": "train.batch_size",
                "num_workers": "train.num_workers",
                "check_val_every_n_epochs": "general.check_val_every_n_epochs",
                "sample_every_val": "general.sample_every_val",
            }
            for option_name, hydra_name in known_overrides.items():
                if option_name in options:
                    command.append(f"{hydra_name}={options[option_name]}")
            arbitrary = options.get("hydra_overrides", []) or []
            if isinstance(arbitrary, (str, bytes)) or not isinstance(arbitrary, Sequence):
                raise TypeError("hydra_overrides must be a sequence of strings.")
            for raw_override in arbitrary:
                override = str(raw_override)
                if _SAFE_OVERRIDE.fullmatch(override) is None or "=" not in override:
                    raise ValueError(f"Invalid Hydra override: {override!r}.")
                override_key = override.split("=", 1)[0].lstrip("+")
                if override_key in _PROTECTED_OVERRIDE_KEYS:
                    raise ValueError(f"Hydra override is controlled by the wrapper: {override}")
                command.append(override)

            configured_horizon = options.get("n_epochs", "upstream default")
            _emit_progress(
                progress,
                "launching upstream DeFoG training: "
                f"experiment={experiment}, gpus={gpus}, "
                f"train.n_epochs={configured_horizon}",
            )
            self._run_external(
                command,
                cwd=defog_root / "src",
                environment=environment,
                log_path=log_path,
                timeout_seconds=timeout_seconds,
                label="DeFoG training",
                progress_enabled=progress["enabled"],
                stream_output=progress["stream_output"],
                progress_interval_seconds=progress["interval_seconds"],
            )
            _emit_progress(progress, "upstream DeFoG training process completed")
            if not runtime_diagnostics_path.is_file():
                raise RuntimeError(
                    "DeFoG training worker did not publish runtime diagnostics."
                )
            runtime_diagnostics = _read_json_object(
                runtime_diagnostics_path,
                label="DeFoG runtime diagnostics",
            )
            if runtime_diagnostics.get("format") != (
                "grapher_defog_runtime_diagnostics_v1"
            ):
                raise RuntimeError(
                    "DeFoG training worker published unsupported runtime "
                    "diagnostics."
                )
            _verify_prepared_sources_unchanged(prepared_splits, splits)
            molecular_statistics_record: dict[str, Any] | None = None
            molecular_statistics_path: Path | None = None
            if native in SUPPORTED_MOLECULAR_DATASETS:
                if not molecular_statistics_capture.is_file():
                    raise RuntimeError(
                        "DeFoG molecular training did not publish empirical "
                        "dataset statistics."
                    )
                molecular_statistics_record = _read_json_object(
                    molecular_statistics_capture,
                    label="DeFoG molecular-statistics record",
                )
                if (
                    molecular_statistics_record.get("format")
                    != "grapher_defog_molecular_statistics_v1"
                    or str(molecular_statistics_record.get("dataset", ""))
                    != native
                    or not molecular_statistics_record.get(
                        "distribution_sha256"
                    )
                ):
                    raise RuntimeError(
                        "DeFoG molecular training published invalid empirical "
                        "statistics metadata."
                    )
                molecular_statistics_path = (
                    stage_train / "molecular_statistics.json"
                )
                shutil.copy2(
                    molecular_statistics_capture,
                    molecular_statistics_path,
                )
            source_checkpoint = _find_checkpoint(native_run)
            _emit_progress(
                progress,
                "selected trained checkpoint: "
                f"{source_checkpoint.relative_to(native_run)}",
            )
            checkpoint_path = stage_train / "checkpoints" / "model.ckpt"
            checkpoint_path.parent.mkdir(parents=True)
            shutil.copy2(source_checkpoint, checkpoint_path)
            final_checkpoint_record: dict[str, Any] | None = None
            final_checkpoint_record_path: Path | None = None
            if source_checkpoint.name == "grapher_final.ckpt":
                source_record = source_checkpoint.with_suffix(".json")
                if not source_record.is_file():
                    raise RuntimeError(
                        "The explicit final DeFoG checkpoint has no completion "
                        f"record: {source_record}."
                    )
                final_checkpoint_record = _read_json_object(
                    source_record,
                    label="DeFoG final-checkpoint record",
                )
                if final_checkpoint_record.get("format") != (
                    "grapher_defog_final_checkpoint_v1"
                ):
                    raise RuntimeError(
                        "Unsupported DeFoG final-checkpoint completion record."
                    )
                final_checkpoint_record_path = (
                    checkpoint_path.parent / "final_checkpoint.json"
                )
                shutil.copy2(source_record, final_checkpoint_record_path)

            native_config = native_run / ".hydra" / "config.yaml"
            resolved_config = stage_train / "resolved_config.yaml"
            if native_config.is_file():
                shutil.copy2(native_config, resolved_config)
            else:
                resolved_config.write_text(
                    yaml.safe_dump(_jsonable(options), sort_keys=False),
                    encoding="utf-8",
                )

            # Keep the DeFoG-formatted data associated with this checkpoint.
            # DeFoG reconstructs dataset metadata at sampling time, so using its
            # bundled default dataset here could silently mismatch a model that
            # was trained on GraphER's prepared splits.
            persisted_native_data = stage_train / "native_dataset"
            native_data.replace(persisted_native_data)
            # Hydra captured the temporary dataset path. Publish a usable
            # resolved configuration that points at this run's durable copy.
            resolved_text = resolved_config.read_text(encoding="utf-8")
            resolved_text = resolved_text.replace(
                str(native_data.resolve()),
                str(layout.native_training_dataset_dir.expanduser().resolve()),
            )
            resolved_config.write_text(resolved_text, encoding="utf-8")
            selected_checkpoint_epoch = _checkpoint_epoch(source_checkpoint)
            selected_checkpoint_is_final = (
                source_checkpoint.name == "grapher_final.ckpt"
            )
            resolved_values = yaml.safe_load(resolved_text) or {}
            configured_n_epochs: int | None = None
            if isinstance(resolved_values, Mapping):
                train_values = resolved_values.get("train")
                if isinstance(train_values, Mapping) and train_values.get(
                    "n_epochs"
                ) is not None:
                    configured_n_epochs = int(train_values["n_epochs"])
            if (
                configured_n_epochs is None
                and selected_checkpoint_is_final
                and final_checkpoint_record is not None
            ):
                configured_n_epochs = int(
                    final_checkpoint_record.get("configured_epochs", -1)
                )
                if configured_n_epochs <= 0:
                    raise RuntimeError(
                        "The DeFoG final-checkpoint record has no valid training "
                        "horizon."
                    )
            final_epoch_verified = False
            if (
                configured_n_epochs is not None
                and selected_checkpoint_epoch is None
                and not selected_checkpoint_is_final
            ):
                raise RuntimeError(
                    "Cannot verify that the selected DeFoG checkpoint is final "
                    f"because its filename has no epoch: {source_checkpoint.name}."
                )
            if configured_n_epochs is not None and selected_checkpoint_is_final:
                if final_checkpoint_record is None:
                    raise AssertionError("Missing explicit final-checkpoint record.")
                completed_epochs = int(
                    final_checkpoint_record.get("completed_epochs", -1)
                )
                recorded_horizon = int(
                    final_checkpoint_record.get("configured_epochs", -1)
                )
                selected_checkpoint_epoch = int(
                    final_checkpoint_record.get("selected_epoch", -1)
                )
                if (
                    completed_epochs != configured_n_epochs
                    or recorded_horizon != configured_n_epochs
                    or selected_checkpoint_epoch != configured_n_epochs - 1
                ):
                    raise RuntimeError(
                        "DeFoG final-checkpoint record does not match the "
                        f"resolved training horizon {configured_n_epochs}: "
                        f"{final_checkpoint_record}."
                    )
                final_epoch_verified = True
            elif (
                configured_n_epochs is not None
                and selected_checkpoint_epoch is not None
            ):
                expected_final_epoch = configured_n_epochs - 1
                if selected_checkpoint_epoch != expected_final_epoch:
                    raise RuntimeError(
                        "DeFoG did not save the final training epoch: selected "
                        f"epoch {selected_checkpoint_epoch}, expected "
                        f"{expected_final_epoch}. Set the checkpoint interval so "
                        "the final epoch is saved before using this run."
                    )
                final_epoch_verified = True
            for split_name, raw_record in prepared_splits.items():
                if not isinstance(raw_record, dict):
                    continue
                output_record = raw_record.get("output")
                if isinstance(output_record, dict):
                    raw_output_path = output_record.get("path")
                    try:
                        relative_output = Path(str(raw_output_path)).relative_to(
                            native_data
                        )
                    except (TypeError, ValueError):
                        relative_output = Path("unknown") / f"{split_name}.pt"
                    output_record["path"] = (
                        Path("native_dataset") / relative_output
                    ).as_posix()
            # Molecular conversion also records sentinel/model-view paths.
            # Normalize any remaining temporary native-data paths before the
            # conversion manifest is published.
            normalized_preparation = json.dumps(_jsonable(preparation), sort_keys=True)
            normalized_preparation = normalized_preparation.replace(
                str(native_data.resolve()),
                "native_dataset",
            )
            preparation = json.loads(normalized_preparation)
            prepared_splits = preparation.get("splits", prepared_splits)
            _atomic_json(preparation_manifest, preparation)

            training_estimates_summary: dict[str, Any] = {
                "enabled": estimates_enabled,
            }
            if estimates_enabled:
                from grapher.models.defog.backend import DeFoGGeneratorConfig

                estimate_count = int(
                    estimate_options.get(
                        "num_graphs",
                        (
                            min(train_graph_count, 1024)
                            if native in SUPPORTED_MOLECULAR_DATASETS
                            else train_graph_count
                        ),
                    )
                )
                if estimate_count <= 0:
                    raise ValueError(
                        "training_estimates.num_graphs must be positive."
                    )
                estimate_seed = int(
                    estimate_options.get("seed", request.run.train_seed)
                )
                estimate_sampling = _mapping_option(
                    estimate_options.get("sampling"),
                    name="training_estimates.sampling",
                )
                estimate_runtime = dict(runtime_cfg)
                estimate_runtime.update(
                    _mapping_option(
                        estimate_options.get("runtime"),
                        name="training_estimates.runtime",
                    )
                )
                estimate_runtime.setdefault("python_executable", python_executable)
                # Match DeFoG training's CUDA-if-available behavior.  Forcing
                # cuda merely because gpus=1 would make an otherwise valid CPU
                # fallback fail only after training had completed.
                estimate_runtime.setdefault("device", "auto" if gpus > 0 else "cpu")
                _emit_progress(
                    progress,
                    "generating the independent post-training source pool: "
                    f"num_graphs={estimate_count}, seed={estimate_seed}",
                )
                estimate_config = DeFoGGeneratorConfig.from_dict(
                    {
                        "type": "defog",
                        "backend": "subprocess",
                        "dataset": native,
                        "experiment": experiment,
                        "checkpoint_path": str(checkpoint_path),
                        "dataset_datadir": str(persisted_native_data),
                        "resolved_config_path": str(resolved_config),
                        "molecular_statistics_path": (
                            str(molecular_statistics_path)
                            if molecular_statistics_path is not None
                            else None
                        ),
                        "source_env": source_env,
                        "python_env": python_env,
                        "sampling": estimate_sampling,
                        "runtime": estimate_runtime,
                    }
                )
                estimate_dir = stage_train / "training_estimates"
                native_estimate_dir = estimate_dir / "native"
                native_estimate_dir.mkdir(parents=True)
                estimate_result = generate_defog_graphs(
                    estimate_config,
                    num_graphs=estimate_count,
                    seed=estimate_seed,
                    output_dir=native_estimate_dir,
                )
                if len(estimate_result.graphs) != estimate_count:
                    raise RuntimeError(
                        "DeFoG returned "
                        f"{len(estimate_result.graphs)} post-training samples; "
                        f"expected {estimate_count}."
                    )
                _emit_progress(
                    progress,
                    "post-training source-pool generation complete: "
                    f"generated={len(estimate_result.graphs)}",
                )
                if native in SUPPORTED_MOLECULAR_DATASETS:
                    estimate_runtime_record = estimate_result.manifest.get("runtime")
                    if not isinstance(estimate_runtime_record, Mapping):
                        raise RuntimeError(
                            "DeFoG molecular estimate manifest has no runtime "
                            "statistics."
                        )
                    estimate_statistics = estimate_runtime_record.get(
                        "molecular_statistics"
                    )
                    if not isinstance(estimate_statistics, Mapping) or (
                        molecular_statistics_record is None
                        or str(estimate_statistics.get("distribution_sha256", ""))
                        != str(
                            molecular_statistics_record.get(
                                "distribution_sha256", ""
                            )
                        )
                    ):
                        raise RuntimeError(
                            "Post-training DeFoG sampling reconstructed molecular "
                            "priors that differ from training."
                        )

                estimated_graphs = estimate_dir / "estimated_graphs.pkl"
                ground_truth_graphs = estimate_dir / "ground_truth_graphs.pkl"
                ground_truth_model_view: Path | None = None
                _atomic_pickle(estimated_graphs, list(estimate_result.graphs))
                # DeFoG is unconditional and supplies no source training index.
                # Preserve the complete target pool rather than pretending that
                # estimate i reconstructs training graph i.
                _verify_prepared_sources_unchanged(prepared_splits, splits)
                shutil.copy2(splits["train"], ground_truth_graphs)
                if native in SUPPORTED_MOLECULAR_DATASETS:
                    source_model_view = (
                        persisted_native_data / "model_view" / "train.pkl"
                    )
                    if not source_model_view.is_file():
                        raise RuntimeError(
                            "Molecular DeFoG conversion did not publish the "
                            "ordered ground-truth model view."
                        )
                    ground_truth_model_view = (
                        estimate_dir / "ground_truth_model_view.pkl"
                    )
                    shutil.copy2(source_model_view, ground_truth_model_view)
                    model_view_record = train_record.get("model_view")
                    if not isinstance(model_view_record, Mapping):
                        raise RuntimeError(
                            "Molecular conversion manifest has no train model-view record."
                        )
                    if int(model_view_record.get("graph_count", -1)) != train_graph_count:
                        raise RuntimeError(
                            "Molecular ground-truth model view does not preserve "
                            "the training split count."
                        )
                    expected_model_view_hash = str(model_view_record.get("sha256", ""))
                    if _sha256(ground_truth_model_view) != expected_model_view_hash:
                        raise RuntimeError(
                            "Molecular ground-truth model-view checksum changed "
                            "after dataset conversion."
                        )

                native_export = native_estimate_dir / "defog_samples.npz"
                if estimate_result.export_path.resolve() != native_export.resolve():
                    shutil.copy2(estimate_result.export_path, native_export)
                native_manifest: Path | None = None
                if (
                    estimate_result.manifest_path is not None
                    and estimate_result.manifest_path.is_file()
                ):
                    native_manifest = native_estimate_dir / "defog_manifest.json"
                    if (
                        estimate_result.manifest_path.resolve()
                        != native_manifest.resolve()
                    ):
                        shutil.copy2(estimate_result.manifest_path, native_manifest)
                path_replacements = {
                    str(estimate_dir.resolve()): str(
                        layout.training_estimates_dir.expanduser().resolve()
                    ),
                    str(persisted_native_data.resolve()): str(
                        layout.native_training_dataset_dir.expanduser().resolve()
                    ),
                    str(checkpoint_path.resolve()): str(
                        (layout.checkpoints_dir / "model.ckpt").expanduser().resolve()
                    ),
                    str(resolved_config.resolve()): str(
                        layout.resolved_training_config_path.expanduser().resolve()
                    ),
                }
                if molecular_statistics_path is not None:
                    path_replacements[str(molecular_statistics_path.resolve())] = str(
                        (layout.train_dir / "molecular_statistics.json")
                        .expanduser()
                        .resolve()
                    )
                normalized_defog_manifest = json.dumps(
                    _jsonable(estimate_result.manifest),
                    sort_keys=True,
                )
                for temporary_path, published_path in path_replacements.items():
                    normalized_defog_manifest = normalized_defog_manifest.replace(
                        temporary_path,
                        published_path,
                    )
                normalized_defog_manifest_value = json.loads(
                    normalized_defog_manifest
                )
                if native_manifest is not None:
                    native_manifest_text = native_manifest.read_text(encoding="utf-8")
                    for temporary_path, published_path in path_replacements.items():
                        native_manifest_text = native_manifest_text.replace(
                            temporary_path,
                            published_path,
                        )
                    native_manifest.write_text(
                        native_manifest_text,
                        encoding="utf-8",
                    )
                estimate_log = estimate_dir / "generate.log"
                if (
                    estimate_result.log_path is not None
                    and estimate_result.log_path.is_file()
                ):
                    shutil.copy2(estimate_result.log_path, estimate_log)
                else:
                    estimate_log.write_text(
                        "DeFoG returned no separate sampling log.\n",
                        encoding="utf-8",
                    )

                estimated_hash = _sha256(estimated_graphs)
                ground_truth_hash = _sha256(ground_truth_graphs)
                source_train = train_record.get("source")
                if not isinstance(source_train, Mapping):
                    raise RuntimeError(
                        "Prepared training split has no source provenance."
                    )
                source_train_hash = str(source_train["sha256"])
                if ground_truth_hash != source_train_hash:
                    raise RuntimeError(
                        "Published ground-truth training graphs do not match "
                        "the prepared input split."
                    )
                estimates_manifest = {
                    "format": TRAINING_ESTIMATES_MANIFEST_FORMAT,
                    "model_id": self.model_id,
                    "dataset": {
                        "benchmark_id": request.dataset.benchmark_id,
                        "serialized_id": request.dataset.serialized_id,
                        "native_id": native,
                        **_dataset_profile(
                            request.dataset.benchmark_id,
                            native,
                        ),
                    },
                    "run_id": request.run.run_id,
                    "checkpoint": {
                        "path": "../checkpoints/model.ckpt",
                        "sha256": _sha256(checkpoint_path),
                    },
                    "estimated_graphs": {
                        "path": "estimated_graphs.pkl",
                        "sha256": estimated_hash,
                        "count": estimate_count,
                        "seed": estimate_seed,
                        "semantics": "independent_unconditional_sample_pool",
                        "sample_order": "defog_raw_index_ascending",
                        "filtered_or_dropped": 0,
                    },
                    "ground_truth_graphs": {
                        "path": "ground_truth_graphs.pkl",
                        "sha256": ground_truth_hash,
                        "count": train_graph_count,
                        "source_split": "train",
                        "source_path": str(splits["train"].resolve()),
                        "source_sha256": source_train_hash,
                        "order": "source_pickle_sequence_order",
                        "storage": "exact_copy",
                        "representation": "project_source",
                    },
                    "ground_truth_model_view": (
                        {
                            "path": "ground_truth_model_view.pkl",
                            "sha256": _sha256(ground_truth_model_view),
                            "count": train_graph_count,
                            "source_split": "train",
                            "representation": "defog_model",
                            "source_to_model_view_index": "identity",
                            "required_for_attributed_pairing": True,
                        }
                        if ground_truth_model_view is not None
                        else None
                    ),
                    "pairing": {
                        "status": "unpaired",
                        "pair_count": 0,
                        "one_to_one": False,
                        "reason": (
                            "The attached DeFoG sampler is unconditional and "
                            "does not expose a source training-graph index."
                        ),
                        "required_next_step": (
                            "Apply and report an explicit training-only matching "
                            "or coupling before constructing GraphER supervision."
                        ),
                    },
                    "native_export": {
                        "path": "native/defog_samples.npz",
                        "sha256": _sha256(native_export),
                        "manifest": (
                            "native/defog_manifest.json"
                            if native_manifest is not None
                            else None
                        ),
                    },
                    "defog_manifest": normalized_defog_manifest_value,
                    "sampling": {
                        "config": estimate_sampling,
                        "runtime": estimate_runtime,
                    },
                    "created_at": _utc_now(),
                }
                _atomic_json(estimate_dir / "manifest.json", estimates_manifest)
                training_estimates_summary = {
                    "enabled": True,
                    "manifest": "training_estimates/manifest.json",
                    "estimated_graphs": "training_estimates/estimated_graphs.pkl",
                    "ground_truth_graphs": (
                        "training_estimates/ground_truth_graphs.pkl"
                    ),
                    "ground_truth_model_view": (
                        "training_estimates/ground_truth_model_view.pkl"
                        if ground_truth_model_view is not None
                        else None
                    ),
                    "estimated_count": estimate_count,
                    "ground_truth_count": train_graph_count,
                    "pairing_status": "unpaired",
                }

            _verify_prepared_sources_unchanged(prepared_splits, splits)
            _verify_source_identity(source_identity, _source_identity(defog_root))
            duration = time.monotonic() - started
            split_hashes = {
                split: _sha256(path) for split, path in request.dataset.split_paths.items()
            }
            manifest = {
                "format": TRAINING_MANIFEST_FORMAT,
                "model_id": self.model_id,
                "dataset": {
                    "benchmark_id": request.dataset.benchmark_id,
                    "serialized_id": request.dataset.serialized_id,
                    "native_id": native,
                    **_dataset_profile(request.dataset.benchmark_id, native),
                    "fingerprint": request.dataset.fingerprint(),
                    "split_sha256": split_hashes,
                },
                "run_id": request.run.run_id,
                "train_seed": request.run.train_seed,
                "started_at": started_at,
                "finished_at": _utc_now(),
                "duration_seconds": duration,
                "runtime": {
                    "requested_gpus": gpus,
                    "cuda_visible_devices": cuda_visible_devices,
                    "single_device_strategy_policy": "disable_ddp_use_auto",
                    "diagnostics": {
                        "path": "runtime_diagnostics.json",
                        "sha256": _sha256(runtime_diagnostics_path),
                    },
                },
                "checkpoint": {
                    "path": "checkpoints/model.ckpt",
                    "sha256": _sha256(checkpoint_path),
                    "native_source": str(source_checkpoint.relative_to(native_run)),
                    "selection": (
                        "post_fit_explicit_final_checkpoint"
                        if selected_checkpoint_is_final
                        else "highest_saved_epoch"
                    ),
                    "selected_is_explicit_final": selected_checkpoint_is_final,
                    "completion_record": (
                        {
                            "path": "checkpoints/final_checkpoint.json",
                            "sha256": _sha256(final_checkpoint_record_path),
                        }
                        if final_checkpoint_record_path is not None
                        else None
                    ),
                    "selected_epoch": selected_checkpoint_epoch,
                    "configured_n_epochs": configured_n_epochs,
                    "final_epoch_verified": final_epoch_verified,
                },
                "resolved_config": {
                    "path": "resolved_config.yaml",
                    "sha256": _sha256(resolved_config),
                },
                "dataset_conversion_manifest": "dataset_conversion.json",
                "native_dataset": {
                    "path": "native_dataset",
                    "purpose": (
                        "Exact DeFoG-formatted data used to reconstruct dataset "
                        "metadata for checkpoint sampling."
                    ),
                },
                "molecular_statistics": (
                    {
                        "policy": "recomputed_from_converted_train_and_val_v1",
                        "applied_during_training": True,
                        "applied_during_generation": True,
                        "path": "molecular_statistics.json",
                        "sha256": _sha256(molecular_statistics_path),
                        "distribution_sha256": molecular_statistics_record[
                            "distribution_sha256"
                        ],
                        "reason": (
                            "Avoid upstream full-benchmark priors when the "
                            "GraphER split or subset differs."
                        ),
                    }
                    if native in SUPPORTED_MOLECULAR_DATASETS
                    and molecular_statistics_path is not None
                    and molecular_statistics_record is not None
                    else None
                ),
                "training_estimates": training_estimates_summary,
                "upstream": {
                    **source_identity,
                    "python_executable": python_executable,
                    "python_environment": python_environment,
                    "experiment": experiment,
                },
                "commands": {
                    "prepare": prepare_command,
                    "train": command,
                    "shell": False,
                },
                "options": options,
            }
            _atomic_json(stage_train / "manifest.json", manifest)
            _emit_progress(
                progress,
                f"publishing training artifacts to {layout.train_dir.resolve()}",
            )
            _publish_directory(stage_train, layout.train_dir, overwrite=request.overwrite)
            _atomic_json(
                layout.run_manifest_path,
                {
                    "format": "grapher_baseline_run_v1",
                    "model_id": self.model_id,
                    "dataset_id": request.run.dataset_id,
                    "run_id": request.run.run_id,
                    "train_seed": request.run.train_seed,
                    "training_manifest": "train/manifest.json",
                    "training_estimates_manifest": (
                        "train/training_estimates/manifest.json"
                        if estimates_enabled
                        else None
                    ),
                },
            )
            _emit_progress(
                progress,
                "training transaction complete: "
                f"checkpoint={layout.checkpoints_dir / 'model.ckpt'}",
            )
        except Exception as exc:
            try:
                failure_path = _preserve_training_failure(
                    layout=layout,
                    log_path=log_path,
                    diagnostics_path=runtime_diagnostics_path,
                    error=exc,
                    started_at=started_at,
                    commands={
                        "prepare": prepare_command,
                        "train": command,
                    },
                    native_run=native_run,
                    working_directory=defog_root / "src",
                    environment=environment,
                )
            except Exception as preservation_error:
                raise RuntimeError(
                    f"{exc}\nGraphER also failed to preserve DeFoG failure "
                    f"artifacts: {preservation_error}"
                ) from exc
            preserved_details = [
                f"Failure artifacts preserved at: {failure_path.resolve()}"
            ]
            preserved_log = failure_path / "train.log"
            if preserved_log.is_file():
                preserved_details.append(f"Full log: {preserved_log.resolve()}")
            preserved_diagnostics = failure_path / "runtime_diagnostics.json"
            if preserved_diagnostics.is_file():
                preserved_details.append(
                    "Runtime diagnostics: "
                    f"{preserved_diagnostics.resolve()}"
                )
            raise RuntimeError(
                f"{exc}\n" + "\n".join(preserved_details)
            ) from exc
        finally:
            if workspace.exists():
                shutil.rmtree(workspace)

        return TrainingArtifacts(
            run_dir=layout.run_dir,
            checkpoint_path=layout.checkpoints_dir / "model.ckpt",
            manifest_path=layout.training_manifest_path,
            log_path=layout.training_log_path,
            artifacts=(
                layout.resolved_training_config_path,
                layout.train_dir / "dataset_conversion.json",
                layout.native_training_dataset_dir,
                layout.train_dir / "runtime_diagnostics.json",
            )
            + (
                (layout.train_dir / "molecular_statistics.json",)
                if native in SUPPORTED_MOLECULAR_DATASETS
                else ()
            ),
            estimated_graphs_path=(
                layout.estimated_training_graphs_path
                if estimates_enabled
                else None
            ),
            ground_truth_graphs_path=(
                layout.ground_truth_training_graphs_path
                if estimates_enabled
                else None
            ),
            ground_truth_model_view_graphs_path=(
                layout.ground_truth_model_view_graphs_path
                if estimates_enabled
                and native in SUPPORTED_MOLECULAR_DATASETS
                else None
            ),
            training_estimates_manifest_path=(
                layout.training_estimates_manifest_path
                if estimates_enabled
                else None
            ),
        )

    def generate(self, request: GenerateRequest) -> GenerationArtifacts:
        self.validate_generate_request(request)
        options = dict(request.options)
        if options.get("generated_path") is not None:
            raise ValueError(
                "DeFoGWrapper.generate() always samples the requested checkpoint. "
                "Use the low-level neutral-export loader to inspect a pre-existing "
                "DeFoG NPZ without attributing it to this trained run."
            )
        layout = request.run.layout
        training_manifest: dict[str, Any] = {}
        if layout.training_manifest_path.is_file():
            training_manifest = _read_json_object(
                layout.training_manifest_path,
                label="DeFoG training manifest",
            )
        checkpoint_hash = _sha256(request.checkpoint_path)
        saved_checkpoint = training_manifest.get("checkpoint", {})
        if not isinstance(saved_checkpoint, Mapping):
            saved_checkpoint = {}
        saved_checkpoint_hash = saved_checkpoint.get("sha256")
        if (
            saved_checkpoint_hash is not None
            and str(saved_checkpoint_hash) != checkpoint_hash
        ):
            raise ValueError(
                "The requested checkpoint does not belong to this DeFoG RunSpec. "
                "Use a new run_id for an external checkpoint instead of mixing it "
                "with saved training metadata and generations."
            )
        managed_run = (
            training_manifest.get("format") == TRAINING_MANIFEST_FORMAT
            and saved_checkpoint_hash is not None
            and str(saved_checkpoint_hash) == checkpoint_hash
        )
        source_env = str(options.get("source_env", DEFOG_ROOT_ENV))
        python_env = str(options.get("python_env", DEFOG_PYTHON_ENV))
        runtime = dict(options.get("runtime", {}) or {})
        progress = _runtime_progress_options(runtime)
        _emit_progress(
            progress,
            "generation preflight: "
            f"run_id={request.run.run_id}, generation_id="
            f"{request.resolved_generation_id}, requested={request.num_graphs}, "
            f"seed={request.generation_seed}",
        )
        generation_source_identity: dict[str, Any] | None = None
        generation_python_environment: dict[str, Any] | None = None
        if managed_run:
            _verify_managed_generation_assets(layout, training_manifest)
            from grapher.models.defog.runtime import (
                resolve_defog_python,
                resolve_defog_root,
            )

            generation_source_root = resolve_defog_root(source_env)
            generation_source_identity = _source_identity(generation_source_root)
            generation_python = resolve_defog_python(
                defog_root=generation_source_root,
                python_executable=runtime.get("python_executable"),
                python_env=python_env,
            )
            generation_python_environment = _python_environment_identity(
                generation_python
            )
            runtime["python_executable"] = generation_python
            expected_source = training_manifest.get("upstream")
            if not isinstance(expected_source, Mapping):
                raise RuntimeError(
                    "Managed DeFoG training manifest has no upstream source "
                    "identity."
                )
            _verify_source_identity(expected_source, generation_source_identity)
            expected_python_environment = expected_source.get(
                "python_environment"
            )
            if not isinstance(expected_python_environment, Mapping):
                raise RuntimeError(
                    "Managed DeFoG training manifest has no Python-environment "
                    "identity."
                )
            _verify_python_environment_identity(
                expected_python_environment,
                generation_python_environment,
            )
        training_dataset = training_manifest.get("dataset", {})
        if not isinstance(training_dataset, Mapping):
            training_dataset = {}
        native = _native_dataset(
            request.run.dataset_id,
            options.get("native_dataset") or training_dataset.get("native_id"),
        )
        if managed_run and native != str(training_dataset.get("native_id", "")):
            raise ValueError(
                "A managed DeFoG run cannot change native_dataset during "
                "generation. Use a new RunSpec for a different profile."
            )
        from grapher.models.defog.backend import DeFoGGeneratorConfig

        training_upstream = training_manifest.get("upstream", {})
        if not isinstance(training_upstream, Mapping):
            training_upstream = {}
        experiment = str(
            options.get("experiment")
            or training_upstream.get("experiment")
            or _default_experiment(native)
        ).lower()
        if (
            managed_run
            and training_upstream.get("experiment") is not None
            and experiment != str(training_upstream["experiment"]).lower()
        ):
            raise ValueError(
                "A managed DeFoG run cannot change its experiment during "
                "generation."
            )
        sampling = dict(options.get("sampling", {}) or {})
        configured_datadir = options.get("dataset_datadir")
        resolved_config_path = options.get("resolved_config_path")
        if managed_run:
            if configured_datadir is not None and (
                Path(configured_datadir).expanduser().resolve()
                != layout.native_training_dataset_dir.resolve()
            ):
                raise ValueError(
                    "A managed DeFoG run cannot replace its persisted "
                    "dataset_datadir during generation."
                )
            if resolved_config_path is not None and (
                Path(resolved_config_path).expanduser().resolve()
                != layout.resolved_training_config_path.resolve()
            ):
                raise ValueError(
                    "A managed DeFoG run cannot replace its resolved training "
                    "configuration during generation."
                )
            configured_datadir = layout.native_training_dataset_dir
            resolved_config_path = layout.resolved_training_config_path
        else:
            if configured_datadir is None and layout.native_training_dataset_dir.is_dir():
                configured_datadir = layout.native_training_dataset_dir
            if (
                resolved_config_path is None
                and layout.resolved_training_config_path.is_file()
            ):
                resolved_config_path = layout.resolved_training_config_path
        config_values = {
            "type": "defog",
            "backend": "subprocess",
            "dataset": native,
            "experiment": experiment,
            "checkpoint_path": str(request.checkpoint_path),
            "source_env": source_env,
            "python_env": python_env,
            "sampling": sampling,
            "runtime": runtime,
        }
        if configured_datadir is not None:
            config_values["dataset_datadir"] = str(configured_datadir)
        if resolved_config_path is not None:
            config_values["resolved_config_path"] = str(resolved_config_path)
        if managed_run and isinstance(
            training_manifest.get("molecular_statistics"), Mapping
        ):
            config_values["molecular_statistics_path"] = str(
                layout.train_dir / "molecular_statistics.json"
            )
        config = DeFoGGeneratorConfig.from_dict(config_values)

        generation_id = request.resolved_generation_id
        target = layout.generation_dir(generation_id)
        ArtifactLayout.require_available(target, overwrite=request.overwrite)
        staging_root = layout.output_root.expanduser().resolve() / ".staging"
        staging_root.mkdir(parents=True, exist_ok=True)
        workspace = Path(tempfile.mkdtemp(prefix="defog-generate-", dir=staging_root))
        stage_generation = workspace / "generation"
        native_dir = stage_generation / "native"
        native_dir.mkdir(parents=True)
        started_at = _utc_now()
        started = time.monotonic()
        try:
            _emit_progress(
                progress,
                "launching DeFoG checkpoint generation: "
                f"dataset={native}, experiment={experiment}, "
                f"batch_size={config.batch_size}",
            )
            result = generate_defog_graphs(
                config,
                num_graphs=request.num_graphs,
                seed=request.generation_seed,
                output_dir=native_dir,
            )
            if len(result.graphs) != request.num_graphs:
                raise RuntimeError(
                    f"DeFoG returned {len(result.graphs)} graphs; "
                    f"expected {request.num_graphs}."
                )
            _emit_progress(
                progress,
                f"checkpoint generation complete: generated={len(result.graphs)}",
            )
            training_statistics = training_manifest.get("molecular_statistics")
            if managed_run and isinstance(training_statistics, Mapping):
                result_runtime = result.manifest.get("runtime")
                if not isinstance(result_runtime, Mapping):
                    raise RuntimeError(
                        "DeFoG molecular generation manifest has no runtime "
                        "statistics."
                    )
                generation_statistics = result_runtime.get(
                    "molecular_statistics"
                )
                if not isinstance(generation_statistics, Mapping) or str(
                    generation_statistics.get("distribution_sha256", "")
                ) != str(training_statistics.get("distribution_sha256", "")):
                    raise RuntimeError(
                        "DeFoG generation reconstructed molecular priors that "
                        "differ from the training checkpoint."
                    )
            graphs_path = stage_generation / "base_graphs.pkl"
            _atomic_pickle(graphs_path, list(result.graphs))

            export_path = native_dir / "defog_samples.npz"
            if result.export_path.resolve() != export_path.resolve():
                shutil.copy2(result.export_path, export_path)
            manifest_copy: Path | None = None
            if result.manifest_path is not None and result.manifest_path.is_file():
                manifest_copy = native_dir / "defog_manifest.json"
                if result.manifest_path.resolve() != manifest_copy.resolve():
                    shutil.copy2(result.manifest_path, manifest_copy)
            normalized_defog_manifest = json.dumps(
                _jsonable(result.manifest),
                sort_keys=True,
            ).replace(
                str(stage_generation.resolve()),
                str(target.expanduser().resolve()),
            )
            normalized_defog_manifest_value = json.loads(
                normalized_defog_manifest
            )
            if manifest_copy is not None:
                manifest_copy.write_text(
                    manifest_copy.read_text(encoding="utf-8").replace(
                        str(stage_generation.resolve()),
                        str(target.expanduser().resolve()),
                    ),
                    encoding="utf-8",
                )
            log_path = stage_generation / "generate.log"
            if result.log_path is not None and result.log_path.is_file():
                shutil.copy2(result.log_path, log_path)
            else:
                log_path.write_text(
                    "Reused a validated DeFoG neutral export.\n", encoding="utf-8"
                )

            graphs_hash = _sha256(graphs_path)
            manifest = {
                "format": GENERATION_MANIFEST_FORMAT,
                "model_id": self.model_id,
                "dataset": {
                    "benchmark_id": request.run.dataset_id,
                    "native_id": native,
                    **_dataset_profile(request.run.dataset_id, native),
                    "datadir": (
                        str(Path(configured_datadir).expanduser().resolve())
                        if configured_datadir is not None
                        else None
                    ),
                },
                "run_id": request.run.run_id,
                "generation_id": generation_id,
                "train_seed": request.run.train_seed,
                "generation_seed": request.generation_seed,
                "started_at": started_at,
                "finished_at": _utc_now(),
                "duration_seconds": time.monotonic() - started,
                "requested_count": request.num_graphs,
                "returned_count": len(result.graphs),
                "sample_order": "defog_raw_index_ascending",
                "base_graphs": {"path": "base_graphs.pkl", "sha256": graphs_hash},
                "checkpoint": {
                    "path": str(request.checkpoint_path.resolve()),
                    "sha256": checkpoint_hash,
                },
                "resolved_training_config": (
                    {
                        "path": str(Path(resolved_config_path).expanduser().resolve()),
                        "sha256": _sha256(
                            Path(resolved_config_path).expanduser().resolve()
                        ),
                    }
                    if resolved_config_path is not None
                    else None
                ),
                "native_export": {
                    "path": "native/defog_samples.npz",
                    "sha256": _sha256(export_path),
                    "manifest": (
                        "native/defog_manifest.json" if manifest_copy is not None else None
                    ),
                },
                "defog_manifest": normalized_defog_manifest_value,
                "upstream": generation_source_identity,
                "python_environment": generation_python_environment,
                "options": options,
            }
            _atomic_json(stage_generation / "manifest.json", manifest)
            _emit_progress(
                progress,
                f"publishing generation artifacts to {target.resolve()}",
            )
            _publish_directory(stage_generation, target, overwrite=request.overwrite)
        finally:
            if workspace.exists():
                shutil.rmtree(workspace)

        final_native = layout.native_generation_dir(generation_id)
        native_artifacts = [final_native / "defog_samples.npz"]
        if (final_native / "defog_manifest.json").is_file():
            native_artifacts.append(final_native / "defog_manifest.json")
        _emit_progress(
            progress,
            "generation transaction complete: "
            f"graphs={layout.generated_graphs_path(generation_id)}",
        )
        return GenerationArtifacts(
            run_dir=layout.run_dir,
            generation_dir=target,
            graphs_path=layout.generated_graphs_path(generation_id),
            manifest_path=layout.generation_manifest_path(generation_id),
            num_requested=request.num_graphs,
            num_generated=request.num_graphs,
            graphs_sha256=graphs_hash,
            log_path=layout.generation_log_path(generation_id),
            native_artifacts=tuple(native_artifacts),
        )

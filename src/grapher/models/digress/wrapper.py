"""GraphER-facing wrapper for the attached external DiGress codebase.

The implementation follows the same artifact and isolation convention as
``DeFoGWrapper``. GraphER-prepared NetworkX splits are converted into native
PyG files, DiGress is trained in its own interpreter, and generated tensors are
exported through a neutral NPZ schema before GraphER constructs NetworkX
objects in its own process.
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

TRAINING_MANIFEST_FORMAT = "grapher_digress_training_v1"
TRAINING_ESTIMATES_MANIFEST_FORMAT = "grapher_digress_training_estimates_v1"
GENERATION_MANIFEST_FORMAT = "grapher_digress_generation_v1"
SUPPORTED_GENERIC_DATASETS = frozenset({"comm20", "planar", "sbm"})
SUPPORTED_MOLECULAR_DATASETS = frozenset({"qm9", "zinc"})
SUPPORTED_NATIVE_DATASETS = SUPPORTED_GENERIC_DATASETS | SUPPORTED_MOLECULAR_DATASETS
_NATIVE_BY_BENCHMARK = {
    "community_small": "comm20",
    "ego_small": "comm20",
    "grid": "planar",
    "comm20": "comm20",
    "planar": "planar",
    "sbm": "sbm",
    "qm9": "qm9",
    "qm9_attributed": "qm9",
    "zinc": "zinc",
    "zinc_attributed": "zinc",
}
_EXPERIMENT_BY_NATIVE = {
    "comm20": "comm20",
    "planar": "planar",
    "sbm": "sbm",
    "qm9": "qm9_no_h",
    "zinc": "zinc_no_h",
}
_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9_.-]+$")
_PROTECTED_HYDRA_KEYS = frozenset(
    {
        "experiment",
        "dataset",
        "dataset.name",
        "dataset.datadir",
        "general.name",
        "general.gpus",
        "general.wandb",
        "train.seed",
        "train.n_epochs",
        "train.batch_size",
        "train.num_workers",
        "general.check_val_every_n_epochs",
    }
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{label} must contain a JSON object: {path}")
    return value


def _load_pickle_graphs(path: Path) -> list[Any]:
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"Expected a graph sequence in {path}.")
    return list(value)


def _deep_update(base: dict[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            base[key] = _deep_update(dict(base[key]), value)
        else:
            base[key] = value
    return base


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    return dict(value)


def _boolean(value: Any, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean.")
    return value


def _load_options(request: TrainRequest) -> dict[str, Any]:
    options: dict[str, Any] = {}
    if request.config_path is not None:
        if not request.config_path.is_file():
            raise FileNotFoundError(
                f"Missing DiGress wrapper config: {request.config_path}"
            )
        loaded = yaml.safe_load(request.config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, Mapping):
            raise TypeError("The DiGress wrapper config must contain a mapping.")
        selected = loaded.get("digress", loaded)
        if not isinstance(selected, Mapping):
            raise TypeError("The digress config section must contain a mapping.")
        options = dict(selected)
    return _deep_update(options, request.options)


def _native_dataset(benchmark: str, explicit: Any = None) -> str:
    requested = str(explicit or benchmark).lower()
    native = _NATIVE_BY_BENCHMARK.get(requested, requested)
    if native not in SUPPORTED_NATIVE_DATASETS:
        raise ValueError(
            f"DiGressWrapper supports native datasets "
            f"{sorted(SUPPORTED_NATIVE_DATASETS)}; benchmark {benchmark!r} has "
            "no declared compatibility profile."
        )
    return native


def _default_experiment(native: str) -> str:
    return _EXPERIMENT_BY_NATIVE[native]


def _upstream_config_templates(native: str, experiment: str) -> tuple[str, str]:
    """Resolve stock DiGress Hydra configs used as adapter templates.

    The attached DiGress checkout has no native ZINC config.  GraphER supplies
    the ZINC data module semantics and empirical categorical priors inside its
    isolated worker, while reusing the stock heavy-atom QM9 config solely for
    loader and model hyperparameter defaults.  The external source tree is
    never modified.
    """

    if native == "zinc":
        template_experiment = (
            "qm9_no_h" if experiment == "zinc_no_h" else experiment
        )
        return "qm9", template_experiment
    return native, experiment


def _default_generation_batch_size(native: str) -> int:
    if native == "qm9":
        return 256
    if native == "zinc":
        return 128
    return 64


def _dataset_profile(benchmark: str, native: str) -> dict[str, str]:
    if native == "zinc":
        representation = "heavy_atom_kekulized_categorical"
        domain = "molecular"
    elif native in SUPPORTED_MOLECULAR_DATASETS:
        representation = "heavy_atom_categorical"
        domain = "molecular"
    else:
        representation = "simple_undirected_topology"
        domain = "generic"
    if benchmark == "ego_small" and native == "comm20":
        role = "generic_loader_compatibility_profile"
    elif benchmark == "grid" and native == "planar":
        role = "generic_loader_and_architecture_compatibility_profile"
    elif benchmark != native:
        role = "declared_native_alias"
    else:
        role = "native"
    return {
        "domain": domain,
        "profile_role": role,
        "model_representation": representation,
    }


def _progress_options(runtime: Mapping[str, Any]) -> dict[str, Any]:
    raw = runtime.get("progress", {}) or {}
    if not isinstance(raw, Mapping):
        raise TypeError("runtime.progress must be a mapping.")
    enabled = _boolean(raw.get("enabled", False), name="runtime.progress.enabled")
    stream = _boolean(
        raw.get("stream_output", enabled),
        name="runtime.progress.stream_output",
    )
    interval = float(raw.get("interval_seconds", 30.0))
    if interval <= 0:
        raise ValueError("runtime.progress.interval_seconds must be positive.")
    epoch_value = raw.get("epoch_interval")
    epoch_interval = None if epoch_value is None else int(epoch_value)
    if epoch_interval is not None and epoch_interval <= 0:
        raise ValueError("runtime.progress.epoch_interval must be positive.")
    generation_interval = int(raw.get("generation_batch_interval", 1))
    if generation_interval <= 0:
        raise ValueError(
            "runtime.progress.generation_batch_interval must be positive."
        )
    return {
        "enabled": enabled,
        "stream_output": stream,
        "interval_seconds": interval,
        "epoch_interval": epoch_interval,
        "generation_batch_interval": generation_interval,
    }


def _emit(progress: Mapping[str, Any], message: str) -> None:
    if bool(progress.get("enabled")):
        print(f"[GraphER/DiGress] {message}", file=sys.stderr, flush=True)


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


def _source_identity(root: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    files: list[Path] = []
    for directory in (root / "configs", root / "src"):
        if directory.is_dir():
            files.extend(
                path
                for path in directory.rglob("*")
                if path.is_file()
                and path.suffix.lower() in {".py", ".yaml", ".yml"}
            )
    for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return {
        "source_root": str(root),
        "revision": _source_revision(root),
        "source_fingerprint": {
            "algorithm": "sha256_names_and_contents_v1",
            "sha256": digest.hexdigest(),
            "file_count": len(files),
        },
    }


def _verify_source(expected: Mapping[str, Any], observed: Mapping[str, Any]) -> None:
    expected_fp = expected.get("source_fingerprint")
    observed_fp = observed.get("source_fingerprint")
    if not isinstance(expected_fp, Mapping) or not isinstance(observed_fp, Mapping):
        raise RuntimeError("DiGress source fingerprint provenance is incomplete.")
    if str(expected_fp.get("sha256", "")) != str(
        observed_fp.get("sha256", "")
    ):
        raise RuntimeError(
            "The external DiGress source tree changed after training. Use a new "
            "run_id or restore the recorded source."
        )


def _python_identity(python: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [str(python), "-c", "import platform,sys; print(platform.python_version()); print(sys.prefix)"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
        check=False,
        shell=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Could not inspect the DiGress Python environment: "
            f"{completed.stderr.strip()}"
        )
    lines = completed.stdout.splitlines()
    return {
        "python_executable": str(python.resolve()),
        "python_version": lines[0].strip() if lines else None,
        "prefix": lines[1].strip() if len(lines) > 1 else None,
    }


def _external_environment(
    root: Path,
    *,
    seed: int,
    cuda_visible_devices: str | None,
) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join([str(root), str(root / "src")]),
            "PYTHONHASHSEED": str(int(seed)),
            "PYTHONUNBUFFERED": "1",
            "WANDB_MODE": "disabled",
            "WANDB_DISABLED": "true",
            "HYDRA_FULL_ERROR": "1",
            "MPLBACKEND": "Agg",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        }
    )
    if cuda_visible_devices is not None:
        environment["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    return environment


def _worker(name: str) -> Path:
    path = Path(__file__).resolve().parent / "workers" / name
    if not path.is_file():
        raise FileNotFoundError(f"Missing DiGress worker: {path}")
    return path


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


def _tail(path: Path, *, lines: int = 200) -> str:
    if not path.is_file():
        return ""
    return "\n".join(
        path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
    )


def _normalize_hydra_overrides(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("hydra_overrides must be a sequence of strings.")
    result: list[str] = []
    for raw in value:
        text = str(raw)
        if not text or any(character in text for character in "\x00\r\n"):
            raise ValueError(f"Invalid Hydra override: {raw!r}")
        key = text.lstrip("+").split("=", 1)[0]
        if key in _PROTECTED_HYDRA_KEYS:
            raise ValueError(
                f"Hydra override {key!r} is managed by DiGressWrapper; use the "
                "corresponding wrapper option or CLI argument."
            )
        result.append(text)
    return result


def _rewrite_config_datadir(path: Path, durable_path: Path) -> None:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise TypeError("Resolved DiGress config must contain a mapping.")
    dataset = loaded.setdefault("dataset", {})
    if not isinstance(dataset, dict):
        raise TypeError("Resolved DiGress dataset config must be a mapping.")
    dataset["datadir"] = str(durable_path.expanduser().resolve())
    path.write_text(yaml.safe_dump(loaded, sort_keys=False), encoding="utf-8")


def _normalize_conversion_manifest(
    manifest: dict[str, Any], *, temporary_root: Path
) -> dict[str, Any]:
    text = json.dumps(_jsonable(manifest), sort_keys=True)
    text = text.replace(str(temporary_root.resolve()), "native_dataset")
    return json.loads(text)


def _preserve_failure(
    *,
    layout: ArtifactLayout,
    workspace: Path,
    log_path: Path,
    error: BaseException,
    command: Sequence[str] | None,
) -> Path:
    target = layout.run_dir / "failures" / f"attempt-{time.time_ns()}"
    target.mkdir(parents=True, exist_ok=False)
    if log_path.is_file():
        shutil.copy2(log_path, target / "train.log")
    # Preserve a completed or periodic checkpoint before the staging tree is
    # removed. This protects long runs from late post-training decode failures.
    checkpoints = sorted(
        (path for path in workspace.rglob("*.ckpt") if path.is_file()),
        key=lambda item: item.stat().st_mtime_ns,
    )
    recovered: str | None = None
    if checkpoints:
        source = checkpoints[-1]
        destination = target / source.name
        shutil.copy2(source, destination)
        recovered = destination.name
    _atomic_json(
        target / "failure.json",
        {
            "format": "grapher_digress_training_failure_v1",
            "model_id": "digress",
            "dataset_id": layout.dataset_id,
            "run_id": layout.run_id,
            "failed_at": _utc_now(),
            "exception_type": type(error).__name__,
            "exception": str(error),
            "command": list(command) if command is not None else None,
            "log": "train.log" if log_path.is_file() else None,
            "recovered_checkpoint": recovered,
        },
    )
    return target


class DiGressWrapper(BaseGeneratorWrapper):
    """Train and sample the attached DiGress implementation in isolation."""

    model_id = "digress"
    display_name = "DiGress"
    capabilities = BaselineCapabilities(
        domains=frozenset({"generic", "attributed"}),
        isolation="subprocess",
        status="ready",
    )
    implementation_note = (
        "Supports generic SPECTRE-compatible graph datasets and heavy-atom "
        "QM9/ZINC through isolated categorical adapters."
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
        progress: Mapping[str, Any],
        append: bool = True,
    ) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        start_offset = log_path.stat().st_size if log_path.is_file() else 0
        reporter = SubprocessLogReporter(
            label=label,
            log_path=log_path,
            enabled=bool(progress["enabled"]),
            stream_output=bool(progress["stream_output"]),
            interval_seconds=float(progress["interval_seconds"]),
            prefix="GraphER/DiGress",
        )
        with log_path.open(mode, encoding="utf-8") as log_file:
            reporter.start(start_offset=start_offset)
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
                reporter.stop(status="timed out")
                raise RuntimeError(
                    f"{label} timed out.\nLog: {log_path}\n"
                    f"Command: {json.dumps(list(command))}\n"
                    f"Last output:\n{_tail(log_path)}"
                ) from exc
            except BaseException:
                reporter.stop(status="failed")
                raise
            else:
                reporter.stop(
                    status=(
                        "completed"
                        if completed.returncode == 0
                        else f"failed with exit code {completed.returncode}"
                    )
                )
        if completed.returncode != 0:
            raise RuntimeError(
                f"{label} exited with code {completed.returncode}.\n"
                f"Working directory: {cwd}\nLog: {log_path}\n"
                f"Command: {json.dumps(list(command))}\n"
                f"Last output:\n{_tail(log_path)}"
            )

    def train(self, request: TrainRequest) -> TrainingArtifacts:
        self.validate_train_request(request)
        options = _load_options(request)
        native = _native_dataset(
            request.dataset.benchmark_id,
            options.get("native_dataset") or request.dataset.native_id,
        )
        experiment = str(
            options.get("experiment", _default_experiment(native))
        ).lower()
        if not _SAFE_IDENTIFIER.fullmatch(experiment):
            raise ValueError("DiGress experiment must be a safe identifier.")

        from grapher.models.digress.runtime import (
            resolve_digress_python,
            resolve_digress_root,
        )

        source_env = str(options.get("source_env", "DIGRESS"))
        python_env = str(options.get("python_env", "DIGRESS_PYTHON"))
        root = resolve_digress_root(source_env)
        runtime = _mapping(options.get("runtime"), name="runtime")
        progress = _progress_options(runtime)
        python = resolve_digress_python(
            digress_root=root,
            python_executable=runtime.get("python_executable"),
            python_env=python_env,
        )
        source_identity = _source_identity(root)
        python_identity = _python_identity(python)
        timeout_value = runtime.get("timeout_seconds")
        timeout_seconds = None if timeout_value is None else float(timeout_value)
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("runtime.timeout_seconds must be positive.")
        gpus = int(runtime.get("gpus", 1))
        if gpus not in {0, 1}:
            raise ValueError("runtime.gpus must be 0 or 1.")
        require_cuda = bool(runtime.get("require_cuda", False))
        if require_cuda and gpus != 1:
            raise ValueError("runtime.require_cuda=true requires runtime.gpus=1.")
        cuda_value = runtime.get("cuda_visible_devices")
        cuda_visible_devices = None if cuda_value is None else str(cuda_value)
        if cuda_visible_devices is not None and any(
            character in cuda_visible_devices for character in "\x00\r\n"
        ):
            raise ValueError("runtime.cuda_visible_devices is invalid.")

        template_dataset, template_experiment = _upstream_config_templates(
            native, experiment
        )
        experiment_config = (
            root / "configs" / "experiment" / f"{template_experiment}.yaml"
        )
        dataset_config = (
            root / "configs" / "dataset" / f"{template_dataset}.yaml"
        )
        required_source_paths = [experiment_config, dataset_config]
        if native in SUPPORTED_MOLECULAR_DATASETS:
            required_source_paths.append(
                root / "src" / "datasets" / "qm9_dataset.py"
            )
        missing = [
            str(path)
            for path in required_source_paths
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "The attached DiGress source lacks required config/loader "
                f"templates: {missing}."
            )

        layout = request.run.layout
        if (
            request.overwrite
            and layout.generations_dir.is_dir()
            and any(layout.generations_dir.iterdir())
        ):
            raise ArtifactCollisionError(
                "Cannot overwrite a trained DiGress run that already has raw "
                "generation batches. Use a new run_id."
            )
        ArtifactLayout.require_available(layout.train_dir, overwrite=request.overwrite)
        staging_root = layout.output_root.expanduser().resolve() / ".staging"
        staging_root.mkdir(parents=True, exist_ok=True)
        workspace = Path(tempfile.mkdtemp(prefix="digress-train-", dir=staging_root))
        stage_train = workspace / "train"
        stage_train.mkdir()
        native_data = workspace / "native_dataset"
        native_run = workspace / "native_run"
        log_path = stage_train / "train.log"
        conversion_manifest_path = stage_train / "dataset_conversion.json"
        worker_manifest_path = native_run / "training_worker_manifest.json"
        started_at = _utc_now()
        started = time.monotonic()
        command: list[str] | None = None
        try:
            environment = _external_environment(
                root,
                seed=request.run.train_seed,
                cuda_visible_devices=cuda_visible_devices,
            )
            preparation_worker = (
                _worker("prepare_molecular_dataset.py")
                if native in SUPPORTED_MOLECULAR_DATASETS
                else _worker("prepare_dataset.py")
            )
            split_paths = request.dataset.split_paths
            prepare_command = [
                str(python),
                str(preparation_worker),
                "--dataset",
                native,
                "--output-root",
                str(native_data),
                "--manifest",
                str(conversion_manifest_path),
                "--train",
                str(split_paths["train"].resolve()),
                "--val",
                str(split_paths["val"].resolve()),
                "--test",
                str(split_paths["test"].resolve()),
            ]
            _emit(
                progress,
                "preparing DiGress-native PyG artifacts from immutable GraphER splits",
            )
            self._run_external(
                prepare_command,
                cwd=root / "src",
                environment=environment,
                log_path=log_path,
                timeout_seconds=timeout_seconds,
                label="DiGress dataset preparation",
                progress=progress,
                append=False,
            )
            if not conversion_manifest_path.is_file():
                raise RuntimeError("DiGress preparation worker published no manifest.")
            conversion = _read_json(
                conversion_manifest_path, label="DiGress conversion manifest"
            )
            expected_conversion_format = (
                "grapher_to_digress_molecular_dataset_v1"
                if native in SUPPORTED_MOLECULAR_DATASETS
                else "grapher_to_digress_generic_dataset_v1"
            )
            if conversion.get("format") != expected_conversion_format:
                raise RuntimeError(
                    "Unsupported DiGress conversion manifest format: "
                    f"{conversion.get('format')!r}."
                )
            if str(conversion.get("dataset", "")).lower() != native:
                raise RuntimeError(
                    "DiGress conversion dataset mismatch: "
                    f"expected {native!r}, got {conversion.get('dataset')!r}."
                )
            if conversion.get("split_order_preserved") is not True or int(
                conversion.get("graphs_dropped", -1)
            ) != 0:
                raise RuntimeError(
                    "DiGress conversion did not preserve every ordered graph."
                )
            if native in SUPPORTED_MOLECULAR_DATASETS:
                from grapher.models.digress.codec import (
                    MOLECULAR_ATOMIC_NUMBERS,
                    MOLECULAR_BOND_TYPES,
                )

                vocabulary = conversion.get("vocabulary")
                if not isinstance(vocabulary, Mapping):
                    raise RuntimeError(
                        "DiGress molecular conversion has no vocabulary record."
                    )
                observed_atoms = tuple(
                    int(value)
                    for value in vocabulary.get("atom_class_to_atomic_number", ())
                )
                observed_bonds = frozenset(
                    int(value)
                    for value in vocabulary.get("present_edge_classes", ())
                )
                if observed_atoms != MOLECULAR_ATOMIC_NUMBERS[native] or (
                    observed_bonds != MOLECULAR_BOND_TYPES[native]
                ):
                    raise RuntimeError(
                        f"DiGress {native.upper()} conversion vocabulary mismatch."
                    )
            split_records = conversion.get("splits")
            if not isinstance(split_records, Mapping):
                raise RuntimeError("DiGress conversion manifest has no split records.")
            for split, source in split_paths.items():
                record = split_records.get(split)
                if not isinstance(record, Mapping):
                    raise RuntimeError(f"Missing DiGress split record {split!r}.")
                source_record = record.get("source")
                if not isinstance(source_record, Mapping) or str(
                    source_record.get("sha256", "")
                ) != _sha256(source):
                    raise RuntimeError(
                        f"Prepared DiGress split {split!r} does not match its "
                        "GraphER source bytes."
                    )
            train_count = int(split_records["train"].get("graph_count", 0))
            if train_count <= 0:
                raise RuntimeError("DiGress preparation produced no training graphs.")

            hydra_overrides = _normalize_hydra_overrides(
                options.get("hydra_overrides")
            )
            command = [
                str(python),
                str(_worker("train.py")),
                "--digress-root",
                str(root),
                "--dataset",
                native,
                "--experiment",
                experiment,
                "--dataset-datadir",
                str(native_data),
                "--output-dir",
                str(native_run),
                "--manifest",
                str(worker_manifest_path),
                "--run-name",
                f"grapher_{request.run.run_id}",
                "--seed",
                str(request.run.train_seed),
                "--gpus",
                str(gpus),
            ]
            if require_cuda:
                command.append("--require-cuda")
            scalar_options = (
                ("n_epochs", "--n-epochs"),
                ("batch_size", "--batch-size"),
                ("num_workers", "--num-workers"),
                ("check_val_every_n_epochs", "--check-val-every-n-epochs"),
                ("save_every_n_epochs", "--save-every-n-epochs"),
            )
            for key, flag in scalar_options:
                if options.get(key) is not None:
                    command.extend([flag, str(int(options[key]))])
            if progress["epoch_interval"] is not None:
                command.extend(
                    ["--epoch-progress-interval", str(progress["epoch_interval"])]
                )
            if request.resume_from is not None:
                if not request.resume_from.is_file():
                    raise FileNotFoundError(request.resume_from)
                command.extend(["--resume-from", str(request.resume_from.resolve())])
            for override in hydra_overrides:
                command.extend(["--override", override])
            _emit(
                progress,
                "launching isolated DiGress training: "
                f"dataset={native}, experiment={experiment}, "
                f"n_epochs={options.get('n_epochs', 'upstream default')}",
            )
            self._run_external(
                command,
                cwd=root / "src",
                environment=environment,
                log_path=log_path,
                timeout_seconds=timeout_seconds,
                label="DiGress training",
                progress=progress,
            )
            if not worker_manifest_path.is_file():
                raise RuntimeError("DiGress training worker published no manifest.")
            worker_manifest = _read_json(
                worker_manifest_path, label="DiGress training-worker manifest"
            )
            if (
                worker_manifest.get("format")
                != "grapher_digress_training_worker_v1"
                or worker_manifest.get("status") != "complete"
            ):
                raise RuntimeError("DiGress training worker did not complete cleanly.")
            if str(worker_manifest.get("dataset", "")).lower() != native:
                raise RuntimeError("DiGress training worker dataset mismatch.")
            if str(worker_manifest.get("experiment", "")).lower() != experiment:
                raise RuntimeError("DiGress training worker experiment mismatch.")
            source_checkpoint = Path(str(worker_manifest.get("checkpoint", "")))
            source_config = Path(str(worker_manifest.get("resolved_config", "")))
            if not source_checkpoint.is_file() or not source_config.is_file():
                raise RuntimeError(
                    "DiGress training completed without checkpoint/config artifacts."
                )

            checkpoints_dir = stage_train / "checkpoints"
            checkpoints_dir.mkdir()
            checkpoint_path = checkpoints_dir / "model.ckpt"
            shutil.copy2(source_checkpoint, checkpoint_path)
            resolved_config = stage_train / "resolved_config.yaml"
            shutil.copy2(source_config, resolved_config)
            persisted_native = stage_train / "native_dataset"
            native_data.replace(persisted_native)
            _rewrite_config_datadir(
                resolved_config, layout.native_training_dataset_dir
            )
            molecular_statistics_path: Path | None = None
            worker_stats = worker_manifest.get("molecular_statistics")
            if worker_stats:
                source_stats = Path(str(worker_stats))
                if not source_stats.is_file():
                    raise RuntimeError("Missing DiGress molecular-statistics file.")
                molecular_statistics_path = stage_train / "molecular_statistics.json"
                shutil.copy2(source_stats, molecular_statistics_path)

            conversion = _normalize_conversion_manifest(
                conversion, temporary_root=native_data
            )
            # The temporary path may already have been moved; normalize the
            # original workspace root as a second pass.
            conversion = json.loads(
                json.dumps(conversion).replace(
                    str((workspace / "native_dataset").resolve()),
                    "native_dataset",
                )
            )
            _atomic_json(conversion_manifest_path, conversion)

            estimate_options = _mapping(
                options.get("training_estimates"), name="training_estimates"
            )
            estimates_enabled = _boolean(
                estimate_options.get("enabled", True),
                name="training_estimates.enabled",
            )
            estimate_summary: dict[str, Any] = {"enabled": estimates_enabled}
            estimated_path: Path | None = None
            ground_truth_path: Path | None = None
            model_view_path: Path | None = None
            estimates_manifest_path: Path | None = None
            if estimates_enabled:
                estimate_count = int(
                    estimate_options.get(
                        "num_graphs",
                        min(train_count, 1024)
                        if native in SUPPORTED_MOLECULAR_DATASETS
                        else train_count,
                    )
                )
                if estimate_count <= 0 or estimate_count > train_count:
                    raise ValueError(
                        "training_estimates.num_graphs must be between 1 and "
                        f"the training-set size ({train_count})."
                    )
                estimate_seed = int(
                    estimate_options.get("seed", request.run.train_seed)
                )
                estimate_batch_size = int(
                    estimate_options.get(
                        "batch_size",
                        options.get(
                            "generation_batch_size",
                            _default_generation_batch_size(native),
                        ),
                    )
                )
                estimate_native = stage_train / "training_estimates" / "native"
                estimate_native.mkdir(parents=True)
                _emit(
                    progress,
                    "generating independent post-training DiGress source pool: "
                    f"num_graphs={estimate_count}, seed={estimate_seed}",
                )
                from grapher.models.digress.backend import generate_digress_graphs

                estimate_result = generate_digress_graphs(
                    digress_root=root,
                    python_executable=python,
                    dataset=native,
                    dataset_datadir=persisted_native,
                    resolved_config_path=resolved_config,
                    checkpoint_path=checkpoint_path,
                    output_dir=estimate_native,
                    num_graphs=estimate_count,
                    generation_seed=estimate_seed,
                    batch_size=estimate_batch_size,
                    molecular_statistics_path=molecular_statistics_path,
                    device=str(estimate_options.get("device", "auto" if gpus else "cpu")),
                    cuda_visible_devices=cuda_visible_devices,
                    timeout_seconds=timeout_seconds,
                    progress_enabled=bool(progress["enabled"]),
                    stream_output=bool(progress["stream_output"]),
                    progress_interval_seconds=float(progress["interval_seconds"]),
                    generation_progress_every_batches=int(
                        progress["generation_batch_interval"]
                    ),
                )
                estimates_dir = stage_train / "training_estimates"
                estimated_path = estimates_dir / "estimated_graphs.pkl"
                _atomic_pickle(estimated_path, list(estimate_result.graphs))
                train_graphs = _load_pickle_graphs(split_paths["train"])
                ground_truth_path = estimates_dir / "ground_truth_graphs.pkl"
                _atomic_pickle(ground_truth_path, train_graphs[:estimate_count])
                if native in SUPPORTED_MOLECULAR_DATASETS:
                    source_model_view = (
                        persisted_native / "model_view" / "train.pkl"
                    )
                    if not source_model_view.is_file():
                        raise RuntimeError(
                            f"{native.upper()} conversion did not persist its "
                            "model-view train split."
                        )
                    model_view_graphs = _load_pickle_graphs(source_model_view)
                    model_view_path = estimates_dir / "ground_truth_model_view.pkl"
                    _atomic_pickle(
                        model_view_path, model_view_graphs[:estimate_count]
                    )
                # Reuse the backend log as the standard estimate log.
                shutil.copy2(
                    estimate_result.log_path,
                    estimates_dir / "generate.log",
                )
                estimates_manifest_path = estimates_dir / "manifest.json"
                estimates_manifest = {
                    "format": TRAINING_ESTIMATES_MANIFEST_FORMAT,
                    "model_id": self.model_id,
                    "dataset_id": request.run.dataset_id,
                    "native_dataset": native,
                    "run_id": request.run.run_id,
                    "training_seed": request.run.train_seed,
                    "generation_seed": estimate_seed,
                    "count": estimate_count,
                    "pairing": {
                        "status": "unpaired",
                        "reason": (
                            "DiGress produces an independent unconditional sample "
                            "pool; a separate declared matching step must construct "
                            "corrector-training pairs."
                        ),
                    },
                    "estimated_graphs": {
                        "path": "estimated_graphs.pkl",
                        "sha256": _sha256(estimated_path),
                    },
                    "ground_truth_graphs": {
                        "path": "ground_truth_graphs.pkl",
                        "sha256": _sha256(ground_truth_path),
                    },
                    "ground_truth_model_view": (
                        {
                            "path": "ground_truth_model_view.pkl",
                            "sha256": _sha256(model_view_path),
                        }
                        if model_view_path is not None
                        else None
                    ),
                    "neutral_export": {
                        "path": "native/digress_samples.npz",
                        "sha256": estimate_result.export_sha256,
                    },
                }
                _atomic_json(estimates_manifest_path, estimates_manifest)
                estimate_summary = {
                    "enabled": True,
                    "count": estimate_count,
                    "manifest": "training_estimates/manifest.json",
                }

            training_manifest = {
                "format": TRAINING_MANIFEST_FORMAT,
                "model_id": self.model_id,
                "dataset": {
                    "benchmark_id": request.dataset.benchmark_id,
                    "serialized_id": request.dataset.serialized_id,
                    "native_id": native,
                    "fingerprint": request.dataset.fingerprint(),
                    **_dataset_profile(request.dataset.benchmark_id, native),
                },
                "run_id": request.run.run_id,
                "train_seed": request.run.train_seed,
                "started_at": started_at,
                "finished_at": _utc_now(),
                "duration_seconds": time.monotonic() - started,
                "upstream": {
                    **source_identity,
                    "experiment": experiment,
                    "config_template": {
                        "dataset": template_dataset,
                        "experiment": template_experiment,
                    },
                    "python_environment": python_identity,
                },
                "training": {
                    "configured_n_epochs": worker_manifest.get(
                        "configured_n_epochs"
                    ),
                    "completed_epochs": worker_manifest.get("completed_epochs"),
                    "global_step": worker_manifest.get("global_step"),
                    "batch_size": worker_manifest.get("batch_size"),
                    "diffusion_steps": worker_manifest.get("diffusion_steps"),
                    "resume_from": (
                        str(request.resume_from.resolve())
                        if request.resume_from is not None
                        else None
                    ),
                },
                "checkpoint": {
                    "path": "checkpoints/model.ckpt",
                    "sha256": _sha256(checkpoint_path),
                },
                "resolved_config": {
                    "path": "resolved_config.yaml",
                    "sha256": _sha256(resolved_config),
                },
                "dataset_conversion": {
                    "path": "dataset_conversion.json",
                    "sha256": _sha256(conversion_manifest_path),
                },
                "molecular_statistics": (
                    {
                        "path": "molecular_statistics.json",
                        "sha256": _sha256(molecular_statistics_path),
                    }
                    if molecular_statistics_path is not None
                    else None
                ),
                "training_estimates": estimate_summary,
                "wrapper_options": _jsonable(options),
                "log": "train.log",
            }
            _atomic_json(stage_train / "manifest.json", training_manifest)
            _atomic_json(
                stage_train / "run.json",
                {
                    "format": "grapher_baseline_run_v1",
                    "model_id": self.model_id,
                    "dataset_id": request.run.dataset_id,
                    "run_id": request.run.run_id,
                    "train_seed": request.run.train_seed,
                },
            )
            _publish_directory(
                stage_train, layout.train_dir, overwrite=request.overwrite
            )
            # run.json belongs at run root under the common contract.
            run_record = layout.train_dir / "run.json"
            if run_record.is_file():
                run_record.replace(layout.run_manifest_path)
            checkpoint_final = layout.checkpoints_dir / "model.ckpt"
            return TrainingArtifacts(
                run_dir=layout.run_dir,
                checkpoint_path=checkpoint_final,
                manifest_path=layout.training_manifest_path,
                log_path=layout.training_log_path,
                artifacts=(
                    layout.resolved_training_config_path,
                    layout.train_dir / "dataset_conversion.json",
                    layout.native_training_dataset_dir,
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
                    if estimates_enabled and native in SUPPORTED_MOLECULAR_DATASETS
                    else None
                ),
                training_estimates_manifest_path=(
                    layout.training_estimates_manifest_path
                    if estimates_enabled
                    else None
                ),
            )
        except BaseException as exc:
            failure = _preserve_failure(
                layout=layout,
                workspace=workspace,
                log_path=log_path,
                error=exc,
                command=command,
            )
            print(
                f"Failure artifacts preserved at: {failure}",
                file=sys.stderr,
                flush=True,
            )
            raise
        finally:
            shutil.rmtree(workspace, ignore_errors=True)

    def generate(self, request: GenerateRequest) -> GenerationArtifacts:
        self.validate_generate_request(request)
        layout = request.run.layout
        managed = layout.training_manifest_path.is_file()
        training_manifest: dict[str, Any] = {}
        if managed:
            training_manifest = _read_json(
                layout.training_manifest_path,
                label="DiGress training manifest",
            )
            if training_manifest.get("format") != TRAINING_MANIFEST_FORMAT:
                raise RuntimeError("Unsupported DiGress training manifest format.")
            checkpoint_record = training_manifest.get("checkpoint")
            if not isinstance(checkpoint_record, Mapping):
                raise RuntimeError("DiGress training manifest has no checkpoint.")
            if str(checkpoint_record.get("sha256", "")) != _sha256(
                request.checkpoint_path
            ):
                raise RuntimeError(
                    "Requested DiGress checkpoint differs from the managed run."
                )

        training_options = training_manifest.get("wrapper_options", {})
        if not isinstance(training_options, Mapping):
            training_options = {}
        options = _deep_update(dict(training_options), request.options)
        dataset_record = training_manifest.get("dataset", {})
        if not isinstance(dataset_record, Mapping):
            dataset_record = {}
        native = _native_dataset(
            request.run.dataset_id,
            options.get("native_dataset") or dataset_record.get("native_id"),
        )
        upstream = training_manifest.get("upstream", {})
        if not isinstance(upstream, Mapping):
            upstream = {}

        from grapher.models.digress.runtime import (
            resolve_digress_python,
            resolve_digress_root,
        )

        runtime = _mapping(options.get("runtime"), name="runtime")
        progress = _progress_options(runtime)
        root = resolve_digress_root(str(options.get("source_env", "DIGRESS")))
        python = resolve_digress_python(
            digress_root=root,
            python_executable=runtime.get("python_executable"),
            python_env=str(options.get("python_env", "DIGRESS_PYTHON")),
        )
        if managed:
            expected_source = upstream
            _verify_source(expected_source, _source_identity(root))
            expected_python = upstream.get("python_environment")
            if isinstance(expected_python, Mapping) and str(
                expected_python.get("python_executable", "")
            ) != str(python.resolve()):
                raise RuntimeError(
                    "Managed DiGress generation resolved a different Python "
                    "interpreter from training."
                )
        timeout_value = runtime.get("timeout_seconds")
        timeout_seconds = None if timeout_value is None else float(timeout_value)
        cuda_value = runtime.get("cuda_visible_devices")
        cuda_visible_devices = None if cuda_value is None else str(cuda_value)
        batch_size = int(
            options.get(
                "generation_batch_size", _default_generation_batch_size(native)
            )
        )
        if batch_size <= 0:
            raise ValueError("generation_batch_size must be positive.")
        if not managed:
            raise RuntimeError(
                "DiGressWrapper.generate currently requires a managed training "
                "run so the exact native dataset and resolved configuration are "
                "available."
            )

        generation_id = request.resolved_generation_id
        target = layout.generation_dir(generation_id)
        ArtifactLayout.require_available(target, overwrite=request.overwrite)
        staging_root = layout.output_root.expanduser().resolve() / ".staging"
        staging_root.mkdir(parents=True, exist_ok=True)
        workspace = Path(tempfile.mkdtemp(prefix="digress-generate-", dir=staging_root))
        stage = workspace / "generation"
        native_dir = stage / "native"
        native_dir.mkdir(parents=True)
        started_at = _utc_now()
        started = time.monotonic()
        try:
            from grapher.models.digress.backend import generate_digress_graphs

            stats_path = layout.train_dir / "molecular_statistics.json"
            if native in SUPPORTED_MOLECULAR_DATASETS and not stats_path.is_file():
                raise RuntimeError(
                    f"Managed DiGress {native.upper()} generation is missing "
                    "its training-time molecular statistics."
                )
            result = generate_digress_graphs(
                digress_root=root,
                python_executable=python,
                dataset=native,
                dataset_datadir=layout.native_training_dataset_dir,
                resolved_config_path=layout.resolved_training_config_path,
                checkpoint_path=request.checkpoint_path,
                output_dir=native_dir,
                num_graphs=request.num_graphs,
                generation_seed=request.generation_seed,
                batch_size=batch_size,
                molecular_statistics_path=(stats_path if stats_path.is_file() else None),
                device=str(runtime.get("device", "auto")),
                cuda_visible_devices=cuda_visible_devices,
                timeout_seconds=timeout_seconds,
                progress_enabled=bool(progress["enabled"]),
                stream_output=bool(progress["stream_output"]),
                progress_interval_seconds=float(progress["interval_seconds"]),
                generation_progress_every_batches=int(
                    progress["generation_batch_interval"]
                ),
            )
            if len(result.graphs) != request.num_graphs:
                raise RuntimeError(
                    f"DiGress returned {len(result.graphs)} graphs; expected "
                    f"{request.num_graphs}."
                )
            graphs_path = stage / "base_graphs.pkl"
            _atomic_pickle(graphs_path, list(result.graphs))
            shutil.copy2(result.log_path, stage / "generate.log")
            graphs_hash = _sha256(graphs_path)
            manifest = {
                "format": GENERATION_MANIFEST_FORMAT,
                "model_id": self.model_id,
                "dataset": {
                    "benchmark_id": request.run.dataset_id,
                    "native_id": native,
                    **_dataset_profile(request.run.dataset_id, native),
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
                "sample_order": "digress_raw_index_ascending",
                "base_graphs": {"path": "base_graphs.pkl", "sha256": graphs_hash},
                "checkpoint": {
                    "path": str(request.checkpoint_path.resolve()),
                    "sha256": _sha256(request.checkpoint_path),
                },
                "resolved_training_config": {
                    "path": str(layout.resolved_training_config_path.resolve()),
                    "sha256": _sha256(layout.resolved_training_config_path),
                },
                "neutral_export": {
                    "path": "native/digress_samples.npz",
                    "sha256": result.export_sha256,
                    "manifest": "native/digress_manifest.json",
                },
                "log": "generate.log",
            }
            _atomic_json(stage / "manifest.json", manifest)
            _publish_directory(stage, target, overwrite=request.overwrite)
            return GenerationArtifacts(
                run_dir=layout.run_dir,
                generation_dir=target,
                graphs_path=layout.generated_graphs_path(generation_id),
                manifest_path=layout.generation_manifest_path(generation_id),
                num_requested=request.num_graphs,
                num_generated=request.num_graphs,
                graphs_sha256=graphs_hash,
                log_path=layout.generation_log_path(generation_id),
                native_artifacts=(
                    layout.native_generation_dir(generation_id)
                    / "digress_samples.npz",
                    layout.native_generation_dir(generation_id)
                    / "digress_manifest.json",
                ),
            )
        finally:
            shutil.rmtree(workspace, ignore_errors=True)

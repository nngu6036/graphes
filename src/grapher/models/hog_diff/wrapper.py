"""GraphER-facing wrapper for the attached HOG-Diff implementation.

HOG-Diff is intentionally executed out of process.  The wrapper preserves its
native two-stage lifecycle -- higher-order VPSDE score training followed by the
conditional OU-bridge score model -- while adapting GraphER's immutable data
splits, artifact layout, exact-count generation contract, and raw molecular
representation.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
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
from grapher.models.hog_diff.codec import (
    HOGDiffDatasetProfile,
    export_dataset,
    load_generated_export,
    profile_for,
)
from grapher.models.hog_diff.runtime import (
    HOGDIFF_PYTHON_ENV,
    HOGDIFF_ROOT_ENV,
    resolve_hogdiff_python,
    resolve_hogdiff_root,
)
from grapher.utils.subprocess_progress import SubprocessLogReporter

TRAINING_MANIFEST_FORMAT = "grapher_hogdiff_training_v1"
TRAINING_ESTIMATES_MANIFEST_FORMAT = "grapher_hogdiff_training_estimates_v1"
GENERATION_MANIFEST_FORMAT = "grapher_hogdiff_generation_v1"

_WRAPPER_OPTION_KEYS = frozenset(
    {
        "source_env",
        "python_env",
        "upstream_config",
        "config_overrides",
        "higher_order",
        "ou",
        "generation_batch_size",
        "num_workers",
        "training_estimates",
        "runtime",
        "native_dataset",
    }
)
_SOURCE_CODE_DIRS = ("models", "utils", "evaluation")


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
        json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _atomic_yaml(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(yaml.safe_dump(_jsonable(value), sort_keys=False), encoding="utf-8")
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


def _load_graphs(path: Path) -> list[Any]:
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"Expected a graph sequence in {path}.")
    return list(value)


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


def _deep_update(base: dict[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            base[key] = _deep_update(dict(base[key]), value)
        else:
            base[key] = value
    return base


def _normalize_optimizer_numbers(config: dict[str, Any]) -> None:
    """Materialize YAML numeric strings before HOG-Diff calls PyTorch.

    PyYAML follows YAML 1.1 scalar resolution, where scientific notation such
    as ``1e-8`` (without a decimal point) is loaded as a string. HOG-Diff's
    published configs use that spelling for Adam epsilon and pass the value
    directly to ``torch.optim``, which requires real numbers.
    """

    numeric_keys = ("lr", "beta1", "eps", "weight_decay", "warmup", "grad_clip")
    for section_name in ("optim", "OUoptim"):
        if section_name not in config:
            continue
        section = _mapping(
            config[section_name],
            name=f"HOG-Diff {section_name} config",
        )
        for key in numeric_keys:
            if key not in section:
                continue
            try:
                section[key] = float(section[key])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"HOG-Diff {section_name}.{key} must be numeric; "
                    f"got {section[key]!r}."
                ) from exc
        config[section_name] = section


def _load_options(request: TrainRequest) -> dict[str, Any]:
    options: dict[str, Any] = {}
    if request.config_path is not None:
        if not request.config_path.is_file():
            raise FileNotFoundError(f"Missing HOG-Diff wrapper config: {request.config_path}")
        loaded = yaml.safe_load(request.config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, Mapping):
            raise TypeError("The HOG-Diff wrapper config must contain a mapping.")
        selected = loaded.get("hog_diff", loaded)
        if not isinstance(selected, Mapping):
            raise TypeError("The hog_diff config section must contain a mapping.")
        options = dict(selected)
    options = _deep_update(options, request.options)
    unknown = set(options).difference(_WRAPPER_OPTION_KEYS)
    if unknown:
        raise ValueError(f"Unknown HOG-Diff wrapper option(s): {sorted(unknown)}.")
    return options


def _progress_options(runtime: Mapping[str, Any]) -> dict[str, Any]:
    raw = runtime.get("progress", {}) or {}
    if not isinstance(raw, Mapping):
        raise TypeError("runtime.progress must be a mapping.")
    enabled = _boolean(raw.get("enabled", False), name="runtime.progress.enabled")
    stream = _boolean(
        raw.get("stream_output", enabled), name="runtime.progress.stream_output"
    )
    interval = float(raw.get("interval_seconds", 30.0))
    if interval <= 0:
        raise ValueError("runtime.progress.interval_seconds must be positive.")
    iteration_value = raw.get("iteration_interval")
    iteration_interval = None if iteration_value is None else int(iteration_value)
    if iteration_interval is not None and iteration_interval <= 0:
        raise ValueError("runtime.progress.iteration_interval must be positive.")
    generation_interval = int(raw.get("generation_batch_interval", 1))
    if generation_interval <= 0:
        raise ValueError("runtime.progress.generation_batch_interval must be positive.")
    return {
        "enabled": enabled,
        "stream_output": stream,
        "interval_seconds": interval,
        "iteration_interval": iteration_interval,
        "generation_batch_interval": generation_interval,
    }


def _emit(progress: Mapping[str, Any], message: str) -> None:
    if bool(progress.get("enabled")):
        print(f"[GraphER/HOG-Diff] {message}", file=sys.stderr, flush=True)


def _source_revision(root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            shell=False,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and value else None


def _source_identity(root: Path) -> dict[str, Any]:
    # Fingerprint every Python source file that can affect training/sampling,
    # including an optional upstream data.py.  Do not recurse through a source-
    # local virtual environment or unrelated data/checkpoint trees.
    candidates = list(root.glob("*.py"))
    for directory in _SOURCE_CODE_DIRS:
        source_dir = root / directory
        if source_dir.is_dir():
            candidates.extend(source_dir.rglob("*.py"))
    files = {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(set(candidates))
        if path.is_file()
    }
    payload = json.dumps(files, sort_keys=True, separators=(",", ":"))
    return {
        "source_root": str(root),
        "revision": _source_revision(root),
        "files": files,
        "source_fingerprint": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "integration_mode": "isolated_upstream_two_stage_score_models",
        "compatibility_shim": (
            "GraphER workers provide only _GENERIC_DATASETS/_MOL_DATASETS when the supplied upstream data.py is absent"
        ),
    }


def _verify_source(expected: Mapping[str, Any], observed: Mapping[str, Any]) -> None:
    if str(expected.get("source_fingerprint", "")) != str(observed.get("source_fingerprint", "")):
        raise RuntimeError(
            "HOG-Diff source differs from the source used for training. Use the same checkout or create a new run."
        )


def _python_identity(python: Path) -> dict[str, Any]:
    script = (
        "import json,platform,sys,numpy,torch,torch_geometric,rdkit,yaml,easydict,wandb;"
        "print(json.dumps({'python_executable':sys.executable,"
        "'python_version':platform.python_version(),'numpy_version':numpy.__version__,"
        "'torch_version':torch.__version__,'torch_geometric_version':torch_geometric.__version__,"
        "'rdkit_version':getattr(rdkit,'__version__','unknown'),"
        "'cuda_available':torch.cuda.is_available()}))"
    )
    completed = subprocess.run(
        [str(python), "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        shell=False,
        timeout=60,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "HOG-Diff Python cannot import its required runtime packages "
            "(torch, torch_geometric, RDKit, PyYAML, easydict, wandb):\n"
            f"{completed.stderr.strip()}"
        )
    value = json.loads(completed.stdout.strip())
    if not isinstance(value, dict):
        raise RuntimeError("HOG-Diff Python identity probe returned invalid JSON.")
    value["python_executable"] = str(python.resolve())
    return value


def _worker(name: str) -> Path:
    path = Path(__file__).resolve().parent / "workers" / name
    if not path.is_file():
        raise FileNotFoundError(f"Missing HOG-Diff worker: {path}")
    return path


def _tail(path: Path, *, lines: int = 200) -> str:
    if not path.is_file():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])


def _publish_directory(stage: Path, target: Path, *, overwrite: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not overwrite:
            raise ArtifactCollisionError(f"Artifact path already exists: {target}")
        shutil.rmtree(target)
    stage.replace(target)


def _resolve_upstream_config(root: Path, profile: HOGDiffDatasetProfile, value: Any) -> Path:
    if value is None:
        path = root / "configs" / profile.config_name
    else:
        candidate = Path(str(value)).expanduser()
        path = candidate if candidate.is_absolute() else root / "configs" / candidate
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing HOG-Diff upstream config: {path}")
    return path


def _resolved_hog_config(
    *,
    source_path: Path,
    options: Mapping[str, Any],
    profile: HOGDiffDatasetProfile,
    dataset_manifest: Mapping[str, Any],
    run_id: str,
    seed: int,
    progress: Mapping[str, Any],
) -> dict[str, Any]:
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, Mapping):
        raise TypeError(f"HOG-Diff source config must contain a mapping: {source_path}")
    config = dict(raw)
    config = _deep_update(config, _mapping(options.get("config_overrides"), name="config_overrides"))
    _normalize_optimizer_numbers(config)
    data = _mapping(config.get("data"), name="HOG-Diff data config")
    data["name"] = profile.native_id
    configured_max = int(data.get("max_node", profile.max_nodes))
    if configured_max != profile.max_nodes:
        raise ValueError(
            f"HOG-Diff {profile.benchmark_id} requires max_node={profile.max_nodes}; got {configured_max}."
        )
    if profile.domain == "attributed":
        channels = int(data.get("atom_channels", len(profile.atomic_numbers)))
        if channels != len(profile.atomic_numbers):
            raise ValueError(
                f"HOG-Diff {profile.benchmark_id} requires atom_channels={len(profile.atomic_numbers)}; got {channels}."
            )
    else:
        projection = dataset_manifest.get("upstream_training_projection", {})
        if not isinstance(projection, Mapping):
            raise RuntimeError("HOG-Diff generic dataset manifest lacks its projection record.")
        data["test_split"] = float(projection["test_split"])
    data["num_workers"] = int(options.get("num_workers", data.get("num_workers", 0)))
    if int(data["num_workers"]) < 0:
        raise ValueError("HOG-Diff num_workers must be non-negative.")
    config["data"] = data

    training = _mapping(config.get("training"), name="HOG-Diff training config")
    ou_training = _mapping(config.get("OUtraining"), name="HOG-Diff OUtraining config")
    training = _deep_update(training, _mapping(options.get("higher_order"), name="higher_order"))
    ou_training = _deep_update(ou_training, _mapping(options.get("ou"), name="ou"))
    for name, section in (("higher_order", training), ("ou", ou_training)):
        section["n_iters"] = int(section["n_iters"])
        section["batch_size"] = int(section["batch_size"])
        if section["n_iters"] < 0 or section["batch_size"] <= 0:
            raise ValueError(f"HOG-Diff {name} n_iters must be non-negative and batch_size positive.")
        section["seed"] = int(seed)
        # Upstream molecular configs can checkpoint-select on test metrics.  The
        # common GraphER protocol instead publishes the final configured state
        # and evaluates only after training, preventing test-set model selection.
        section["snapshot_sampling"] = False
        if progress.get("iteration_interval") is not None:
            section["log_freq"] = int(progress["iteration_interval"])
    config["training"] = training
    config["OUtraining"] = ou_training

    evaluation = _mapping(config.get("eval"), name="HOG-Diff eval config")
    evaluation["seed"] = int(seed)
    evaluation["save_graph"] = False
    config["eval"] = evaluation
    experiment = _mapping(config.get("exp"), name="HOG-Diff exp config")
    experiment["plot"] = False
    config["exp"] = experiment
    config["exp_name"] = f"grapher_{profile.benchmark_id}_{run_id}"
    config["ckpt"] = f"checkpoints/{profile.native_id}/hog_diff.pth"
    return config


def _environment(
    root: Path,
    *,
    data_root: Path,
    runtime_root: Path,
    seed: int,
    device: str,
    cuda_visible_devices: str | None,
) -> tuple[dict[str, str], bool]:
    environment = dict(os.environ)
    existing = environment.get("PYTHONPATH")
    entries = [str(root)]
    if existing:
        entries.append(existing)
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join(entries),
            "PYTHONHASHSEED": str(int(seed)),
            "PYTHONUNBUFFERED": "1",
            "MPLBACKEND": "Agg",
            "WANDB_MODE": "disabled",
            "WANDB_SILENT": "true",
            "DATA_ROOT": str(data_root.resolve()),
            "CKPT_ROOT": str((runtime_root / "checkpoint_runtime").resolve()),
            "LOG_ROOT": str((runtime_root / "log_runtime").resolve()),
            "WANDB_LOG_ROOT": str((runtime_root / "wandb_runtime").resolve()),
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        }
    )
    selected = str(device).strip().lower()
    require_cuda = selected in {"gpu", "cuda"} or selected.startswith("cuda:")
    if selected == "cpu":
        environment["CUDA_VISIBLE_DEVICES"] = ""
        require_cuda = False
    elif cuda_visible_devices is not None:
        environment["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    elif selected.startswith("cuda:"):
        index = selected.split(":", maxsplit=1)[1]
        int(index)  # validate
        environment["CUDA_VISIBLE_DEVICES"] = index
    elif selected not in {"auto", "gpu", "cuda"}:
        raise ValueError("runtime.device must be auto, cpu, gpu, cuda, or cuda:N.")
    return environment, require_cuda


class HOGDiffWrapper(BaseGeneratorWrapper):
    """Train and sample the attached HOG-Diff baseline in isolated workers."""

    model_id = "hog_diff"
    display_name = "HOG-Diff"
    capabilities = BaselineCapabilities(
        domains=frozenset({"generic", "attributed"}), isolation="subprocess", status="ready"
    )
    implementation_note = (
        "Supports Community-small, Ego-small, QM9, and the GraphER ZINC benchmark. "
        "Training preserves HOG-Diff's higher-order then OU-bridge stages; raw molecular "
        "outputs are exported before HOG-Diff's MoFlow/RDKit validity correction."
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
        append: bool,
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
            prefix="GraphER/HOG-Diff",
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
                    f"{label} timed out.\nLog: {log_path}\nCommand: {json.dumps(list(command))}\n"
                    f"Last output:\n{_tail(log_path)}"
                ) from exc
            except BaseException:
                reporter.stop(status="failed")
                raise
            else:
                reporter.stop(
                    status=("completed" if completed.returncode == 0 else f"failed with exit code {completed.returncode}")
                )
        if completed.returncode != 0:
            raise RuntimeError(
                f"{label} exited with code {completed.returncode}.\nWorking directory: {cwd}\n"
                f"Log: {log_path}\nCommand: {json.dumps(list(command))}\nLast output:\n{_tail(log_path)}"
            )

    def _generate_worker(
        self,
        *,
        root: Path,
        python: Path,
        resolved_config: Path,
        checkpoint: Path,
        profile: HOGDiffDatasetProfile,
        data_root: Path,
        output_dir: Path,
        num_graphs: int,
        seed: int,
        batch_size: int,
        runtime: Mapping[str, Any],
        progress: Mapping[str, Any],
        timeout_seconds: float | None,
        log_path: Path,
    ) -> tuple[list[Any], Path, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        export_path = output_dir / "hog_diff_samples.npz"
        export_manifest_path = output_dir / "hog_diff_manifest.json"
        cuda_value = runtime.get("cuda_visible_devices")
        environment, require_cuda = _environment(
            root,
            data_root=data_root,
            runtime_root=output_dir,
            seed=seed,
            device=str(runtime.get("device", "auto")),
            cuda_visible_devices=None if cuda_value is None else str(cuda_value),
        )
        command = [
            str(python),
            str(_worker("generate.py")),
            "--hogdiff-root",
            str(root),
            "--config",
            str(resolved_config),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(export_path),
            "--manifest",
            str(export_manifest_path),
            "--domain",
            profile.domain,
            "--num-graphs",
            str(int(num_graphs)),
            "--batch-size",
            str(int(batch_size)),
            "--seed",
            str(int(seed)),
            "--progress-every-batches",
            str(int(progress["generation_batch_interval"])),
        ]
        if require_cuda:
            command.append("--require-cuda")
        self._run_external(
            command,
            cwd=root,
            environment=environment,
            log_path=log_path,
            timeout_seconds=timeout_seconds,
            label="HOG-Diff two-stage generation",
            progress=progress,
            append=False,
        )
        for artifact in (export_path, export_manifest_path):
            if not artifact.is_file():
                raise RuntimeError(f"HOG-Diff generation worker did not publish {artifact}.")
        manifest = _read_json(export_manifest_path, label="HOG-Diff export manifest")
        if manifest.get("format") != "grapher_hogdiff_export_v1":
            raise RuntimeError("Unsupported HOG-Diff export manifest format.")
        if int(manifest.get("num_generated", -1)) != int(num_graphs):
            raise RuntimeError("HOG-Diff export count does not match the request.")
        output_record = manifest.get("output", {})
        if not isinstance(output_record, Mapping) or str(output_record.get("sha256", "")) != _sha256(export_path):
            raise RuntimeError("HOG-Diff neutral export SHA-256 mismatch.")
        graphs = load_generated_export(export_path, profile=profile)
        if len(graphs) != int(num_graphs):
            raise RuntimeError(f"Decoded {len(graphs)} HOG-Diff graphs; expected {num_graphs}.")
        return graphs, export_path, export_manifest_path

    def train(self, request: TrainRequest) -> TrainingArtifacts:
        self.validate_train_request(request)
        profile = profile_for(request.dataset.benchmark_id)
        options = _load_options(request)
        if options.get("native_dataset") not in {None, profile.native_id}:
            raise ValueError(
                f"HOG-Diff native_dataset for {profile.benchmark_id} must be {profile.native_id!r}."
            )
        source_env = str(options.get("source_env", HOGDIFF_ROOT_ENV))
        python_env = str(options.get("python_env", HOGDIFF_PYTHON_ENV))
        root = resolve_hogdiff_root(source_env)
        runtime = _mapping(options.get("runtime"), name="runtime")
        progress = _progress_options(runtime)
        python = resolve_hogdiff_python(
            hogdiff_root=root,
            python_executable=runtime.get("python_executable"),
            python_env=python_env,
        )
        source_identity = _source_identity(root)
        python_identity = _python_identity(python)
        timeout_value = runtime.get("timeout_seconds")
        timeout_seconds = None if timeout_value is None else float(timeout_value)
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("runtime.timeout_seconds must be positive.")

        layout = request.run.layout
        if request.overwrite and layout.generations_dir.is_dir() and any(layout.generations_dir.iterdir()):
            raise ArtifactCollisionError(
                "Cannot overwrite a trained HOG-Diff run that already has raw generation batches. Use a new run_id."
            )
        ArtifactLayout.require_available(layout.train_dir, overwrite=request.overwrite)
        staging_root = layout.output_root.expanduser().resolve() / ".staging"
        staging_root.mkdir(parents=True, exist_ok=True)
        workspace = Path(tempfile.mkdtemp(prefix="hogdiff-train-", dir=staging_root))
        stage_train = workspace / "train"
        stage_train.mkdir()
        native_dataset = stage_train / "native_dataset"
        native_dataset.mkdir()
        native_runtime = workspace / "native_runtime"
        native_runtime.mkdir()
        log_path = stage_train / "train.log"
        started_at = _utc_now()
        started = time.monotonic()
        try:
            split_graphs = {
                split: _load_graphs(path) for split, path in request.dataset.split_paths.items()
            }
            _emit(progress, "projecting immutable GraphER splits into HOG-Diff's native dataset representation")
            dataset_manifest = export_dataset(
                train_graphs=split_graphs["train"],
                val_graphs=split_graphs["val"],
                test_graphs=split_graphs["test"],
                profile=profile,
                output_dir=native_dataset,
            )
            dataset_manifest_path = native_dataset / "manifest.json"
            source_config = _resolve_upstream_config(root, profile, options.get("upstream_config"))
            resolved_config = _resolved_hog_config(
                source_path=source_config,
                options=options,
                profile=profile,
                dataset_manifest=dataset_manifest,
                run_id=request.run.run_id,
                seed=request.run.train_seed,
                progress=progress,
            )
            resolved_config_path = stage_train / "resolved_config.yaml"
            _atomic_yaml(resolved_config_path, resolved_config)
            checkpoint_path = stage_train / "checkpoints" / "hog_diff.pth"
            checkpoint_path.parent.mkdir(parents=True)
            molecular_npz = (
                native_dataset / profile.native_id / "processed" / "grapher_atom_bond.npz"
                if profile.domain == "attributed"
                else None
            )
            cuda_value = runtime.get("cuda_visible_devices")
            environment, require_cuda = _environment(
                root,
                data_root=native_dataset,
                runtime_root=native_runtime,
                seed=request.run.train_seed,
                device=str(runtime.get("device", "auto")),
                cuda_visible_devices=None if cuda_value is None else str(cuda_value),
            )
            stage_manifests: dict[str, dict[str, Any]] = {}
            for stage_index, mode in enumerate(("higher-order", "OU")):
                mode_manifest = stage_train / f"{('higher_order' if mode == 'higher-order' else 'ou')}_worker_manifest.json"
                command = [
                    str(python),
                    str(_worker("train.py")),
                    "--hogdiff-root",
                    str(root),
                    "--config",
                    str(resolved_config_path),
                    "--mode",
                    mode,
                    "--checkpoint",
                    str(checkpoint_path),
                    "--manifest",
                    str(mode_manifest),
                    "--seed",
                    str(request.run.train_seed),
                    "--num-workers",
                    str(int(options.get("num_workers", 0))),
                ]
                if request.resume_from is not None:
                    if not request.resume_from.is_file():
                        raise FileNotFoundError(request.resume_from)
                    command.extend(["--resume-from", str(request.resume_from.resolve())])
                if molecular_npz is not None:
                    command.extend(["--molecular-npz", str(molecular_npz)])
                if require_cuda:
                    command.append("--require-cuda")
                _emit(
                    progress,
                    "training HOG-Diff phase 1/2 (higher-order VPSDE)"
                    if mode == "higher-order"
                    else "training HOG-Diff phase 2/2 (conditional OU bridge)",
                )
                self._run_external(
                    command,
                    cwd=root,
                    environment=environment,
                    log_path=log_path,
                    timeout_seconds=timeout_seconds,
                    label=f"HOG-Diff {mode} training",
                    progress=progress,
                    append=stage_index > 0,
                )
                if not mode_manifest.is_file():
                    raise RuntimeError(f"HOG-Diff {mode} worker published no manifest.")
                stage_manifests[mode] = _read_json(mode_manifest, label=f"HOG-Diff {mode} worker manifest")
            if not checkpoint_path.is_file() or checkpoint_path.stat().st_size <= 0:
                raise RuntimeError("HOG-Diff final two-stage checkpoint was not published.")

            estimate_cfg = _mapping(options.get("training_estimates"), name="training_estimates")
            estimates_enabled = _boolean(
                estimate_cfg.get("enabled", False), name="training_estimates.enabled"
            )
            estimate_summary: dict[str, Any] = {"enabled": False}
            if estimates_enabled:
                train_graphs = split_graphs["train"]
                estimate_count = int(estimate_cfg.get("num_graphs", len(train_graphs)))
                if estimate_count <= 0 or estimate_count > len(train_graphs):
                    raise ValueError(
                        "training_estimates.num_graphs must be between 1 and the training split size "
                        f"({len(train_graphs)})."
                    )
                estimate_seed = int(estimate_cfg.get("seed", request.run.train_seed + 1_000_003))
                batch_size = int(options.get("generation_batch_size", resolved_config["eval"]["batch_size"]))
                if batch_size <= 0:
                    raise ValueError("generation_batch_size must be positive.")
                estimates_dir = stage_train / "training_estimates"
                native_estimates = estimates_dir / "native"
                _emit(progress, f"generating {estimate_count} unpaired post-training HOG-Diff estimates")
                estimated_graphs, estimate_export, estimate_export_manifest = self._generate_worker(
                    root=root,
                    python=python,
                    resolved_config=resolved_config_path,
                    checkpoint=checkpoint_path,
                    profile=profile,
                    data_root=native_dataset,
                    output_dir=native_estimates,
                    num_graphs=estimate_count,
                    seed=estimate_seed,
                    batch_size=batch_size,
                    runtime=runtime,
                    progress=progress,
                    timeout_seconds=timeout_seconds,
                    log_path=estimates_dir / "generate.log",
                )
                estimated_path = estimates_dir / "estimated_graphs.pkl"
                ground_truth_path = estimates_dir / "ground_truth_graphs.pkl"
                _atomic_pickle(estimated_path, estimated_graphs)
                _atomic_pickle(ground_truth_path, train_graphs[:estimate_count])
                estimates_manifest = {
                    "format": TRAINING_ESTIMATES_MANIFEST_FORMAT,
                    "model_id": self.model_id,
                    "dataset_id": request.dataset.benchmark_id,
                    "count": estimate_count,
                    "seed": estimate_seed,
                    "pairing": "unpaired_independent_samples",
                    "estimated_graphs": {"path": "estimated_graphs.pkl", "sha256": _sha256(estimated_path)},
                    "ground_truth_graphs": {"path": "ground_truth_graphs.pkl", "sha256": _sha256(ground_truth_path)},
                    "neutral_export": {
                        "path": f"native/{estimate_export.name}",
                        "sha256": _sha256(estimate_export),
                        "manifest": f"native/{estimate_export_manifest.name}",
                    },
                }
                _atomic_json(estimates_dir / "manifest.json", estimates_manifest)
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
                    "native_id": profile.native_id,
                    "fingerprint": request.dataset.fingerprint(),
                    "domain": profile.domain,
                    "representation": (
                        "simple_undirected_topology"
                        if profile.domain == "generic"
                        else "implicit_hydrogen_atom_bond_graph"
                    ),
                },
                "run_id": request.run.run_id,
                "train_seed": request.run.train_seed,
                "started_at": started_at,
                "finished_at": _utc_now(),
                "duration_seconds": time.monotonic() - started,
                "upstream": {**source_identity, "python_environment": python_identity},
                "upstream_config": {
                    "source_path": str(source_config),
                    "source_sha256": _sha256(source_config),
                    "resolved_path": "resolved_config.yaml",
                    "resolved_sha256": _sha256(resolved_config_path),
                },
                "protocol_adaptation": {
                    "two_stage_training": ["higher-order", "OU"],
                    "snapshot_sampling_disabled": True,
                    "reason": "avoid test-set checkpoint selection; publish final configured states",
                    "validation_used_for_training": False,
                    "molecular_generation_correction": "disabled_in_wrapper_export",
                },
                "training_stages": {
                    mode: {
                        "configured_n_iters": stage_manifests[mode].get("configured_n_iters"),
                        "configured_batch_size": stage_manifests[mode].get("configured_batch_size"),
                        "initial_step": stage_manifests[mode].get("initial_step"),
                        "checkpoint_step": stage_manifests[mode].get("checkpoint_step"),
                        "device": stage_manifests[mode].get("device"),
                        "manifest": (
                            "higher_order_worker_manifest.json" if mode == "higher-order" else "ou_worker_manifest.json"
                        ),
                    }
                    for mode in ("higher-order", "OU")
                },
                "checkpoint": {"path": "checkpoints/hog_diff.pth", "sha256": _sha256(checkpoint_path)},
                "dataset_conversion": {
                    "path": "native_dataset/manifest.json",
                    "sha256": _sha256(dataset_manifest_path),
                },
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
            _publish_directory(stage_train, layout.train_dir, overwrite=request.overwrite)
            run_record = layout.train_dir / "run.json"
            if run_record.is_file():
                run_record.replace(layout.run_manifest_path)
            return TrainingArtifacts(
                run_dir=layout.run_dir,
                checkpoint_path=layout.checkpoints_dir / "hog_diff.pth",
                manifest_path=layout.training_manifest_path,
                log_path=layout.training_log_path,
                artifacts=(
                    layout.resolved_training_config_path,
                    layout.native_training_dataset_dir,
                    layout.train_dir / "higher_order_worker_manifest.json",
                    layout.train_dir / "ou_worker_manifest.json",
                ),
                estimated_graphs_path=(layout.estimated_training_graphs_path if estimates_enabled else None),
                ground_truth_graphs_path=(layout.ground_truth_training_graphs_path if estimates_enabled else None),
                training_estimates_manifest_path=(layout.training_estimates_manifest_path if estimates_enabled else None),
            )
        finally:
            shutil.rmtree(workspace, ignore_errors=True)

    def generate(self, request: GenerateRequest) -> GenerationArtifacts:
        self.validate_generate_request(request)
        profile = profile_for(request.run.dataset_id)
        layout = request.run.layout
        if not layout.training_manifest_path.is_file():
            raise RuntimeError("HOGDiffWrapper.generate requires a managed HOG-Diff training run.")
        training_manifest = _read_json(layout.training_manifest_path, label="HOG-Diff training manifest")
        if training_manifest.get("format") != TRAINING_MANIFEST_FORMAT:
            raise RuntimeError("Unsupported HOG-Diff training manifest format.")
        checkpoint_record = training_manifest.get("checkpoint")
        if not isinstance(checkpoint_record, Mapping):
            raise RuntimeError("HOG-Diff training manifest has no checkpoint record.")
        if str(checkpoint_record.get("sha256", "")) != _sha256(request.checkpoint_path):
            raise RuntimeError("Requested HOG-Diff checkpoint differs from the managed run.")
        dataset_record = training_manifest.get("dataset", {})
        if not isinstance(dataset_record, Mapping) or str(dataset_record.get("benchmark_id")) != profile.benchmark_id:
            raise RuntimeError("HOG-Diff managed training dataset identity is inconsistent with the run.")
        if not layout.resolved_training_config_path.is_file() or not layout.native_training_dataset_dir.is_dir():
            raise RuntimeError("Managed HOG-Diff resolved config/native dataset is missing.")

        training_options = training_manifest.get("wrapper_options", {})
        if not isinstance(training_options, Mapping):
            training_options = {}
        generation_options = dict(request.options)
        unknown_generation = set(generation_options).difference({"runtime", "generation_batch_size"})
        if unknown_generation:
            raise ValueError(
                "HOG-Diff generation may override only runtime and generation_batch_size; got "
                f"{sorted(unknown_generation)}. Sampling/model settings are frozen in the managed training config."
            )
        options = _deep_update(dict(training_options), generation_options)
        runtime = _mapping(options.get("runtime"), name="runtime")
        progress = _progress_options(runtime)
        root = resolve_hogdiff_root(str(options.get("source_env", HOGDIFF_ROOT_ENV)))
        python = resolve_hogdiff_python(
            hogdiff_root=root,
            python_executable=runtime.get("python_executable"),
            python_env=str(options.get("python_env", HOGDIFF_PYTHON_ENV)),
        )
        upstream = training_manifest.get("upstream", {})
        if not isinstance(upstream, Mapping):
            raise RuntimeError("HOG-Diff training manifest has no upstream identity.")
        _verify_source(upstream, _source_identity(root))
        expected_python = upstream.get("python_environment")
        if isinstance(expected_python, Mapping) and str(expected_python.get("python_executable", "")) != str(python.resolve()):
            raise RuntimeError("Managed HOG-Diff generation resolved a different Python interpreter from training.")
        timeout_value = runtime.get("timeout_seconds")
        timeout_seconds = None if timeout_value is None else float(timeout_value)
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("runtime.timeout_seconds must be positive.")
        resolved = yaml.safe_load(layout.resolved_training_config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(resolved, Mapping):
            raise TypeError("Managed HOG-Diff resolved config is invalid.")
        batch_size = int(options.get("generation_batch_size", (resolved.get("eval") or {}).get("batch_size", 1)))
        if batch_size <= 0:
            raise ValueError("generation_batch_size must be positive.")

        generation_id = request.resolved_generation_id
        target = layout.generation_dir(generation_id)
        ArtifactLayout.require_available(target, overwrite=request.overwrite)
        staging_root = layout.output_root.expanduser().resolve() / ".staging"
        staging_root.mkdir(parents=True, exist_ok=True)
        workspace = Path(tempfile.mkdtemp(prefix="hogdiff-generate-", dir=staging_root))
        stage = workspace / "generation"
        native_dir = stage / "native"
        native_dir.mkdir(parents=True)
        started_at = _utc_now()
        started = time.monotonic()
        try:
            _emit(progress, f"generating {request.num_graphs} raw HOG-Diff samples through both score stages")
            graphs, export_path, export_manifest_path = self._generate_worker(
                root=root,
                python=python,
                resolved_config=layout.resolved_training_config_path,
                checkpoint=request.checkpoint_path,
                profile=profile,
                data_root=layout.native_training_dataset_dir,
                output_dir=native_dir,
                num_graphs=request.num_graphs,
                seed=request.generation_seed,
                batch_size=batch_size,
                runtime=runtime,
                progress=progress,
                timeout_seconds=timeout_seconds,
                log_path=stage / "generate.log",
            )
            graphs_path = stage / "base_graphs.pkl"
            _atomic_pickle(graphs_path, graphs)
            graphs_hash = _sha256(graphs_path)
            export_manifest = _read_json(export_manifest_path, label="HOG-Diff export manifest")
            generation_manifest = {
                "format": GENERATION_MANIFEST_FORMAT,
                "model_id": self.model_id,
                "dataset": {
                    "benchmark_id": profile.benchmark_id,
                    "native_id": profile.native_id,
                    "domain": profile.domain,
                    "representation": (
                        "simple_undirected_topology"
                        if profile.domain == "generic"
                        else "implicit_hydrogen_atom_bond_graph"
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
                "returned_count": len(graphs),
                "sample_order": "hog_diff_worker_index_ascending",
                "two_stage_generation": ["higher-order VPSDE", "conditional OU bridge"],
                "molecular_posthoc_correction": False,
                "base_graphs": {"path": "base_graphs.pkl", "sha256": graphs_hash},
                "checkpoint": {
                    "path": str(request.checkpoint_path.resolve()),
                    "sha256": _sha256(request.checkpoint_path),
                },
                "neutral_export": {
                    "path": f"native/{export_path.name}",
                    "sha256": _sha256(export_path),
                    "manifest": f"native/{export_manifest_path.name}",
                },
                "native_diagnostics": {
                    key: export_manifest.get(key)
                    for key in ("batch_size", "sampling_rounds", "device", "postprocessing")
                },
                "log": "generate.log",
            }
            _atomic_json(stage / "manifest.json", generation_manifest)
            _publish_directory(stage, target, overwrite=request.overwrite)
            return GenerationArtifacts(
                run_dir=layout.run_dir,
                generation_dir=target,
                graphs_path=layout.generated_graphs_path(generation_id),
                manifest_path=layout.generation_manifest_path(generation_id),
                num_requested=request.num_graphs,
                num_generated=len(graphs),
                graphs_sha256=graphs_hash,
                log_path=layout.generation_log_path(generation_id),
                native_artifacts=(
                    layout.native_generation_dir(generation_id) / "hog_diff_samples.npz",
                    layout.native_generation_dir(generation_id) / "hog_diff_manifest.json",
                ),
            )
        finally:
            shutil.rmtree(workspace, ignore_errors=True)

"""GraphES-facing wrapper for the attached external GraphRNN codebase.

The legacy upstream entry point is not imported into the GraphES process.
Prepared NetworkX graphs cross the subprocess boundary as neutral binary
adjacency tensors; current-PyTorch compatibility workers instantiate the
upstream ``GRU_plain`` and ``MLP_plain`` modules, train the requested GraphRNN
variant, and export generated adjacency tensors for validation and decoding.
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
from grapher.models.graphrnn.codec import export_graphrnn_dataset
from grapher.utils.subprocess_progress import SubprocessLogReporter

TRAINING_MANIFEST_FORMAT = "grapher_graphrnn_training_v1"
TRAINING_ESTIMATES_MANIFEST_FORMAT = "grapher_graphrnn_training_estimates_v1"
GENERATION_MANIFEST_FORMAT = "grapher_graphrnn_generation_v1"
SUPPORTED_BENCHMARKS = frozenset({"community_small", "ego_small", "grid"})

_WRAPPER_OPTION_KEYS = frozenset(
    {"source_env", "python_env", "runtime", "training_estimates", "native_dataset"}
)
_MODEL_OPTION_KEYS = frozenset(
    {
        "variant",
        "max_num_node",
        "max_prev_node",
        "hidden_size_rnn",
        "hidden_size_rnn_output",
        "embedding_size_rnn",
        "embedding_size_rnn_output",
        "embedding_size_output",
        "num_layers",
        "batch_size",
        "batch_ratio",
        "epochs",
        "learning_rate",
        "milestones",
        "lr_rate",
        "scheduler_step_unit",
        "num_workers",
        "save_every_epochs",
        "log_every_epochs",
        "gradient_clip_norm",
        "sample_time",
        "generation_batch_size",
        "deterministic",
        "torch_num_threads",
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


def _atomic_yaml(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        yaml.safe_dump(_jsonable(value), sort_keys=False), encoding="utf-8"
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


def _load_graphs(path: Path) -> list[Any]:
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
                f"Missing GraphRNN wrapper config: {request.config_path}"
            )
        loaded = yaml.safe_load(request.config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, Mapping):
            raise TypeError("The GraphRNN wrapper config must contain a mapping.")
        selected = loaded.get("graphrnn", loaded)
        if not isinstance(selected, Mapping):
            raise TypeError("The graphrnn config section must contain a mapping.")
        options = dict(selected)
    options = _deep_update(options, request.options)
    unknown = set(options).difference(_WRAPPER_OPTION_KEYS | _MODEL_OPTION_KEYS)
    if unknown:
        raise ValueError(f"Unknown GraphRNN wrapper option(s): {sorted(unknown)}.")
    return options


def _model_options(options: Mapping[str, Any], *, inferred_max_nodes: int) -> dict[str, Any]:
    selected = {
        key: value for key, value in options.items() if key in _MODEL_OPTION_KEYS
    }
    raw_max = selected.get("max_num_node")
    if raw_max is None:
        selected["max_num_node"] = int(inferred_max_nodes)
    else:
        selected["max_num_node"] = int(raw_max)
    if int(selected["max_num_node"]) < int(inferred_max_nodes):
        raise ValueError(
            "GraphRNN max_num_node is smaller than an input graph: "
            f"configured={selected['max_num_node']}, observed={inferred_max_nodes}."
        )
    return selected


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
        print(f"[GraphES/GraphRNN] {message}", file=sys.stderr, flush=True)


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
    files = {}
    for relative in ("model.py", "data.py", "train.py", "args.py", "README.md"):
        path = root / relative
        if path.is_file():
            files[relative] = _sha256(path)
    payload = json.dumps(files, sort_keys=True, separators=(",", ":"))
    return {
        "source_root": str(root),
        "revision": _source_revision(root),
        "files": files,
        "source_fingerprint": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "compatibility_mode": (
            "upstream_model_modules_with_modern_training_and_sampling_worker"
        ),
    }


def _verify_source(expected: Mapping[str, Any], observed: Mapping[str, Any]) -> None:
    expected_fingerprint = str(expected.get("source_fingerprint", ""))
    if not expected_fingerprint or expected_fingerprint != str(
        observed.get("source_fingerprint", "")
    ):
        raise RuntimeError(
            "GraphRNN source differs from the source used for training. "
            "Use the same attached checkout or start a new run."
        )


def _python_identity(python: Path) -> dict[str, Any]:
    script = (
        "import json,platform,sys,numpy,torch;"
        "print(json.dumps({'python_executable':sys.executable,"
        "'python_version':platform.python_version(),"
        "'numpy_version':numpy.__version__,'torch_version':torch.__version__,"
        "'cuda_available':torch.cuda.is_available()}))"
    )
    completed = subprocess.run(
        [str(python), "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        shell=False,
        timeout=30,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "GraphRNN Python cannot import the required runtime packages "
            f"(torch and numpy):\n{completed.stderr.strip()}"
        )
    value = json.loads(completed.stdout.strip())
    if not isinstance(value, dict):
        raise RuntimeError("GraphRNN Python identity probe returned invalid JSON.")
    value["python_executable"] = str(python.resolve())
    return value


def _external_environment(
    root: Path,
    *,
    seed: int,
    cuda_visible_devices: str | None,
) -> dict[str, str]:
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
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        }
    )
    if cuda_visible_devices is not None:
        environment["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    return environment


def _worker(name: str) -> Path:
    path = Path(__file__).resolve().parent / "workers" / name
    if not path.is_file():
        raise FileNotFoundError(f"Missing GraphRNN worker: {path}")
    return path


def _publish_directory(stage: Path, target: Path, *, overwrite: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not overwrite:
            raise ArtifactCollisionError(f"Artifact path already exists: {target}")
        shutil.rmtree(target)
    stage.replace(target)


def _tail(path: Path, *, lines: int = 200) -> str:
    if not path.is_file():
        return ""
    return "\n".join(
        path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
    )


class GraphRNNWrapper(BaseGeneratorWrapper):
    """Train and sample the attached GraphRNN implementation in isolation."""

    model_id = "graphrnn"
    display_name = "GraphRNN"
    capabilities = BaselineCapabilities(
        domains=frozenset({"generic"}), isolation="subprocess", status="ready"
    )
    implementation_note = (
        "Supports generic unlabelled graphs. The attached GraphRNN formulation "
        "does not jointly generate molecular node and edge categories."
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
            prefix="GraphES/GraphRNN",
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
        if request.dataset.benchmark_id not in SUPPORTED_BENCHMARKS:
            raise ValueError(
                f"GraphRNNWrapper supports {sorted(SUPPORTED_BENCHMARKS)}; got "
                f"{request.dataset.benchmark_id!r}."
            )
        options = _load_options(request)
        from grapher.models.graphrnn.runtime import (
            resolve_graphrnn_python,
            resolve_graphrnn_root,
        )

        source_env = str(options.get("source_env", "GRAPHRNN"))
        python_env = str(options.get("python_env", "GRAPHRNN_PYTHON"))
        root = resolve_graphrnn_root(source_env)
        runtime = _mapping(options.get("runtime"), name="runtime")
        progress = _progress_options(runtime)
        python = resolve_graphrnn_python(
            graphrnn_root=root,
            python_executable=runtime.get("python_executable"),
            python_env=python_env,
        )
        source_identity = _source_identity(root)
        python_identity = _python_identity(python)
        timeout_value = runtime.get("timeout_seconds")
        timeout_seconds = None if timeout_value is None else float(timeout_value)
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("runtime.timeout_seconds must be positive.")
        cuda_value = runtime.get("cuda_visible_devices")
        cuda_visible_devices = None if cuda_value is None else str(cuda_value)
        device = str(runtime.get("device", "auto"))

        layout = request.run.layout
        if (
            request.overwrite
            and layout.generations_dir.is_dir()
            and any(layout.generations_dir.iterdir())
        ):
            raise ArtifactCollisionError(
                "Cannot overwrite a trained GraphRNN run that already has raw "
                "generation batches. Use a new run_id."
            )
        ArtifactLayout.require_available(layout.train_dir, overwrite=request.overwrite)
        staging_root = layout.output_root.expanduser().resolve() / ".staging"
        staging_root.mkdir(parents=True, exist_ok=True)
        workspace = Path(tempfile.mkdtemp(prefix="graphrnn-train-", dir=staging_root))
        stage_train = workspace / "train"
        stage_train.mkdir()
        native_dataset = stage_train / "native_dataset"
        native_dataset.mkdir()
        native_run = workspace / "native_run"
        native_run.mkdir()
        log_path = stage_train / "train.log"
        started_at = _utc_now()
        started = time.monotonic()
        try:
            split_graphs = {
                split: _load_graphs(path)
                for split, path in request.dataset.split_paths.items()
            }
            observed_max = max(
                graph.number_of_nodes()
                for graphs in split_graphs.values()
                for graph in graphs
            )
            model_options = _model_options(
                options, inferred_max_nodes=int(observed_max)
            )
            # The runtime progress interval is a presentation-level alias for
            # the worker's stable epoch progress cadence. An explicit CLI
            # progress override should not require editing the model section.
            if progress.get("epoch_interval") is not None:
                model_options["log_every_epochs"] = int(
                    progress["epoch_interval"]
                )
            dataset_export = native_dataset / "graphrnn_dataset.npz"
            dataset_manifest_path = native_dataset / "manifest.json"
            _emit(progress, "converting immutable GraphES splits to neutral adjacency tensors")
            dataset_manifest = export_graphrnn_dataset(
                split_graphs,
                output_path=dataset_export,
                manifest_path=dataset_manifest_path,
                benchmark_id=request.dataset.benchmark_id,
                max_num_node=int(model_options["max_num_node"]),
            )
            worker_config_path = workspace / "graphrnn_worker_config.json"
            _atomic_json(worker_config_path, model_options)
            worker_manifest_path = native_run / "training_worker_manifest.json"
            command = [
                str(python),
                str(_worker("train.py")),
                "--graphrnn-root",
                str(root),
                "--dataset",
                str(dataset_export),
                "--dataset-manifest",
                str(dataset_manifest_path),
                "--config",
                str(worker_config_path),
                "--output-dir",
                str(native_run),
                "--manifest",
                str(worker_manifest_path),
                "--seed",
                str(request.run.train_seed),
                "--device",
                device,
            ]
            if request.resume_from is not None:
                if not request.resume_from.is_file():
                    raise FileNotFoundError(request.resume_from)
                command.extend(["--resume-from", str(request.resume_from.resolve())])
            environment = _external_environment(
                root,
                seed=request.run.train_seed,
                cuda_visible_devices=cuda_visible_devices,
            )
            _emit(
                progress,
                "starting isolated GraphRNN training with the upstream GRU modules",
            )
            self._run_external(
                command,
                cwd=root,
                environment=environment,
                log_path=log_path,
                timeout_seconds=timeout_seconds,
                label="GraphRNN training",
                progress=progress,
                append=False,
            )
            if not worker_manifest_path.is_file():
                raise RuntimeError("GraphRNN training worker published no manifest.")
            worker_manifest = _read_json(
                worker_manifest_path, label="GraphRNN worker manifest"
            )
            resolved_worker_config = worker_manifest.get("resolved_config")
            if not isinstance(resolved_worker_config, Mapping):
                raise RuntimeError(
                    "GraphRNN training worker manifest has no resolved config."
                )
            worker_checkpoint = Path(str(worker_manifest.get("checkpoint", "")))
            if not worker_checkpoint.is_file():
                raise RuntimeError(
                    f"GraphRNN worker checkpoint is missing: {worker_checkpoint}"
                )
            worker_checkpoints = native_run / "checkpoints"
            if not worker_checkpoints.is_dir():
                raise RuntimeError("GraphRNN worker published no checkpoint directory.")
            shutil.copytree(worker_checkpoints, stage_train / "checkpoints")
            checkpoint_path = stage_train / "checkpoints" / "graphrnn.pt"
            if not checkpoint_path.is_file():
                raise RuntimeError("GraphRNN final checkpoint was not published.")
            history = native_run / "loss_history.jsonl"
            if history.is_file():
                shutil.copy2(history, stage_train / "loss_history.jsonl")

            # The worker executes below a temporary staging directory. Publish
            # a durable, relocatable copy of its manifest rather than retaining
            # paths that disappear as soon as this transaction completes.
            published_worker_manifest = dict(worker_manifest)
            published_worker_manifest["checkpoint"] = "checkpoints/graphrnn.pt"
            published_worker_manifest["history"] = (
                "loss_history.jsonl" if history.is_file() else None
            )
            worker_dataset_export = worker_manifest.get("dataset_export")
            if isinstance(worker_dataset_export, Mapping):
                published_dataset_export = dict(worker_dataset_export)
                published_dataset_export.update(
                    {
                        "path": "native_dataset/graphrnn_dataset.npz",
                        "manifest": "native_dataset/manifest.json",
                    }
                )
                published_worker_manifest["dataset_export"] = (
                    published_dataset_export
                )
            _atomic_json(
                stage_train / "worker_manifest.json", published_worker_manifest
            )
            _atomic_yaml(
                stage_train / "resolved_config.yaml",
                {
                    "graphrnn": {
                        **dict(resolved_worker_config),
                        "runtime": _jsonable(runtime),
                        "training_estimates": _jsonable(
                            options.get("training_estimates", {})
                        ),
                    }
                },
            )

            estimate_cfg = _mapping(
                options.get("training_estimates"), name="training_estimates"
            )
            estimates_enabled = _boolean(
                estimate_cfg.get("enabled", False),
                name="training_estimates.enabled",
            )
            estimate_summary: dict[str, Any] = {"enabled": False}
            if estimates_enabled:
                from grapher.models.graphrnn.backend import (
                    generate_graphrnn_graphs,
                )

                train_graphs = split_graphs["train"]
                estimate_count = int(
                    estimate_cfg.get("num_graphs", len(train_graphs))
                )
                if estimate_count <= 0 or estimate_count > len(train_graphs):
                    raise ValueError(
                        "training_estimates.num_graphs must be between 1 and the "
                        f"training split size ({len(train_graphs)})."
                    )
                estimate_seed = int(
                    estimate_cfg.get("seed", request.run.train_seed + 1_000_003)
                )
                generation_batch_size = int(
                    model_options.get("generation_batch_size", 32)
                )
                estimates_dir = stage_train / "training_estimates"
                native_estimates = estimates_dir / "native"
                _emit(
                    progress,
                    f"generating {estimate_count} unpaired post-training estimates",
                )
                estimate_result = generate_graphrnn_graphs(
                    graphrnn_root=root,
                    python_executable=python,
                    checkpoint_path=checkpoint_path,
                    output_dir=native_estimates,
                    num_graphs=estimate_count,
                    generation_seed=estimate_seed,
                    batch_size=generation_batch_size,
                    device=device,
                    sample_time=(
                        int(model_options["sample_time"])
                        if model_options.get("sample_time") is not None
                        else None
                    ),
                    cuda_visible_devices=cuda_visible_devices,
                    timeout_seconds=timeout_seconds,
                    progress_enabled=bool(progress["enabled"]),
                    stream_output=bool(progress["stream_output"]),
                    progress_interval_seconds=float(progress["interval_seconds"]),
                    generation_progress_every_batches=int(
                        progress["generation_batch_interval"]
                    ),
                )
                # This generation occurs before the staged training directory
                # is published. Rewrite the checkpoint reference so the native
                # estimate manifest remains valid after the atomic move.
                estimate_native_manifest_path = (
                    native_estimates / "graphrnn_manifest.json"
                )
                estimate_native_manifest = _read_json(
                    estimate_native_manifest_path,
                    label="GraphRNN training-estimate manifest",
                )
                estimate_checkpoint = estimate_native_manifest.get("checkpoint")
                if isinstance(estimate_checkpoint, Mapping):
                    durable_checkpoint = dict(estimate_checkpoint)
                    durable_checkpoint["path"] = "../../checkpoints/graphrnn.pt"
                    estimate_native_manifest["checkpoint"] = durable_checkpoint
                _atomic_json(
                    estimate_native_manifest_path, estimate_native_manifest
                )
                estimated_path = estimates_dir / "estimated_graphs.pkl"
                ground_truth_path = estimates_dir / "ground_truth_graphs.pkl"
                _atomic_pickle(estimated_path, list(estimate_result.graphs))
                _atomic_pickle(ground_truth_path, train_graphs[:estimate_count])
                shutil.copy2(estimate_result.log_path, estimates_dir / "generate.log")
                estimates_manifest = {
                    "format": TRAINING_ESTIMATES_MANIFEST_FORMAT,
                    "model_id": self.model_id,
                    "dataset_id": request.run.dataset_id,
                    "run_id": request.run.run_id,
                    "training_seed": request.run.train_seed,
                    "generation_seed": estimate_seed,
                    "count": estimate_count,
                    "pairing": {
                        "status": "unpaired",
                        "reason": (
                            "GraphRNN produces an independent unconditional sample "
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
                    "neutral_export": {
                        "path": "native/graphrnn_samples.npz",
                        "sha256": estimate_result.export_sha256,
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
                    "native_id": request.dataset.native_id,
                    "fingerprint": request.dataset.fingerprint(),
                    "domain": "generic",
                    "representation": "simple_undirected_topology",
                },
                "run_id": request.run.run_id,
                "train_seed": request.run.train_seed,
                "started_at": started_at,
                "finished_at": _utc_now(),
                "duration_seconds": time.monotonic() - started,
                "upstream": {
                    **source_identity,
                    "python_environment": python_identity,
                },
                "training": {
                    "configured_epochs": worker_manifest.get("configured_epochs"),
                    "completed_epochs": worker_manifest.get("checkpoint_epoch"),
                    "resumed_from_epoch": worker_manifest.get(
                        "resumed_from_epoch"
                    ),
                    "last_loss": worker_manifest.get("last_loss"),
                    "device": worker_manifest.get("device"),
                },
                "checkpoint": {
                    "path": "checkpoints/graphrnn.pt",
                    "sha256": _sha256(checkpoint_path),
                },
                "resolved_config": {
                    "path": "resolved_config.yaml",
                    "sha256": _sha256(stage_train / "resolved_config.yaml"),
                },
                "dataset_conversion": {
                    "path": "native_dataset/manifest.json",
                    "sha256": _sha256(dataset_manifest_path),
                    "neutral_export_sha256": str(
                        (dataset_manifest.get("output") or {}).get("sha256", "")
                    ),
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
            _publish_directory(
                stage_train, layout.train_dir, overwrite=request.overwrite
            )
            run_record = layout.train_dir / "run.json"
            if run_record.is_file():
                run_record.replace(layout.run_manifest_path)
            return TrainingArtifacts(
                run_dir=layout.run_dir,
                checkpoint_path=layout.checkpoints_dir / "graphrnn.pt",
                manifest_path=layout.training_manifest_path,
                log_path=layout.training_log_path,
                artifacts=(
                    layout.resolved_training_config_path,
                    layout.native_training_dataset_dir,
                    layout.train_dir / "worker_manifest.json",
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
                training_estimates_manifest_path=(
                    layout.training_estimates_manifest_path
                    if estimates_enabled
                    else None
                ),
            )
        finally:
            shutil.rmtree(workspace, ignore_errors=True)

    def generate(self, request: GenerateRequest) -> GenerationArtifacts:
        self.validate_generate_request(request)
        layout = request.run.layout
        if not layout.training_manifest_path.is_file():
            raise RuntimeError(
                "GraphRNNWrapper.generate requires a managed GraphRNN training run."
            )
        training_manifest = _read_json(
            layout.training_manifest_path, label="GraphRNN training manifest"
        )
        if training_manifest.get("format") != TRAINING_MANIFEST_FORMAT:
            raise RuntimeError("Unsupported GraphRNN training manifest format.")
        checkpoint_record = training_manifest.get("checkpoint")
        if not isinstance(checkpoint_record, Mapping):
            raise RuntimeError("GraphRNN training manifest has no checkpoint record.")
        if str(checkpoint_record.get("sha256", "")) != _sha256(
            request.checkpoint_path
        ):
            raise RuntimeError(
                "Requested GraphRNN checkpoint differs from the managed run."
            )

        training_options = training_manifest.get("wrapper_options", {})
        if not isinstance(training_options, Mapping):
            training_options = {}
        options = _deep_update(dict(training_options), request.options)
        runtime = _mapping(options.get("runtime"), name="runtime")
        progress = _progress_options(runtime)
        from grapher.models.graphrnn.runtime import (
            resolve_graphrnn_python,
            resolve_graphrnn_root,
        )

        root = resolve_graphrnn_root(str(options.get("source_env", "GRAPHRNN")))
        python = resolve_graphrnn_python(
            graphrnn_root=root,
            python_executable=runtime.get("python_executable"),
            python_env=str(options.get("python_env", "GRAPHRNN_PYTHON")),
        )
        upstream = training_manifest.get("upstream", {})
        if not isinstance(upstream, Mapping):
            raise RuntimeError("GraphRNN training manifest has no upstream identity.")
        _verify_source(upstream, _source_identity(root))
        expected_python = upstream.get("python_environment")
        if isinstance(expected_python, Mapping) and str(
            expected_python.get("python_executable", "")
        ) != str(python.resolve()):
            raise RuntimeError(
                "Managed GraphRNN generation resolved a different Python "
                "interpreter from training."
            )
        timeout_value = runtime.get("timeout_seconds")
        timeout_seconds = None if timeout_value is None else float(timeout_value)
        cuda_value = runtime.get("cuda_visible_devices")
        cuda_visible_devices = None if cuda_value is None else str(cuda_value)
        device = str(runtime.get("device", "auto"))
        batch_size = int(options.get("generation_batch_size", 32))
        if batch_size <= 0:
            raise ValueError("generation_batch_size must be positive.")
        sample_value = options.get("sample_time")
        sample_time = None if sample_value is None else int(sample_value)

        generation_id = request.resolved_generation_id
        target = layout.generation_dir(generation_id)
        ArtifactLayout.require_available(target, overwrite=request.overwrite)
        staging_root = layout.output_root.expanduser().resolve() / ".staging"
        staging_root.mkdir(parents=True, exist_ok=True)
        workspace = Path(
            tempfile.mkdtemp(prefix="graphrnn-generate-", dir=staging_root)
        )
        stage = workspace / "generation"
        native_dir = stage / "native"
        native_dir.mkdir(parents=True)
        started_at = _utc_now()
        started = time.monotonic()
        try:
            from grapher.models.graphrnn.backend import generate_graphrnn_graphs

            result = generate_graphrnn_graphs(
                graphrnn_root=root,
                python_executable=python,
                checkpoint_path=request.checkpoint_path,
                output_dir=native_dir,
                num_graphs=request.num_graphs,
                generation_seed=request.generation_seed,
                batch_size=batch_size,
                device=device,
                sample_time=sample_time,
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
                    f"GraphRNN returned {len(result.graphs)} graphs; expected "
                    f"{request.num_graphs}."
                )
            graphs_path = stage / "base_graphs.pkl"
            _atomic_pickle(graphs_path, list(result.graphs))
            shutil.copy2(result.log_path, stage / "generate.log")
            graphs_hash = _sha256(graphs_path)
            generation_manifest = {
                "format": GENERATION_MANIFEST_FORMAT,
                "model_id": self.model_id,
                "dataset": {
                    "benchmark_id": request.run.dataset_id,
                    "domain": "generic",
                    "representation": "simple_undirected_topology",
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
                "sample_order": "graphrnn_worker_index_ascending",
                "base_graphs": {"path": "base_graphs.pkl", "sha256": graphs_hash},
                "checkpoint": {
                    "path": str(request.checkpoint_path.resolve()),
                    "sha256": _sha256(request.checkpoint_path),
                },
                "neutral_export": {
                    "path": "native/graphrnn_samples.npz",
                    "sha256": result.export_sha256,
                    "manifest": "native/graphrnn_manifest.json",
                },
                "raw_diagnostics": result.manifest.get("statistics"),
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
                num_generated=request.num_graphs,
                graphs_sha256=graphs_hash,
                log_path=layout.generation_log_path(generation_id),
                native_artifacts=(
                    layout.native_generation_dir(generation_id)
                    / "graphrnn_samples.npz",
                    layout.native_generation_dir(generation_id)
                    / "graphrnn_manifest.json",
                ),
            )
        finally:
            shutil.rmtree(workspace, ignore_errors=True)

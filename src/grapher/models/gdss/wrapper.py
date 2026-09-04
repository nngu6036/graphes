"""GraphER-facing wrapper for the attached GDSS implementation.

GDSS is executed out of process so its legacy PyTorch environment remains
isolated.  GraphER supplies immutable numeric train/validation/test projections;
GDSS optimizes on train, monitors validation in place of the upstream test
loader, and never touches the frozen GraphER test split during training.
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
from grapher.models.gdss.codec import GDSSDatasetProfile, export_dataset, load_generated_export, profile_for
from grapher.models.gdss.runtime import (
    GDSS_PYTHON_ENV,
    GDSS_ROOT_ENV,
    resolve_gdss_python,
    resolve_gdss_root,
)
from grapher.utils.subprocess_progress import SubprocessLogReporter

TRAINING_MANIFEST_FORMAT = "grapher_gdss_training_v1"
TRAINING_ESTIMATES_MANIFEST_FORMAT = "grapher_gdss_training_estimates_v1"
GENERATION_MANIFEST_FORMAT = "grapher_gdss_generation_v1"

_WRAPPER_OPTION_KEYS = frozenset(
    {
        "source_env",
        "python_env",
        "upstream_config",
        "sampling_config",
        "config_overrides",
        "train",
        "sampler",
        "sample",
        "batch_size",
        "num_workers",
        "generation_batch_size",
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
    temporary.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
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


def _load_options(request: TrainRequest) -> dict[str, Any]:
    options: dict[str, Any] = {}
    if request.config_path is not None:
        if not request.config_path.is_file():
            raise FileNotFoundError(f"Missing GDSS wrapper config: {request.config_path}")
        loaded = yaml.safe_load(request.config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, Mapping):
            raise TypeError("The GDSS wrapper config must contain a mapping.")
        selected = loaded.get("gdss", loaded)
        if not isinstance(selected, Mapping):
            raise TypeError("The gdss config section must contain a mapping.")
        options = dict(selected)
    options = _deep_update(options, request.options)
    unknown = set(options).difference(_WRAPPER_OPTION_KEYS)
    if unknown:
        raise ValueError(f"Unknown GDSS wrapper option(s): {sorted(unknown)}.")
    return options


def _progress_options(runtime: Mapping[str, Any]) -> dict[str, Any]:
    raw = runtime.get("progress", {}) or {}
    if not isinstance(raw, Mapping):
        raise TypeError("runtime.progress must be a mapping.")
    enabled = _boolean(raw.get("enabled", False), name="runtime.progress.enabled")
    stream = _boolean(raw.get("stream_output", enabled), name="runtime.progress.stream_output")
    interval = float(raw.get("interval_seconds", 30.0))
    if interval <= 0:
        raise ValueError("runtime.progress.interval_seconds must be positive.")
    epoch_value = raw.get("epoch_interval")
    epoch_interval = None if epoch_value is None else int(epoch_value)
    if epoch_interval is not None and epoch_interval <= 0:
        raise ValueError("runtime.progress.epoch_interval must be positive.")
    generation_interval = int(raw.get("generation_batch_interval", 1))
    if generation_interval <= 0:
        raise ValueError("runtime.progress.generation_batch_interval must be positive.")
    return {
        "enabled": enabled,
        "stream_output": stream,
        "interval_seconds": interval,
        "epoch_interval": epoch_interval,
        "generation_batch_interval": generation_interval,
    }


def _emit(progress: Mapping[str, Any], message: str) -> None:
    if bool(progress.get("enabled")):
        print(f"[GraphER/GDSS] {message}", file=sys.stderr, flush=True)


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
        "integration_mode": "isolated_upstream_joint_sde_with_grapher_data_adapter",
    }


def _verify_source(expected: Mapping[str, Any], observed: Mapping[str, Any]) -> None:
    if str(expected.get("source_fingerprint", "")) != str(observed.get("source_fingerprint", "")):
        raise RuntimeError(
            "GDSS source differs from the source used for training. Use the same checkout or create a new run."
        )


def _python_identity(python: Path) -> dict[str, Any]:
    script = (
        "import json,platform,sys,numpy,torch,networkx,scipy,yaml,tqdm;"
        "print(json.dumps({'python_executable':sys.executable,'python_version':platform.python_version(),"
        "'numpy_version':numpy.__version__,'torch_version':torch.__version__,"
        "'networkx_version':networkx.__version__,'scipy_version':scipy.__version__,"
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
            "GDSS Python cannot import its required runtime packages "
            "(torch, NumPy, NetworkX, SciPy, PyYAML, tqdm):\n"
            f"{completed.stderr.strip()}"
        )
    value = json.loads(completed.stdout.strip())
    if not isinstance(value, dict):
        raise RuntimeError("GDSS Python identity probe returned invalid JSON.")
    value["python_executable"] = str(python.resolve())
    return value


def _worker(name: str) -> Path:
    path = Path(__file__).resolve().parent / "workers" / name
    if not path.is_file():
        raise FileNotFoundError(f"Missing GDSS worker: {path}")
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


def _resolve_config_path(root: Path, value: Any, default_name: str) -> Path:
    if value is None:
        path = root / "config" / default_name
    else:
        candidate = Path(str(value)).expanduser()
        path = candidate if candidate.is_absolute() else root / "config" / candidate
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing GDSS upstream config: {path}")
    return path


def _resolved_gdss_config(
    *,
    source_path: Path,
    sampling_path: Path | None,
    options: Mapping[str, Any],
    profile: GDSSDatasetProfile,
    run_id: str,
    seed: int,
    progress: Mapping[str, Any],
) -> dict[str, Any]:
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, Mapping):
        raise TypeError(f"GDSS source config must contain a mapping: {source_path}")
    config = dict(raw)
    config = _deep_update(config, _mapping(options.get("config_overrides"), name="config_overrides"))

    if sampling_path is not None:
        sampling_raw = yaml.safe_load(sampling_path.read_text(encoding="utf-8")) or {}
        if not isinstance(sampling_raw, Mapping):
            raise TypeError(f"GDSS sampling config must contain a mapping: {sampling_path}")
        for section in ("sampler", "sample"):
            if section in sampling_raw:
                config[section] = dict(sampling_raw[section])

    data = _mapping(config.get("data"), name="GDSS data config")
    data["data"] = profile.native_id
    configured_max = int(data.get("max_node_num", profile.max_nodes))
    configured_feat = int(data.get("max_feat_num", profile.max_feat_num))
    if configured_max != profile.max_nodes:
        raise ValueError(
            f"GDSS {profile.benchmark_id} requires max_node_num={profile.max_nodes}; got {configured_max}."
        )
    if configured_feat != profile.max_feat_num:
        raise ValueError(
            f"GDSS {profile.benchmark_id} requires max_feat_num={profile.max_feat_num}; got {configured_feat}."
        )
    data["max_node_num"] = profile.max_nodes
    data["max_feat_num"] = profile.max_feat_num
    data["dir"] = "."
    data["num_workers"] = int(options.get("num_workers", data.get("num_workers", 0)))
    if int(data["num_workers"]) < 0:
        raise ValueError("GDSS num_workers must be non-negative.")
    if options.get("batch_size") is not None:
        data["batch_size"] = int(options["batch_size"])
    data["batch_size"] = int(data["batch_size"])
    if data["batch_size"] <= 0:
        raise ValueError("GDSS training batch_size must be positive.")
    config["data"] = data

    train = _mapping(config.get("train"), name="GDSS train config")
    train = _deep_update(train, _mapping(options.get("train"), name="train"))
    train["num_epochs"] = int(train["num_epochs"])
    if train["num_epochs"] <= 0:
        raise ValueError("GDSS train.num_epochs must be positive.")
    # Publish the final configured state deterministically. Intermediate native
    # checkpoints are not needed by the managed GraphER run.
    train["save_interval"] = train["num_epochs"]
    if progress.get("epoch_interval") is not None:
        train["print_interval"] = int(progress["epoch_interval"])
    train["name"] = f"grapher_{profile.benchmark_id}_{run_id}"
    config["train"] = train

    sampler = _mapping(config.get("sampler"), name="GDSS sampler config")
    sampler = _deep_update(sampler, _mapping(options.get("sampler"), name="sampler"))
    sample = _mapping(config.get("sample"), name="GDSS sample config")
    sample = _deep_update(sample, _mapping(options.get("sample"), name="sample"))
    required_sampler = {"predictor", "corrector", "snr", "scale_eps", "n_steps"}
    required_sample = {"use_ema", "noise_removal", "probability_flow", "eps"}
    if required_sampler.difference(sampler):
        raise ValueError(f"GDSS sampler config is incomplete: missing {sorted(required_sampler.difference(sampler))}.")
    if required_sample.difference(sample):
        raise ValueError(f"GDSS sample config is incomplete: missing {sorted(required_sample.difference(sample))}.")
    sample["seed"] = int(seed)
    config["sampler"] = sampler
    config["sample"] = sample
    config["seed"] = int(seed)
    return config


def _environment(
    root: Path,
    *,
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
        int(index)
        environment["CUDA_VISIBLE_DEVICES"] = index
    elif selected not in {"auto", "gpu", "cuda"}:
        raise ValueError("runtime.device must be auto, cpu, gpu, cuda, or cuda:N.")
    return environment, require_cuda


class GDSSWrapper(BaseGeneratorWrapper):
    """Train and sample the attached GDSS baseline in isolated workers."""

    model_id = "gdss"
    display_name = "GDSS"
    capabilities = BaselineCapabilities(
        domains=frozenset({"generic", "attributed"}), isolation="subprocess", status="ready"
    )
    implementation_note = (
        "Supports Community-small, Ego-small, Grid, heavy-atom QM9, and the GraphER ZINC benchmark. "
        "Training uses GraphER train/validation splits and keeps test frozen; molecular generation exports "
        "raw GDSS categorical graphs before valence correction or largest-component filtering."
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
            prefix="GraphER/GDSS",
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
                reporter.stop(status="completed" if completed.returncode == 0 else f"failed with exit code {completed.returncode}")
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
        profile: GDSSDatasetProfile,
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
        export_path = output_dir / "gdss_samples.npz"
        export_manifest_path = output_dir / "gdss_manifest.json"
        cuda_value = runtime.get("cuda_visible_devices")
        environment, require_cuda = _environment(
            root,
            seed=seed,
            device=str(runtime.get("device", "auto")),
            cuda_visible_devices=None if cuda_value is None else str(cuda_value),
        )
        command = [
            str(python),
            str(_worker("generate.py")),
            "--gdss-root", str(root),
            "--config", str(resolved_config),
            "--dataset-dir", str(data_root),
            "--checkpoint", str(checkpoint),
            "--output", str(export_path),
            "--manifest", str(export_manifest_path),
            "--domain", profile.domain,
            "--num-graphs", str(int(num_graphs)),
            "--batch-size", str(int(batch_size)),
            "--seed", str(int(seed)),
            "--atom-channels", str(len(profile.atomic_numbers)),
            "--progress-every-batches", str(int(progress["generation_batch_interval"])),
        ]
        if require_cuda:
            command.append("--require-cuda")
        self._run_external(
            command,
            cwd=root,
            environment=environment,
            log_path=log_path,
            timeout_seconds=timeout_seconds,
            label="GDSS generation",
            progress=progress,
            append=False,
        )
        for artifact in (export_path, export_manifest_path):
            if not artifact.is_file():
                raise RuntimeError(f"GDSS generation worker did not publish {artifact}.")
        manifest = _read_json(export_manifest_path, label="GDSS export manifest")
        if manifest.get("format") != "grapher_gdss_export_v1":
            raise RuntimeError("Unsupported GDSS export manifest format.")
        if int(manifest.get("num_generated", -1)) != int(num_graphs):
            raise RuntimeError("GDSS export count does not match the request.")
        output_record = manifest.get("output", {})
        if not isinstance(output_record, Mapping) or str(output_record.get("sha256", "")) != _sha256(export_path):
            raise RuntimeError("GDSS neutral export SHA-256 mismatch.")
        graphs = load_generated_export(export_path, profile=profile)
        if len(graphs) != int(num_graphs):
            raise RuntimeError(f"Decoded {len(graphs)} GDSS graphs; expected {num_graphs}.")
        return graphs, export_path, export_manifest_path

    def train(self, request: TrainRequest) -> TrainingArtifacts:
        self.validate_train_request(request)
        if request.resume_from is not None:
            raise ValueError("The attached GDSS checkpoint does not include optimizer/scheduler state; managed resume is unsupported.")
        profile = profile_for(request.dataset.benchmark_id)
        options = _load_options(request)
        if options.get("native_dataset") not in {None, profile.native_id}:
            raise ValueError(f"GDSS native_dataset for {profile.benchmark_id} must be {profile.native_id!r}.")
        source_env = str(options.get("source_env", GDSS_ROOT_ENV))
        python_env = str(options.get("python_env", GDSS_PYTHON_ENV))
        root = resolve_gdss_root(source_env)
        runtime = _mapping(options.get("runtime"), name="runtime")
        progress = _progress_options(runtime)
        python = resolve_gdss_python(
            gdss_root=root,
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
                "Cannot overwrite a trained GDSS run that already has raw generation batches. Use a new run_id."
            )
        ArtifactLayout.require_available(layout.train_dir, overwrite=request.overwrite)
        staging_root = layout.output_root.expanduser().resolve() / ".staging"
        staging_root.mkdir(parents=True, exist_ok=True)
        workspace = Path(tempfile.mkdtemp(prefix="gdss-train-", dir=staging_root))
        stage_train = workspace / "train"
        stage_train.mkdir()
        native_dataset = stage_train / "native_dataset"
        native_dataset.mkdir()
        log_path = stage_train / "train.log"
        started_at = _utc_now()
        started = time.monotonic()
        try:
            split_graphs = {split: _load_graphs(path) for split, path in request.dataset.split_paths.items()}
            _emit(progress, "projecting GraphER splits into neutral GDSS tensors")
            dataset_manifest = export_dataset(
                train_graphs=split_graphs["train"],
                val_graphs=split_graphs["val"],
                test_graphs=split_graphs["test"],
                profile=profile,
                output_dir=native_dataset,
            )
            dataset_manifest_path = native_dataset / "manifest.json"
            source_config = _resolve_config_path(root, options.get("upstream_config"), profile.config_name)
            sampling_config = (
                _resolve_config_path(root, options.get("sampling_config"), profile.sampling_config_name)
                if profile.sampling_config_name is not None
                else (
                    _resolve_config_path(root, options.get("sampling_config"), profile.config_name)
                    if options.get("sampling_config") is not None else None
                )
            )
            resolved_config = _resolved_gdss_config(
                source_path=source_config,
                sampling_path=sampling_config,
                options=options,
                profile=profile,
                run_id=request.run.run_id,
                seed=request.run.train_seed,
                progress=progress,
            )
            resolved_config_path = stage_train / "resolved_config.yaml"
            _atomic_yaml(resolved_config_path, resolved_config)

            checkpoint_path = stage_train / "checkpoints" / "gdss.pth"
            worker_manifest_path = stage_train / "training_worker_manifest.json"
            cuda_value = runtime.get("cuda_visible_devices")
            environment, require_cuda = _environment(
                root,
                seed=request.run.train_seed,
                device=str(runtime.get("device", "auto")),
                cuda_visible_devices=None if cuda_value is None else str(cuda_value),
            )
            command = [
                str(python), str(_worker("train.py")),
                "--gdss-root", str(root),
                "--config", str(resolved_config_path),
                "--dataset-dir", str(native_dataset),
                "--checkpoint", str(checkpoint_path),
                "--manifest", str(worker_manifest_path),
                "--domain", profile.domain,
                "--seed", str(int(request.run.train_seed)),
            ]
            if require_cuda:
                command.append("--require-cuda")
            _emit(progress, f"training joint GDSS score networks for {resolved_config['train']['num_epochs']} epochs")
            self._run_external(
                command,
                cwd=root,
                environment=environment,
                log_path=log_path,
                timeout_seconds=timeout_seconds,
                label="GDSS training",
                progress=progress,
                append=False,
            )
            if not checkpoint_path.is_file() or not worker_manifest_path.is_file():
                raise RuntimeError("GDSS training worker did not publish its managed artifacts.")
            worker_manifest = _read_json(worker_manifest_path, label="GDSS training worker manifest")
            if worker_manifest.get("format") != "grapher_gdss_training_worker_v1":
                raise RuntimeError("Unsupported GDSS training worker manifest format.")
            worker_ckpt = worker_manifest.get("checkpoint", {})
            if not isinstance(worker_ckpt, Mapping) or str(worker_ckpt.get("sha256", "")) != _sha256(checkpoint_path):
                raise RuntimeError("GDSS training checkpoint SHA-256 mismatch.")

            estimate_cfg = _mapping(options.get("training_estimates"), name="training_estimates")
            estimates_enabled = _boolean(estimate_cfg.get("enabled", False), name="training_estimates.enabled")
            estimate_summary: dict[str, Any] = {"enabled": False}
            if estimates_enabled:
                train_graphs = split_graphs["train"]
                estimate_count = int(estimate_cfg.get("num_graphs", len(train_graphs)))
                if estimate_count < 1 or estimate_count > len(train_graphs):
                    raise ValueError(
                        "training_estimates.num_graphs must be between 1 and the training split size."
                    )
                estimate_seed = int(estimate_cfg.get("seed", request.run.train_seed))
                generation_batch_size = int(options.get("generation_batch_size", resolved_config["data"]["batch_size"]))
                estimates_dir = stage_train / "training_estimates"
                native_estimates = estimates_dir / "native"
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
                    batch_size=generation_batch_size,
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
                    "pairing": {"status": "unpaired", "method": "independent_gdss_samples"},
                    "estimated_graphs": {"path": "estimated_graphs.pkl", "sha256": _sha256(estimated_path)},
                    "ground_truth_graphs": {"path": "ground_truth_graphs.pkl", "sha256": _sha256(ground_truth_path)},
                    "neutral_export": {
                        "path": f"native/{estimate_export.name}",
                        "sha256": _sha256(estimate_export),
                        "manifest": f"native/{estimate_export_manifest.name}",
                    },
                }
                _atomic_json(estimates_dir / "manifest.json", estimates_manifest)
                estimate_summary = {"enabled": True, "count": estimate_count, "manifest": "training_estimates/manifest.json"}

            training_manifest = {
                "format": TRAINING_MANIFEST_FORMAT,
                "model_id": self.model_id,
                "dataset": {
                    "benchmark_id": request.dataset.benchmark_id,
                    "serialized_id": request.dataset.serialized_id,
                    "native_id": profile.native_id,
                    "fingerprint": request.dataset.fingerprint(),
                    "domain": profile.domain,
                    "representation": "simple_undirected_topology" if profile.domain == "generic" else "implicit_hydrogen_atom_bond_graph",
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
                    "sampling_path": str(sampling_config) if sampling_config is not None else None,
                    "sampling_sha256": _sha256(sampling_config) if sampling_config is not None else None,
                    "resolved_path": "resolved_config.yaml",
                    "resolved_sha256": _sha256(resolved_config_path),
                },
                "protocol_adaptation": {
                    "optimizer_split": "train",
                    "monitor_split": "val",
                    "test_used_during_training": False,
                    "checkpoint_selection": "final_configured_epoch",
                    "molecular_generation_correction": "disabled_in_wrapper_export",
                    "molecular_largest_component_filter": False,
                },
                "training": {
                    "configured_num_epochs": worker_manifest.get("configured_num_epochs"),
                    "configured_batch_size": worker_manifest.get("configured_batch_size"),
                    "device": worker_manifest.get("device"),
                    "manifest": "training_worker_manifest.json",
                },
                "checkpoint": {"path": "checkpoints/gdss.pth", "sha256": _sha256(checkpoint_path)},
                "dataset_conversion": {"path": "native_dataset/manifest.json", "sha256": _sha256(dataset_manifest_path)},
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
                checkpoint_path=layout.checkpoints_dir / "gdss.pth",
                manifest_path=layout.training_manifest_path,
                log_path=layout.training_log_path,
                artifacts=(
                    layout.resolved_training_config_path,
                    layout.native_training_dataset_dir,
                    layout.train_dir / "training_worker_manifest.json",
                ),
                estimated_graphs_path=layout.estimated_training_graphs_path if estimates_enabled else None,
                ground_truth_graphs_path=layout.ground_truth_training_graphs_path if estimates_enabled else None,
                training_estimates_manifest_path=layout.training_estimates_manifest_path if estimates_enabled else None,
            )
        finally:
            shutil.rmtree(workspace, ignore_errors=True)

    def generate(self, request: GenerateRequest) -> GenerationArtifacts:
        self.validate_generate_request(request)
        profile = profile_for(request.run.dataset_id)
        layout = request.run.layout
        if not layout.training_manifest_path.is_file():
            raise RuntimeError("GDSSWrapper.generate requires a managed GDSS training run.")
        training_manifest = _read_json(layout.training_manifest_path, label="GDSS training manifest")
        if training_manifest.get("format") != TRAINING_MANIFEST_FORMAT:
            raise RuntimeError("Unsupported GDSS training manifest format.")
        checkpoint_record = training_manifest.get("checkpoint")
        if not isinstance(checkpoint_record, Mapping) or str(checkpoint_record.get("sha256", "")) != _sha256(request.checkpoint_path):
            raise RuntimeError("Requested GDSS checkpoint differs from the managed run.")
        dataset_record = training_manifest.get("dataset", {})
        if not isinstance(dataset_record, Mapping) or str(dataset_record.get("benchmark_id")) != profile.benchmark_id:
            raise RuntimeError("GDSS managed training dataset identity is inconsistent with the run.")
        if not layout.resolved_training_config_path.is_file() or not layout.native_training_dataset_dir.is_dir():
            raise RuntimeError("Managed GDSS resolved config/native dataset is missing.")

        training_options = training_manifest.get("wrapper_options", {})
        if not isinstance(training_options, Mapping):
            training_options = {}
        generation_options = dict(request.options)
        unknown_generation = set(generation_options).difference({"runtime", "generation_batch_size"})
        if unknown_generation:
            raise ValueError(
                "GDSS generation may override only runtime and generation_batch_size; got "
                f"{sorted(unknown_generation)}. Sampler/model settings are frozen in the managed training config."
            )
        options = _deep_update(dict(training_options), generation_options)
        runtime = _mapping(options.get("runtime"), name="runtime")
        progress = _progress_options(runtime)
        root = resolve_gdss_root(str(options.get("source_env", GDSS_ROOT_ENV)))
        python = resolve_gdss_python(
            gdss_root=root,
            python_executable=runtime.get("python_executable"),
            python_env=str(options.get("python_env", GDSS_PYTHON_ENV)),
        )
        upstream = training_manifest.get("upstream", {})
        if not isinstance(upstream, Mapping):
            raise RuntimeError("GDSS training manifest has no upstream identity.")
        _verify_source(upstream, _source_identity(root))
        expected_python = upstream.get("python_environment")
        if isinstance(expected_python, Mapping) and str(expected_python.get("python_executable", "")) != str(python.resolve()):
            raise RuntimeError("Managed GDSS generation resolved a different Python interpreter from training.")
        timeout_value = runtime.get("timeout_seconds")
        timeout_seconds = None if timeout_value is None else float(timeout_value)
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("runtime.timeout_seconds must be positive.")
        resolved = yaml.safe_load(layout.resolved_training_config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(resolved, Mapping):
            raise TypeError("Managed GDSS resolved config is invalid.")
        default_batch = int((resolved.get("data") or {}).get("batch_size", 1))
        batch_size = int(options.get("generation_batch_size", default_batch))
        if batch_size <= 0:
            raise ValueError("generation_batch_size must be positive.")

        generation_id = request.resolved_generation_id
        target = layout.generation_dir(generation_id)
        ArtifactLayout.require_available(target, overwrite=request.overwrite)
        staging_root = layout.output_root.expanduser().resolve() / ".staging"
        staging_root.mkdir(parents=True, exist_ok=True)
        workspace = Path(tempfile.mkdtemp(prefix="gdss-generate-", dir=staging_root))
        stage = workspace / "generation"
        native_dir = stage / "native"
        native_dir.mkdir(parents=True)
        started_at = _utc_now()
        started = time.monotonic()
        try:
            _emit(progress, f"generating {request.num_graphs} raw GDSS samples")
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
            export_manifest = _read_json(export_manifest_path, label="GDSS export manifest")
            generation_manifest = {
                "format": GENERATION_MANIFEST_FORMAT,
                "model_id": self.model_id,
                "dataset": {
                    "benchmark_id": profile.benchmark_id,
                    "native_id": profile.native_id,
                    "domain": profile.domain,
                    "representation": "simple_undirected_topology" if profile.domain == "generic" else "implicit_hydrogen_atom_bond_graph",
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
                "sample_order": "gdss_worker_index_ascending",
                "molecular_posthoc_correction": False,
                "molecular_largest_component_filter": False,
                "base_graphs": {"path": "base_graphs.pkl", "sha256": graphs_hash},
                "checkpoint": {"path": str(request.checkpoint_path.resolve()), "sha256": _sha256(request.checkpoint_path)},
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
                    layout.native_generation_dir(generation_id) / "gdss_samples.npz",
                    layout.native_generation_dir(generation_id) / "gdss_manifest.json",
                ),
            )
        finally:
            shutil.rmtree(workspace, ignore_errors=True)

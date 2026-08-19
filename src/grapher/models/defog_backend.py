"""Isolated DeFoG base-generator adapter for GraphER graph batches.

The attached DeFoG reference implementation is not an installable, namespaced
library: it mixes ``src.*`` imports with bare ``datasets``/``models`` imports
and targets a different dependency stack.  This module therefore never imports
DeFoG in the GraphER process.  A small worker runs under the DeFoG interpreter
and publishes a numeric ``.npz`` export, which is then validated here before it
is converted to NetworkX graphs.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from grapher.utils.io import ensure_dir
from grapher.utils.subprocess_progress import SubprocessLogReporter

DEFOG_EXPORT_FORMAT = "defog_graph_batch_v2"
LEGACY_DEFOG_EXPORT_FORMAT = "defog_generic_topology_v1"
DEFOG_ROOT_ENV = "DEFOG"
DEFOG_PYTHON_ENV = "DEFOG_PYTHON"
SUPPORTED_GENERIC_DATASETS = frozenset({"comm20", "planar", "sbm", "tree"})
SUPPORTED_MOLECULAR_DATASETS = frozenset({"qm9", "zinc"})
SUPPORTED_DATASETS = SUPPORTED_GENERIC_DATASETS | SUPPORTED_MOLECULAR_DATASETS
_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9_.-]+$")
_SAFE_DEVICE = re.compile(r"^(?:auto|cpu|cuda(?::[0-9]+)?)$")
_SAFE_ENV_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _normalize_generic_graph(graph: nx.Graph) -> nx.Graph:
    """Normalize a generic graph without importing Torch-heavy rewiring code."""

    if graph.is_directed() or graph.is_multigraph():
        raise ValueError("A generic DeFoG export must be simple and undirected.")
    normalized = nx.convert_node_labels_to_integers(
        nx.Graph(graph), first_label=0, ordering="sorted"
    )
    if nx.number_of_selfloops(normalized):
        raise ValueError("A generic DeFoG export cannot contain self-loops.")
    return normalized


@dataclass(frozen=True)
class DeFoGGeneratorConfig:
    """Configuration for one isolated DeFoG sampling/export run."""

    dataset: str
    experiment: str
    checkpoint_path: Path | None = None
    generated_path: Path | None = None
    manifest_path: Path | None = None
    dataset_datadir: Path | None = None
    resolved_config_path: Path | None = None
    molecular_statistics_path: Path | None = None
    source_env: str = DEFOG_ROOT_ENV
    python_env: str = DEFOG_PYTHON_ENV
    python_executable: str | None = None
    device: str = "auto"
    batch_size: int = 64
    sample_steps: int | None = None
    eta: float | None = None
    omega: float | None = None
    time_distortion: str | None = None
    rdb: str | None = None
    rdb_crit: str | None = None
    timeout_seconds: float | None = None
    cuda_visible_devices: str | None = None
    progress_enabled: bool = False
    stream_subprocess_output: bool = False
    progress_interval_seconds: float = 30.0
    generation_progress_interval: int = 1

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any] | None,
        *,
        checkpoint_path: str | Path | None = None,
        generated_path: str | Path | None = None,
        python_executable: str | None = None,
        device: str | None = None,
    ) -> DeFoGGeneratorConfig:
        values = dict(data or {})
        generator_type = str(values.get("type", "defog")).lower()
        if generator_type != "defog":
            raise ValueError("base_generator.type must be 'defog'.")
        backend = str(values.get("backend", "subprocess")).lower()
        if backend != "subprocess":
            raise ValueError(
                "The reference DeFoG adapter supports only subprocess isolation."
            )

        dataset = str(values.get("dataset", "")).lower().strip()
        experiment = str(values.get("experiment", dataset)).lower().strip()
        for name, value in (("dataset", dataset), ("experiment", experiment)):
            if not value or _SAFE_IDENTIFIER.fullmatch(value) is None:
                raise ValueError(f"base_generator.{name} is not a safe identifier.")
        if dataset not in SUPPORTED_DATASETS:
            raise ValueError(
                f"Unsupported DeFoG dataset {dataset!r}; expected one of "
                f"{sorted(SUPPORTED_DATASETS)}."
            )

        sampling = dict(values.get("sampling", {}) or {})
        runtime = dict(values.get("runtime", {}) or {})
        resolved_checkpoint = checkpoint_path or values.get("checkpoint_path")
        resolved_generated = generated_path or values.get("generated_path")
        if not resolved_checkpoint and not resolved_generated:
            raise ValueError(
                "base_generator.checkpoint_path is required unless generated_path "
                "reuses an existing DeFoG export."
            )

        selected_device = str(
            device if device is not None else runtime.get("device", "auto")
        ).lower()
        if _SAFE_DEVICE.fullmatch(selected_device) is None:
            raise ValueError(
                "base_generator.runtime.device must be auto, cpu, cuda, or cuda:N."
            )
        batch_size = int(sampling.get("batch_size", 64))
        sample_steps_value = sampling.get("sample_steps")
        sample_steps = (
            None if sample_steps_value is None else int(sample_steps_value)
        )
        if batch_size <= 0 or (sample_steps is not None and sample_steps <= 0):
            raise ValueError("DeFoG batch_size and sample_steps must be positive.")
        time_distortion_value = sampling.get("time_distortion")
        time_distortion = (
            None
            if time_distortion_value is None
            else str(time_distortion_value).lower()
        )
        if time_distortion is not None and time_distortion not in {
            "identity",
            "cosine",
            "polyinc",
            "polydec",
        }:
            raise ValueError("Unsupported DeFoG time_distortion.")
        rdb_value = sampling.get("rdb")
        rdb = None if rdb_value is None else str(rdb_value).lower()
        if rdb is not None and rdb not in {"general", "column", "entry"}:
            raise ValueError("DeFoG sampling.rdb must be general, column, or entry.")
        rdb_crit_value = sampling.get("rdb_crit")
        rdb_crit = None if rdb_crit_value is None else str(rdb_crit_value).strip()
        if rdb_crit is not None and _SAFE_IDENTIFIER.fullmatch(rdb_crit) is None:
            raise ValueError("base_generator.sampling.rdb_crit is invalid.")
        eta_value = sampling.get("eta")
        omega_value = sampling.get("omega")
        eta = None if eta_value is None else float(eta_value)
        omega = None if omega_value is None else float(omega_value)
        if (eta is not None and not np.isfinite(eta)) or (
            omega is not None and not np.isfinite(omega)
        ):
            raise ValueError("DeFoG eta and omega must be finite.")

        timeout_value = runtime.get("timeout_seconds")
        timeout_seconds = None if timeout_value is None else float(timeout_value)
        if timeout_seconds is not None and timeout_seconds <= 0.0:
            raise ValueError("base_generator.runtime.timeout_seconds must be positive.")
        raw_progress = runtime.get("progress", {}) or {}
        if not isinstance(raw_progress, Mapping):
            raise TypeError("base_generator.runtime.progress must be a mapping.")
        progress_enabled = raw_progress.get("enabled", False)
        if not isinstance(progress_enabled, bool):
            raise TypeError(
                "base_generator.runtime.progress.enabled must be a boolean."
            )
        stream_output = raw_progress.get("stream_output", progress_enabled)
        if not isinstance(stream_output, bool):
            raise TypeError(
                "base_generator.runtime.progress.stream_output must be a boolean."
            )
        progress_interval_seconds = float(
            raw_progress.get("interval_seconds", 30.0)
        )
        if progress_interval_seconds <= 0:
            raise ValueError(
                "base_generator.runtime.progress.interval_seconds must be positive."
            )
        generation_progress_interval = int(
            raw_progress.get("generation_batch_interval", 1)
        )
        if generation_progress_interval <= 0:
            raise ValueError(
                "base_generator.runtime.progress.generation_batch_interval "
                "must be positive."
            )
        cuda_visible = runtime.get("cuda_visible_devices")
        if cuda_visible is not None:
            cuda_visible = str(cuda_visible)
            if not cuda_visible or any(ch in cuda_visible for ch in "\n\r\0"):
                raise ValueError("runtime.cuda_visible_devices is invalid.")
            if selected_device.startswith("cuda:") and selected_device != "cuda:0":
                raise ValueError(
                    "When cuda_visible_devices is set, device must be 'cuda' or "
                    "'cuda:0' because CUDA renumbers visible devices locally."
                )

        source_env = str(values.get("source_env", DEFOG_ROOT_ENV))
        python_env = str(values.get("python_env", DEFOG_PYTHON_ENV))
        if any(
            _SAFE_ENV_KEY.fullmatch(value) is None for value in (source_env, python_env)
        ):
            raise ValueError(
                "DeFoG source_env and python_env must be environment keys."
            )
        manifest_value = values.get("manifest_path")
        datadir_value = values.get("dataset_datadir")
        resolved_config_value = values.get("resolved_config_path")
        molecular_statistics_value = values.get("molecular_statistics_path")
        return cls(
            dataset=dataset,
            experiment=experiment,
            checkpoint_path=(
                Path(resolved_checkpoint).expanduser() if resolved_checkpoint else None
            ),
            generated_path=(
                Path(resolved_generated).expanduser() if resolved_generated else None
            ),
            manifest_path=(
                Path(manifest_value).expanduser() if manifest_value else None
            ),
            dataset_datadir=(
                Path(datadir_value).expanduser() if datadir_value else None
            ),
            resolved_config_path=(
                Path(resolved_config_value).expanduser()
                if resolved_config_value
                else None
            ),
            molecular_statistics_path=(
                Path(molecular_statistics_value).expanduser()
                if molecular_statistics_value
                else None
            ),
            source_env=source_env,
            python_env=python_env,
            python_executable=(
                python_executable
                if python_executable is not None
                else runtime.get("python_executable")
            ),
            device=selected_device,
            batch_size=batch_size,
            sample_steps=sample_steps,
            eta=eta,
            omega=omega,
            time_distortion=time_distortion,
            rdb=rdb,
            rdb_crit=rdb_crit,
            timeout_seconds=timeout_seconds,
            cuda_visible_devices=cuda_visible,
            progress_enabled=progress_enabled,
            stream_subprocess_output=stream_output,
            progress_interval_seconds=progress_interval_seconds,
            generation_progress_interval=generation_progress_interval,
        )


@dataclass(frozen=True)
class DeFoGGenerationResult:
    """Validated GraphER-facing result from DeFoG."""

    graphs: list[nx.Graph]
    export_path: Path
    manifest_path: Path | None
    log_path: Path | None
    manifest: dict[str, Any]


def resolve_defog_root(source_env: str = DEFOG_ROOT_ENV) -> Path:
    """Resolve and validate the DeFoG repository root from an environment key."""

    if _SAFE_ENV_KEY.fullmatch(source_env) is None:
        raise ValueError("base_generator.source_env is invalid.")
    raw = os.environ.get(source_env)
    if not raw:
        raise OSError(
            f"Environment variable {source_env} must point to the DeFoG source root."
        )
    root = Path(raw).expanduser().resolve()
    required = (root / "src" / "main.py", root / "configs" / "config.yaml")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{source_env}={root} is not a DeFoG source root; missing {missing}."
        )
    return root


def resolve_defog_python(
    config: DeFoGGeneratorConfig,
    *,
    defog_root: Path,
) -> str:
    """Choose the isolated DeFoG interpreter without invoking a shell."""

    if _SAFE_ENV_KEY.fullmatch(config.python_env) is None:
        raise ValueError("base_generator.python_env is invalid.")
    candidate = config.python_executable or os.environ.get(config.python_env)
    if not candidate:
        local_python = defog_root / ".venv" / "bin" / "python"
        candidate = str(local_python) if local_python.is_file() else sys.executable
    resolved = shutil.which(str(candidate))
    if resolved is None:
        path = Path(str(candidate)).expanduser()
        if path.is_file():
            resolved = str(path.resolve())
    if resolved is None:
        raise FileNotFoundError(
            f"Could not resolve a DeFoG Python interpreter from {candidate!r}."
        )
    return resolved


def _worker_path() -> Path:
    path = Path(__file__).resolve().parents[3] / "scripts" / "defog_export_worker.py"
    if not path.is_file():
        raise FileNotFoundError(f"Missing DeFoG export worker: {path}")
    return path


def build_defog_worker_command(
    config: DeFoGGeneratorConfig,
    *,
    defog_root: Path,
    python_executable: str,
    export_path: Path,
    manifest_path: Path,
    num_graphs: int,
    seed: int,
    raw_pickle_path: Path | None = None,
) -> list[str]:
    """Build the direct argv used to run the isolated export worker."""

    if num_graphs <= 0:
        raise ValueError("num_graphs must be positive.")
    command = [
        python_executable,
        str(_worker_path()),
        "--defog-root",
        str(defog_root),
        "--output",
        str(export_path),
        "--manifest",
        str(manifest_path),
        "--dataset",
        config.dataset,
        "--experiment",
        config.experiment,
        "--num-samples",
        str(num_graphs),
        "--seed",
        str(seed),
    ]
    if raw_pickle_path is not None:
        command.extend(["--input-pickle", str(raw_pickle_path)])
        return command
    if config.checkpoint_path is None:
        raise ValueError("A DeFoG checkpoint is required for generation.")
    if config.resolved_config_path is not None:
        resolved_config = config.resolved_config_path.expanduser().resolve()
        if not resolved_config.is_file():
            raise FileNotFoundError(
                f"Missing DeFoG resolved training config: {resolved_config}."
            )
        command.extend(["--resolved-config", str(resolved_config)])
    if config.dataset_datadir is not None:
        datadir = config.dataset_datadir.expanduser().resolve()
        if not datadir.is_dir():
            raise FileNotFoundError(
                f"Missing DeFoG-formatted dataset directory: {datadir}."
            )
        command.extend(["--dataset-datadir", str(datadir)])
    if config.molecular_statistics_path is not None:
        statistics = config.molecular_statistics_path.expanduser().resolve()
        if not statistics.is_file():
            raise FileNotFoundError(
                f"Missing DeFoG molecular-statistics cache: {statistics}."
            )
        command.extend(["--molecular-statistics", str(statistics)])
    command.extend(
        [
            "--checkpoint",
            str(config.checkpoint_path.resolve()),
            "--device",
            config.device,
            "--batch-size",
            str(config.batch_size),
        ]
    )
    optional_sampling_arguments = (
        ("--sample-steps", config.sample_steps),
        ("--eta", config.eta),
        ("--omega", config.omega),
        ("--time-distortion", config.time_distortion),
        ("--rdb", config.rdb),
        ("--rdb-crit", config.rdb_crit),
    )
    for flag, value in optional_sampling_arguments:
        if value is not None:
            command.extend([flag, str(value)])
    return command


def _worker_environment(
    config: DeFoGGeneratorConfig,
    *,
    defog_root: Path,
    seed: int,
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
            "GRAPHER_DEFOG_PROGRESS_ENABLED": (
                "1" if config.progress_enabled else "0"
            ),
            "GRAPHER_DEFOG_PROGRESS_INTERVAL_SECONDS": str(
                config.progress_interval_seconds
            ),
            "GRAPHER_DEFOG_GENERATION_PROGRESS_INTERVAL": str(
                config.generation_progress_interval
            ),
        }
    )
    if config.cuda_visible_devices is not None:
        environment["CUDA_VISIBLE_DEVICES"] = config.cuda_visible_devices
    return environment


def _log_tail(path: Path, lines: int = 60) -> str:
    if not path.is_file():
        return ""
    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(content[-lines:])


def _run_worker(
    command: list[str],
    *,
    config: DeFoGGeneratorConfig,
    defog_root: Path,
    log_path: Path,
    seed: int,
) -> None:
    ensure_dir(log_path.parent)
    reporter = SubprocessLogReporter(
        label="DeFoG generation worker",
        log_path=log_path,
        enabled=config.progress_enabled,
        stream_output=config.stream_subprocess_output,
        interval_seconds=config.progress_interval_seconds,
    )
    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            log_file.flush()
            reporter.start(start_offset=0)
            completed = subprocess.run(
                command,
                cwd=str(defog_root / "src"),
                env=_worker_environment(config, defog_root=defog_root, seed=seed),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                timeout=config.timeout_seconds,
                check=False,
                shell=False,
                start_new_session=True,
            )
            log_file.flush()
    except subprocess.TimeoutExpired as exc:
        reporter.stop(status="timed out")
        raise RuntimeError(
            f"DeFoG generation timed out. Last worker output:\n{_log_tail(log_path)}"
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
            f"DeFoG worker exited with code {completed.returncode}. "
            f"Last worker output:\n{_log_tail(log_path)}"
        )


def _require_integer_array(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite numeric values.")
    rounded = np.rint(array)
    if not np.array_equal(array, rounded):
        raise ValueError(f"{name} must contain integral values.")
    return rounded.astype(np.int64, copy=False)


def load_defog_export(
    path: str | Path,
    *,
    dataset: str | None = None,
    expected_count: int | None = None,
) -> list[nx.Graph]:
    """Load and strictly validate a neutral DeFoG graph export.

    Molecular batches use semantic atomic-number/bond-type arrays and are
    decoded by the dedicated molecular codec. Generic batches retain the
    compact topology representation used by the original adapter.
    """

    export_path = Path(path)
    if not export_path.is_file():
        raise FileNotFoundError(f"Missing DeFoG export: {export_path}")
    normalized_dataset = str(dataset).lower() if dataset is not None else None
    if normalized_dataset is None:
        with np.load(export_path, allow_pickle=False) as probe:
            if "format" in probe.files:
                encoded_format = np.asarray(probe["format"])
                if (
                    encoded_format.ndim == 0
                    and str(encoded_format.item()) == "grapher_defog_molecular_v1"
                ):
                    if "dataset" not in probe.files:
                        raise ValueError("Molecular DeFoG export has no dataset metadata.")
                    encoded_dataset = np.asarray(probe["dataset"])
                    if encoded_dataset.ndim != 0:
                        raise ValueError(
                            "Molecular DeFoG dataset metadata must be scalar."
                        )
                    normalized_dataset = str(encoded_dataset.item()).lower()
    if normalized_dataset in SUPPORTED_MOLECULAR_DATASETS:
        from grapher.models.defog_molecular_codec import (
            MODEL_REPRESENTATION,
            load_molecular_export,
        )

        return load_molecular_export(
            export_path,
            expected_dataset=normalized_dataset,
            expected_representation=MODEL_REPRESENTATION,
            expected_count=expected_count,
        )
    if normalized_dataset is not None and normalized_dataset not in SUPPORTED_GENERIC_DATASETS:
        raise ValueError(f"Unsupported DeFoG dataset {dataset!r}.")
    with np.load(export_path, allow_pickle=False) as payload:
        required = {
            "node_ptr",
            "node_labels",
            "edge_ptr",
            "edge_endpoints",
            "edge_labels",
            "raw_indices",
        }
        missing = sorted(required - set(payload.files))
        if missing:
            raise ValueError(f"DeFoG export is missing arrays: {missing}.")
        node_ptr = _require_integer_array(payload["node_ptr"], name="node_ptr")
        node_labels = _require_integer_array(payload["node_labels"], name="node_labels")
        edge_ptr = _require_integer_array(payload["edge_ptr"], name="edge_ptr")
        endpoints = _require_integer_array(
            payload["edge_endpoints"], name="edge_endpoints"
        )
        edge_labels = _require_integer_array(payload["edge_labels"], name="edge_labels")
        raw_indices = _require_integer_array(payload["raw_indices"], name="raw_indices")
        if "dataset" in payload.files:
            encoded_dataset_values = np.asarray(payload["dataset"]).reshape(-1)
            if encoded_dataset_values.size != 1:
                raise ValueError("DeFoG dataset metadata must contain one value.")
            encoded_dataset = str(encoded_dataset_values[0]).lower()
            if normalized_dataset is not None and encoded_dataset != normalized_dataset:
                raise ValueError(
                    f"DeFoG export dataset {encoded_dataset!r} does not match "
                    f"expected dataset {normalized_dataset!r}."
                )

    if node_ptr.ndim != 1 or edge_ptr.ndim != 1 or node_ptr.size != edge_ptr.size:
        raise ValueError("DeFoG node_ptr and edge_ptr must be aligned vectors.")
    num_graphs = int(node_ptr.size - 1)
    if num_graphs <= 0:
        raise ValueError("DeFoG export contains no graphs.")
    if expected_count is not None and num_graphs != int(expected_count):
        raise ValueError(
            f"DeFoG exported {num_graphs} graphs, expected {expected_count}."
        )
    if raw_indices.shape != (num_graphs,) or not np.array_equal(
        raw_indices, np.arange(num_graphs, dtype=np.int64)
    ):
        raise ValueError("DeFoG raw_indices must preserve contiguous sample order.")
    if node_ptr[0] != 0 or edge_ptr[0] != 0:
        raise ValueError("DeFoG pointer arrays must start at zero.")
    if np.any(np.diff(node_ptr) <= 0) or np.any(np.diff(edge_ptr) < 0):
        raise ValueError("DeFoG pointer arrays are not monotone.")
    if node_ptr[-1] != node_labels.size:
        raise ValueError("node_ptr does not cover node_labels.")
    if endpoints.ndim != 2 or endpoints.shape[1] != 2:
        raise ValueError("edge_endpoints must have shape [M, 2].")
    if edge_ptr[-1] != endpoints.shape[0] or edge_labels.shape != (endpoints.shape[0],):
        raise ValueError("edge_ptr, edge_endpoints, and edge_labels disagree.")

    graphs: list[nx.Graph] = []
    for index in range(num_graphs):
        node_start, node_stop = int(node_ptr[index]), int(node_ptr[index + 1])
        edge_start, edge_stop = int(edge_ptr[index]), int(edge_ptr[index + 1])
        labels = node_labels[node_start:node_stop]
        graph_edges = endpoints[edge_start:edge_stop]
        graph_edge_labels = edge_labels[edge_start:edge_stop]
        n = int(labels.size)
        if n <= 0 or np.any(labels != 0):
            raise ValueError(
                f"Generic DeFoG sample {index} must have one zero-valued node class."
            )
        if np.any(graph_edge_labels != 1):
            raise ValueError(
                f"Generic DeFoG sample {index} contains a non-edge-present label."
            )
        if graph_edges.size:
            if np.any(graph_edges < 0) or np.any(graph_edges >= n):
                raise ValueError(f"DeFoG sample {index} has an invalid endpoint.")
            if np.any(graph_edges[:, 0] >= graph_edges[:, 1]):
                raise ValueError(
                    f"DeFoG sample {index} edges must use canonical u < v order."
                )
            if np.unique(graph_edges, axis=0).shape[0] != graph_edges.shape[0]:
                raise ValueError(f"DeFoG sample {index} contains duplicate edges.")
        graph = nx.Graph()
        graph.add_nodes_from(range(n))
        graph.add_edges_from((int(u), int(v)) for u, v in graph_edges)
        graph.graph["base_generator"] = "defog"
        graph.graph["defog_raw_index"] = index
        graphs.append(_normalize_generic_graph(graph))
    return graphs


def _read_manifest(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"DeFoG manifest {path} must contain a JSON object.")
    if value.get("format") not in {
        DEFOG_EXPORT_FORMAT,
        LEGACY_DEFOG_EXPORT_FORMAT,
    }:
        raise ValueError(f"Unsupported DeFoG manifest format {value.get('format')!r}.")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_export_manifest(
    manifest: Mapping[str, Any],
    *,
    export_path: Path,
    expected_count: int,
    expected_dataset: str | None = None,
) -> None:
    if not manifest:
        return
    if int(manifest.get("exported_samples", -1)) != expected_count:
        raise ValueError("DeFoG manifest sample count does not match the export.")
    manifest_dataset = manifest.get("dataset")
    if (
        expected_dataset is not None
        and manifest_dataset is not None
        and str(manifest_dataset).lower() != str(expected_dataset).lower()
    ):
        raise ValueError("DeFoG manifest dataset does not match the requested dataset.")
    export = manifest.get("export", {}) or {}
    if not isinstance(export, Mapping):
        raise TypeError("DeFoG manifest export metadata must be a mapping.")
    expected_hash = export.get("sha256")
    if expected_hash is not None and str(expected_hash) != _sha256(export_path):
        raise ValueError("DeFoG export checksum does not match its manifest.")


def generate_defog_graphs(
    config: DeFoGGeneratorConfig,
    *,
    num_graphs: int,
    seed: int,
    output_dir: str | Path,
) -> DeFoGGenerationResult:
    """Generate or reuse DeFoG samples and expose validated NetworkX graphs."""

    if num_graphs <= 0:
        raise ValueError("num_graphs must be positive.")
    destination = ensure_dir(output_dir).resolve()
    configured_generated = (
        config.generated_path.resolve() if config.generated_path is not None else None
    )
    if (
        configured_generated is not None
        and configured_generated.suffix.lower() == ".npz"
    ):
        manifest_path = (
            config.manifest_path.resolve()
            if config.manifest_path is not None
            else configured_generated.with_name("defog_manifest.json")
        )
        graphs = load_defog_export(
            configured_generated,
            dataset=config.dataset,
            expected_count=num_graphs,
        )
        manifest = _read_manifest(manifest_path)
        _verify_export_manifest(
            manifest,
            export_path=configured_generated,
            expected_count=num_graphs,
            expected_dataset=config.dataset,
        )
        return DeFoGGenerationResult(
            graphs=graphs,
            export_path=configured_generated,
            manifest_path=manifest_path if manifest_path.is_file() else None,
            log_path=None,
            manifest=manifest,
        )

    if configured_generated is not None and configured_generated.suffix.lower() not in {
        ".pkl",
        ".pickle",
    }:
        raise ValueError(
            "base_generator.generated_path must be a neutral .npz export or a "
            "trusted DeFoG .pkl file."
        )

    defog_root = resolve_defog_root(config.source_env)
    python_executable = resolve_defog_python(config, defog_root=defog_root)
    if configured_generated is None and (
        config.checkpoint_path is None or not config.checkpoint_path.is_file()
    ):
        raise FileNotFoundError(f"Missing DeFoG checkpoint: {config.checkpoint_path}")
    if configured_generated is not None and not configured_generated.is_file():
        raise FileNotFoundError(
            f"Missing pre-generated DeFoG sample file: {configured_generated}"
        )

    export_path = destination / "defog_samples.npz"
    manifest_path = destination / "defog_manifest.json"
    log_path = destination / "defog.log"
    command = build_defog_worker_command(
        config,
        defog_root=defog_root,
        python_executable=python_executable,
        export_path=export_path,
        manifest_path=manifest_path,
        num_graphs=num_graphs,
        seed=seed,
        raw_pickle_path=(
            configured_generated
            if configured_generated is not None
            and configured_generated.suffix.lower() in {".pkl", ".pickle"}
            else None
        ),
    )
    _run_worker(
        command,
        config=config,
        defog_root=defog_root,
        log_path=log_path,
        seed=seed,
    )
    graphs = load_defog_export(
        export_path,
        dataset=config.dataset,
        expected_count=num_graphs,
    )
    manifest = _read_manifest(manifest_path)
    if not manifest:
        raise RuntimeError("DeFoG worker completed without publishing a manifest.")
    _verify_export_manifest(
        manifest,
        export_path=export_path,
        expected_count=num_graphs,
        expected_dataset=config.dataset,
    )
    return DeFoGGenerationResult(
        graphs=graphs,
        export_path=export_path,
        manifest_path=manifest_path,
        log_path=log_path,
        manifest=manifest,
    )

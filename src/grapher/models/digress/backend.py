"""Validated subprocess backend for DiGress generation."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import networkx as nx

from grapher.models.digress.codec import load_digress_export
from grapher.utils.subprocess_progress import SubprocessLogReporter


@dataclass(frozen=True)
class DiGressGenerationResult:
    graphs: list[nx.Graph]
    export_path: Path
    manifest_path: Path
    log_path: Path
    export_sha256: str
    manifest: Mapping[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _tail(path: Path, *, lines: int = 200) -> str:
    if not path.is_file():
        return ""
    return "\n".join(
        path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
    )


def _worker_path() -> Path:
    path = Path(__file__).resolve().parent / "workers" / "export.py"
    if not path.is_file():
        raise FileNotFoundError(f"Missing DiGress export worker: {path}")
    return path


def _environment(
    digress_root: Path,
    *,
    seed: int,
    cuda_visible_devices: str | None,
) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join(
                [str(digress_root), str(digress_root / "src")]
            ),
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
        environment["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    return environment


def generate_digress_graphs(
    *,
    digress_root: str | Path,
    python_executable: str | Path,
    dataset: str,
    dataset_datadir: str | Path,
    resolved_config_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    num_graphs: int,
    generation_seed: int,
    batch_size: int,
    molecular_statistics_path: str | Path | None = None,
    device: str = "auto",
    cuda_visible_devices: str | None = None,
    timeout_seconds: float | None = None,
    progress_enabled: bool = False,
    stream_output: bool = False,
    progress_interval_seconds: float = 30.0,
    generation_progress_every_batches: int = 1,
) -> DiGressGenerationResult:
    """Run the isolated exporter and decode its neutral NPZ output."""

    root = Path(digress_root).expanduser().resolve()
    python = Path(python_executable).expanduser().resolve()
    dataset_dir = Path(dataset_datadir).expanduser().resolve()
    config = Path(resolved_config_path).expanduser().resolve()
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if not python.is_file():
        raise FileNotFoundError(f"Missing DiGress Python executable: {python}")
    for required in (dataset_dir, config, checkpoint):
        if not required.exists():
            raise FileNotFoundError(required)
    if int(num_graphs) <= 0 or int(batch_size) <= 0:
        raise ValueError("num_graphs and batch_size must be positive.")
    if timeout_seconds is not None and float(timeout_seconds) <= 0:
        raise ValueError("timeout_seconds must be positive.")

    destination.mkdir(parents=True, exist_ok=True)
    export_path = destination / "digress_samples.npz"
    manifest_path = destination / "digress_manifest.json"
    log_path = destination / "digress.log"
    command = [
        str(python),
        str(_worker_path()),
        "--digress-root",
        str(root),
        "--dataset",
        str(dataset),
        "--dataset-datadir",
        str(dataset_dir),
        "--config",
        str(config),
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(export_path),
        "--manifest",
        str(manifest_path),
        "--num-graphs",
        str(int(num_graphs)),
        "--batch-size",
        str(int(batch_size)),
        "--seed",
        str(int(generation_seed)),
        "--progress-every-batches",
        str(int(generation_progress_every_batches)),
        "--device",
        str(device),
    ]
    if molecular_statistics_path is not None:
        statistics = Path(molecular_statistics_path).expanduser().resolve()
        if not statistics.is_file():
            raise FileNotFoundError(statistics)
        command.extend(["--molecular-statistics", str(statistics)])

    environment = _environment(
        root,
        seed=generation_seed,
        cuda_visible_devices=cuda_visible_devices,
    )
    reporter = SubprocessLogReporter(
        label="DiGress generation worker",
        log_path=log_path,
        enabled=progress_enabled,
        stream_output=stream_output,
        interval_seconds=progress_interval_seconds,
        prefix="GraphER/DiGress",
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        reporter.start(start_offset=0)
        try:
            completed = subprocess.run(
                command,
                cwd=str(root / "src"),
                env=environment,
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
                "DiGress generation timed out.\n"
                f"Log: {log_path}\nCommand: {json.dumps(command)}\n"
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
            f"DiGress generation exited with code {completed.returncode}.\n"
            f"Log: {log_path}\nCommand: {json.dumps(command)}\n"
            f"Last output:\n{_tail(log_path)}"
        )
    for artifact in (export_path, manifest_path):
        if not artifact.is_file():
            raise RuntimeError(f"DiGress exporter did not publish {artifact}.")
    manifest = _read_json(manifest_path)
    if manifest.get("format") != "grapher_digress_export_v1":
        raise RuntimeError("Unsupported DiGress export manifest format.")
    if int(manifest.get("num_generated", -1)) != int(num_graphs):
        raise RuntimeError(
            "DiGress export count mismatch: "
            f"manifest={manifest.get('num_generated')}, requested={num_graphs}."
        )
    expected_hash = str((manifest.get("output") or {}).get("sha256", ""))
    observed_hash = _sha256(export_path)
    if not expected_hash or expected_hash != observed_hash:
        raise RuntimeError("DiGress neutral export SHA-256 mismatch.")
    graphs = load_digress_export(export_path, dataset=dataset)
    if len(graphs) != int(num_graphs):
        raise RuntimeError(
            f"Decoded {len(graphs)} DiGress graphs; expected {num_graphs}."
        )
    return DiGressGenerationResult(
        graphs=graphs,
        export_path=export_path,
        manifest_path=manifest_path,
        log_path=log_path,
        export_sha256=observed_hash,
        manifest=manifest,
    )

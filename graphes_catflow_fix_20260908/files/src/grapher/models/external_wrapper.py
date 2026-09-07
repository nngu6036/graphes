"""Managed subprocess contract shared by four source-backed research baselines.

The upstream checkout is read-only. Neural code runs in its own interpreter;
GraphER exports numeric datasets and validates every published raw graph batch.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
import uuid
from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from grapher.models.base import (BaseGeneratorWrapper, TrainRequest, TrainingArtifacts,
                                 GenerateRequest, GenerationArtifacts)
from grapher.models.errors import ArtifactCollisionError
from grapher.models.external_codec import PROFILES, GraphProfile, save_graphs, decode_graphs
from grapher.utils.networkx_pickle import load_trusted_networkx_pickle
from grapher.utils.subprocess_progress import SubprocessLogReporter


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n")
    tmp.replace(path)


def merge(base: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for k, v in update.items():
        result[k] = merge(result[k], v) if isinstance(v, Mapping) and isinstance(result.get(k), Mapping) else copy.deepcopy(v)
    return result


def source_identity(root: Path) -> dict[str, Any]:
    files = {p.relative_to(root).as_posix(): sha256(p) for p in sorted(root.rglob("*"))
             if p.is_file() and p.suffix in {".py", ".yaml", ".yml"}
             and not any(part in {".git", "__pycache__", "outputs", "wandb", "logs", ".venv"} for part in p.relative_to(root).parts)}
    signature = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {"root": str(root), "fingerprint": signature, "files": files}


class SourceBackedWrapper(BaseGeneratorWrapper):
    """Subclasses declare the source markers, supported datasets and worker."""
    supported_datasets: frozenset[str]
    source_markers: tuple[str, ...]
    default_options: dict[str, Any] = {}
    source_aliases: tuple[str, ...] = ()

    def _profile(self, dataset: str, options: Mapping[str, Any]) -> GraphProfile:
        if dataset not in self.supported_datasets:
            raise ValueError(f"{self.display_name} supports {sorted(self.supported_datasets)}, not {dataset!r}. "
                             "No categorical molecular support is inferred from an adjacency-only release.")
        p = PROFILES[dataset]
        maximum = int(options.get("max_nodes", p.max_nodes))
        if maximum < 2:
            raise ValueError("max_nodes must be at least 2.")
        return replace(p, max_nodes=maximum)

    def _options(self, request: TrainRequest) -> dict[str, Any]:
        options = copy.deepcopy(self.default_options)
        if request.config_path:
            raw = yaml.safe_load(request.config_path.read_text()) or {}
            if not isinstance(raw, Mapping):
                raise TypeError("Wrapper configuration must be a YAML mapping.")
            selected = raw.get(self.model_id, raw.get("gsdm", raw) if self.model_id == "gdsm" else raw)
            if not isinstance(selected, Mapping):
                raise TypeError("Model configuration section must be a mapping.")
            options = merge(options, selected)
        options = merge(options, request.options)
        allowed = {"source_env", "python_env", "source_root", "python", "runtime", "train", "model", "sample",
                   "generation_batch_size", "max_nodes", "upstream_config", "config_overrides", "training_estimates"}
        if set(options) - allowed:
            raise ValueError(f"Unknown wrapper options: {sorted(set(options) - allowed)}.")
        train = options.get("train", {})
        for key in ("epochs", "batch_size", "log_every"):
            if int(train.get(key, 1 if key == "log_every" else 0)) <= 0:
                raise ValueError(f"train.{key} must be positive.")
        if int(options.get("generation_batch_size", 32)) <= 0:
            raise ValueError("generation_batch_size must be positive.")
        estimates = options.get("training_estimates", {})
        if estimates.get("enabled", False):
            raise ValueError("These wrappers do not create corrector-training pairs. Generate a separate raw training batch explicitly.")
        return options

    def _resolve_runtime(self, options: Mapping[str, Any], fallback: Mapping[str, Any] | None = None) -> tuple[Path, Path]:
        env_name = str(options.get("source_env", self.model_id.upper()))
        root_value = options.get("source_root") or os.environ.get(env_name)
        for alias in self.source_aliases:
            root_value = root_value or os.environ.get(alias)
        if not root_value and fallback:
            root_value = fallback.get("source_root")
        if not root_value:
            raise RuntimeError(f"Set {env_name} to the extracted {self.display_name} source directory.")
        root = Path(root_value).expanduser().resolve()
        missing = [f for f in self.source_markers if not (root / f).is_file()]
        if missing:
            raise FileNotFoundError(f"Invalid {self.display_name} checkout {root}; missing {missing}.")
        python_env = str(options.get("python_env", self.model_id.upper() + "_PYTHON"))
        py_value = options.get("python") or os.environ.get(python_env)
        for alias in self.source_aliases:
            py_value = py_value or os.environ.get(alias + "_PYTHON")
        py_value = py_value or (fallback or {}).get("python") or sys.executable
        located = shutil.which(str(py_value))
        executable = Path(located or str(py_value)).expanduser().absolute()
        # Do not resolve a venv's python symlink: doing so discards the venv.
        if not executable.is_file():
            raise FileNotFoundError(f"Missing upstream Python: {executable}")
        return root, executable

    def _run_worker(self, job: dict[str, Any], python: Path, work: Path, log_path: Path) -> None:
        # Public run layouts may be relative to the GraphER checkout. Workers
        # change cwd, so every file passed across the process boundary must be
        # absolute (including the job path itself).
        work = work.resolve()
        log_path = log_path.resolve()
        for key in ("source_root", "dataset_dir", "checkpoint", "worker_manifest", "output"):
            if key in job:
                job[key] = str(Path(job[key]).resolve())
        write_json(work / "job.json", job)
        env = dict(os.environ)
        env.update(PYTHONHASHSEED=str(job["seed"]), PYTHONUNBUFFERED="1", MPLBACKEND="Agg",
                   PYTHONDONTWRITEBYTECODE="1", WANDB_MODE="disabled", CUBLAS_WORKSPACE_CONFIG=":4096:8")
        # Absolute source root and worker-local helper imports only. Inheriting
        # PYTHONPATH=src would accidentally import GraphER into an old runtime.
        env["PYTHONPATH"] = job["source_root"]
        runtime = job["options"].get("runtime", {})
        device = str(runtime.get("device", "auto"))
        if device not in {"cpu", "auto", "cuda", "gpu"} and not (device.startswith("cuda:") and device[5:].isdigit()):
            raise ValueError("runtime.device must be auto, cpu, gpu, cuda or cuda:N.")
        if device == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""
        elif device.startswith("cuda:"):
            env["CUDA_VISIBLE_DEVICES"] = device.split(":", 1)[1]
        elif runtime.get("cuda_visible_devices") is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(runtime["cuda_visible_devices"])
        worker = Path(__file__).parent / "external_workers" / (self.model_id + ".py")
        progress = runtime.get("progress", {})
        reporter = SubprocessLogReporter(label=job["stage"], log_path=log_path,
                    enabled=bool(progress.get("enabled", True)), stream_output=bool(progress.get("stream_output", True)),
                    interval_seconds=float(progress.get("interval_seconds", 15)), prefix="GraphER/" + self.display_name)
        timeout = runtime.get("timeout_seconds")
        if timeout is not None and float(timeout) <= 0:
            raise ValueError("timeout_seconds must be positive or null.")
        with log_path.open("w") as log:
            reporter.start(start_offset=0)
            try:
                result = subprocess.run([str(python), str(worker), str(work / "job.json")], cwd=work,
                            env=env, stdout=log, stderr=subprocess.STDOUT, check=False, timeout=timeout)
            except BaseException:
                reporter.stop(status="failed")
                raise
            else:
                reporter.stop(status="completed" if result.returncode == 0 else "failed")
        if result.returncode:
            tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-60:])
            raise RuntimeError(f"{self.display_name} {job['stage']} failed (exit {result.returncode}).\nLog: {log_path}\n{tail}")
        record = json.loads(Path(job["worker_manifest"]).read_text())
        if record.get("format") != "grapher_external_worker_v1" or record.get("model") != self.model_id or record.get("stage") != job["stage"]:
            raise RuntimeError("Invalid worker completion manifest.")

    def _training_artifacts(self, request: TrainRequest) -> TrainingArtifacts:
        layout = request.run.layout
        return TrainingArtifacts(run_dir=layout.run_dir, checkpoint_path=layout.checkpoints_dir / (self.model_id + ".pt"),
                                 manifest_path=layout.training_manifest_path, log_path=layout.training_log_path)

    def train(self, request: TrainRequest) -> TrainingArtifacts:
        self.validate_train_request(request)
        if request.resume_from is not None:
            raise ValueError("Training resume is not implemented; cached completed runs are reusable, not resumable.")
        options = self._options(request)
        profile = self._profile(request.run.dataset_id, options)
        root, python = self._resolve_runtime(options)
        source = source_identity(root)
        layout = request.run.layout
        fingerprint = request.dataset.fingerprint()
        if layout.training_manifest_path.is_file() and not request.overwrite:
            old = json.loads(layout.training_manifest_path.read_text())
            same = (old.get("wrapper_options") == options and old["dataset"]["fingerprint"] == fingerprint
                    and old["source"]["fingerprint"] == source["fingerprint"] and old["train_seed"] == request.run.train_seed)
            checkpoint = layout.checkpoints_dir / (self.model_id + ".pt")
            if same and checkpoint.is_file() and sha256(checkpoint) == old["checkpoint"]["sha256"]:
                return self._training_artifacts(request)
            raise ArtifactCollisionError("Completed run differs from this request. Choose a new run-id or use --overwrite explicitly.")
        if layout.train_dir.exists() and not request.overwrite:
            raise ArtifactCollisionError(f"Training directory exists: {layout.train_dir}")
        layout.run_dir.mkdir(parents=True, exist_ok=True)
        work = layout.run_dir / (".train_work_" + uuid.uuid4().hex)
        work.mkdir()
        start = time.monotonic()
        try:
            data_dir = work / "native_dataset"
            data_dir.mkdir()
            conversion = {}
            for split in ("train", "val"):
                graphs = load_trusted_networkx_pickle(request.dataset.split_paths[split])
                if not isinstance(graphs, (list, tuple)) or not len(graphs):
                    raise ValueError(f"{split} must contain a nonempty graph list.")
                save_graphs(data_dir / (split + ".npz"), graphs, profile)
                conversion[split] = {"count": len(graphs), "sha256": sha256(data_dir / (split + ".npz"))}
            write_json(data_dir / "manifest.json", {"profile": profile.to_dict(), "splits": conversion,
                       "test_exported": False, "formal_charge_modelled": False})
            checkpoint = work / "checkpoints" / (self.model_id + ".pt")
            job = {"model": self.model_id, "stage": "train", "source_root": str(root),
                   "dataset_dir": str(data_dir), "profile": profile.to_dict(), "options": options,
                   "seed": request.run.train_seed, "checkpoint": str(checkpoint),
                   "worker_manifest": str(work / "worker_manifest.json")}
            self._run_worker(job, python, work, work / "train.log")
            if not checkpoint.is_file():
                raise RuntimeError("Training worker did not save a checkpoint.")
            if fingerprint != request.dataset.fingerprint():
                raise RuntimeError("Prepared dataset changed during training; refusing to publish a mixed-provenance run.")
            (work / "resolved_config.yaml").write_text(yaml.safe_dump({self.model_id: options}, sort_keys=False))
            manifest = {"format": "grapher_source_baseline_training_v1", "model_id": self.model_id,
                        "run_id": request.run.run_id, "train_seed": request.run.train_seed,
                        "created_at": datetime.now(timezone.utc).isoformat(), "duration_seconds": time.monotonic() - start,
                        "dataset": {"benchmark_id": request.dataset.benchmark_id, "serialized_id": request.dataset.serialized_id,
                                    "fingerprint": fingerprint, "root": str(request.dataset.root.resolve()),
                                    "split_sha256": {s: sha256(p) for s, p in request.dataset.split_paths.items()}},
                        "profile": profile.to_dict(), "source": source, "runtime": {"source_root": str(root), "python": str(python)},
                        "wrapper_options": options, "checkpoint": {"path": "checkpoints/" + checkpoint.name, "sha256": sha256(checkpoint)},
                        "data_conversion": conversion, "checkpoint_selection": json.loads((work / "worker_manifest.json").read_text()).get(
                            "checkpoint_selection", "final_configured_epoch"), "test_used_for_training": False}
            write_json(work / "manifest.json", manifest)
            if layout.train_dir.exists():
                shutil.rmtree(layout.train_dir)
            work.replace(layout.train_dir)
            write_json(layout.run_manifest_path, {"format": "grapher_baseline_run_v1", "model_id": self.model_id,
                       "dataset_id": request.run.dataset_id, "run_id": request.run.run_id, "train_seed": request.run.train_seed})
            return self._training_artifacts(request)
        except BaseException:
            # Keep failed worker jobs/logs: debugging an upstream dependency
            # failure should not depend on the terminal scrollback.
            print(f"Failed training artifacts retained at {work}", file=sys.stderr)
            raise

    def generate(self, request: GenerateRequest) -> GenerationArtifacts:
        self.validate_generate_request(request)
        layout = request.run.layout
        record = json.loads(layout.training_manifest_path.read_text())
        if (record.get("model_id") != self.model_id or record["dataset"]["benchmark_id"] != request.run.dataset_id
                or record.get("run_id") != request.run.run_id or record.get("train_seed") != request.run.train_seed):
            raise RuntimeError("Managed checkpoint model/dataset does not match the generation request.")
        checkpoint_hash = sha256(request.checkpoint_path)
        if record["checkpoint"]["sha256"] != checkpoint_hash:
            raise RuntimeError("Checkpoint differs from the trained run's manifest.")
        allowed = {"runtime", "generation_batch_size", "sample", "source_root", "python"}
        if set(request.options) - allowed:
            raise ValueError("Generation cannot alter training/model options.")
        options = merge(record["wrapper_options"], request.options)
        if int(options.get("generation_batch_size", 32)) <= 0:
            raise ValueError("generation_batch_size must be positive.")
        profile = self._profile(request.run.dataset_id, options)
        root, python = self._resolve_runtime(options, record["runtime"])
        if source_identity(root)["fingerprint"] != record["source"]["fingerprint"]:
            raise RuntimeError("Upstream source changed since training; use the recorded checkout.")
        data_dir = layout.train_dir / "native_dataset"
        for s, value in record["data_conversion"].items():
            if sha256(data_dir / (s + ".npz")) != value["sha256"]:
                raise RuntimeError("Exported training data changed since training.")
        generation_dir = layout.generation_dir(request.resolved_generation_id)
        if generation_dir.exists() and not request.overwrite:
            raise ArtifactCollisionError(f"Generation already exists: {generation_dir}. Use a new generation-id or --overwrite.")
        layout.generations_dir.mkdir(parents=True, exist_ok=True)
        work = layout.generations_dir / (".generation_work_" + uuid.uuid4().hex)
        work.mkdir()
        start = time.monotonic()
        job = {"model": self.model_id, "stage": "generate", "source_root": str(root), "dataset_dir": str(data_dir),
               "profile": profile.to_dict(), "options": options, "seed": request.generation_seed,
               "checkpoint": str(request.checkpoint_path.resolve()), "output": str(work / "samples.npz"),
               "num_graphs": request.num_graphs, "worker_manifest": str(work / "worker_manifest.json")}
        try:
            self._run_worker(job, python, work, work / "generate.log")
            graphs = decode_graphs(work / "samples.npz", profile, request.num_graphs)
            graph_path = work / "base_graphs.pkl"
            with graph_path.open("wb") as f:
                pickle.dump(graphs, f, protocol=pickle.HIGHEST_PROTOCOL)
            checksum = sha256(graph_path)
            manifest = {"format": "grapher_source_baseline_generation_v1", "model_id": self.model_id,
                        "run_id": request.run.run_id, "generation_seed": request.generation_seed,
                        "generation_id": request.resolved_generation_id, "dataset": record["dataset"],
                        "num_requested": request.num_graphs, "num_generated": len(graphs),
                        "base_graphs": {"path": "base_graphs.pkl", "sha256": checksum},
                        "checkpoint_sha256": checkpoint_hash, "source_fingerprint": record["source"]["fingerprint"],
                        "duration_seconds": time.monotonic() - start, "wrapper_options": options,
                        "posthoc_repair": False, "largest_component_filter": False, "test_conditioning": False}
            write_json(work / "manifest.json", manifest)
            if generation_dir.exists():
                shutil.rmtree(generation_dir)
            work.replace(generation_dir)
            return GenerationArtifacts(run_dir=layout.run_dir, generation_dir=generation_dir,
                    graphs_path=generation_dir / "base_graphs.pkl", manifest_path=generation_dir / "manifest.json",
                    num_requested=request.num_graphs, num_generated=len(graphs), graphs_sha256=checksum,
                    log_path=generation_dir / "generate.log", native_artifacts=(generation_dir / "samples.npz",))
        except BaseException:
            print(f"Failed generation artifacts retained at {work}", file=sys.stderr)
            raise

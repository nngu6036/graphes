"""Independent validation selections with epoch-matched embedded/exported priors.

Selection is independent of the optimization objective.  Only scalar validation
metrics are inspected; saving never evaluates the model or consumes RNG draws.
Each file is replaced atomically; the registry is committed after the pair and
its selection.json marker.  Registry hashes detect incomplete/interrupted writes.
These are inference/diagnostic snapshots, not optimizer-resume checkpoints.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, Mapping

import torch

from grapher.models.dhvae_hh.degree_vae import save_degree_vae_checkpoint
from grapher.rewiring_mlp.generic.spectral_model import save_topology_spectral_checkpoint


REGISTRY_NAME = "checkpoint_registry.json"
SELECTION_METRICS = {
    "best_joint": "val_joint_loss",
    "best_histogram": "val_clustering_histogram_w1",
    "best_orbit": "val_orbit_summary_log_rmse",
}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Device-independent exact tensor-state fingerprint (also supports bfloat16)."""
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        header = [name, str(tensor.dtype), list(tensor.shape)]
        digest.update(json.dumps(header, separators=(",", ":")).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    try:
        with temp.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _atomic_copy(source: Path, destination: Path) -> None:
    temp = destination.with_name(destination.name + ".tmp")
    try:
        shutil.copyfile(source, temp)
        os.replace(temp, destination)
    finally:
        temp.unlink(missing_ok=True)


def ensure_fresh_joint_output(output: Path) -> None:
    """Never reuse prior training artifacts, including an interrupted partial run."""
    names = ("checkpoint.pt", "degree_checkpoint.pt", "report.json", "history.json",
             REGISTRY_NAME, "checkpoints", "selection.json")
    occupied = [str(output / name) for name in names if (output / name).exists()]
    if occupied:
        raise FileExistsError(
            "Refusing to overwrite joint training artifacts: " + ", ".join(occupied)
            + ". Use a new output directory. This trainer does not resume from history.json."
        )


def resolve_checkpoint_policy(config: Mapping[str, Any]) -> dict[str, bool]:
    policy = dict(config.get("joint_degree", {}).get("checkpointing", {}) or {})
    unknown = set(policy) - {"enabled", "save_last"}
    if unknown:
        raise ValueError(f"Unknown joint_degree.checkpointing keys: {sorted(unknown)}")
    resolved = {"enabled": policy.get("enabled", True), "save_last": policy.get("save_last", True)}
    for name, value in resolved.items():
        if not isinstance(value, bool):
            raise ValueError(f"joint_degree.checkpointing.{name} must be a YAML boolean.")
    return resolved


class JointCheckpointManager:
    """Retain the earliest best epoch for each metric and the latest completed epoch.

    All best selections share the caller's eligibility rule (post-warmup when the
    degree prior is trainable).  `last` is saved after every completed validation
    epoch, including warmup, and explicitly records best-selection eligibility.
    Missing heads are recorded as unavailable rather than selecting a fake zero.
    Nonfinite/missing active metrics raise before any selection is written.
    """

    def __init__(self, output: Path, *, config: dict, histogram_enabled: bool,
                 orbit_enabled: bool) -> None:
        self.output = Path(output)
        self.config = config
        self.policy = resolve_checkpoint_policy(config)
        self.criteria = {"best_joint": SELECTION_METRICS["best_joint"]}
        self.unavailable: dict[str, str] = {}
        if self.policy["enabled"]:
            for kind, enabled in (("best_histogram", histogram_enabled), ("best_orbit", orbit_enabled)):
                if enabled:
                    self.criteria[kind] = SELECTION_METRICS[kind]
                else:
                    self.unavailable[kind] = "corresponding prediction head is disabled"
        self.records: dict[str, dict[str, Any]] = {}
        self.last_completed_epoch = 0
        self.training_complete = False

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "policy": self.policy,
            "checkpoint_scope": "inference_snapshot_without_optimizer_state",
            "selection_scope": "fixed_bridge_validation_not_generation_mmd",
            "tie_policy": "keep_earliest_strict_improvement",
            "best_eligibility": "degree_trainable_this_epoch_or_prior_permanently_frozen",
            "last_completed_epoch": self.last_completed_epoch,
            "training_complete": self.training_complete,
            "selections": deepcopy(self.records),
            "unavailable": self.unavailable,
            "legacy_aliases": {
                "checkpoint.pt": "best_joint",
                "degree_checkpoint.pt": "best_joint",
            },
            "dataset_graph_fingerprints": next(iter(self.records.values()), {}).get(
                "dataset_graph_fingerprints", {}
            ),
        }

    def _commit_manifest(self) -> None:
        atomic_json(self.manifest(), self.output / REGISTRY_NAME)

    def update(self, model, *, epoch: int, eligible: bool, report: dict,
               val_metrics: dict, summary_config=None) -> list[str]:
        if not isinstance(epoch, int) or epoch <= self.last_completed_epoch:
            raise ValueError("Checkpoint updates require strictly increasing positive epochs.")
        metric_values = {}
        for kind, metric in self.criteria.items():
            value = report.get(metric)
            if value is None or not math.isfinite(float(value)):
                raise ValueError(f"Missing or nonfinite checkpoint metric {metric!r} at epoch {epoch}.")
            metric_values[kind] = float(value)
        selected = [
            kind for kind in self.criteria
            if eligible and (kind not in self.records or metric_values[kind] < self.records[kind]["value"])
        ]
        if self.policy["enabled"] and self.policy["save_last"]:
            selected.append("last")
        if selected:
            model_hash = state_dict_sha256(model.state_dict())
            degree_hash = state_dict_sha256(model.degree_model.state_dict())
            for kind in selected:
                self.records[kind] = self._save_pair(
                    model, kind=kind, epoch=epoch, eligible=eligible,
                    report=report, val_metrics=val_metrics, summary_config=summary_config,
                    model_hash=model_hash, degree_hash=degree_hash,
                    metric=self.criteria.get(kind), value=metric_values.get(kind),
                )
        self.last_completed_epoch = epoch
        self._commit_manifest()
        return selected

    def _save_pair(self, model, *, kind, epoch, eligible, report, val_metrics,
                   summary_config, model_hash, degree_hash, metric, value) -> dict:
        folder = self.output / "checkpoints" / kind if self.policy["enabled"] else self.output
        folder.mkdir(parents=True, exist_ok=True)
        joint_path = folder / "checkpoint.pt"
        degree_path = folder / "degree_checkpoint.pt"
        metadata = {
            "kind": kind, "epoch": epoch, "metric": metric, "value": value,
            "direction": "min" if metric else "latest_completed_epoch",
            "eligible_for_best": bool(eligible),
            "selection_scope": "fixed_bridge_validation_not_generation_mmd",
            "seed": self.config.get("seed"),
            "model_state_sha256": model_hash, "degree_model_state_sha256": degree_hash,
        }
        metadata["pair_id"] = hashlib.sha256(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        saved_report = {**report, "checkpoint_selection": metadata}
        temp_joint = joint_path.with_name("checkpoint.pt.tmp")
        temp_degree = degree_path.with_name("degree_checkpoint.pt.tmp")
        try:
            save_topology_spectral_checkpoint(
                model, temp_joint, summary_config=summary_config,
                config=self.config, report=saved_report,
            )
            save_degree_vae_checkpoint(
                temp_degree, model.degree_model, model.degree_vectorizer,
                config=self.config,
                metrics={**val_metrics, "joint_epoch": epoch, "joint_checkpoint": str(joint_path),
                         "checkpoint_selection": metadata},
            )
            # Both serializations must finish before either destination is replaced.
            os.replace(temp_joint, joint_path)
            os.replace(temp_degree, degree_path)
        finally:
            temp_joint.unlink(missing_ok=True)
            temp_degree.unlink(missing_ok=True)
        record = {
            **metadata,
            "checkpoint": joint_path.relative_to(self.output).as_posix(),
            "degree_checkpoint": degree_path.relative_to(self.output).as_posix(),
            "checkpoint_sha256": file_sha256(joint_path),
            "degree_checkpoint_sha256": file_sha256(degree_path),
            "validation_metrics": {key: val for key, val in report.items() if key.startswith("val_")},
            "dataset_graph_fingerprints": report.get("dataset_graph_fingerprints", {}),
        }
        atomic_json(record, folder / "selection.json")
        if kind == "best_joint" and self.policy["enabled"]:
            _atomic_copy(joint_path, self.output / "checkpoint.pt")
            _atomic_copy(degree_path, self.output / "degree_checkpoint.pt")
        return record

    def finish(self) -> None:
        if "best_joint" not in self.records:
            raise RuntimeError("No eligible best-joint checkpoint was saved.")
        self.training_complete = True
        self._commit_manifest()


def verify_checkpoint_registry(output: str | Path) -> dict:
    """Check all published pairs and aliases without unpickling model files."""
    output = Path(output).resolve()
    with (output / REGISTRY_NAME).open(encoding="utf-8") as handle:
        registry = json.load(handle)
    if registry.get("schema_version") != 1 or not registry.get("selections"):
        raise ValueError("Invalid or empty joint checkpoint registry.")
    for kind, record in registry["selections"].items():
        for key in ("checkpoint", "degree_checkpoint"):
            path = (output / record[key]).resolve()
            if not path.is_relative_to(output):
                raise ValueError(f"Checkpoint registry path escapes training directory: {record[key]}")
            if not path.is_file() or file_sha256(path) != record[key + "_sha256"]:
                raise ValueError(f"Integrity check failed for {kind} {key}: {path}")
        marker = output / Path(record["checkpoint"]).parent / "selection.json"
        if not marker.is_file() or json.loads(marker.read_text(encoding="utf-8")) != record:
            raise ValueError(f"Selection marker disagrees with registry for {kind}.")
    best = registry["selections"].get("best_joint")
    if best:
        for filename, key in (("checkpoint.pt", "checkpoint"), ("degree_checkpoint.pt", "degree_checkpoint")):
            alias = output / filename
            if not alias.is_file() or file_sha256(alias) != best[key + "_sha256"]:
                raise ValueError(f"Legacy best-joint alias is stale or incomplete: {alias}")
    return registry

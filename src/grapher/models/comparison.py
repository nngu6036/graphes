"""Common baseline-comparison configuration helpers.

The common YAML is a DeFoG *reference* profile.  It supplies portable defaults
and protocol provenance, but a baseline's own YAML is merged afterwards so
model-native/equivalent training budgets can override raw DeFoG epoch counts.
Explicit CLI options remain highest priority.
"""
from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from grapher.models.registry import normalize_baseline_id

COMMON_CONFIG_SCHEMA_VERSION = 1
_REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class ComparisonProfile:
    path: Path
    dataset: str
    common: dict[str, Any]
    provenance: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must contain a mapping.")
    return copy.deepcopy(dict(value))


def default_common_config(dataset: str) -> Path | None:
    path = _REPO_ROOT / "configs" / "baselines" / f"common_{dataset}.yaml"
    return path if path.is_file() else None


def default_wrapper_config(model_id: str, dataset: str) -> Path | None:
    model = normalize_baseline_id(model_id)
    # Community-small CatFlow has a corrected-path v2 config that should be the
    # default for new training.  The legacy filename remains readable.
    candidates: list[Path] = []
    if model == "catflow" and dataset == "community_small":
        candidates.append(
            _REPO_ROOT
            / "configs"
            / "baselines"
            / "catflow_community_small_linear_v2.yaml"
        )
    candidates.append(
        _REPO_ROOT / "configs" / "baselines" / f"{model}_{dataset}.yaml"
    )
    for path in candidates:
        if path.is_file():
            return path
    return None


def load_common_config(path: Path, *, dataset: str) -> ComparisonProfile:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing common baseline config: {resolved}")
    loaded = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, Mapping):
        raise TypeError("The common baseline config must contain a mapping.")
    common = loaded.get("baseline_common", loaded)
    common = _mapping(common, name="baseline_common")
    version = int(common.get("schema_version", 0))
    if version != COMMON_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported common baseline schema_version {version}; "
            f"expected {COMMON_CONFIG_SCHEMA_VERSION}."
        )
    configured_dataset = str(common.get("dataset", "")).strip()
    if configured_dataset != dataset:
        raise ValueError(
            f"Common config targets {configured_dataset!r}, but --dataset is {dataset!r}."
        )
    provenance = {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "schema_version": version,
        "dataset": configured_dataset,
        "reference": _mapping(common.get("reference"), name="baseline_common.reference"),
        "reference_training": _mapping(common.get("training"), name="baseline_common.training"),
        "data": _mapping(common.get("data"), name="baseline_common.data"),
        "policy": _mapping(common.get("policy"), name="baseline_common.policy"),
        "protocol": _mapping(common.get("protocol"), name="baseline_common.protocol"),
    }
    return ComparisonProfile(resolved, dataset, common, provenance)


def resolve_common_config(
    dataset: str,
    explicit: Path | None,
    *,
    disabled: bool = False,
) -> ComparisonProfile | None:
    if disabled:
        if explicit is not None:
            raise ValueError("--common-config and --no-common-config cannot be used together.")
        return None
    path = explicit or default_common_config(dataset)
    return load_common_config(path, dataset=dataset) if path is not None else None


def resolve_wrapper_config(
    model_id: str,
    dataset: str,
    explicit: Path | None,
) -> Path | None:
    path = explicit or default_wrapper_config(model_id, dataset)
    if path is None:
        return None
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing baseline wrapper config: {resolved}")
    return resolved


def _append_override(overrides: list[str], key: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        text = "true" if value else "false"
    else:
        text = str(value)
    overrides.append(f"{key}={text}")


def common_defaults_for_model(
    model_id: str,
    profile: ComparisonProfile | None,
) -> dict[str, Any]:
    """Translate portable DeFoG-reference controls to wrapper-level defaults.

    These values are *fallbacks only*.  Wrapper YAML is merged after them.
    Model-specific schedules/optimizers therefore remain authoritative where a
    raw DeFoG epoch/LR/EMA value is not comparable.
    """

    if profile is None:
        return {}
    model = normalize_baseline_id(model_id)
    training = _mapping(profile.common.get("training"), name="baseline_common.training")
    data = _mapping(profile.common.get("data"), name="baseline_common.data")

    epochs = training.get("epochs")
    batch = training.get("batch_size")
    lr = training.get("learning_rate")
    wd = training.get("weight_decay")
    clip = training.get("gradient_clip_norm")
    ema = training.get("ema_decay")
    workers = training.get("num_workers")
    val_every = training.get("validation_every_epochs")

    if model == "catflow":
        train: dict[str, Any] = {}
        if epochs is not None: train["epochs"] = epochs
        if batch is not None: train["batch_size"] = batch
        if lr is not None: train["lr"] = lr
        if ema is not None: train["ema"] = ema
        if val_every is not None: train["validation_every"] = val_every
        return {"train": train}

    if model in {"gdsm", "gdsm_simple"}:
        train = {}
        for key, value in (
            ("epochs", epochs), ("batch_size", batch), ("lr", lr),
            ("weight_decay", wd), ("ema", ema), ("grad_norm", clip),
        ):
            if value is not None:
                train[key] = value
        return {"train": train}

    if model == "edge":
        train = {}
        for key, value in (("epochs", epochs), ("batch_size", batch), ("lr", lr)):
            if value is not None:
                train[key] = value
        if clip is not None:
            train["clip_value"] = clip
        return {"train": train}

    if model == "spectre":
        result: dict[str, Any] = {"train": {}, "model": {}}
        if epochs is not None: result["train"]["epochs"] = epochs
        if batch is not None: result["train"]["batch_size"] = batch
        if lr is not None:
            result["model"]["lr_g"] = lr
            result["model"]["lr_d"] = lr
        if wd is not None: result["model"]["weight_decay"] = wd
        if ema is not None: result["model"]["ema"] = ema
        if clip is not None: result["model"]["clip_grad_norm"] = clip
        return result

    if model == "digress":
        result = {}
        for key, value in (
            ("n_epochs", epochs), ("batch_size", batch),
            ("num_workers", workers), ("check_val_every_n_epochs", val_every),
        ):
            if value is not None:
                result[key] = value
        hydra: list[str] = []
        _append_override(hydra, "train.lr", lr)
        _append_override(hydra, "train.weight_decay", wd)
        _append_override(hydra, "train.ema_decay", ema)
        if clip is not None:
            _append_override(hydra, "train.clip_grad", clip)
        optimizer = training.get("optimizer")
        _append_override(hydra, "train.optimizer", optimizer)
        if hydra:
            result["hydra_overrides"] = hydra
        return result

    if model == "gdss":
        result: dict[str, Any] = {"train": {}}
        if epochs is not None: result["train"]["num_epochs"] = epochs
        if lr is not None: result["train"]["lr"] = lr
        if wd is not None: result["train"]["weight_decay"] = wd
        if ema is not None: result["train"]["ema"] = ema
        if clip is not None: result["train"]["grad_norm"] = clip
        if batch is not None: result["batch_size"] = batch
        if workers is not None: result["num_workers"] = workers
        return result

    if model == "graphrnn":
        result = {}
        for key, value in (
            ("epochs", epochs), ("batch_size", batch),
            ("learning_rate", lr), ("num_workers", workers),
            ("gradient_clip_norm", clip),
        ):
            if value is not None:
                result[key] = value
        return result

    if model == "hog_diff":
        # HOG-Diff has two semantically different training stages (VPSDE and
        # OU bridge).  A single DeFoG epoch count/LR cannot be translated
        # honestly.  Only the loader-level worker count is portable here;
        # stage budgets live explicitly in hog_diff_<dataset>.yaml.
        return {"num_workers": workers} if workers is not None else {}

    if model == "defog":
        result = {}
        key_map = {
            "epochs": "n_epochs",
            "batch_size": "batch_size",
            "num_workers": "num_workers",
            "validation_every_epochs": "check_val_every_n_epochs",
            "optimizer": "optimizer",
            "learning_rate": "learning_rate",
            "weight_decay": "weight_decay",
            "gradient_clip_norm": "gradient_clip_norm",
            "ema_decay": "ema_decay",
        }
        for source, target in key_map.items():
            if source in training and training[source] is not None:
                result[target] = training[source]
        for source, target in (
            ("remove_hydrogens", "remove_hydrogens"),
            ("pin_memory", "pin_memory"),
            ("aromatic", "aromatic"),
        ):
            if source in data and data[source] is not None:
                result[target] = data[source]
        reference = _mapping(profile.common.get("reference"), name="baseline_common.reference")
        if reference.get("native_dataset"):
            result["native_dataset"] = reference["native_dataset"]
        if reference.get("experiment"):
            result["experiment"] = reference["experiment"]
        return result

    return {}


def model_comparison_metadata(config_path: Path | None) -> dict[str, Any]:
    if config_path is None:
        return {}
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, Mapping):
        return {}
    comparison = loaded.get("comparison", {}) or {}
    return _mapping(comparison, name="comparison")


def comparison_request_options(
    model_id: str,
    profile: ComparisonProfile | None,
    *,
    model_config: Path | None,
) -> dict[str, Any]:
    metadata = model_comparison_metadata(model_config)
    if profile is None:
        configured_model = metadata.get("model")
        if configured_model and normalize_baseline_id(str(configured_model)) != normalize_baseline_id(model_id):
            raise ValueError(
                f"Model comparison metadata targets {configured_model!r}, not {model_id!r}."
            )
        if not metadata:
            return {}
        return {
            "comparison_reference": {
                "model": normalize_baseline_id(model_id),
                "common_profile": None,
                "model_config": str(model_config) if model_config is not None else None,
                "model_comparison": metadata,
            }
        }
    configured_dataset = str(metadata.get("dataset", profile.dataset))
    if configured_dataset and configured_dataset != profile.dataset:
        raise ValueError(
            f"Model comparison metadata targets {configured_dataset!r}, "
            f"but common profile targets {profile.dataset!r}."
        )
    configured_model = metadata.get("model")
    if configured_model:
        if normalize_baseline_id(str(configured_model)) != normalize_baseline_id(model_id):
            raise ValueError(
                f"Model comparison metadata targets {configured_model!r}, not {model_id!r}."
            )
    expected_common = metadata.get("common_config")
    if expected_common and Path(str(expected_common)).name != profile.path.name:
        raise ValueError(
            f"Model config expects common profile {expected_common!r}, "
            f"but {profile.path.name!r} was selected."
        )
    provenance = copy.deepcopy(profile.provenance)
    provenance["model"] = normalize_baseline_id(model_id)
    provenance["model_config"] = str(model_config) if model_config is not None else None
    provenance["model_comparison"] = metadata
    return {
        "comparison_defaults": common_defaults_for_model(model_id, profile),
        "comparison_reference": provenance,
    }

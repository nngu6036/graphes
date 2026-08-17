"""Common GraphER wrapper for the project-owned DH-VAE + HH baseline.

The wrapper delegates model fitting to the maintained
``train_degree_generator.py`` entrypoint and delegates generation to the
canonical samplers and constructors in this package. It adds experiment
orchestration: immutable dataset references, run-scoped artifacts, exact-count
generation, checksums, manifests, and atomic publication.
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
import tempfile
import time
from collections import Counter
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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


TRAINING_MANIFEST_FORMAT = "grapher_dhvae_hh_training_v1"
TRAINING_ESTIMATES_MANIFEST_FORMAT = (
    "grapher_dhvae_hh_training_estimates_v1"
)
GENERATION_MANIFEST_FORMAT = "grapher_dhvae_hh_generation_v1"

_GENERIC_TYPES = frozenset(
    {"degree_histogram_vae", "degree_vae", "vae", "learned"}
)
_TYPED_TYPES = frozenset(
    {
        "typed_degree_histogram_vae",
        "typed_signature_histogram_vae",
        "typed_signature_vae",
    }
)
_DEFAULT_CONFIG_BY_BENCHMARK = {
    "community_small": "community_small.yaml",
    "ego_small": "ego_small.yaml",
    "grid": "grid.yaml",
    "qm9": "qm9_typed.yaml",
    "qm9_attributed": "qm9_typed.yaml",
    "zinc": "zinc_typed.yaml",
    "zinc_attributed": "zinc_typed.yaml",
}
_TRAIN_OPTION_KEYS = frozenset(
    {
        "config_overrides",
        "constructor",
        "degree_generator",
        "runtime",
        "summary",
        "training_estimates",
        "typed_signature",
    }
)
_GENERATE_OPTION_KEYS = frozenset(
    {
        "config_overrides",
        "constructor",
        "max_attempts_per_graph",
        "runtime",
        "sampling",
        "typed_signature",
    }
)
_TRAIN_RUNTIME_KEYS = frozenset({"device", "timeout_seconds"})
_GENERATE_RUNTIME_KEYS = frozenset({"device"})
_TRAINING_ESTIMATE_OPTION_KEYS = frozenset(
    {
        "constructor",
        "enabled",
        "max_attempts_per_graph",
        "num_graphs",
        "runtime",
        "sampling",
        "seed",
    }
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[4]


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
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
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


def _atomic_yaml(path: Path, value: Mapping[str, Any]) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        yaml.safe_dump(_jsonable(value), sort_keys=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{label} must contain a JSON object: {path}.")
    return value


def _read_yaml(path: Path, *, label: str) -> dict[str, Any]:
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise TypeError(f"{label} must contain a YAML mapping: {path}.")
    return value


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    return dict(value)


def _boolean(value: Any, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a YAML/Python boolean.")
    return value


def _deep_update(
    base: dict[str, Any], update: Mapping[str, Any]
) -> dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            base[key] = _deep_update(dict(base[key]), value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _validate_option_keys(
    options: Mapping[str, Any], allowed: frozenset[str], *, operation: str
) -> None:
    unknown = sorted(str(key) for key in options if str(key) not in allowed)
    if unknown:
        raise ValueError(
            f"Unknown DH-VAE+HH {operation} options: {unknown}. "
            f"Supported keys are {sorted(allowed)}."
        )


def _generator_type(config: Mapping[str, Any]) -> str:
    degree = _mapping(config.get("degree_generator"), name="degree_generator")
    value = str(degree.get("type", "degree_histogram_vae")).lower()
    if value not in _GENERIC_TYPES | _TYPED_TYPES:
        raise ValueError(f"Unknown DH-VAE generator type: {value!r}.")
    return value


def _is_typed(config: Mapping[str, Any]) -> bool:
    return _generator_type(config) in _TYPED_TYPES


def _default_experiment_config(benchmark_id: str) -> Path:
    filename = _DEFAULT_CONFIG_BY_BENCHMARK.get(str(benchmark_id).lower())
    if filename is None:
        raise ValueError(
            "No default DH-VAE+HH configuration is declared for benchmark "
            f"{benchmark_id!r}. Pass TrainRequest.config_path explicitly."
        )
    return _project_root() / "configs" / "experiments" / "dhvae" / filename


def _resolve_dataset_config(request: TrainRequest) -> Path:
    if request.dataset.config_path is not None:
        path = request.dataset.config_path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing dataset config: {path}.")
        return path
    candidates = (
        _project_root()
        / "configs"
        / "datasets"
        / f"{request.dataset.benchmark_id}.yaml",
        _project_root()
        / "configs"
        / "datasets"
        / f"{request.dataset.serialized_id}.yaml",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "No dataset configuration was provided and no project default exists "
        f"for {request.dataset.benchmark_id!r}."
    )


def _load_training_config(request: TrainRequest) -> tuple[dict[str, Any], Path]:
    path = (
        request.config_path.expanduser().resolve()
        if request.config_path is not None
        else _default_experiment_config(request.dataset.benchmark_id).resolve()
    )
    if not path.is_file():
        raise FileNotFoundError(f"Missing DH-VAE experiment config: {path}.")
    config = _read_yaml(path, label="DH-VAE experiment config")
    return config, path


def _dataset_graphs(path: Path) -> list[Any]:
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, list):
        raise TypeError(f"Prepared graph split must contain a list: {path}.")
    for index, graph in enumerate(value):
        if not hasattr(graph, "nodes") or not hasattr(graph, "edges"):
            raise TypeError(
                f"Prepared graph split item {index} is not graph-like: {path}."
            )
    return value


def _split_hashes(split_paths: Mapping[str, Path]) -> dict[str, str]:
    return {name: _sha256(path) for name, path in split_paths.items()}


def _verify_split_hashes(
    split_paths: Mapping[str, Path], expected: Mapping[str, str]
) -> None:
    observed = _split_hashes(split_paths)
    if observed != dict(expected):
        raise RuntimeError(
            "A prepared dataset split changed while DH-VAE was training: "
            f"expected={dict(expected)}, observed={observed}."
        )


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


def _log_tail(path: Path, *, lines: int = 200) -> str:
    if not path.is_file():
        return ""
    return "\n".join(
        path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
    )


def _training_environment(seed: int) -> dict[str, str]:
    environment = dict(os.environ)
    source = str((_project_root() / "src").resolve())
    inherited = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = os.pathsep.join(
        [value for value in (source, inherited) if value]
    )
    environment["PYTHONHASHSEED"] = str(int(seed))
    environment.setdefault("MPLBACKEND", "Agg")
    return environment


def _run_training_subprocess(
    *, config_path: Path, log_path: Path, seed: int, timeout: float | None
) -> list[str]:
    script = _project_root() / "scripts" / "train_degree_generator.py"
    if not script.is_file():
        raise FileNotFoundError(f"Missing DH-VAE training script: {script}.")
    command = [sys.executable, str(script), "--config", str(config_path)]
    environment = _training_environment(seed)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"[DH-VAE+HH training] started_at={_utc_now()}\n")
        log.write(f"[DH-VAE+HH training] cwd={_project_root()}\n")
        log.write(f"[DH-VAE+HH training] argv={json.dumps(command)}\n")
        log.flush()
        try:
            result = subprocess.run(
                command,
                cwd=str(_project_root()),
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
                shell=False,
                start_new_session=True,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                "DH-VAE training timed out.\n"
                f"Log: {log_path.resolve()}\n"
                f"Command: {json.dumps(command)}\n"
                f"Last output:\n{_log_tail(log_path)}"
            ) from exc
    if result.returncode != 0:
        raise RuntimeError(
            f"DH-VAE training exited with code {result.returncode}.\n"
            f"Log: {log_path.resolve()}\n"
            f"Command: {json.dumps(command)}\n"
            f"Last output:\n{_log_tail(log_path)}"
        )
    return command


def _torch_load(path: Path) -> dict[str, Any]:
    import torch

    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise TypeError(f"DH-VAE checkpoint must contain a mapping: {path}.")
    return value


def _normalize_checkpoint_config(
    checkpoint_path: Path, resolved_config: Mapping[str, Any]
) -> None:
    """Replace staging paths embedded by the existing training CLI."""

    import torch

    checkpoint = _torch_load(checkpoint_path)
    config_record = checkpoint.get("config")
    if not isinstance(config_record, Mapping):
        config_record = {}
    normalized = dict(config_record)
    normalized["experiment_config"] = copy.deepcopy(dict(resolved_config))
    checkpoint["config"] = normalized
    temporary = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(checkpoint_path)


def _checkpoint_experiment_config(path: Path) -> dict[str, Any]:
    checkpoint = _torch_load(path)
    config_record = checkpoint.get("config")
    if not isinstance(config_record, Mapping):
        raise RuntimeError(
            "DH-VAE checkpoint does not contain its training configuration. "
            "Use a checkpoint produced by the maintained trainer or provide a "
            "managed training run."
        )
    experiment = config_record.get("experiment_config")
    if not isinstance(experiment, Mapping):
        raise RuntimeError(
            "DH-VAE checkpoint has no embedded experiment_config mapping."
        )
    return copy.deepcopy(dict(experiment))


def _sampling_record(summary: Mapping[str, Any]) -> dict[str, Any]:
    raw = summary.get("sampling_diagnostics")
    if not isinstance(raw, Mapping):
        return {}
    keys = (
        "attempts_used",
        "raw_graphical",
        "raw_connected_feasible",
        "raw_even_degree_sum",
        "raw_degree_bounds_valid",
        "first_raw_feasible",
        "accepted_without_postprocessing",
        "fallback_used",
        "repair_used",
        "repair_l1_adjustment",
    )
    return {key: _jsonable(raw[key]) for key in keys if key in raw}


def _decorate_graph(graph: Any, *, index: int, seed: int, typed: bool) -> None:
    graph.graph["base_model"] = "dhvae_hh"
    graph.graph["raw_index"] = int(index)
    graph.graph["generation_seed"] = int(seed)
    graph.graph["invariant_kind"] = (
        "typed_degree_signature" if typed else "degree_sequence"
    )


def _complete_molecular_aliases(graph: Any, config: Mapping[str, Any]) -> None:
    from grapher.rewiring_mlp.molecular.constraints import bond_order

    typed = _mapping(config.get("typed_signature"), name="typed_signature")
    node_attribute = str(typed.get("node_attribute", "atomic_num"))
    edge_attribute = str(typed.get("edge_attribute", "bond_type"))
    if node_attribute == "atomic_num":
        for _node, data in graph.nodes(data=True):
            data.setdefault("atom_type", int(data["atomic_num"]))
    if edge_attribute == "bond_type":
        for _left, _right, data in graph.edges(data=True):
            data.setdefault("bond_order", float(bond_order(int(data["bond_type"]))))


def _effective_generation_config(
    base: Mapping[str, Any], options: Mapping[str, Any]
) -> dict[str, Any]:
    config = copy.deepcopy(dict(base))
    config = _deep_update(
        config,
        _mapping(options.get("config_overrides"), name="config_overrides"),
    )
    for section in ("constructor", "typed_signature"):
        if section in options:
            config[section] = _deep_update(
                _mapping(config.get(section), name=section),
                _mapping(options[section], name=section),
            )
    degree = _mapping(config.get("degree_generator"), name="degree_generator")
    degree = _deep_update(
        degree,
        _mapping(options.get("sampling"), name="sampling"),
    )
    runtime = _mapping(options.get("runtime"), name="runtime")
    if "device" in runtime:
        degree["device"] = str(runtime["device"])
    config["degree_generator"] = degree
    return config


def _generate_batch(
    *,
    checkpoint_path: Path,
    config: Mapping[str, Any],
    num_graphs: int,
    seed: int,
    max_attempts_per_graph: int,
) -> tuple[list[Any], dict[str, Any]]:
    import numpy as np
    import torch

    from grapher.models.dhvae_hh.degree_sampler import (
        DegreeVAESampler,
        TypedDegreeVAESampler,
    )
    from grapher.models.dhvae_hh.havel_hakimi import (
        assert_constructor_validity,
        construct_coarse_graph,
    )
    from grapher.models.dhvae_hh.typed_constructor import (
        TypedConstructionError,
        construct_typed_graph,
    )
    from grapher.rewiring_mlp.molecular.typed_invariants import TypedInvariant

    if int(num_graphs) <= 0:
        raise ValueError("num_graphs must be positive.")
    if int(max_attempts_per_graph) <= 0:
        raise ValueError("max_attempts_per_graph must be positive.")
    typed = _is_typed(config)
    degree = _mapping(config.get("degree_generator"), name="degree_generator")
    degree["checkpoint_path"] = str(checkpoint_path.resolve())
    constructor = _mapping(config.get("constructor"), name="constructor")
    if typed:
        typed_config = _mapping(
            config.get("typed_signature"), name="typed_signature"
        )
        if typed_config.get("max_ordinary_degree") is not None:
            constructor.setdefault(
                "max_ordinary_degree", typed_config["max_ordinary_degree"]
            )
        if typed_config.get("max_weighted_valence") is not None:
            constructor.setdefault(
                "max_weighted_valence", typed_config["max_weighted_valence"]
            )
        sampler: Any = TypedDegreeVAESampler.from_config(degree, seed=seed)
    else:
        constructor_type = str(constructor.get("type", "havel_hakimi")).lower()
        if constructor_type != "havel_hakimi":
            raise ValueError(
                "Ordinary DH-VAE generation requires constructor.type: "
                "havel_hakimi."
            )
        degree.setdefault("postprocess_policy", "reject_only")
        degree.setdefault("fallback", "error")
        sampler = DegreeVAESampler.from_config(degree, seed=seed)

    rng = np.random.default_rng(int(seed))
    torch.manual_seed(int(seed))
    graphs: list[Any] = []
    records: list[dict[str, Any]] = []
    rejection_reasons: Counter[str] = Counter()
    total_attempts = 0
    for graph_index in range(int(num_graphs)):
        for local_attempt in range(1, int(max_attempts_per_graph) + 1):
            total_attempts += 1
            try:
                summary = sampler.sample(rng)
            except (RuntimeError, TypeError, ValueError) as exc:
                rejection_reasons[
                    f"degree_sample:{type(exc).__name__}"
                ] += 1
                continue
            try:
                if typed:
                    invariant = TypedInvariant.from_dict(summary["typed_invariant"])
                    graph, constructor_record = construct_typed_graph(
                        invariant,
                        constructor,
                        rng,
                    )
                    _complete_molecular_aliases(graph, config)
                else:
                    graph = construct_coarse_graph(summary, constructor, rng)
                    assert_constructor_validity(
                        graph,
                        summary,
                        require_connected=bool(
                            constructor.get("ensure_connected", True)
                        ),
                    )
                    constructor_record = {"success": True}
            except TypedConstructionError as exc:
                reason = str(
                    exc.diagnostics.get("failure_reason")
                    or "typed_construction_failed"
                )
                rejection_reasons[f"constructor:{reason}"] += 1
                continue
            except (
                AssertionError,
                KeyError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as exc:
                rejection_reasons[
                    f"constructor:{type(exc).__name__}"
                ] += 1
                continue
            _decorate_graph(
                graph,
                index=graph_index,
                seed=seed,
                typed=typed,
            )
            graphs.append(graph)
            records.append(
                {
                    "raw_index": graph_index,
                    "attempts": local_attempt,
                    "sampling": _sampling_record(summary),
                    "constructor": _jsonable(constructor_record),
                }
            )
            break
        else:
            raise RuntimeError(
                "DH-VAE+HH generation exhausted "
                f"{max_attempts_per_graph} attempts for graph {graph_index}; "
                f"returned={len(graphs)}/{num_graphs}, "
                f"rejections={dict(rejection_reasons)}."
            )
    if len(graphs) != int(num_graphs):
        raise RuntimeError(
            f"DH-VAE+HH returned {len(graphs)} graphs; expected {num_graphs}."
        )
    return graphs, {
        "requested": int(num_graphs),
        "returned": len(graphs),
        "total_attempts": total_attempts,
        "max_attempts_per_graph": int(max_attempts_per_graph),
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "sample_order": "raw_index_0_to_n_minus_1",
        "dropped_after_acceptance": 0,
        "records": records,
    }


def _managed_training_manifest(
    layout: ArtifactLayout,
) -> dict[str, Any] | None:
    if not layout.training_manifest_path.is_file():
        return None
    return _read_json(
        layout.training_manifest_path,
        label="DH-VAE+HH training manifest",
    )


def _generation_base_config(
    request: GenerateRequest,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    layout = request.run.layout
    training_manifest = _managed_training_manifest(layout)
    if training_manifest is not None:
        checkpoint_record = training_manifest.get("checkpoint")
        if not isinstance(checkpoint_record, Mapping):
            raise RuntimeError("Managed DH-VAE training manifest has no checkpoint.")
        expected_hash = str(checkpoint_record.get("sha256", ""))
        observed_hash = _sha256(request.checkpoint_path)
        if not expected_hash or observed_hash != expected_hash:
            raise RuntimeError(
                "The requested DH-VAE checkpoint does not match this managed "
                f"run: expected {expected_hash}, observed {observed_hash}."
            )
        if not layout.resolved_training_config_path.is_file():
            raise FileNotFoundError(
                "Managed DH-VAE run is missing resolved_config.yaml."
            )
        config_record = training_manifest.get("resolved_config")
        if not isinstance(config_record, Mapping):
            raise RuntimeError(
                "Managed DH-VAE training manifest has no resolved config record."
            )
        expected_config_hash = str(config_record.get("sha256", ""))
        if (
            not expected_config_hash
            or _sha256(layout.resolved_training_config_path)
            != expected_config_hash
        ):
            raise RuntimeError(
                "The persisted DH-VAE resolved configuration changed after "
                "training."
            )
        return (
            _read_yaml(
                layout.resolved_training_config_path,
                label="DH-VAE resolved config",
            ),
            training_manifest,
        )
    return _checkpoint_experiment_config(request.checkpoint_path), None


def _preserve_failure(
    *,
    layout: ArtifactLayout,
    operation: str,
    error: BaseException,
    log_path: Path | None,
) -> Path:
    target = layout.run_dir / "failures" / f"attempt-{time.time_ns()}"
    target.mkdir(parents=True, exist_ok=False)
    if log_path is not None and log_path.is_file():
        shutil.copy2(log_path, target / log_path.name)
    _atomic_json(
        target / "failure.json",
        {
            "format": "grapher_dhvae_hh_failure_v1",
            "operation": operation,
            "model_id": "dhvae_hh",
            "dataset_id": layout.dataset_id,
            "run_id": layout.run_id,
            "failed_at": _utc_now(),
            "exception_type": type(error).__name__,
            "exception": str(error),
            "log": (
                log_path.name
                if log_path is not None and log_path.is_file()
                else None
            ),
        },
    )
    return target


class DHVAEHHWrapper(BaseGeneratorWrapper):
    """Train DH-VAE and generate exact HH realizations of sampled invariants."""

    model_id = "dhvae_hh"
    display_name = "DH-VAE + randomized Havel--Hakimi"
    capabilities = BaselineCapabilities(
        domains=frozenset({"generic", "attributed"}),
        isolation="subprocess",
        status="ready",
    )
    implementation_note = (
        "Training delegates to the maintained project DH-VAE trainer; "
        "generation uses the canonical ordinary or typed sampler and exact "
        "Havel--Hakimi/typed constructor."
    )

    def train(self, request: TrainRequest) -> TrainingArtifacts:
        self.validate_train_request(request)
        if request.resume_from is not None:
            raise NotImplementedError(
                "DHVAEHHWrapper does not support optimizer-state resume. Use a "
                "new run_id for a fresh, fully recorded training run."
            )
        options = dict(request.options)
        _validate_option_keys(options, _TRAIN_OPTION_KEYS, operation="training")
        config, source_config_path = _load_training_config(request)
        config = _deep_update(
            config,
            _mapping(options.get("config_overrides"), name="config_overrides"),
        )
        for section in (
            "degree_generator",
            "constructor",
            "summary",
            "typed_signature",
        ):
            if section in options:
                config[section] = _deep_update(
                    _mapping(config.get(section), name=section),
                    _mapping(options[section], name=section),
                )
        runtime = _mapping(options.get("runtime"), name="runtime")
        _validate_option_keys(
            runtime,
            _TRAIN_RUNTIME_KEYS,
            operation="training runtime",
        )
        if "device" in runtime:
            degree = _mapping(
                config.get("degree_generator"), name="degree_generator"
            )
            degree["device"] = str(runtime["device"])
            config["degree_generator"] = degree
        timeout_value = runtime.get("timeout_seconds")
        timeout = None if timeout_value is None else float(timeout_value)
        if timeout is not None and timeout <= 0:
            raise ValueError("runtime.timeout_seconds must be positive.")
        estimates_options = _mapping(
            options.get("training_estimates"), name="training_estimates"
        )
        _validate_option_keys(
            estimates_options,
            _TRAINING_ESTIMATE_OPTION_KEYS,
            operation="training-estimate",
        )
        estimates_enabled = _boolean(
            estimates_options.get("enabled", True),
            name="training_estimates.enabled",
        )
        estimate_runtime = _mapping(
            estimates_options.get("runtime"),
            name="training_estimates.runtime",
        )
        _validate_option_keys(
            estimate_runtime,
            _GENERATE_RUNTIME_KEYS,
            operation="training-estimate runtime",
        )

        layout = request.run.layout
        if (
            request.overwrite
            and layout.generations_dir.is_dir()
            and any(layout.generations_dir.iterdir())
        ):
            raise ArtifactCollisionError(
                "Cannot overwrite a DH-VAE training run that already has "
                "generated batches. Use a new run_id."
            )
        ArtifactLayout.require_available(
            layout.train_dir, overwrite=request.overwrite
        )
        split_paths = request.dataset.split_paths
        initial_split_hashes = _split_hashes(split_paths)
        dataset_fingerprint = request.dataset.fingerprint()
        train_graphs = _dataset_graphs(split_paths["train"])
        if not train_graphs:
            raise RuntimeError("DH-VAE training requires a non-empty train split.")
        dataset_config_path = _resolve_dataset_config(request)
        source_config_hash = _sha256(source_config_path)
        dataset_config_hash = _sha256(dataset_config_path)
        generator_type = _generator_type(config)
        typed = generator_type in _TYPED_TYPES
        default_estimate_count = (
            min(len(train_graphs), 1024) if typed else len(train_graphs)
        )
        estimate_count = (
            int(estimates_options.get("num_graphs", default_estimate_count))
            if estimates_enabled
            else 0
        )
        if estimates_enabled and estimate_count <= 0:
            raise ValueError("training_estimates.num_graphs must be positive.")
        estimate_seed = int(
            estimates_options.get("seed", request.run.train_seed)
        )
        estimate_max_attempts = int(
            estimates_options.get("max_attempts_per_graph", 100)
        )
        if estimates_enabled and estimate_max_attempts <= 0:
            raise ValueError(
                "training_estimates.max_attempts_per_graph must be positive."
            )

        staging_root = layout.output_root.expanduser().resolve() / ".staging"
        staging_root.mkdir(parents=True, exist_ok=True)
        workspace = Path(
            tempfile.mkdtemp(prefix="dhvae-hh-train-", dir=staging_root)
        )
        stage_train = workspace / "train"
        stage_train.mkdir()
        log_path = stage_train / "train.log"
        worker_config_path = workspace / "worker_config.yaml"
        checkpoint_path = stage_train / "checkpoints" / "checkpoint.pt"
        final_checkpoint_path = layout.checkpoints_dir / "checkpoint.pt"
        started_at = _utc_now()
        started = time.monotonic()
        try:
            config["seed"] = int(request.run.train_seed)
            dataset_section = _mapping(config.get("dataset"), name="dataset")
            dataset_section.update(
                {
                    "name": str(request.dataset.serialized_id),
                    "benchmark": request.dataset.benchmark_id,
                    "root": str(request.dataset.root.expanduser().resolve()),
                    "config_path": str(dataset_config_path),
                    "build_if_missing": False,
                }
            )
            config["dataset"] = dataset_section
            degree = _mapping(
                config.get("degree_generator"), name="degree_generator"
            )
            degree["checkpoint_path"] = str(final_checkpoint_path.resolve())
            config["degree_generator"] = degree
            worker_config = copy.deepcopy(config)
            worker_config["degree_generator"]["checkpoint_path"] = str(
                checkpoint_path.resolve()
            )
            _atomic_yaml(worker_config_path, worker_config)
            command = _run_training_subprocess(
                config_path=worker_config_path,
                log_path=log_path,
                seed=request.run.train_seed,
                timeout=timeout,
            )
            if not checkpoint_path.is_file():
                raise RuntimeError(
                    "DH-VAE training finished without checkpoint.pt."
                )
            metrics_path = checkpoint_path.parent / "training_metrics.json"
            vectorizer_name = (
                "typed_signature_vectorizer.json"
                if generator_type in _TYPED_TYPES
                else "degree_vectorizer.json"
            )
            vectorizer_path = checkpoint_path.parent / vectorizer_name
            for required in (metrics_path, vectorizer_path):
                if not required.is_file():
                    raise RuntimeError(
                        f"DH-VAE training did not publish {required.name}."
                    )
            _normalize_checkpoint_config(checkpoint_path, config)
            _verify_split_hashes(split_paths, initial_split_hashes)
            _atomic_yaml(stage_train / "resolved_config.yaml", config)

            if estimates_enabled:
                estimate_options = {
                    "sampling": _mapping(
                        estimates_options.get("sampling"),
                        name="training_estimates.sampling",
                    ),
                    "constructor": _mapping(
                        estimates_options.get("constructor"),
                        name="training_estimates.constructor",
                    ),
                    "runtime": estimate_runtime,
                }
                estimate_config = _effective_generation_config(
                    config, estimate_options
                )
                estimate_graphs, estimate_diagnostics = _generate_batch(
                    checkpoint_path=checkpoint_path,
                    config=estimate_config,
                    num_graphs=estimate_count,
                    seed=estimate_seed,
                    max_attempts_per_graph=estimate_max_attempts,
                )
                estimate_dir = stage_train / "training_estimates"
                estimated_path = estimate_dir / "estimated_graphs.pkl"
                ground_truth_path = estimate_dir / "ground_truth_graphs.pkl"
                _atomic_pickle(estimated_path, estimate_graphs)
                ground_truth_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(split_paths["train"], ground_truth_path)
                estimate_log = estimate_dir / "generate.log"
                estimate_log.write_text(
                    json.dumps(_jsonable(estimate_diagnostics), indent=2) + "\n",
                    encoding="utf-8",
                )
                _atomic_json(
                    estimate_dir / "manifest.json",
                    {
                        "format": TRAINING_ESTIMATES_MANIFEST_FORMAT,
                        "semantics": "independent_unconditional_sample_pool",
                        "model_id": self.model_id,
                        "dataset_id": request.run.dataset_id,
                        "run_id": request.run.run_id,
                        "checkpoint_sha256": _sha256(checkpoint_path),
                        "estimated_graphs": {
                            "path": "estimated_graphs.pkl",
                            "count": len(estimate_graphs),
                            "sha256": _sha256(estimated_path),
                            "seed": estimate_seed,
                        },
                        "ground_truth_graphs": {
                            "path": "ground_truth_graphs.pkl",
                            "count": len(train_graphs),
                            "sha256": _sha256(ground_truth_path),
                            "split": "train",
                            "order": "exact_prepared_split_order",
                        },
                        "pairing": {
                            "status": "unpaired",
                            "pair_count": 0,
                            "reason": "unconditional_prior_sampling",
                        },
                        "generation": estimate_diagnostics,
                    },
                )
            _verify_split_hashes(split_paths, initial_split_hashes)
            if _sha256(source_config_path) != source_config_hash:
                raise RuntimeError(
                    "The DH-VAE source experiment config changed during training."
                )
            if _sha256(dataset_config_path) != dataset_config_hash:
                raise RuntimeError(
                    "The dataset config changed during DH-VAE training."
                )
            resolved_path = stage_train / "resolved_config.yaml"
            duration = time.monotonic() - started
            constructor_type = str(
                _mapping(config.get("constructor"), name="constructor").get(
                    "type",
                    (
                        "typed_backtracking"
                        if generator_type in _TYPED_TYPES
                        else "havel_hakimi"
                    ),
                )
            )
            manifest = {
                "format": TRAINING_MANIFEST_FORMAT,
                "model_id": self.model_id,
                "dataset": {
                    "benchmark_id": request.dataset.benchmark_id,
                    "serialized_id": request.dataset.serialized_id,
                    "fingerprint": dataset_fingerprint,
                    "split_sha256": initial_split_hashes,
                    "train_graph_count": len(train_graphs),
                },
                "run_id": request.run.run_id,
                "train_seed": request.run.train_seed,
                "started_at": started_at,
                "finished_at": _utc_now(),
                "duration_seconds": duration,
                "generator_type": generator_type,
                "constructor_type": constructor_type,
                "source_config": {
                    "path": str(source_config_path),
                    "sha256": source_config_hash,
                },
                "dataset_config": {
                    "path": str(dataset_config_path),
                    "sha256": dataset_config_hash,
                },
                "checkpoint": {
                    "path": "checkpoints/checkpoint.pt",
                    "sha256": _sha256(checkpoint_path),
                },
                "vectorizer": {
                    "path": f"checkpoints/{vectorizer_name}",
                    "sha256": _sha256(vectorizer_path),
                },
                "training_metrics": {
                    "path": "checkpoints/training_metrics.json",
                    "sha256": _sha256(metrics_path),
                },
                "resolved_config": {
                    "path": "resolved_config.yaml",
                    "sha256": _sha256(resolved_path),
                },
                "command": {"argv": command, "shell": False},
                "training_estimates": {
                    "enabled": estimates_enabled,
                    "count": estimate_count,
                    "manifest": (
                        "training_estimates/manifest.json"
                        if estimates_enabled
                        else None
                    ),
                    "pairing_status": (
                        "unpaired" if estimates_enabled else None
                    ),
                },
            }
            _atomic_json(stage_train / "manifest.json", manifest)
            _publish_directory(
                stage_train, layout.train_dir, overwrite=request.overwrite
            )
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
        except Exception as exc:
            failure_path = _preserve_failure(
                layout=layout,
                operation="train",
                error=exc,
                log_path=log_path,
            )
            raise RuntimeError(
                f"{exc}\nFailure artifacts preserved at: "
                f"{failure_path.resolve()}"
            ) from exc
        finally:
            if workspace.exists():
                shutil.rmtree(workspace)

        return TrainingArtifacts(
            run_dir=layout.run_dir,
            checkpoint_path=layout.checkpoints_dir / "checkpoint.pt",
            manifest_path=layout.training_manifest_path,
            log_path=layout.training_log_path,
            artifacts=(
                layout.resolved_training_config_path,
                layout.checkpoints_dir / vectorizer_name,
                layout.checkpoints_dir / "training_metrics.json",
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

    def generate(self, request: GenerateRequest) -> GenerationArtifacts:
        self.validate_generate_request(request)
        options = dict(request.options)
        _validate_option_keys(options, _GENERATE_OPTION_KEYS, operation="generation")
        generation_runtime = _mapping(options.get("runtime"), name="runtime")
        _validate_option_keys(
            generation_runtime,
            _GENERATE_RUNTIME_KEYS,
            operation="generation runtime",
        )
        base_config, training_manifest = _generation_base_config(request)
        config = _effective_generation_config(base_config, options)
        max_attempts = int(options.get("max_attempts_per_graph", 100))
        if max_attempts <= 0:
            raise ValueError("max_attempts_per_graph must be positive.")

        layout = request.run.layout
        generation_id = request.resolved_generation_id
        generation_dir = layout.generation_dir(generation_id)
        ArtifactLayout.require_available(
            generation_dir, overwrite=request.overwrite
        )
        staging_root = layout.output_root.expanduser().resolve() / ".staging"
        staging_root.mkdir(parents=True, exist_ok=True)
        workspace = Path(
            tempfile.mkdtemp(prefix="dhvae-hh-generate-", dir=staging_root)
        )
        stage_generation = workspace / "generation"
        stage_generation.mkdir()
        log_path = stage_generation / "generate.log"
        started_at = _utc_now()
        started = time.monotonic()
        try:
            graphs, diagnostics = _generate_batch(
                checkpoint_path=request.checkpoint_path,
                config=config,
                num_graphs=request.num_graphs,
                seed=request.generation_seed,
                max_attempts_per_graph=max_attempts,
            )
            graphs_path = stage_generation / "base_graphs.pkl"
            _atomic_pickle(graphs_path, graphs)
            log_path.write_text(
                json.dumps(_jsonable(diagnostics), indent=2) + "\n",
                encoding="utf-8",
            )
            manifest = {
                "format": GENERATION_MANIFEST_FORMAT,
                "model_id": self.model_id,
                "dataset_id": request.run.dataset_id,
                "run_id": request.run.run_id,
                "generation_id": generation_id,
                "generation_seed": request.generation_seed,
                "started_at": started_at,
                "finished_at": _utc_now(),
                "duration_seconds": time.monotonic() - started,
                "generator_type": _generator_type(config),
                "checkpoint": {
                    "path": str(request.checkpoint_path.resolve()),
                    "sha256": _sha256(request.checkpoint_path),
                    "managed_training_run": training_manifest is not None,
                },
                "graphs": {
                    "path": "base_graphs.pkl",
                    "sha256": _sha256(graphs_path),
                    "requested": request.num_graphs,
                    "returned": len(graphs),
                    "order": "raw_index_0_to_n_minus_1",
                    "dropped": 0,
                },
                "sampling": _mapping(
                    config.get("degree_generator"), name="degree_generator"
                ),
                "constructor": _mapping(
                    config.get("constructor"), name="constructor"
                ),
                "diagnostics": diagnostics,
            }
            _atomic_json(stage_generation / "manifest.json", manifest)
            _publish_directory(
                stage_generation,
                generation_dir,
                overwrite=request.overwrite,
            )
        except Exception as exc:
            failure_path = _preserve_failure(
                layout=layout,
                operation="generate",
                error=exc,
                log_path=log_path,
            )
            raise RuntimeError(
                f"{exc}\nFailure artifacts preserved at: "
                f"{failure_path.resolve()}"
            ) from exc
        finally:
            if workspace.exists():
                shutil.rmtree(workspace)

        final_graphs_path = layout.generated_graphs_path(generation_id)
        return GenerationArtifacts(
            run_dir=layout.run_dir,
            generation_dir=generation_dir,
            graphs_path=final_graphs_path,
            manifest_path=layout.generation_manifest_path(generation_id),
            num_requested=request.num_graphs,
            num_generated=request.num_graphs,
            graphs_sha256=_sha256(final_graphs_path),
            log_path=layout.generation_log_path(generation_id),
        )

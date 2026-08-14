#!/usr/bin/env python
"""Safely materialize and optionally execute the fixed research protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import string
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from grapher.rewiring_mlp.evaluation.studies import (
    DEFAULT_EVALUATION_SEEDS,
    aggregate_pipeline_diagnostics,
    aggregate_three_seed_results,
    generation_error_decomposition,
    paired_ablation_comparison,
    quality_cost_pareto_summary,
)

PROTOCOL_FORMAT = "grapher_research_protocol_v1"
MANIFEST_FORMAT = "grapher_research_protocol_manifest_v1"
REPORT_FORMAT = "grapher_research_protocol_report_v1"
RUN_CATEGORIES = (
    ("ablations", "ablation"),
    ("cost_sweeps", "cost_sweep"),
    ("external_baselines", "external_baseline"),
)
ALLOWED_TEMPLATE_FIELDS = frozenset(
    {"seed", "variant", "config", "output_dir", "run_id", "protocol_dir"}
)
SHELL_EXECUTABLES = frozenset(
    {"sh", "bash", "dash", "zsh", "fish", "cmd", "cmd.exe", "powershell", "pwsh"}
)
SHELL_COMMAND_FLAGS = frozenset({"-c", "/c", "--command", "-command"})


def external_baseline_adapter_placeholder(baseline_name: str) -> None:
    """Fail explicitly when a declared external baseline has no argv adapter."""

    raise NotImplementedError(
        f"External baseline {baseline_name!r} has no executable argv adapter. "
        "Declare a pinned executable and an argv list before including its metrics."
    )


external_baseline_without_adapter = external_baseline_adapter_placeholder


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (Path, date, datetime)):
        return str(value)
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return repr(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _slug(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    return normalized or "run"


def _expand_string(template: str, context: Mapping[str, Any], *, location: str) -> str:
    formatter = string.Formatter()
    fields = {
        field_name
        for _, field_name, _, _ in formatter.parse(template)
        if field_name is not None
    }
    unknown = fields - ALLOWED_TEMPLATE_FIELDS
    if unknown:
        raise ValueError(
            f"{location} uses unsupported template field(s) {sorted(unknown)}; "
            f"allowed fields are {sorted(ALLOWED_TEMPLATE_FIELDS)}."
        )
    try:
        return template.format_map(dict(context))
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Could not expand {location}: {template!r}.") from exc


def _expand_value(value: Any, context: Mapping[str, Any], *, location: str) -> Any:
    if isinstance(value, str):
        return _expand_string(value, context, location=location)
    if isinstance(value, Mapping):
        return {
            str(key): _expand_value(item, context, location=f"{location}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _expand_value(item, context, location=f"{location}[{index}]")
            for index, item in enumerate(value)
        ]
    return value


def _resolve_path(raw: str | Path, *, base: Path) -> Path:
    path = Path(raw).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_protocol(path: str | Path) -> dict[str, Any]:
    """Load and validate the top-level protocol structure."""

    protocol_path = Path(path).resolve()
    with protocol_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        raise TypeError("A research protocol must be a YAML mapping.")
    protocol = dict(raw)
    declared_format = protocol.get("format", PROTOCOL_FORMAT)
    if declared_format != PROTOCOL_FORMAT:
        raise ValueError(
            f"Unsupported protocol format {declared_format!r}; expected {PROTOCOL_FORMAT!r}."
        )
    seeds = tuple(int(seed) for seed in protocol.get("seeds", DEFAULT_EVALUATION_SEEDS))
    if seeds != DEFAULT_EVALUATION_SEEDS:
        raise ValueError(
            f"The fixed protocol requires seeds {DEFAULT_EVALUATION_SEEDS} in that "
            f"order, got {seeds}."
        )
    protocol["format"] = PROTOCOL_FORMAT
    protocol["seeds"] = list(seeds)
    for plural, _ in RUN_CATEGORIES:
        entries = protocol.get(plural, [])
        if entries is None:
            entries = []
        if not isinstance(entries, list) or any(
            not isinstance(entry, Mapping) for entry in entries
        ):
            raise TypeError(f"protocol.{plural} must be a YAML list of mappings.")
        protocol[plural] = [dict(entry) for entry in entries]
    if not any(protocol[plural] for plural, _ in RUN_CATEGORIES):
        raise ValueError(
            "The protocol must declare at least one ablation, cost sweep, or "
            "external baseline."
        )
    return protocol


def _validate_argv(raw: Any, *, location: str) -> list[str]:
    if isinstance(raw, str):
        raise TypeError(
            f"{location}.argv must be a YAML list of tokens, never a shell string."
        )
    if not isinstance(raw, list) or not raw:
        raise TypeError(f"{location}.argv must be a non-empty YAML list of tokens.")
    if any(not isinstance(token, str) for token in raw):
        raise TypeError(f"Every token in {location}.argv must be a string.")
    executable = Path(raw[0]).name.lower()
    if executable in SHELL_EXECUTABLES and any(
        token.lower() in SHELL_COMMAND_FLAGS for token in raw[1:]
    ):
        raise ValueError(
            f"{location}.argv uses a shell command-string adapter; provide a direct "
            "executable argv adapter instead."
        )
    return list(raw)


def _capture_config(
    raw: Any,
    *,
    context: dict[str, Any],
    cwd: Path,
    output_dir: Path,
    run_id: str,
    location: str,
) -> dict[str, Any] | None:
    if raw is None:
        context["config"] = ""
        return None
    if isinstance(raw, Mapping):
        expanded = _expand_value(dict(raw), context, location=f"{location}.config")
        materialized_path = output_dir / "configs" / f"{_slug(run_id)}.yaml"
        materialized_path.parent.mkdir(parents=True, exist_ok=True)
        encoded = yaml.safe_dump(expanded, sort_keys=True).encode("utf-8")
        materialized_path.write_bytes(encoded)
        context["config"] = str(materialized_path.resolve())
        return {
            "source": "inline",
            "path": str(materialized_path.resolve()),
            "exists": True,
            "sha256": _sha256_bytes(encoded),
            "content": _jsonable(expanded),
        }
    if not isinstance(raw, str):
        raise TypeError(f"{location}.config must be a path string or YAML mapping.")
    expanded_path = _expand_string(raw, context, location=f"{location}.config")
    path = _resolve_path(expanded_path, base=cwd)
    context["config"] = str(path)
    if not path.is_file():
        return {
            "source": "file",
            "path": str(path),
            "exists": False,
            "sha256": None,
            "content": None,
        }
    encoded = path.read_bytes()
    try:
        if path.suffix.lower() == ".json":
            parsed = json.loads(encoded.decode("utf-8"))
        else:
            parsed = yaml.safe_load(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, yaml.YAMLError):
        parsed = None
    return {
        "source": "file",
        "path": str(path),
        "exists": True,
        "sha256": _sha256_bytes(encoded),
        "content": _jsonable(parsed),
    }


def _report_paths(
    entry: Mapping[str, Any],
    *,
    context: Mapping[str, Any],
    cwd: Path,
    location: str,
) -> dict[str, str]:
    if "report" in entry and "reports" in entry:
        raise ValueError(f"{location} cannot define both report and reports.")
    if "report" in entry:
        raw_reports: Mapping[str, Any] = {"default": entry["report"]}
    else:
        value = entry.get("reports", {})
        if not isinstance(value, Mapping):
            raise TypeError(f"{location}.reports must be a name-to-path mapping.")
        raw_reports = value
    result: dict[str, str] = {}
    for name, raw_path in raw_reports.items():
        if not isinstance(raw_path, str):
            raise TypeError(f"{location}.reports.{name} must be a path string.")
        expanded = _expand_string(
            raw_path, context, location=f"{location}.reports.{name}"
        )
        result[str(name)] = str(_resolve_path(expanded, base=cwd))
    return result


def materialize_protocol(
    protocol: Mapping[str, Any],
    *,
    protocol_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Expand all declared variants over exactly seeds 42, 43, and 44."""

    seeds = tuple(int(seed) for seed in protocol.get("seeds", ()))
    if seeds != DEFAULT_EVALUATION_SEEDS:
        raise ValueError(
            f"The fixed protocol requires seeds {DEFAULT_EVALUATION_SEEDS}, got {seeds}."
        )
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    protocol_file = Path(protocol_path).resolve()
    protocol_dir = protocol_file.parent
    default_cwd_raw = protocol.get("cwd", str(Path.cwd()))
    if not isinstance(default_cwd_raw, str):
        raise TypeError("protocol.cwd must be a path string.")
    variants_seen: set[tuple[str, str]] = set()
    runs: list[dict[str, Any]] = []
    for plural, category in RUN_CATEGORIES:
        for entry_index, raw_entry in enumerate(protocol.get(plural, [])):
            entry = dict(raw_entry)
            location = f"protocol.{plural}[{entry_index}]"
            variant_raw = entry.get("variant", entry.get("name"))
            if not isinstance(variant_raw, str) or not variant_raw.strip():
                raise ValueError(f"{location} must define a non-empty name or variant.")
            variant = variant_raw.strip()
            identity = (category, variant)
            if identity in variants_seen:
                raise ValueError(f"Duplicate {category} variant {variant!r}.")
            variants_seen.add(identity)
            if "shell" in entry:
                raise ValueError(
                    f"{location}.shell is forbidden; commands always use shell=False."
                )
            adapter_missing = category == "external_baseline" and "argv" not in entry
            argv_template = (
                None
                if adapter_missing
                else _validate_argv(entry.get("argv"), location=location)
            )
            for seed in seeds:
                run_id = f"{category}.{variant}.seed_{seed}"
                context: dict[str, Any] = {
                    "seed": seed,
                    "variant": variant,
                    "output_dir": str(destination),
                    "run_id": run_id,
                    "protocol_dir": str(protocol_dir),
                    "config": "",
                }
                cwd_template = entry.get("cwd", default_cwd_raw)
                if not isinstance(cwd_template, str):
                    raise TypeError(f"{location}.cwd must be a path string.")
                cwd_expanded = _expand_string(
                    cwd_template, context, location=f"{location}.cwd"
                )
                cwd = _resolve_path(cwd_expanded, base=Path.cwd())
                config_capture = _capture_config(
                    entry.get("config", entry.get("config_path")),
                    context=context,
                    cwd=cwd,
                    output_dir=destination,
                    run_id=run_id,
                    location=location,
                )
                argv = (
                    None
                    if argv_template is None
                    else [
                        _expand_string(
                            token,
                            context,
                            location=f"{location}.argv[{token_index}]",
                        )
                        for token_index, token in enumerate(argv_template)
                    ]
                )
                raw_environment = entry.get("env", {})
                if not isinstance(raw_environment, Mapping) or any(
                    not isinstance(key, str) or not isinstance(value, str)
                    for key, value in raw_environment.items()
                ):
                    raise TypeError(f"{location}.env must map strings to strings.")
                environment = _expand_value(
                    dict(raw_environment), context, location=f"{location}.env"
                )
                timeout = entry.get("timeout_seconds")
                if timeout is not None and (
                    isinstance(timeout, bool) or float(timeout) <= 0
                ):
                    raise ValueError(f"{location}.timeout_seconds must be positive.")
                reports = _report_paths(
                    entry,
                    context=context,
                    cwd=cwd,
                    location=location,
                )
                log_stem = _slug(run_id)
                parameters = _expand_value(
                    entry.get("parameters", {}),
                    context,
                    location=f"{location}.parameters",
                )
                run = {
                    "run_id": run_id,
                    "category": category,
                    "variant": variant,
                    "seed": seed,
                    "argv": argv,
                    "command_display": shlex.join(argv) if argv is not None else None,
                    "adapter_missing": adapter_missing,
                    "cwd": str(cwd),
                    "environment_overrides": environment,
                    "timeout_seconds": float(timeout) if timeout is not None else None,
                    "config": config_capture,
                    "reports": reports,
                    "primary_report": entry.get("primary_report"),
                    "selectors": {
                        key: entry.get(key)
                        for key in (
                            "metrics_path",
                            "pipeline_path",
                            "ablation_records_path",
                            "stage_metrics_path",
                        )
                        if entry.get(key) is not None
                    },
                    "parameters": parameters,
                    "baseline": bool(entry.get("baseline", False)),
                    "stdout_log": str(
                        (destination / "logs" / f"{log_stem}.stdout.log").resolve()
                    ),
                    "stderr_log": str(
                        (destination / "logs" / f"{log_stem}.stderr.log").resolve()
                    ),
                    "failure_log": None,
                    "status": "materialized",
                    "return_code": None,
                    "runtime_seconds": 0.0,
                    "started_at": None,
                    "finished_at": None,
                    "report_load": {},
                }
                runs.append(run)
    return {
        "format": MANIFEST_FORMAT,
        "protocol_format": protocol.get("format", PROTOCOL_FORMAT),
        "protocol_name": str(protocol.get("name", protocol_file.stem)),
        "protocol_path": str(protocol_file),
        "protocol_sha256": _sha256_bytes(protocol_file.read_bytes()),
        "output_dir": str(destination),
        "seeds": list(seeds),
        "created_at": _utc_now(),
        "mode": None,
        "runs": runs,
        "status_counts": {},
    }


def _write_failure(run: dict[str, Any], message: str, output_dir: Path) -> None:
    path = output_dir / "failures" / f"{_slug(str(run['run_id']))}.failure.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(message.rstrip() + "\n", encoding="utf-8")
    run["failure_log"] = str(path.resolve())


def _execute_or_plan_run(
    run: dict[str, Any], *, execute: bool, output_dir: Path
) -> None:
    if run["adapter_missing"]:
        try:
            external_baseline_adapter_placeholder(str(run["variant"]))
        except NotImplementedError as exc:
            run["status"] = "not_implemented"
            _write_failure(run, str(exc), output_dir)
        return
    if not execute:
        run["status"] = "planned"
        return

    stdout_path = Path(str(run["stdout_log"]))
    stderr_path = Path(str(run["stderr_log"]))
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(dict(run["environment_overrides"]))
    run["started_at"] = _utc_now()
    started = time.perf_counter()
    try:
        with (
            stdout_path.open("w", encoding="utf-8") as stdout_handle,
            stderr_path.open("w", encoding="utf-8") as stderr_handle,
        ):
            completed = subprocess.run(
                list(run["argv"]),
                cwd=str(run["cwd"]),
                env=environment,
                stdout=stdout_handle,
                stderr=stderr_handle,
                timeout=run["timeout_seconds"],
                check=False,
                shell=False,
            )
        run["return_code"] = int(completed.returncode)
        if completed.returncode == 0:
            run["status"] = "completed"
        else:
            run["status"] = "failed"
            _write_failure(
                run,
                f"Command returned {completed.returncode}.\n"
                f"argv: {run['argv']!r}\n"
                f"stdout: {stdout_path}\nstderr: {stderr_path}",
                output_dir,
            )
    except subprocess.TimeoutExpired as exc:
        run["return_code"] = 124
        run["status"] = "timed_out"
        _write_failure(
            run,
            f"Command exceeded timeout_seconds={run['timeout_seconds']}: {exc}",
            output_dir,
        )
    except OSError as exc:
        run["return_code"] = 127
        run["status"] = "failed"
        _write_failure(
            run, f"Could not execute argv {run['argv']!r}: {exc}", output_dir
        )
    finally:
        run["runtime_seconds"] = float(time.perf_counter() - started)
        run["finished_at"] = _utc_now()


def _load_reports(
    run: dict[str, Any], *, execute: bool, output_dir: Path
) -> dict[str, Any]:
    loaded: dict[str, Any] = {}
    if execute and run["status"] == "completed" and not run["reports"]:
        run["status"] = "missing_report"
        _write_failure(
            run,
            "Command completed but the protocol declares no JSON report path; "
            "research runs must expose an aggregatable report.",
            output_dir,
        )
        return loaded
    for name, raw_path in dict(run["reports"]).items():
        path = Path(raw_path)
        if not path.is_file():
            run["report_load"][name] = {"status": "missing", "path": str(path)}
            if execute and run["status"] == "completed":
                run["status"] = "missing_report"
                _write_failure(
                    run,
                    f"Command completed but declared JSON report is missing: {path}",
                    output_dir,
                )
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            run["report_load"][name] = {
                "status": "invalid",
                "path": str(path),
                "error": str(exc),
            }
            if execute and run["status"] == "completed":
                run["status"] = "invalid_report"
                _write_failure(run, f"Invalid JSON report {path}: {exc}", output_dir)
            continue
        if not isinstance(value, Mapping):
            error = "JSON report root must be an object."
            run["report_load"][name] = {
                "status": "invalid",
                "path": str(path),
                "error": error,
            }
            if execute and run["status"] == "completed":
                run["status"] = "invalid_report"
                _write_failure(run, f"Invalid JSON report {path}: {error}", output_dir)
            continue
        loaded[name] = dict(value)
        run["report_load"][name] = {
            "status": "loaded",
            "path": str(path),
            "sha256": _sha256_bytes(path.read_bytes()),
        }
    return loaded


def _select_primary_report(
    run: Mapping[str, Any], loaded: Mapping[str, Any]
) -> Mapping[str, Any] | None:
    if not loaded:
        return None
    selected = run.get("primary_report")
    if selected is None:
        selected = next(iter(loaded))
    if selected not in loaded:
        raise ValueError(
            f"Run {run['run_id']} selected unknown/unloaded primary_report {selected!r}."
        )
    value = loaded[str(selected)]
    if not isinstance(value, Mapping):
        raise TypeError("The selected primary report must be an object.")
    return value


def _resolve_json_path(value: Any, path: str) -> Any:
    current = value
    for component in path.split("."):
        if isinstance(current, Mapping) and component in current:
            current = current[component]
        elif isinstance(current, list) and component.isdigit():
            current = current[int(component)]
        else:
            raise KeyError(f"JSON path {path!r} is missing component {component!r}.")
    return current


def _selected_value(
    report: Mapping[str, Any], selector: str | None, defaults: Sequence[str]
) -> Any | None:
    if selector is not None:
        return _resolve_json_path(report, selector)
    for path in defaults:
        try:
            return _resolve_json_path(report, path)
        except (KeyError, IndexError):
            continue
    return None


def _numeric_leaf_metrics(value: Any, *, prefix: str = "") -> dict[str, float]:
    result: dict[str, float] = {}
    if isinstance(value, Mapping):
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            result.update(_numeric_leaf_metrics(item, prefix=name))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            label = str(index)
            if isinstance(item, Mapping):
                for identity_key in ("comparison", "stage", "name"):
                    if identity_key in item:
                        label = _slug(str(item[identity_key]))
                        break
            name = f"{prefix}.{label}" if prefix else label
            result.update(_numeric_leaf_metrics(item, prefix=name))
    elif isinstance(value, (int, float, np.integer, np.floating, bool, np.bool_)):
        number = float(value)
        if np.isfinite(number):
            result[prefix] = number
    return result


def _extract_metrics(
    run: Mapping[str, Any], report: Mapping[str, Any]
) -> dict[str, float]:
    selector = dict(run.get("selectors", {})).get("metrics_path")
    selected = _selected_value(
        report,
        selector,
        ("metrics", "hybrid_refined", "evaluation", "diagnostics"),
    )
    if selected is None:
        selected = report
    metrics = _numeric_leaf_metrics(selected)
    metrics.pop("seed", None)
    if not metrics:
        raise ValueError(
            f"Run {run['run_id']} report exposes no finite numeric metrics."
        )
    return metrics


def _extract_pipeline_records(
    run: Mapping[str, Any], report: Mapping[str, Any]
) -> list[Mapping[str, Any]]:
    selector = dict(run.get("selectors", {})).get("pipeline_path")
    selected = _selected_value(
        report,
        selector,
        ("pipeline_diagnostics", "diagnostics"),
    )
    if selected is None:
        return []
    if isinstance(selected, Mapping):
        return [selected]
    if isinstance(selected, list) and all(
        isinstance(item, Mapping) for item in selected
    ):
        return selected
    raise TypeError(
        f"Run {run['run_id']} pipeline diagnostics must be an object or list."
    )


def _extract_ablation_records(
    run: Mapping[str, Any], report: Mapping[str, Any]
) -> list[dict[str, Any]]:
    selector = dict(run.get("selectors", {})).get("ablation_records_path")
    selected = _selected_value(report, selector, ("ablation_records", "samples"))
    if selected is None:
        return []
    if not isinstance(selected, list) or any(
        not isinstance(item, Mapping) for item in selected
    ):
        raise TypeError(
            f"Run {run['run_id']} ablation records must be a list of objects."
        )
    records = []
    for index, item in enumerate(selected):
        record = dict(item)
        original_id = record.get("sample_id", index)
        record["sample_id"] = f"{run['seed']}:{original_id}"
        record.setdefault("seed", int(run["seed"]))
        records.append(record)
    return records


def _extract_stage_decomposition(
    run: Mapping[str, Any],
    report: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any] | None:
    selector = dict(run.get("selectors", {})).get("stage_metrics_path")
    selected = _selected_value(report, selector, ("stage_metrics",))
    if selected is None:
        return None
    if not isinstance(selected, Mapping):
        raise TypeError(f"Run {run['run_id']} stage_metrics must be an object.")
    directions = dict(protocol.get("metric_higher_is_better", {}) or {})
    return generation_error_decomposition(
        selected,
        metric_higher_is_better=directions,
    )


def aggregate_protocol_reports(
    protocol: Mapping[str, Any],
    manifest: Mapping[str, Any],
    loaded_reports: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate matched reports with the research-study APIs."""

    errors: list[dict[str, str]] = []
    by_variant: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for run in manifest["runs"]:
        by_variant.setdefault((str(run["category"]), str(run["variant"])), []).append(
            run
        )

    variant_aggregates: dict[str, Any] = {}
    ablation_records: dict[str, list[dict[str, Any]]] = {}
    stage_decompositions: dict[str, Any] = {}
    for (category, variant), runs in by_variant.items():
        key = f"{category}.{variant}"
        seed_metrics: dict[int, dict[str, float]] = {}
        pipeline_records: list[Mapping[str, Any]] = []
        variant_ablation_records: list[dict[str, Any]] = []
        per_seed_stages: dict[str, Any] = {}
        for run in runs:
            report = loaded_reports.get(str(run["run_id"]))
            if report is None:
                continue
            try:
                seed_metrics[int(run["seed"])] = _extract_metrics(run, report)
                pipeline_records.extend(_extract_pipeline_records(run, report))
                variant_ablation_records.extend(_extract_ablation_records(run, report))
                decomposition = _extract_stage_decomposition(run, report, protocol)
                if decomposition is not None:
                    per_seed_stages[str(run["seed"])] = decomposition
            except (KeyError, TypeError, ValueError) as exc:
                errors.append({"scope": str(run["run_id"]), "error": str(exc)})
        variant_result: dict[str, Any] = {
            "category": category,
            "variant": variant,
            "available_seeds": sorted(seed_metrics),
            "parameters": _jsonable(runs[0].get("parameters", {})),
            "metrics": None,
            "pipeline_diagnostics": None,
        }
        if set(seed_metrics) == set(DEFAULT_EVALUATION_SEEDS):
            common_metrics = set.intersection(
                *(set(values) for values in seed_metrics.values())
            )
            all_metrics = set.union(*(set(values) for values in seed_metrics.values()))
            excluded = sorted(all_metrics - common_metrics)
            if common_metrics:
                try:
                    variant_result["metrics"] = aggregate_three_seed_results(
                        {
                            seed: {metric: values[metric] for metric in common_metrics}
                            for seed, values in seed_metrics.items()
                        }
                    )
                    variant_result["excluded_noncommon_metrics"] = excluded
                except (TypeError, ValueError) as exc:
                    errors.append({"scope": key, "error": str(exc)})
            else:
                errors.append(
                    {"scope": key, "error": "No finite metric is common to all seeds."}
                )
        elif seed_metrics:
            errors.append(
                {
                    "scope": key,
                    "error": "Reports are required for exactly seeds 42, 43, and 44; "
                    f"available={sorted(seed_metrics)}.",
                }
            )
        if pipeline_records:
            try:
                variant_result["pipeline_diagnostics"] = aggregate_pipeline_diagnostics(
                    pipeline_records,
                    require_complete=False,
                    allow_fallback=False,
                )
            except (TypeError, ValueError) as exc:
                errors.append({"scope": f"{key}.pipeline", "error": str(exc)})
        if variant_ablation_records:
            ablation_records[variant] = variant_ablation_records
        if per_seed_stages:
            stage_decompositions[key] = per_seed_stages
        variant_aggregates[key] = variant_result

    ablation_comparisons: dict[str, Any] = {}
    ablation_config = dict(protocol.get("ablation_comparison", {}) or {})
    baseline_candidates = [
        str(run["variant"])
        for run in manifest["runs"]
        if run["category"] == "ablation" and bool(run.get("baseline"))
    ]
    baseline_variant = ablation_config.get("baseline_variant")
    if baseline_variant is None and baseline_candidates:
        baseline_variant = baseline_candidates[0]
    if len(set(baseline_candidates)) > 1:
        errors.append(
            {
                "scope": "ablation_comparison",
                "error": f"Multiple baseline variants declared: {sorted(set(baseline_candidates))}.",
            }
        )
    if baseline_variant is not None and str(baseline_variant) in ablation_records:
        for variant, records in ablation_records.items():
            if variant == str(baseline_variant):
                continue
            try:
                ablation_comparisons[variant] = paired_ablation_comparison(
                    ablation_records[str(baseline_variant)],
                    records,
                    metrics=ablation_config.get("metrics"),
                    metric_higher_is_better=ablation_config.get(
                        "metric_higher_is_better"
                    ),
                )
            except (TypeError, ValueError) as exc:
                errors.append(
                    {"scope": f"ablation_comparison.{variant}", "error": str(exc)}
                )

    quality_cost_result = None
    quality_cost = protocol.get("quality_cost")
    if quality_cost is not None:
        if not isinstance(quality_cost, Mapping):
            errors.append(
                {"scope": "quality_cost", "error": "quality_cost must be a mapping."}
            )
        else:
            pareto_records = []
            for key, aggregate in variant_aggregates.items():
                if (
                    aggregate["category"] != "cost_sweep"
                    or aggregate["metrics"] is None
                ):
                    continue
                record: dict[str, Any] = {"name": aggregate["variant"]}
                record.update(
                    {
                        metric: summary["mean"]
                        for metric, summary in aggregate["metrics"]["aggregate"].items()
                    }
                )
                record.update(_numeric_leaf_metrics(aggregate.get("parameters", {})))
                pareto_records.append(record)
            if pareto_records:
                try:
                    quality_cost_result = quality_cost_pareto_summary(
                        pareto_records,
                        quality_keys=quality_cost["quality_keys"],
                        cost_keys=quality_cost["cost_keys"],
                        higher_is_better=quality_cost.get("higher_is_better"),
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    errors.append({"scope": "quality_cost", "error": str(exc)})
            else:
                errors.append(
                    {
                        "scope": "quality_cost",
                        "error": "No complete cost-sweep reports.",
                    }
                )

    failed_statuses = {
        "failed",
        "timed_out",
        "not_implemented",
        "missing_report",
        "invalid_report",
    }
    failures = [
        {
            "run_id": run["run_id"],
            "status": run["status"],
            "return_code": run["return_code"],
            "failure_log": run["failure_log"],
        }
        for run in manifest["runs"]
        if run["status"] in failed_statuses
    ]
    return {
        "format": REPORT_FORMAT,
        "protocol_name": manifest["protocol_name"],
        "mode": manifest["mode"],
        "seeds": list(DEFAULT_EVALUATION_SEEDS),
        "variant_aggregates": variant_aggregates,
        "stage_decompositions": stage_decompositions,
        "ablation_comparisons": ablation_comparisons,
        "quality_cost": quality_cost_result,
        "failures": failures,
        "aggregation_errors": errors,
        "success": not failures and not errors,
        "created_at": _utc_now(),
    }


def run_protocol(
    protocol_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    execute: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Materialize a protocol, optionally execute it, and write both artifacts."""

    protocol_file = Path(protocol_path).resolve()
    protocol = load_protocol(protocol_file)
    if output_dir is None:
        raw_destination = protocol.get(
            "output_dir", f"outputs/research_protocol/{protocol_file.stem}"
        )
        if not isinstance(raw_destination, str):
            raise TypeError("protocol.output_dir must be a path string.")
        destination = _resolve_path(raw_destination, base=Path.cwd())
    else:
        destination = Path(output_dir).resolve()
    manifest = materialize_protocol(
        protocol,
        protocol_path=protocol_file,
        output_dir=destination,
    )
    manifest["mode"] = "execute" if execute else "dry_run"
    loaded_by_run: dict[str, Mapping[str, Any]] = {}
    for run in manifest["runs"]:
        _execute_or_plan_run(run, execute=execute, output_dir=destination)
        loaded = _load_reports(run, execute=execute, output_dir=destination)
        try:
            primary = _select_primary_report(run, loaded)
        except (TypeError, ValueError) as exc:
            run["status"] = "invalid_report"
            _write_failure(run, str(exc), destination)
            primary = None
        if primary is not None:
            loaded_by_run[str(run["run_id"])] = primary
    manifest["status_counts"] = dict(
        sorted(Counter(str(run["status"]) for run in manifest["runs"]).items())
    )
    report = aggregate_protocol_reports(protocol, manifest, loaded_by_run)
    _write_json(destination / "manifest.json", manifest)
    _write_json(destination / "report.json", report)
    return manifest, report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize the fixed three-seed ablation/cost protocol. The default "
            "is a safe dry run; pass --execute to launch argv adapters."
        )
    )
    parser.add_argument("--protocol", required=True, help="YAML research protocol.")
    parser.add_argument("--output-dir", default=None)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Materialize commands without executing them (the default).",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help="Explicitly execute every direct argv adapter with shell=False.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest, report = run_protocol(
            args.protocol,
            output_dir=args.output_dir,
            execute=bool(args.execute),
        )
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        print(f"Protocol error: {exc}", file=sys.stderr)
        return 2
    destination = Path(manifest["output_dir"])
    print(f"Manifest: {destination / 'manifest.json'}")
    print(f"Report: {destination / 'report.json'}")
    print(f"Mode: {manifest['mode']}; statuses: {manifest['status_counts']}")
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

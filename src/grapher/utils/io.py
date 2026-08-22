from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import yaml


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def apply_config_overrides(
    config: dict[str, Any],
    overrides: list[str] | tuple[str, ...] | None,
) -> dict[str, Any]:
    """Apply repeatable ``dot.path=value`` YAML overrides in place.

    Values are parsed with ``yaml.safe_load`` so command-line strings such as
    ``true``, ``null``, ``[1, 2]``, and numeric literals keep their YAML types.
    Missing nested mappings are created, while attempting to descend through a
    scalar value is treated as a configuration error.
    """

    for expression in overrides or ():
        if "=" not in expression:
            raise ValueError(
                "Config overrides must use KEY=VALUE syntax; "
                f"received {expression!r}."
            )
        raw_key, raw_value = expression.split("=", 1)
        key = raw_key.strip()
        if not key:
            raise ValueError("Config override key cannot be empty.")
        parts = [part.strip() for part in key.split(".")]
        if any(not part for part in parts):
            raise ValueError(
                f"Invalid config override path {key!r}: empty path component."
            )
        try:
            value = yaml.safe_load(raw_value)
        except yaml.YAMLError as exc:
            raise ValueError(
                f"Invalid YAML value in config override {expression!r}."
            ) from exc

        cursor: dict[str, Any] = config
        for index, part in enumerate(parts[:-1]):
            existing = cursor.get(part)
            if existing is None:
                child: dict[str, Any] = {}
                cursor[part] = child
                cursor = child
            elif isinstance(existing, dict):
                cursor = existing
            else:
                prefix = ".".join(parts[: index + 1])
                raise ValueError(
                    f"Cannot apply override {expression!r}: {prefix!r} "
                    "is not a mapping."
                )
        cursor[parts[-1]] = value
    return config


def require_config(
    mapping: dict[str, Any], key: str, *, context: str = "config"
) -> Any:
    if key not in mapping:
        raise KeyError(f"Missing required config parameter: {context}.{key}")
    return mapping[key]


def require_config_section(
    mapping: dict[str, Any], key: str, *, context: str = "config"
) -> dict[str, Any]:
    value = require_config(mapping, key, context=context)
    if not isinstance(value, dict):
        raise TypeError(f"Config parameter {context}.{key} must be a mapping.")
    return value


def save_yaml(obj: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def save_pickle(obj: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as f:
        return pickle.load(f)

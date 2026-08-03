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

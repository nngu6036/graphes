from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def save_yaml(obj: dict[str, Any], path: str | Path, *, force: bool = True) -> None:
    path = Path(path)
    if path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing YAML: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def save_json(obj: dict[str, Any], path: str | Path, *, force: bool = True) -> None:
    path = Path(path)
    if path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing JSON: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_pickle(obj: Any, path: str | Path, *, force: bool = True) -> None:
    path = Path(path)
    if path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing pickle: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str | Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def stable_hash(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]

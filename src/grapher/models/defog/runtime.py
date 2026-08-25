"""Dependency-light runtime resolution for isolated DeFoG processes."""

from __future__ import annotations

import os
import re
import shutil
import sys
from pathlib import Path

DEFOG_ROOT_ENV = "DEFOG"
DEFOG_PYTHON_ENV = "DEFOG_PYTHON"
_SAFE_ENV_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def resolve_defog_root(source_env: str = DEFOG_ROOT_ENV) -> Path:
    if _SAFE_ENV_KEY.fullmatch(source_env) is None:
        raise ValueError("source_env must be a valid environment-variable name.")
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
    *,
    defog_root: Path,
    python_executable: str | None = None,
    python_env: str = DEFOG_PYTHON_ENV,
) -> str:
    if _SAFE_ENV_KEY.fullmatch(python_env) is None:
        raise ValueError("python_env must be a valid environment-variable name.")
    candidate = python_executable or os.environ.get(python_env)
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

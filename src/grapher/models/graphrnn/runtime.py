"""Runtime discovery for the isolated GraphRNN wrapper."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

GRAPHRNN_ROOT_ENV = "GRAPHRNN"
GRAPHRNN_PYTHON_ENV = "GRAPHRNN_PYTHON"


def resolve_graphrnn_root(source_env: str = GRAPHRNN_ROOT_ENV) -> Path:
    """Resolve and validate the attached GraphRNN source checkout."""

    raw = os.environ.get(str(source_env))
    if not raw:
        raise RuntimeError(
            f"Set {source_env} to the attached GraphRNN source root, or pass "
            "--graphrnn-root to scripts/run_graphrnn_baseline.py."
        )
    root = Path(raw).expanduser().resolve()
    required = (
        root / "model.py",
        root / "data.py",
        root / "args.py",
        root / "README.md",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Invalid GraphRNN source root {root}; missing: {missing}."
        )
    return root


def resolve_graphrnn_python(
    *,
    graphrnn_root: Path,
    python_executable: str | Path | None = None,
    python_env: str = GRAPHRNN_PYTHON_ENV,
) -> Path:
    """Resolve the Python interpreter used by GraphRNN worker processes."""

    raw = python_executable or os.environ.get(str(python_env)) or sys.executable
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        resolved = shutil.which(str(path))
        if resolved is None:
            raise FileNotFoundError(f"Could not resolve GraphRNN Python: {raw}")
        path = Path(resolved)
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing GraphRNN Python executable: {path}")
    del graphrnn_root  # retained for symmetry with the other external wrappers
    return path

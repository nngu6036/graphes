"""Runtime discovery for the isolated DiGress wrapper."""

from __future__ import annotations

import os
import sys
from pathlib import Path

DIGRESS_ROOT_ENV = "DIGRESS"
DIGRESS_PYTHON_ENV = "DIGRESS_PYTHON"


def resolve_digress_root(source_env: str = DIGRESS_ROOT_ENV) -> Path:
    """Resolve and validate the external DiGress source checkout."""

    raw = os.environ.get(str(source_env))
    if not raw:
        raise RuntimeError(
            f"Set {source_env} to the attached DiGress source root, or pass "
            "--digress-root to run_digress_baseline.py."
        )
    root = Path(raw).expanduser().resolve()
    required = (
        root / "src" / "diffusion_model_discrete.py",
        root / "src" / "datasets" / "spectre_dataset.py",
        root / "configs" / "config.yaml",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Invalid DiGress source root {root}; missing: {missing}."
        )
    return root


def resolve_digress_python(
    *,
    digress_root: Path,
    python_executable: str | Path | None = None,
    python_env: str = DIGRESS_PYTHON_ENV,
) -> Path:
    """Resolve the Python interpreter used for the external DiGress process."""

    raw = python_executable or os.environ.get(str(python_env)) or sys.executable
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        import shutil

        resolved = shutil.which(str(path))
        if resolved is None:
            raise FileNotFoundError(f"Could not resolve DiGress Python: {raw}")
        path = Path(resolved)
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing DiGress Python executable: {path}")
    del digress_root  # retained in the signature for symmetry with other wrappers
    return path

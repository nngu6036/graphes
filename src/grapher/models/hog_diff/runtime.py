"""Runtime discovery for the isolated HOG-Diff baseline wrapper."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

HOGDIFF_ROOT_ENV = "HOGDIFF"
HOGDIFF_PYTHON_ENV = "HOGDIFF_PYTHON"

_REQUIRED_FILES = (
    "main.py",
    "trainer.py",
    "sampler.py",
    "models/ScoreNet.py",
    "utils/dataloader.py",
    "utils/file_utils.py",
    "utils/solver.py",
    "configs/cs.yaml",
    "configs/ego.yaml",
    "configs/qm9.yaml",
    "configs/zinc250k.yaml",
)


def resolve_hogdiff_root(source_env: str = HOGDIFF_ROOT_ENV) -> Path:
    """Resolve and validate the attached HOG-Diff checkout.

    The release supplied with GraphER does not contain ``data.py`` even though
    the upstream modules import dataset-family constants from it.  The wrapper
    supplies a tiny compatibility shim in its worker directory, so ``data.py``
    is deliberately not part of the source-root validation contract.
    """

    raw = os.environ.get(str(source_env))
    if not raw:
        raise RuntimeError(
            f"Set {source_env} to the HOG-Diff source root, or pass "
            "--hogdiff-root to scripts/run_hog_diff_baseline.py."
        )
    root = Path(raw).expanduser().resolve()
    missing = [str(root / relative) for relative in _REQUIRED_FILES if not (root / relative).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Invalid HOG-Diff source root {root}; missing: {missing}."
        )
    return root


def resolve_hogdiff_python(
    *,
    hogdiff_root: Path,
    python_executable: str | Path | None = None,
    python_env: str = HOGDIFF_PYTHON_ENV,
) -> Path:
    """Resolve the Python interpreter used by HOG-Diff subprocess workers."""

    raw = python_executable or os.environ.get(str(python_env))
    if raw is None:
        local = hogdiff_root / ".venv" / "bin" / "python"
        raw = local if local.is_file() else sys.executable
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        resolved = shutil.which(str(path))
        if resolved is None:
            raise FileNotFoundError(f"Could not resolve HOG-Diff Python: {raw}")
        path = Path(resolved)
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing HOG-Diff Python executable: {path}")
    return path

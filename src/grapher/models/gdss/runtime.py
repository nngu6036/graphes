"""Runtime discovery for the isolated GDSS baseline wrapper."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

GDSS_ROOT_ENV = "GDSS"
GDSS_PYTHON_ENV = "GDSS_PYTHON"

_REQUIRED = (
    "main.py",
    "trainer.py",
    "sampler.py",
    "losses.py",
    "sde.py",
    "solver.py",
    "models/ScoreNetwork_A.py",
    "models/ScoreNetwork_X.py",
    "utils/loader.py",
    "utils/graph_utils.py",
    "utils/ema.py",
    "config/community_small.yaml",
    "config/ego_small.yaml",
    "config/grid.yaml",
    "config/qm9.yaml",
    "config/zinc250k.yaml",
    "config/sample_qm9.yaml",
    "config/sample_zinc250k.yaml",
)


def resolve_gdss_root(source_env: str = GDSS_ROOT_ENV) -> Path:
    raw = os.environ.get(source_env)
    if not raw:
        raise FileNotFoundError(
            f"Set {source_env} to the GDSS source root, or pass --gdss-root "
            "to scripts/run_gdss_baseline.py."
        )
    root = Path(raw).expanduser().resolve()
    missing = [relative for relative in _REQUIRED if not (root / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"Invalid GDSS source root {root}; missing: {missing}.")
    return root


def resolve_gdss_python(
    *,
    gdss_root: Path,
    python_executable: str | Path | None = None,
    python_env: str = GDSS_PYTHON_ENV,
) -> Path:
    raw = python_executable or os.environ.get(python_env)
    if raw:
        candidate = str(raw)
        located = shutil.which(candidate)
        path = Path(located or candidate).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Could not resolve GDSS Python: {raw}")
        return path

    # Prefer a source-local virtual environment if one exists; otherwise use
    # the current interpreter.  Dependency probing in the wrapper gives a
    # precise error if this environment is incompatible with GDSS.
    candidates = (
        gdss_root / ".venv" / "bin" / "python",
        gdss_root / "venv" / "bin" / "python",
        gdss_root / ".venv" / "Scripts" / "python.exe",
        gdss_root / "venv" / "Scripts" / "python.exe",
    )
    for path in candidates:
        if path.is_file():
            return path.resolve()
    return Path(sys.executable).resolve()

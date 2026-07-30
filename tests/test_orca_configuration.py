from pathlib import Path

import pytest

from grapher.properties import summary


def test_configure_orca_executable_accepts_explicit_executable() -> None:
    executable = Path("/bin/true")
    if not executable.exists():
        pytest.skip("/bin/true is unavailable on this platform")

    previous = summary.ORCA_EXEC
    try:
        resolved = summary.configure_orca_executable(executable)
        assert resolved == str(executable.resolve())
        assert summary.ORCA_EXEC == resolved
    finally:
        summary.ORCA_EXEC = previous


def test_configure_orca_executable_rejects_missing_path(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match="ORCA evaluation is enabled"):
        summary.configure_orca_executable(tmp_path / "missing-orca")

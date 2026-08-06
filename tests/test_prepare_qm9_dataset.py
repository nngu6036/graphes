from __future__ import annotations

from pathlib import Path

import pytest

from scripts.prepare_qm9_dataset import (
    _clear_dataset_outputs,
    _dataset_output_paths,
)


def test_dataset_output_paths_require_distinct_direct_children(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be different"):
        _dataset_output_paths(tmp_path, ("qm9", "qm9"))

    for unsafe_name in ("", ".", "..", "nested/qm9"):
        with pytest.raises(ValueError):
            _dataset_output_paths(tmp_path, (unsafe_name, "qm9_attributed"))


def test_clear_dataset_outputs_removes_only_selected_directories(
    tmp_path: Path,
) -> None:
    topology_path, attributed_path = _dataset_output_paths(
        tmp_path,
        ("qm9_topology", "qm9_attributed"),
    )
    topology_path.mkdir()
    attributed_path.mkdir()
    (topology_path / "stale.pkl").write_text("old", encoding="utf-8")
    (attributed_path / "stale.json").write_text("old", encoding="utf-8")
    unrelated = tmp_path / "keep_me"
    unrelated.mkdir()

    removed = _clear_dataset_outputs((topology_path, attributed_path))

    assert removed == [topology_path, attributed_path]
    assert not topology_path.exists()
    assert not attributed_path.exists()
    assert unrelated.is_dir()

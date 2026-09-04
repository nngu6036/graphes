from __future__ import annotations

from pathlib import Path

import pytest

from grapher.models.hog_diff.runtime import resolve_hogdiff_root


def test_runtime_accepts_release_without_data_py(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    required = (
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
    for relative in required:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# stub\n", encoding="utf-8")
    assert not (tmp_path / "data.py").exists()
    monkeypatch.setenv("HOGDIFF_TEST", str(tmp_path))
    assert resolve_hogdiff_root("HOGDIFF_TEST") == tmp_path.resolve()

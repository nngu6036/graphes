from __future__ import annotations

import sys
from pathlib import Path

import pytest

from grapher.models.graphrnn.runtime import (
    resolve_graphrnn_python,
    resolve_graphrnn_root,
)


def _make_root(path: Path) -> Path:
    path.mkdir()
    for name in ("model.py", "data.py", "args.py", "README.md"):
        (path / name).write_text("# fixture\n", encoding="utf-8")
    return path


def test_runtime_resolves_root_and_current_python(tmp_path: Path, monkeypatch) -> None:
    root = _make_root(tmp_path / "GraphRNN")
    monkeypatch.setenv("TEST_GRAPHRNN", str(root))
    monkeypatch.delenv("TEST_GRAPHRNN_PYTHON", raising=False)
    assert resolve_graphrnn_root("TEST_GRAPHRNN") == root.resolve()
    assert resolve_graphrnn_python(
        graphrnn_root=root,
        python_env="TEST_GRAPHRNN_PYTHON",
    ) == Path(sys.executable).resolve()


def test_runtime_rejects_incomplete_checkout(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "GraphRNN"
    root.mkdir()
    (root / "model.py").write_text("# fixture\n", encoding="utf-8")
    monkeypatch.setenv("TEST_GRAPHRNN", str(root))
    with pytest.raises(FileNotFoundError, match="Invalid GraphRNN source root"):
        resolve_graphrnn_root("TEST_GRAPHRNN")

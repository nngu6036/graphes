from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from grapher.models.hog_diff.runtime import resolve_hogdiff_root
from grapher.models.hog_diff.workers._compat import install_torch_functional_alias
from grapher.models.hog_diff.workers.generate import (
    _is_predictor_nan_error,
    _numerical_retry_seed,
    _singleton_retry_seed,
)


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


def test_runtime_repairs_missing_layers_functional_alias_without_overwriting() -> None:
    layers = SimpleNamespace()
    functional = object()

    assert install_torch_functional_alias(layers, functional) is True
    assert layers.F is functional
    assert install_torch_functional_alias(layers, object()) is False
    assert layers.F is functional


def test_generation_numerical_retry_detection_and_seeds_are_narrow_and_deterministic() -> None:
    assert _is_predictor_nan_error(ValueError("NaNs in predictor output: tensors"))
    assert not _is_predictor_nan_error(ValueError("different upstream failure"))
    assert not _is_predictor_nan_error(RuntimeError("NaNs in predictor output"))
    assert _numerical_retry_seed(42, 0, 0) == 42
    assert _numerical_retry_seed(42, 2, 1) == 1_000_047
    assert _singleton_retry_seed(42, 2, 3, 1) == 1_002_030_173

from __future__ import annotations

from pathlib import Path

import pytest

from grapher.models.gdss.runtime import resolve_gdss_root


def test_runtime_accepts_attached_gdss_release_layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    required = (
        "main.py", "trainer.py", "sampler.py", "losses.py", "sde.py", "solver.py",
        "models/ScoreNetwork_A.py", "models/ScoreNetwork_X.py", "utils/loader.py",
        "utils/graph_utils.py", "utils/ema.py", "config/community_small.yaml",
        "config/ego_small.yaml", "config/grid.yaml", "config/qm9.yaml",
        "config/zinc250k.yaml", "config/sample_qm9.yaml", "config/sample_zinc250k.yaml",
    )
    for relative in required:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# stub\n", encoding="utf-8")
    # The supplied archive has data_generators.py at repository root and no
    # data/ package; GraphER does not depend on that broken native loader path.
    assert not (tmp_path / "data").exists()
    monkeypatch.setenv("GDSS_TEST", str(tmp_path))
    assert resolve_gdss_root("GDSS_TEST") == tmp_path.resolve()

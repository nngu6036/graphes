from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def test_registry_resolution_does_not_import_torch_or_upstream_graphrnn() -> None:
    code = """
import sys
from grapher.models.registry import get_wrapper_class
wrapper = get_wrapper_class('graphrnn')
assert wrapper.__name__ == 'GraphRNNWrapper'
assert 'torch' not in sys.modules
assert 'graphrnn_upstream_model_adapter' not in sys.modules
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(SRC)
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        cwd=ROOT,
        env=environment,
    )

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PACKAGE = SRC / "grapher" / "rewiring_mlp"


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_rewiring_components_are_colocated() -> None:
    assert {"generic", "attributed", "core", "molecular", "evaluation"} <= {
        path.name for path in PACKAGE.iterdir() if path.is_dir()
    }


def test_old_rewiring_packages_are_absent() -> None:
    for name in ("topology", "hybrid", "refinement", "molecular", "evaluation"):
        assert not (SRC / "grapher" / name).exists()


def test_python_code_has_no_retired_imports() -> None:
    retired = (
        "grapher.topology",
        "grapher.hybrid",
        "grapher.refinement",
        "grapher.molecular",
        "grapher.evaluation",
    )
    paths = list((SRC / "grapher").rglob("*.py"))
    paths.extend((ROOT / "scripts").glob("*.py"))
    paths.extend((ROOT / "tests").glob("*.py"))
    for path in paths:
        for module in _imported_modules(path):
            assert not module.startswith(retired), (path, module)


def test_package_root_is_lightweight() -> None:
    code = """
import sys
import grapher.rewiring_mlp
for name in (
    'grapher.rewiring_mlp.generic.model',
    'grapher.rewiring_mlp.attributed.model',
    'grapher.rewiring_mlp.molecular.graph_io',
):
    assert name not in sys.modules, name
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC)
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT, env=env)

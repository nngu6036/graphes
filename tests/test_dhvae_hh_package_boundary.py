from __future__ import annotations

import ast
import os
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PACKAGE = SRC / "grapher" / "models" / "dhvae_hh"


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _imported_modules(path: Path, *, top_level_only: bool = False) -> set[str]:
    tree = _tree(path)
    nodes = tree.body if top_level_only else ast.walk(tree)
    modules: set[str] = set()
    for node in nodes:
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


class DHVAEHHBoundaryTests(unittest.TestCase):
    def test_project_owned_baseline_is_colocated(self) -> None:
        expected = {
            "__init__.py",
            "wrapper.py",
            "degree_vae.py",
            "typed_degree_vae.py",
            "degree_sampler.py",
            "havel_hakimi.py",
            "typed_constructor.py",
            "training.py",
            "evaluation.py",
        }
        self.assertTrue(expected.issubset({path.name for path in PACKAGE.iterdir()}))

    def test_retired_baseline_packages_are_absent(self) -> None:
        self.assertFalse((SRC / "grapher" / "generators").exists())
        self.assertFalse((SRC / "grapher" / "construction").exists())

    def test_baseline_does_not_import_rewiring_packages(self) -> None:
        forbidden = (
            "grapher.rewiring_mlp.core",
            "grapher.rewiring_mlp.generic",
            "grapher.rewiring_mlp.attributed",
        )
        for path in PACKAGE.glob("*.py"):
            for module in _imported_modules(path):
                with self.subTest(path=path.name, module=module):
                    self.assertFalse(module.startswith(forbidden))

    def test_production_code_uses_canonical_baseline_imports(self) -> None:
        forbidden = (
            "grapher.generators.degree_vae",
            "grapher.generators.degree_sampler",
            "grapher.construction.coarse",
            "grapher.construction.typed",
        )
        production_files = list((SRC / "grapher").rglob("*.py")) + list(
            (ROOT / "scripts").glob("*.py")
        )
        for path in production_files:
            imports = _imported_modules(path)
            for module in forbidden:
                with self.subTest(path=path.relative_to(ROOT), module=module):
                    self.assertNotIn(module, imports)

    def test_neutral_typed_invariants_have_no_eager_model_or_torch_import(self) -> None:
        path = (
            SRC
            / "grapher"
            / "rewiring_mlp"
            / "molecular"
            / "typed_invariants.py"
        )
        imports = _imported_modules(path, top_level_only=True)
        self.assertNotIn("torch", imports)
        self.assertFalse(
            any(module.startswith("grapher.models.dhvae_hh") for module in imports)
        )

    def test_registry_resolves_wrapper_without_heavy_baseline_imports(self) -> None:
        code = """
import sys
from grapher.models.registry import get_wrapper_class
wrapper = get_wrapper_class('dhvae_hh')
assert wrapper.__name__ == 'DHVAEHHWrapper'
for name in (
    'grapher.models.dhvae_hh.degree_vae',
    'grapher.models.dhvae_hh.typed_degree_vae',
    'grapher.models.dhvae_hh.havel_hakimi',
    'grapher.models.dhvae_hh.typed_constructor',
):
    assert name not in sys.modules, name
"""
        env = dict(os.environ)
        env["PYTHONPATH"] = str(SRC)
        subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            cwd=ROOT,
            env=env,
        )

if __name__ == "__main__":
    unittest.main()

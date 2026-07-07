#!/usr/bin/env python
"""Generate molecules with topology-first GraphER + mixture CatFlow.

Alias for sample_qm9_mixture_catflow.py with a method-oriented name.
"""
from __future__ import annotations

from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("sample_qm9_mixture_catflow.py")), run_name="__main__")

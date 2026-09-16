#!/usr/bin/env python3
"""Run the project-owned minimal GSDM reference through the shared baseline CLI."""
from grapher.models.external_cli import main

if __name__ == "__main__":
    raise SystemExit(main("gdsm_simple"))

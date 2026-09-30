#!/usr/bin/env python3
"""Compatibility alias for the project-owned GSDM-simple baseline runner."""
from grapher.models.external_cli import main

if __name__ == "__main__":
    raise SystemExit(main("gdsm_simple"))

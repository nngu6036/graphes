#!/usr/bin/env python3
"""Compatibility alias for GSDM/GDSM through the shared GraphER baseline CLI."""
from grapher.models.external_cli import main

if __name__ == "__main__":
    raise SystemExit(main("gsdm"))

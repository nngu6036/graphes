#!/usr/bin/env python3
"""Run gdsm through the shared GraphER baseline CLI."""
from grapher.models.external_cli import main

if __name__ == "__main__":
    raise SystemExit(main("gdsm"))

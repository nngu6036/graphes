#!/usr/bin/env python3
"""Train/generate GDSM via the GraphER baseline API."""
from grapher.models.external_cli import main

if __name__ == "__main__":
    raise SystemExit(main("gdsm"))

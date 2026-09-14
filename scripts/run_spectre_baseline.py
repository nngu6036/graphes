#!/usr/bin/env python3
"""Train/generate SPECTRE with common-reference then model-specific config precedence."""
from grapher.models.external_cli import main

if __name__ == "__main__":
    raise SystemExit(main("spectre"))

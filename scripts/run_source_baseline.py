#!/usr/bin/env python3
"""Train or generate a source-backed baseline through the shared GraphER CLI.

This is the consolidated entry point for CatFlow, GSDM/GDSM, EDGE and SPECTRE.
The runner automatically loads the DeFoG-reference common profile (when one
exists) and then the model-specific YAML, so native/equivalent training budgets
override raw common epoch counts. Model-specific behavior remains in
``src/grapher/models`` and each model's YAML.
"""
from __future__ import annotations

import argparse
from collections.abc import Sequence

from grapher.models.external_cli import main as external_main
from grapher.models.registry import normalize_baseline_id

_SOURCE_BACKED = ("catflow", "gdsm", "edge", "spectre")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train/generate a source-backed GraphER baseline.",
        add_help=False,
    )
    parser.add_argument("--model", required=True, help="catflow, gdsm/gsdm, edge or spectre")
    args, remaining = parser.parse_known_args(argv)
    model = normalize_baseline_id(args.model)
    if model not in _SOURCE_BACKED:
        parser.error(
            f"{args.model!r} is not a source-backed model handled by this entry point; "
            f"choose one of {', '.join(_SOURCE_BACKED)}."
        )
    return external_main(model, remaining)


if __name__ == "__main__":
    raise SystemExit(main())

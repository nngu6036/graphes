#!/usr/bin/env python
"""Print fast statistics for one prepared graph dataset.

Run from the repository root, for example:

    PYTHONPATH=src python scripts/print_dataset_statistics.py \
        --dataset community_small

The command is read-only. It never builds, downloads, or rewrites a dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

from grapher.data.statistics import (
    compute_prepared_dataset_statistics,
    format_graph_statistics_table,
    resolve_prepared_dataset,
)
from grapher.utils.io import save_json


def _format_counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{key}={value}" for key, value in counts.items()) or "none"


def _format_attribute_fields(report: dict[str, Any], kind: str) -> str:
    fields = report[f"{kind}_attributes"]["fields"]
    if not fields:
        return "none"
    values = []
    for name, details in fields.items():
        coverage = details.get("coverage")
        suffix = f" ({100.0 * coverage:.1f}%)" if coverage is not None else ""
        values.append(f"{name}{suffix}")
    return ", ".join(values)


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.expanduser().resolve(strict=False).relative_to(
            directory.expanduser().resolve(strict=False)
        )
    except ValueError:
        return False
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Report aggregate and split statistics for a prepared NetworkX "
            "graph dataset."
        )
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help=(
            "Prepared dataset name or configs/datasets/<name>.yaml stem, for "
            "example community_small, sbm, qm9_attributed, or zinc."
        ),
    )
    parser.add_argument(
        "--root",
        "--dataset-root",
        dest="root",
        default="outputs/datasets",
        help="Root containing <dataset>/{train,val,test}.pkl.",
    )
    parser.add_argument(
        "--json-out",
        help="Optionally save the complete statistics report as JSON.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow --json-out to overwrite an existing file.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    output_path = Path(args.json_out) if args.json_out else None
    if output_path is not None and output_path.exists() and not args.force:
        parser.error(
            f"JSON output already exists: {output_path}. Use --force to "
            "overwrite it."
        )
    try:
        dataset = resolve_prepared_dataset(args.dataset, root=args.root)
    except (FileNotFoundError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    if output_path is not None and _is_within(output_path, dataset.directory):
        parser.error(
            "--json-out must be outside the prepared dataset directory so the "
            "reporter cannot overwrite dataset artifacts."
        )

    print(f"Dataset:    {dataset.requested_name}", flush=True)
    if dataset.serialized_name != dataset.requested_name:
        print(f"Resolved:   {dataset.serialized_name}", flush=True)
    print(f"Directory:  {dataset.directory}", flush=True)
    try:
        report = compute_prepared_dataset_statistics(
            dataset,
            progress=lambda message: print(message, flush=True),
        )
    except (FileNotFoundError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print()
    print(format_graph_statistics_table(report["statistics"]))

    overall = report["statistics"]["all"]
    print("\nDegree histogram (all splits):")
    print(f"  {_format_counts(overall['degree_histogram'])}")
    print("\nAttribute coverage (all splits):")
    print(f"  nodes: {_format_attribute_fields(overall, 'node')}")
    print(f"  edges: {_format_attribute_fields(overall, 'edge')}")

    molecular = overall.get("molecular_attributes")
    if molecular is not None:
        print("\nMolecular categories (all splits):")
        print(f"  atom types: {_format_counts(molecular['atom_type_counts'])}")
        print(f"  bond types: {_format_counts(molecular['bond_type_counts'])}")

    if output_path is not None:
        save_json(report, output_path)
        print(f"\nSaved JSON report: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def _format_counts(counts: Mapping[str, int]) -> str:
    return ", ".join(f"{name}={int(value)}" for name, value in counts.items())


def print_preparation_summary(
    *,
    dataset: str,
    source: str | Path,
    input_records: int,
    processed_records: int,
    accepted_graphs: int,
    rejection_reasons: Mapping[str, int],
    split_sizes: Mapping[str, int],
    outputs: Sequence[tuple[str, str | Path, str]],
) -> None:
    """Print one stable summary schema for every prepared graph dataset."""

    input_records = int(input_records)
    processed_records = int(processed_records)
    accepted_graphs = int(accepted_graphs)
    rejected_records = int(sum(int(value) for value in rejection_reasons.values()))
    unexamined_records = input_records - processed_records

    if min(input_records, processed_records, accepted_graphs, rejected_records) < 0:
        raise ValueError("Dataset preparation counts must be non-negative.")
    if processed_records > input_records:
        raise ValueError("processed_records cannot exceed input_records.")
    if accepted_graphs + rejected_records != processed_records:
        raise ValueError(
            "accepted_graphs plus rejected records must equal processed_records."
        )
    if sum(int(value) for value in split_sizes.values()) != accepted_graphs:
        raise ValueError("Split sizes must sum to accepted_graphs.")

    print(f"Prepared dataset: {dataset}")
    print(f"  source: {source}")
    print(f"  input records: {input_records}")
    print(f"  processed records: {processed_records}")
    print(f"  accepted graphs: {accepted_graphs}")
    print(f"  rejected records: {rejected_records}")
    print(f"  unexamined records: {unexamined_records}")
    print(f"  rejection reasons: {dict(sorted(rejection_reasons.items()))}")
    print(f"  splits: {_format_counts(split_sizes)}")
    for label, path, schema in outputs:
        print(f"  {label}: {path} ({schema})")


def common_preparation_report(
    *,
    input_records: int,
    processed_records: int,
    accepted_graphs: int,
    rejection_reasons: Mapping[str, int],
) -> dict[str, Any]:
    """Return common machine-readable count fields for preparation reports."""

    normalized_reasons = {
        str(name): int(value) for name, value in sorted(rejection_reasons.items())
    }
    rejected_records = int(sum(normalized_reasons.values()))
    input_records = int(input_records)
    processed_records = int(processed_records)
    accepted_graphs = int(accepted_graphs)
    if accepted_graphs + rejected_records != processed_records:
        raise ValueError(
            "accepted_graphs plus rejected records must equal processed_records."
        )
    if processed_records > input_records:
        raise ValueError("processed_records cannot exceed input_records.")
    return {
        "num_input_records": input_records,
        "num_processed_records": processed_records,
        "num_accepted_graphs": accepted_graphs,
        "num_rejected_records": rejected_records,
        "num_unexamined_records": input_records - processed_records,
        "rejection_reasons": normalized_reasons,
    }

#!/usr/bin/env python3
"""Summarize paired GraphER final-connectivity rejection ablations.

Each pair must use the same trained checkpoint and generation seed.  The baseline
is the ordinary sampler; the fixed run rejects disconnected final graphs as whole
samples and continues sampling until the requested returned count is reached.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any


MOLECULAR_METRICS = (
    "validity_without_correction",
    "uniqueness_rate",
    "novelty_rate",
    "nspdk_mmd",
    "fcd",
)
GENERATION_METRICS = (
    "connectedness_rate",
    "raw_final_connectedness_rate",
    "generation_yield",
    "generation_attempts",
    "rejected_disconnected_final_graphs",
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _optional_qed(generation_dir: Path) -> float | None:
    candidates = [
        generation_dir / "evaluation_qed" / "qed_metrics.json",
        generation_dir / "qed" / "qed_metrics.json",
        generation_dir / "qed_metrics.json",
    ]
    for path in candidates:
        if path.is_file():
            value = _read_json(path).get("metrics", {}).get("qed_mean")
            return None if value is None else float(value)
    return None


def load_run(generation_dir: Path) -> dict[str, Any]:
    manifest = _read_json(generation_dir / "manifest.json")
    rewiring = _read_json(generation_dir / "rewiring_diagnostics.json")
    molecular = _read_json(
        generation_dir / "evaluation_molecules" / "molecular_evaluation_metrics.json"
    )
    metrics = molecular["metrics"]
    aggregate = rewiring["aggregate"]
    row: dict[str, Any] = {
        "generation_dir": str(generation_dir),
        "run_id": manifest.get("run_id"),
        "generation_id": manifest.get("generation_id"),
        "seed": manifest.get("generation_seed"),
        "checkpoint_sha256": manifest.get("checkpoint", {}).get("sha256"),
        "num_requested": manifest.get("num_requested"),
        "num_generated": manifest.get("num_generated"),
    }
    for key in MOLECULAR_METRICS:
        value = metrics.get(key)
        row[key] = None if value is None else float(value)
    for key in GENERATION_METRICS:
        value = aggregate.get(key)
        row[key] = None if value is None else float(value)
    row["qed_mean"] = _optional_qed(generation_dir)
    return row


def validate_pair(baseline: dict[str, Any], fixed: dict[str, Any]) -> None:
    if baseline["seed"] != fixed["seed"]:
        raise ValueError(f"Generation seed mismatch: {baseline['seed']} vs {fixed['seed']}")
    if baseline["checkpoint_sha256"] != fixed["checkpoint_sha256"]:
        raise ValueError(
            f"Checkpoint mismatch for seed {baseline['seed']}; connectivity ablation must reuse the same checkpoint"
        )
    if baseline["num_requested"] != fixed["num_requested"]:
        raise ValueError(f"Requested-count mismatch for seed {baseline['seed']}")
    if baseline["num_generated"] != fixed["num_generated"]:
        raise ValueError(f"Returned-count mismatch for seed {baseline['seed']}")


def _summary(values: list[float]) -> dict[str, float | int | None]:
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return {
        "n": len(finite),
        "mean": mean(finite) if finite else None,
        "std": stdev(finite) if len(finite) > 1 else None,
    }


def summarize(pairs: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
    keys = (*MOLECULAR_METRICS, *GENERATION_METRICS, "qed_mean")
    per_seed = []
    for baseline, fixed in pairs:
        validate_pair(baseline, fixed)
        row = {"seed": baseline["seed"], "baseline": baseline, "connectivity_filter": fixed, "delta": {}}
        for key in keys:
            a, b = baseline.get(key), fixed.get(key)
            row["delta"][key] = None if a is None or b is None else float(b) - float(a)
        per_seed.append(row)
    aggregate = {"baseline": {}, "connectivity_filter": {}, "paired_delta": {}}
    for key in keys:
        aggregate["baseline"][key] = _summary([p[0].get(key) for p in pairs])
        aggregate["connectivity_filter"][key] = _summary([p[1].get(key) for p in pairs])
        aggregate["paired_delta"][key] = _summary([
            (float(p[1][key]) - float(p[0][key]))
            for p in pairs if p[0].get(key) is not None and p[1].get(key) is not None
        ])
    return {
        "schema_version": 1,
        "comparison": "main_vs_final_connectivity_reject_and_resample",
        "paired_by": "same checkpoint SHA256 and generation seed",
        "delta_definition": "connectivity_filter - baseline",
        "per_seed": per_seed,
        "aggregate": aggregate,
    }


def write_csv(report: dict[str, Any], path: Path) -> None:
    keys = (*MOLECULAR_METRICS, *GENERATION_METRICS, "qed_mean")
    fields = ["seed", "variant", *keys]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in report["per_seed"]:
            for variant in ("baseline", "connectivity_filter"):
                data = row[variant]
                writer.writerow({"seed": row["seed"], "variant": variant, **{k: data.get(k) for k in keys}})
            writer.writerow({"seed": row["seed"], "variant": "delta", **row["delta"]})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-dir", type=Path, action="append", required=True,
                   help="Baseline generation directory; repeat once per seed")
    p.add_argument("--fixed-dir", type=Path, action="append", required=True,
                   help="Connectivity-filter generation directory; repeat once per seed")
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    if len(args.baseline_dir) != len(args.fixed_dir):
        p.error("Use the same number of --baseline-dir and --fixed-dir arguments")
    pairs = [(load_run(a), load_run(b)) for a, b in zip(args.baseline_dir, args.fixed_dir)]
    report = summarize(pairs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "connectivity_ablation_summary.json"
    csv_path = args.output_dir / "connectivity_ablation_per_seed.csv"
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    write_csv(report, csv_path)
    print(json.dumps(report["aggregate"], indent=2, allow_nan=False))
    print(f"Saved {json_path}")
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()

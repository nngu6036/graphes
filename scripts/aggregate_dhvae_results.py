from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from grapher.evaluation.run_utils import aggregate_numeric_results, parse_run_ids
from grapher.registry import available_datasets
from grapher.utils.io import load_json, save_json
from grapher.utils.logging import get_logger

logger = get_logger(__name__)

MODEL = "dhvae"
METRIC_FILENAME = "dhvae_metrics.json"


def _run_dir_name(run_id: int) -> str:
    return f"run_{int(run_id):03d}"


def _metric_path(metric_dir: str | Path, dataset: str, run_id: int) -> Path:
    return Path(metric_dir) / dataset / MODEL / _run_dir_name(run_id) / METRIC_FILENAME


def _aggregate_metric_path(metric_dir: str | Path, dataset: str) -> Path:
    return Path(metric_dir) / dataset / MODEL / "dhvae_metrics.aggregate.json"


def _discover_run_ids(metric_dir: str | Path, dataset: str) -> list[int]:
    base = Path(metric_dir) / dataset / MODEL
    run_ids: list[int] = []
    for path in sorted(base.glob(f"run_*/{METRIC_FILENAME}")):
        match = re.fullmatch(r"run_(\d+)", path.parent.name)
        if match:
            run_ids.append(int(match.group(1)))
    return sorted(dict.fromkeys(run_ids))


def _load_run_payload(path: Path, *, dataset: str, run_id: int) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"DH-VAE metric file not found for run_id={run_id}: {path}")
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}, got {type(payload).__name__}")
    if "results" not in payload:
        raise KeyError(f"Metric file is missing a 'results' object: {path}")
    if bool(payload.get("is_aggregate", False)):
        raise ValueError(f"Expected an individual run metric file, got aggregate metric file: {path}")

    payload_dataset = payload.get("dataset")
    payload_model = payload.get("model")
    if payload_dataset is not None and str(payload_dataset) != str(dataset):
        raise ValueError(f"Metric file dataset mismatch in {path}: expected {dataset!r}, got {payload_dataset!r}")
    if payload_model is not None and str(payload_model) != MODEL:
        raise ValueError(f"Metric file model mismatch in {path}: expected {MODEL!r}, got {payload_model!r}")
    return payload


def aggregate_dhvae_results(
    *,
    dataset: str,
    run_ids: Sequence[int] | None = None,
    metric_dir: str | Path = "outputs/metrics",
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Average numeric DH-VAE metric results across existing per-run JSON files."""

    resolved_run_ids = [int(run_id) for run_id in run_ids] if run_ids is not None else _discover_run_ids(metric_dir, dataset)
    if not resolved_run_ids:
        raise FileNotFoundError(f"No DH-VAE run metric files found under {Path(metric_dir) / dataset / MODEL}")
    if len(set(resolved_run_ids)) != len(resolved_run_ids):
        raise ValueError(f"Duplicate run ids are not allowed: {resolved_run_ids}")

    source_paths = [_metric_path(metric_dir, dataset, run_id) for run_id in resolved_run_ids]
    run_payloads = [
        _load_run_payload(path, dataset=dataset, run_id=run_id)
        for path, run_id in zip(source_paths, resolved_run_ids, strict=True)
    ]
    numeric = aggregate_numeric_results(run_payloads)
    payload = {
        "dataset": dataset,
        "model": MODEL,
        "metric_family": "dhvae_metrics",
        "runtime_seconds": sum(float(item.get("runtime_seconds", 0.0) or 0.0) for item in run_payloads),
        "is_aggregate": True,
        "run_ids": resolved_run_ids,
        "num_runs": len(resolved_run_ids),
        "protocol": {
            "run_ids": resolved_run_ids,
            "aggregation": "numeric results are averaged across run_ids; *_std values are population standard deviations across run_ids",
            "source_metric_files": [str(path) for path in source_paths],
        },
        "results": numeric["flat"],
        "run_result_summary": numeric["nested"],
    }

    output_path = Path(output) if output else _aggregate_metric_path(metric_dir, dataset)
    save_json(payload, output_path, force=True)
    print("aggregate:")
    for key in sorted(numeric["nested"]):
        summary = numeric["nested"][key]
        print(f"{key}: {summary['mean']:.8g} +- {summary['std']:.8g}")
    logger.info("Saved aggregate DH-VAE metrics to %s", output_path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Average DH-VAE metric JSON outputs across multiple run ids.")
    parser.add_argument("--dataset", required=True, choices=available_datasets())
    parser.add_argument("--run-id", type=int, default=None, help="Aggregate one explicit run id.")
    parser.add_argument("--run-ids", type=int, nargs="+", default=None, help="Run ids to average. Defaults to all discovered DH-VAE run metrics.")
    parser.add_argument("--metric-dir", type=str, default="outputs/metrics")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    try:
        run_ids = parse_run_ids(run_id=args.run_id, run_ids=args.run_ids) if args.run_id is not None or args.run_ids else None
        aggregate_dhvae_results(
            dataset=args.dataset,
            run_ids=run_ids,
            metric_dir=args.metric_dir,
            output=args.output,
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from grapher.evaluation.data_io import load_dataset_splits
from grapher.evaluation.metrics import (
    degree_distribution_kl,
    degree_sequence_histogram,
    mmd_gaussian_emd,
)
from grapher.evaluation.run_utils import (
    aggregate_metric_path,
    aggregate_numeric_results,
    metric_path,
    parse_run_ids,
    run_seed,
    sample_path,
)
from grapher.generation.rewiring import degree_sequence
from grapher.registry import available_datasets
from grapher.utils.io import load_pickle, save_json
from grapher.utils.logging import get_logger

logger = get_logger(__name__)

METRIC_FILENAME = "dhvae_metrics.json"


def _subsample(items: Sequence[Any], max_items: int | None, seed: int) -> list[Any]:
    items = list(items)
    if max_items is None or max_items <= 0 or len(items) <= max_items:
        return items
    rng = np.random.default_rng(int(seed))
    indices = rng.choice(len(items), size=int(max_items), replace=False)
    return [items[int(index)] for index in indices]


def _load_generated_sequences(dataset: str, run_id: int | None, sample_file: str | None) -> tuple[list[list[int]], Path]:
    candidates = [Path(sample_file)] if sample_file else [sample_path(dataset, "dhvae", run_id=run_id)]
    for path in candidates:
        if not path.exists():
            continue
        payload = load_pickle(path)
        if isinstance(payload, dict) and "degree_sequences" in payload:
            payload = payload["degree_sequences"]
        if not isinstance(payload, list):
            raise TypeError(f"Expected generated DH-VAE payload to be a list, got {type(payload).__name__}: {path}")
        sequences = [[int(value) for value in sequence] for sequence in payload]
        return sequences, path
    raise FileNotFoundError("Could not find generated DH-VAE samples. Tried: " + ", ".join(str(path) for path in candidates))


def _degree_histogram_matrix(sequences: Sequence[Sequence[int]], max_degree: int) -> np.ndarray:
    if not sequences:
        raise ValueError("Cannot compute DH-VAE metrics from an empty sequence set.")
    return np.asarray([degree_sequence_histogram(sequence, max_degree) for sequence in sequences], dtype=np.float64)


def _max_degree(*sequence_groups: Sequence[Sequence[int]], degree_bins: int) -> int:
    value = max(
        [0, int(degree_bins) - 1]
        + [max(sequence) if sequence else 0 for group in sequence_groups for sequence in group]
    )
    return int(value)


def _sequence_summary(sequences: Sequence[Sequence[int]]) -> dict[str, float | int]:
    sizes = np.asarray([len(sequence) for sequence in sequences], dtype=np.float64)
    max_degrees = np.asarray([max(sequence) if sequence else 0 for sequence in sequences], dtype=np.float64)
    edge_counts = np.asarray([sum(sequence) / 2.0 for sequence in sequences], dtype=np.float64)

    def summary(values: np.ndarray, key: str) -> dict[str, float]:
        if values.size == 0:
            return {f"{key}_mean": 0.0, f"{key}_std": 0.0}
        return {f"{key}_mean": float(values.mean()), f"{key}_std": float(values.std(ddof=0))}

    out: dict[str, float | int] = {"num_sequences": int(len(sequences))}
    out.update(summary(sizes, "num_nodes"))
    out.update(summary(max_degrees, "max_degree"))
    out.update(summary(edge_counts, "num_edges"))
    return out


def _comparison_metrics(
    *,
    target_sequences: Sequence[Sequence[int]],
    candidate_sequences: Sequence[Sequence[int]],
    max_degree: int,
    sigma: float,
) -> dict[str, float]:
    target_hist = _degree_histogram_matrix(target_sequences, max_degree)
    candidate_hist = _degree_histogram_matrix(candidate_sequences, max_degree)
    return {
        "kl": degree_distribution_kl(target_sequences, candidate_sequences, max_degree=max_degree),
        "mmd": mmd_gaussian_emd(target_hist, candidate_hist, sigma=float(sigma)),
    }


def evaluate(
    *,
    dataset: str,
    run_id: int | None,
    dataset_root: str,
    sample_file: str | None,
    output: str | None,
    seed: int,
    max_train_sequences: int | None,
    max_test_sequences: int | None,
    max_generated_sequences: int | None,
    degree_bins: int,
    sigma: float,
) -> dict[str, Any]:
    start = time.perf_counter()
    splits = load_dataset_splits(dataset, output_root=dataset_root, build_if_missing=True)
    for split in ("train", "test"):
        if split not in splits:
            raise KeyError(f"Dataset {dataset!r} is missing split {split!r}; available={sorted(splits)}")

    train_graphs = _subsample(splits["train"], max_train_sequences, seed)
    test_graphs = _subsample(splits["test"], max_test_sequences, seed + 1)
    train_sequences = [degree_sequence(nx.Graph(graph)) for graph in train_graphs]
    test_sequences = [degree_sequence(nx.Graph(graph)) for graph in test_graphs]
    generated_sequences, sample_path_used = _load_generated_sequences(dataset, run_id, sample_file)
    generated_sequences = _subsample(generated_sequences, max_generated_sequences, seed + 2)

    if not train_sequences:
        raise ValueError("Train split produced no degree sequences.")
    if not test_sequences:
        raise ValueError("Test split produced no degree sequences.")
    if not generated_sequences:
        raise ValueError("Generated DH-VAE samples produced no degree sequences.")

    max_degree = _max_degree(train_sequences, test_sequences, generated_sequences, degree_bins=degree_bins)
    train_test = _comparison_metrics(
        target_sequences=test_sequences,
        candidate_sequences=train_sequences,
        max_degree=max_degree,
        sigma=sigma,
    )
    generated_test = _comparison_metrics(
        target_sequences=test_sequences,
        candidate_sequences=generated_sequences,
        max_degree=max_degree,
        sigma=sigma,
    )
    elapsed = time.perf_counter() - start

    results = {
        "train_test_kl": train_test["kl"],
        "train_test_mmd": train_test["mmd"],
        "generated_test_kl": generated_test["kl"],
        "generated_test_mmd": generated_test["mmd"],
    }
    payload = {
        "dataset": dataset,
        "model": "dhvae",
        "run_id": run_id,
        "metric_family": "dhvae_metrics",
        "runtime_seconds": elapsed,
        "num_train_sequences": len(train_sequences),
        "num_test_sequences": len(test_sequences),
        "num_generated_sequences": len(generated_sequences),
        "protocol": {
            "seed": int(seed),
            "dataset_root": str(dataset_root),
            "sample_path": str(sample_path_used),
            "max_train_sequences": max_train_sequences,
            "max_test_sequences": max_test_sequences,
            "max_generated_sequences": max_generated_sequences,
            "degree_bins": degree_bins,
            "max_degree": max_degree,
            "sigma": sigma,
            "kl_direction": "KL(test || train) and KL(test || generated), matching degree_distribution_kl(reference, candidate).",
        },
        "sequence_summaries": {
            "train": _sequence_summary(train_sequences),
            "test": _sequence_summary(test_sequences),
            "generated": _sequence_summary(generated_sequences),
        },
        "results": results,
    }

    output_path = Path(output) if output else metric_path(dataset, "dhvae", METRIC_FILENAME, run_id=run_id)
    save_json(payload, output_path, force=True)
    print(f"train_test_kl: {results['train_test_kl']:.8g}")
    print(f"train_test_mmd: {results['train_test_mmd']:.8g}")
    print(f"generated_test_kl: {results['generated_test_kl']:.8g}")
    print(f"generated_test_mmd: {results['generated_test_mmd']:.8g}")
    logger.info("Saved DH-VAE metrics to %s", output_path)
    return payload


def evaluate_run_ids(
    *,
    dataset: str,
    run_ids: Sequence[int],
    dataset_root: str,
    output: str | None,
    seed: int,
    max_train_sequences: int | None,
    max_test_sequences: int | None,
    max_generated_sequences: int | None,
    degree_bins: int,
    sigma: float,
) -> dict[str, Any]:
    run_payloads = [
        evaluate(
            dataset=dataset,
            run_id=int(run_id),
            dataset_root=dataset_root,
            sample_file=None,
            output=None,
            seed=run_seed(seed, int(run_id)),
            max_train_sequences=max_train_sequences,
            max_test_sequences=max_test_sequences,
            max_generated_sequences=max_generated_sequences,
            degree_bins=degree_bins,
            sigma=sigma,
        )
        for run_id in run_ids
    ]
    numeric = aggregate_numeric_results(run_payloads)
    payload = {
        "dataset": dataset,
        "model": "dhvae",
        "metric_family": "dhvae_metrics",
        "runtime_seconds": sum(float(item.get("runtime_seconds", 0.0) or 0.0) for item in run_payloads),
        "is_aggregate": True,
        "run_ids": [int(run_id) for run_id in run_ids],
        "num_runs": len(run_ids),
        "protocol": {
            "base_seed": int(seed),
            "run_ids": [int(run_id) for run_id in run_ids],
            "seed_stride": 1000,
            "aggregation": "numeric results are averaged across run_ids; *_std values are population standard deviations across run_ids",
            "source_metric_files": [str(metric_path(dataset, "dhvae", METRIC_FILENAME, run_id=int(run_id))) for run_id in run_ids],
        },
        "results": numeric["flat"],
        "run_result_summary": numeric["nested"],
    }
    output_path = Path(output) if output else aggregate_metric_path(dataset, "dhvae", METRIC_FILENAME)
    save_json(payload, output_path, force=True)
    print("aggregate:")
    for key in ("train_test_kl", "train_test_mmd", "generated_test_kl", "generated_test_mmd"):
        print(f"{key}: {payload['results'].get(key):.8g} +- {payload['results'].get(key + '_std'):.8g}")
    logger.info("Saved aggregate DH-VAE metrics to %s", output_path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DH-VAE degree-sequence KL and MMD against prepared train/test sequences.")
    parser.add_argument("--dataset", required=True, choices=available_datasets())
    parser.add_argument("--run-id", type=int, default=None)
    parser.add_argument("--run-ids", type=int, nargs="+", default=None)
    parser.add_argument("--dataset-root", type=str, default="outputs/datasets")
    parser.add_argument("--sample-path", type=str, default=None, help="Optional explicit generated DH-VAE pickle path. Not valid with --run-ids.")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-sequences", type=int, default=None)
    parser.add_argument("--max-test-sequences", type=int, default=None)
    parser.add_argument("--max-generated-sequences", type=int, default=None)
    parser.add_argument("--degree-bins", type=int, default=20)
    parser.add_argument("--sigma", type=float, default=1.0)
    args = parser.parse_args()

    if args.run_ids and args.sample_path:
        parser.error("--sample-path cannot be used with --run-ids because each run must resolve its own sample file.")
    try:
        run_ids = parse_run_ids(run_id=args.run_id, run_ids=args.run_ids) if (args.run_id is not None or args.run_ids) else None
    except ValueError as exc:
        parser.error(str(exc))

    if run_ids is not None and (args.run_ids or len(run_ids) > 1):
        evaluate_run_ids(
            dataset=args.dataset,
            run_ids=run_ids,
            dataset_root=args.dataset_root,
            output=args.output,
            seed=args.seed,
            max_train_sequences=args.max_train_sequences,
            max_test_sequences=args.max_test_sequences,
            max_generated_sequences=args.max_generated_sequences,
            degree_bins=args.degree_bins,
            sigma=args.sigma,
        )
    else:
        evaluate(
            dataset=args.dataset,
            run_id=args.run_id,
            dataset_root=args.dataset_root,
            sample_file=args.sample_path,
            output=args.output,
            seed=args.seed,
            max_train_sequences=args.max_train_sequences,
            max_test_sequences=args.max_test_sequences,
            max_generated_sequences=args.max_generated_sequences,
            degree_bins=args.degree_bins,
            sigma=args.sigma,
        )


if __name__ == "__main__":
    main()

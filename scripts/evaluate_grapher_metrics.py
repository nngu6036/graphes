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
    clustering_histogram,
    degree_distribution_kl,
    degree_histogram,
    degree_sequence_histogram,
    descriptor_matrix,
    mmd_gaussian_emd,
    mmd_rbf,
    motif_proxy_vector,
    spectral_histogram,
    structural_summary,
)
from grapher.evaluation.run_utils import make_model_run_config, metric_path, sample_path
from grapher.generation.rewiring import check_sequence_validity, connected_sequence_feasible, degree_sequence
from grapher.generation.validity import quality_metrics
from grapher.registry import available_datasets
from grapher.utils.io import load_pickle, load_yaml, save_json
from grapher.utils.logging import get_logger

logger = get_logger(__name__)

METRIC_FILENAME = "grapher_metrics.json"


def _resolved_sample_candidate(cfg: dict, dataset: str, model: str, run_id: int | None) -> Path | None:
    configured = cfg.get("samples_path")
    if run_id is not None:
        if isinstance(configured, str) and "run_id" in configured:
            return Path(configured)
        if isinstance(configured, str) and configured:
            base = Path(configured)
            return base.parent / model / f"run_{run_id:03d}.pkl"
        return sample_path(dataset, model, run_id=run_id)
    return Path(configured) if configured else sample_path(dataset, model, run_id=None)


def _subsample(items: Sequence[Any], max_items: int | None, seed: int) -> list[Any]:
    items = list(items)
    if max_items is None or max_items <= 0 or len(items) <= max_items:
        return items
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(items), size=int(max_items), replace=False)
    return [items[int(i)] for i in idx]


def _default_model_config(model: str) -> Path:
    if model == "dhvae":
        return Path("configs/models/dhvae.yaml")
    return Path("configs/models/grapher_generic.yaml")


def _load_generated_payload(dataset: str, model: str, cfg: dict, run_id: int | None, explicit_sample_path: str | None):
    candidates: list[Path] = []
    if explicit_sample_path:
        candidates.append(Path(explicit_sample_path))
    resolved = _resolved_sample_candidate(cfg, dataset, model, run_id)
    if resolved is not None:
        candidates.append(resolved)
    candidates.append(sample_path(dataset, model, run_id=run_id))
    configured = cfg.get("samples_path")
    if configured:
        candidates.append(Path(configured))
    if model == "dhvae":
        candidates.append(Path("outputs/samples") / dataset / "dhvae_degree_sequences.pkl")
    for path in candidates:
        if path.exists():
            return load_pickle(path), path
    raise FileNotFoundError("Could not find generated samples. Tried: " + ", ".join(str(p) for p in candidates))


def _payload_to_graphs_and_sequences(payload) -> tuple[list[nx.Graph], list[list[int]], str]:
    if not isinstance(payload, list):
        raise TypeError(f"Expected generated sample payload to be a list, got {type(payload)}")
    if not payload:
        return [], [], "empty"
    if all(isinstance(x, nx.Graph) for x in payload):
        graphs = [nx.convert_node_labels_to_integers(nx.Graph(g), ordering="sorted") for g in payload]
        return graphs, [degree_sequence(g) for g in graphs], "graphs"
    if all(isinstance(x, (list, tuple, np.ndarray)) for x in payload):
        return [], [[int(v) for v in x] for x in payload], "degree_sequences"
    raise TypeError("Generated sample list must contain either NetworkX graphs or degree sequences.")


def _graph_summary(graphs: Sequence[nx.Graph]) -> dict[str, float | int]:
    if not graphs:
        return {
            "connectedness_rate": 0.0,
            "zero_degree_graph_count": 0,
            "clustering_mean": 0.0,
            "clustering_std": 0.0,
        }
    connected = [float(nx.is_connected(g)) if g.number_of_nodes() > 0 else 0.0 for g in graphs]
    zero_degree_graph_count = sum(1 for g in graphs if any(int(degree) == 0 for _, degree in g.degree()))
    clustering = np.asarray([nx.average_clustering(g) if g.number_of_nodes() > 0 else 0.0 for g in graphs], dtype=np.float64)
    return {
        "connectedness_rate": float(np.mean(connected)),
        "zero_degree_graph_count": int(zero_degree_graph_count),
        "clustering_mean": float(clustering.mean()),
        "clustering_std": float(clustering.std(ddof=0)),
    }


def _print_graph_summary(reference_graphs: Sequence[nx.Graph], generated_graphs: Sequence[nx.Graph]) -> None:
    reference = _graph_summary(reference_graphs)
    print(f"reference connectedness rate: {reference['connectedness_rate']:.6f}")
    print(f"reference zero-degree graph count: {reference['zero_degree_graph_count']}")
    print(f"reference clustering mean/std: {reference['clustering_mean']:.6f} / {reference['clustering_std']:.6f}")
    if generated_graphs:
        generated = _graph_summary(generated_graphs)
        print(f"generated clustering mean/std: {generated['clustering_mean']:.6f} / {generated['clustering_std']:.6f}")
    else:
        print("generated clustering mean/std: n/a")


def evaluate(
    *,
    dataset: str,
    model: str,
    model_config: dict,
    dataset_root: str,
    reference_split: str,
    max_reference_graphs: int | None,
    max_generated_graphs: int | None,
    seed: int,
    run_id: int | None,
    sample_file: str | None,
    output: str | None,
    degree_bins: int,
    clustering_bins: int,
    spectral_bins: int,
    sigma: float,
) -> dict:
    start = time.perf_counter()
    cfg = make_model_run_config(model_config, dataset=dataset, model=model, run_id=run_id, seed=seed, use_run_paths=run_id is not None)
    splits = load_dataset_splits(dataset, output_root=dataset_root, build_if_missing=True)
    if reference_split not in splits:
        raise KeyError(f"Unknown split {reference_split!r}; available={sorted(splits)}")
    reference_graphs = _subsample(splits[reference_split], max_reference_graphs, seed)
    train_graphs = list(splits.get("train", []))
    payload, sample_path_used = _load_generated_payload(dataset, model, cfg, run_id, sample_file)
    generated_graphs, generated_sequences, payload_type = _payload_to_graphs_and_sequences(payload)
    if generated_graphs:
        generated_graphs = _subsample(generated_graphs, max_generated_graphs, seed + 17)
        generated_sequences = [degree_sequence(g) for g in generated_graphs]
    elif generated_sequences:
        generated_sequences = _subsample(generated_sequences, max_generated_graphs, seed + 17)

    _print_graph_summary(reference_graphs, generated_graphs)

    reference_sequences = [degree_sequence(g) for g in reference_graphs]
    max_degree = max(
        [0]
        + [max(seq) if seq else 0 for seq in reference_sequences]
        + [max(seq) if seq else 0 for seq in generated_sequences]
    )
    max_degree = max(max_degree, int(degree_bins) - 1)

    results: dict[str, float | int | None] = {}
    debug: dict[str, Any] = {"sample_path": str(sample_path_used), "payload_type": payload_type}

    if generated_sequences:
        ref_seq_desc = np.asarray([degree_sequence_histogram(seq, max_degree) for seq in reference_sequences], dtype=np.float64)
        gen_seq_desc = np.asarray([degree_sequence_histogram(seq, max_degree) for seq in generated_sequences], dtype=np.float64)
        results["degree_sequence_mmd"] = mmd_gaussian_emd(ref_seq_desc, gen_seq_desc, sigma=sigma)
        results["degree_sequence_kl"] = degree_distribution_kl(reference_sequences, generated_sequences, max_degree=max_degree)
        validity = [check_sequence_validity(seq)[0] for seq in generated_sequences]
        connected_feasible = [connected_sequence_feasible(seq)[0] for seq in generated_sequences]
        results["degree_sequence_graphicality_rate"] = float(np.mean(validity)) if validity else 0.0
        results["degree_sequence_connected_feasible_rate"] = float(np.mean(connected_feasible)) if connected_feasible else 0.0

    if generated_graphs:
        descriptor_specs = {
            "degree_mmd": (lambda g: degree_histogram(g, max_degree=max_degree), "emd"),
            "clustering_mmd": (lambda g: clustering_histogram(g, bins=clustering_bins), "emd"),
            "spectral_mmd": (lambda g: spectral_histogram(g, bins=spectral_bins), "emd"),
            "motif_proxy_mmd": (motif_proxy_vector, "rbf"),
            "structural_summary_mmd": (structural_summary, "rbf"),
        }
        for name, (fn, kind) in descriptor_specs.items():
            ref_desc = descriptor_matrix(reference_graphs, fn)
            gen_desc = descriptor_matrix(generated_graphs, fn)
            results[name] = mmd_gaussian_emd(ref_desc, gen_desc, sigma=sigma) if kind == "emd" else mmd_rbf(ref_desc, gen_desc, sigma=None)
            debug[name] = {"reference_shape": list(ref_desc.shape), "generated_shape": list(gen_desc.shape)}
        results.update(quality_metrics(generated_graphs, reference_graphs=train_graphs, dataset=dataset))

    elapsed = time.perf_counter() - start
    output_path = Path(output) if output else metric_path(dataset, model, METRIC_FILENAME, run_id=run_id)
    payload_out = {
        "dataset": dataset,
        "model": model,
        "run_id": run_id,
        "metric_family": "grapher_metrics",
        "num_reference_graphs": len(reference_graphs),
        "num_generated_graphs": len(generated_graphs),
        "num_generated_degree_sequences": len(generated_sequences),
        "runtime_seconds": elapsed,
        "protocol": {
            "seed": seed,
            "reference_split": reference_split,
            "max_reference_graphs": max_reference_graphs,
            "max_generated_graphs": max_generated_graphs,
            "degree_bins": degree_bins,
            "clustering_bins": clustering_bins,
            "spectral_bins": spectral_bins,
            "sigma": sigma,
            "motif_note": "motif_proxy_mmd is a lightweight higher-order structural proxy; use ORCA externally for exact orbit-count MMD if needed.",
        },
        "debug": debug,
        "results": results,
    }
    save_json(payload_out, output_path, force=True)
    logger.info("Saved metrics to %s", output_path)
    return payload_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated GraphER graphs or DH-VAE degree sequences.")
    parser.add_argument("--dataset", required=True, choices=available_datasets())
    parser.add_argument("--model", choices=["grapher", "dhvae"], default="grapher")
    parser.add_argument("--model-config", type=str, default=None)
    parser.add_argument("--dataset-root", type=str, default="outputs/datasets")
    parser.add_argument("--reference-split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--max-reference-graphs", type=int, default=None)
    parser.add_argument("--max-generated-graphs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id", type=int, default=None)
    parser.add_argument("--sample-path", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--degree-bins", type=int, default=20)
    parser.add_argument("--clustering-bins", type=int, default=20)
    parser.add_argument("--spectral-bins", type=int, default=20)
    parser.add_argument("--sigma", type=float, default=1.0)
    args = parser.parse_args()
    cfg_path = Path(args.model_config) if args.model_config else _default_model_config(args.model)
    evaluate(
        dataset=args.dataset,
        model=args.model,
        model_config=load_yaml(cfg_path),
        dataset_root=args.dataset_root,
        reference_split=args.reference_split,
        max_reference_graphs=args.max_reference_graphs,
        max_generated_graphs=args.max_generated_graphs,
        seed=args.seed,
        run_id=args.run_id,
        sample_file=args.sample_path,
        output=args.output,
        degree_bins=args.degree_bins,
        clustering_bins=args.clustering_bins,
        spectral_bins=args.spectral_bins,
        sigma=args.sigma,
    )


if __name__ == "__main__":
    main()

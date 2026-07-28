#!/usr/bin/env python
"""Evaluate DH-VAE before target-summary training or graph refinement."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import torch

from grapher.construction.coarse import (
    assert_constructor_validity,
    construct_coarse_graph,
)
from grapher.data.io import load_dataset_splits
from grapher.evaluation.degree_sequences import evaluate_degree_sequence_sets
from grapher.generators.degree_vae import (
    connected_feasible_degree_sequence,
    load_degree_vae_checkpoint,
)
from grapher.utils.io import ensure_dir, load_yaml, save_json


def _degree_sequences(graphs: list[nx.Graph]) -> list[list[int]]:
    return [
        sorted([int(degree) for _, degree in graph.degree()], reverse=True)
        for graph in graphs
    ]


def _subsample(
    values: list[Any],
    limit: int | None,
    rng: np.random.Generator,
) -> list[Any]:
    if limit is None or int(limit) <= 0 or len(values) <= int(limit):
        return list(values)
    indices = rng.choice(len(values), size=int(limit), replace=False)
    return [values[int(index)] for index in indices]


def _mean_bool(diagnostics: list[dict[str, Any]], key: str) -> float:
    if not diagnostics:
        return 0.0
    return float(np.mean([bool(item.get(key, False)) for item in diagnostics]))


def _sample_degree_sequences(
    *,
    checkpoint_path: str | Path,
    degree_cfg: dict[str, Any],
    num_samples: int,
    batch_size: int,
    seed: int,
    device: str,
    prior_mode: str = "model",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    model, vectorizer, _checkpoint = load_degree_vae_checkpoint(
        checkpoint_path, device=device
    )
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    summaries: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    remaining = int(num_samples)
    while remaining > 0:
        current_batch = min(int(batch_size), remaining)
        node_counts = None
        if str(degree_cfg.get("sample_num_nodes", "empirical")).lower() == "empirical":
            node_counts = [
                vectorizer.sample_empirical_node_count(rng)
                for _ in range(current_batch)
            ]
        with torch.no_grad():
            outputs = model.sample_outputs(
                current_batch,
                node_counts=node_counts,
                deterministic_node_count=bool(
                    degree_cfg.get("deterministic", False)
                ),
                prior_mode=prior_mode,
                device=next(model.parameters()).device,
            )
        batch = vectorizer.outputs_to_summaries(
            outputs,
            rng=rng,
            deterministic=bool(degree_cfg.get("deterministic", False)),
            sample_num_nodes=str(
                degree_cfg.get("sample_num_nodes", "empirical")
            ),
            max_resample=int(degree_cfg.get("max_resample", 200)),
            fallback=str(degree_cfg.get("fallback", "empirical_nearest_n")),
            parity_conditioned=bool(
                degree_cfg.get("parity_conditioned", True)
            ),
            max_parity_resample=int(
                degree_cfg.get("max_parity_resample", 32)
            ),
            include_diagnostics=True,
        )
        for summary in batch:
            diagnostics.append(dict(summary.pop("sampling_diagnostics")))
            summaries.append(summary)
        remaining -= current_batch
    return summaries, diagnostics


def _aggregate_posterior_sequences(
    *,
    checkpoint_path: str | Path,
    graphs: list[nx.Graph],
    degree_cfg: dict[str, Any],
    num_samples: int,
    batch_size: int,
    seed: int,
    device: str,
) -> list[list[int]]:
    """Sample z from the encoded training aggregate posterior and decode raw."""

    model, vectorizer, _checkpoint = load_degree_vae_checkpoint(
        checkpoint_path, device=device
    )
    x_np, targets_np = vectorizer.to_training_arrays(graphs)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model_device = next(model.parameters()).device
    sampled: list[list[int]] = []
    remaining = int(num_samples)
    while remaining > 0:
        current_batch = min(max(int(batch_size), 1), remaining)
        indices = rng.choice(len(graphs), size=current_batch, replace=True)
        batch_x = torch.as_tensor(
            x_np[indices], dtype=torch.float32, device=model_device
        )
        node_counts = torch.as_tensor(
            targets_np["num_nodes_count"][indices],
            dtype=torch.long,
            device=model_device,
        )
        with torch.no_grad():
            mu, logvar = model.encode(batch_x)
            z = model.reparameterize(mu, logvar)
            outputs = model.decode(z, node_counts)
        decoded = vectorizer.outputs_to_summaries(
            outputs,
            rng=rng,
            deterministic=False,
            sample_num_nodes="conditioned",
            max_resample=1,
            fallback="empirical_nearest_n",
            parity_conditioned=bool(
                degree_cfg.get("parity_conditioned", True)
            ),
            max_parity_resample=int(
                degree_cfg.get("max_parity_resample", 32)
            ),
            include_diagnostics=True,
        )
        for summary in decoded:
            sampled.append(
                [
                    int(degree)
                    for degree in summary["sampling_diagnostics"][
                        "first_raw_degree_sequence"
                    ]
                ]
            )
        remaining -= current_batch
    return sampled


def _posterior_reconstruction_sequences(
    *,
    checkpoint_path: str | Path,
    graphs: list[nx.Graph],
    batch_size: int,
    seed: int,
    device: str,
) -> list[list[int]]:
    """Decode q(z|h_D,n) means without repair or empirical fallback."""

    model, vectorizer, _checkpoint = load_degree_vae_checkpoint(
        checkpoint_path, device=device
    )
    x_np, targets_np = vectorizer.to_training_arrays(graphs)
    rng = np.random.default_rng(seed)
    reconstructed: list[list[int]] = []
    model_device = next(model.parameters()).device
    for start in range(0, len(graphs), max(int(batch_size), 1)):
        stop = min(start + max(int(batch_size), 1), len(graphs))
        batch_x = torch.as_tensor(
            x_np[start:stop], dtype=torch.float32, device=model_device
        )
        node_counts = torch.as_tensor(
            targets_np["num_nodes_count"][start:stop],
            dtype=torch.long,
            device=model_device,
        )
        with torch.no_grad():
            outputs = model.reconstruct_outputs(
                batch_x, node_counts, use_mean=True
            )
        decoded = vectorizer.outputs_to_summaries(
            outputs,
            rng=rng,
            deterministic=True,
            sample_num_nodes="conditioned",
            max_resample=1,
            fallback="empirical_nearest_n",
            parity_conditioned=False,
            include_diagnostics=True,
        )
        for summary in decoded:
            diagnostic = summary["sampling_diagnostics"]
            reconstructed.append(
                [
                    int(degree)
                    for degree in diagnostic["first_raw_degree_sequence"]
                ]
            )
    return reconstructed


def _quality_metrics(
    summaries: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
    constructor_cfg: dict[str, Any],
    *,
    seed: int,
    check_constructor: bool,
) -> dict[str, float]:
    sequences = [
        [int(degree) for degree in summary["degree_sequence"]]
        for summary in summaries
    ]
    graphical = [
        nx.is_graphical(sequence, method="eg") for sequence in sequences
    ]
    connected_feasible = [
        connected_feasible_degree_sequence(sequence) for sequence in sequences
    ]
    even_sum = [sum(sequence) % 2 == 0 for sequence in sequences]
    bounds = [
        len(sequence) == int(summary["num_nodes"])
        and all(0 <= degree < int(summary["num_nodes"]) for degree in sequence)
        for sequence, summary in zip(sequences, summaries)
    ]

    constructor_success: list[bool] = []
    if check_constructor:
        rng = np.random.default_rng(seed)
        require_connected = bool(constructor_cfg.get("ensure_connected", True))
        for summary in summaries:
            try:
                graph = construct_coarse_graph(summary, constructor_cfg, rng)
                assert_constructor_validity(
                    graph,
                    summary,
                    require_connected=require_connected,
                )
                constructor_success.append(True)
            except Exception:
                constructor_success.append(False)

    return {
        "raw_graphicality_rate": _mean_bool(
            diagnostics, "raw_graphical"
        ),
        "raw_connected_feasible_rate": _mean_bool(
            diagnostics, "raw_connected_feasible"
        ),
        "raw_even_degree_sum_rate": _mean_bool(
            diagnostics, "raw_even_degree_sum"
        ),
        "raw_degree_bounds_rate": _mean_bool(
            diagnostics, "raw_degree_bounds_valid"
        ),
        "repair_usage_rate": _mean_bool(diagnostics, "repair_used"),
        "fallback_usage_rate": _mean_bool(diagnostics, "fallback_used"),
        "accepted_without_postprocessing_rate": _mean_bool(
            diagnostics, "accepted_without_postprocessing"
        ),
        "mean_repair_l1_adjustment": float(
            np.mean(
                [
                    float(item.get("repair_l1_adjustment", 0.0))
                    for item in diagnostics
                ]
            )
        )
        if diagnostics
        else 0.0,
        "mean_sampling_attempts": float(
            np.mean(
                [
                    float(item.get("attempts_used", 0.0))
                    for item in diagnostics
                ]
            )
        )
        if diagnostics
        else 0.0,
        "mean_parity_redraws": float(
            np.mean(
                [
                    float(item.get("parity_redraws", 0.0))
                    for item in diagnostics
                ]
            )
        )
        if diagnostics
        else 0.0,
        "accepted_graphicality_rate": float(np.mean(graphical))
        if graphical
        else 0.0,
        "accepted_connected_feasible_rate": float(
            np.mean(connected_feasible)
        )
        if connected_feasible
        else 0.0,
        "accepted_even_degree_sum_rate": float(np.mean(even_sum))
        if even_sum
        else 0.0,
        "accepted_degree_bounds_rate": float(np.mean(bounds))
        if bounds
        else 0.0,
        "constructor_success_rate": float(np.mean(constructor_success))
        if constructor_success
        else float("nan"),
    }


def _compact_comparison(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "degree_kl": float(
            metrics["degree_marginal_kl_reference_to_candidate"]
        ),
        "degree_mmd": float(metrics["degree_histogram_mmd"]),
        "node_count_tv": float(metrics["node_count_total_variation"]),
        "edge_count_tv": float(metrics["edge_count_total_variation"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained DH-VAE against held-out degree sequences, "
            "including raw decoder quality and post-processing guarantees."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--max-reference-sequences", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-constructor-check", action="store_true")
    args = parser.parse_args()

    config = load_yaml(args.config)
    dataset_cfg = config.get("dataset", {}) or {}
    degree_cfg = config.get("degree_generator", {}) or {}
    eval_cfg = config.get("degree_evaluation", {}) or {}
    constructor_cfg = config.get("constructor", {}) or {}

    seed = int(args.seed if args.seed is not None else config.get("seed", 0))
    rng = np.random.default_rng(seed)
    dataset_name = str(dataset_cfg.get("name", "sbm"))
    dataset_root = str(dataset_cfg.get("root", "outputs/datasets"))
    dataset_config_path = dataset_cfg.get(
        "config_path", f"configs/datasets/{dataset_name}.yaml"
    )
    splits = load_dataset_splits(
        dataset_name,
        root=dataset_root,
        build_if_missing=bool(dataset_cfg.get("build_if_missing", False)),
        config_path=dataset_config_path,
    )
    train_graphs = list(splits["train"])
    test_graphs = list(splits["test"])
    if not train_graphs or not test_graphs:
        raise RuntimeError("Degree evaluation requires non-empty train and test splits.")

    reference_limit = (
        args.max_reference_sequences
        if args.max_reference_sequences is not None
        else eval_cfg.get("max_reference_sequences", 1024)
    )
    train_sequences = _subsample(
        _degree_sequences(train_graphs), reference_limit, rng
    )
    test_sequences = _subsample(
        _degree_sequences(test_graphs), reference_limit, rng
    )

    checkpoint_path = Path(
        args.checkpoint
        or degree_cfg.get(
            "checkpoint_path",
            "outputs/degree_generators/degree/checkpoint.pt",
        )
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing degree-generator checkpoint: {checkpoint_path}")

    num_samples = int(
        args.num_samples
        if args.num_samples is not None
        else eval_cfg.get("num_samples", 1024)
    )
    batch_size = int(
        args.batch_size
        if args.batch_size is not None
        else eval_cfg.get("batch_size", 256)
    )
    device = str(args.device or degree_cfg.get("device", "auto"))
    output_dir = ensure_dir(
        args.output_dir
        or eval_cfg.get(
            "output_dir",
            checkpoint_path.parent / "evaluation",
        )
    )

    summaries, diagnostics = _sample_degree_sequences(
        checkpoint_path=checkpoint_path,
        degree_cfg=degree_cfg,
        num_samples=num_samples,
        batch_size=batch_size,
        seed=seed,
        device=device,
        prior_mode="model",
    )
    generated_sequences = [
        [int(degree) for degree in summary["degree_sequence"]]
        for summary in summaries
    ]
    raw_prior_sequences = [
        [int(degree) for degree in item["first_raw_degree_sequence"]]
        for item in diagnostics
    ]
    _, standard_diagnostics = _sample_degree_sequences(
        checkpoint_path=checkpoint_path,
        degree_cfg=degree_cfg,
        num_samples=num_samples,
        batch_size=batch_size,
        seed=seed,
        device=device,
        prior_mode="standard_normal",
    )
    standard_normal_sequences = [
        [int(degree) for degree in item["first_raw_degree_sequence"]]
        for item in standard_diagnostics
    ]
    aggregate_posterior_sequences = _aggregate_posterior_sequences(
        checkpoint_path=checkpoint_path,
        graphs=train_graphs,
        degree_cfg=degree_cfg,
        num_samples=num_samples,
        batch_size=batch_size,
        seed=seed,
        device=device,
    )
    posterior_sequences = _posterior_reconstruction_sequences(
        checkpoint_path=checkpoint_path,
        graphs=test_graphs,
        batch_size=batch_size,
        seed=seed,
        device=device,
    )

    train_test = evaluate_degree_sequence_sets(
        test_sequences,
        train_sequences,
    )
    generated_test = evaluate_degree_sequence_sets(
        test_sequences,
        generated_sequences,
        train=train_sequences,
        degree_mmd_sigma=float(train_test["degree_mmd_sigma"]),
    )
    raw_prior_test = evaluate_degree_sequence_sets(
        test_sequences,
        raw_prior_sequences,
        train=train_sequences,
        degree_mmd_sigma=float(train_test["degree_mmd_sigma"]),
    )
    posterior_test = evaluate_degree_sequence_sets(
        test_sequences,
        posterior_sequences,
        train=train_sequences,
        degree_mmd_sigma=float(train_test["degree_mmd_sigma"]),
    )
    aggregate_posterior_test = evaluate_degree_sequence_sets(
        test_sequences,
        aggregate_posterior_sequences,
        train=train_sequences,
        degree_mmd_sigma=float(train_test["degree_mmd_sigma"]),
    )
    standard_normal_test = evaluate_degree_sequence_sets(
        test_sequences,
        standard_normal_sequences,
        train=train_sequences,
        degree_mmd_sigma=float(train_test["degree_mmd_sigma"]),
    )
    quality = _quality_metrics(
        summaries,
        diagnostics,
        constructor_cfg,
        seed=seed,
        check_constructor=not args.skip_constructor_check,
    )

    report = {
        "dataset": dataset_name,
        "checkpoint": str(checkpoint_path),
        "seed": seed,
        "protocol": {
            "num_generated_sequences": len(generated_sequences),
            "num_train_sequences": len(train_sequences),
            "num_test_sequences": len(test_sequences),
            "degree_kl_direction": "KL(test || candidate)",
            "degree_mmd_descriptor": (
                "per-graph normalized degree histogram with an RBF kernel"
            ),
            "degree_mmd_sigma": float(train_test["degree_mmd_sigma"]),
            "constructor_check": not args.skip_constructor_check,
            "postprocessing_note": (
                "Posterior reconstruction, aggregate-posterior, standard-normal, "
                "and learned-prior-raw metrics are measured before repair. "
                "Learned-prior-accepted metrics are measured after rejection "
                "sampling, repair, or fallback."
            ),
            "learned_prior_type": str(
                load_degree_vae_checkpoint(checkpoint_path, device=device)[0].prior_type
            ),
        },
        "comparison_table": {
            "train_to_test": _compact_comparison(train_test),
            "posterior_reconstruction_to_test": _compact_comparison(
                posterior_test
            ),
            "aggregate_posterior_to_test": _compact_comparison(
                aggregate_posterior_test
            ),
            "standard_normal_prior_to_test": _compact_comparison(
                standard_normal_test
            ),
            "learned_prior_raw_to_test": _compact_comparison(raw_prior_test),
            "learned_prior_accepted_to_test": _compact_comparison(
                generated_test
            ),
        },
        "dh_vae_quality": quality,
        "posterior_reconstruction_distribution": posterior_test,
        "aggregate_posterior_distribution": aggregate_posterior_test,
        "standard_normal_prior_distribution": standard_normal_test,
        "learned_prior_raw_distribution": raw_prior_test,
        "learned_prior_accepted_distribution": generated_test,
        # Backward-compatible aliases for downstream report readers.
        "prior_raw_distribution": raw_prior_test,
        "prior_accepted_distribution": generated_test,
        "train_test_baseline": train_test,
    }
    save_json(report, output_dir / "degree_evaluation.json")
    save_json(
        {
            "accepted_degree_sequences": generated_sequences,
            "raw_prior_degree_sequences": raw_prior_sequences,
            "standard_normal_prior_degree_sequences": standard_normal_sequences,
            "aggregate_posterior_degree_sequences": (
                aggregate_posterior_sequences
            ),
            "posterior_reconstruction_degree_sequences": posterior_sequences,
        },
        output_dir / "generated_degree_sequences.json",
    )

    print("\nDegree-sequence distribution matching (lower is better)")
    print(
        f"{'Comparison':<38} {'KL(test||candidate)':>20} "
        f"{'MMD':>12} {'Node TV':>12} {'Edge TV':>12}"
    )
    for name, metrics in report["comparison_table"].items():
        print(
            f"{name:<38} {metrics['degree_kl']:>20.6f} "
            f"{metrics['degree_mmd']:>12.6f} "
            f"{metrics['node_count_tv']:>12.6f} "
            f"{metrics['edge_count_tv']:>12.6f}"
        )

    print("\nDH-VAE feasibility and post-processing")
    for key, value in quality.items():
        print(f"{key}: {value:.6f}")
    print(f"\nSaved report to: {output_dir / 'degree_evaluation.json'}")
    print(
        "Saved generated sequences to: "
        f"{output_dir / 'generated_degree_sequences.json'}"
    )


if __name__ == "__main__":
    main()

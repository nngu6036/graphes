#!/usr/bin/env python
"""Evaluate DH-VAE before target-summary training or graph refinement."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import torch

from grapher.models.dhvae_hh.havel_hakimi import (
    assert_constructor_validity,
    construct_coarse_graph,
)
from grapher.models.dhvae_hh.typed_constructor import (
    TypedConstructionError,
    construct_typed_graph,
)
from grapher.data.io import load_dataset_splits
from grapher.rewiring_mlp.evaluation.degree_sequences import evaluate_degree_sequence_sets
from grapher.rewiring_mlp.evaluation.metrics import mmd_gaussian_emd
from grapher.models.dhvae_hh.degree_vae import (
    connected_feasible_degree_sequence,
    load_degree_vae_checkpoint,
)
from grapher.rewiring_mlp.molecular.typed_invariants import (
    TypedInvariant,
    extract_typed_invariant,
    typed_invariant_errors,
)
from grapher.models.dhvae_hh.typed_degree_vae import (
    TypedSignatureVectorizer,
    load_typed_signature_checkpoint,
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


def _slice_model_outputs(
    outputs: dict[str, Any], index: int
) -> dict[str, Any]:
    """Return one row of a batched DH-VAE output dictionary."""

    row: dict[str, Any] = {}
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            row[key] = value[index : index + 1]
        else:
            array = np.asarray(value)
            row[key] = array[index : index + 1]
    return row


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
    """Draw prior samples without letting one infeasible draw abort evaluation.

    Evaluation must measure native failure probability rather than silently
    replacing invalid model draws.  When the configured production fallback is
    ``error``, a temporary empirical placeholder is used only to recover the
    first raw sequence and diagnostics from that row; the placeholder is *not*
    included in the accepted distribution.
    """

    model, vectorizer, _checkpoint = load_degree_vae_checkpoint(
        checkpoint_path, device=device
    )
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    summaries: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    remaining = int(num_samples)
    configured_fallback = str(degree_cfg.get("fallback", "empirical_nearest_n"))
    while remaining > 0:
        current_batch = min(int(batch_size), remaining)
        node_counts = None
        if str(degree_cfg.get("sample_num_nodes", "empirical")).lower() == "empirical":
            node_counts = [
                vectorizer.sample_empirical_node_count(rng)
                for _ in range(current_batch)
            ]
        edge_counts = None
        if (
            node_counts is not None
            and str(degree_cfg.get("sample_num_edges", "model")).lower() == "empirical"
        ):
            edge_counts = [
                vectorizer.sample_empirical_edge_count(int(n), rng) for n in node_counts
            ]
        with torch.no_grad():
            outputs = model.sample_outputs(
                current_batch,
                node_counts=node_counts,
                edge_counts=edge_counts,
                deterministic_node_count=bool(degree_cfg.get("deterministic", False)),
                deterministic_edge_count=bool(degree_cfg.get("deterministic", False)),
                prior_mode=prior_mode,
                device=next(model.parameters()).device,
            )

        # Process rows independently.  ``outputs_to_summaries`` raises on a
        # reject-only failure, so batching the post-processor made a single rare
        # Ego-small failure terminate a 10k-sample diagnostic run.
        for index in range(current_batch):
            row_outputs = _slice_model_outputs(outputs, index)
            try:
                row_batch = vectorizer.outputs_to_summaries(
                    row_outputs,
                    rng=rng,
                    deterministic=bool(degree_cfg.get("deterministic", False)),
                    sample_num_nodes=str(
                        degree_cfg.get("sample_num_nodes", "empirical")
                    ),
                    sample_num_edges=str(
                        degree_cfg.get("sample_num_edges", "model")
                    ),
                    exact_degree_sum_conditioning=bool(
                        degree_cfg.get("exact_degree_sum_conditioning", True)
                    ),
                    max_resample=int(degree_cfg.get("max_resample", 200)),
                    fallback=configured_fallback,
                    parity_conditioned=bool(
                        degree_cfg.get("parity_conditioned", True)
                    ),
                    max_parity_resample=int(
                        degree_cfg.get("max_parity_resample", 32)
                    ),
                    postprocess_policy=str(
                        degree_cfg.get("postprocess_policy", "repair")
                    ),
                    include_diagnostics=True,
                )
                summary = row_batch[0]
                diagnostic = dict(summary.pop("sampling_diagnostics"))
                diagnostic["native_sampling_failed"] = False
                diagnostic["evaluation_placeholder_fallback_used"] = False
                summaries.append(summary)
                diagnostics.append(diagnostic)
            except RuntimeError as exc:
                if configured_fallback.lower() != "error":
                    raise
                # Re-run the *same model output* only to expose its raw first
                # sequence.  The empirical fallback is a diagnostic placeholder
                # and is never counted as an accepted model sample.
                placeholder = vectorizer.outputs_to_summaries(
                    row_outputs,
                    rng=rng,
                    deterministic=bool(degree_cfg.get("deterministic", False)),
                    sample_num_nodes=str(
                        degree_cfg.get("sample_num_nodes", "empirical")
                    ),
                    sample_num_edges=str(
                        degree_cfg.get("sample_num_edges", "model")
                    ),
                    exact_degree_sum_conditioning=bool(
                        degree_cfg.get("exact_degree_sum_conditioning", True)
                    ),
                    max_resample=int(degree_cfg.get("max_resample", 200)),
                    fallback="empirical_nearest_n",
                    parity_conditioned=bool(
                        degree_cfg.get("parity_conditioned", True)
                    ),
                    max_parity_resample=int(
                        degree_cfg.get("max_parity_resample", 32)
                    ),
                    postprocess_policy=str(
                        degree_cfg.get("postprocess_policy", "repair")
                    ),
                    include_diagnostics=True,
                )[0]
                diagnostic = dict(placeholder.pop("sampling_diagnostics"))
                diagnostic["native_sampling_failed"] = True
                diagnostic["native_sampling_error"] = str(exc)
                diagnostic["evaluation_placeholder_fallback_used"] = True
                # Do not report the evaluation-only placeholder as a production
                # fallback.
                diagnostic["fallback_used"] = False
                diagnostics.append(diagnostic)
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
        edge_counts = torch.as_tensor(
            targets_np["num_edges_count"][indices],
            dtype=torch.long,
            device=model_device,
        )
        with torch.no_grad():
            mu, logvar = model.encode(batch_x)
            z = model.reparameterize(mu, logvar)
            outputs = model.decode(z, node_counts, edge_counts)
        decoded = vectorizer.outputs_to_summaries(
            outputs,
            rng=rng,
            deterministic=False,
            sample_num_nodes="conditioned",
            sample_num_edges="conditioned",
            exact_degree_sum_conditioning=bool(
                degree_cfg.get("exact_degree_sum_conditioning", True)
            ),
            max_resample=1,
            fallback="empirical_nearest_n",
            parity_conditioned=bool(degree_cfg.get("parity_conditioned", True)),
            max_parity_resample=int(degree_cfg.get("max_parity_resample", 32)),
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
        edge_counts = torch.as_tensor(
            targets_np["num_edges_count"][start:stop],
            dtype=torch.long,
            device=model_device,
        )
        with torch.no_grad():
            outputs = model.reconstruct_outputs(
                batch_x, node_counts, edge_counts=edge_counts, use_mean=True
            )
        decoded = vectorizer.outputs_to_summaries(
            outputs,
            rng=rng,
            deterministic=True,
            sample_num_nodes="conditioned",
            sample_num_edges="conditioned",
            exact_degree_sum_conditioning=True,
            max_resample=1,
            fallback="empirical_nearest_n",
            parity_conditioned=False,
            include_diagnostics=True,
        )
        for summary in decoded:
            diagnostic = summary["sampling_diagnostics"]
            reconstructed.append(
                [int(degree) for degree in diagnostic["first_raw_degree_sequence"]]
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
        [int(degree) for degree in summary["degree_sequence"]] for summary in summaries
    ]
    graphical = [nx.is_graphical(sequence, method="eg") for sequence in sequences]
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
        "native_sampling_failure_rate": _mean_bool(
            diagnostics, "native_sampling_failed"
        ),
        "num_attempted_prior_draws": float(len(diagnostics)),
        "num_native_returned_sequences": float(len(summaries)),
        "raw_graphicality_rate": _mean_bool(diagnostics, "raw_graphical"),
        "raw_connected_feasible_rate": _mean_bool(
            diagnostics, "raw_connected_feasible"
        ),
        "raw_even_degree_sum_rate": _mean_bool(diagnostics, "raw_even_degree_sum"),
        "raw_degree_bounds_rate": _mean_bool(diagnostics, "raw_degree_bounds_valid"),
        "exact_degree_sum_conditioned_rate": _mean_bool(
            diagnostics, "exact_degree_sum_conditioned"
        ),
        "raw_edge_count_matches_target_rate": _mean_bool(
            diagnostics, "raw_edge_count_matches_target"
        ),
        "repair_usage_rate": _mean_bool(diagnostics, "repair_used"),
        "fallback_usage_rate": _mean_bool(diagnostics, "fallback_used"),
        "accepted_without_postprocessing_rate": _mean_bool(
            diagnostics, "accepted_without_postprocessing"
        ),
        "mean_repair_l1_adjustment": float(
            np.mean(
                [float(item.get("repair_l1_adjustment", 0.0)) for item in diagnostics]
            )
        )
        if diagnostics
        else 0.0,
        "mean_sampling_attempts": float(
            np.mean([float(item.get("attempts_used", 0.0)) for item in diagnostics])
        )
        if diagnostics
        else 0.0,
        "mean_parity_redraws": float(
            np.mean([float(item.get("parity_redraws", 0.0)) for item in diagnostics])
        )
        if diagnostics
        else 0.0,
        "accepted_graphicality_rate": float(np.mean(graphical)) if graphical else 0.0,
        "accepted_connected_feasible_rate": float(np.mean(connected_feasible))
        if connected_feasible
        else 0.0,
        "accepted_even_degree_sum_rate": float(np.mean(even_sum)) if even_sum else 0.0,
        "accepted_degree_bounds_rate": float(np.mean(bounds)) if bounds else 0.0,
        "constructor_success_rate": float(np.mean(constructor_success))
        if constructor_success
        else float("nan"),
    }


def _compact_comparison(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "degree_kl": float(metrics["degree_marginal_kl_reference_to_candidate"]),
        "degree_mmd_graphrnn": float(metrics["degree_histogram_mmd_graphrnn"]),
        "degree_mmd": float(metrics["degree_histogram_mmd"]),
        "node_count_tv": float(metrics["node_count_total_variation"]),
        "edge_count_tv": float(metrics["edge_count_total_variation"]),
    }


def _shape_comparison(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "mean_degree_abs_error": float(metrics.get("mean_degree_mean_abs_error", 0.0)),
        "degree_variance_abs_error": float(
            metrics.get("degree_variance_mean_abs_error", 0.0)
        ),
        "degree_second_moment_abs_error": float(
            metrics.get("degree_second_moment_mean_abs_error", 0.0)
        ),
        "max_degree_tv": float(metrics.get("max_degree_total_variation", 0.0)),
        "wedge_count_tv": float(metrics.get("wedge_count_total_variation", 0.0)),
    }


def _typed_histogram_matrix(
    invariants: list[TypedInvariant],
    vectorizer: TypedSignatureVectorizer,
) -> np.ndarray:
    if not invariants:
        return np.zeros((0, vectorizer.signature_dim), dtype=np.float64)
    return np.stack(
        [vectorizer.invariant_histogram(invariant) for invariant in invariants]
    ).astype(np.float64)


def _evaluate_typed_prior(
    *,
    checkpoint_path: Path,
    degree_cfg: dict[str, Any],
    constructor_cfg: dict[str, Any],
    train_graphs: list[nx.Graph],
    test_graphs: list[nx.Graph],
    reference_limit: int | None,
    num_samples: int,
    batch_size: int,
    seed: int,
    device: str,
    output_dir: Path,
    check_constructor: bool,
) -> None:
    """Evaluate the joint typed-signature prior and exact constructor."""

    model, vectorizer, checkpoint = load_typed_signature_checkpoint(
        checkpoint_path,
        device=device,
    )
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model_device = next(model.parameters()).device
    summaries: list[dict[str, Any]] = []
    sampling_diagnostics: list[dict[str, Any]] = []
    remaining = int(num_samples)
    while remaining > 0:
        current_batch = min(max(int(batch_size), 1), remaining)
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
                device=model_device,
            )
        decoded = vectorizer.outputs_to_summaries(
            outputs,
            rng=rng,
            deterministic=bool(degree_cfg.get("deterministic", False)),
            max_resample=int(degree_cfg.get("max_resample", 1000)),
            fallback=str(degree_cfg.get("fallback", "error")),
            include_diagnostics=True,
        )
        for summary in decoded:
            sampling_diagnostics.append(dict(summary.pop("sampling_diagnostics")))
            summaries.append(summary)
        remaining -= current_batch

    generated_invariants = [
        TypedInvariant.from_dict(summary["typed_invariant"]) for summary in summaries
    ]
    train_reference = _subsample(train_graphs, reference_limit, rng)
    test_reference = _subsample(test_graphs, reference_limit, rng)
    train_invariants = [
        extract_typed_invariant(
            graph,
            edge_types=vectorizer.vocabulary.edge_types,
            node_attribute=vectorizer.vocabulary.node_attribute,
            edge_attribute=vectorizer.vocabulary.edge_attribute,
        )
        for graph in train_reference
    ]
    test_invariants = [
        extract_typed_invariant(
            graph,
            edge_types=vectorizer.vocabulary.edge_types,
            node_attribute=vectorizer.vocabulary.node_attribute,
            edge_attribute=vectorizer.vocabulary.edge_attribute,
        )
        for graph in test_reference
    ]
    generated_histograms = _typed_histogram_matrix(generated_invariants, vectorizer)
    train_histograms = _typed_histogram_matrix(train_invariants, vectorizer)
    test_histograms = _typed_histogram_matrix(test_invariants, vectorizer)

    constructor_diagnostics: list[dict[str, Any]] = []
    construction_success: list[bool] = []
    if check_constructor:
        for invariant in generated_invariants:
            try:
                _graph, diagnostic = construct_typed_graph(
                    invariant,
                    constructor_cfg,
                    rng,
                )
                construction_success.append(True)
                constructor_diagnostics.append(dict(diagnostic))
            except TypedConstructionError as exc:
                construction_success.append(False)
                constructor_diagnostics.append(dict(exc.diagnostics))

    feasibility_errors = [
        typed_invariant_errors(
            invariant,
            require_connected=vectorizer.require_connected,
            max_ordinary_degree=vectorizer.max_ordinary_degree,
            max_weighted_valence=vectorizer.max_weighted_valence,
        )
        for invariant in generated_invariants
    ]
    generated_sequences = [
        invariant.degree_sequence for invariant in generated_invariants
    ]
    train_sequences = [invariant.degree_sequence for invariant in train_invariants]
    test_sequences = [invariant.degree_sequence for invariant in test_invariants]
    degree_baseline = evaluate_degree_sequence_sets(test_sequences, train_sequences)
    degree_distribution = evaluate_degree_sequence_sets(
        test_sequences,
        generated_sequences,
        train=train_sequences,
        degree_mmd_sigma=float(degree_baseline["degree_mmd_sigma"]),
    )

    failure_reasons: dict[str, int] = {}
    for diagnostic in constructor_diagnostics:
        reason = diagnostic.get("failure_reason")
        if reason:
            failure_reasons[str(reason)] = failure_reasons.get(str(reason), 0) + 1
    attempts = [float(item.get("attempts_used", 0.0)) for item in sampling_diagnostics]
    report = {
        "format": "typed_signature_prior_evaluation_v1",
        "dataset": checkpoint.get("config", {}).get("dataset", {}).get("name"),
        "checkpoint": str(checkpoint_path),
        "seed": int(seed),
        "protocol": {
            "num_generated_invariants": len(generated_invariants),
            "num_train_invariants": len(train_invariants),
            "num_test_invariants": len(test_invariants),
            "signature_vocabulary_source": "training_split_checkpoint",
            "typed_histogram_kernel": "Gaussian Earth-Mover",
            "constructor_check": bool(check_constructor),
        },
        "typed_signature_distribution": {
            "train_to_test_mmd": mmd_gaussian_emd(test_histograms, train_histograms),
            "generated_to_test_mmd": mmd_gaussian_emd(
                test_histograms,
                generated_histograms,
            ),
        },
        "aggregate_degree_distribution": degree_distribution,
        "feasibility": {
            "first_draw_feasible_rate": _mean_bool(
                sampling_diagnostics,
                "first_raw_feasible",
            ),
            "accepted_necessary_feasibility_rate": float(
                np.mean([not errors for errors in feasibility_errors])
            )
            if feasibility_errors
            else 0.0,
            "fallback_usage_rate": _mean_bool(
                sampling_diagnostics,
                "fallback_used",
            ),
            "mean_sampling_attempts": float(np.mean(attempts)) if attempts else 0.0,
            "constructor_success_rate": float(np.mean(construction_success))
            if construction_success
            else (float("nan") if not check_constructor else 0.0),
            "mean_constructor_restarts": float(
                np.mean(
                    [
                        float(item.get("restarts", 0.0))
                        for item in constructor_diagnostics
                    ]
                )
            )
            if constructor_diagnostics
            else 0.0,
            "mean_constructor_backtracks": float(
                np.mean(
                    [
                        float(item.get("backtracks", 0.0))
                        for item in constructor_diagnostics
                    ]
                )
            )
            if constructor_diagnostics
            else 0.0,
            "constructor_failure_reasons": failure_reasons,
        },
        "sampling_diagnostics": sampling_diagnostics,
        "constructor_diagnostics": constructor_diagnostics,
    }
    save_json(report, output_dir / "degree_evaluation.json")
    save_json(
        {
            "typed_invariants": [value.to_dict() for value in generated_invariants],
            "aggregate_degree_sequences": generated_sequences,
        },
        output_dir / "generated_typed_invariants.json",
    )
    print("Typed-signature prior evaluation", flush=True)
    for key, value in report["feasibility"].items():
        print(f"  {key}: {value}", flush=True)
    print(f"Saved report to: {output_dir / 'degree_evaluation.json'}", flush=True)


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
        raise RuntimeError(
            "Degree evaluation requires non-empty train and test splits."
        )

    reference_limit = (
        args.max_reference_sequences
        if args.max_reference_sequences is not None
        else eval_cfg.get("max_reference_sequences", 1024)
    )
    train_sequences = _subsample(_degree_sequences(train_graphs), reference_limit, rng)
    test_sequences = _subsample(_degree_sequences(test_graphs), reference_limit, rng)

    checkpoint_path = Path(
        args.checkpoint
        or degree_cfg.get(
            "checkpoint_path",
            "outputs/degree_generators/degree/checkpoint.pt",
        )
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing degree-generator checkpoint: {checkpoint_path}"
        )

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

    generator_type = str(degree_cfg.get("type", "degree_histogram_vae")).lower()
    if generator_type in {
        "typed_degree_histogram_vae",
        "typed_signature_histogram_vae",
        "typed_signature_vae",
    }:
        _evaluate_typed_prior(
            checkpoint_path=checkpoint_path,
            degree_cfg=degree_cfg,
            constructor_cfg=constructor_cfg,
            train_graphs=train_graphs,
            test_graphs=test_graphs,
            reference_limit=reference_limit,
            num_samples=num_samples,
            batch_size=batch_size,
            seed=seed,
            device=device,
            output_dir=output_dir,
            check_constructor=not args.skip_constructor_check,
        )
        return
    if generator_type not in {
        "degree_histogram_vae",
        "degree_vae",
        "vae",
        "learned",
    }:
        raise ValueError(f"Unknown degree_generator.type: {generator_type!r}")

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
        [int(degree) for degree in summary["degree_sequence"]] for summary in summaries
    ]
    raw_prior_sequences = [
        [int(degree) for degree in item["first_raw_degree_sequence"]]
        for item in diagnostics
        if item.get("first_raw_degree_sequence")
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
        if item.get("first_raw_degree_sequence")
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
    empirical_rng = np.random.default_rng(seed + 991)
    empirical_train_sequences = [
        list(train_sequences[int(empirical_rng.integers(0, len(train_sequences)))])
        for _ in range(int(num_samples))
    ]
    empirical_train_test = evaluate_degree_sequence_sets(
        test_sequences,
        empirical_train_sequences,
        train=train_sequences,
        degree_mmd_sigma=float(train_test["degree_mmd_sigma"]),
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
            "num_attempted_prior_draws": len(diagnostics),
            "native_sampling_failures": int(
                sum(bool(item.get("native_sampling_failed", False)) for item in diagnostics)
            ),
            "num_train_sequences": len(train_sequences),
            "num_test_sequences": len(test_sequences),
            "degree_kl_direction": "KL(test || candidate)",
            "degree_mmd_descriptor": (
                "per-graph normalized degree histogram with a Gaussian "
                "Earth-Mover kernel"
            ),
            "generic_degree_mmd_protocol": "graphrnn",
            "generic_degree_mmd_sigma": 1.0,
            "adaptive_degree_mmd_sigma": float(train_test["degree_mmd_sigma"]),
            # Backward-compatible name for readers that expect the adaptive
            # diagnostic bandwidth under this key.
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
            "train_empirical_resample_to_test": _compact_comparison(empirical_train_test),
            "posterior_reconstruction_to_test": _compact_comparison(posterior_test),
            "aggregate_posterior_to_test": _compact_comparison(
                aggregate_posterior_test
            ),
            "standard_normal_prior_to_test": _compact_comparison(standard_normal_test),
            "learned_prior_raw_to_test": _compact_comparison(raw_prior_test),
            "learned_prior_accepted_to_test": _compact_comparison(generated_test),
        },
        "degree_shape_table": {
            "train_to_test": _shape_comparison(train_test),
            "train_empirical_resample_to_test": _shape_comparison(empirical_train_test),
            "posterior_reconstruction_to_test": _shape_comparison(posterior_test),
            "aggregate_posterior_to_test": _shape_comparison(aggregate_posterior_test),
            "standard_normal_prior_to_test": _shape_comparison(standard_normal_test),
            "learned_prior_raw_to_test": _shape_comparison(raw_prior_test),
            "learned_prior_accepted_to_test": _shape_comparison(generated_test),
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
        "train_empirical_resample_distribution": empirical_train_test,
    }
    save_json(report, output_dir / "degree_evaluation.json")
    save_json(
        {
            "accepted_degree_sequences": generated_sequences,
            "train_empirical_resample_degree_sequences": empirical_train_sequences,
            "raw_prior_degree_sequences": raw_prior_sequences,
            "standard_normal_prior_degree_sequences": standard_normal_sequences,
            "aggregate_posterior_degree_sequences": (aggregate_posterior_sequences),
            "posterior_reconstruction_degree_sequences": posterior_sequences,
        },
        output_dir / "generated_degree_sequences.json",
    )

    print("\nDegree-sequence distribution matching (lower is better)")
    print(
        f"{'Comparison':<38} {'KL(test||candidate)':>20} "
        f"{'Generic MMD':>12} {'Adaptive MMD':>13} "
        f"{'Node TV':>12} {'Edge TV':>12}"
    )
    for name, metrics in report["comparison_table"].items():
        print(
            f"{name:<38} {metrics['degree_kl']:>20.6f} "
            f"{metrics['degree_mmd_graphrnn']:>12.6f} "
            f"{metrics['degree_mmd']:>13.6f} "
            f"{metrics['node_count_tv']:>12.6f} "
            f"{metrics['edge_count_tv']:>12.6f}"
        )
    print(
        "  Generic MMD uses the same GraphRNN/GDSS/HOG-Diff degree "
        "protocol as evaluate_graph_generation_report.py (Gaussian EMD, "
        "sigma=1.0)."
    )
    print(
        "  Adaptive MMD retains the DH-VAE median-bandwidth diagnostic "
        "for backward compatibility."
    )

    print("\nDegree-shape diagnostics (lower is better)")
    print(
        f"{'Comparison':<38} {'MeanDegErr':>12} {'VarErr':>12} "
        f"{'SecondMomErr':>14} {'MaxDegTV':>12} {'WedgeTV':>12}"
    )
    for name, metrics in report["degree_shape_table"].items():
        print(
            f"{name:<38} {metrics['mean_degree_abs_error']:>12.6f} "
            f"{metrics['degree_variance_abs_error']:>12.6f} "
            f"{metrics['degree_second_moment_abs_error']:>14.6f} "
            f"{metrics['max_degree_tv']:>12.6f} "
            f"{metrics['wedge_count_tv']:>12.6f}"
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

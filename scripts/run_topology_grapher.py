#!/usr/bin/env python
from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import torch

from grapher.models.dhvae_hh.havel_hakimi import (
    assert_constructor_validity,
    construct_coarse_graph,
)
from grapher.data.io import load_dataset_splits
from grapher.rewiring_mlp.evaluation.metrics import (
    degree_preservation_rate,
    degree_target_match_rate,
    evaluate_graph_sets,
)
from grapher.rewiring_mlp.evaluation.studies import aggregate_pipeline_diagnostics
from grapher.models.dhvae_hh.degree_sampler import (
    EmpiricalDegreeSampler,
    build_degree_sampler,
)
from grapher.properties.summary import configure_orca_executable
from grapher.rewiring_mlp.generic.model import load_topology_checkpoint
from grapher.rewiring_mlp.generic.refiner import (
    TopologyRefinerConfig,
    refine_graph_with_topology_predictions,
)
from grapher.utils.io import ensure_dir, load_yaml, save_json, save_pickle


def _oracle_degree_summary(graph: nx.Graph) -> dict[str, Any]:
    sequence = sorted((int(degree) for _, degree in graph.degree()), reverse=True)
    n = len(sequence)
    m = int(sum(sequence) // 2)
    histogram = np.bincount(
        sequence,
        minlength=max(max(sequence, default=0) + 1, 1),
    ).astype(np.float64)
    histogram /= max(float(histogram.sum()), 1.0)
    return {
        "num_nodes": n,
        "num_edges": m,
        "degree_sequence": sequence,
        "degree_hist": histogram,
        "density": 2.0 * m / (n * (n - 1)) if n > 1 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate generic graph topologies with DH-VAE, connected "
            "Havel-Hakimi construction, and structural-summary GraphER rewiring."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-generate", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    run_started = time.perf_counter()

    config = load_yaml(args.config)
    pipeline_stage = str(
        (config.get("pipeline", {}) or {}).get("stage", "topology")
    ).lower()
    if pipeline_stage != "topology":
        raise ValueError("run_topology_grapher.py requires pipeline.stage: topology.")
    if config.get("categorical_state") or config.get("molecular_generation"):
        raise ValueError("The topology generator accepts generic datasets only.")
    seed = int(args.seed if args.seed is not None else config.get("seed", 0))
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    dataset_cfg = dict(config.get("dataset", {}) or {})
    splits = load_dataset_splits(
        str(dataset_cfg.get("name", "sbm")),
        root=dataset_cfg.get("root", "outputs/datasets"),
        build_if_missing=bool(dataset_cfg.get("build_if_missing", True)),
        config_path=dataset_cfg.get("config_path"),
    )
    train_graphs = list(splits["train"])
    reference_graphs = list(splits.get("test", []))
    generation_cfg = dict(config.get("generation", {}) or {})
    num_generate = int(
        args.num_generate
        if args.num_generate is not None
        else generation_cfg.get("num_generate", len(reference_graphs))
    )
    if num_generate <= 0:
        raise ValueError("num_generate must be positive.")

    predictor_cfg = dict(config.get("topology_predictor", {}) or {})
    checkpoint_path = args.checkpoint or predictor_cfg.get("checkpoint_path")
    if not checkpoint_path:
        raise ValueError("topology_predictor.checkpoint_path is required.")
    device = args.device or predictor_cfg.get("device", "auto")
    model, graphlet_basis, summary_config, checkpoint = load_topology_checkpoint(
        checkpoint_path,
        device=device,
    )
    model_device = next(model.parameters()).device
    predictor_report = checkpoint.get("report", {}) or {}
    predictor_graphlet_error = predictor_report.get("val_graphlet_mae")
    if predictor_graphlet_error is None:
        raise ValueError(
            "The topology checkpoint is missing held-out val_graphlet_mae; "
            "generate from a checkpoint selected by train_topology_grapher.py."
        )
    predictor_clustering_error = predictor_report.get("val_clustering_mae")
    predictor_orbit_log_error = predictor_report.get("val_orbit_log_mae")

    degree_source = str(generation_cfg.get("degree_source", "learned")).lower()
    degree_cfg = dict(config.get("degree_generator", {}) or {})
    degree_type = str(degree_cfg.get("type", "degree_histogram_vae")).lower()
    if "typed" in degree_type:
        raise ValueError("The generic topology stage requires the ordinary DH-VAE.")
    degree_sampler = None
    if degree_source in {"learned", "degree_vae"}:
        if str(degree_cfg.get("postprocess_policy", "")).lower() != "reject_only":
            raise ValueError(
                "Learned topology generation requires degree_generator."
                "postprocess_policy: reject_only."
            )
        if str(degree_cfg.get("fallback", "")).lower() != "error":
            raise ValueError(
                "Learned topology generation requires degree_generator."
                "fallback: error."
            )
        degree_cfg["enabled"] = True
        degree_sampler = build_degree_sampler(degree_cfg, train_graphs, seed=seed)
    elif degree_source in {"empirical", "train_empirical"}:
        degree_sampler = EmpiricalDegreeSampler.fit_from_graphs(
            train_graphs,
            seed=seed,
        )
    elif degree_source not in {"oracle", "test_oracle"}:
        raise ValueError(f"Unknown generation.degree_source: {degree_source!r}")

    constructor_cfg = dict(config.get("constructor", {}) or {})
    if str(constructor_cfg.get("type", "havel_hakimi")).lower() != "havel_hakimi":
        raise ValueError(
            "The generic topology stage requires Havel-Hakimi construction."
        )
    if not bool(constructor_cfg.get("ensure_connected", True)):
        raise ValueError("Topology generation requires constructor.ensure_connected.")
    refiner_cfg = dict(config.get("topology_refiner", {}) or {})
    refiner_settings = TopologyRefinerConfig.from_dict(refiner_cfg)
    if not refiner_settings.preserve_connectivity:
        raise ValueError(
            "Topology generation requires topology_refiner.preserve_connectivity."
        )
    coarse_graphs: list[nx.Graph] = []
    refined_graphs: list[nx.Graph] = []
    target_degree_sequences: list[list[int]] = []
    traces: list[list[dict[str, Any]]] = []
    graph_runtimes: list[float] = []
    pipeline_records: list[dict[str, Any]] = []
    max_attempts_per_graph = int(
        generation_cfg.get("max_attempts_per_graph", 8)
    )
    if max_attempts_per_graph <= 0:
        raise ValueError("generation.max_attempts_per_graph must be positive.")

    for index in range(num_generate):
        graph_started = time.perf_counter()
        generation_rejections: Counter[str] = Counter()
        for generation_attempt in range(1, max_attempts_per_graph + 1):
            try:
                if degree_source in {"oracle", "test_oracle"}:
                    if not reference_graphs:
                        raise ValueError(
                            "Oracle degree generation requires test graphs."
                        )
                    degree_summary = _oracle_degree_summary(
                        reference_graphs[index % len(reference_graphs)]
                    )
                else:
                    if degree_sampler is None:
                        raise RuntimeError("Degree sampler was not initialized.")
                    degree_summary = degree_sampler.sample(rng)
            except RuntimeError:
                generation_rejections["degree_prior_rejected"] += 1
                continue

            try:
                coarse = construct_coarse_graph(
                    degree_summary,
                    constructor_cfg,
                    rng,
                )
                assert_constructor_validity(
                    coarse,
                    degree_summary,
                    require_connected=True,
                )
            except (ValueError, RuntimeError, AssertionError):
                generation_rejections["constructor_rejected"] += 1
                continue
            break
        else:
            raise RuntimeError(
                "Topology generation exhausted "
                f"{max_attempts_per_graph} attempts for graph {index}; "
                f"rejections={dict(generation_rejections)}."
            )

        refined, trace = refine_graph_with_topology_predictions(
            coarse,
            model=model,
            graphlet_basis=graphlet_basis,
            summary_config=summary_config,
            refiner_config=refiner_settings,
            device=model_device,
            rng=rng,
            return_trace=True,
        )
        runtime = float(time.perf_counter() - graph_started)
        coarse_graphs.append(coarse)
        refined_graphs.append(refined)
        target_degree_sequences.append(
            [int(value) for value in degree_summary["degree_sequence"]]
        )
        traces.append(trace)
        graph_runtimes.append(runtime)
        sampling_diagnostics = dict(
            degree_summary.get("sampling_diagnostics", {}) or {}
        )

        decision_rows = [row for row in trace if "num_proposals" in row]
        proposals = sum(int(row.get("num_proposals", 0)) for row in decision_rows)
        passes = sum(
            int(row.get("num_valid_candidates", 0)) for row in decision_rows
        )
        accepted = sum(bool(row.get("accepted")) for row in trace)
        terminal_rows = [
            row for row in trace if bool(row.get("terminal_stop", False))
        ]
        terminal_stop = bool(terminal_rows)
        prediction_calls = max(
            (int(row.get("prediction_calls", 0)) for row in trace),
            default=0,
        )
        prediction_horizon_rows = [
            row for row in trace if bool(row.get("prediction_refreshed", False))
        ]
        realized_prediction_horizons = [
            int(row["prediction_horizon"])
            for row in prediction_horizon_rows
            if "prediction_horizon" in row
        ]
        plateau_refreshes = sum(
            row.get("reason") == "prediction_plateau_refresh" for row in trace
        )
        rejection_reasons: Counter[str] = Counter()
        rejection_reasons.update(generation_rejections)
        for row in decision_rows:
            rejection_reasons.update(
                {
                    str(key): int(value)
                    for key, value in (
                        row.get("candidate_rejection_reasons", {}) or {}
                    ).items()
                }
            )
        pipeline_record = {
            "pipeline_mode": "topology",
            "graphlet_error": float(predictor_graphlet_error),
            "clustering_error": (
                float(predictor_clustering_error)
                if predictor_clustering_error is not None
                else None
            ),
            "orbit_log_error": (
                float(predictor_orbit_log_error)
                if predictor_orbit_log_error is not None
                else None
            ),
            "invariant_feasible": 1.0,
            "constructor_success": 1.0,
            "accepted_swaps": accepted,
            "runtime_seconds": runtime,
            "fallback_used": float(
                bool(sampling_diagnostics.get("fallback_used", False))
                or bool(sampling_diagnostics.get("repair_used", False))
            ),
            "degree_raw_graphical": float(
                bool(sampling_diagnostics.get("raw_graphical", True))
            ),
            "degree_raw_connected_feasible": float(
                bool(
                    sampling_diagnostics.get(
                        "raw_connected_feasible",
                        True,
                    )
                )
            ),
            "degree_sampling_attempts": int(
                sampling_diagnostics.get("attempts_used", 1)
            ),
            "degree_repair_used": float(
                bool(sampling_diagnostics.get("repair_used", False))
            ),
            "degree_repair_l1": int(
                sampling_diagnostics.get("repair_l1_adjustment", 0)
            ),
            "candidate_proposals": proposals,
            "candidate_passes": passes,
            "candidate_pass_rate": float(passes / max(proposals, 1)),
            "prediction_calls": prediction_calls,
            "accepted_swaps_per_prediction": float(
                accepted / max(prediction_calls, 1)
            ),
            "mean_realized_prediction_horizon": (
                float(np.mean(realized_prediction_horizons))
                if realized_prediction_horizons
                else 0.0
            ),
            "plateau_refreshes": plateau_refreshes,
            "stopped": float(terminal_stop),
            "stop_opportunities": 1,
            "stop_rate": float(terminal_stop),
            "stop_reason": (
                str(terminal_rows[-1].get("reason"))
                if terminal_rows
                else "maximum_accepted_steps"
            ),
            "generation_attempts": generation_attempt,
            "generation_successes": 1,
            "end_to_end_yield": float(1.0 / generation_attempt),
            "rejection_reasons": dict(sorted(rejection_reasons.items())),
        }
        if accepted > 0:
            pipeline_record["proposals_per_accepted_swap"] = float(
                proposals / accepted
            )
        pipeline_records.append(pipeline_record)
        print(
            f"graph={index + 1}/{num_generate} "
            f"n={refined.number_of_nodes()} m={refined.number_of_edges()} "
            f"accepted_steps={accepted} prediction_calls={prediction_calls} "
            f"plateau_refreshes={plateau_refreshes}",
            flush=True,
        )

    evaluation_cfg = dict(config.get("evaluation", {}) or {})
    inline_evaluation = bool(evaluation_cfg.get("inline_during_generation", False))
    orca_exec = None
    coarse_metrics: dict[str, Any] = {}
    refined_metrics: dict[str, Any] = {}
    if inline_evaluation:
        compute_orbit = bool(evaluation_cfg.get("compute_orbit", True))
        graphlet_backend = str(
            evaluation_cfg.get("graphlet_backend", "sampled")
        ).lower()
        if compute_orbit or graphlet_backend in {"orca", "exact_orca", "exact"}:
            orca_exec = configure_orca_executable(
                evaluation_cfg.get("orca_exec"),
                required=True,
            )
        references = reference_graphs[:num_generate] or reference_graphs
        metric_kwargs = {
            "compute_orbit": compute_orbit,
            "compute_graphlet_history": bool(
                evaluation_cfg.get("compute_graphlet_history", True)
            ),
            "graphlet_k_min": int(
                evaluation_cfg.get(
                    "graphlet_k_min", summary_config.graphlet_k_min
                )
            ),
            "graphlet_k_max": int(
                evaluation_cfg.get(
                    "graphlet_k_max", summary_config.graphlet_k_max
                )
            ),
            "graphlet_connected_only": bool(
                evaluation_cfg.get(
                    "graphlet_connected_only",
                    summary_config.graphlet_connected_only,
                )
            ),
            "graphlet_num_samples": evaluation_cfg.get(
                "graphlet_num_samples", summary_config.graphlet_num_samples
            ),
            "graphlet_backend": graphlet_backend,
        }
        coarse_metrics = evaluate_graph_sets(
            references,
            coarse_graphs,
            train_graphs,
            **metric_kwargs,
        )
        refined_metrics = evaluate_graph_sets(
            references,
            refined_graphs,
            train_graphs,
            **metric_kwargs,
        )

    aggregated_pipeline = aggregate_pipeline_diagnostics(
        pipeline_records,
        require_complete=True,
        allow_fallback=False,
    )
    accepted_steps = [
        sum(bool(row.get("accepted")) for row in trace) for trace in traces
    ]
    trace_rows = [row for trace in traces for row in trace]
    accepted_rows = [row for row in trace_rows if bool(row.get("accepted"))]
    prediction_refresh_rows = [
        row for row in trace_rows if bool(row.get("prediction_refreshed", False))
    ]
    prediction_call_counts = [
        max(
            (int(row.get("prediction_calls", 0)) for row in trace),
            default=0,
        )
        for trace in traces
    ]
    prediction_horizons = [
        int(row["prediction_horizon"])
        for row in prediction_refresh_rows
        if "prediction_horizon" in row
    ]
    plateau_refresh_count = sum(
        row.get("reason") == "prediction_plateau_refresh" for row in trace_rows
    )
    diagnostics = {
        "pipeline_mode": "topology",
        "degree_preservation_rate": degree_preservation_rate(
            coarse_graphs,
            refined_graphs,
        ),
        "constructor_target_degree_match_rate": degree_target_match_rate(
            coarse_graphs,
            target_degree_sequences,
        ),
        "final_target_degree_match_rate": degree_target_match_rate(
            refined_graphs,
            target_degree_sequences,
        ),
        "connectedness_rate": float(
            np.mean(
                [
                    graph.number_of_nodes() > 0
                    and (graph.number_of_nodes() == 1 or nx.is_connected(graph))
                    for graph in refined_graphs
                ]
            )
        ),
        "mean_accepted_steps": float(np.mean(accepted_steps)),
        "mean_accepted_graphlet_gain": (
            float(np.mean([row["graphlet_gain"] for row in accepted_rows]))
            if accepted_rows
            else 0.0
        ),
        "mean_accepted_clustering_gain": (
            float(np.mean([row["clustering_gain"] for row in accepted_rows]))
            if accepted_rows
            else 0.0
        ),
        "mean_accepted_orbit_gain": (
            float(np.mean([row["orbit_gain"] for row in accepted_rows]))
            if accepted_rows
            else 0.0
        ),
        "mean_accepted_structural_gain": (
            float(np.mean([row["structural_gain"] for row in accepted_rows]))
            if accepted_rows
            else 0.0
        ),
        "all_accepted_moves_improve_frozen_energy": bool(
            all(float(row["energy_improvement"]) > 0.0 for row in accepted_rows)
        ),
        "all_accepted_moves_pass_relative_threshold": bool(
            all(
                float(row.get("relative_energy_improvement", 0.0))
                > float(refiner_settings.min_relative_improvement)
                for row in accepted_rows
            )
        ),
        "prediction_horizon_mode": refiner_settings.prediction_horizon_mode,
        "prediction_horizon_schedule": (
            refiner_settings.prediction_horizon_schedule
        ),
        "prediction_horizon_initial_k": (
            refiner_settings.prediction_horizon_initial_k
        ),
        "prediction_horizon_final_k": (
            refiner_settings.prediction_horizon_final_k
        ),
        "mean_realized_prediction_horizon": (
            float(np.mean(prediction_horizons)) if prediction_horizons else 0.0
        ),
        "mean_prediction_calls": float(np.mean(prediction_call_counts)),
        "mean_accepted_swaps_per_prediction_call": float(
            sum(accepted_steps) / max(sum(prediction_call_counts), 1)
        ),
        "plateau_refresh_count": int(plateau_refresh_count),
        "predictor_graphlet_error": float(predictor_graphlet_error),
        "predictor_clustering_error": (
            float(predictor_clustering_error)
            if predictor_clustering_error is not None
            else None
        ),
        "predictor_orbit_log_error": (
            float(predictor_orbit_log_error)
            if predictor_orbit_log_error is not None
            else None
        ),
        "mean_graph_runtime_seconds": float(np.mean(graph_runtimes)),
        "runtime_seconds": float(time.perf_counter() - run_started),
        "inline_evaluation": inline_evaluation,
    }
    report = {
        "format": "topology_structural_generation_v2",
        "pipeline_mode": "topology",
        "checkpoint_format": checkpoint.get("format"),
        "degree_source": degree_source,
        "prediction_horizon": {
            "mode": refiner_settings.prediction_horizon_mode,
            "initial_k": refiner_settings.prediction_horizon_initial_k,
            "final_k": refiner_settings.prediction_horizon_final_k,
            "schedule": refiner_settings.prediction_horizon_schedule,
            "refresh_on_plateau": refiner_settings.refresh_on_plateau,
            "min_improvement": refiner_settings.min_improvement,
            "min_relative_improvement": (
                refiner_settings.min_relative_improvement
            ),
        },
        "orca_exec": orca_exec,
        "num_generated": len(refined_graphs),
        "hh_source": coarse_metrics,
        "topology_refined": refined_metrics,
        # Compatibility aliases for existing report consumers.
        "coarse": coarse_metrics,
        "hybrid_refined": refined_metrics,
        "diagnostics": diagnostics,
        "pipeline_diagnostics": aggregated_pipeline,
        "pipeline_records": pipeline_records,
        "traces": traces,
        "seed": seed,
        "config": config,
    }
    output_dir = ensure_dir(args.output_dir)
    save_pickle(coarse_graphs, output_dir / "coarse_graphs.pkl")
    save_pickle(refined_graphs, output_dir / "topology_refined_graphs.pkl")
    if bool(generation_cfg.get("write_legacy_hybrid_alias", False)):
        save_pickle(refined_graphs, output_dir / "hybrid_refined_graphs.pkl")
    save_json(report, output_dir / "report.json")
    print("Topology generation diagnostics", flush=True)
    for key, value in diagnostics.items():
        print(f"  {key}: {value}", flush=True)
    print(f"Saved results to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()

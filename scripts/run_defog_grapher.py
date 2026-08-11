#!/usr/bin/env python
"""Generate with frozen DeFoG, then apply post-hoc topology GraphER correction."""

from __future__ import annotations

import argparse
import time
from collections import Counter
from collections.abc import Sequence
from typing import Any

import networkx as nx
import numpy as np
import torch

from grapher.data.io import load_dataset_splits
from grapher.evaluation.metrics import degree_preservation_rate, evaluate_graph_sets
from grapher.evaluation.studies import aggregate_pipeline_diagnostics
from grapher.generators.defog import (
    DeFoGGeneratorConfig,
    generate_defog_graphs,
)
from grapher.properties.summary import configure_orca_executable
from grapher.topology.data import normalize_topology_graph
from grapher.topology.model import load_topology_checkpoint
from grapher.topology.refiner import refine_graph_with_topology_predictions
from grapher.utils.io import ensure_dir, load_yaml, save_json, save_pickle


def _is_connected(graph: nx.Graph) -> bool:
    return graph.number_of_nodes() > 0 and (
        graph.number_of_nodes() == 1 or nx.is_connected(graph)
    )


def _indexed_degrees(graph: nx.Graph) -> list[int]:
    return [int(graph.degree(node)) for node in range(graph.number_of_nodes())]


def correct_defog_base_graphs(
    base_graphs: Sequence[nx.Graph],
    *,
    model: Any,
    graphlet_basis: Any,
    summary_config: Any,
    refiner_config: dict[str, Any],
    device: torch.device | str,
    rng: np.random.Generator,
    predictor_graphlet_error: float,
    disconnected_policy: str = "no_op_and_report",
    show_progress: bool = True,
) -> tuple[
    list[nx.Graph],
    list[list[dict[str, Any]]],
    list[dict[str, Any]],
    list[float],
]:
    """Correct connected base graphs and retain disconnected samples unchanged."""

    policy = str(disconnected_policy).lower()
    if policy not in {"no_op_and_report", "error"}:
        raise ValueError(
            "generation.disconnected_policy must be no_op_and_report or error."
        )
    refined_graphs: list[nx.Graph] = []
    traces: list[list[dict[str, Any]]] = []
    records: list[dict[str, Any]] = []
    runtimes: list[float] = []

    for index, raw_graph in enumerate(base_graphs):
        started = time.perf_counter()
        base = normalize_topology_graph(raw_graph)
        initial_degrees = _indexed_degrees(base)
        connected = _is_connected(base)
        correction_attempted = connected
        if not connected:
            if policy == "error":
                raise ValueError(
                    f"DeFoG base graph {index} is disconnected and cannot be "
                    "processed by the connectivity-preserving topology refiner."
                )
            refined = base.copy()
            trace = [
                {
                    "step": 0,
                    "accepted": False,
                    "reason": "source_disconnected_noop",
                    "num_proposals": 0,
                    "num_valid_candidates": 0,
                    "candidate_rejection_reasons": {
                        "source_disconnected": 1,
                    },
                }
            ]
        else:
            refined, trace = refine_graph_with_topology_predictions(
                base,
                model=model,
                graphlet_basis=graphlet_basis,
                summary_config=summary_config,
                refiner_config=refiner_config,
                device=device,
                rng=rng,
                return_trace=True,
            )

        runtime = float(time.perf_counter() - started)
        if refined.number_of_nodes() != base.number_of_nodes():
            raise AssertionError("GraphER correction changed the DeFoG node count.")
        if _indexed_degrees(refined) != initial_degrees:
            raise AssertionError("GraphER correction changed indexed DeFoG degrees.")
        if connected and not _is_connected(refined):
            raise AssertionError("GraphER correction disconnected a DeFoG base graph.")

        decision_rows = [row for row in trace if "num_proposals" in row]
        proposals = sum(int(row.get("num_proposals", 0)) for row in decision_rows)
        passes = sum(int(row.get("num_valid_candidates", 0)) for row in decision_rows)
        accepted = sum(bool(row.get("accepted")) for row in trace)
        rejection_reasons: Counter[str] = Counter()
        for row in decision_rows:
            rejection_reasons.update(
                {
                    str(key): int(value)
                    for key, value in (
                        row.get("candidate_rejection_reasons", {}) or {}
                    ).items()
                }
            )
        stopped = bool(trace and not trace[-1].get("accepted", False))
        records.append(
            {
                "pipeline_mode": "topology",
                "base_generator": "defog",
                "graphlet_error": float(predictor_graphlet_error),
                "invariant_feasible": 1.0,
                "constructor_success": 1.0,
                "accepted_swaps": accepted,
                "runtime_seconds": runtime,
                "fallback_used": 0.0,
                "candidate_proposals": proposals,
                "candidate_passes": passes,
                "candidate_pass_rate": float(passes / max(proposals, 1)),
                "proposals_per_accepted_swap": float(
                    proposals / accepted if accepted else 0.0
                ),
                "stopped": float(stopped),
                "stop_opportunities": 1,
                "stop_rate": float(stopped),
                "generation_attempts": 1,
                "generation_successes": 1,
                "end_to_end_yield": 1.0,
                "base_connected": float(connected),
                "correction_attempted": float(correction_attempted),
                "rejection_reasons": dict(sorted(rejection_reasons.items())),
            }
        )
        refined_graphs.append(refined)
        traces.append(trace)
        runtimes.append(runtime)
        if show_progress:
            print(
                f"graph={index + 1}/{len(base_graphs)} "
                f"n={refined.number_of_nodes()} m={refined.number_of_edges()} "
                f"connected={connected} accepted_steps={accepted}",
                flush=True,
            )
    return refined_graphs, traces, records, runtimes


def _inline_metrics(
    *,
    config: dict[str, Any],
    train_graphs: list[nx.Graph],
    reference_graphs: list[nx.Graph],
    base_graphs: list[nx.Graph],
    refined_graphs: list[nx.Graph],
    summary_config: Any,
    num_generate: int,
) -> tuple[dict[str, Any], dict[str, Any], str | None]:
    evaluation_cfg = dict(config.get("evaluation", {}) or {})
    if not bool(evaluation_cfg.get("inline_during_generation", False)):
        return {}, {}, None
    compute_orbit = bool(evaluation_cfg.get("compute_orbit", True))
    graphlet_backend = str(evaluation_cfg.get("graphlet_backend", "sampled")).lower()
    orca_exec = None
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
            evaluation_cfg.get("graphlet_k_min", summary_config.graphlet_k_min)
        ),
        "graphlet_k_max": int(
            evaluation_cfg.get("graphlet_k_max", summary_config.graphlet_k_max)
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
    return (
        evaluate_graph_sets(references, base_graphs, train_graphs, **metric_kwargs),
        evaluate_graph_sets(
            references,
            refined_graphs,
            train_graphs,
            **metric_kwargs,
        ),
        orca_exec,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate generic graphs with DeFoG and correct them with GraphER."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--defog-checkpoint", default=None)
    parser.add_argument("--defog-generated", default=None)
    parser.add_argument("--defog-python", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-generate", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--defog-device", default=None)
    args = parser.parse_args()
    run_started = time.perf_counter()

    config = load_yaml(args.config)
    pipeline_stage = str(
        (config.get("pipeline", {}) or {}).get("stage", "posthoc_correction")
    ).lower()
    if pipeline_stage not in {"posthoc_correction", "topology_correction"}:
        raise ValueError(
            "run_defog_grapher.py requires pipeline.stage: posthoc_correction."
        )
    if config.get("categorical_state") or config.get("molecular_generation"):
        raise ValueError("The DeFoG topology corrector accepts generic graphs only.")
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
    train_graphs = list(splits.get("train", []))
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
    model_device_name = args.device or predictor_cfg.get("device", "auto")
    model, graphlet_basis, summary_config, checkpoint = load_topology_checkpoint(
        checkpoint_path,
        device=model_device_name,
    )
    model_device = next(model.parameters()).device
    predictor_report = checkpoint.get("report", {}) or {}
    predictor_graphlet_error = predictor_report.get("val_graphlet_mae")
    if predictor_graphlet_error is None:
        raise ValueError(
            "The topology checkpoint is missing held-out val_graphlet_mae."
        )

    output_dir = ensure_dir(args.output_dir)
    base_cfg_raw = dict(config.get("base_generator", {}) or {})
    base_cfg = DeFoGGeneratorConfig.from_dict(
        base_cfg_raw,
        checkpoint_path=args.defog_checkpoint,
        generated_path=args.defog_generated,
        python_executable=args.defog_python,
        device=args.defog_device,
    )
    base_result = generate_defog_graphs(
        base_cfg,
        num_graphs=num_generate,
        seed=seed,
        output_dir=output_dir,
    )
    base_graphs = [normalize_topology_graph(graph) for graph in base_result.graphs]
    refiner_cfg = dict(config.get("topology_refiner", {}) or {})
    if not bool(refiner_cfg.get("preserve_connectivity", True)):
        raise ValueError("DeFoG correction requires preserve_connectivity: true.")
    refined_graphs, traces, pipeline_records, graph_runtimes = (
        correct_defog_base_graphs(
            base_graphs,
            model=model,
            graphlet_basis=graphlet_basis,
            summary_config=summary_config,
            refiner_config=refiner_cfg,
            device=model_device,
            rng=rng,
            predictor_graphlet_error=float(predictor_graphlet_error),
            disconnected_policy=str(
                generation_cfg.get("disconnected_policy", "no_op_and_report")
            ),
        )
    )

    base_metrics, refined_metrics, orca_exec = _inline_metrics(
        config=config,
        train_graphs=train_graphs,
        reference_graphs=reference_graphs,
        base_graphs=base_graphs,
        refined_graphs=refined_graphs,
        summary_config=summary_config,
        num_generate=num_generate,
    )
    aggregated_pipeline = aggregate_pipeline_diagnostics(
        pipeline_records,
        require_complete=True,
        allow_fallback=False,
    )
    accepted_rows = [
        row for trace in traces for row in trace if bool(row.get("accepted"))
    ]
    accepted_steps = [
        sum(bool(row.get("accepted")) for row in trace) for trace in traces
    ]
    base_connected = np.asarray(
        [_is_connected(graph) for graph in base_graphs], dtype=np.float64
    )
    changed = np.asarray(
        [
            set(base.edges()) != set(refined.edges())
            for base, refined in zip(base_graphs, refined_graphs)
        ],
        dtype=np.float64,
    )
    diagnostics = {
        "pipeline_mode": "topology",
        "base_generator": "defog",
        "degree_preservation_rate": degree_preservation_rate(
            base_graphs,
            refined_graphs,
        ),
        "connected_base_rate": float(base_connected.mean()),
        "connected_final_rate": float(
            np.mean([_is_connected(graph) for graph in refined_graphs])
        ),
        "correction_attempt_rate": float(base_connected.mean()),
        "disconnected_noop_rate": float(1.0 - base_connected.mean()),
        "correction_changed_rate": float(changed.mean()),
        "mean_accepted_steps": float(np.mean(accepted_steps)),
        "mean_accepted_graphlet_gain": (
            float(np.mean([row["graphlet_gain"] for row in accepted_rows]))
            if accepted_rows
            else 0.0
        ),
        "all_accepted_moves_improve_frozen_energy": bool(
            all(float(row["energy_improvement"]) > 0.0 for row in accepted_rows)
        ),
        "predictor_graphlet_error": float(predictor_graphlet_error),
        "mean_graph_runtime_seconds": float(np.mean(graph_runtimes)),
        "runtime_seconds": float(time.perf_counter() - run_started),
        "inline_evaluation": bool(
            (config.get("evaluation", {}) or {}).get("inline_during_generation", False)
        ),
    }
    report = {
        "format": "defog_topology_correction_v1",
        "pipeline_mode": "topology",
        "pipeline_stage": pipeline_stage,
        "base_generator": "defog",
        "checkpoint_format": checkpoint.get("format"),
        "orca_exec": orca_exec,
        "num_generated": len(refined_graphs),
        "defog_base": base_metrics,
        "topology_refined": refined_metrics,
        "diagnostics": diagnostics,
        "pipeline_diagnostics": aggregated_pipeline,
        "pipeline_records": pipeline_records,
        "traces": traces,
        "defog_export_path": str(base_result.export_path),
        "defog_manifest_path": (
            str(base_result.manifest_path) if base_result.manifest_path else None
        ),
        "defog_manifest": base_result.manifest,
        "dataset_provenance": {
            "grapher_reference_dataset": {
                "name": dataset_cfg.get("name"),
                "benchmark": dataset_cfg.get("benchmark"),
                "config_path": dataset_cfg.get("config_path"),
            },
            "defog_model_dataset": {
                "dataset": base_cfg.dataset,
                "experiment": base_cfg.experiment,
            },
            "contract": dict(base_cfg_raw.get("dataset_contract", {}) or {}),
        },
        "seed": seed,
        "config": config,
    }
    save_pickle(base_graphs, output_dir / "defog_base_graphs.pkl")
    save_pickle(refined_graphs, output_dir / "topology_refined_graphs.pkl")
    save_json(report, output_dir / "report.json")
    print("DeFoG + GraphER correction diagnostics", flush=True)
    for key, value in diagnostics.items():
        print(f"  {key}: {value}", flush=True)
    print(f"Saved results to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()

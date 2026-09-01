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

from grapher.data.io import load_dataset_splits
from grapher.models.dhvae_hh.degree_sampler import (
    EmpiricalDegreeSampler,
    build_degree_sampler,
)
from grapher.models.dhvae_hh.havel_hakimi import (
    assert_constructor_validity,
    construct_coarse_graph,
)
from grapher.properties.summary import configure_orca_executable
from grapher.rewiring_mlp.evaluation.metrics import (
    degree_preservation_rate,
    degree_target_match_rate,
    evaluate_graph_sets,
)
from grapher.rewiring_mlp.evaluation.studies import aggregate_pipeline_diagnostics
from grapher.rewiring_mlp.generic.model import (
    TOPOLOGY_CHECKPOINT_FORMAT,
    load_topology_checkpoint,
)
from grapher.rewiring_mlp.generic.flow_model import (
    TOPOLOGY_FLOW_GRAPHLET_CHECKPOINT_FORMAT,
    load_topology_flow_graphlet_checkpoint,
)
from grapher.rewiring_mlp.generic.flow_graphlet_refiner import (
    FlowGraphletRefinerConfig,
    refine_graph_with_flow_graphlet_predictions,
)
from grapher.rewiring_mlp.generic.refiner import (
    TopologyRefinerConfig,
    refine_graph_with_topology_predictions,
)
from grapher.rewiring_mlp.generic.spectral_model import (
    TOPOLOGY_SPECTRAL_CHECKPOINT_FORMAT,
    TOPOLOGY_SPECTRAL_GRAPHLET_CHECKPOINT_FORMAT,
    load_topology_spectral_checkpoint,
    load_topology_spectral_graphlet_checkpoint,
    training_time_horizon_from_config,
)
from grapher.rewiring_mlp.generic.spectral_refiner import (
    SpectralRefinerConfig,
    refine_graph_with_spectral_predictions,
)
from grapher.rewiring_mlp.generic.spectral_graphlet_refiner import (
    SpectralGraphletRefinerConfig,
    refine_graph_with_spectral_graphlet_predictions,
)
from grapher.utils.io import (
    apply_config_overrides,
    ensure_dir,
    load_yaml,
    save_json,
    save_pickle,
)


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


def _checkpoint_format(path: str | Path) -> str:
    checkpoint = torch.load(Path(path), map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Topology checkpoint must be a mapping: {path}")
    return str(checkpoint.get("format", ""))


def _mean_or_zero(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if key in row and row[key] is not None]
    return float(np.mean(values)) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate generic graph topologies with DH-VAE/empirical degrees, "
            "connected Havel-Hakimi construction, and GraphER rewiring. The "
            "refiner is selected automatically from the checkpoint format "
            "(structural-summary, spectral, joint spectral + graphlet-logit "
            "diffusion guidance, or edge-flow matching + graphlet guidance)."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-generate", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--set",
        "--override",
        dest="config_overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Override any YAML option using a dotted path. Repeat this flag for "
            "multiple values, e.g. --set topology_refiner.steps=40 --set "
            "topology_refiner.spectral_guidance.min_clean_mix=0.25. Values are "
            "parsed as YAML, so booleans/lists/null/numbers keep their types."
        ),
    )
    args = parser.parse_args()
    run_started = time.perf_counter()

    config = load_yaml(args.config)
    apply_config_overrides(config, args.config_overrides)
    pipeline_stage = str((config.get("pipeline", {}) or {}).get("stage", "topology")).lower()
    if pipeline_stage != "topology":
        raise ValueError("run_topology_grapher.py requires pipeline.stage: topology.")
    if config.get("categorical_state") or config.get("molecular_generation"):
        raise ValueError("The topology generator accepts generic datasets only.")

    seed = int(args.seed if args.seed is not None else config.get("seed", 0))
    seed_sequence = np.random.SeedSequence(seed)
    source_seed_sequence, refiner_seed_sequence = seed_sequence.spawn(2)
    source_rng = np.random.default_rng(source_seed_sequence)
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
    refiner_graph_seeds = refiner_seed_sequence.spawn(num_generate)

    predictor_cfg = dict(config.get("topology_predictor", {}) or {})
    checkpoint_path = args.checkpoint or predictor_cfg.get("checkpoint_path")
    if not checkpoint_path:
        raise ValueError("topology_predictor.checkpoint_path is required.")
    device = args.device or predictor_cfg.get("device", "auto")
    checkpoint_format = _checkpoint_format(checkpoint_path)

    graphlet_basis = None
    predictor_graphlet_error: float | None = None
    predictor_clustering_error: float | None = None
    predictor_orbit_log_error: float | None = None
    predictor_spectral_error: float | None = None
    predictor_flow_error: float | None = None
    if checkpoint_format == TOPOLOGY_SPECTRAL_GRAPHLET_CHECKPOINT_FORMAT:
        guidance_mode = "spectral_graphlet"
        model, graphlet_basis, summary_config, checkpoint = (
            load_topology_spectral_graphlet_checkpoint(
                checkpoint_path,
                device=device,
            )
        )
        predictor_report = checkpoint.get("report", {}) or {}
        predictor_spectral_error_raw = predictor_report.get(
            "val_spectral_normalized_rmse",
            predictor_report.get("val_spectral_normalized_mae"),
        )
        predictor_graphlet_error_raw = predictor_report.get(
            "val_graphlet_logit_rmse",
            predictor_report.get("val_graphlet_probability_mae"),
        )
        if predictor_spectral_error_raw is None or predictor_graphlet_error_raw is None:
            raise ValueError(
                "The spectral+graphlet checkpoint is missing held-out spectral "
                "or graphlet-logit validation error."
            )
        predictor_spectral_error = float(predictor_spectral_error_raw)
        predictor_graphlet_error = float(predictor_graphlet_error_raw)
    elif checkpoint_format == TOPOLOGY_FLOW_GRAPHLET_CHECKPOINT_FORMAT:
        guidance_mode = "flow_graphlet"
        model, graphlet_basis, summary_config, checkpoint = (
            load_topology_flow_graphlet_checkpoint(
                checkpoint_path,
                device=device,
            )
        )
        predictor_report = checkpoint.get("report", {}) or {}
        predictor_flow_error_raw = predictor_report.get(
            "val_flow_rmse",
            predictor_report.get("val_flow_mae"),
        )
        predictor_graphlet_error_raw = predictor_report.get(
            "val_graphlet_logit_rmse",
            predictor_report.get("val_graphlet_probability_mae"),
        )
        if predictor_flow_error_raw is None or predictor_graphlet_error_raw is None:
            raise ValueError(
                "The flow+graphlet checkpoint is missing held-out flow or graphlet "
                "validation error."
            )
        predictor_flow_error = float(predictor_flow_error_raw)
        predictor_graphlet_error = float(predictor_graphlet_error_raw)
    elif checkpoint_format == TOPOLOGY_SPECTRAL_CHECKPOINT_FORMAT:
        guidance_mode = "spectral"
        model, summary_config, checkpoint = load_topology_spectral_checkpoint(
            checkpoint_path,
            device=device,
        )
        predictor_report = checkpoint.get("report", {}) or {}
        predictor_spectral_error_raw = predictor_report.get(
            "val_spectral_normalized_rmse",
            predictor_report.get("val_spectral_normalized_mae"),
        )
        if predictor_spectral_error_raw is None:
            raise ValueError(
                "The spectral checkpoint is missing held-out "
                "val_spectral_normalized_rmse/mae; generate from a checkpoint "
                "selected by train_topology_grapher.py."
            )
        predictor_spectral_error = float(predictor_spectral_error_raw)
    elif checkpoint_format == TOPOLOGY_CHECKPOINT_FORMAT:
        guidance_mode = "structural_summary"
        model, graphlet_basis, summary_config, checkpoint = load_topology_checkpoint(
            checkpoint_path,
            device=device,
        )
        predictor_report = checkpoint.get("report", {}) or {}
        predictor_graphlet_error_raw = predictor_report.get("val_graphlet_mae")
        if predictor_graphlet_error_raw is None:
            raise ValueError(
                "The topology checkpoint is missing held-out val_graphlet_mae; "
                "generate from a checkpoint selected by train_topology_grapher.py."
            )
        predictor_graphlet_error = float(predictor_graphlet_error_raw)
        clustering = predictor_report.get("val_clustering_mae")
        orbit = predictor_report.get("val_orbit_log_mae")
        predictor_clustering_error = None if clustering is None else float(clustering)
        predictor_orbit_log_error = None if orbit is None else float(orbit)
    else:
        raise ValueError(
            f"Unsupported topology checkpoint format {checkpoint_format!r}. "
            f"Expected {TOPOLOGY_CHECKPOINT_FORMAT!r}, "
            f"{TOPOLOGY_SPECTRAL_CHECKPOINT_FORMAT!r}, or "
            f"{TOPOLOGY_SPECTRAL_GRAPHLET_CHECKPOINT_FORMAT!r}, or "
            f"{TOPOLOGY_FLOW_GRAPHLET_CHECKPOINT_FORMAT!r}."
        )
    model_device = next(model.parameters()).device

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
                "Learned topology generation requires degree_generator.fallback: error."
            )
        degree_cfg["enabled"] = True
        degree_sampler = build_degree_sampler(degree_cfg, train_graphs, seed=seed)
    elif degree_source in {"empirical", "train_empirical"}:
        degree_sampler = EmpiricalDegreeSampler.fit_from_graphs(train_graphs, seed=seed)
    elif degree_source not in {"oracle", "test_oracle"}:
        raise ValueError(f"Unknown generation.degree_source: {degree_source!r}")

    constructor_cfg = dict(config.get("constructor", {}) or {})
    if str(constructor_cfg.get("type", "havel_hakimi")).lower() != "havel_hakimi":
        raise ValueError("The generic topology stage requires Havel-Hakimi construction.")
    if not bool(constructor_cfg.get("ensure_connected", True)):
        raise ValueError("Topology generation requires constructor.ensure_connected.")

    refiner_cfg = dict(config.get("topology_refiner", {}) or {})
    if guidance_mode in {"spectral", "spectral_graphlet"}:
        time_parameterization = str(
            checkpoint.get("time_parameterization", "") or ""
        ).lower()
        if time_parameterization == "normalized_diffusion_progress_0_source_1_clean":
            # V2 checkpoints are queried with normalized reverse progress in
            # [0,1].  Generation step budget is therefore a pure projection
            # compute knob and never rescales the neural time input.
            refiner_cfg.pop("time_horizon", None)
            prefix = (
                "[GraphER/SpectralGraphlet]"
                if guidance_mode == "spectral_graphlet"
                else "[GraphER/Spectral]"
            )
            print(
                f"{prefix} predictor time=normalized diffusion progress "
                "(0=source, 1=clean); independent of topology_refiner.steps.",
                flush=True,
            )
        else:
            # Legacy v1 checkpoints normalized time by topology_trajectory.steps.
            training_horizon = checkpoint.get("training_time_horizon")
            if training_horizon is None:
                training_horizon = training_time_horizon_from_config(
                    checkpoint.get("config", {}) or {}
                )
            if training_horizon is None:
                prefix = (
                    "[GraphER/SpectralGraphlet]"
                    if guidance_mode == "spectral_graphlet"
                    else "[GraphER/Spectral]"
                )
                print(
                    f"{prefix} WARNING: legacy checkpoint has no training time "
                    "horizon; falling back to topology_refiner.steps.",
                    flush=True,
                )
            else:
                training_horizon = int(training_horizon)
                explicit = refiner_cfg.get("time_horizon")
                if explicit is not None and int(explicit) != training_horizon:
                    raise ValueError(
                        "topology_refiner.time_horizon "
                        f"({int(explicit)}) disagrees with the legacy checkpoint "
                        f"horizon ({training_horizon})."
                    )
                refiner_cfg["time_horizon"] = training_horizon

        if guidance_mode == "spectral_graphlet":
            refiner_settings: Any = SpectralGraphletRefinerConfig.from_dict(refiner_cfg)
            if graphlet_basis is None:
                raise ValueError("Spectral+graphlet checkpoint is missing its graphlet basis.")
            print(
                "[GraphER/SpectralGraphlet] loaded joint Spectral Transformer + "
                f"graphlet-logit checkpoint format={checkpoint_format} device={model_device}",
                flush=True,
            )
            print(
                "[GraphER/SpectralGraphlet] guidance: spectrum supplies global "
                "denoising, graphlet CLR/logit diffusion supplies local higher-order "
                "denoising, and exact local graphlet deltas score valid degree-preserving swaps.",
                flush=True,
            )
        else:
            refiner_settings = SpectralRefinerConfig.from_dict(refiner_cfg)
            print(
                "[GraphER/Spectral] loaded Spectral Transformer checkpoint "
                f"format={checkpoint_format} device={model_device}",
                flush=True,
            )
            print(
                "[GraphER/Spectral] guidance: model predicts the full clean Laplacian "
                "eigenvalue vector jointly; the bridge derives the next spectral target; "
                "valid degree-preserving swaps project the graph toward that target.",
                flush=True,
            )
        if not refiner_settings.preserve_connectivity:
            raise ValueError("Spectral-family topology generation requires connectivity preservation.")
        debug_prefix = (
            "[GraphER/SpectralGraphlet]"
            if guidance_mode == "spectral_graphlet"
            else "[GraphER/Spectral]"
        )
        print(
            f"{debug_prefix} debug="
            f"{refiner_settings.debug_enabled} print_every={refiner_settings.debug_print_every} "
            f"top_candidates={refiner_settings.debug_top_candidates} "
            f"spectrum_values={refiner_settings.debug_spectrum_values}",
            flush=True,
        )
    elif guidance_mode == "flow_graphlet":
        # Flow checkpoints use normalized accepted-step progress directly;
        # legacy spectral time-horizon compatibility does not apply.
        refiner_cfg.pop("time_horizon", None)
        refiner_settings = FlowGraphletRefinerConfig.from_dict(refiner_cfg)
        if graphlet_basis is None:
            raise ValueError("Flow+graphlet checkpoint is missing its graphlet basis.")
        if not refiner_settings.preserve_connectivity:
            raise ValueError(
                "Flow+graphlet topology generation requires connectivity "
                "preservation."
            )
        print(
            "[GraphER/FlowGraphlet] loaded joint edge-flow matching + graphlet "
            f"checkpoint format={checkpoint_format} device={model_device}",
            flush=True,
        )
        print(
            "[GraphER/FlowGraphlet] guidance: the learned degree-tangent edge "
            "velocity scores each valid double-edge swap directly; exact local "
            "graphlet deltas provide higher-order structural guidance. Rewiring "
            "is projection only and was not used to create training states.",
            flush=True,
        )
    else:
        refiner_settings = TopologyRefinerConfig.from_dict(refiner_cfg)
        if not refiner_settings.preserve_connectivity:
            raise ValueError("Topology generation requires topology_refiner.preserve_connectivity.")

    coarse_graphs: list[nx.Graph] = []
    refined_graphs: list[nx.Graph] = []
    target_degree_sequences: list[list[int]] = []
    traces: list[list[dict[str, Any]]] = []
    graph_runtimes: list[float] = []
    pipeline_records: list[dict[str, Any]] = []
    max_attempts_per_graph = int(generation_cfg.get("max_attempts_per_graph", 8))
    if max_attempts_per_graph <= 0:
        raise ValueError("generation.max_attempts_per_graph must be positive.")

    for index in range(num_generate):
        graph_started = time.perf_counter()
        generation_rejections: Counter[str] = Counter()
        for generation_attempt in range(1, max_attempts_per_graph + 1):
            try:
                if degree_source in {"oracle", "test_oracle"}:
                    if not reference_graphs:
                        raise ValueError("Oracle degree generation requires test graphs.")
                    degree_summary = _oracle_degree_summary(
                        reference_graphs[index % len(reference_graphs)]
                    )
                else:
                    if degree_sampler is None:
                        raise RuntimeError("Degree sampler was not initialized.")
                    degree_summary = degree_sampler.sample(source_rng)
            except RuntimeError:
                generation_rejections["degree_prior_rejected"] += 1
                continue

            try:
                coarse = construct_coarse_graph(degree_summary, constructor_cfg, source_rng)
                assert_constructor_validity(coarse, degree_summary, require_connected=True)
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

        if guidance_mode == "spectral_graphlet":
            assert graphlet_basis is not None
            refined, trace = refine_graph_with_spectral_graphlet_predictions(
                coarse,
                model=model,
                graphlet_basis=graphlet_basis,
                refiner_config=refiner_settings,
                device=model_device,
                rng=np.random.default_rng(refiner_graph_seeds[index]),
                return_trace=True,
                debug_context=(
                    f"graph={index + 1}/{num_generate} "
                    f"n={coarse.number_of_nodes()} m={coarse.number_of_edges()}"
                ),
            )
        elif guidance_mode == "flow_graphlet":
            assert graphlet_basis is not None
            refined, trace = refine_graph_with_flow_graphlet_predictions(
                coarse,
                model=model,
                graphlet_basis=graphlet_basis,
                refiner_config=refiner_settings,
                device=model_device,
                rng=np.random.default_rng(refiner_graph_seeds[index]),
                return_trace=True,
                debug_context=(
                    f"graph={index + 1}/{num_generate} "
                    f"n={coarse.number_of_nodes()} m={coarse.number_of_edges()}"
                ),
            )
        elif guidance_mode == "spectral":
            refined, trace = refine_graph_with_spectral_predictions(
                coarse,
                model=model,
                refiner_config=refiner_settings,
                device=model_device,
                rng=np.random.default_rng(refiner_graph_seeds[index]),
                return_trace=True,
                debug_context=(
                    f"graph={index + 1}/{num_generate} "
                    f"n={coarse.number_of_nodes()} m={coarse.number_of_edges()}"
                ),
            )
        else:
            assert graphlet_basis is not None
            refined, trace = refine_graph_with_topology_predictions(
                coarse,
                model=model,
                graphlet_basis=graphlet_basis,
                summary_config=summary_config,
                refiner_config=refiner_settings,
                device=model_device,
                rng=np.random.default_rng(refiner_graph_seeds[index]),
                return_trace=True,
            )

        runtime = float(time.perf_counter() - graph_started)
        coarse_graphs.append(coarse)
        refined_graphs.append(refined)
        target_degree_sequences.append([int(value) for value in degree_summary["degree_sequence"]])
        traces.append(trace)
        graph_runtimes.append(runtime)
        sampling_diagnostics = dict(degree_summary.get("sampling_diagnostics", {}) or {})

        decision_rows = [row for row in trace if "num_proposals" in row]
        proposals = sum(int(row.get("num_proposals", 0)) for row in decision_rows)
        passes = sum(int(row.get("num_valid_candidates", 0)) for row in decision_rows)
        accepted = sum(bool(row.get("accepted")) for row in trace)
        terminal_rows = [row for row in trace if bool(row.get("terminal_stop", False))]
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
                    for key, value in (row.get("candidate_rejection_reasons", {}) or {}).items()
                }
            )

        pipeline_record: dict[str, Any] = {
            "pipeline_mode": "topology",
            "guidance_mode": guidance_mode,
            "invariant_feasible": 1.0,
            "constructor_success": 1.0,
            "accepted_swaps": accepted,
            "runtime_seconds": runtime,
            "fallback_used": float(
                bool(sampling_diagnostics.get("fallback_used", False))
                or bool(sampling_diagnostics.get("repair_used", False))
            ),
            "degree_raw_graphical": float(bool(sampling_diagnostics.get("raw_graphical", True))),
            "degree_raw_connected_feasible": float(
                bool(sampling_diagnostics.get("raw_connected_feasible", True))
            ),
            "degree_sampling_attempts": int(sampling_diagnostics.get("attempts_used", 1)),
            "degree_repair_used": float(bool(sampling_diagnostics.get("repair_used", False))),
            "degree_repair_l1": int(sampling_diagnostics.get("repair_l1_adjustment", 0)),
            "candidate_proposals": proposals,
            "candidate_passes": passes,
            "candidate_pass_rate": float(passes / max(proposals, 1)),
            "prediction_calls": prediction_calls,
            "accepted_swaps_per_prediction": float(accepted / max(prediction_calls, 1)),
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
        if guidance_mode == "spectral_graphlet":
            pipeline_record["spectral_error"] = float(predictor_spectral_error)
            pipeline_record["graphlet_error"] = float(predictor_graphlet_error)
        elif guidance_mode == "flow_graphlet":
            pipeline_record["flow_error"] = float(predictor_flow_error)
            pipeline_record["graphlet_error"] = float(predictor_graphlet_error)
        elif guidance_mode == "spectral":
            pipeline_record["spectral_error"] = float(predictor_spectral_error)
        else:
            pipeline_record.update(
                {
                    "graphlet_error": float(predictor_graphlet_error),
                    "clustering_error": predictor_clustering_error,
                    "orbit_log_error": predictor_orbit_log_error,
                }
            )
        if accepted > 0:
            pipeline_record["proposals_per_accepted_swap"] = float(proposals / accepted)
        pipeline_records.append(pipeline_record)
        print(
            f"graph={index + 1}/{num_generate} guidance={guidance_mode} "
            f"n={refined.number_of_nodes()} m={refined.number_of_edges()} "
            f"accepted_steps={accepted} prediction_calls={prediction_calls} "
            f"plateau_refreshes={plateau_refreshes} runtime={runtime:.3f}s",
            flush=True,
        )

    evaluation_cfg = dict(config.get("evaluation", {}) or {})
    inline_evaluation = bool(evaluation_cfg.get("inline_during_generation", False))
    orca_exec = None
    coarse_metrics: dict[str, Any] = {}
    refined_metrics: dict[str, Any] = {}
    if inline_evaluation:
        compute_orbit = bool(evaluation_cfg.get("compute_orbit", True))
        graphlet_backend = str(evaluation_cfg.get("graphlet_backend", "sampled")).lower()
        if compute_orbit or graphlet_backend in {"orca", "exact_orca", "exact"}:
            orca_exec = configure_orca_executable(evaluation_cfg.get("orca_exec"), required=True)
        references = reference_graphs[:num_generate] or reference_graphs
        metric_kwargs = {
            "compute_orbit": compute_orbit,
            "compute_graphlet_history": bool(evaluation_cfg.get("compute_graphlet_history", True)),
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
                "graphlet_num_samples",
                summary_config.graphlet_num_samples,
            ),
            "graphlet_backend": graphlet_backend,
        }
        coarse_metrics = evaluate_graph_sets(references, coarse_graphs, train_graphs, **metric_kwargs)
        refined_metrics = evaluate_graph_sets(references, refined_graphs, train_graphs, **metric_kwargs)

    aggregated_pipeline = aggregate_pipeline_diagnostics(
        pipeline_records,
        require_complete=True,
        allow_fallback=False,
    )
    accepted_steps = [sum(bool(row.get("accepted")) for row in trace) for trace in traces]
    trace_rows = [row for trace in traces for row in trace]
    accepted_rows = [row for row in trace_rows if bool(row.get("accepted"))]
    prediction_refresh_rows = [
        row for row in trace_rows if bool(row.get("prediction_refreshed", False))
    ]
    prediction_call_counts = [
        max((int(row.get("prediction_calls", 0)) for row in trace), default=0)
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

    diagnostics: dict[str, Any] = {
        "pipeline_mode": "topology",
        "guidance_mode": guidance_mode,
        "degree_preservation_rate": degree_preservation_rate(coarse_graphs, refined_graphs),
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
        "prediction_horizon_schedule": refiner_settings.prediction_horizon_schedule,
        "prediction_horizon_initial_k": refiner_settings.prediction_horizon_initial_k,
        "prediction_horizon_final_k": refiner_settings.prediction_horizon_final_k,
        "mean_realized_prediction_horizon": (
            float(np.mean(prediction_horizons)) if prediction_horizons else 0.0
        ),
        "mean_prediction_calls": float(np.mean(prediction_call_counts)),
        "mean_accepted_swaps_per_prediction_call": float(
            sum(accepted_steps) / max(sum(prediction_call_counts), 1)
        ),
        "plateau_refresh_count": int(plateau_refresh_count),
        "mean_graph_runtime_seconds": float(np.mean(graph_runtimes)),
        "runtime_seconds": float(time.perf_counter() - run_started),
        "inline_evaluation": inline_evaluation,
    }
    if guidance_mode == "spectral_graphlet":
        diagnostics.update(
            {
                "mean_accepted_spectral_gain": _mean_or_zero(accepted_rows, "spectral_gain"),
                "mean_accepted_clean_spectral_gain": _mean_or_zero(accepted_rows, "clean_spectral_gain"),
                "mean_accepted_graphlet_gain": _mean_or_zero(accepted_rows, "graphlet_gain"),
                "mean_accepted_clean_graphlet_gain": _mean_or_zero(accepted_rows, "clean_graphlet_gain"),
                "mean_projection_residual": _mean_or_zero(accepted_rows, "projection_residual"),
                "mean_spectral_projection_residual": _mean_or_zero(accepted_rows, "spectral_projection_residual"),
                "mean_graphlet_projection_residual": _mean_or_zero(accepted_rows, "graphlet_projection_residual"),
                "mean_spectral_weight": _mean_or_zero(accepted_rows, "spectral_weight"),
                "mean_graphlet_weight": _mean_or_zero(accepted_rows, "graphlet_weight"),
                "mean_spectral_clean_mix": _mean_or_zero(accepted_rows, "spectral_clean_mix"),
                "mean_graphlet_clean_mix": _mean_or_zero(accepted_rows, "graphlet_clean_mix"),
                "mean_bridge_expansions": _mean_or_zero(accepted_rows, "bridge_expansions"),
                "predictor_spectral_normalized_error": float(predictor_spectral_error),
                "predictor_graphlet_logit_error": float(predictor_graphlet_error),
                "spectral_distance": refiner_settings.distance,
                "graphlet_distance": refiner_settings.graphlet_distance,
                "spectral_normalization": refiner_settings.normalization,
                "spectral_bridge_schedule": refiner_settings.bridge_schedule,
                "graphlet_bridge_schedule": refiner_settings.graphlet_bridge_schedule,
                "global_to_local_schedule": refiner_settings.guidance_weight_schedule,
                "spectral_debug_enabled": refiner_settings.debug_enabled,
            }
        )
        refresh_on_plateau = refiner_settings.refresh_on_prediction_plateau
        report_format = "topology_spectral_graphlet_generation_v1"
    elif guidance_mode == "flow_graphlet":
        diagnostics.update(
            {
                "mean_accepted_flow_gain": _mean_or_zero(
                    accepted_rows, "flow_gain"
                ),
                "mean_accepted_graphlet_gain": _mean_or_zero(
                    accepted_rows, "graphlet_gain"
                ),
                "mean_flow_weight": _mean_or_zero(accepted_rows, "flow_weight"),
                "mean_graphlet_weight": _mean_or_zero(
                    accepted_rows, "graphlet_weight"
                ),
                "mean_graphlet_clean_mix": _mean_or_zero(
                    accepted_rows, "graphlet_clean_mix"
                ),
                "mean_soft_degree_residual": _mean_or_zero(
                    accepted_rows, "soft_degree_residual"
                ),
                "mean_predicted_flow_degree_tangent_residual": _mean_or_zero(
                    accepted_rows,
                    "predicted_flow_degree_tangent_residual",
                ),
                "predictor_flow_error": float(predictor_flow_error),
                "predictor_graphlet_logit_error": float(predictor_graphlet_error),
                "graphlet_distance": refiner_settings.graphlet_distance,
                "graphlet_bridge_schedule": (
                    refiner_settings.graphlet_bridge_schedule
                ),
                "global_to_local_schedule": (
                    refiner_settings.guidance_weight_schedule
                ),
                "flow_normalize_per_swap": (
                    refiner_settings.flow_normalize_per_swap
                ),
            }
        )
        refresh_on_plateau = refiner_settings.refresh_on_prediction_plateau
        report_format = "topology_flow_graphlet_generation_v1"
    elif guidance_mode == "spectral":
        diagnostics.update(
            {
                "mean_accepted_spectral_gain": _mean_or_zero(accepted_rows, "spectral_gain"),
                "mean_accepted_clean_spectral_gain": _mean_or_zero(
                    accepted_rows,
                    "clean_spectral_gain",
                ),
                "mean_projection_residual": _mean_or_zero(
                    accepted_rows,
                    "projection_residual",
                ),
                "mean_bridge_clean_mix": _mean_or_zero(accepted_rows, "clean_mix"),
                "mean_bridge_expansions": _mean_or_zero(
                    accepted_rows,
                    "bridge_expansions",
                ),
                "predictor_spectral_normalized_error": float(predictor_spectral_error),
                "spectral_distance": refiner_settings.distance,
                "spectral_normalization": refiner_settings.normalization,
                "spectral_bridge_schedule": refiner_settings.bridge_schedule,
                "spectral_debug_enabled": refiner_settings.debug_enabled,
            }
        )
        refresh_on_plateau = refiner_settings.refresh_on_prediction_plateau
        report_format = "topology_spectral_generation_v1"
    else:
        diagnostics.update(
            {
                "mean_accepted_graphlet_gain": _mean_or_zero(accepted_rows, "graphlet_gain"),
                "mean_accepted_clustering_gain": _mean_or_zero(
                    accepted_rows,
                    "clustering_gain",
                ),
                "mean_accepted_orbit_gain": _mean_or_zero(accepted_rows, "orbit_gain"),
                "mean_accepted_structural_gain": _mean_or_zero(
                    accepted_rows,
                    "structural_gain",
                ),
                "predictor_graphlet_error": float(predictor_graphlet_error),
                "predictor_clustering_error": predictor_clustering_error,
                "predictor_orbit_log_error": predictor_orbit_log_error,
            }
        )
        refresh_on_plateau = refiner_settings.refresh_on_plateau
        report_format = "topology_structural_generation_v2"

    report = {
        "format": report_format,
        "pipeline_mode": "topology",
        "guidance_mode": guidance_mode,
        "checkpoint_format": checkpoint.get("format"),
        "degree_source": degree_source,
        "prediction_horizon": {
            "mode": refiner_settings.prediction_horizon_mode,
            "initial_k": refiner_settings.prediction_horizon_initial_k,
            "final_k": refiner_settings.prediction_horizon_final_k,
            "schedule": refiner_settings.prediction_horizon_schedule,
            "refresh_on_plateau": refresh_on_plateau,
            "min_improvement": refiner_settings.min_improvement,
            "min_relative_improvement": refiner_settings.min_relative_improvement,
        },
        "orca_exec": orca_exec,
        "num_generated": len(refined_graphs),
        "hh_source": coarse_metrics,
        "topology_refined": refined_metrics,
        "coarse": coarse_metrics,
        "hybrid_refined": refined_metrics,
        "diagnostics": diagnostics,
        "pipeline_diagnostics": aggregated_pipeline,
        "pipeline_records": pipeline_records,
        "traces": traces,
        "seed": seed,
        "config_overrides": list(args.config_overrides),
        "rng_streams": {
            "source_and_refiner_decoupled": True,
            "refiner_rng_per_graph": True,
        },
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

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
from grapher.models.dhvae_hh.degree_perturbation import (
    DegreePerturbationError,
    PerturbedEmpiricalDegreeSampler,
    sequence_fingerprint,
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
from grapher.rewiring_mlp.generic.joint_edge_spectral_generation import (
    JointEdgeSpectralRefinerConfig,
    sample_soft_endpoint as sample_joint_edge_spectral_endpoint,
    refine_graph as refine_joint_edge_spectral_graph,
)
from grapher.rewiring_mlp.generic.spectral_graphlet_refiner import (
    SpectralGraphletRefinerConfig,
    enrich_graph_with_degree_summary,
    predict_degree_conditioned_summary,
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


def _build_generation_degree_sampler(
    degree_source: str,
    degree_cfg: dict[str, Any],
    *,
    train_graphs: list[nx.Graph],
    reference_graphs: list[nx.Graph],
    seed: int,
    perturbation_cfg: dict[str, Any] | None = None,
    support_max_degree: int | None = None,
    parent_failure_policy: str = "error",
    max_parent_attempts: int = 128,
):
    """Build the configured ordinary-degree sampler for generic generation.

    ``test_oracle`` is handled per reference graph in the generation loop and
    therefore intentionally returns ``None``. ``test_empirical`` samples only
    from held-out degree sequences and is intended for an explicitly labelled
    diagnostic, not unconditional main-table generation.
    """

    source = str(degree_source).lower()
    if source not in {"train_empirical_perturbed", "edge_relocation"} and (
        parent_failure_policy != "error" or max_parent_attempts != 128
    ):
        raise ValueError("Parent retry settings require degree_source=train_empirical_perturbed or edge_relocation.")
    cfg = dict(degree_cfg or {})
    degree_type = str(cfg.get("type", "degree_histogram_vae")).lower()
    if "typed" in degree_type:
        raise ValueError(
            "The generic topology stage requires the ordinary DH-VAE."
        )

    if source in {"learned", "degree_vae"}:
        if str(cfg.get("postprocess_policy", "")).lower() != "reject_only":
            raise ValueError(
                "Learned topology generation requires degree_generator."
                "postprocess_policy: reject_only."
            )
        if str(cfg.get("fallback", "")).lower() != "error":
            raise ValueError(
                "Learned topology generation requires degree_generator.fallback: error."
            )
        cfg["enabled"] = True
        return build_degree_sampler(cfg, train_graphs, seed=seed)

    if source in {"train_empirical_perturbed", "edge_relocation"}:
        settings = dict(perturbation_cfg or {})
        if source == "edge_relocation":
            conflicting = {k: settings[k] for k in ("method", "probability", "steps", "failure_policy") if k in settings}
            expected = {"method": "edge_relocation", "probability": 1.0, "steps": 1, "failure_policy": "error"}
            for key, value in conflicting.items():
                if value != expected[key]:
                    raise ValueError(f"generation.degree_source=edge_relocation fixes {key}={expected[key]!r}; received {value!r}.")
            settings.update(expected)
        return PerturbedEmpiricalDegreeSampler.fit_from_graphs(
            train_graphs, settings, seed=seed,
            support_max_degree=support_max_degree,
            parent_failure_policy=parent_failure_policy,
            max_parent_attempts=max_parent_attempts,
        )

    if source in {"empirical", "train_empirical"}:
        return EmpiricalDegreeSampler.fit_from_graphs(train_graphs, seed=seed)

    if source == "test_empirical":
        if not reference_graphs:
            raise ValueError(
                "generation.degree_source=test_empirical requires a non-empty test split."
            )
        return EmpiricalDegreeSampler.fit_from_graphs(reference_graphs, seed=seed)

    if source in {"oracle", "test_oracle"}:
        return None

    raise ValueError(f"Unknown generation.degree_source: {source!r}")


def _checkpoint_format(path: str | Path) -> str:
    checkpoint = torch.load(Path(path), map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Topology checkpoint must be a mapping: {path}")
    return str(checkpoint.get("format", ""))


def _generation_degree_prior_report(
    degree_sampler, *, degree_source, seed, degree_rng_mode,
    degree_sampling_records, target_degree_sequences,
):
    """Separate all prior attempts, accepted summaries, and completed graphs."""
    if isinstance(degree_sampler, PerturbedEmpiricalDegreeSampler):
        report = degree_sampler.report()
        report["sampling_scope"] = "all_prior_attempts_including_constructor_rejections"
    else:
        report = {"format": "degree_prior_audit_v1", "degree_source": degree_source,
                  "num_samples": len(degree_sampling_records)}
    report.update({
        "degree_source": degree_source, "seed": seed, "degree_rng_mode": degree_rng_mode,
        "num_returned": len(degree_sampling_records),
        "returned_parent_degree_fingerprint": sequence_fingerprint(r["parent_degree_sequence"] for r in degree_sampling_records),
        "returned_degree_fingerprint": sequence_fingerprint(target_degree_sequences),
        "returned_records": degree_sampling_records,
    })
    return report


def _mean_or_zero(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if key in row and row[key] is not None]
    return float(np.mean(values)) if values else 0.0


def _guidance_diagnostic_summary(
    settings: SpectralRefinerConfig,
    accepted_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Separate scored components from optional measurements on selected moves.

    All discrepancies are against the prediction frozen for ONE decision, not a
    trajectory-wide target. A missing/inactive measurement is null, never zero.
    """
    components = set(settings.guidance_mode.split("_"))
    result: dict[str, Any] = {
        "edge_guidance_weight": getattr(settings, "edge_weight", 0.0) if "edge" in components else 0.0,
        "spectral_guidance_weight": settings.spectral_weight if "spectral" in components else 0.0,
        "clustering_guidance_weight": settings.clustering_weight if "clustering" in components else 0.0,
        "orbit_guidance_weight": settings.orbit_weight if "orbit" in components else 0.0,
        "induced_graphlet_guidance_weight": settings.induced_graphlet_weight if "graphlet" in components else 0.0,
        "induced_graphlet_guidance_k": settings.induced_graphlet_k if "graphlet" in components else None,
        "induced_graphlet_guidance_scope": settings.induced_graphlet_scope if "graphlet" in components else None,
        "induced_graphlet_guidance_distance": settings.induced_graphlet_distance if "graphlet" in components else None,
        "cycle_guidance_weight": settings.cycle_weight if "cycle" in components else 0.0,
        "cycle_guidance_k": settings.cycle_k if "cycle" in components else None,
        "cycle_guidance_distance": settings.cycle_distance if "cycle" in components else None,
        "scoring_components": sorted(components),
        "discrepancy_scope": "same_step_frozen_prediction; accepted_moves_only",
        "candidate_spectral_diagnostics_requested": settings.compute_candidate_spectral_diagnostics,
        "accepted_spectral_diagnostics_computed": bool(accepted_rows),
    }
    for component in ("edge", "clustering", "orbit", "cycle", "graphlet"):
        active = component in components
        measured = [r for r in accepted_rows if active and r.get(f"current_{component}_discrepancy") is not None]
        result[f"{component}_diagnostics_computed"] = bool(measured)
        result[f"accepted_{component}_measurements"] = len(measured)
        gains = [float(r[f"{component}_gain"]) for r in measured]
        for suffix, key in (
            ("gain", f"{component}_gain"),
            ("discrepancy_before", f"current_{component}_discrepancy"),
            ("discrepancy_after", f"candidate_{component}_discrepancy"),
        ):
            result[f"mean_accepted_{component}_{suffix}"] = (
                float(np.mean([float(r[key]) for r in measured])) if measured else None
            )
        result[f"accepted_{component}_improved_fraction"] = (
            float(np.mean(np.asarray(gains) > 1e-10)) if gains else None
        )
        result[f"accepted_{component}_worsened_fraction"] = (
            float(np.mean(np.asarray(gains) < -1e-10)) if gains else None
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate generic graph topologies with DH-VAE/empirical degrees, "
            "connected Havel-Hakimi construction, and GraphER rewiring. The "
            "refiner is selected automatically from the checkpoint format "
            "(structural-summary, spectral, or joint spectral + graphlet-logit "
            "diffusion guidance)."
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
    # Keep construction/refinement as the first two streams so their seeded
    # behavior stays unchanged when source enrichment is enabled.
    source_seed_sequence, refiner_seed_sequence, enrichment_seed_sequence = (
        seed_sequence.spawn(3)
    )
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
    max_total_graph_attempts = generation_cfg.get("max_total_graph_attempts", 10 * num_generate)
    if (isinstance(max_total_graph_attempts, bool)
            or not isinstance(max_total_graph_attempts, (int, np.integer))
            or max_total_graph_attempts < num_generate):
        raise ValueError("generation.max_total_graph_attempts must be an integer >= num_generate.")
    max_total_graph_attempts = int(max_total_graph_attempts)
    # Extending the child streams preserves the original first num_generate
    # seeds and gives every replacement attempt its own refinement randomness.
    refiner_graph_seeds = refiner_seed_sequence.spawn(max_total_graph_attempts)
    enrichment_graph_seeds = enrichment_seed_sequence.spawn(max_total_graph_attempts)

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
    elif checkpoint_format == TOPOLOGY_SPECTRAL_CHECKPOINT_FORMAT:
        guidance_mode = "spectral"
        model, summary_config, checkpoint = load_topology_spectral_checkpoint(
            checkpoint_path,
            device=device,
        )
        predictor_report = checkpoint.get("report", {}) or {}
        if str(getattr(model, "spectral_representation", "eigenvalues")) == "heat_kernel":
            predictor_spectral_error_raw = predictor_report.get(
                "val_heat_kernel_rmse", predictor_report.get("val_heat_kernel_mae")
            )
            missing_message = "val_heat_kernel_rmse/mae"
        else:
            predictor_spectral_error_raw = predictor_report.get(
                "val_spectral_normalized_rmse",
                predictor_report.get("val_spectral_normalized_mae"),
            )
            missing_message = "val_spectral_normalized_rmse/mae"
        if predictor_spectral_error_raw is None:
            raise ValueError(
                "The spectral checkpoint is missing held-out " + missing_message +
                "; generate from a checkpoint selected by train_topology_grapher.py."
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
            f"{TOPOLOGY_SPECTRAL_GRAPHLET_CHECKPOINT_FORMAT!r}."
        )
    model_device = next(model.parameters()).device
    joint_edge_diffusion = bool(
        guidance_mode == "spectral"
        and (
            getattr(model, "predict_edge_state", False)
            or str(getattr(model, "spectral_representation", "eigenvalues")) == "heat_kernel"
        )
    )

    degree_source = str(generation_cfg.get("degree_source", "learned")).lower()
    perturbation_cfg = dict(generation_cfg.get("degree_perturbation", {}) or {})
    if degree_source not in {"train_empirical_perturbed", "edge_relocation"} and any(
        key in generation_cfg for key in ("degree_failure_policy", "max_degree_parent_attempts")
    ):
        raise ValueError("generation.degree_failure_policy and max_degree_parent_attempts require a perturbed training-degree prior.")
    if perturbation_cfg and degree_source not in {"train_empirical_perturbed", "edge_relocation"}:
        raise ValueError("generation.degree_perturbation requires degree_source=train_empirical_perturbed or edge_relocation.")
    degree_rng_mode = str(generation_cfg.get(
        "degree_rng_mode", "independent" if degree_source in {"train_empirical_perturbed", "edge_relocation"} else "legacy"
    )).lower()
    if degree_rng_mode not in {"legacy", "independent"}:
        raise ValueError("generation.degree_rng_mode must be legacy or independent.")
    if degree_source in {"train_empirical_perturbed", "edge_relocation"} and degree_rng_mode != "independent":
        raise ValueError("Perturbed/edge-relocation degrees require degree_rng_mode=independent for parent pairing.")
    # The old first-three construction/refinement/enrichment streams are untouched.
    # The new control and all perturbation variants share this separate parent RNG.
    degree_rng = (np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(3,)))
                  if degree_rng_mode == "independent" else source_rng)
    degree_cfg = dict(config.get("degree_generator", {}) or {})
    joint_degree_enabled = bool(getattr(model, "joint_degree_enabled", False))
    if bool((config.get("joint_degree", {}) or {}).get("enabled", False)) and not joint_degree_enabled:
        raise ValueError("Joint-degree config requires a joint checkpoint; train with this config first.")
    checkpoint_selection = (checkpoint.get("report", {}) or {}).get("checkpoint_selection")
    # Prior ablations require checkpoint provenance for every predictor family,
    # including the multi-size spectral+graphlet model (not only joint DH-VAE).
    from grapher.rewiring_mlp.generic.joint_checkpointing import file_sha256
    checkpoint_file_sha256 = file_sha256(checkpoint_path)
    if joint_degree_enabled:
        from grapher.rewiring_mlp.generic.joint_degree_training import graph_fingerprint
        recorded = dict((checkpoint.get("report", {}) or {}).get("dataset_graph_fingerprints", {}) or {})
        for split_name, expected in recorded.items():
            if graph_fingerprint(list(splits.get(split_name, []))) != expected:
                raise ValueError(f"Joint checkpoint dataset fingerprint mismatch for {split_name}; refusing stale-dataset generation.")
    if joint_degree_enabled and degree_source in {"learned", "degree_vae"}:
        from grapher.rewiring_mlp.generic.joint_degree_model import build_embedded_degree_sampler
        degree_sampler = build_embedded_degree_sampler(model, degree_cfg, seed=seed)
        degree_sampler_source = "joint_checkpoint_embedded"
    else:
        degree_sampler = _build_generation_degree_sampler(
            degree_source, degree_cfg, train_graphs=train_graphs,
            reference_graphs=reference_graphs, seed=seed,
            perturbation_cfg=perturbation_cfg,
            support_max_degree=(int(model.degree_vectorizer.max_degree) if joint_degree_enabled else None),
            parent_failure_policy=generation_cfg.get("degree_failure_policy", "error"),
            max_parent_attempts=generation_cfg.get("max_degree_parent_attempts", 128),
        )
        degree_sampler_source = "external_checkpoint" if degree_source in {"learned", "degree_vae"} else degree_source
    if joint_degree_enabled:
        print(f"[GraphER/JointDegree] degree_source={degree_source} sampler={degree_sampler_source}; "
              f"conditioning=realized_degree_histogram; orbit_consistency={model.orbit_consistency}", flush=True)

    if isinstance(degree_sampler, PerturbedEmpiricalDegreeSampler):
        suffix = (
            "; temporary connected HH witness is used only to relocate an endpoint; "
            "its adjacency is discarded and the actual GraphER source is reconstructed from the new degrees"
            if degree_sampler.config.method == "edge_relocation" else ""
        )
        print(f"[GraphER/DegreePerturbation] method={degree_sampler.config.method} "
              f"probability={degree_sampler.config.probability} steps={degree_sampler.config.steps} "
              f"failure_policy={degree_sampler.config.failure_policy}; "
              "training degrees only; n,m preserved; actual changed degrees condition predictor" + suffix, flush=True)
        if degree_sampler.parent_failure_policy == "resample_parent":
            print("[GraphER/DegreePerturbation] degree_failure_policy=resample_parent "
                  f"max_degree_parent_attempts={degree_sampler.max_parent_attempts}; "
                  "accepted parents are conditioned on successful perturbations; rejected draws are recorded.", flush=True)

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
            if joint_edge_diffusion:
                refiner_settings = JointEdgeSpectralRefinerConfig.from_dict(refiner_cfg, model=model)
                active_components = set(refiner_settings.guidance_mode.split("_"))
                print(
                    "[GraphER/JointSpectral] loaded generic soft-edge / spectral-space diffusion "
                    f"checkpoint representation={getattr(model, 'spectral_representation', 'eigenvalues')}; "
                    "hard realization preserves indexed degrees.",
                    flush=True,
                )
            else:
                refiner_settings = SpectralRefinerConfig.from_dict(refiner_cfg)
                active_components = set(refiner_settings.guidance_mode.split("_"))
            if "clustering" in active_components:
                if refiner_settings.clustering_statistic == "histogram":
                    if not getattr(model, "predict_clustering_histogram", False):
                        raise ValueError("Histogram guidance requested, but checkpoint has no histogram head. Train with structure_summary_prediction.clustering_histogram=true.")
                    bins = refiner_settings.clustering_histogram_bins
                    if bins is not None and int(bins) != model.clustering_histogram_bins:
                        raise ValueError("clustering_guidance.histogram_bins disagrees with the checkpoint.")
                elif not getattr(model, "predict_clustering_coefficient", False):
                    raise ValueError("Mean-clustering guidance requested, but checkpoint has no scalar clustering head.")
            if "graphlet" in active_components:
                if not getattr(model, "predict_induced_graphlet_histogram", False):
                    raise ValueError("Graphlet guidance requested but checkpoint has no induced graphlet head; train a new checkpoint.")
                if not joint_edge_diffusion:
                    from grapher.rewiring_mlp.generic.induced_graphlets import InducedGraphletSpec
                    if InducedGraphletSpec(refiner_settings.induced_graphlet_k, refiner_settings.induced_graphlet_scope) != model.induced_graphlet_spec:
                        raise ValueError("Induced graphlet guidance catalogue differs from the checkpoint.")
            if "cycle" in active_components:
                if not getattr(model, "predict_cycle_graphlet_histogram", False):
                    raise ValueError("Cycle guidance requested, but checkpoint has no cycle graphlet histogram head. Train with structure_summary_prediction.cycle_graphlet_histogram=true.")
                if refiner_settings.cycle_k != model.cycle_graphlet_k:
                    raise ValueError("cycle_guidance.k does not match the checkpoint.")
            if "orbit" in active_components and not getattr(model, "predict_orbit_summary", False):
                raise ValueError(
                    "Orbit guidance requested, but checkpoint has no orbit-summary head. "
                    "Train with structure_summary_prediction.orbit_summary=true."
                )
            print(
                f"[GraphER/Spectral] rewiring_guidance={refiner_settings.guidance_mode} "
                f"clustering_statistic={refiner_settings.clustering_statistic} "
                f"histogram_bins={getattr(model, 'clustering_histogram_bins', None) if getattr(model, 'predict_clustering_histogram', False) else None} "
                f"orbit_summary={getattr(model, 'predict_orbit_summary', False)} "
                f"weights=(spectral={refiner_settings.spectral_weight}, "
                f"clustering={refiner_settings.clustering_weight}, orbit={refiner_settings.orbit_weight})",
                flush=True,
            )
            print(
                "[GraphER/Spectral] loaded Spectral Transformer checkpoint "
                f"format={checkpoint_format} device={model_device}",
                flush=True,
            )
            print(
                "[GraphER/Spectral] predictor uses current/source spectra; "
                f"valid degree-preserving swaps are ranked by {refiner_settings.guidance_mode}. "
                "Predicted targets are frozen within each decision and refreshed after the configured horizon.",
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
    else:
        refiner_settings = TopologyRefinerConfig.from_dict(refiner_cfg)
        if not refiner_settings.preserve_connectivity:
            raise ValueError("Topology generation requires topology_refiner.preserve_connectivity.")

    source_enrichment_cfg = dict(config.get("source_enrichment", {}) or {})
    source_enrichment_enabled = bool(source_enrichment_cfg.get("enabled", False))
    source_enrichment_settings = None
    if source_enrichment_enabled:
        if guidance_mode != "spectral_graphlet":
            raise ValueError(
                "Degree-conditioned source enrichment currently requires the "
                "spectral_graphlet predictor."
            )
        if not bool(getattr(model, "degree_summary_enabled", False)):
            raise ValueError(
                "source_enrichment.enabled requires a checkpoint trained with "
                "source_enrichment.summary_estimator.enabled: true."
            )
        source_enrichment_settings = SpectralGraphletRefinerConfig.from_dict(
            dict(source_enrichment_cfg.get("rewiring", {}) or {})
        )

    coarse_graphs: list[nx.Graph] = []
    enriched_base_graphs: list[nx.Graph] = []
    refined_graphs: list[nx.Graph] = []
    target_degree_sequences: list[list[int]] = []
    degree_sampling_records: list[dict[str, Any]] = []
    traces: list[list[dict[str, Any]]] = []
    enrichment_traces: list[list[dict[str, Any]]] = []
    graph_runtimes: list[float] = []
    pipeline_records: list[dict[str, Any]] = []
    skipped_graphs: list[dict[str, Any]] = []
    source_rejection_totals: Counter[str] = Counter()
    num_attempted = 0
    max_attempts_per_graph = int(generation_cfg.get("max_attempts_per_graph", 8))
    if max_attempts_per_graph <= 0:
        raise ValueError("generation.max_attempts_per_graph must be positive.")

    def generation_completion():
        return {
            "num_requested": num_generate, "num_attempted": num_attempted,
            "num_generated": len(refined_graphs), "num_skipped": len(skipped_graphs),
            "skipped_graphs": skipped_graphs,
            "requested_count_reached": len(refined_graphs) == num_generate,
            "max_total_graph_attempts": max_total_graph_attempts,
            "replacement_attempts": max(0, num_attempted - num_generate),
            "replacement_budget_exhausted": (
                len(refined_graphs) < num_generate and num_attempted >= max_total_graph_attempts
            ),
            "generation_success_fraction": len(refined_graphs) / max(num_attempted, 1),
            "generation_rejections": dict(sorted(source_rejection_totals.items())),
            "completed_parent_distribution": (
                "conditioned_on_successful_generation"
                if degree_source in {"empirical", "train_empirical", "train_empirical_perturbed",
                                     "edge_relocation", "test_empirical"} else None
            ),
        }

    def skip_graph(index, stage, attempts_used, reason, rejections):
        skipped_graphs.append({
            "generation_index": index, "stage": stage, "attempts_used": attempts_used,
            "reason": str(reason), "generation_rejections": dict(rejections),
            "runtime_seconds": float(time.perf_counter() - graph_started),
        })
        action = ("resampling a replacement" if num_attempted < max_total_graph_attempts
                  else "total graph attempt budget exhausted")
        print(f"attempt={index + 1}/{max_total_graph_attempts} generated={len(refined_graphs)}/{num_generate} "
              f"skipped=True stage={stage} attempts={attempts_used} reason={reason}; {action}.", flush=True)

    def save_source_failure(error, index, rejections):
        output_dir = ensure_dir(args.output_dir)
        failed_report = _generation_degree_prior_report(
            degree_sampler, degree_source=degree_source, seed=seed, degree_rng_mode=degree_rng_mode,
            degree_sampling_records=degree_sampling_records, target_degree_sequences=target_degree_sequences,
        )
        completion = generation_completion()
        completion["generation_rejections"] = dict(sorted((source_rejection_totals + Counter(rejections)).items()))
        failed_report.update(completion, complete=False, generation_aborted=True, failure=str(error))
        save_json(failed_report, output_dir / "degree_prior_report.json")
        save_pickle(coarse_graphs, output_dir / "partial_coarse_graphs.pkl")
        save_pickle(refined_graphs, output_dir / "partial_topology_refined_graphs.pkl")
        if source_enrichment_enabled:
            save_pickle(enriched_base_graphs, output_dir / "partial_enriched_base_graphs.pkl")
        save_json(target_degree_sequences, output_dir / "partial_sampled_degree_sequences.json")
        save_json({
            "format": "topology_partial_generation_v1", "complete": False,
            **completion,
            "failure": str(error), "failed_generation_index": index,
            "seed": seed, "degree_source": degree_source,
            "degree_rng_mode": degree_rng_mode, "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_file_sha256, "config": config,
            "degree_prior_report_file": "degree_prior_report.json",
            "parent_degree_fingerprint": failed_report["returned_parent_degree_fingerprint"],
            "sampled_degree_fingerprint": failed_report["returned_degree_fingerprint"],
            "pipeline_records": pipeline_records, "traces": traces,
        }, output_dir / "partial_report.json")

    for index in range(max_total_graph_attempts):
        if len(refined_graphs) == num_generate:
            break
        num_attempted = index + 1
        graph_started = time.perf_counter()
        generation_rejections: Counter[str] = Counter()
        source_ready = False
        for generation_attempt in range(1, max_attempts_per_graph + 1):
            prior_draws_before = len(degree_sampler.records) if isinstance(degree_sampler, PerturbedEmpiricalDegreeSampler) else 0
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
                    degree_summary = degree_sampler.sample(degree_rng)
            except DegreePerturbationError as exc:
                if not exc.sampling_failure:
                    save_source_failure(exc, index, generation_rejections)
                    raise
                # The sampler has exhausted this request. A later attempt will
                # draw a replacement while preserving every sampling constraint.
                draws = len(degree_sampler.records) - prior_draws_before
                generation_rejections["degree_prior_rejected"] += draws
                skip_graph(index, "degree_prior", draws, exc, generation_rejections)
                break
            except RuntimeError as exc:
                if not str(exc).startswith("Degree generator exhausted "):
                    save_source_failure(exc, index, generation_rejections)
                    raise
                generation_rejections["degree_prior_rejected"] += 1
                continue

            try:
                coarse = construct_coarse_graph(degree_summary, constructor_cfg, source_rng)
                assert_constructor_validity(coarse, degree_summary, require_connected=True)
            except (ValueError, RuntimeError, AssertionError):
                generation_rejections["constructor_rejected"] += 1
                continue
            source_ready = True
            break
        else:
            error = RuntimeError(
                "Topology generation exhausted "
                f"{max_attempts_per_graph} attempts for graph {index}; "
                f"rejections={dict(generation_rejections)}."
            )
            stage = "source_construction" if generation_rejections["constructor_rejected"] else "degree_prior"
            skip_graph(index, stage, max_attempts_per_graph, error, generation_rejections)

        source_rejection_totals.update(generation_rejections)
        if not source_ready:
            continue

        if isinstance(degree_sampler, PerturbedEmpiricalDegreeSampler):
            degree_summary["sampling_diagnostics"]["returned_generation_index"] = index
        generation_context = (f"output={len(refined_graphs) + 1}/{num_generate} "
                              f"attempt={index + 1}/{max_total_graph_attempts}")
        base_graph = coarse
        enrichment_trace: list[dict[str, Any]] = []
        if source_enrichment_enabled:
            assert graphlet_basis is not None
            assert source_enrichment_settings is not None
            degree_target = predict_degree_conditioned_summary(
                model,
                coarse,
                graphlet_basis=graphlet_basis,
                device=model_device,
                graphlet_logit_epsilon=source_enrichment_settings.graphlet_logit_epsilon,
            )
            base_graph, enrichment_trace = enrich_graph_with_degree_summary(
                coarse,
                target=degree_target,
                graphlet_basis=graphlet_basis,
                refiner_config=source_enrichment_settings,
                device=model_device,
                rng=np.random.default_rng(enrichment_graph_seeds[index]),
                return_trace=True,
            )

        if guidance_mode == "spectral_graphlet":
            assert graphlet_basis is not None
            refined, trace = refine_graph_with_spectral_graphlet_predictions(
                base_graph,
                model=model,
                graphlet_basis=graphlet_basis,
                refiner_config=refiner_settings,
                device=model_device,
                rng=np.random.default_rng(refiner_graph_seeds[index]),
                return_trace=True,
                conditioning_graph=coarse,
                debug_context=(
                    f"{generation_context} "
                    f"n={base_graph.number_of_nodes()} m={base_graph.number_of_edges()}"
                ),
            )
        elif guidance_mode == "spectral":
            if joint_edge_diffusion:
                bridge_seed = int(np.random.default_rng(refiner_graph_seeds[index]).integers(0, 2**31 - 1))
                targets, bridge_report = sample_joint_edge_spectral_endpoint(
                    model, base_graph, config, seed=bridge_seed
                )
                refined, trace = refine_joint_edge_spectral_graph(
                    base_graph, targets, model, config,
                    rng=np.random.default_rng(refiner_graph_seeds[index]),
                    prediction_calls=int(bridge_report["prediction_calls"]),
                )
                if trace:
                    trace[0]["bridge_report"] = bridge_report
            else:
                refined, trace = refine_graph_with_spectral_predictions(
                    coarse,
                    model=model,
                    refiner_config=refiner_settings,
                    device=model_device,
                    rng=np.random.default_rng(refiner_graph_seeds[index]),
                    return_trace=True,
                    debug_context=(
                        f"{generation_context} "
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
        enriched_base_graphs.append(base_graph)
        refined_graphs.append(refined)
        target_degree_sequences.append([int(value) for value in degree_summary["degree_sequence"]])
        traces.append(trace)
        enrichment_traces.append(enrichment_trace)
        graph_runtimes.append(runtime)
        sampling_diagnostics = dict(degree_summary.get("sampling_diagnostics", {}) or {})
        degree_sampling_records.append({
            "generation_index": index,
            "degree_sequence": list(degree_summary["degree_sequence"]),
            "parent_degree_sequence": list(sampling_diagnostics.get("parent_degree_sequence", degree_summary["degree_sequence"])),
            "sampling_diagnostics": sampling_diagnostics,
        })

        decision_rows = [row for row in trace if "num_proposals" in row]
        proposals = sum(int(row.get("num_proposals", 0)) for row in decision_rows)
        passes = sum(int(row.get("num_valid_candidates", 0)) for row in decision_rows)
        accepted = sum(bool(row.get("accepted")) for row in trace)
        enrichment_accepted = sum(bool(row.get("accepted")) for row in enrichment_trace)
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
            # Keep the internal aggregator's legacy family field; explicit active
            # guidance is separate. Top-level report/diagnostics use active mode.
            "guidance_mode": guidance_mode,
            "predictor_family": guidance_mode,
            "rewiring_guidance_mode": refiner_settings.guidance_mode if guidance_mode == "spectral" else guidance_mode,
            "invariant_feasible": 1.0,
            "constructor_success": 1.0,
            "accepted_swaps": accepted,
            "source_enrichment_enabled": float(source_enrichment_enabled),
            "source_enrichment_accepted_swaps": enrichment_accepted,
            "runtime_seconds": runtime,
            "fallback_used": float(
                bool(sampling_diagnostics.get("fallback_used", False))
                or bool(sampling_diagnostics.get("repair_used", False))
            ),
            "degree_raw_graphical": float(bool(sampling_diagnostics.get("raw_graphical", True))),
            "degree_raw_connected_feasible": float(
                bool(sampling_diagnostics.get("raw_connected_feasible", True))
            ),
            "degree_sampling_attempts": int(sampling_diagnostics.get("parent_attempt", sampling_diagnostics.get("attempts_used", 1))),
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
            f"generated={len(refined_graphs)}/{num_generate} attempt={index + 1}/{max_total_graph_attempts} "
            f"guidance={guidance_mode} "
            f"n={refined.number_of_nodes()} m={refined.number_of_edges()} "
            f"enrichment_steps={enrichment_accepted} accepted_steps={accepted} "
            f"prediction_calls={prediction_calls} "
            f"plateau_refreshes={plateau_refreshes} runtime={runtime:.3f}s",
            flush=True,
        )

    evaluation_cfg = dict(config.get("evaluation", {}) or {})
    inline_evaluation = bool(evaluation_cfg.get("inline_during_generation", False))
    orca_exec = None
    coarse_metrics: dict[str, Any] = {}
    enriched_metrics: dict[str, Any] = {}
    refined_metrics: dict[str, Any] = {}
    if inline_evaluation and refined_graphs:
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
        if source_enrichment_enabled:
            enriched_metrics = evaluate_graph_sets(
                references,
                enriched_base_graphs,
                train_graphs,
                **metric_kwargs,
            )
        refined_metrics = evaluate_graph_sets(references, refined_graphs, train_graphs, **metric_kwargs)

    aggregated_pipeline = aggregate_pipeline_diagnostics(
        pipeline_records,
        require_complete=True,
        allow_fallback=False,
    ) if pipeline_records else {"num_records": 0, "status": "no_generated_graphs"}
    aggregated_pipeline["sampling_scope"] = "completed_graphs_only"
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
        "guidance_mode": refiner_settings.guidance_mode if guidance_mode == "spectral" else guidance_mode,
        "predictor_family": guidance_mode,
        "legacy_predictor_guidance_mode": guidance_mode,
        "degree_preservation_rate": degree_preservation_rate(coarse_graphs, refined_graphs),
        "source_enrichment_degree_preservation_rate": degree_preservation_rate(
            coarse_graphs, enriched_base_graphs
        ),
        "source_enrichment_connectedness_rate": float(
            np.mean(
                [
                    graph.number_of_nodes() > 0
                    and (graph.number_of_nodes() == 1 or nx.is_connected(graph))
                    for graph in enriched_base_graphs
                ]
            )
        ) if enriched_base_graphs else None,
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
        ) if refined_graphs else None,
        "mean_accepted_steps": float(np.mean(accepted_steps)) if accepted_steps else None,
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
        "mean_prediction_calls": float(np.mean(prediction_call_counts)) if prediction_call_counts else None,
        "mean_accepted_swaps_per_prediction_call": float(
            sum(accepted_steps) / max(sum(prediction_call_counts), 1)
        ),
        "plateau_refresh_count": int(plateau_refresh_count),
        "mean_graph_runtime_seconds": float(np.mean(graph_runtimes)) if graph_runtimes else None,
        "source_enrichment_enabled": bool(source_enrichment_enabled),
        "mean_source_enrichment_accepted_steps": float(
            np.mean(
                [sum(bool(row.get("accepted")) for row in trace) for trace in enrichment_traces]
            )
        ) if enrichment_traces else 0.0,
        "runtime_seconds": float(time.perf_counter() - run_started),
        "inline_evaluation": inline_evaluation and bool(refined_graphs),
        "inline_evaluation_skipped_reason": "no_generated_graphs" if inline_evaluation and not refined_graphs else None,
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
    elif guidance_mode == "spectral":
        diagnostics.update(
            {
                "rewiring_guidance_mode": refiner_settings.guidance_mode,
                "clustering_guidance_statistic": refiner_settings.clustering_statistic,
                "clustering_histogram_bins": getattr(model, "clustering_histogram_bins", None) if getattr(model, "predict_clustering_histogram", False) else None,
                "predictor_clustering_histogram_w1": (checkpoint.get("report", {}) or {}).get("val_clustering_histogram_w1"),
                "mean_accepted_clustering_gain": _mean_or_zero(accepted_rows, "clustering_gain"),
                "mean_accepted_clustering_discrepancy_before": _mean_or_zero(accepted_rows, "current_clustering_discrepancy"),
                "mean_accepted_clustering_discrepancy_after": _mean_or_zero(accepted_rows, "candidate_clustering_discrepancy"),
                "orbit_guidance_weight": refiner_settings.orbit_weight,
                "orbit_guidance_distance": refiner_settings.orbit_distance,
                "predictor_orbit_summary_enabled": bool(getattr(model, "predict_orbit_summary", False)),
                "predictor_induced_graphlet_histogram_enabled": bool(getattr(model, "predict_induced_graphlet_histogram", False)),
                "induced_graphlet_metadata": model.induced_graphlet_spec.metadata() if getattr(model, "predict_induced_graphlet_histogram", False) else None,
                "predictor_induced_graphlet_histogram_tv": (checkpoint.get("report", {}) or {}).get("val_induced_graphlet_histogram_tv"),
                "predictor_cycle_graphlet_histogram_enabled": bool(getattr(model, "predict_cycle_graphlet_histogram", False)),
                "predictor_cycle_graphlet_histogram_tv": (checkpoint.get("report", {}) or {}).get("val_cycle_graphlet_histogram_tv"),
                "cycle_graphlet_representation": (f"[C{model.cycle_graphlet_k},other] / choose(n,{model.cycle_graphlet_k})" if getattr(model, "predict_cycle_graphlet_histogram", False) else None),
                "mean_accepted_orbit_gain": _mean_or_zero(accepted_rows, "orbit_gain"),
                "mean_accepted_orbit_discrepancy_before": _mean_or_zero(accepted_rows, "current_orbit_discrepancy"),
                "mean_accepted_orbit_discrepancy_after": _mean_or_zero(accepted_rows, "candidate_orbit_discrepancy"),
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
                "predictor_heat_kernel_error": (
                    float(predictor_spectral_error)
                    if str(getattr(model, "spectral_representation", "eigenvalues")) == "heat_kernel"
                    else None
                ),
                "spectral_representation": str(getattr(model, "spectral_representation", "eigenvalues")),
                "spectral_distance": refiner_settings.distance,
                "spectral_normalization": refiner_settings.normalization,
                "spectral_bridge_schedule": refiner_settings.bridge_schedule,
                "spectral_debug_enabled": refiner_settings.debug_enabled,
            }
        )
        diagnostics.update(_guidance_diagnostic_summary(refiner_settings, accepted_rows))
        # Historical predictor_* aliases above are retained for readers of old reports.
        # Explicitly identify their source; they are NOT measured on generated graphs.
        diagnostics["predictor_error_scope"] = "checkpoint_validation_report"
        diagnostics["checkpoint_validation_metrics"] = {
            key: value for key, value in (checkpoint.get("report", {}) or {}).items()
            if key.startswith("val_") and isinstance(value, (int, float, str, bool, type(None)))
        }
        refresh_on_plateau = refiner_settings.refresh_on_prediction_plateau
        report_format = "topology_spectral_generation_v2"
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

    diagnostics.update({
        "checkpoint_selection_kind": checkpoint_selection.get("kind") if checkpoint_selection else None,
        "checkpoint_epoch": (checkpoint.get("report", {}) or {}).get("epoch"),
        "joint_degree_enabled": joint_degree_enabled,
        "degree_sampler_source": degree_sampler_source,
        "degree_conditioning": "actual_histogram_posterior_mean_decoder_features" if joint_degree_enabled else None,
        "orbit_consistency": model.orbit_consistency if joint_degree_enabled else None,
    })
    degree_prior_report = _generation_degree_prior_report(
        degree_sampler, degree_source=degree_source, seed=seed, degree_rng_mode=degree_rng_mode,
        degree_sampling_records=degree_sampling_records, target_degree_sequences=target_degree_sequences,
    )
    completion = generation_completion()
    degree_prior_report.update(completion, complete=True, generation_aborted=False)
    if isinstance(degree_sampler, PerturbedEmpiricalDegreeSampler):
        diagnostics.update({
            "degree_perturbation_method": degree_prior_report["method"],
            "degree_parent_failure_policy": degree_prior_report["parent_failure_policy"],
            "degree_prior_parent_draws": degree_prior_report["num_parent_draws"],
            "degree_prior_rejected_parent_draws": degree_prior_report["num_rejected_parent_draws"],
            "degree_prior_accepted_samples": degree_prior_report["num_accepted_samples"],
            "degree_perturbation_requested_fraction": degree_prior_report["requested_fraction"],
            "degree_perturbation_changed_fraction": degree_prior_report["changed_fraction"],
            "degree_perturbation_success_given_requested": degree_prior_report["success_given_requested"],
            "degree_perturbation_identity_fallbacks": degree_prior_report["num_identity_fallbacks"],
            "degree_perturbation_failure_reasons": degree_prior_report["failure_reasons"],
            "degree_prior_novel_fraction": degree_prior_report["novel_degree_fraction"],
            "degree_perturbation_mean_distance_half_l1": degree_prior_report["mean_distance_half_l1"],
            "degree_perturbation_preserves_n_m": degree_prior_report["all_preserve_n_m"],
            "degree_perturbation_preserves_second_moment": degree_prior_report["all_preserve_second_moment"],
        })
    report = {
        "format": report_format,
        "complete": True,
        **completion,
        "pipeline_mode": "topology",
        "guidance_mode": refiner_settings.guidance_mode if guidance_mode == "spectral" else guidance_mode,
        "predictor_family": guidance_mode,
        "legacy_predictor_guidance_mode": guidance_mode,
        "checkpoint_format": checkpoint.get("format"),
        "degree_source": degree_source,
        "degree_rng_mode": degree_rng_mode,
        "joint_soft_edge_diffusion": joint_edge_diffusion,
        "spectral_representation": str(getattr(model, "spectral_representation", "eigenvalues")),
        "heat_kernel_times": (
            list(getattr(model, "heat_kernel_times", ()))
            if str(getattr(model, "spectral_representation", "eigenvalues")) == "heat_kernel"
            else None
        ),
        "laplacian_eigenvalue_diffusion": bool(guidance_mode == "spectral"),
        "degree_prior_report_file": "degree_prior_report.json",
        "parent_degree_fingerprint": degree_prior_report["returned_parent_degree_fingerprint"],
        "sampled_degree_fingerprint": degree_prior_report["returned_degree_fingerprint"],
        "degree_sampling_records": degree_sampling_records,
        "degree_sampler_source": degree_sampler_source,
        "joint_degree_enabled": joint_degree_enabled,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_file_sha256,
        "checkpoint_selection": checkpoint_selection,
        "checkpoint_epoch": (checkpoint.get("report", {}) or {}).get("epoch"),
        "source_graph_fingerprint": graph_fingerprint(coarse_graphs) if joint_degree_enabled else None,
        "final_graph_fingerprint": graph_fingerprint(refined_graphs) if joint_degree_enabled else None,
        "dataset_graph_fingerprints": (checkpoint.get("report", {}) or {}).get("dataset_graph_fingerprints"),
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
        "hh_source": coarse_metrics,
        "enriched_base": enriched_metrics,
        "topology_refined": refined_metrics,
        "coarse": coarse_metrics,
        "hybrid_refined": refined_metrics,
        "diagnostics": diagnostics,
        "pipeline_diagnostics": aggregated_pipeline,
        "pipeline_records": pipeline_records,
        "traces": traces,
        "source_enrichment_traces": enrichment_traces,
        "seed": seed,
        "config_overrides": list(args.config_overrides),
        "rng_streams": {
            "source_and_refiner_decoupled": True,
            "refiner_rng_per_graph": True,
            "source_enrichment_decoupled": True,
            "source_enrichment_rng_per_graph": True,
            "degree_sampling_independent": degree_rng_mode == "independent",
            "degree_perturbation_rng_per_sample": isinstance(degree_sampler, PerturbedEmpiricalDegreeSampler),
        },
        "config": config,
    }
    output_dir = ensure_dir(args.output_dir)
    save_pickle(coarse_graphs, output_dir / "coarse_graphs.pkl")
    if source_enrichment_enabled:
        save_pickle(enriched_base_graphs, output_dir / "enriched_base_graphs.pkl")
    save_pickle(refined_graphs, output_dir / "topology_refined_graphs.pkl")
    if bool(generation_cfg.get("write_legacy_hybrid_alias", False)):
        save_pickle(refined_graphs, output_dir / "hybrid_refined_graphs.pkl")
    save_json(degree_prior_report, output_dir / "degree_prior_report.json")
    save_json(target_degree_sequences, output_dir / "sampled_degree_sequences.json")
    save_json(report, output_dir / "report.json")
    print(f"Generation finished: requested={num_generate} generated={len(refined_graphs)} "
          f"attempted={num_attempted}/{max_total_graph_attempts} skipped={len(skipped_graphs)}", flush=True)
    if completion["replacement_budget_exhausted"]:
        print(f"Replacement budget exhausted: generation.max_total_graph_attempts={max_total_graph_attempts}; "
              f"saved {len(refined_graphs)} of {num_generate} requested graphs.", flush=True)
    print("Topology generation diagnostics", flush=True)
    for key, value in diagnostics.items():
        print(f"  {key}: {value}", flush=True)
    print(f"Saved results to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()

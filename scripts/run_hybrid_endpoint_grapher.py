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

from grapher.construction.coarse import (
    assert_constructor_validity,
    construct_coarse_graph,
)
from grapher.construction.typed import construct_typed_graph
from grapher.data.io import load_dataset_splits
from grapher.evaluation.metrics import (
    degree_preservation_rate,
    degree_target_match_rate,
    evaluate_graph_sets,
)
from grapher.generators.degree_sampler import (
    EmpiricalDegreeSampler,
    build_degree_sampler,
)
from grapher.hybrid.model import load_hybrid_endpoint_checkpoint
from grapher.hybrid.refiner import refine_graph_with_hybrid_predictions
from grapher.hybrid.selector import load_selector_checkpoint
from grapher.molecular.constraints import (
    fit_molecular_attribute_priors,
    initialize_molecular_attributes,
    molecular_valence_errors,
)
from grapher.molecular.graph_io import is_valid_molecular_graph
from grapher.molecular.typed_invariants import (
    TypedInvariant,
    extract_typed_invariant,
    typed_invariant_matches_graph,
)
from grapher.properties.summary import configure_orca_executable
from grapher.utils.io import ensure_dir, load_yaml, save_json, save_pickle


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate graphs with endpoint categorical sampling and "
            "graphlet-guided valid rewiring."
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
    seed = int(args.seed if args.seed is not None else config.get("seed", 0))
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    evaluation_cfg = config.get("evaluation", {}) or {}
    compute_orbit = bool(evaluation_cfg.get("compute_orbit", True))
    graphlet_backend = str(evaluation_cfg.get("graphlet_backend", "sampled")).lower()
    orca_required = compute_orbit or graphlet_backend in {
        "orca",
        "exact_orca",
        "exact",
    }
    orca_exec = (
        configure_orca_executable(
            evaluation_cfg.get("orca_exec"),
            required=True,
        )
        if orca_required
        else None
    )
    if orca_exec:
        print(f"ORCA evaluation enabled: {orca_exec}", flush=True)

    dataset_cfg = config.get("dataset", {}) or {}
    splits = load_dataset_splits(
        str(dataset_cfg.get("name", "sbm")),
        root=dataset_cfg.get("root", "outputs/datasets"),
        build_if_missing=bool(dataset_cfg.get("build_if_missing", True)),
        config_path=dataset_cfg.get("config_path"),
    )
    train_graphs = list(splits["train"])
    reference_graphs = list(splits.get("test", []))
    generation_cfg = config.get("generation", {}) or {}
    num_generate = int(
        args.num_generate
        if args.num_generate is not None
        else generation_cfg.get("num_generate", len(reference_graphs))
    )
    if num_generate <= 0:
        raise ValueError("num_generate must be positive.")

    predictor_cfg = config.get("endpoint_predictor", {}) or {}
    checkpoint_path = args.checkpoint or predictor_cfg.get("checkpoint_path")
    if not checkpoint_path:
        raise ValueError("endpoint_predictor.checkpoint_path is required.")
    device = args.device or predictor_cfg.get("device", "auto")
    (
        model,
        vocabulary,
        graphlet_basis,
        summary_config,
        checkpoint,
    ) = load_hybrid_endpoint_checkpoint(checkpoint_path, device=device)
    model_device = next(model.parameters()).device

    molecular_cfg = dict(config.get("molecular_generation", {}) or {})
    molecular_mode = bool(
        molecular_cfg.get(
            "enabled",
            vocabulary.node_attribute in {"atomic_num", "atom_type"}
            and vocabulary.edge_attribute in {"bond_type", "bond_order"},
        )
    )
    molecular_priors = None
    molecular_atom_types: tuple[int, ...] = ()
    molecular_bond_types: tuple[int, ...] = ()
    constructor_cfg = dict(config.get("constructor", {}) or {})
    constructor_type = str(constructor_cfg.get("type", "havel_hakimi")).lower()
    degree_cfg = dict(config.get("degree_generator", {}) or {})
    typed_mode = constructor_type in {
        "typed_backtracking",
        "typed",
        "molecular_typed",
    } or str(degree_cfg.get("type", "")).lower() in {
        "typed_degree_histogram_vae",
        "typed_signature_histogram_vae",
        "typed_signature_vae",
    }
    if typed_mode and not molecular_mode:
        raise ValueError("Typed invariant generation requires molecular mode.")

    if molecular_mode:
        molecular_atom_types = tuple(
            int(value)
            for value in molecular_cfg.get(
                "allowed_atom_types",
                vocabulary.node_values,
            )
        )
        molecular_bond_types = tuple(
            int(value)
            for value in molecular_cfg.get(
                "allowed_bond_types",
                [1, 2, 3],
            )
        )
        if not typed_mode:
            molecular_priors = fit_molecular_attribute_priors(
                train_graphs,
                allowed_atom_types=molecular_atom_types,
                allowed_bond_types=molecular_bond_types,
            )
            print(
                "WARNING: topology-first molecular baseline mode is enabled; "
                "attributes are initialized after ordinary-degree construction.",
                flush=True,
            )

    degree_source = str(generation_cfg.get("degree_source", "empirical")).lower()
    degree_sampler = None
    if degree_source in {"learned", "degree_vae"}:
        degree_cfg["enabled"] = True
        degree_sampler = build_degree_sampler(
            degree_cfg,
            train_graphs,
            seed=seed,
        )
    elif degree_source in {"empirical", "train_empirical"}:
        if not typed_mode:
            degree_sampler = EmpiricalDegreeSampler.fit_from_graphs(
                train_graphs,
                seed=seed,
            )
    elif degree_source not in {"oracle", "test_oracle"}:
        raise ValueError(f"Unknown generation.degree_source: {degree_source!r}")

    # Independent random relabeling is unnecessary at generation and would make
    # debugging endpoint probabilities harder. The predictor itself is
    # permutation equivariant.
    constructor_cfg.setdefault("random_relabel", False)
    refiner_cfg = config.get("hybrid_refiner", {}) or {}
    refiner_mode = str(refiner_cfg.get("mode", "energy")).lower()
    learned_selector = None
    selector_checkpoint = None
    if refiner_mode in {"policy", "hybrid"}:
        selector_cfg = config.get("candidate_selector", {}) or {}
        selector_checkpoint = selector_cfg.get(
            "checkpoint_path",
            str(Path(checkpoint_path).parent / "selector.pt"),
        )
        if not Path(selector_checkpoint).is_file():
            raise FileNotFoundError(
                f"{refiner_mode} inference requires the trained selector "
                f"checkpoint: {selector_checkpoint}"
            )
        learned_selector, _selector_metadata = load_selector_checkpoint(
            selector_checkpoint,
            device=model_device,
        )
    coarse_graphs: list[nx.Graph] = []
    refined_graphs: list[nx.Graph] = []
    target_degree_sequences: list[list[int]] = []
    traces: list[list[dict[str, Any]]] = []
    source_rdkit_valid: list[float] = []
    final_rdkit_valid: list[float] = []
    typed_invariant_preserved: list[float] = []
    constructor_diagnostics: list[dict[str, Any]] = []
    graph_runtimes: list[float] = []

    for index in range(num_generate):
        graph_started = time.perf_counter()
        typed_invariant: TypedInvariant | None = None
        if degree_source in {"oracle", "test_oracle"}:
            if not reference_graphs:
                raise ValueError("Oracle degree generation requires test graphs.")
            oracle_graph = reference_graphs[index % len(reference_graphs)]
            if typed_mode:
                typed_invariant = extract_typed_invariant(
                    oracle_graph,
                    edge_types=molecular_bond_types,
                    node_attribute=str(vocabulary.node_attribute or "atomic_num"),
                    edge_attribute=str(vocabulary.edge_attribute or "bond_type"),
                )
            sequence = sorted(
                [int(degree) for _, degree in oracle_graph.degree()],
                reverse=True,
            )
            node_count = len(sequence)
            edge_count = int(sum(sequence) // 2)
            histogram = np.bincount(
                sequence,
                minlength=max(max(sequence, default=0) + 1, 1),
            ).astype(np.float64)
            histogram /= max(float(histogram.sum()), 1.0)
            degree_summary = {
                "num_nodes": node_count,
                "num_edges": edge_count,
                "degree_sequence": sequence,
                "degree_hist": histogram,
                "density": (
                    2.0 * edge_count / (node_count * (node_count - 1))
                    if node_count > 1
                    else 0.0
                ),
            }
            if typed_invariant is not None:
                degree_summary["typed_invariant"] = typed_invariant.to_dict()
        elif typed_mode and degree_source in {"empirical", "train_empirical"}:
            empirical_graph = train_graphs[int(rng.integers(0, len(train_graphs)))]
            typed_invariant = extract_typed_invariant(
                empirical_graph,
                edge_types=molecular_bond_types,
                node_attribute=str(vocabulary.node_attribute or "atomic_num"),
                edge_attribute=str(vocabulary.edge_attribute or "bond_type"),
            )
            degree_summary = {
                "num_nodes": typed_invariant.num_nodes,
                "num_edges": int(sum(typed_invariant.degree_sequence) // 2),
                "degree_sequence": typed_invariant.degree_sequence,
                "typed_invariant": typed_invariant.to_dict(),
            }
        else:
            assert degree_sampler is not None
            degree_summary = degree_sampler.sample(rng)
            if typed_mode:
                typed_invariant = TypedInvariant.from_dict(
                    degree_summary["typed_invariant"]
                )
        if typed_mode:
            if typed_invariant is None:
                typed_invariant = TypedInvariant.from_dict(
                    degree_summary["typed_invariant"]
                )
            typed_constructor_cfg = dict(constructor_cfg)
            typed_signature_cfg = config.get("typed_signature", {}) or {}
            typed_constructor_cfg.setdefault(
                "max_ordinary_degree",
                typed_signature_cfg.get("max_ordinary_degree"),
            )
            typed_constructor_cfg.setdefault(
                "max_weighted_valence",
                typed_signature_cfg.get("max_weighted_valence"),
            )
            coarse, construction_report = construct_typed_graph(
                typed_invariant,
                typed_constructor_cfg,
                rng,
            )
            constructor_diagnostics.append(construction_report)
            typed_invariant = extract_typed_invariant(
                coarse,
                edge_types=typed_invariant.edge_types,
                node_attribute=typed_invariant.node_attribute,
                edge_attribute=typed_invariant.edge_attribute,
            )
        else:
            coarse = construct_coarse_graph(
                degree_summary,
                constructor_cfg,
                rng,
            )
            assert_constructor_validity(
                coarse,
                degree_summary,
                require_connected=bool(constructor_cfg.get("ensure_connected", True)),
            )
            constructor_diagnostics.append({"success": True, "type": constructor_type})
        if molecular_mode and not typed_mode:
            assert molecular_priors is not None
            max_attempts = max(
                int(molecular_cfg.get("max_initialization_attempts", 16)),
                1,
            )
            initialized = None
            for attempt in range(max_attempts):
                candidate = initialize_molecular_attributes(
                    coarse,
                    molecular_priors,
                    rng=rng,
                    allowed_atom_types=molecular_atom_types,
                    allowed_bond_types=molecular_bond_types,
                    sample=bool(molecular_cfg.get("sample_attributes", True)),
                    smoothing=float(molecular_cfg.get("smoothing", 0.05)),
                    force_single_bonds=attempt == max_attempts - 1,
                )
                if not bool(
                    molecular_cfg.get("require_rdkit_source_validity", True)
                ) or is_valid_molecular_graph(candidate):
                    initialized = candidate
                    break
            if initialized is None:
                errors = molecular_valence_errors(
                    candidate,
                    allowed_atom_types=molecular_atom_types,
                    allowed_bond_types=molecular_bond_types,
                )
                raise RuntimeError(
                    "Could not initialize an RDKit-valid molecular source graph. "
                    f"Valence diagnostics: {errors[:3]}"
                )
            coarse = initialized
        if molecular_mode:
            source_is_valid = bool(is_valid_molecular_graph(coarse))
            source_rdkit_valid.append(float(source_is_valid))
            if (
                bool(molecular_cfg.get("require_rdkit_source_validity", True))
                and not source_is_valid
            ):
                raise RuntimeError(
                    "The exact typed constructor produced a source that failed "
                    "the declared RDKit validity mask."
                )
        refined, trace = refine_graph_with_hybrid_predictions(
            coarse,
            model=model,
            vocabulary=vocabulary,
            graphlet_basis=graphlet_basis,
            summary_config=summary_config,
            refiner_config=refiner_cfg,
            device=model_device,
            rng=rng,
            return_trace=True,
            selector=learned_selector,
        )
        if molecular_mode:
            final_rdkit_valid.append(float(is_valid_molecular_graph(refined)))
        if typed_invariant is not None:
            typed_invariant_preserved.append(
                float(typed_invariant_matches_graph(refined, typed_invariant))
            )
        coarse_graphs.append(coarse)
        refined_graphs.append(refined)
        target_degree_sequences.append(
            [int(degree) for degree in degree_summary["degree_sequence"]]
        )
        traces.append(trace)
        graph_runtimes.append(float(time.perf_counter() - graph_started))
        accepted = sum(bool(row.get("accepted")) for row in trace)
        print(
            f"graph={index + 1}/{num_generate} "
            f"n={refined.number_of_nodes()} m={refined.number_of_edges()} "
            f"accepted_steps={accepted}",
            flush=True,
        )

    references = reference_graphs[:num_generate] or reference_graphs
    metric_kwargs = {
        "compute_orbit": compute_orbit,
        "compute_graphlet_history": bool(
            evaluation_cfg.get("compute_graphlet_history", True)
        ),
        "graphlet_k_min": int(
            evaluation_cfg.get(
                "graphlet_k_min",
                summary_config.graphlet_k_min,
            )
        ),
        "graphlet_k_max": int(
            evaluation_cfg.get(
                "graphlet_k_max",
                summary_config.graphlet_k_max,
            )
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
        "graphlet_node_label_attr": (
            vocabulary.node_attribute if graphlet_basis.attributed else None
        ),
        "graphlet_edge_label_attr": (
            vocabulary.edge_attribute if graphlet_basis.attributed else None
        ),
        "attributed_graphlet_backend": graphlet_basis.attributed_backend,
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
    accepted_steps = [
        sum(bool(row.get("accepted")) for row in trace) for trace in traces
    ]
    predicted_endpoint_degree_matches = [
        float(row["sampled_target_degree_match"])
        for trace in traces
        for row in trace
        if "sampled_target_degree_match" in row
    ]
    trace_rows = [row for trace in traces for row in trace]
    accepted_rows = [row for row in trace_rows if bool(row.get("accepted"))]
    decision_rows = [row for row in trace_rows if "num_proposals" in row]
    proposal_count = sum(int(row.get("num_proposals", 0)) for row in decision_rows)
    valid_candidate_count = sum(
        int(row.get("num_valid_candidates", 0)) for row in decision_rows
    )
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
    stop_reasons = Counter(
        str(trace[-1].get("reason", "step_budget")) for trace in traces if trace
    )
    predictor_report = checkpoint.get("report", {}) or {}
    diagnostics = {
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
                    nx.is_connected(graph) if graph.number_of_nodes() > 0 else False
                    for graph in refined_graphs
                ]
            )
        ),
        "mean_accepted_steps": float(np.mean(accepted_steps)),
        "predictor_nll": predictor_report.get(
            "val_edge_nll",
            predictor_report.get("val_edge_loss"),
        ),
        "predictor_macro_f1": predictor_report.get("val_edge_macro_f1"),
        "predictor_graphlet_error": predictor_report.get("val_graphlet_mae"),
        "predictor_consistency_residual": predictor_report.get(
            "val_consistency_residual"
        ),
        "constructor_success_rate": float(
            np.mean(
                [bool(item.get("success", False)) for item in constructor_diagnostics]
            )
        )
        if constructor_diagnostics
        else 0.0,
        "mean_constructor_restarts": float(
            np.mean(
                [float(item.get("restarts", 0.0)) for item in constructor_diagnostics]
            )
        )
        if constructor_diagnostics
        else 0.0,
        "mean_constructor_backtracks": float(
            np.mean(
                [float(item.get("backtracks", 0.0)) for item in constructor_diagnostics]
            )
        )
        if constructor_diagnostics
        else 0.0,
        "candidate_pass_rate": float(valid_candidate_count / max(proposal_count, 1)),
        "proposals_per_accepted_swap": float(
            proposal_count / max(len(accepted_rows), 1)
        ),
        "num_proposals": int(proposal_count),
        "num_valid_candidates": int(valid_candidate_count),
        "num_accepted_swaps": len(accepted_rows),
        "stop_rate": float(
            np.mean(
                [
                    bool(trace and not trace[-1].get("accepted", False))
                    for trace in traces
                ]
            )
        ),
        "stop_reasons": dict(sorted(stop_reasons.items())),
        "candidate_rejection_reasons": dict(sorted(rejection_reasons.items())),
        "mean_graph_runtime_seconds": float(np.mean(graph_runtimes))
        if graph_runtimes
        else 0.0,
        "runtime_seconds": float(time.perf_counter() - run_started),
        "end_to_end_yield": float(len(refined_graphs) / max(num_generate, 1)),
        "predictor_sampled_endpoint_degree_match_rate": (
            float(np.mean(predicted_endpoint_degree_matches))
            if predicted_endpoint_degree_matches
            else float("nan")
        ),
    }
    if molecular_mode:
        diagnostics.update(
            {
                "molecular_source_rdkit_validity": float(np.mean(source_rdkit_valid)),
                "molecular_final_rdkit_validity": float(np.mean(final_rdkit_valid)),
                "molecular_valence_preservation_rate": float(
                    np.mean(
                        [
                            not molecular_valence_errors(
                                graph,
                                allowed_atom_types=molecular_atom_types,
                                allowed_bond_types=molecular_bond_types,
                            )
                            for graph in refined_graphs
                        ]
                    )
                ),
                "typed_invariant_preservation_rate": float(
                    np.mean(typed_invariant_preserved)
                )
                if typed_invariant_preserved
                else float("nan"),
            }
        )
    report = {
        "format": "hybrid_endpoint_graphlet_generation_v3",
        "checkpoint_format": checkpoint.get("format"),
        "selector_checkpoint": selector_checkpoint,
        "selector_mode": refiner_mode,
        "degree_source": degree_source,
        "orca_exec": orca_exec,
        "num_generated": len(refined_graphs),
        "coarse": coarse_metrics,
        "hybrid_refined": refined_metrics,
        "diagnostics": diagnostics,
        "traces": traces,
        "constructor_diagnostics": constructor_diagnostics,
        "seed": seed,
        "config": config,
    }
    output_dir = ensure_dir(args.output_dir)
    save_pickle(coarse_graphs, output_dir / "coarse_graphs.pkl")
    save_pickle(refined_graphs, output_dir / "hybrid_refined_graphs.pkl")
    save_json(report, output_dir / "report.json")
    print("Hybrid generation diagnostics", flush=True)
    for key, value in diagnostics.items():
        print(f"  {key}: {value}", flush=True)
    print(f"Saved results to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from grapher.construction.coarse import (
    assert_constructor_validity,
    construct_coarse_graph,
)
from grapher.data.io import load_dataset_splits
from grapher.evaluation.metrics import (
    degree_preservation_rate,
    degree_target_match_rate,
    evaluate_graph_sets,
)
from grapher.generators.degree_sampler import EmpiricalDegreeSampler
from grapher.hybrid.model import load_hybrid_endpoint_checkpoint
from grapher.hybrid.refiner import refine_graph_with_hybrid_predictions
from grapher.properties.sampler import build_degree_sampler_from_config
from grapher.utils.io import ensure_dir, load_yaml, save_json, save_pickle


def _degree_summary_from_graph(graph: nx.Graph) -> dict[str, Any]:
    sequence = sorted([int(degree) for _, degree in graph.degree()], reverse=True)
    node_count = len(sequence)
    edge_count = int(sum(sequence) // 2)
    histogram = np.bincount(
        sequence,
        minlength=max(max(sequence, default=0) + 1, 1),
    ).astype(np.float64)
    histogram /= max(float(histogram.sum()), 1.0)
    return {
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

    config = load_yaml(args.config)
    seed = int(args.seed if args.seed is not None else config.get("seed", 0))
    rng = np.random.default_rng(seed)
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

    degree_source = str(generation_cfg.get("degree_source", "empirical")).lower()
    degree_sampler = None
    if degree_source in {"learned", "degree_vae"}:
        degree_cfg = dict(config.get("degree_generator", {}) or {})
        degree_cfg["enabled"] = True
        degree_sampler = build_degree_sampler_from_config(
            degree_cfg,
            train_graphs,
            seed=seed,
        )
    elif degree_source in {"empirical", "train_empirical"}:
        degree_sampler = EmpiricalDegreeSampler.fit_from_graphs(
            train_graphs,
            seed=seed,
        )
    elif degree_source not in {"oracle", "test_oracle"}:
        raise ValueError(f"Unknown generation.degree_source: {degree_source!r}")

    constructor_cfg = dict(config.get("constructor", {}) or {})
    # Independent random relabeling is unnecessary at generation and would make
    # debugging endpoint probabilities harder. The predictor itself is
    # permutation equivariant.
    constructor_cfg.setdefault("random_relabel", False)
    refiner_cfg = config.get("hybrid_refiner", {}) or {}
    evaluation_cfg = config.get("evaluation", {}) or {}
    coarse_graphs: list[nx.Graph] = []
    refined_graphs: list[nx.Graph] = []
    target_degree_sequences: list[list[int]] = []
    traces: list[list[dict[str, Any]]] = []

    for index in range(num_generate):
        if degree_source in {"oracle", "test_oracle"}:
            if not reference_graphs:
                raise ValueError("Oracle degree generation requires test graphs.")
            degree_summary = _degree_summary_from_graph(
                reference_graphs[index % len(reference_graphs)]
            )
        else:
            assert degree_sampler is not None
            degree_summary = degree_sampler.sample(rng)
        coarse = construct_coarse_graph(
            degree_summary,
            constructor_cfg,
            rng,
        )
        assert_constructor_validity(
            coarse,
            degree_summary,
            require_connected=bool(
                constructor_cfg.get("ensure_connected", True)
            ),
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
        )
        coarse_graphs.append(coarse)
        refined_graphs.append(refined)
        target_degree_sequences.append(
            [int(degree) for degree in degree_summary["degree_sequence"]]
        )
        traces.append(trace)
        accepted = sum(bool(row.get("accepted")) for row in trace)
        print(
            f"graph={index + 1}/{num_generate} "
            f"n={refined.number_of_nodes()} m={refined.number_of_edges()} "
            f"accepted_steps={accepted}",
            flush=True,
        )

    references = reference_graphs[:num_generate] or reference_graphs
    metric_kwargs = {
        "compute_orbit": bool(evaluation_cfg.get("compute_orbit", False)),
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
        "graphlet_backend": str(
            evaluation_cfg.get("graphlet_backend", "sampled")
        ),
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
                    nx.is_connected(graph)
                    if graph.number_of_nodes() > 0
                    else False
                    for graph in refined_graphs
                ]
            )
        ),
        "mean_accepted_steps": float(np.mean(accepted_steps)),
        "predictor_sampled_endpoint_degree_match_rate": (
            float(np.mean(predicted_endpoint_degree_matches))
            if predicted_endpoint_degree_matches
            else float("nan")
        ),
    }
    report = {
        "format": "hybrid_endpoint_graphlet_generation_v2",
        "checkpoint_format": checkpoint.get("format"),
        "degree_source": degree_source,
        "num_generated": len(refined_graphs),
        "coarse": coarse_metrics,
        "hybrid_refined": refined_metrics,
        "diagnostics": diagnostics,
        "traces": traces,
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

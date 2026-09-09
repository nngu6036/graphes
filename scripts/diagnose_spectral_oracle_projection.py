#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from grapher.data.io import load_dataset_splits
from grapher.models.dhvae_hh.havel_hakimi import construct_coarse_graph
from grapher.rewiring_mlp.generic.spectral import (
    laplacian_eigenvalues,
    spectral_distance,
    spectral_scale,
    spectrum_moments,
)
from grapher.rewiring_mlp.generic.spectral_refiner import (
    SpectralPrediction,
    SpectralRefinerConfig,
    refine_graph_with_spectral_predictions,
)
from grapher.utils.io import ensure_dir, load_yaml, save_json, save_pickle


def _limited(values: list[Any], limit: int | None) -> list[Any]:
    if limit is None or int(limit) <= 0:
        return values
    return values[: int(limit)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Test only the spectral rewiring projection using the TRUE clean spectrum. "
            "No neural denoiser and no learned degree prior are involved."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--max-graphs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config = load_yaml(args.config)
    dataset_cfg = dict(config.get("dataset", {}) or {})
    splits = load_dataset_splits(
        str(dataset_cfg.get("name", "sbm")),
        root=dataset_cfg.get("root", "outputs/datasets"),
        build_if_missing=bool(dataset_cfg.get("build_if_missing", False)),
        config_path=dataset_cfg.get("config_path"),
    )
    targets = _limited(list(splits[args.split]), args.max_graphs)
    if not targets:
        raise ValueError(f"Dataset split {args.split!r} is empty.")

    constructor_cfg = dict(config.get("constructor", {}) or {})
    refiner_cfg = SpectralRefinerConfig.from_dict(config.get("topology_refiner", {}) or {})
    spectral_cfg = dict(config.get("spectral_prediction", {}) or {})
    metric = str(spectral_cfg.get("distance", "rmse"))
    normalization = str(spectral_cfg.get("normalization", "mean_degree"))
    low_weight = float(spectral_cfg.get("low_frequency_weight", 1.0))
    low_cutoff = int(spectral_cfg.get("low_frequency_cutoff", 0))

    seed_sequence = np.random.SeedSequence(int(args.seed))
    child_seeds = seed_sequence.spawn(len(targets))
    sources: list[nx.Graph] = []
    refined_graphs: list[nx.Graph] = []
    rows: list[dict[str, Any]] = []

    for index, (target, child_seed) in enumerate(zip(targets, child_seeds)):
        target = nx.convert_node_labels_to_integers(nx.Graph(target), ordering="sorted")
        degree_sequence = sorted((int(d) for _, d in target.degree()), reverse=True)
        summary = {
            "num_nodes": len(degree_sequence),
            "num_edges": int(sum(degree_sequence) // 2),
            "degree_sequence": degree_sequence,
        }
        rng = np.random.default_rng(child_seed)
        source = construct_coarse_graph(summary, constructor_cfg, rng=rng)
        target_spectrum = laplacian_eigenvalues(target)
        target_trace, target_second = spectrum_moments(target_spectrum)

        def oracle_predictor(_model, graph, *, time, device, **kwargs):
            del time, device, kwargs
            return SpectralPrediction(
                clean_spectrum=target_spectrum.copy(),
                current_spectrum=laplacian_eigenvalues(graph),
                trace=target_trace,
                second_moment=target_second,
            )

        refined, trace = refine_graph_with_spectral_predictions(
            source,
            model=None,  # custom oracle predictor ignores the model argument
            refiner_config=refiner_cfg,
            device=args.device,
            rng=rng,
            return_trace=True,
            prediction_fn=oracle_predictor,
            debug_context=f"oracle-{index}",
        )
        scale = spectral_scale(source, mode=normalization)
        initial_distance = spectral_distance(
            laplacian_eigenvalues(source),
            target_spectrum,
            metric=metric,
            scale=scale,
            low_frequency_weight=low_weight,
            low_frequency_cutoff=low_cutoff,
        )
        final_distance = spectral_distance(
            laplacian_eigenvalues(refined),
            target_spectrum,
            metric=metric,
            scale=scale,
            low_frequency_weight=low_weight,
            low_frequency_cutoff=low_cutoff,
        )
        accepted = sum(bool(row.get("accepted")) for row in trace)
        rows.append(
            {
                "index": index,
                "num_nodes": target.number_of_nodes(),
                "num_edges": target.number_of_edges(),
                "initial_spectral_distance": float(initial_distance),
                "final_spectral_distance": float(final_distance),
                "spectral_gain": float(initial_distance - final_distance),
                "accepted_steps": int(accepted),
                "degree_preserved": sorted(dict(source.degree()).values())
                == sorted(dict(refined.degree()).values()),
                "connected": refined.number_of_nodes() <= 1 or nx.is_connected(refined),
            }
        )
        sources.append(source)
        refined_graphs.append(refined)

    initial = np.asarray([row["initial_spectral_distance"] for row in rows], dtype=float)
    final = np.asarray([row["final_spectral_distance"] for row in rows], dtype=float)
    report = {
        "format": "grapher_spectral_oracle_projection_v1",
        "split": args.split,
        "num_graphs": len(rows),
        "mean_initial_spectral_distance": float(initial.mean()),
        "mean_final_spectral_distance": float(final.mean()),
        "mean_spectral_gain": float((initial - final).mean()),
        "improved_fraction": float(np.mean(final < initial - 1.0e-12)),
        "degree_preservation_rate": float(np.mean([row["degree_preserved"] for row in rows])),
        "connectedness_rate": float(np.mean([row["connected"] for row in rows])),
        "mean_accepted_steps": float(np.mean([row["accepted_steps"] for row in rows])),
        "rows": rows,
    }

    print("Spectral oracle projection diagnostic")
    print(f"  split={args.split} graphs={len(rows)}")
    print(f"  mean source -> target spectral distance: {report['mean_initial_spectral_distance']:.6f}")
    print(f"  mean final  -> target spectral distance: {report['mean_final_spectral_distance']:.6f}")
    print(f"  mean spectral gain:                     {report['mean_spectral_gain']:.6f}")
    print(f"  improved fraction:                      {report['improved_fraction']:.3f}")
    print(f"  degree preservation:                    {report['degree_preservation_rate']:.3f}")
    print(f"  connectedness:                          {report['connectedness_rate']:.3f}")
    print(f"  mean accepted steps:                    {report['mean_accepted_steps']:.2f}")

    if args.output_dir:
        out = ensure_dir(args.output_dir)
        save_json(report, out / "oracle_projection_report.json")
        save_pickle(sources, out / "hh_source_graphs.pkl")
        save_pickle(refined_graphs, out / "oracle_spectral_refined_graphs.pkl")
        print(f"  saved={out}")


if __name__ == "__main__":
    main()

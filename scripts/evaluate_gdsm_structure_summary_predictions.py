#!/usr/bin/env python3
"""Evaluate joint graphlet, clustering, and orbit summary predictions."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import networkx as nx
import numpy as np

from grapher.rewiring_mlp.generic.basis import TopologyGraphletBasis
from grapher.rewiring_mlp.generic.graphlets import extract_topology_graphlet_target
from grapher.rewiring_mlp.properties.summary import (
    SummaryConfig,
    clustering_histogram,
    python_orbit_count_vector,
)


def _load(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def _resolve_graph_path(root: Path, graph_arg: Path | None) -> Path:
    if graph_arg is None:
        return root / "base_graphs.pkl"
    if graph_arg.is_absolute() or graph_arg.exists():
        return graph_arg
    return root / graph_arg


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument("--graphs", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    root = args.generated_dir
    graph_path = _resolve_graph_path(root, args.graphs)
    graphs = _load(graph_path)
    prediction_path = root / "predicted_structure_summaries.pkl"
    if not prediction_path.is_file():
        prediction_path = root / "predicted_graphlet_summaries.pkl"
    predictions = _load(prediction_path)
    if len(graphs) != len(predictions):
        raise RuntimeError(f"graph/prediction count mismatch: {len(graphs)} vs {len(predictions)}")

    cfg = SummaryConfig.from_dict({
        "clustering_summary": False,
        "spectral_summary": False,
        "motif_proxy": False,
        "orbit_count": False,
        "graphlet_history": True,
        "graphlet_k_min": 3,
        "graphlet_k_max": 5,
        "graphlet_connected_only": True,
        "graphlet_topology_filter": "all",
        "graphlet_backend": "exact",
        "graphlet_num_samples": None,
    })
    basis = TopologyGraphletBasis.from_config(cfg)

    graphlet_hist_mae = {str(k): [] for k in (3, 4, 5)}
    graphlet_mass_mae = {str(k): [] for k in (3, 4, 5)}
    clustering_hist_mae: list[float] = []
    clustering_cdf_mae: list[float] = []
    orbit_hist_mae: list[float] = []
    orbit_log_total_sq: list[float] = []
    orbit_log_count_sq: list[float] = []

    for graph, pred in zip(graphs, predictions):
        g = nx.convert_node_labels_to_integers(graph, ordering="sorted")
        target, mass = extract_topology_graphlet_target(g, graphlet_basis=basis, summary_config=cfg)
        ph = np.asarray(pred["graphlet_histogram"], dtype=np.float64).reshape(-1)
        pm = np.asarray(pred["graphlet_mass"], dtype=np.float64).reshape(-1)
        if not np.isfinite(ph).all() or not np.isfinite(pm).all():
            raise FloatingPointError(f"Non-finite graphlet prediction at graph {pred.get('graph_index', '?')}")
        for order, (start, stop) in zip((3, 4, 5), basis.slices):
            block = target[start:stop]
            if float(block.sum()) > 0.0:
                graphlet_hist_mae[str(order)].append(float(np.mean(np.abs(ph[start:stop] - block))))
            graphlet_mass_mae[str(order)].append(float(abs(pm[(3,4,5).index(order)] - mass[(3,4,5).index(order)])))

        if "clustering_histogram" in pred:
            pc = np.asarray(pred["clustering_histogram"], dtype=np.float64).reshape(-1)
            tc = clustering_histogram(g, bins=pc.size)
            if not np.isfinite(pc).all():
                raise FloatingPointError("Non-finite clustering prediction")
            clustering_hist_mae.append(float(np.mean(np.abs(pc - tc))))
            clustering_cdf_mae.append(float(np.mean(np.abs(np.cumsum(pc - tc)[:-1]))))

        if "orbit_histogram" in pred and "orbit_log_total" in pred:
            po = np.asarray(pred["orbit_histogram"], dtype=np.float64).reshape(-1)
            plt = float(np.asarray(pred["orbit_log_total"], dtype=np.float64).reshape(-1)[0])
            counts = np.maximum(np.asarray(python_orbit_count_vector(g), dtype=np.float64), 0.0)
            total = float(counts.sum())
            to = counts / total if total > 0.0 else np.zeros_like(counts)
            tlt = float(np.log1p(total))
            if not np.isfinite(po).all() or not np.isfinite(plt):
                raise FloatingPointError("Non-finite orbit prediction")
            orbit_hist_mae.append(float(np.mean(np.abs(po - to))))
            orbit_log_total_sq.append((plt - tlt) ** 2)
            pred_counts = po * np.expm1(max(plt, 0.0))
            orbit_log_count_sq.append(float(np.mean((np.log1p(pred_counts) - np.log1p(counts)) ** 2)))

    result = {
        "num_graphs": len(graphs),
        "graphs_path": str(graph_path),
        "predictions_path": str(prediction_path),
        "graphlet_histogram_mae": {k: (float(np.mean(v)) if v else None) for k, v in graphlet_hist_mae.items()},
        "graphlet_mass_mae": {k: (float(np.mean(v)) if v else None) for k, v in graphlet_mass_mae.items()},
        "clustering_histogram_mae": float(np.mean(clustering_hist_mae)) if clustering_hist_mae else None,
        "clustering_cdf_mae": float(np.mean(clustering_cdf_mae)) if clustering_cdf_mae else None,
        "orbit_histogram_mae": float(np.mean(orbit_hist_mae)) if orbit_hist_mae else None,
        "orbit_log_total_rmse": float(np.sqrt(np.mean(orbit_log_total_sq))) if orbit_log_total_sq else None,
        "orbit_log_count_rmse": float(np.sqrt(np.mean(orbit_log_count_sq))) if orbit_log_count_sq else None,
    }
    out = args.output or (root / "structure_summary_prediction_evaluation.json")
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

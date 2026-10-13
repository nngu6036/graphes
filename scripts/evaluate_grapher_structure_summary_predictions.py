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


def _prediction_metadata(root: Path) -> dict:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return dict(
        manifest.get("predicted_topology_summaries", {})
        or manifest.get("predicted_structure_summaries", {})
        or {}
    )


def _degree_histogram(graph: nx.Graph, bins: int) -> np.ndarray:
    degrees = np.asarray([int(d) for _, d in graph.degree()], dtype=np.int64)
    if degrees.size == 0:
        raise ValueError("Degree-summary evaluation requires non-empty graphs")
    observed_max = int(degrees.max(initial=0))
    if observed_max >= bins:
        raise ValueError(
            f"Observed degree {observed_max} does not fit prediction width {bins}."
        )
    hist = np.bincount(degrees, minlength=bins).astype(np.float64)
    return hist / float(degrees.size)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument("--graphs", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--graphlet-num-samples",
        type=int,
        default=None,
        help="Sample this many induced subsets per graph/order; default is exact.",
    )
    parser.add_argument(
        "--max-graphs",
        type=int,
        default=None,
        help="Optional deterministic prefix for expensive summary diagnostics.",
    )
    args = parser.parse_args()

    root = args.generated_dir
    graph_path = _resolve_graph_path(root, args.graphs)
    graphs = _load(graph_path)
    prediction_path = root / "predicted_topology_summaries.pkl"
    if not prediction_path.is_file():
        prediction_path = root / "predicted_structure_summaries.pkl"
    if not prediction_path.is_file():
        prediction_path = root / "predicted_graphlet_summaries.pkl"
    predictions = _load(prediction_path)
    if len(graphs) != len(predictions):
        raise RuntimeError(f"graph/prediction count mismatch: {len(graphs)} vs {len(predictions)}")
    if args.max_graphs is not None:
        if args.max_graphs < 1:
            raise ValueError("--max-graphs must be positive")
        graphs = graphs[: args.max_graphs]
        predictions = predictions[: args.max_graphs]
    metadata = _prediction_metadata(root)
    graphlet_orders = [int(k) for k in metadata.get("graphlet_orders", [3, 4, 5])]
    if not graphlet_orders:
        graphlet_orders = [3, 4, 5]
    if graphlet_orders[0] != 3 or graphlet_orders != list(range(3, graphlet_orders[-1] + 1)):
        raise ValueError(f"Unsupported graphlet order metadata: {graphlet_orders}")

    cfg = SummaryConfig.from_dict({
        "clustering_summary": False,
        "spectral_summary": False,
        "motif_proxy": False,
        "orbit_count": False,
        "graphlet_history": True,
        "graphlet_k_min": 3,
        "graphlet_k_max": max(graphlet_orders),
        "graphlet_connected_only": True,
        "graphlet_topology_filter": "all",
        "graphlet_backend": "sampled" if args.graphlet_num_samples else "exact",
        "graphlet_num_samples": args.graphlet_num_samples,
    })
    basis = TopologyGraphletBasis.from_config(cfg)

    graphlet_hist_mae = {str(k): [] for k in graphlet_orders}
    graphlet_mass_mae = {str(k): [] for k in graphlet_orders}
    clustering_hist_mae: list[float] = []
    clustering_cdf_mae: list[float] = []
    orbit_hist_mae: list[float] = []
    orbit_log_total_sq: list[float] = []
    orbit_log_count_sq: list[float] = []
    degree_hist_mae: list[float] = []
    degree_cdf_mae: list[float] = []
    degree_mean_abs: list[float] = []

    for graph, pred in zip(graphs, predictions):
        g = nx.convert_node_labels_to_integers(graph, ordering="sorted")
        target, mass = extract_topology_graphlet_target(g, graphlet_basis=basis, summary_config=cfg)
        graphlet_hist_key = (
            "graphlet_histogram"
            if "graphlet_histogram" in pred
            else "topology_graphlet_histogram"
        )
        graphlet_mass_key = (
            "graphlet_mass" if "graphlet_mass" in pred else "topology_graphlet_mass"
        )
        ph = np.asarray(pred[graphlet_hist_key], dtype=np.float64).reshape(-1)
        pm = np.asarray(pred[graphlet_mass_key], dtype=np.float64).reshape(-1)
        if not np.isfinite(ph).all() or not np.isfinite(pm).all():
            raise FloatingPointError(f"Non-finite graphlet prediction at graph {pred.get('graph_index', '?')}")
        for order_index, (order, (start, stop)) in enumerate(zip(graphlet_orders, basis.slices)):
            block = target[start:stop]
            if float(block.sum()) > 0.0:
                graphlet_hist_mae[str(order)].append(float(np.mean(np.abs(ph[start:stop] - block))))
            graphlet_mass_mae[str(order)].append(float(abs(pm[order_index] - mass[order_index])))

        clustering_key = (
            "clustering_histogram"
            if "clustering_histogram" in pred
            else "topology_clustering_histogram"
        )
        if clustering_key in pred:
            pc = np.asarray(pred[clustering_key], dtype=np.float64).reshape(-1)
            tc = clustering_histogram(g, bins=pc.size)
            if not np.isfinite(pc).all():
                raise FloatingPointError("Non-finite clustering prediction")
            clustering_hist_mae.append(float(np.mean(np.abs(pc - tc))))
            clustering_cdf_mae.append(float(np.mean(np.abs(np.cumsum(pc - tc)[:-1]))))

        orbit_hist_key = (
            "orbit_histogram" if "orbit_histogram" in pred else "topology_orbit_histogram"
        )
        orbit_total_key = (
            "orbit_log_total" if "orbit_log_total" in pred else "topology_orbit_log_total"
        )
        if orbit_hist_key in pred and orbit_total_key in pred:
            po = np.asarray(pred[orbit_hist_key], dtype=np.float64).reshape(-1)
            plt = float(np.asarray(pred[orbit_total_key], dtype=np.float64).reshape(-1)[0])
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

        degree_key = (
            "degree_histogram" if "degree_histogram" in pred else "topology_degree_histogram"
        )
        if degree_key in pred:
            pd = np.asarray(pred[degree_key], dtype=np.float64).reshape(-1)
            td = _degree_histogram(g, pd.size)
            if not np.isfinite(pd).all():
                raise FloatingPointError("Non-finite degree-histogram prediction")
            degree_hist_mae.append(float(np.mean(np.abs(pd - td))))
            if pd.size > 1:
                degree_cdf_mae.append(float(np.mean(np.abs(np.cumsum(pd - td)[:-1]))))
            axis = np.arange(pd.size, dtype=np.float64)
            degree_mean_abs.append(float(abs(np.dot(pd, axis) - np.dot(td, axis))))

    result = {
        "num_graphs": len(graphs),
        "graphs_path": str(graph_path),
        "predictions_path": str(prediction_path),
        "summary_type": "topology_only",
        "graphlet_histogram_mae": {k: (float(np.mean(v)) if v else None) for k, v in graphlet_hist_mae.items()},
        "graphlet_mass_mae": {k: (float(np.mean(v)) if v else None) for k, v in graphlet_mass_mae.items()},
        "clustering_histogram_mae": float(np.mean(clustering_hist_mae)) if clustering_hist_mae else None,
        "clustering_cdf_mae": float(np.mean(clustering_cdf_mae)) if clustering_cdf_mae else None,
        "orbit_histogram_mae": float(np.mean(orbit_hist_mae)) if orbit_hist_mae else None,
        "orbit_log_total_rmse": float(np.sqrt(np.mean(orbit_log_total_sq))) if orbit_log_total_sq else None,
        "orbit_log_count_rmse": float(np.sqrt(np.mean(orbit_log_count_sq))) if orbit_log_count_sq else None,
        "degree_histogram_mae": float(np.mean(degree_hist_mae)) if degree_hist_mae else None,
        "degree_cdf_mae": float(np.mean(degree_cdf_mae)) if degree_cdf_mae else None,
        "degree_mean_mae": float(np.mean(degree_mean_abs)) if degree_mean_abs else None,
        "graphlet_orders": graphlet_orders,
        "degree_max_degree": (
            int(metadata.get("degree_max_degree", 0))
            if bool(metadata.get("degree_enabled", False)) else None
        ),
    }
    out = args.output or (root / "structure_summary_prediction_evaluation.json")
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

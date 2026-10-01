#!/usr/bin/env python3
"""Evaluate joint graphlet-summary predictions saved by log-gap Laplacian GSDM."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import networkx as nx
import numpy as np

from grapher.rewiring_mlp.generic.basis import TopologyGraphletBasis
from grapher.rewiring_mlp.generic.graphlets import extract_topology_graphlet_target
from grapher.rewiring_mlp.properties.summary import SummaryConfig


def _load(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument(
        "--graphs",
        type=Path,
        default=None,
        help="Graph pickle to compare against predictions (default: <generated-dir>/base_graphs.pkl).",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    root = args.generated_dir
    graph_path = args.graphs or (root / "base_graphs.pkl")
    if not graph_path.is_absolute():
        graph_path = root / graph_path
    graphs = _load(graph_path)
    predictions = _load(root / "predicted_graphlet_summaries.pkl")
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

    hist_mae_by_order = {str(k): [] for k in (3, 4, 5)}
    mass_abs_by_order = {str(k): [] for k in (3, 4, 5)}
    for graph, pred in zip(graphs, predictions):
        g = nx.convert_node_labels_to_integers(graph, ordering="sorted")
        target, mass = extract_topology_graphlet_target(
            g, graphlet_basis=basis, summary_config=cfg
        )
        ph = np.asarray(pred["graphlet_histogram"], dtype=np.float64)
        pm = np.asarray(pred["graphlet_mass"], dtype=np.float64)
        for order, (start, stop) in zip((3, 4, 5), basis.slices):
            block = target[start:stop]
            if float(block.sum()) > 0.0:
                hist_mae_by_order[str(order)].append(float(np.mean(np.abs(ph[start:stop] - block))))
            mass_abs_by_order[str(order)].append(float(abs(pm[(3,4,5).index(order)] - mass[(3,4,5).index(order)])))

    result = {
        "num_graphs": len(graphs),
        "graphs_path": str(graph_path),
        "graphlet_histogram_mae": {
            k: (float(np.mean(v)) if v else None) for k, v in hist_mae_by_order.items()
        },
        "graphlet_mass_mae": {
            k: (float(np.mean(v)) if v else None) for k, v in mass_abs_by_order.items()
        },
    }
    out = args.output or (root / "graphlet_summary_prediction_evaluation.json")
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

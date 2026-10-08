#!/usr/bin/env python3
"""Evaluate hierarchical GDSM typed-graphlet auxiliary predictions."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from grapher.properties.summary import SummaryConfig
from grapher.rewiring_mlp.attributed.data import GraphletBasis


def _load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument("--graphs", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    root = args.generated_dir
    graph_path = args.graphs or (root / "base_graphs.pkl")
    if not graph_path.is_absolute() and not graph_path.exists():
        graph_path = root / graph_path
    prediction_path = root / "predicted_typed_graphlet_summaries.pkl"
    if not prediction_path.is_file():
        prediction_path = root / "predicted_structure_summaries.pkl"

    manifest = json.loads((root / "manifest.json").read_text())
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = Path(manifest["checkpoint"]["path"])
    state = _torch_load(checkpoint_path)
    summary = dict(state["attribute_summary"])
    basis = GraphletBasis.from_dict(dict(summary["graphlet_basis"]))
    cfg = SummaryConfig.from_dict(dict(summary["summary_config"]))

    graphs = _load_pickle(graph_path)
    predictions = _load_pickle(prediction_path)
    if len(graphs) != len(predictions):
        raise RuntimeError(
            f"graph/prediction count mismatch: {len(graphs)} vs {len(predictions)}"
        )

    hist_mae = {str(k): [] for k in basis.sizes}
    mass_mae = {str(k): [] for k in basis.sizes}
    overflow_mass = {str(k): [] for k in basis.sizes}
    for index, (graph, pred) in enumerate(zip(graphs, predictions)):
        history, mass = basis.statistics_for_graph(
            graph, cfg, rng=np.random.default_rng(index)
        )
        target = basis.flatten_history(history).astype(np.float64)
        target_mass = basis.flatten_mass(mass).astype(np.float64)
        pred_hist = np.asarray(
            pred.get("typed_graphlet_histogram"), dtype=np.float64
        ).reshape(-1)
        pred_mass = np.asarray(
            pred.get("typed_graphlet_mass"), dtype=np.float64
        ).reshape(-1)
        if pred_hist.size != basis.width or pred_mass.size != len(basis.sizes):
            raise ValueError("Typed graphlet prediction has an incompatible width")
        for order_index, (order, (start, stop)) in enumerate(
            zip(basis.sizes, basis.slices)
        ):
            block = target[start:stop]
            if float(block.sum()) > 0.0:
                hist_mae[order].append(
                    float(np.mean(np.abs(pred_hist[start:stop] - block)))
                )
            mass_mae[order].append(
                float(abs(pred_mass[order_index] - target_mass[order_index]))
            )
            if basis.overflow_key in basis.keys_by_k[order]:
                overflow_index = basis.keys_by_k[order].index(basis.overflow_key)
                overflow_mass[order].append(float(block[overflow_index]))

    result = {
        "num_graphs": len(graphs),
        "graphs_path": str(graph_path),
        "predictions_path": str(prediction_path),
        "checkpoint_path": str(checkpoint_path),
        "summary_type": "typed_graphlet",
        "graphlet_histogram_mae": {
            key: (float(np.mean(values)) if values else None)
            for key, values in hist_mae.items()
        },
        "graphlet_mass_mae": {
            key: (float(np.mean(values)) if values else None)
            for key, values in mass_mae.items()
        },
        "target_overflow_mass": {
            key: (float(np.mean(values)) if values else None)
            for key, values in overflow_mass.items()
        },
    }
    output = args.output or (root / "typed_graphlet_prediction_evaluation.json")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

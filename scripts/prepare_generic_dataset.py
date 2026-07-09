#!/usr/bin/env python
"""Prepare generic graph datasets used by the coarse-to-fine pipeline.

This script creates and verifies NetworkX graph splits for SBM, grid, and ego
datasets. It writes split files to:

    outputs/datasets/<dataset_name>/train.pkl
    outputs/datasets/<dataset_name>/val.pkl
    outputs/datasets/<dataset_name>/test.pkl

Run from the repository root with:

    PYTHONPATH=src python scripts/prepare_generic_dataset.py \
        --dataset sbm \
        --root outputs/datasets
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from grapher.data.builders import SPLIT_NAMES, build_graphs_from_config, infer_dataset_type, split_graphs
from grapher.data.io import dataset_dir, save_dataset_splits
from grapher.utils.io import load_yaml, save_json


def _verify_graph(g: nx.Graph, *, require_connected: bool, reject_zero_degree: bool) -> list[str]:
    errors: list[str] = []
    if g.is_directed():
        errors.append("graph is directed")
    if isinstance(g, (nx.MultiGraph, nx.MultiDiGraph)):
        errors.append("graph is a multigraph")
    if nx.number_of_selfloops(g) != 0:
        errors.append("graph has self-loops")
    if g.number_of_nodes() <= 0:
        errors.append("graph has no nodes")
    if require_connected and g.number_of_nodes() > 1 and not nx.is_connected(g):
        errors.append("graph is disconnected")
    if reject_zero_degree and g.number_of_nodes() > 1:
        zero_degree = [node for node, degree in g.degree() if degree == 0]
        if zero_degree:
            errors.append(f"graph has {len(zero_degree)} zero-degree nodes")
    return errors


def _summarize_split(graphs: list[nx.Graph]) -> dict[str, Any]:
    if not graphs:
        return {
            "num_graphs": 0,
            "min_nodes": None,
            "max_nodes": None,
            "mean_nodes": None,
            "min_edges": None,
            "max_edges": None,
            "mean_edges": None,
            "connected_rate": None,
            "zero_self_loop_rate": None,
        }
    node_counts = np.asarray([g.number_of_nodes() for g in graphs], dtype=float)
    edge_counts = np.asarray([g.number_of_edges() for g in graphs], dtype=float)
    connected = np.asarray([nx.is_connected(g) if g.number_of_nodes() > 0 else False for g in graphs], dtype=float)
    no_self_loops = np.asarray([nx.number_of_selfloops(g) == 0 for g in graphs], dtype=float)
    return {
        "num_graphs": len(graphs),
        "min_nodes": int(node_counts.min()),
        "max_nodes": int(node_counts.max()),
        "mean_nodes": float(node_counts.mean()),
        "min_edges": int(edge_counts.min()),
        "max_edges": int(edge_counts.max()),
        "mean_edges": float(edge_counts.mean()),
        "connected_rate": float(connected.mean()),
        "zero_self_loop_rate": float(no_self_loops.mean()),
    }


def verify_splits(splits: dict[str, list[nx.Graph]], config: dict[str, Any]) -> dict[str, Any]:
    filters = config.get("filters", {}) or {}
    require_connected = bool(filters.get("require_connected", True))
    reject_zero_degree = bool(filters.get("reject_zero_degree", True))

    errors: list[str] = []
    split_stats: dict[str, Any] = {}
    for split, graphs in splits.items():
        split_stats[split] = _summarize_split(graphs)
        for idx, graph in enumerate(graphs):
            graph_errors = _verify_graph(
                graph,
                require_connected=require_connected,
                reject_zero_degree=reject_zero_degree,
            )
            errors.extend([f"{split}[{idx}]: {msg}" for msg in graph_errors])

    expected_total = int(config.get("num_graphs", sum(len(v) for v in splits.values())))
    actual_total = sum(len(v) for v in splits.values())
    if actual_total != expected_total:
        errors.append(f"expected {expected_total} graphs, found {actual_total}")

    report = {
        "dataset": str(config.get("name", "sbm")),
        "dataset_type": infer_dataset_type(config),
        "status": "pass" if not errors else "fail",
        "split_sizes": {split: len(splits.get(split, [])) for split in SPLIT_NAMES},
        "split_stats": split_stats,
        "errors": errors,
    }
    if errors:
        preview = "\n".join(errors[:20])
        raise AssertionError(f"Dataset verification failed with {len(errors)} error(s):\n{preview}")
    return report


def _print_report(report: dict[str, Any], out_dir: Path) -> None:
    def _fmt_stat(value: Any, *, digits: int = 2) -> str:
        if value is None:
            return "n/a"
        if isinstance(value, float):
            return f"{value:.{digits}f}"
        return str(value)

    print(f"Dataset: {report['dataset']}")
    print(f"Type:     {report['dataset_type']}")
    print(f"Output:  {out_dir}")
    print(f"Status:  {report['status']}")
    print("Splits:")
    for split in SPLIT_NAMES:
        stats = report["split_stats"][split]
        print(
            "  "
            f"{split}: {stats['num_graphs']} graphs, "
            f"nodes {stats['min_nodes']}..{stats['max_nodes']} "
            f"mean={_fmt_stat(stats['mean_nodes'])}, "
            f"edges {stats['min_edges']}..{stats['max_edges']} "
            f"mean={_fmt_stat(stats['mean_edges'])}, "
            f"connected_rate={_fmt_stat(stats['connected_rate'], digits=3)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and verify SBM, grid, or ego graph datasets.")
    parser.add_argument("--dataset", required=True, help="Dataset name; loads configs/datasets/<dataset>.yaml.")
    parser.add_argument("--root", default="outputs/datasets", help="Output root for dataset splits.")
    args = parser.parse_args()

    config_path = Path("configs/datasets") / f"{args.dataset}.yaml"
    config = dict(load_yaml(config_path))
    config["name"] = args.dataset
    dataset_name = str(config["name"])
    out_dir = dataset_dir(dataset_name, args.root)
    graphs = build_graphs_from_config(config)
    splits = split_graphs(graphs, config)
    save_dataset_splits(dataset_name, splits, config, args.root)

    report = verify_splits(splits, config)
    report["action"] = "built"
    save_json(report, out_dir / "prep_report.json")
    _print_report(report, out_dir)
    print("Action:  built")
    print(f"Report:  {out_dir / 'prep_report.json'}")


if __name__ == "__main__":
    main()

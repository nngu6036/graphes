#!/usr/bin/env python
"""Prepare the Stage-0 SPECTRE-style SBM dataset.

This script creates and verifies the first generic graph dataset used by the
coarse-to-fine proposal. It writes NetworkX graph splits to:

    outputs/datasets/sbm_spectre/train.pkl
    outputs/datasets/sbm_spectre/val.pkl
    outputs/datasets/sbm_spectre/test.pkl

Run from the repository root with:

    PYTHONPATH=src python scripts/prepare_sbm_spectre_dataset.py \
        --config configs/datasets/sbm_spectre.yaml \
        --root outputs/datasets
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from grapher.data.io import dataset_dir, save_dataset_splits
from grapher.data.sbm import build_sbm_graphs, split_graphs
from grapher.utils.io import load_pickle, load_yaml, save_json


SPLIT_NAMES = ("train", "val", "test")


def _apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    config = dict(config)
    if args.dataset_name is not None:
        config["name"] = args.dataset_name
    if args.num_graphs is not None:
        config["num_graphs"] = int(args.num_graphs)
    if args.seed is not None:
        config["seed"] = int(args.seed)
    if args.train_frac is not None or args.val_frac is not None:
        split_cfg = dict(config.get("split", {}) or {})
        if args.train_frac is not None:
            split_cfg["train"] = float(args.train_frac)
        if args.val_frac is not None:
            split_cfg["val"] = float(args.val_frac)
        test_frac = 1.0 - float(split_cfg.get("train", 0.8)) - float(split_cfg.get("val", 0.1))
        if test_frac <= 0.0:
            raise ValueError("train + val fractions must be less than 1.0")
        split_cfg["test"] = test_frac
        config["split"] = split_cfg
    return config


def _split_file_paths(dataset_name: str, root: str | Path) -> dict[str, Path]:
    out_dir = dataset_dir(dataset_name, root)
    return {split: out_dir / f"{split}.pkl" for split in SPLIT_NAMES}


def _load_existing_splits(dataset_name: str, root: str | Path) -> dict[str, list[nx.Graph]]:
    paths = _split_file_paths(dataset_name, root)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing dataset split files: {missing}")
    return {split: load_pickle(path) for split, path in paths.items()}


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
        "dataset": str(config.get("name", "sbm_spectre")),
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
    print(f"Dataset: {report['dataset']}")
    print(f"Output:  {out_dir}")
    print(f"Status:  {report['status']}")
    print("Splits:")
    for split in SPLIT_NAMES:
        stats = report["split_stats"][split]
        print(
            "  "
            f"{split}: {stats['num_graphs']} graphs, "
            f"nodes {stats['min_nodes']}..{stats['max_nodes']} "
            f"mean={stats['mean_nodes']:.2f}, "
            f"edges {stats['min_edges']}..{stats['max_edges']} "
            f"mean={stats['mean_edges']:.2f}, "
            f"connected_rate={stats['connected_rate']:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and verify the Stage-0 SPECTRE-style SBM dataset.")
    parser.add_argument("--config", default="configs/datasets/sbm_spectre.yaml", help="Dataset YAML config.")
    parser.add_argument("--root", default="outputs/datasets", help="Output root for dataset splits.")
    parser.add_argument("--dataset-name", default=None, help="Override dataset name from the config.")
    parser.add_argument("--num-graphs", type=int, default=None, help="Override the number of graphs.")
    parser.add_argument("--seed", type=int, default=None, help="Override the dataset seed.")
    parser.add_argument("--train-frac", type=float, default=None, help="Override train split fraction.")
    parser.add_argument("--val-frac", type=float, default=None, help="Override validation split fraction.")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild dataset even if split files already exist.")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing split files; do not build.")
    args = parser.parse_args()

    config = _apply_overrides(load_yaml(args.config), args)
    dataset_name = str(config.get("name", "sbm_spectre"))
    out_dir = dataset_dir(dataset_name, args.root)
    split_paths = _split_file_paths(dataset_name, args.root)
    splits_exist = all(path.exists() for path in split_paths.values())

    if args.verify_only:
        splits = _load_existing_splits(dataset_name, args.root)
        action = "verified existing"
    elif splits_exist and not args.overwrite:
        splits = _load_existing_splits(dataset_name, args.root)
        action = "found existing"
    else:
        if dataset_name not in {"sbm_spectre", "sbm"}:
            raise ValueError(f"This preparation script supports only sbm_spectre/sbm, got {dataset_name!r}.")
        graphs = build_sbm_graphs(config)
        splits = split_graphs(graphs, config)
        save_dataset_splits(dataset_name, splits, config, args.root)
        action = "built"

    report = verify_splits(splits, config)
    report["action"] = action
    save_json(report, out_dir / "prep_report.json")
    _print_report(report, out_dir)
    print(f"Action:  {action}")
    print(f"Report:  {out_dir / 'prep_report.json'}")


if __name__ == "__main__":
    main()

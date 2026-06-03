from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx

from grapher.registry import DATASET_REGISTRY
from grapher.utils.io import load_pickle, load_yaml, save_pickle, save_json, save_yaml, stable_hash
from grapher.datasets.base import graph_statistics
from grapher.graphs.attributes import attribute_coverage, canonicalize_graph_attributes, fit_attribute_statistics, normalize_schema


def dataset_output_dir(dataset: str, output_root: str | Path = "outputs/datasets") -> Path:
    return Path(output_root) / dataset


def split_path(dataset: str, split: str, output_root: str | Path = "outputs/datasets") -> Path:
    return dataset_output_dir(dataset, output_root) / f"{split}.pkl"


def metadata_path(dataset: str, output_root: str | Path = "outputs/datasets") -> Path:
    return dataset_output_dir(dataset, output_root) / "metadata.json"


def build_dataset_splits(dataset: str, config: dict[str, Any]) -> dict[str, list[nx.Graph]]:
    if dataset not in DATASET_REGISTRY:
        raise KeyError(f"Dataset '{dataset}' is not registered. Available: {sorted(DATASET_REGISTRY.keys())}")
    return DATASET_REGISTRY[dataset](config).build()


def save_dataset_splits(
    dataset: str,
    splits: dict[str, list[nx.Graph]],
    config: dict[str, Any],
    *,
    output_root: str | Path = "outputs/datasets",
    force: bool = False,
) -> Path:
    out_dir = dataset_output_dir(dataset, output_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    attr_schema = normalize_schema(config)
    all_raw_graphs: list[nx.Graph] = []
    for graphs in splits.values():
        all_raw_graphs.extend(list(graphs))
    all_attr_stats = fit_attribute_statistics(all_raw_graphs, attr_schema)
    canonical_splits: dict[str, list[nx.Graph]] = {}
    for split, graphs in splits.items():
        canonical_splits[split], _ = canonicalize_graph_attributes(list(graphs), attr_schema, all_attr_stats)
        save_pickle(canonical_splits[split], split_path(dataset, split, output_root), force=force)
    splits = canonical_splits
    train_attr_stats = fit_attribute_statistics(list(splits.get("train", [])), attr_schema)
    metadata = {
        "dataset": dataset,
        "config_hash": stable_hash(config),
        "config": config,
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "statistics": {k: graph_statistics(list(v)) for k, v in splits.items()},
        "graph_attributes": {
            "schema": attr_schema,
            "all_attribute_stats_raw": all_attr_stats.to_dict(),
            "train_attribute_stats": train_attr_stats.to_dict(),
            "coverage": {k: attribute_coverage(list(v), attr_schema) for k, v in splits.items()},
            "canonical_attribute_names": {
                "node_label": "node_label",
                "node_features": "feats",
                "edge_type": "edge_type",
                "edge_features": "edge_attr",
                "graph_label": "graph_label",
            },
        },
    }
    save_json(metadata, metadata_path(dataset, output_root), force=force)
    save_yaml(config, out_dir / "resolved_dataset_config.yaml", force=force)
    return out_dir


def load_dataset_splits(
    dataset: str,
    *,
    output_root: str | Path = "outputs/datasets",
    build_if_missing: bool = True,
    config_path: str | Path | None = None,
    force: bool = False,
) -> dict[str, list[nx.Graph]]:
    required = [split_path(dataset, s, output_root) for s in ("train", "val", "test")]
    if all(p.exists() for p in required) and not force:
        return {s: load_pickle(split_path(dataset, s, output_root)) for s in ("train", "val", "test")}

    if not build_if_missing:
        missing = [str(p) for p in required if not p.exists()]
        raise FileNotFoundError(f"Missing persisted dataset split files: {missing}")

    cfg_path = Path(config_path) if config_path else Path("configs/datasets") / f"{dataset}.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {cfg_path}")
    cfg = load_yaml(cfg_path)
    splits = build_dataset_splits(dataset, cfg)
    save_dataset_splits(dataset, splits, cfg, output_root=output_root, force=True)
    # Return the canonicalized persisted splits, not the raw builder output.
    # This keeps force=True consistent with the non-force load path.
    return {s: load_pickle(split_path(dataset, s, output_root)) for s in ("train", "val", "test")}

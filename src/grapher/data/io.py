from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx

from grapher.data.builders import build_splits_from_config
from grapher.utils.io import ensure_dir, load_pickle, load_yaml, save_json, save_pickle, save_yaml


def dataset_dir(dataset: str, root: str | Path = "outputs/datasets") -> Path:
    return Path(root) / dataset


def split_path(dataset: str, split: str, root: str | Path = "outputs/datasets") -> Path:
    return dataset_dir(dataset, root) / f"{split}.pkl"


def save_dataset_splits(dataset: str, splits: dict[str, list[nx.Graph]], config: dict[str, Any], root: str | Path) -> None:
    out = ensure_dir(dataset_dir(dataset, root))
    for split, graphs in splits.items():
        save_pickle(graphs, out / f"{split}.pkl")
    save_yaml(config, out / "resolved_dataset_config.yaml")
    save_json({"dataset": dataset, "split_sizes": {k: len(v) for k, v in splits.items()}}, out / "metadata.json")


def build_dataset_from_config(config_path: str | Path) -> dict[str, list[nx.Graph]]:
    config = load_yaml(config_path)
    return build_splits_from_config(config)


def load_dataset_splits(
    dataset: str,
    *,
    root: str | Path = "outputs/datasets",
    build_if_missing: bool = True,
    config_path: str | Path | None = None,
) -> dict[str, list[nx.Graph]]:
    paths = {split: split_path(dataset, split, root) for split in ("train", "val", "test")}
    if all(path.exists() for path in paths.values()):
        return {split: load_pickle(path) for split, path in paths.items()}
    if not build_if_missing:
        missing = [str(p) for p in paths.values() if not p.exists()]
        raise FileNotFoundError(f"Missing dataset split files: {missing}")
    if config_path is None:
        config_path = Path("configs/datasets") / f"{dataset}.yaml"
    config = load_yaml(config_path)
    splits = build_dataset_from_config(config_path)
    save_dataset_splits(dataset, splits, config, root)
    return splits

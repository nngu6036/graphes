"""Reproducible subsets of an existing prepared training split."""
from __future__ import annotations

from typing import Any, Sequence, TypeVar

import numpy as np

T = TypeVar("T")


def sample_training_graphs(
    graphs: Sequence[T], limit: int | None, *, seed: int
) -> tuple[list[T], dict[str, Any]]:
    """Sample without replacement, retaining prepared order within the subset.

    None or zero uses the full split. The returned selection can be saved in a
    checkpoint and restored independently of generation's random seed.
    """

    requested = None if limit is None else int(limit)
    if requested is not None and requested < 0:
        raise ValueError("--max-train-graphs / dataset.max_train_graphs must be non-negative.")
    count = len(graphs) if not requested else min(requested, len(graphs))
    indices = None
    if count < len(graphs):
        indices = sorted(np.random.default_rng(int(seed)).choice(
            len(graphs), size=count, replace=False
        ).tolist())
    selected = list(graphs) if indices is None else [graphs[index] for index in indices]
    selection = {
        "strategy": "all" if indices is None else "random_without_replacement",
        "seed": int(seed),
        "available_graphs": len(graphs),
        "requested_graphs": requested,
        "selected_graphs": len(selected),
        "indices": indices,
    }
    return selected, selection


def restore_training_graphs(graphs: Sequence[T], dataset_config: dict) -> list[T]:
    """Restore the checkpoint's training pool; old checkpoints retain prefixes."""

    selection = dataset_config.get("training_subset")
    if selection is None:
        limit = dataset_config.get("max_train_graphs")
        return list(graphs[:int(limit)]) if limit else list(graphs)
    if selection["available_graphs"] != len(graphs):
        raise ValueError("Prepared training split size differs from the saved training subset.")
    indices = selection["indices"]
    if selection["strategy"] == "all":
        if indices is not None or selection["selected_graphs"] != len(graphs):
            raise ValueError("Invalid saved full training subset.")
        return list(graphs)
    if selection["strategy"] != "random_without_replacement" or indices is None:
        raise ValueError("Unknown saved training subset strategy.")
    if (len(indices) != selection["selected_graphs"]
            or len(set(indices)) != len(indices)
            or any(not isinstance(index, int) or not 0 <= index < len(graphs) for index in indices)):
        raise ValueError("Invalid saved training subset indices.")
    return [graphs[index] for index in indices]

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import networkx as nx
import numpy as np


def _as_rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(0 if seed is None else int(seed))


def split_counts(n: int, split_cfg: dict[str, float] | None) -> dict[str, int]:
    split_cfg = split_cfg or {"train": 0.8, "val": 0.1, "test": 0.1}
    train_frac = float(split_cfg.get("train", 0.8))
    val_frac = float(split_cfg.get("val", 0.1))
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    n_train = min(max(n_train, 0), n)
    n_val = min(max(n_val, 0), n - n_train)
    return {"train": n_train, "val": n_val, "test": n - n_train - n_val}


def split_graphs(graphs: Sequence[nx.Graph], config: dict[str, Any], *, shuffle: bool = True) -> dict[str, list[nx.Graph]]:
    graphs = list(graphs)
    if shuffle:
        rng = _as_rng(config.get("seed", 42))
        order = rng.permutation(len(graphs))
        graphs = [graphs[int(i)] for i in order]
    counts = split_counts(len(graphs), config.get("split"))
    train_end = counts["train"]
    val_end = train_end + counts["val"]
    return {
        "train": graphs[:train_end],
        "val": graphs[train_end:val_end],
        "test": graphs[val_end:],
    }


def preprocess_graph(graph: nx.Graph, config: dict[str, Any]) -> nx.Graph | None:
    cfg = config.get("preprocessing", {}) or {}
    g = nx.Graph(graph) if cfg.get("make_undirected", True) else graph.copy()
    if config.get("remove_self_loops", True):
        g.remove_edges_from(nx.selfloop_edges(g))
    size_filter = cfg.get("graph_size_filter", {}) or {}
    min_nodes = size_filter.get("min_nodes")
    max_nodes = size_filter.get("max_nodes")
    if min_nodes is not None and g.number_of_nodes() < int(min_nodes):
        return None
    if max_nodes is not None and g.number_of_nodes() > int(max_nodes):
        return None
    if cfg.get("relabel_nodes", True):
        g = nx.convert_node_labels_to_integers(g, ordering="sorted")
    return g


def graph_statistics(graphs: Sequence[nx.Graph]) -> dict[str, Any]:
    graphs = list(graphs)
    node_counts = np.asarray([g.number_of_nodes() for g in graphs], dtype=float)
    edge_counts = np.asarray([g.number_of_edges() for g in graphs], dtype=float)
    degrees = [d for g in graphs for _, d in g.degree()]
    max_degrees = np.asarray([max((d for _, d in g.degree()), default=0) for g in graphs], dtype=float)
    densities = np.asarray([nx.density(g) if g.number_of_nodes() > 1 else 0.0 for g in graphs], dtype=float)
    connected = np.asarray([float(nx.is_connected(g)) if g.number_of_nodes() > 0 else 0.0 for g in graphs], dtype=float)

    def summary(values: np.ndarray) -> dict[str, float]:
        if values.size == 0:
            return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
        return {
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
        }

    return {
        "num_graphs": len(graphs),
        "nodes": summary(node_counts),
        "edges": summary(edge_counts),
        "max_degree": summary(max_degrees),
        "avg_degree": float(np.mean(degrees)) if degrees else 0.0,
        "density": summary(densities),
        "connected_rate": float(connected.mean()) if connected.size else 0.0,
    }


@dataclass
class BaseDatasetBuilder:
    config: dict[str, Any]

    @property
    def seed(self) -> int:
        return int(self.config.get("seed", 42))

    @property
    def rng(self) -> np.random.Generator:
        return _as_rng(self.seed)

    def finalize(self, graphs: Sequence[nx.Graph], *, shuffle: bool | None = None) -> dict[str, list[nx.Graph]]:
        processed = []
        for graph in graphs:
            g = preprocess_graph(graph, self.config)
            if g is not None:
                processed.append(g)
        use_shuffle = bool(self.config.get("shuffle", True)) if shuffle is None else bool(shuffle)
        return split_graphs(processed, self.config, shuffle=use_shuffle)

    def build(self) -> dict[str, list[nx.Graph]]:
        raise NotImplementedError

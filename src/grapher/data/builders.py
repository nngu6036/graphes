from __future__ import annotations

import pickle
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import networkx as nx
import numpy as np

from grapher.data.sbm import build_sbm_graphs


SPLIT_NAMES = ("train", "val", "test")


class GraphDatasetBuilder(Protocol):
    def build_graphs(self) -> list[nx.Graph]:
        ...


def infer_dataset_type(config: dict[str, Any]) -> str:
    dataset_type = config.get("type", config.get("dataset_type"))
    if dataset_type is not None:
        return str(dataset_type).lower()

    name = str(config.get("name", "")).lower()
    if name.startswith("sbm"):
        return "sbm"
    if name.startswith("grid"):
        return "grid"
    if name.startswith("ego"):
        return "ego"

    raise ValueError(
        "Could not infer dataset type. Set config field 'type' to one of: "
        "sbm, grid, ego."
    )


def split_graphs(graphs: list[nx.Graph], config: dict[str, Any]) -> dict[str, list[nx.Graph]]:
    split_cfg = config.get("split", {}) or {}
    train_frac = float(split_cfg.get("train", 0.8))
    val_frac = float(split_cfg.get("val", 0.1))
    seed = int(config.get("seed", 0))

    if train_frac < 0.0 or val_frac < 0.0 or train_frac + val_frac >= 1.0:
        raise ValueError("split train/val fractions must be non-negative and sum to less than 1.0")

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(graphs)).tolist()
    n_train = int(round(len(graphs) * train_frac))
    n_val = int(round(len(graphs) * val_frac))
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]
    return {
        "train": [graphs[i] for i in train_idx],
        "val": [graphs[i] for i in val_idx],
        "test": [graphs[i] for i in test_idx],
    }


def build_graphs_from_config(config: dict[str, Any]) -> list[nx.Graph]:
    dataset_type = infer_dataset_type(config)
    builder = get_dataset_builder(dataset_type, config)
    return builder.build_graphs()


def build_splits_from_config(config: dict[str, Any]) -> dict[str, list[nx.Graph]]:
    return split_graphs(build_graphs_from_config(config), config)


def get_dataset_builder(dataset_type: str, config: dict[str, Any]) -> GraphDatasetBuilder:
    dataset_type = dataset_type.lower()
    if dataset_type == "sbm":
        return SBMDatasetBuilder(config)
    if dataset_type == "grid":
        return GridDatasetBuilder(config)
    if dataset_type == "ego":
        return EgoDatasetBuilder(config)
    raise ValueError(f"Unsupported dataset type {dataset_type!r}; expected sbm, grid, or ego.")


@dataclass
class SBMDatasetBuilder:
    config: dict[str, Any]

    def build_graphs(self) -> list[nx.Graph]:
        graphs = build_sbm_graphs(self.config)
        source_dataset = str(self.config.get("name", "sbm"))
        for graph in graphs:
            graph.graph["source_dataset"] = source_dataset
        return graphs


@dataclass
class GridDatasetBuilder:
    config: dict[str, Any]

    def build_graphs(self) -> list[nx.Graph]:
        cfg = self.config.get("grid", {}) or {}
        filters = self.config.get("filters", {}) or {}
        num_graphs = int(self.config.get("num_graphs", 200))
        seed = int(self.config.get("seed", 0))
        min_rows = int(cfg.get("min_rows", cfg.get("min_side", 10)))
        max_rows = int(cfg.get("max_rows", cfg.get("max_side", 20)))
        min_cols = int(cfg.get("min_cols", cfg.get("min_side", 10)))
        max_cols = int(cfg.get("max_cols", cfg.get("max_side", 20)))
        random_relabel = bool(cfg.get("random_relabel", True))
        enumerate_dimensions = bool(cfg.get("enumerate_dimensions", False))
        max_attempts_per_graph = int(filters.get("max_attempts_per_graph", 50))

        if min_rows <= 0 or min_cols <= 0:
            raise ValueError("grid min rows/cols must be positive.")
        if max_rows < min_rows or max_cols < min_cols:
            raise ValueError("grid max rows/cols must be greater than or equal to min rows/cols.")

        rng = np.random.default_rng(seed)
        graphs: list[nx.Graph] = []
        attempts = 0
        max_attempts = num_graphs * max_attempts_per_graph
        source_dataset = str(self.config.get("name", "grid"))

        if enumerate_dimensions:
            dimension_pairs = [
                (rows, cols)
                for rows in range(min_rows, max_rows + 1)
                for cols in range(min_cols, max_cols + 1)
            ]
            if len(dimension_pairs) < num_graphs:
                raise RuntimeError(
                    f"Need {num_graphs} grid dimension pairs, but only {len(dimension_pairs)} are configured."
                )
            dimension_pairs = dimension_pairs[:num_graphs]
        else:
            dimension_pairs = []

        while len(graphs) < num_graphs and attempts < max_attempts:
            attempts += 1
            if enumerate_dimensions:
                rows, cols = dimension_pairs[len(graphs)]
            else:
                rows = int(rng.integers(min_rows, max_rows + 1))
                cols = int(rng.integers(min_cols, max_cols + 1))
            graph_seed = int(rng.integers(0, 2**31 - 1))
            g = nx.grid_2d_graph(rows, cols)
            g = nx.convert_node_labels_to_integers(nx.Graph(g), first_label=0, ordering="sorted")

            if random_relabel:
                labels = rng.permutation(g.number_of_nodes()).tolist()
                mapping = {node: int(labels[node]) for node in g.nodes()}
                g = nx.relabel_nodes(g, mapping)
                g = nx.convert_node_labels_to_integers(g, first_label=0, ordering="sorted")

            g.graph.update(
                {
                    "source_dataset": source_dataset,
                    "dataset_type": "grid",
                    "rows": rows,
                    "cols": cols,
                    "seed": graph_seed,
                }
            )
            graphs.append(g)

        if len(graphs) < num_graphs:
            raise RuntimeError(f"Could not build {num_graphs} grid graphs; got {len(graphs)}.")
        return graphs


@dataclass
class EgoDatasetBuilder:
    config: dict[str, Any]

    def build_graphs(self) -> list[nx.Graph]:
        cfg = self.config.get("ego", {}) or {}
        filters = self.config.get("filters", {}) or {}
        num_graphs = int(self.config.get("num_graphs", 200))
        seed = int(self.config.get("seed", 0))
        radius = int(cfg.get("radius", 2))
        min_nodes = int(cfg.get("min_nodes", 8))
        max_nodes = int(cfg.get("max_nodes", 100))
        sample_with_replacement = bool(cfg.get("sample_with_replacement", False))
        max_attempts_per_graph = int(filters.get("max_attempts_per_graph", 300))
        source_dataset = str(self.config.get("name", "ego"))

        if radius < 0:
            raise ValueError("ego radius must be non-negative.")
        if min_nodes <= 0 or max_nodes < min_nodes:
            raise ValueError("ego min/max node bounds are invalid.")

        rng = np.random.default_rng(seed)
        source = self._build_source_graph(cfg, rng)
        candidates = [node for node in source.nodes() if min_nodes <= len(nx.ego_graph(source, node, radius=radius)) <= max_nodes]
        if not candidates:
            raise RuntimeError(
                "No ego centers satisfy the configured min_nodes/max_nodes bounds. "
                "Relax ego filters or increase the source graph size."
            )

        if not sample_with_replacement and len(candidates) < num_graphs:
            raise RuntimeError(
                f"Need {num_graphs} ego centers, but only {len(candidates)} satisfy the configured bounds."
            )

        if sample_with_replacement:
            center_sequence = [candidates[int(rng.integers(0, len(candidates)))] for _ in range(num_graphs)]
        else:
            chosen = rng.choice(len(candidates), size=num_graphs, replace=False)
            center_sequence = [candidates[int(idx)] for idx in chosen]

        graphs: list[nx.Graph] = []
        attempts = 0
        max_attempts = num_graphs * max_attempts_per_graph
        for center in center_sequence:
            attempts += 1
            g = nx.ego_graph(source, center, radius=radius).copy()
            if not (min_nodes <= g.number_of_nodes() <= max_nodes):
                continue

            g = nx.convert_node_labels_to_integers(nx.Graph(g), first_label=0, ordering="sorted")
            g.graph.update(
                {
                    "source_dataset": source_dataset,
                    "dataset_type": "ego",
                    "radius": radius,
                    "center": int(center) if isinstance(center, (int, np.integer)) else str(center),
                }
            )
            graphs.append(g)

        if len(graphs) < num_graphs and attempts >= max_attempts:
            raise RuntimeError(f"Could not build {num_graphs} ego graphs; got {len(graphs)}.")
        if len(graphs) < num_graphs:
            raise RuntimeError(f"Could not build {num_graphs} ego graphs; got {len(graphs)}.")
        return graphs

    def _build_source_graph(self, cfg: dict[str, Any], rng: np.random.Generator) -> nx.Graph:
        edge_list_path = cfg.get("edge_list_path")
        if edge_list_path:
            g = nx.read_edgelist(Path(edge_list_path), nodetype=int)
            return nx.convert_node_labels_to_integers(nx.Graph(g), first_label=0, ordering="sorted")

        model = str(cfg.get("source_model", "barabasi_albert")).lower()
        source_nodes = int(cfg.get("source_nodes", 5000))
        graph_seed = int(rng.integers(0, 2**31 - 1))
        if model in {"barabasi_albert", "ba"}:
            m = int(cfg.get("ba_m", 4))
            if source_nodes <= m:
                raise ValueError("ego source_nodes must be greater than ba_m.")
            return nx.barabasi_albert_graph(source_nodes, m, seed=graph_seed)
        if model in {"erdos_renyi", "er"}:
            p = float(cfg.get("edge_probability", 0.002))
            return nx.fast_gnp_random_graph(source_nodes, p, seed=graph_seed)
        if model == "citeseer":
            return self._load_citeseer_source(cfg)
        raise ValueError(
            f"Unsupported ego source_model {model!r}; expected citeseer, barabasi_albert, or erdos_renyi."
        )

    def _load_citeseer_source(self, cfg: dict[str, Any]) -> nx.Graph:
        graph_path = cfg.get("source_graph_path")
        if graph_path is None:
            root = Path(cfg.get("source_root", "outputs/datasets/planetoid"))
            graph_path = root / "citeseer" / "raw" / "ind.citeseer.graph"
            if not graph_path.exists():
                graph_path.parent.mkdir(parents=True, exist_ok=True)
                url = str(
                    cfg.get(
                        "source_graph_url",
                        "https://github.com/kimiyoung/planetoid/raw/master/data/ind.citeseer.graph",
                    )
                )
                urllib.request.urlretrieve(url, graph_path)
        else:
            graph_path = Path(graph_path)

        with Path(graph_path).open("rb") as f:
            adjacency = pickle.load(f, encoding="latin1")

        g = nx.Graph()
        for node, neighbors in adjacency.items():
            node_id = int(node)
            g.add_node(node_id)
            for neighbor in neighbors:
                neighbor_id = int(neighbor)
                if node_id != neighbor_id:
                    g.add_edge(node_id, neighbor_id)
        return nx.convert_node_labels_to_integers(g, first_label=0, ordering="sorted")

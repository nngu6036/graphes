from __future__ import annotations

import pickle
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import networkx as nx
import numpy as np

SPLIT_NAMES = ("train", "val", "test")


class GraphDatasetBuilder(Protocol):
    def build_graphs(self) -> list[nx.Graph]: ...


@dataclass(frozen=True)
class SBMSpec:
    num_graphs: int = 200
    seed: int = 0
    min_blocks: int = 2
    max_blocks: int = 5
    min_nodes_per_block: int = 20
    max_nodes_per_block: int = 40
    min_total_nodes: int | None = None
    max_total_nodes: int | None = None
    equal_block_sizes: bool = False
    p_in: float = 0.30
    p_out: float = 0.005
    p_inter: float | None = None
    inter_edges_per_node_fraction: float | None = None
    ensure_inter_community_edge: bool = False
    require_connected: bool = True
    reject_zero_degree: bool = True
    max_attempts_per_graph: int = 300


def _sbm_spec_from_config(config: dict[str, Any]) -> SBMSpec:
    communities = config.get("communities", {}) or {}
    edge_probs = config.get("edge_probs", {}) or {}
    filters = config.get("filters", {}) or {}
    return SBMSpec(
        num_graphs=int(config.get("num_graphs", 200)),
        seed=int(config.get("seed", 0)),
        min_blocks=int(communities.get("min_blocks", 2)),
        max_blocks=int(communities.get("max_blocks", 5)),
        min_nodes_per_block=int(communities.get("min_nodes_per_block", 20)),
        max_nodes_per_block=int(communities.get("max_nodes_per_block", 40)),
        min_total_nodes=(
            int(communities["min_total_nodes"])
            if communities.get("min_total_nodes") is not None
            else None
        ),
        max_total_nodes=(
            int(communities["max_total_nodes"])
            if communities.get("max_total_nodes") is not None
            else None
        ),
        equal_block_sizes=bool(communities.get("equal_block_sizes", False)),
        p_in=float(edge_probs.get("p_in", 0.30)),
        p_out=float(edge_probs.get("p_out", 0.005)),
        p_inter=(
            float(edge_probs["p_inter"])
            if edge_probs.get("p_inter") is not None
            else None
        ),
        inter_edges_per_node_fraction=(
            float(edge_probs["inter_edges_per_node_fraction"])
            if edge_probs.get("inter_edges_per_node_fraction") is not None
            else None
        ),
        ensure_inter_community_edge=bool(
            edge_probs.get("ensure_inter_community_edge", False)
        ),
        require_connected=bool(filters.get("require_connected", True)),
        reject_zero_degree=bool(filters.get("reject_zero_degree", True)),
        max_attempts_per_graph=int(filters.get("max_attempts_per_graph", 300)),
    )


def _acceptable_sbm_graph(graph: nx.Graph, spec: SBMSpec) -> bool:
    if graph.number_of_nodes() <= 0:
        return False
    if (
        spec.require_connected
        and graph.number_of_nodes() > 1
        and not nx.is_connected(graph)
    ):
        return False
    if spec.reject_zero_degree and graph.number_of_nodes() > 1:
        if any(deg == 0 for _, deg in graph.degree()):
            return False
    return True


def _add_uniform_inter_community_edges(
    graph: nx.Graph,
    sizes: list[int],
    num_edges: int,
    rng: np.random.Generator,
) -> None:
    if num_edges <= 0:
        return

    communities: list[list[int]] = []
    offset = 0
    for size in sizes:
        communities.append(list(range(offset, offset + size)))
        offset += size

    candidates: list[tuple[int, int]] = []
    for i, left in enumerate(communities):
        for right in communities[i + 1 :]:
            candidates.extend(
                (u, v) for u in left for v in right if not graph.has_edge(u, v)
            )

    if num_edges > len(candidates):
        raise RuntimeError(
            f"Cannot add {num_edges} inter-community edges; only "
            f"{len(candidates)} candidates exist."
        )

    chosen = rng.choice(len(candidates), size=num_edges, replace=False)
    graph.add_edges_from(candidates[int(idx)] for idx in chosen)


def build_sbm_graphs(config: dict[str, Any]) -> list[nx.Graph]:
    spec = _sbm_spec_from_config(config)
    if spec.min_blocks <= 0 or spec.max_blocks < spec.min_blocks:
        raise ValueError("SBM min_blocks/max_blocks are invalid.")
    if (spec.min_total_nodes is None) != (spec.max_total_nodes is None):
        raise ValueError(
            "communities.min_total_nodes and max_total_nodes must be set together."
        )
    if (
        spec.min_total_nodes is not None
        and spec.max_total_nodes is not None
        and (
            spec.min_total_nodes < spec.max_blocks
            or spec.max_total_nodes < spec.min_total_nodes
        )
    ):
        raise ValueError("SBM min_total_nodes/max_total_nodes are invalid.")
    if not 0.0 <= spec.p_in <= 1.0 or not 0.0 <= spec.p_out <= 1.0:
        raise ValueError("SBM p_in and p_out must be in [0, 1].")
    if spec.p_inter is not None and spec.inter_edges_per_node_fraction is not None:
        raise ValueError(
            "SBM p_inter and inter_edges_per_node_fraction are mutually exclusive."
        )
    if spec.p_inter is not None and spec.p_inter < 0.0:
        raise ValueError("SBM p_inter must be non-negative.")
    if spec.inter_edges_per_node_fraction is not None and not (
        0.0 <= spec.inter_edges_per_node_fraction <= 1.0
    ):
        raise ValueError("SBM inter_edges_per_node_fraction must be in [0, 1].")
    rng = np.random.default_rng(spec.seed)
    graphs: list[nx.Graph] = []
    attempts = 0
    max_attempts = spec.num_graphs * spec.max_attempts_per_graph

    while len(graphs) < spec.num_graphs and attempts < max_attempts:
        attempts += 1
        k = int(rng.integers(spec.min_blocks, spec.max_blocks + 1))
        if spec.min_total_nodes is not None and spec.max_total_nodes is not None:
            sampled_total = int(
                rng.integers(spec.min_total_nodes, spec.max_total_nodes + 1)
            )
            if spec.equal_block_sizes:
                # Community-small follows the GDSS generator: sample a maximum
                # total size, then give every block floor(max_nodes / k) nodes.
                block_size = sampled_total // k
                sizes = [block_size] * k
            else:
                # Random positive composition of the requested total.
                cuts = sorted(
                    int(value)
                    for value in rng.choice(
                        np.arange(1, sampled_total),
                        size=k - 1,
                        replace=False,
                    )
                )
                sizes = [
                    right - left
                    for left, right in zip(
                        [0, *cuts],
                        [*cuts, sampled_total],
                    )
                ]
        elif spec.equal_block_sizes:
            block_size = int(
                rng.integers(
                    spec.min_nodes_per_block,
                    spec.max_nodes_per_block + 1,
                )
            )
            sizes = [block_size] * k
        else:
            sizes = (
                rng.integers(
                    spec.min_nodes_per_block,
                    spec.max_nodes_per_block + 1,
                    size=k,
                )
                .astype(int)
                .tolist()
            )

        inter_probability = spec.p_out
        if spec.p_inter is not None:
            total_cross_pairs = sum(
                sizes[i] * sizes[j] for i in range(k) for j in range(i + 1, k)
            )
            expected_inter_edges = spec.p_inter * sum(sizes)
            inter_probability = expected_inter_edges / total_cross_pairs
            if inter_probability > 1.0:
                raise ValueError(
                    "SBM p_inter requests more expected cross-community edges "
                    "than the sampled block sizes can realize."
                )
        probs = [
            [spec.p_in if i == j else inter_probability for j in range(k)]
            for i in range(k)
        ]
        graph_seed = int(rng.integers(0, 2**31 - 1))
        g = nx.stochastic_block_model(
            sizes,
            probs,
            seed=graph_seed,
            selfloops=False,
        )
        g = nx.convert_node_labels_to_integers(
            nx.Graph(g),
            first_label=0,
            ordering="sorted",
        )

        if spec.inter_edges_per_node_fraction is not None:
            num_inter_edges = int(
                round(spec.inter_edges_per_node_fraction * g.number_of_nodes())
            )
            if spec.ensure_inter_community_edge:
                num_inter_edges = max(num_inter_edges, 1)
            _add_uniform_inter_community_edges(g, sizes, num_inter_edges, rng)
        elif spec.ensure_inter_community_edge:
            offsets = np.cumsum([0, *sizes])
            community_of = {
                node: community
                for community, (start, stop) in enumerate(
                    zip(offsets[:-1], offsets[1:])
                )
                for node in range(int(start), int(stop))
            }
            if not any(
                community_of[int(u)] != community_of[int(v)] for u, v in g.edges()
            ):
                _add_uniform_inter_community_edges(g, sizes, 1, rng)

        community_labels: list[int] = []
        for community_id, size in enumerate(sizes):
            community_labels.extend([community_id] * size)
        nx.set_node_attributes(
            g,
            {i: int(c) for i, c in enumerate(community_labels)},
            "community",
        )
        g.graph.update(
            {
                "source_dataset": "sbm",
                "num_blocks": k,
                "block_sizes": sizes,
                "p_in": spec.p_in,
                "p_out": spec.p_out,
                "p_inter": spec.p_inter,
                "p_inter_edge_probability": (
                    inter_probability if spec.p_inter is not None else None
                ),
                "inter_edges_per_node_fraction": spec.inter_edges_per_node_fraction,
                "seed": graph_seed,
            }
        )

        if not _acceptable_sbm_graph(g, spec):
            continue
        graphs.append(g)

    if len(graphs) < spec.num_graphs:
        raise RuntimeError(
            f"Could not build {spec.num_graphs} acceptable SBM graphs; "
            f"got {len(graphs)} after {attempts} attempts."
        )
    return graphs


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


def split_graphs(
    graphs: list[nx.Graph],
    config: dict[str, Any],
) -> dict[str, list[nx.Graph]]:
    split_cfg = config.get("split", {}) or {}
    train_frac = float(split_cfg.get("train", 0.8))
    val_frac = float(split_cfg.get("val", 0.1))
    test_frac = float(split_cfg.get("test", 1.0 - train_frac - val_frac))
    seed = int(config.get("seed", 0))

    if min(train_frac, val_frac, test_frac) < 0.0:
        raise ValueError("split train/val/test fractions must be non-negative.")
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError("split train/val/test fractions must sum to 1.0.")

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


def get_dataset_builder(
    dataset_type: str,
    config: dict[str, Any],
) -> GraphDatasetBuilder:
    dataset_type = dataset_type.lower()
    if dataset_type == "sbm":
        return SBMDatasetBuilder(config)
    if dataset_type == "grid":
        return GridDatasetBuilder(config)
    if dataset_type == "ego":
        return EgoDatasetBuilder(config)
    raise ValueError(
        f"Unsupported dataset type {dataset_type!r}; expected sbm, grid, or ego."
    )


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
            raise ValueError(
                "grid max rows/cols must be greater than or equal to min rows/cols."
            )

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
                    f"Need {num_graphs} grid dimension pairs, but only "
                    f"{len(dimension_pairs)} are configured."
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
            g = nx.convert_node_labels_to_integers(
                nx.Graph(g),
                first_label=0,
                ordering="sorted",
            )

            if random_relabel:
                labels = rng.permutation(g.number_of_nodes()).tolist()
                mapping = {node: int(labels[node]) for node in g.nodes()}
                g = nx.relabel_nodes(g, mapping)
                g = nx.convert_node_labels_to_integers(
                    g,
                    first_label=0,
                    ordering="sorted",
                )

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
            raise RuntimeError(
                f"Could not build {num_graphs} grid graphs; got {len(graphs)}."
            )
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
        selection = str(cfg.get("selection", "random")).lower()
        max_attempts_per_graph = int(filters.get("max_attempts_per_graph", 300))
        source_dataset = str(self.config.get("name", "ego"))

        if radius < 0:
            raise ValueError("ego radius must be non-negative.")
        if min_nodes <= 0 or max_nodes < min_nodes:
            raise ValueError("ego min/max node bounds are invalid.")
        if selection not in {"first", "random"}:
            raise ValueError("ego.selection must be 'first' or 'random'.")
        if selection == "first" and sample_with_replacement:
            raise ValueError(
                "ego.selection='first' is incompatible with sample_with_replacement."
            )

        rng = np.random.default_rng(seed)
        source = self._build_source_graph(cfg, rng)
        if (
            bool(cfg.get("largest_connected_component", False))
            and source.number_of_nodes()
        ):
            components = list(nx.connected_components(source))
            largest = max(
                enumerate(components),
                key=lambda item: (len(item[1]), -item[0]),
            )[1]
            source = source.subgraph(largest).copy()
            source = nx.convert_node_labels_to_integers(
                nx.Graph(source),
                first_label=0,
                ordering="sorted",
            )
        candidates = [
            node
            for node in source.nodes()
            if min_nodes <= len(nx.ego_graph(source, node, radius=radius)) <= max_nodes
        ]
        if not candidates:
            raise RuntimeError(
                "No ego centers satisfy the configured min_nodes/max_nodes bounds. "
                "Relax ego filters or increase the source graph size."
            )

        if not sample_with_replacement and len(candidates) < num_graphs:
            raise RuntimeError(
                f"Need {num_graphs} ego centers, but only {len(candidates)} "
                "satisfy the configured bounds."
            )

        if selection == "first":
            center_sequence = candidates[:num_graphs]
        elif sample_with_replacement:
            center_sequence = [
                candidates[int(rng.integers(0, len(candidates)))]
                for _ in range(num_graphs)
            ]
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

            g = nx.convert_node_labels_to_integers(
                nx.Graph(g),
                first_label=0,
                ordering="sorted",
            )
            g.graph.update(
                {
                    "source_dataset": source_dataset,
                    "dataset_type": "ego",
                    "radius": radius,
                    "center": (
                        int(center)
                        if isinstance(center, (int, np.integer))
                        else str(center)
                    ),
                }
            )
            graphs.append(g)

        if len(graphs) < num_graphs and attempts >= max_attempts:
            raise RuntimeError(
                f"Could not build {num_graphs} ego graphs; got {len(graphs)}."
            )
        if len(graphs) < num_graphs:
            raise RuntimeError(
                f"Could not build {num_graphs} ego graphs; got {len(graphs)}."
            )
        return graphs

    def _build_source_graph(
        self,
        cfg: dict[str, Any],
        rng: np.random.Generator,
    ) -> nx.Graph:
        edge_list_path = cfg.get("edge_list_path")
        if edge_list_path:
            g = nx.read_edgelist(Path(edge_list_path), nodetype=int)
            return nx.convert_node_labels_to_integers(
                nx.Graph(g),
                first_label=0,
                ordering="sorted",
            )

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
            f"Unsupported ego source_model {model!r}; expected citeseer, "
            "barabasi_albert, or erdos_renyi."
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
                        "https://github.com/kimiyoung/planetoid/raw/master/"
                        "data/ind.citeseer.graph",
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

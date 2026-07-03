from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np


@dataclass(frozen=True)
class SBMSpec:
    num_graphs: int = 200
    seed: int = 0
    min_blocks: int = 2
    max_blocks: int = 5
    min_nodes_per_block: int = 20
    max_nodes_per_block: int = 40
    p_in: float = 0.30
    p_out: float = 0.005
    require_connected: bool = True
    reject_zero_degree: bool = True
    max_attempts_per_graph: int = 300


def _spec_from_config(config: dict[str, Any]) -> SBMSpec:
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
        p_in=float(edge_probs.get("p_in", 0.30)),
        p_out=float(edge_probs.get("p_out", 0.005)),
        require_connected=bool(filters.get("require_connected", True)),
        reject_zero_degree=bool(filters.get("reject_zero_degree", True)),
        max_attempts_per_graph=int(filters.get("max_attempts_per_graph", 300)),
    )


def _acceptable(graph: nx.Graph, spec: SBMSpec) -> bool:
    if graph.number_of_nodes() <= 0:
        return False
    if spec.require_connected and graph.number_of_nodes() > 1 and not nx.is_connected(graph):
        return False
    if spec.reject_zero_degree and graph.number_of_nodes() > 1:
        if any(deg == 0 for _, deg in graph.degree()):
            return False
    return True


def build_sbm_graphs(config: dict[str, Any]) -> list[nx.Graph]:
    """Build SPECTRE-style stochastic block model graphs.

    This is intentionally small and dependency-free. It is the first generic
    dataset for the coarse-to-fine proposal.
    """

    spec = _spec_from_config(config)
    rng = np.random.default_rng(spec.seed)
    graphs: list[nx.Graph] = []
    attempts = 0
    max_attempts = spec.num_graphs * spec.max_attempts_per_graph

    while len(graphs) < spec.num_graphs and attempts < max_attempts:
        attempts += 1
        k = int(rng.integers(spec.min_blocks, spec.max_blocks + 1))
        sizes = rng.integers(spec.min_nodes_per_block, spec.max_nodes_per_block + 1, size=k).astype(int).tolist()
        probs = [[spec.p_in if i == j else spec.p_out for j in range(k)] for i in range(k)]
        graph_seed = int(rng.integers(0, 2**31 - 1))
        g = nx.stochastic_block_model(sizes, probs, seed=graph_seed, selfloops=False)
        g = nx.convert_node_labels_to_integers(nx.Graph(g), first_label=0, ordering="sorted")

        # Preserve community labels as optional metadata for later diagnostics.
        community_labels: list[int] = []
        for community_id, size in enumerate(sizes):
            community_labels.extend([community_id] * size)
        nx.set_node_attributes(g, {i: int(c) for i, c in enumerate(community_labels)}, "community")
        g.graph.update(
            {
                "source_dataset": "sbm_spectre",
                "num_blocks": k,
                "block_sizes": sizes,
                "p_in": spec.p_in,
                "p_out": spec.p_out,
                "seed": graph_seed,
            }
        )

        if not _acceptable(g, spec):
            continue
        graphs.append(g)

    if len(graphs) < spec.num_graphs:
        raise RuntimeError(
            f"Could not build {spec.num_graphs} acceptable SBM graphs; "
            f"got {len(graphs)} after {attempts} attempts."
        )
    return graphs


def split_graphs(graphs: list[nx.Graph], config: dict[str, Any]) -> dict[str, list[nx.Graph]]:
    split_cfg = config.get("split", {}) or {}
    train_frac = float(split_cfg.get("train", 0.8))
    val_frac = float(split_cfg.get("val", 0.1))
    seed = int(config.get("seed", 0))
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

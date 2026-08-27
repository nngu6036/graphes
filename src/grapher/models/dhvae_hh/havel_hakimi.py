from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np


@dataclass(frozen=True)
class ConstructorConfig:
    type: str = "havel_hakimi"
    ensure_connected: bool = True
    random_relabel: bool = True
    max_repair_trials: int = 10000
    deterministic_seed: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None = None) -> "ConstructorConfig":
        data = data or {}
        return cls(
            type=str(data.get("type", "havel_hakimi")),
            ensure_connected=bool(data.get("ensure_connected", True)),
            random_relabel=bool(data.get("random_relabel", True)),
            max_repair_trials=int(data.get("max_repair_trials", 10000)),
            deterministic_seed=(
                None
                if data.get("deterministic_seed") in {None, "", "none", "None"}
                else int(data.get("deterministic_seed"))
            ),
        )


def _random_relabel(graph: nx.Graph, rng: np.random.Generator) -> nx.Graph:
    nodes = list(graph.nodes())
    permuted = rng.permutation(nodes).tolist()
    mapping = {old: int(new) for old, new in zip(nodes, permuted)}
    out = nx.relabel_nodes(graph, mapping, copy=True)
    return nx.convert_node_labels_to_integers(out, first_label=0, ordering="sorted")


def _try_connect_two_components(
    graph: nx.Graph,
    c1: set[int],
    c2: set[int],
    rng: np.random.Generator,
    max_trials: int,
) -> bool:
    edges1 = list(graph.subgraph(c1).edges())
    edges2 = list(graph.subgraph(c2).edges())
    if not edges1 or not edges2:
        return False
    for _ in range(max_trials):
        u, v = edges1[int(rng.integers(0, len(edges1)))]
        x, y = edges2[int(rng.integers(0, len(edges2)))]
        if len({u, v, x, y}) < 4:
            continue
        proposals = [((u, x), (v, y)), ((u, y), (v, x))]
        rng.shuffle(proposals)
        for e_new1, e_new2 in proposals:
            a, b = e_new1
            c, d = e_new2
            if a == b or c == d:
                continue
            if graph.has_edge(a, b) or graph.has_edge(c, d):
                continue
            graph.remove_edge(u, v)
            graph.remove_edge(x, y)
            graph.add_edge(a, b)
            graph.add_edge(c, d)
            return True
    return False


def repair_connectivity_degree_preserving(
    graph: nx.Graph, rng: np.random.Generator, max_trials: int = 10000
) -> nx.Graph:
    """Connect components using degree-preserving switches.

    This cannot repair isolated zero-degree components because no degree-
    preserving operation can connect an isolated node.
    """

    g = graph.copy()
    if g.number_of_nodes() <= 1 or nx.is_connected(g):
        return g
    attempts = 0
    while not nx.is_connected(g):
        components = [set(c) for c in nx.connected_components(g)]
        components = sorted(components, key=len, reverse=True)
        c1, c2 = components[0], components[1]
        success = _try_connect_two_components(
            g, c1, c2, rng, max_trials=max(1, max_trials // max(len(components), 1))
        )
        attempts += 1
        if not success or attempts > len(components) + max_trials:
            raise RuntimeError(
                "Could not connect graph with degree-preserving switches."
            )
    return g


def _deterministic_constructor_rng(
    degree_sequence: list[int],
    seed: int,
) -> np.random.Generator:
    """Create a stable RNG without relying on Python's randomized hash()."""

    payload = (
        int(seed).to_bytes(8, byteorder="little", signed=True)
        + np.asarray(degree_sequence, dtype=np.int64).tobytes()
    )
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    stable_seed = int.from_bytes(digest, byteorder="little", signed=False)
    return np.random.default_rng(stable_seed)


def construct_coarse_graph(
    summary: dict[str, Any],
    config: ConstructorConfig | dict[str, Any] | None = None,
    rng: np.random.Generator | None = None,
) -> nx.Graph:
    cfg = (
        config
        if isinstance(config, ConstructorConfig)
        else ConstructorConfig.from_dict(config)
    )
    degree_sequence = sorted(
        [int(d) for d in summary["degree_sequence"]],
        reverse=True,
    )
    generator = (
        _deterministic_constructor_rng(degree_sequence, cfg.deterministic_seed)
        if cfg.deterministic_seed is not None
        else (rng if rng is not None else np.random.default_rng(0))
    )
    if cfg.type != "havel_hakimi":
        raise ValueError(
            f"Unsupported coarse constructor {cfg.type!r}; expected 'havel_hakimi'."
        )
    if not nx.is_graphical(degree_sequence, method="eg"):
        raise ValueError("Target degree sequence is not graphical.")
    graph = nx.havel_hakimi_graph(degree_sequence)
    graph = nx.convert_node_labels_to_integers(
        nx.Graph(graph), first_label=0, ordering="sorted"
    )
    if (
        cfg.ensure_connected
        and graph.number_of_nodes() > 1
        and not nx.is_connected(graph)
    ):
        graph = repair_connectivity_degree_preserving(
            graph, generator, cfg.max_repair_trials
        )
    if cfg.random_relabel:
        graph = _random_relabel(graph, generator)
    graph.graph["constructor"] = cfg.type
    return graph


def construct_indexed_havel_hakimi(
    degree_sequence: list[int],
    *,
    ensure_connected: bool = True,
    max_repair_trials: int = 10000,
    rng: np.random.Generator | None = None,
) -> nx.Graph:
    """Realize an ordered degree sequence while retaining node correspondence.

    Unlike :func:`construct_coarse_graph`, this helper does not sort or randomly
    relabel the sequence. It supports the legacy endpoint-supervision path,
    where entry ``i`` must remain the degree of node ``i``. The implementation
    lives with the baseline so no Havel--Hakimi algorithm remains embedded in
    the rewiring modules.
    """

    degrees = [int(value) for value in degree_sequence]
    if not nx.is_graphical(degrees, method="eg"):
        raise ValueError("Target degree sequence is not graphical.")
    graph = nx.havel_hakimi_graph(degrees)
    graph = nx.convert_node_labels_to_integers(
        nx.Graph(graph),
        first_label=0,
        ordering="sorted",
    )
    if [int(graph.degree(node)) for node in range(len(degrees))] != degrees:
        raise RuntimeError("Havel--Hakimi did not preserve indexed-node degrees.")
    if (
        ensure_connected
        and graph.number_of_nodes() > 1
        and not nx.is_connected(graph)
    ):
        graph = repair_connectivity_degree_preserving(
            graph,
            rng if rng is not None else np.random.default_rng(0),
            max_trials=max_repair_trials,
        )
    if (
        ensure_connected
        and graph.number_of_nodes() > 1
        and not nx.is_connected(graph)
    ):
        raise RuntimeError("Could not construct a connected indexed HH source.")
    graph.graph["constructor"] = "indexed_havel_hakimi"
    return graph


def assert_constructor_validity(
    graph: nx.Graph, summary: dict[str, Any], require_connected: bool = True
) -> None:
    expected_degrees = sorted(
        [int(d) for d in summary["degree_sequence"]], reverse=True
    )
    actual_degrees = sorted([int(d) for _, d in graph.degree()], reverse=True)
    if graph.number_of_nodes() != int(summary["num_nodes"]):
        raise AssertionError("Node count mismatch.")
    if actual_degrees != expected_degrees:
        raise AssertionError("Degree sequence mismatch.")
    if nx.number_of_selfloops(graph) != 0:
        raise AssertionError("Self-loops found.")
    if require_connected and graph.number_of_nodes() > 1 and not nx.is_connected(graph):
        raise AssertionError("Graph is disconnected.")

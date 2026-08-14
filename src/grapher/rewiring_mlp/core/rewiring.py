from __future__ import annotations

from typing import Iterable, TypeAlias

import networkx as nx
import numpy as np

Edge: TypeAlias = tuple[int, int]
Action: TypeAlias = tuple[tuple[Edge, Edge], tuple[Edge, Edge]]


def canonical_edge(u: int, v: int) -> Edge:
    if u == v:
        raise ValueError("Self-loop edge is not allowed.")
    return (int(u), int(v)) if u < v else (int(v), int(u))


def canonical_edges(edges: Iterable[tuple[int, int]]) -> tuple[Edge, Edge]:
    canonical = tuple(sorted(canonical_edge(u, v) for u, v in edges))
    if len(canonical) != 2:
        raise ValueError("A rewiring action must contain exactly two edges.")
    return canonical  # type: ignore[return-value]


def make_action(
    removed: Iterable[tuple[int, int]], added: Iterable[tuple[int, int]]
) -> Action:
    return (canonical_edges(removed), canonical_edges(added))


def apply_action(graph: nx.Graph, action: Action) -> nx.Graph:
    removed, added = action
    g = graph.copy()
    for u, v in removed:
        g.remove_edge(u, v)
    for u, v in added:
        g.add_edge(u, v)
    return g


def is_valid_action(
    graph: nx.Graph, action: Action, preserve_connectivity: bool = True
) -> bool:
    removed, added = action
    if len(set(removed)) != 2 or len(set(added)) != 2:
        return False
    endpoints = [node for edge in removed for node in edge]
    if len(set(endpoints)) != 4:
        return False
    for edge in removed:
        if not graph.has_edge(*edge):
            return False
    for u, v in added:
        if u == v:
            return False
        if graph.has_edge(u, v):
            return False
    if set(removed) & set(added):
        return False
    try:
        candidate = apply_action(graph, action)
    except Exception:
        return False
    if candidate.number_of_edges() != graph.number_of_edges():
        return False
    if nx.number_of_selfloops(candidate) != 0:
        return False
    if (
        preserve_connectivity
        and candidate.number_of_nodes() > 1
        and not nx.is_connected(candidate)
    ):
        return False
    return True


def candidate_actions_from_edge_pair(e1: Edge, e2: Edge) -> list[Action]:
    u, v = e1
    x, y = e2
    if len({u, v, x, y}) < 4:
        return []
    raw = [
        make_action([e1, e2], [(u, x), (v, y)]),
        make_action([e1, e2], [(u, y), (v, x)]),
    ]
    # Deduplicate in case of symmetries.
    return list(dict.fromkeys(raw))


def enumerate_valid_double_edge_swaps(
    graph: nx.Graph, preserve_connectivity: bool = True
) -> list[Action]:
    edges = [canonical_edge(u, v) for u, v in graph.edges()]
    out: list[Action] = []
    seen: set[Action] = set()
    for i in range(len(edges)):
        for j in range(i + 1, len(edges)):
            for action in candidate_actions_from_edge_pair(edges[i], edges[j]):
                if action in seen:
                    continue
                if is_valid_action(
                    graph, action, preserve_connectivity=preserve_connectivity
                ):
                    out.append(action)
                    seen.add(action)
    return out


def sample_valid_double_edge_swaps(
    graph: nx.Graph,
    budget: int,
    rng: np.random.Generator,
    *,
    preserve_connectivity: bool = True,
    max_attempts: int | None = None,
) -> list[Action]:
    edges = [canonical_edge(u, v) for u, v in graph.edges()]
    if len(edges) < 2 or budget <= 0:
        return []
    max_attempts = max_attempts or max(100, budget * 50)
    out: list[Action] = []
    seen: set[Action] = set()
    attempts = 0
    while len(out) < budget and attempts < max_attempts:
        attempts += 1
        idx = rng.choice(len(edges), size=2, replace=False)
        e1, e2 = edges[int(idx[0])], edges[int(idx[1])]
        actions = candidate_actions_from_edge_pair(e1, e2)
        rng.shuffle(actions)
        for action in actions:
            if action in seen:
                continue
            if is_valid_action(
                graph, action, preserve_connectivity=preserve_connectivity
            ):
                out.append(action)
                seen.add(action)
                if len(out) >= budget:
                    break
    return out

from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection
from typing import Any

import networkx as nx
import numpy as np

from grapher.rewiring_mlp.core.rewiring import (
    Action,
    apply_action,
    candidate_actions_from_edge_pair,
    canonical_edge,
)


def topology_state_key(graph: nx.Graph) -> bytes:
    """Return a label-sensitive key for cycle detection inside one degree fibre."""

    nodes = sorted(graph.nodes())
    adjacency = nx.to_numpy_array(graph, nodelist=nodes, dtype=np.uint8)
    upper = adjacency[np.triu_indices(len(nodes), k=1)]
    return int(len(nodes)).to_bytes(8, "little") + np.packbits(upper).tobytes()


def _fast_action_is_connected(
    *,
    nodes: list[int],
    adjacency: dict[int, set[int]],
    action: Action,
) -> bool:
    """Connectivity check on a virtual swap without constructing a graph.

    Only four incidences change in a double-edge swap.  A small BFS over the
    original adjacency with those incidences virtually removed/added avoids a
    NetworkX copy and ``nx.is_connected`` call for every raw proposal.
    """

    if len(nodes) <= 1:
        return True
    removed, added = action
    removed_set = {canonical_edge(*edge) for edge in removed}
    added_by_node: defaultdict[int, set[int]] = defaultdict(set)
    for u, v in added:
        added_by_node[u].add(v)
        added_by_node[v].add(u)

    start = nodes[0]
    seen = {start}
    stack = [start]
    while stack:
        u = stack.pop()
        for v in adjacency[u]:
            if canonical_edge(u, v) in removed_set:
                continue
            if v not in seen:
                seen.add(v)
                stack.append(v)
        for v in added_by_node.get(u, ()):
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return len(seen) == len(nodes)


def _fast_valid_action(
    graph: nx.Graph,
    action: Action,
    *,
    edge_set: set[tuple[int, int]],
    nodes: list[int],
    adjacency: dict[int, set[int]],
    preserve_connectivity: bool,
) -> bool:
    removed, added = action
    if len(set(removed)) != 2 or len(set(added)) != 2:
        return False
    endpoints = [node for edge in removed for node in edge]
    if len(set(endpoints)) != 4:
        return False
    if any(canonical_edge(*edge) not in edge_set for edge in removed):
        return False
    for u, v in added:
        if u == v or canonical_edge(u, v) in edge_set:
            return False
    if set(removed) & set(added):
        return False
    return (not preserve_connectivity) or _fast_action_is_connected(
        nodes=nodes,
        adjacency=adjacency,
        action=action,
    )


def _state_key_after_action(
    *,
    node_count: int,
    base_upper: np.ndarray,
    edge_position: dict[tuple[int, int], int],
    action: Action,
) -> bytes:
    upper = base_upper.copy()
    removed, added = action
    for edge in removed:
        upper[edge_position[canonical_edge(*edge)]] = False
    for edge in added:
        upper[edge_position[canonical_edge(*edge)]] = True
    return int(node_count).to_bytes(8, "little") + np.packbits(upper).tobytes()


def propose_valid_topology_swaps(
    graph: nx.Graph,
    *,
    proposal_budget: int,
    valid_candidate_budget: int,
    preserve_connectivity: bool,
    rng: np.random.Generator,
    excluded_states: Collection[bytes] | None = None,
) -> tuple[list[Action], dict[Action, nx.Graph], dict[str, Any]]:
    """Propose ordinary double-edge swaps without access to a target adjacency.

    A negative proposal budget requests exhaustive edge-pair enumeration. A
    negative valid-candidate budget retains every valid proposal. The function
    deliberately knows nothing about a terminal graph: candidates are filtered
    only by hard topology constraints and optional tabu state keys.
    """

    if int(proposal_budget) == 0:
        raise ValueError(
            "proposal_budget must be positive or negative for exhaustive search."
        )
    if int(valid_candidate_budget) == 0:
        raise ValueError(
            "valid_candidate_budget must be positive or negative to keep every "
            "candidate."
        )
    if graph.is_directed() or graph.is_multigraph():
        raise ValueError("Topology rewiring requires a simple undirected graph.")
    if nx.number_of_selfloops(graph):
        raise ValueError("Topology rewiring does not support self-loops.")

    edges = [canonical_edge(u, v) for u, v in graph.edges()]
    edge_set = set(edges)
    nodes = sorted(int(node) for node in graph.nodes())
    adjacency = {node: {int(v) for v in graph.neighbors(node)} for node in nodes}
    seen: set[Action] = set()
    excluded = set(excluded_states or ())

    edge_position: dict[tuple[int, int], int] = {}
    base_upper = np.zeros(len(nodes) * (len(nodes) - 1) // 2, dtype=np.bool_)
    position = 0
    for left in range(len(nodes)):
        for right in range(left + 1, len(nodes)):
            edge = canonical_edge(nodes[left], nodes[right])
            edge_position[edge] = position
            base_upper[position] = edge in edge_set
            position += 1

    def raw_actions():
        if len(edges) < 2:
            return
        if int(proposal_budget) < 0:
            for left in range(len(edges)):
                for right in range(left + 1, len(edges)):
                    for action in candidate_actions_from_edge_pair(
                        edges[left], edges[right]
                    ):
                        if action not in seen:
                            seen.add(action)
                            yield action
            return

        attempts = 0
        max_attempts = max(100, int(proposal_budget) * 50)
        while len(seen) < int(proposal_budget) and attempts < max_attempts:
            attempts += 1
            indices = rng.choice(len(edges), size=2, replace=False)
            actions = candidate_actions_from_edge_pair(
                edges[int(indices[0])],
                edges[int(indices[1])],
            )
            rng.shuffle(actions)
            for action in actions:
                if action in seen:
                    continue
                seen.add(action)
                yield action
                if len(seen) >= int(proposal_budget):
                    break

    candidates: list[Action] = []
    candidate_graphs: dict[Action, nx.Graph] = {}
    rejections: defaultdict[str, int] = defaultdict(int)
    for action in raw_actions():
        if not _fast_valid_action(
            graph,
            action,
            edge_set=edge_set,
            nodes=nodes,
            adjacency=adjacency,
            preserve_connectivity=preserve_connectivity,
        ):
            rejections["topology_or_connectivity"] += 1
            continue
        if excluded and _state_key_after_action(
            node_count=len(nodes),
            base_upper=base_upper,
            edge_position=edge_position,
            action=action,
        ) in excluded:
            rejections["visited_state"] += 1
            continue
        # Materialize the candidate only after all cheap validity/tabu checks.
        candidate = apply_action(graph, action)
        candidates.append(action)
        candidate_graphs[action] = candidate
        if (
            int(valid_candidate_budget) > 0
            and len(candidates) >= int(valid_candidate_budget)
        ):
            break

    diagnostics = {
        "proposal_budget": int(proposal_budget),
        "valid_candidate_budget": int(valid_candidate_budget),
        "num_proposals": len(seen),
        "num_valid_candidates": len(candidates),
        "candidate_pass_rate": float(len(candidates) / max(len(seen), 1)),
        "candidate_rejection_reasons": dict(sorted(rejections.items())),
    }
    return candidates, candidate_graphs, diagnostics

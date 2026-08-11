from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection
from typing import Any

import networkx as nx
import numpy as np

from grapher.refinement.rewiring import (
    Action,
    apply_action,
    candidate_actions_from_edge_pair,
    canonical_edge,
    is_valid_action,
)


def topology_state_key(graph: nx.Graph) -> bytes:
    """Return a label-sensitive key for cycle detection inside one degree fibre."""

    nodes = sorted(graph.nodes())
    adjacency = nx.to_numpy_array(graph, nodelist=nodes, dtype=np.uint8)
    upper = adjacency[np.triu_indices(len(nodes), k=1)]
    return int(len(nodes)).to_bytes(8, "little") + np.packbits(upper).tobytes()


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
    seen: set[Action] = set()
    excluded = set(excluded_states or ())

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
        if not is_valid_action(
            graph,
            action,
            preserve_connectivity=preserve_connectivity,
        ):
            rejections["topology_or_connectivity"] += 1
            continue
        candidate = apply_action(graph, action)
        if excluded and topology_state_key(candidate) in excluded:
            rejections["visited_state"] += 1
            continue
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

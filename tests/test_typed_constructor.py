from __future__ import annotations

from collections import Counter

import networkx as nx
import numpy as np
import pytest

from grapher.construction.typed import (
    TypedConstructionError,
    construct_typed_graph,
)
from grapher.molecular.typed_invariants import (
    TypedDegreeSignature,
    TypedInvariant,
    extract_typed_invariant,
    typed_invariant_matches_graph,
)


def _typed_cycle() -> nx.Graph:
    graph = nx.Graph()
    for node, atom in enumerate((6, 7, 8, 9)):
        graph.add_node(node, atomic_num=atom)
    graph.add_edge(0, 1, bond_type=1)
    graph.add_edge(1, 2, bond_type=2)
    graph.add_edge(2, 3, bond_type=1)
    graph.add_edge(3, 0, bond_type=2)
    return graph


def test_exact_indexed_typed_realization_and_diagnostics() -> None:
    invariant = extract_typed_invariant(_typed_cycle(), edge_types=(1, 2))
    graph, diagnostics = construct_typed_graph(
        invariant,
        {
            "randomize_assignment": False,
            "max_restarts": 0,
            "max_backtracks": 1_000,
        },
        np.random.default_rng(4),
    )

    assert typed_invariant_matches_graph(graph, invariant)
    assert nx.is_connected(graph)
    assert nx.number_of_selfloops(graph) == 0
    assert graph.number_of_edges() == 4
    assert diagnostics["success"] is True
    assert diagnostics["trials"] == 1
    assert diagnostics["restarts"] == 0
    assert diagnostics["failure_reason"] is None
    assert diagnostics["assignment"] == [0, 1, 2, 3]


def test_random_assignment_preserves_complete_signature_multiset() -> None:
    invariant = extract_typed_invariant(_typed_cycle(), edge_types=(1, 2))
    graph, diagnostics = construct_typed_graph(
        invariant,
        {"randomize_assignment": True, "max_restarts": 0},
        np.random.default_rng(0),
    )
    observed = extract_typed_invariant(graph, edge_types=(1, 2))
    assignment = diagnostics["assignment"]

    assert assignment != [0, 1, 2, 3]
    assert sorted(assignment) == [0, 1, 2, 3]
    assert Counter(observed.signatures) == Counter(invariant.signatures)
    assert observed.signatures == tuple(
        invariant.signatures[index] for index in assignment
    )


def test_endpoint_compatibility_is_enforced_during_search() -> None:
    invariant = TypedInvariant(
        signatures=(
            TypedDegreeSignature("C", (2,)),
            TypedDegreeSignature("C", (2,)),
            TypedDegreeSignature("N", (2,)),
            TypedDegreeSignature("N", (2,)),
        ),
        edge_types=(1,),
        node_attribute="atom",
        edge_attribute="bond",
    )

    def compatible(left, right, _edge_type):
        return left != right

    graph, _ = construct_typed_graph(
        invariant,
        {"randomize_assignment": False, "max_restarts": 0},
        np.random.default_rng(2),
        endpoint_compatible=compatible,
    )

    assert nx.is_connected(graph)
    assert typed_invariant_matches_graph(graph, invariant)
    assert all(
        graph.nodes[u]["atom"] != graph.nodes[v]["atom"] for u, v in graph.edges()
    )


def test_empirical_ranking_hook_guides_order_without_changing_feasibility() -> None:
    invariant = extract_typed_invariant(_typed_cycle(), edge_types=(1, 2))
    calls: list[tuple[int, object, object]] = []

    def score(edge_type, left, right) -> float:
        calls.append((edge_type, left.node_type, right.node_type))
        return 10.0 if {left.node_type, right.node_type} == {6, 7} else 1.0

    graph, diagnostics = construct_typed_graph(
        invariant,
        {
            "randomize_assignment": False,
            "candidate_ranking": "empirical",
            "candidate_temperature": 0.0,
            "max_restarts": 0,
        },
        np.random.default_rng(8),
        initialization_score=score,
    )

    assert calls
    assert typed_invariant_matches_graph(graph, invariant)
    assert diagnostics["candidate_ranking"] == "empirical"


def test_simultaneously_unrealizable_types_fail_with_search_diagnostics() -> None:
    # Both per-type degree sequences and the aggregate degree sequence are
    # graphical, but their realizations cannot be edge-disjoint.  This checks
    # the one-edge-per-pair constraint beyond the inexpensive precheck.
    invariant = TypedInvariant(
        signatures=tuple(
            TypedDegreeSignature(6, value)
            for value in ((2, 1), (1, 3), (1, 0), (0, 2), (0, 2))
        ),
        edge_types=(1, 2),
    )

    with pytest.raises(TypedConstructionError) as captured:
        construct_typed_graph(
            invariant,
            {
                "randomize_assignment": False,
                "max_restarts": 0,
                "max_backtracks": 20_000,
            },
            np.random.default_rng(5),
        )

    diagnostics = captured.value.diagnostics
    assert diagnostics["success"] is False
    assert diagnostics["trials"] == 1
    assert diagnostics["restarts"] == 0
    assert diagnostics["failure_reason"] == "no_typed_realization"


def test_precheck_failure_exposes_zero_trial_diagnostics() -> None:
    invariant = TypedInvariant(
        signatures=(
            TypedDegreeSignature(6, (1,)),
            TypedDegreeSignature(6, (0,)),
        ),
        edge_types=(1,),
    )

    with pytest.raises(TypedConstructionError) as captured:
        construct_typed_graph(
            invariant,
            {"ensure_connected": False},
            np.random.default_rng(1),
        )

    diagnostics = captured.value.diagnostics
    assert diagnostics["success"] is False
    assert diagnostics["trials"] == 0
    assert diagnostics["restarts"] == 0
    assert diagnostics["failure_reason"] == "precheck_failed"
    assert diagnostics["precheck_errors"]

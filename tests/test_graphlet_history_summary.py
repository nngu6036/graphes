from __future__ import annotations

import networkx as nx

from grapher.properties.summary import (
    SummaryConfig,
    distance_to_summary,
    extract_summary,
)
from grapher.utils.motifs import (
    graphlet_history,
    graphlet_history_l2_distance,
    induced_simple_cycle_node_sets,
    topology_graphlet_keys_by_size,
)


def test_graphlet_history_is_permutation_invariant():
    g = nx.cycle_graph(6)
    g.add_edge(0, 3)
    gp = nx.relabel_nodes(g, {i: (i * 5) % 6 for i in range(6)}, copy=True)
    h1 = graphlet_history(g, k_min=3, k_max=4)
    h2 = graphlet_history(gp, k_min=3, k_max=4)
    assert h1 == h2


def test_graphlet_energy_detects_difference():
    cfg = SummaryConfig(graphlet_history=True, graphlet_k_min=3, graphlet_k_max=4)
    target = extract_summary(nx.cycle_graph(6), cfg)
    same = distance_to_summary(nx.cycle_graph(6), target, cfg, {"graphlet_weight": 1.0})
    other = distance_to_summary(nx.path_graph(6), target, cfg, {"graphlet_weight": 1.0})
    assert same <= other
    assert (
        graphlet_history_l2_distance(
            target["graphlet_history"],
            target["graphlet_history"],
        )
        == 0.0
    )


def test_cycle_only_summary_alias_and_complete_topology_basis() -> None:
    cfg = SummaryConfig.from_dict(
        {
            "graphlet_history": True,
            "graphlet_cycle_only": True,
            "graphlet_k_min": 3,
            "graphlet_k_max": 4,
        }
    )
    assert cfg.graphlet_topology_filter == "simple_cycle"
    cycle_history = graphlet_history(
        nx.cycle_graph(4),
        k_min=4,
        k_max=4,
        topology_filter="simple_cycle",
    )
    path_history = graphlet_history(
        nx.path_graph(4),
        k_min=4,
        k_max=4,
        topology_filter="simple_cycle",
    )
    assert sum(cycle_history["4"].values()) == 1.0
    assert path_history["4"] == {}

    all_keys = topology_graphlet_keys_by_size(3, 4)
    cycle_keys = topology_graphlet_keys_by_size(
        3, 4, topology_filter="simple_cycle"
    )
    assert [len(all_keys[str(k)]) for k in (3, 4)] == [2, 6]
    assert [len(cycle_keys[str(k)]) for k in (3, 4)] == [1, 1]


def test_direct_induced_cycle_enumerator_returns_each_ring_once() -> None:
    graph = nx.cycle_graph(4)
    graph.add_edges_from(((3, 4), (4, 5), (5, 3)))

    triangles = {
        frozenset(nodes) for nodes in induced_simple_cycle_node_sets(graph, 3)
    }
    squares = {
        frozenset(nodes) for nodes in induced_simple_cycle_node_sets(graph, 4)
    }

    assert triangles == {frozenset((3, 4, 5))}
    assert squares == {frozenset((0, 1, 2, 3))}

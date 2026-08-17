from __future__ import annotations

import networkx as nx

from grapher.properties.summary import (
    SummaryConfig,
    distance_to_summary,
    extract_summary,
)
from grapher.utils.motifs import graphlet_history, graphlet_history_l2_distance


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

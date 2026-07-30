from __future__ import annotations

import networkx as nx
import numpy as np

from grapher.evaluation.metrics import degree_target_match_rate
from grapher.properties.summary import extract_summary
from grapher.refinement.grapher_opt import refine_graph


def test_refiner_preserves_degree_sequence():
    g = nx.connected_watts_strogatz_graph(20, 4, 0.2, seed=0)
    target = extract_summary(nx.connected_watts_strogatz_graph(20, 4, 0.5, seed=1))
    out = refine_graph(
        g,
        target,
        refiner_config={"steps": 3, "candidate_budget": 16, "selection": "greedy"},
        rng=np.random.default_rng(0),
    )
    assert sorted(dict(g.degree()).values()) == sorted(dict(out.degree()).values())
    assert nx.is_connected(out)


def test_degree_target_match_rate_is_permutation_invariant():
    graph = nx.path_graph(5)
    relabeled = nx.relabel_nodes(
        graph,
        {0: 4, 1: 2, 2: 0, 3: 3, 4: 1},
        copy=True,
    )
    target = [1, 2, 2, 2, 1]

    assert degree_target_match_rate([relabeled], [target]) == 1.0


def test_degree_target_match_rate_detects_mismatch():
    graph = nx.path_graph(5)

    assert degree_target_match_rate([graph], [[2, 2, 2, 2, 0]]) == 0.0

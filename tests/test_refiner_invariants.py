from __future__ import annotations

import networkx as nx
import numpy as np

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

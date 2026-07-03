from __future__ import annotations

import networkx as nx
import numpy as np

from grapher.properties.summary import SummaryConfig, distance_to_summary, extract_summary
from grapher.refinement.rewiring import apply_action, enumerate_valid_double_edge_swaps, permute_action


def test_candidate_scores_are_equivariant():
    g = nx.connected_watts_strogatz_graph(12, 4, 0.3, seed=0)
    cfg = SummaryConfig(degree_hist_max_degree=6, clustering_bins=8, spectral_bins=8)
    target = extract_summary(g, cfg)
    mapping = {i: (i * 5) % 12 for i in range(12)}
    inv = {v: k for k, v in mapping.items()}
    gp = nx.relabel_nodes(g, mapping, copy=True)
    c1 = set(enumerate_valid_double_edge_swaps(g, preserve_connectivity=True))
    c2 = set(permute_action(a, inv) for a in enumerate_valid_double_edge_swaps(gp, preserve_connectivity=True))
    assert c1 == c2
    weights = {"clustering_weight": 1.0, "spectral_weight": 0.2, "motif_weight": 0.5}
    for action in list(c1)[:20]:
        ap = permute_action(action, mapping)
        delta = distance_to_summary(g, target, cfg, weights) - distance_to_summary(apply_action(g, action), target, cfg, weights)
        delta_p = distance_to_summary(gp, target, cfg, weights) - distance_to_summary(apply_action(gp, ap), target, cfg, weights)
        assert abs(delta - delta_p) < 1e-6

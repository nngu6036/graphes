from __future__ import annotations

import networkx as nx
import numpy as np

from grapher.properties.summary import SummaryConfig, extract_summary


def test_summary_is_permutation_invariant():
    g = nx.cycle_graph(8)
    g.add_edge(0, 4)
    mapping = {i: (i * 3) % 8 for i in range(8)}
    gp = nx.relabel_nodes(g, mapping, copy=True)
    cfg = SummaryConfig(degree_hist_max_degree=4, clustering_bins=10, spectral_bins=10)
    s1 = extract_summary(g, cfg)
    s2 = extract_summary(gp, cfg)
    for key in ["degree_hist", "clustering_hist", "spectral_hist", "motif_proxy"]:
        assert np.allclose(s1[key], s2[key])

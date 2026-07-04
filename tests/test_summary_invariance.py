from __future__ import annotations

import networkx as nx
import numpy as np

from grapher.properties.summary import SummaryConfig, extract_summary, python_orbit_count_vector


def test_summary_is_permutation_invariant():
    g = nx.cycle_graph(8)
    g.add_edge(0, 4)
    mapping = {i: (i * 3) % 8 for i in range(8)}
    gp = nx.relabel_nodes(g, mapping, copy=True)
    cfg = SummaryConfig(degree_hist_max_degree=4, clustering_bins=10, spectral_bins=10)
    s1 = extract_summary(g, cfg)
    s2 = extract_summary(gp, cfg)
    for key in ["degree_hist", "clustering_hist", "spectral_hist", "motif_proxy", "orbit_count"]:
        assert np.allclose(s1[key], s2[key])


def test_orbit_count_for_triangle():
    counts = python_orbit_count_vector(nx.complete_graph(3))
    expected = np.zeros(15, dtype=np.float64)
    expected[0] = 2.0
    expected[3] = 1.0
    assert np.allclose(counts, expected)


def test_orbit_count_for_path_four():
    counts = python_orbit_count_vector(nx.path_graph(4))
    expected = np.zeros(15, dtype=np.float64)
    expected[0] = 1.5
    expected[1] = 1.0
    expected[2] = 0.5
    expected[6] = 0.5
    expected[7] = 0.5
    assert np.allclose(counts, expected)

from __future__ import annotations

import networkx as nx
import numpy as np

from grapher.models.dhvae_hh.havel_hakimi import (
    assert_constructor_validity,
    construct_coarse_graph,
)
from grapher.rewiring_mlp.properties.summary import extract_summary


def test_constructor_preserves_degree_sequence():
    g = nx.barbell_graph(5, 2)
    summary = extract_summary(g)
    out = construct_coarse_graph(
        summary,
        {"ensure_connected": True, "random_relabel": False},
        np.random.default_rng(0),
    )
    assert_constructor_validity(out, summary, require_connected=True)

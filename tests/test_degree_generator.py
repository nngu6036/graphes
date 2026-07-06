from __future__ import annotations

import networkx as nx
import numpy as np
import torch

from grapher.generators.degree_vectorizer import DegreeVectorizer, connected_feasible_degree_sequence
from grapher.generators.degree_vae import build_degree_vae


def test_degree_vectorizer_outputs_graphical_summaries():
    graphs = [nx.cycle_graph(12), nx.path_graph(12), nx.watts_strogatz_graph(20, 4, 0.2, seed=1)]
    vectorizer = DegreeVectorizer.fit(graphs, require_connected=True)
    model = build_degree_vae(vectorizer, latent_dim=4, hidden_dim=16)
    torch.manual_seed(0)
    outputs = model.sample_outputs(3, device="cpu")
    summaries = vectorizer.outputs_to_summaries(outputs, rng=np.random.default_rng(0))
    assert len(summaries) == 3
    for summary in summaries:
        seq = summary["degree_sequence"]
        assert len(seq) == summary["num_nodes"]
        assert nx.is_graphical(seq, method="eg")
        assert connected_feasible_degree_sequence(seq)
        assert summary["num_edges"] == sum(seq) // 2

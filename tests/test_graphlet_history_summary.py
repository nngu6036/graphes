from __future__ import annotations

import networkx as nx
import numpy as np
import torch

from grapher.generators.summary_vae import build_summary_vae, summary_vae_loss
from grapher.generators.summary_vectorizer import SummaryVectorizer
from grapher.properties.summary import SummaryConfig, distance_to_summary, extract_summary
from grapher.utils.motifs import graphlet_history, graphlet_history_l2_distance


def test_graphlet_history_is_permutation_invariant():
    g = nx.cycle_graph(6)
    g.add_edge(0, 3)
    gp = nx.relabel_nodes(g, {i: (i * 5) % 6 for i in range(6)}, copy=True)
    h1 = graphlet_history(g, k_min=3, k_max=4)
    h2 = graphlet_history(gp, k_min=3, k_max=4)
    assert h1 == h2


def test_graphlet_history_vectorizer_and_vae_forward():
    graphs = [nx.path_graph(6), nx.cycle_graph(6), nx.complete_graph(5)]
    cfg = SummaryConfig(degree_hist_max_degree=5, graphlet_history=True, graphlet_k_min=3, graphlet_k_max=4)
    summaries = [extract_summary(g, cfg) for g in graphs]
    vectorizer = SummaryVectorizer.fit(summaries, cfg, require_connected=True)
    assert vectorizer.graphlet_dim > 0
    x_np, targets_np = vectorizer.to_training_arrays(summaries)
    model = build_summary_vae(vectorizer, latent_dim=4, hidden_dim=16)
    x = torch.tensor(x_np, dtype=torch.float32)
    outputs, mu, logvar = model(x)
    targets = {k: torch.tensor(v).long() if k == "num_nodes" else torch.tensor(v).float() for k, v in targets_np.items()}
    loss, metrics = summary_vae_loss(outputs, targets, mu, logvar)
    assert torch.isfinite(loss)
    assert metrics["graphlet_loss"] >= 0.0
    sampled = vectorizer.outputs_to_summaries(model.sample_outputs(2, device="cpu"), rng=np.random.default_rng(0))
    assert "graphlet_history" in sampled[0]


def test_graphlet_energy_detects_difference():
    cfg = SummaryConfig(graphlet_history=True, graphlet_k_min=3, graphlet_k_max=4)
    target = extract_summary(nx.cycle_graph(6), cfg)
    same = distance_to_summary(nx.cycle_graph(6), target, cfg, {"graphlet_weight": 1.0})
    other = distance_to_summary(nx.path_graph(6), target, cfg, {"graphlet_weight": 1.0})
    assert same <= other
    assert graphlet_history_l2_distance(target["graphlet_history"], target["graphlet_history"]) == 0.0

from __future__ import annotations

import networkx as nx
import numpy as np
import torch

from grapher.evaluation.degree_sequences import evaluate_degree_sequence_sets
from grapher.generators.degree_vae import DegreeVectorizer, build_degree_vae, connected_feasible_degree_sequence


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


def test_degree_sampling_diagnostics_distinguish_raw_and_accepted_quality():
    graphs = [nx.cycle_graph(4)]
    vectorizer = DegreeVectorizer.fit(graphs, require_connected=True)
    outputs = {
        "num_nodes_logits": torch.tensor([[10.0]]),
        "degree_logits": torch.tensor([[-10.0, -10.0, 10.0]]),
        "edge_scalar": torch.tensor([[1.0]]),
    }
    summaries = vectorizer.outputs_to_summaries(
        outputs,
        rng=np.random.default_rng(0),
        deterministic=True,
        sample_num_nodes="model",
        include_diagnostics=True,
    )
    diagnostics = summaries[0]["sampling_diagnostics"]
    assert diagnostics["raw_graphical"]
    assert diagnostics["raw_connected_feasible"]
    assert not diagnostics["repair_used"]
    assert not diagnostics["fallback_used"]


def test_degree_sequence_evaluation_is_zero_for_identical_sets():
    sequences = [[2, 2, 2, 2], [2, 1, 1]]
    metrics = evaluate_degree_sequence_sets(
        sequences,
        sequences,
        train=sequences,
    )
    assert np.isclose(metrics["degree_histogram_mmd"], 0.0)
    assert np.isclose(
        metrics["degree_marginal_kl_reference_to_candidate"], 0.0
    )
    assert np.isclose(metrics["node_count_total_variation"], 0.0)
    assert np.isclose(metrics["edge_count_total_variation"], 0.0)
    assert metrics["sequence_novelty_rate"] == 0.0

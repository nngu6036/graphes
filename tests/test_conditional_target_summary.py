import networkx as nx
import numpy as np
import torch

from grapher.generators.summary_vae import (
    ConditionalSummaryVAE,
    SummaryVectorizer,
    build_conditional_summary_vae,
)
from grapher.properties.summary import SummaryConfig, extract_summary


def test_conditional_target_summary_preserves_degree_condition():
    graphs = [nx.path_graph(6), nx.cycle_graph(6), nx.complete_graph(5)]
    cfg = SummaryConfig(
        degree_hist_max_degree=5,
        clustering_bins=8,
        spectral_bins=8,
    )
    summaries = [extract_summary(graph, cfg) for graph in graphs]
    vectorizer = SummaryVectorizer.fit(summaries, cfg)
    model = build_conditional_summary_vae(
        vectorizer,
        latent_dim=4,
        hidden_dim=16,
        num_layers=1,
    )
    assert isinstance(model, ConditionalSummaryVAE)

    condition_summary = summaries[1]
    condition = torch.as_tensor(
        vectorizer.to_condition_vector(condition_summary)[None, :],
        dtype=torch.float32,
    )
    with torch.no_grad():
        outputs = model.sample_outputs(condition, device="cpu")
    decoded = vectorizer.outputs_to_summaries(
        outputs,
        deterministic=True,
        condition_summaries=[condition_summary],
    )[0]

    assert decoded["num_nodes"] == condition_summary["num_nodes"]
    assert decoded["num_edges"] == condition_summary["num_edges"]
    assert decoded["degree_sequence"] == condition_summary["degree_sequence"]
    assert np.allclose(decoded["degree_hist"], condition_summary["degree_hist"])


def test_conditional_target_summary_forward_shapes():
    graph = nx.cycle_graph(6)
    cfg = SummaryConfig(
        degree_hist_max_degree=3,
        clustering_bins=6,
        spectral_bins=6,
    )
    summary = extract_summary(graph, cfg)
    vectorizer = SummaryVectorizer.fit([summary], cfg)
    model = build_conditional_summary_vae(
        vectorizer,
        latent_dim=3,
        hidden_dim=12,
        num_layers=1,
    )
    x = torch.as_tensor(
        vectorizer.to_feature_vector(summary)[None, :],
        dtype=torch.float32,
    )
    condition = torch.as_tensor(
        vectorizer.to_condition_vector(summary)[None, :],
        dtype=torch.float32,
    )
    outputs, mu, logvar = model(x, condition)
    assert outputs["clustering_logits"].shape == (1, 6)
    assert mu.shape == logvar.shape == (1, 3)

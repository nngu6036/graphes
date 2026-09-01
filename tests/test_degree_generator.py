from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
import torch

from grapher.rewiring_mlp.evaluation.degree_sequences import evaluate_degree_sequence_sets
from grapher.models.dhvae_hh.degree_vae import (
    DegreeVectorizer,
    build_degree_vae,
    connected_feasible_degree_sequence,
    degree_vae_loss,
    load_degree_vae_checkpoint,
    save_degree_vae_checkpoint,
)


def test_degree_vectorizer_outputs_graphical_summaries():
    graphs = [
        nx.cycle_graph(12),
        nx.path_graph(12),
        nx.watts_strogatz_graph(20, 4, 0.2, seed=1),
    ]
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
    assert diagnostics["accepted_without_postprocessing"]


def test_degree_decoder_is_explicitly_conditioned_on_graph_size():
    graphs = [nx.cycle_graph(8), nx.cycle_graph(16)]
    vectorizer = DegreeVectorizer.fit(graphs, require_connected=True)
    model = build_degree_vae(
        vectorizer,
        latent_dim=4,
        hidden_dim=16,
        size_condition_dim=8,
    )
    z = torch.zeros(2, 4)
    outputs = model.decode(z, torch.tensor([8, 16]))
    assert outputs["conditioned_num_nodes"].tolist() == [8, 16]
    # The same latent receives different continuous size embeddings.
    assert not torch.allclose(
        outputs["degree_logits"][0],
        outputs["degree_logits"][1],
    )


def test_training_targets_include_true_size_and_degree_moment():
    graph = nx.path_graph(6)
    vectorizer = DegreeVectorizer.fit([graph], require_connected=True)
    targets = vectorizer.to_targets(graph)
    assert int(targets["num_nodes_count"]) == 6
    assert np.isclose(float(targets["mean_degree"][0]), 10.0 / 6.0)
    assert "edge_scalar" not in targets


def test_size_conditioned_vae_loss_and_checkpoint_round_trip(tmp_path):
    graphs = [nx.path_graph(8), nx.cycle_graph(12)]
    vectorizer = DegreeVectorizer.fit(graphs, require_connected=True)
    x_np, targets_np = vectorizer.to_training_arrays(graphs)
    model = build_degree_vae(
        vectorizer,
        latent_dim=4,
        hidden_dim=16,
        size_condition_dim=8,
    )
    x = torch.as_tensor(x_np, dtype=torch.float32)
    targets = {
        key: torch.as_tensor(
            value,
            dtype=(
                torch.long if key in {"num_nodes", "num_nodes_count"} else torch.float32
            ),
        )
        for key, value in targets_np.items()
    }
    outputs, mu, logvar = model(x, targets["num_nodes_count"])
    loss, metrics = degree_vae_loss(
        outputs,
        targets,
        mu,
        logvar,
        weights={"degree_moment": 1.0},
    )
    loss.backward()
    assert np.isfinite(float(loss.detach()))
    assert np.isfinite(metrics["degree_moment_loss"])
    assert model.degree_head.weight.grad is not None
    assert model.prior_type == "conditional_gmm"
    assert model.prior_components == 4
    assert any(
        parameter.grad is not None for parameter in model.conditional_prior.parameters()
    )

    checkpoint = tmp_path / "degree_vae.pt"
    save_degree_vae_checkpoint(checkpoint, model, vectorizer)
    loaded, loaded_vectorizer, _ = load_degree_vae_checkpoint(checkpoint, device="cpu")
    sampled = loaded.sample_outputs(2, node_counts=[8, 12], device="cpu")
    assert sampled["conditioned_num_nodes"].tolist() == [8, 12]
    assert loaded_vectorizer.input_dim == vectorizer.input_dim
    assert loaded.prior_type == "conditional_gmm"
    assert loaded.prior_components == 4


def test_parity_conditioned_sampling_produces_even_raw_sequence():
    graphs = [nx.cycle_graph(4)]
    vectorizer = DegreeVectorizer.fit(graphs, require_connected=True)
    outputs = {
        "num_nodes_logits": torch.tensor([[10.0]]),
        "degree_logits": torch.tensor([[-10.0, 0.0, 0.0]]),
        "conditioned_num_nodes": torch.tensor([4]),
    }
    summaries = vectorizer.outputs_to_summaries(
        outputs,
        rng=np.random.default_rng(7),
        deterministic=False,
        max_resample=1,
        parity_conditioned=True,
        max_parity_resample=128,
        include_diagnostics=True,
    )
    diagnostic = summaries[0]["sampling_diagnostics"]
    assert diagnostic["raw_even_degree_sum"]
    assert diagnostic["parity_draws"] >= 1


def test_reject_only_policy_does_not_silently_repair_invalid_draw():
    vectorizer = DegreeVectorizer.fit(
        [nx.cycle_graph(4)],
        require_connected=True,
    )
    outputs = {
        "num_nodes_logits": torch.tensor([[10.0]]),
        # Four degree-one nodes are graphical but cannot form a connected graph.
        "degree_logits": torch.tensor([[-10.0, 10.0, -10.0]]),
        "conditioned_num_nodes": torch.tensor([4]),
    }
    with pytest.raises(RuntimeError, match="exhausted its samples"):
        vectorizer.outputs_to_summaries(
            outputs,
            rng=np.random.default_rng(0),
            deterministic=True,
            fallback="error",
            postprocess_policy="reject_only",
        )


def test_conditional_gmm_prior_shapes_and_standard_normal_override():
    vectorizer = DegreeVectorizer.fit(
        [nx.path_graph(8), nx.cycle_graph(12)],
        require_connected=True,
    )
    model = build_degree_vae(
        vectorizer,
        latent_dim=5,
        hidden_dim=16,
        prior_type="conditional_gmm",
        prior_components=3,
    )
    node_counts = torch.tensor([8, 12])
    params = model.prior_parameters(node_counts)
    assert params["prior_logits"].shape == (2, 3)
    assert params["prior_means"].shape == (2, 3, 5)
    assert params["prior_logvars"].shape == (2, 3, 5)
    learned_z = model.sample_prior(node_counts, prior_mode="model")
    standard_z = model.sample_prior(node_counts, prior_mode="standard_normal")
    assert learned_z.shape == standard_z.shape == (2, 5)


def test_degree_sequence_evaluation_is_zero_for_identical_sets():
    sequences = [[2, 2, 2, 2], [2, 1, 1]]
    metrics = evaluate_degree_sequence_sets(
        sequences,
        sequences,
        train=sequences,
    )
    assert np.isclose(metrics["degree_histogram_mmd"], 0.0)
    assert np.isclose(metrics["degree_marginal_kl_reference_to_candidate"], 0.0)
    assert np.isclose(metrics["node_count_total_variation"], 0.0)
    assert np.isclose(metrics["edge_count_total_variation"], 0.0)
    assert metrics["sequence_novelty_rate"] == 0.0


def test_edge_conditioned_dhvae_models_p_m_given_n_and_p_d_given_n_m():
    graphs = [
        nx.path_graph(8),          # m=7
        nx.cycle_graph(8),         # m=8
        nx.watts_strogatz_graph(8, 4, 0.0),  # m=16
    ]
    vectorizer = DegreeVectorizer.fit(graphs, require_connected=True)
    model = build_degree_vae(
        vectorizer,
        latent_dim=4,
        hidden_dim=24,
        size_condition_dim=8,
        edge_condition_dim=8,
        use_edge_count_conditioning=True,
        prior_condition_on_edges=True,
    )
    z = torch.zeros(2, 4)
    n = torch.tensor([8, 8])
    outputs = model.decode(z, n, torch.tensor([7, 16]))
    assert outputs["num_edges_logits"].shape[-1] == vectorizer.edge_count_classes
    assert outputs["conditioned_num_edges"].tolist() == [7, 16]
    assert not torch.allclose(outputs["degree_logits"][0], outputs["degree_logits"][1])
    # Connected simple 8-node graphs require 7 <= m <= 28.
    assert float(outputs["num_edges_logits"][0, 6].detach()) < -1.0e8


def test_exact_degree_sum_conditioning_preserves_sampled_edge_count_before_rejection():
    vectorizer = DegreeVectorizer.fit(
        [nx.path_graph(8), nx.cycle_graph(8)], require_connected=True
    )
    outputs = {
        "num_nodes_logits": torch.zeros(1, vectorizer.node_count_classes),
        "num_edges_logits": torch.zeros(1, vectorizer.edge_count_classes),
        "conditioned_num_nodes": torch.tensor([8]),
        "conditioned_num_edges": torch.tensor([8]),
        "degree_logits": torch.tensor(
            [[-20.0, 0.0, 2.0, 1.0] + [-20.0] * (vectorizer.degree_dim - 4)]
        ),
    }
    summaries = vectorizer.outputs_to_summaries(
        outputs,
        rng=np.random.default_rng(5),
        max_resample=100,
        fallback="error",
        postprocess_policy="reject_only",
        exact_degree_sum_conditioning=True,
        include_diagnostics=True,
    )
    summary = summaries[0]
    diagnostic = summary["sampling_diagnostics"]
    assert summary["num_edges"] == 8
    assert sum(diagnostic["first_raw_degree_sequence"]) == 16
    assert diagnostic["exact_degree_sum_conditioned"]
    assert diagnostic["raw_edge_count_matches_target"]
    assert diagnostic["raw_even_degree_sum"]
    assert not diagnostic["repair_used"]


def test_enhanced_prior_loss_reaches_edge_and_prior_parameters():
    graphs = [nx.path_graph(8), nx.cycle_graph(8), nx.watts_strogatz_graph(8, 4, 0.0)]
    vectorizer = DegreeVectorizer.fit(graphs, require_connected=True)
    x_np, targets_np = vectorizer.to_training_arrays(graphs)
    model = build_degree_vae(
        vectorizer,
        latent_dim=4,
        hidden_dim=24,
        size_condition_dim=8,
        edge_condition_dim=8,
        use_edge_count_conditioning=True,
        prior_condition_on_edges=True,
        prior_components=2,
    )
    x = torch.as_tensor(x_np, dtype=torch.float32)
    targets = {
        key: torch.as_tensor(
            value,
            dtype=(
                torch.long
                if key in {"num_nodes", "num_nodes_count", "num_edges_count"}
                else torch.float32
            ),
        )
        for key, value in targets_np.items()
    }
    outputs, mu, logvar = model(
        x, targets["num_nodes_count"], targets["num_edges_count"]
    )
    prior_z = model.sample_prior(
        targets["num_nodes_count"], edge_counts=targets["num_edges_count"]
    )
    prior_outputs = model.decode(
        prior_z, targets["num_nodes_count"], targets["num_edges_count"]
    )
    loss, metrics = degree_vae_loss(
        outputs,
        targets,
        mu,
        logvar,
        weights={
            "num_nodes": 1.0,
            "num_edges": 2.0,
            "degree": 5.0,
            "degree_moment": 0.25,
            "aggregate_prior_moment": 0.05,
            "prior_distribution": 1.0,
        },
        prior_outputs=prior_outputs,
        prior_distribution_sigma=0.2,
    )
    loss.backward()
    assert np.isfinite(metrics["num_edges_loss"])
    assert np.isfinite(metrics["prior_distribution_loss"])
    assert np.isfinite(metrics["aggregate_prior_moment_loss"])
    assert model.num_edges_head.weight.grad is not None
    assert model.degree_head.weight.grad is not None
    assert any(p.grad is not None for p in model.conditional_prior.parameters())


def test_edge_conditioned_checkpoint_round_trip(tmp_path):
    graphs = [nx.path_graph(8), nx.cycle_graph(8)]
    vectorizer = DegreeVectorizer.fit(graphs, require_connected=True)
    model = build_degree_vae(
        vectorizer,
        latent_dim=4,
        hidden_dim=16,
        use_edge_count_conditioning=True,
        prior_condition_on_edges=True,
    )
    checkpoint = tmp_path / "edge_conditioned.pt"
    save_degree_vae_checkpoint(checkpoint, model, vectorizer)
    loaded, restored, _ = load_degree_vae_checkpoint(checkpoint, device="cpu")
    assert loaded.architecture_version == 4
    assert loaded.use_edge_count_conditioning
    assert loaded.prior_condition_on_edges
    outputs = loaded.sample_outputs(4, node_counts=[8, 8, 8, 8], device="cpu")
    summaries = restored.outputs_to_summaries(
        outputs,
        rng=np.random.default_rng(9),
        max_resample=100,
        fallback="error",
        postprocess_policy="reject_only",
        exact_degree_sum_conditioning=True,
    )
    assert all(summary["num_edges"] * 2 == sum(summary["degree_sequence"]) for summary in summaries)

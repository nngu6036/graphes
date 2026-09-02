from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
import torch

from grapher.rewiring_mlp.core.rewiring import enumerate_valid_double_edge_swaps
from grapher.rewiring_mlp.generic.basis import TopologyGraphletBasis
from grapher.rewiring_mlp.generic.graphlet_diffusion import (
    extract_topology_graphlet_simplex,
    graphlet_simplex_to_clr,
)
from grapher.rewiring_mlp.generic.rewiring import propose_valid_topology_swaps
from grapher.rewiring_mlp.generic.spectral import (
    batched_laplacian_eigenvalues,
    laplacian_eigenvalues,
)
from grapher.rewiring_mlp.generic.spectral_data import (
    TopologySpectralDiffusionIterableDataset,
    TopologySpectralExample,
    collate_spectral_examples,
)
from grapher.rewiring_mlp.generic.spectral_graphlet_refiner import (
    DegreeConditionedSummaryPrediction,
    SpectralGraphletRefinerConfig,
    enrich_graph_with_degree_summary,
    predict_degree_conditioned_summary,
)
from grapher.rewiring_mlp.generic.spectral_model import (
    TopologySpectralGraphletTransformerPredictor,
    load_topology_spectral_graphlet_checkpoint,
    save_topology_spectral_graphlet_checkpoint,
)


def _graph() -> nx.Graph:
    graph = nx.Graph()
    graph.add_edges_from(
        [(0, 1), (0, 2), (0, 3), (1, 2), (2, 4), (3, 5), (4, 5)]
    )
    return graph


def _basis() -> TopologyGraphletBasis:
    return TopologyGraphletBasis.from_config(
        {
            "graphlet_history": True,
            "graphlet_k_min": 3,
            "graphlet_k_max": 4,
            "graphlet_connected_only": True,
        }
    )


def _model(basis: TopologyGraphletBasis) -> TopologySpectralGraphletTransformerPredictor:
    return TopologySpectralGraphletTransformerPredictor(
        graphlet_block_widths=basis.simplex_block_widths,
        graphlet_dim=24,
        graphlet_dropout=0.0,
        hidden_dim=16,
        edge_dim=8,
        graph_dim=16,
        num_layers=1,
        spectral_dim=16,
        spectral_layers=1,
        spectral_heads=4,
        spectral_ff_dim=32,
        dropout=0.0,
        degree_summary_enabled=True,
        degree_summary_dim=16,
        degree_summary_layers=1,
        degree_summary_dropout=0.0,
    )


def test_batched_candidate_spectra_match_scalar() -> None:
    source = _graph()
    actions, candidates, _ = propose_valid_topology_swaps(
        source,
        proposal_budget=-1,
        valid_candidate_budget=-1,
        preserve_connectivity=True,
        rng=np.random.default_rng(0),
    )
    graphs = [candidates[action] for action in actions]
    batch = batched_laplacian_eigenvalues(graphs, device="cpu", backend="numpy")
    assert len(batch) == len(graphs)
    for graph, spectrum in zip(graphs, batch):
        np.testing.assert_allclose(spectrum, laplacian_eigenvalues(graph), atol=1.0e-10)


def test_fast_proposal_exhaustive_matches_reference_validity() -> None:
    graph = _graph()
    reference = set(enumerate_valid_double_edge_swaps(graph, preserve_connectivity=True))
    proposed, _graphs, _diag = propose_valid_topology_swaps(
        graph,
        proposal_budget=-1,
        valid_candidate_budget=-1,
        preserve_connectivity=True,
        rng=np.random.default_rng(0),
    )
    assert set(proposed) == reference


def test_degree_summary_is_permutation_invariant_and_trace_preserving() -> None:
    graph = _graph()
    basis = _basis()
    model = _model(basis).eval()
    prediction = predict_degree_conditioned_summary(
        model,
        graph,
        graphlet_basis=basis,
        device="cpu",
        graphlet_logit_epsilon=1.0e-5,
    )
    mapping = {0: 3, 1: 5, 2: 1, 3: 4, 4: 0, 5: 2}
    relabeled = nx.relabel_nodes(graph, mapping, copy=True)
    prediction_relabel = predict_degree_conditioned_summary(
        model,
        relabeled,
        graphlet_basis=basis,
        device="cpu",
        graphlet_logit_epsilon=1.0e-5,
    )
    np.testing.assert_allclose(
        prediction.clean_spectrum,
        prediction_relabel.clean_spectrum,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        prediction.clean_graphlet_logits,
        prediction_relabel.clean_graphlet_logits,
        atol=1.0e-6,
    )
    assert prediction.clean_spectrum[0] == pytest.approx(0.0, abs=1.0e-7)
    assert float(prediction.clean_spectrum.sum()) == pytest.approx(
        2.0 * graph.number_of_edges(), rel=1.0e-5, abs=1.0e-5
    )


def test_joint_degree_summary_loss_backpropagates() -> None:
    graph = _graph()
    basis = _basis()
    prob, mask, _ = extract_topology_graphlet_simplex(graph, graphlet_basis=basis)
    logits = graphlet_simplex_to_clr(
        prob, graphlet_basis=basis, coordinate_mask=mask
    )
    spectrum = laplacian_eigenvalues(graph).astype(np.float32)
    batch = collate_spectral_examples(
        [
            TopologySpectralExample(
                current_graph=graph,
                time=0.5,
                current_spectrum=spectrum,
                source_spectrum=spectrum,
                clean_spectrum_target=spectrum,
                current_graphlet_probabilities=prob.astype(np.float32),
                source_graphlet_probabilities=prob.astype(np.float32),
                clean_graphlet_probabilities_target=prob.astype(np.float32),
                current_graphlet_logits=logits.astype(np.float32),
                source_graphlet_logits=logits.astype(np.float32),
                clean_graphlet_logits_target=logits.astype(np.float32),
                graphlet_coordinate_mask=mask,
            )
        ]
    )
    model = _model(basis)
    loss, metrics = model.loss(
        batch,
        loss_weights={
            "spectrum": 0.75,
            "moment2": 0.25,
            "low_frequency": 0.75,
            "low_frequency_k": 4,
            "graphlet_logit": 2.0,
            "graphlet_probability": 1.0,
            "degree_summary": 0.35,
        },
    )
    loss.backward()
    assert metrics["degree_summary_loss"] >= 0.0
    assert model.degree_spectral_head[-1].weight.grad is not None
    assert model.degree_graphlet_heads[0][-1].weight.grad is not None


def test_degree_summary_checkpoint_roundtrip(tmp_path) -> None:
    basis = _basis()
    model = _model(basis)
    path = tmp_path / "enriched.pt"
    save_topology_spectral_graphlet_checkpoint(
        model,
        path,
        graphlet_basis=basis,
    )
    loaded, loaded_basis, _summary, _checkpoint = (
        load_topology_spectral_graphlet_checkpoint(path, device="cpu")
    )
    assert loaded.degree_summary_enabled is True
    assert loaded.degree_summary_dim == model.degree_summary_dim
    assert loaded_basis.simplex_block_widths == basis.simplex_block_widths


def test_source_enrichment_preserves_degree_and_improves_fixed_summary() -> None:
    source = _graph()
    basis = _basis()
    actions, candidate_graphs, _ = propose_valid_topology_swaps(
        source,
        proposal_budget=-1,
        valid_candidate_budget=-1,
        preserve_connectivity=True,
        rng=np.random.default_rng(0),
    )
    target_graph = candidate_graphs[actions[0]]
    prob, mask, _ = extract_topology_graphlet_simplex(target_graph, graphlet_basis=basis)
    logits = graphlet_simplex_to_clr(prob, graphlet_basis=basis, coordinate_mask=mask)
    spectrum = laplacian_eigenvalues(target_graph)
    target = DegreeConditionedSummaryPrediction(
        clean_spectrum=spectrum,
        clean_graphlet_logits=logits,
        clean_graphlet_probabilities=prob,
        graphlet_coordinate_mask=mask,
        trace=float(spectrum.sum()),
        second_moment=float(np.square(spectrum).sum()),
    )
    cfg = SpectralGraphletRefinerConfig.from_dict(
        {
            "mode": "spectral_graphlet",
            "steps": 1,
            "proposal_budget": -1,
            "valid_candidate_budget": -1,
            "preserve_connectivity": True,
            "selection": "greedy",
            "accept_only_improving": True,
            "min_improvement": 1.0e-12,
            "min_relative_improvement": 0.0,
            "spectral_guidance": {
                "distance": "rmse",
                "normalization": "mean_degree",
                "low_frequency_weight": 1.0,
                "low_frequency_cutoff": 0,
                "expand_on_plateau": False,
            },
            "graphlet_guidance": {
                "distance": "clr_rmse",
                "logit_epsilon": 1.0e-5,
                "size_weights": {"3": 1.0, "4": 1.0},
            },
            "global_to_local": {
                "schedule": "linear",
                "spectral_initial": 1.0,
                "spectral_final": 1.0,
                "graphlet_initial": 1.0,
                "graphlet_final": 1.0,
            },
        }
    )
    enriched, trace = enrich_graph_with_degree_summary(
        source,
        target=target,
        graphlet_basis=basis,
        refiner_config=cfg,
        device="cpu",
        rng=np.random.default_rng(0),
        return_trace=True,
    )
    assert trace and trace[0]["accepted"]
    assert trace[0]["energy_improvement"] > 0.0
    assert [enriched.degree(i) for i in sorted(enriched)] == [
        source.degree(i) for i in sorted(source)
    ]
    assert nx.is_connected(enriched)


def test_streaming_diffusion_caches_fixed_endpoints() -> None:
    graph = _graph()
    basis = _basis()
    dataset = TopologySpectralDiffusionIterableDataset(
        [graph],
        diffusion_config={
            "storage": "streaming",
            "cache_endpoints": True,
            "samples_per_graph": 2,
            "paths_per_graph": 1,
            "spectral_sigma": 0.2,
            "graphlet_sigma": 0.35,
        },
        source_config={
            "ensure_connected_source": True,
            "random_relabel_source": True,
            "source_randomization_steps": 0,
        },
        spectral_config={"require_same_degree_sequence": True},
        graphlet_basis=basis,
        seed=42,
    )
    assert dataset._endpoint_cache is not None
    first_source = dataset._endpoint_cache[0].source_spectrum.copy()
    list(iter(dataset))
    dataset.set_epoch(1)
    list(iter(dataset))
    np.testing.assert_allclose(dataset._endpoint_cache[0].source_spectrum, first_source)
    assert dataset.last_diagnostics[-1]["endpoint_cache"] is True

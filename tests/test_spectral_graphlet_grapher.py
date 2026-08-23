from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
import torch

from grapher.rewiring_mlp.generic.basis import TopologyGraphletBasis
from grapher.rewiring_mlp.generic.graphlet_diffusion import (
    extract_topology_graphlet_simplex,
    graphlet_clr_to_simplex,
    graphlet_simplex_to_clr,
)
from grapher.rewiring_mlp.generic.rewiring import propose_valid_topology_swaps
from grapher.rewiring_mlp.generic.spectral import laplacian_eigenvalues, spectrum_moments
from grapher.rewiring_mlp.generic.spectral_data import (
    TopologySpectralExample,
    collate_spectral_examples,
)
from grapher.rewiring_mlp.generic.spectral_model import (
    TOPOLOGY_SPECTRAL_GRAPHLET_CHECKPOINT_FORMAT,
    TopologySpectralGraphletTransformerPredictor,
    load_topology_spectral_graphlet_checkpoint,
    save_topology_spectral_graphlet_checkpoint,
)
from grapher.rewiring_mlp.generic.spectral_graphlet_refiner import (
    SpectralGraphletPrediction,
    SpectralGraphletRefinerConfig,
    refine_graph_with_spectral_graphlet_predictions,
)


def _graph() -> nx.Graph:
    graph = nx.Graph()
    graph.add_edges_from(
        [
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 2),
            (2, 4),
            (3, 5),
            (4, 5),
        ]
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


def _example(graph: nx.Graph, basis: TopologyGraphletBasis) -> TopologySpectralExample:
    prob, mask, _ = extract_topology_graphlet_simplex(graph, graphlet_basis=basis)
    logits = graphlet_simplex_to_clr(
        prob,
        graphlet_basis=basis,
        epsilon=1.0e-5,
        coordinate_mask=mask,
    )
    return TopologySpectralExample(
        current_graph=graph,
        time=0.5,
        clean_spectrum_target=laplacian_eigenvalues(graph),
        current_graphlet_probabilities=prob.astype(np.float32),
        clean_graphlet_probabilities_target=prob.astype(np.float32),
        current_graphlet_logits=logits.astype(np.float32),
        clean_graphlet_logits_target=logits.astype(np.float32),
        graphlet_coordinate_mask=mask,
    )


def _model(basis: TopologyGraphletBasis) -> TopologySpectralGraphletTransformerPredictor:
    return TopologySpectralGraphletTransformerPredictor(
        graphlet_block_widths=basis.simplex_block_widths,
        graphlet_dim=24,
        graphlet_dropout=0.0,
        graphlet_logit_epsilon=1.0e-5,
        hidden_dim=16,
        edge_dim=8,
        graph_dim=16,
        num_layers=2,
        spectral_dim=16,
        spectral_layers=2,
        spectral_heads=4,
        spectral_ff_dim=32,
        dropout=0.0,
        min_gap=1.0e-6,
        input_normalization="mean_degree",
    )


def test_graphlet_simplex_includes_disconnected_bin_and_roundtrips_clr() -> None:
    graph = _graph()
    basis = _basis()
    prob, mask, _ = extract_topology_graphlet_simplex(graph, graphlet_basis=basis)
    assert prob.size == basis.simplex_width
    for start, stop in basis.simplex_slices:
        assert np.all(mask[start:stop])
        assert float(prob[start:stop].sum()) == pytest.approx(1.0, abs=1.0e-10)
    logits = graphlet_simplex_to_clr(
        prob,
        graphlet_basis=basis,
        epsilon=1.0e-8,
        coordinate_mask=mask,
    )
    recovered = graphlet_clr_to_simplex(
        logits,
        graphlet_basis=basis,
        coordinate_mask=mask,
    )
    np.testing.assert_allclose(recovered, prob, atol=2.0e-7, rtol=2.0e-7)


def test_joint_predictor_outputs_valid_graphlet_simplexes_and_backpropagates() -> None:
    basis = _basis()
    batch = collate_spectral_examples([_example(_graph(), basis)])
    model = _model(basis)
    outputs = model(batch)
    assert outputs["clean_graphlet_logits"].shape == (1, basis.simplex_width)
    prob = outputs["clean_graphlet_probabilities"]
    for start, stop in basis.simplex_slices:
        assert float(prob[0, start:stop].sum().detach()) == pytest.approx(1.0, abs=1.0e-6)
    loss, metrics = model.loss(
        batch,
        loss_weights={
            "spectrum": 1.0,
            "moment2": 0.1,
            "graphlet_logit": 1.0,
            "graphlet_probability": 0.25,
        },
    )
    loss.backward()
    assert metrics["graphlet_logit_rmse"] >= 0.0
    assert model.graphlet_heads[0][-1].weight.grad is not None
    assert model.gap_head[-1].weight.grad is not None


def test_joint_checkpoint_roundtrip(tmp_path) -> None:
    basis = _basis()
    model = _model(basis)
    path = tmp_path / "joint.pt"
    save_topology_spectral_graphlet_checkpoint(
        model,
        path,
        graphlet_basis=basis,
        report={
            "val_spectral_normalized_rmse": 0.1,
            "val_graphlet_logit_rmse": 0.2,
        },
    )
    loaded, loaded_basis, _summary, checkpoint = load_topology_spectral_graphlet_checkpoint(
        path,
        device="cpu",
    )
    assert checkpoint["format"] == TOPOLOGY_SPECTRAL_GRAPHLET_CHECKPOINT_FORMAT
    assert loaded_basis.simplex_block_widths == basis.simplex_block_widths
    assert isinstance(loaded, TopologySpectralGraphletTransformerPredictor)


def test_joint_refiner_uses_local_graphlet_delta_and_prints_debug(capsys) -> None:
    source = _graph()
    basis = _basis()
    candidates, candidate_graphs, _ = propose_valid_topology_swaps(
        source,
        proposal_budget=-1,
        valid_candidate_budget=-1,
        preserve_connectivity=True,
        rng=np.random.default_rng(4),
    )
    source_spectrum = laplacian_eigenvalues(source)
    target_action = next(
        action
        for action in candidates
        if not np.allclose(laplacian_eigenvalues(candidate_graphs[action]), source_spectrum)
    )
    target = candidate_graphs[target_action]
    target_spectrum = laplacian_eigenvalues(target)
    target_prob, target_mask, _ = extract_topology_graphlet_simplex(target, graphlet_basis=basis)
    target_logits = graphlet_simplex_to_clr(
        target_prob,
        graphlet_basis=basis,
        epsilon=1.0e-5,
        coordinate_mask=target_mask,
    )

    def fake_predictor(_model, graph, **kwargs):
        del kwargs
        current_prob, current_mask, _ = extract_topology_graphlet_simplex(
            graph,
            graphlet_basis=basis,
        )
        current_logits = graphlet_simplex_to_clr(
            current_prob,
            graphlet_basis=basis,
            epsilon=1.0e-5,
            coordinate_mask=current_mask,
        )
        first, second = spectrum_moments(target_spectrum)
        return SpectralGraphletPrediction(
            clean_spectrum=target_spectrum,
            current_spectrum=laplacian_eigenvalues(graph),
            clean_graphlet_logits=target_logits,
            clean_graphlet_probabilities=target_prob,
            current_graphlet_logits=current_logits,
            current_graphlet_probabilities=current_prob,
            graphlet_coordinate_mask=current_mask,
            trace=first,
            second_moment=second,
        )

    config = SpectralGraphletRefinerConfig.from_dict(
        {
            "mode": "spectral_graphlet",
            "steps": 1,
            "proposal_budget": -1,
            "valid_candidate_budget": -1,
            "preserve_connectivity": True,
            "reject_revisited_states": False,
            "selection": "greedy",
            "min_improvement": 0.0,
            "min_relative_improvement": 0.0,
            "spectral_guidance": {
                "distance": "rmse",
                "normalization": "mean_degree",
                "min_clean_mix": 1.0,
                "max_clean_mix": 1.0,
                "expand_on_plateau": False,
            },
            "graphlet_guidance": {
                "distance": "clr_rmse",
                "logit_epsilon": 1.0e-5,
                "min_clean_mix": 1.0,
                "max_clean_mix": 1.0,
                "size_weights": {"3": 1.0, "4": 1.0},
            },
            "global_to_local": {
                "spectral_initial": 1.0,
                "spectral_final": 1.0,
                "graphlet_initial": 1.0,
                "graphlet_final": 1.0,
            },
            "debug": {"enabled": True, "top_candidates": 2, "spectrum_values": 8},
        }
    )
    refined, trace = refine_graph_with_spectral_graphlet_predictions(
        source,
        model=_model(basis),
        graphlet_basis=basis,
        refiner_config=config,
        device="cpu",
        rng=np.random.default_rng(5),
        return_trace=True,
        prediction_fn=fake_predictor,
        debug_context="unit-test",
    )
    captured = capsys.readouterr().out
    assert "[GraphER/SpectralGraphlet]" in captured
    assert "graphlet_current=" in captured
    assert "graphlet_next_target=" in captured
    assert "graphlet_gain=" in captured
    assert "ACCEPT" in captured
    assert sorted(dict(refined.degree()).values()) == sorted(dict(source.degree()).values())
    assert nx.is_connected(refined)
    accepted = [row for row in trace if row.get("accepted")]
    assert len(accepted) == 1
    assert accepted[0]["energy_improvement"] > 0.0
    assert "graphlet_projection_residual" in accepted[0]

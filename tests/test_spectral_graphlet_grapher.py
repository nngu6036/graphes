from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
import torch

from grapher.rewiring_mlp.generic.basis import TopologyGraphletBasis
from grapher.rewiring_mlp.generic.graphlet_diffusion import (
    extract_topology_graphlet_simplex,
    graphlet_clr_to_simplex,
    graphlet_simplex_from_counts,
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
import grapher.rewiring_mlp.generic.spectral_graphlet_refiner as spectral_graphlet_refiner_module
from grapher.rewiring_mlp.generic.spectral_graphlet_refiner import (
    SpectralGraphletPrediction,
    SpectralGraphletRefinerConfig,
    predict_clean_spectrum_and_graphlets,
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



def test_prediction_refresh_reuses_cached_graphlet_counts(monkeypatch) -> None:
    graph = _graph()
    basis = _basis()
    source_prob, source_mask, source_counts = extract_topology_graphlet_simplex(
        graph, graphlet_basis=basis
    )
    source_logits = graphlet_simplex_to_clr(
        source_prob,
        graphlet_basis=basis,
        epsilon=1.0e-5,
        coordinate_mask=source_mask,
    )

    def fail_full_recount(*args, **kwargs):
        del args, kwargs
        raise AssertionError("prediction refresh unexpectedly performed a full graphlet recount")

    monkeypatch.setattr(
        spectral_graphlet_refiner_module,
        "extract_topology_graphlet_simplex",
        fail_full_recount,
    )
    prediction = predict_clean_spectrum_and_graphlets(
        _model(basis),
        graph,
        graphlet_basis=basis,
        time=0.5,
        device="cpu",
        graphlet_logit_epsilon=1.0e-5,
        conditioning_graph=graph,
        source_spectrum=laplacian_eigenvalues(graph),
        source_graphlet_probabilities=source_prob,
        source_graphlet_logits=source_logits,
        current_graphlet_counts=source_counts,
    )
    cached_prob, cached_mask = graphlet_simplex_from_counts(
        source_counts,
        num_nodes=graph.number_of_nodes(),
        graphlet_basis=basis,
    )
    np.testing.assert_allclose(prediction.current_graphlet_probabilities, cached_prob)
    np.testing.assert_array_equal(prediction.graphlet_coordinate_mask, cached_mask)


def test_refiner_performs_only_one_full_graphlet_count(monkeypatch) -> None:
    source = _graph()
    basis = _basis()
    candidates, candidate_graphs, _ = propose_valid_topology_swaps(
        source,
        proposal_budget=-1,
        valid_candidate_budget=-1,
        preserve_connectivity=True,
        rng=np.random.default_rng(14),
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

    original_extract = spectral_graphlet_refiner_module.extract_topology_graphlet_simplex
    full_recounts = 0

    def counted_extract(*args, **kwargs):
        nonlocal full_recounts
        full_recounts += 1
        return original_extract(*args, **kwargs)

    monkeypatch.setattr(
        spectral_graphlet_refiner_module,
        "extract_topology_graphlet_simplex",
        counted_extract,
    )

    def fake_predictor(_model, graph, **kwargs):
        counts = kwargs["current_graphlet_counts"]
        current_prob, current_mask = graphlet_simplex_from_counts(
            counts,
            num_nodes=graph.number_of_nodes(),
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
        }
    )
    refine_graph_with_spectral_graphlet_predictions(
        source,
        model=_model(basis),
        graphlet_basis=basis,
        refiner_config=config,
        device="cpu",
        rng=np.random.default_rng(15),
        prediction_fn=fake_predictor,
    )
    assert full_recounts == 1

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


def test_k5_local_delta_fast_lookup_matches_full_recount() -> None:
    from grapher.rewiring_mlp.generic.graphlets import (
        candidate_topology_graphlet_counts,
        extract_topology_graphlet_counts,
    )

    graph = nx.gnm_random_graph(12, 24, seed=123)
    while not nx.is_connected(graph):
        graph = nx.gnm_random_graph(12, 24, seed=int(np.random.default_rng().integers(1, 100000)))
    basis = TopologyGraphletBasis.from_config(
        {
            "graphlet_history": True,
            "graphlet_k_min": 3,
            "graphlet_k_max": 5,
            "graphlet_connected_only": True,
        }
    )
    counts = extract_topology_graphlet_counts(graph, graphlet_basis=basis)
    candidates, candidate_graphs, _ = propose_valid_topology_swaps(
        graph,
        proposal_budget=128,
        valid_candidate_budget=8,
        preserve_connectivity=True,
        rng=np.random.default_rng(124),
    )
    assert candidates
    for action in candidates:
        candidate = candidate_graphs[action]
        incremental = candidate_topology_graphlet_counts(
            graph,
            candidate,
            action,
            current_counts=counts,
            graphlet_basis=basis,
        )
        exact = extract_topology_graphlet_counts(candidate, graphlet_basis=basis)
        assert incremental == exact


def test_prepared_candidate_states_are_reused_across_target_rescoring(monkeypatch) -> None:
    basis = _basis()
    graph = _graph()
    prob, mask, counts = extract_topology_graphlet_simplex(graph, graphlet_basis=basis)
    logits = graphlet_simplex_to_clr(
        prob,
        graphlet_basis=basis,
        epsilon=1.0e-5,
        coordinate_mask=mask,
    )
    candidates, candidate_graphs, _ = propose_valid_topology_swaps(
        graph,
        proposal_budget=-1,
        valid_candidate_budget=-1,
        preserve_connectivity=True,
        rng=np.random.default_rng(125),
    )
    original = spectral_graphlet_refiner_module.candidate_graphlet_logits_from_counts
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        spectral_graphlet_refiner_module,
        "candidate_graphlet_logits_from_counts",
        counted,
    )
    cfg = SpectralGraphletRefinerConfig.from_dict(
        {
            "mode": "spectral_graphlet",
            "steps": 2,
            "proposal_budget": -1,
            "valid_candidate_budget": -1,
            "graphlet_guidance": {"size_weights": {"3": 1.0, "4": 1.0}},
        }
    )
    prepared = spectral_graphlet_refiner_module.prepare_spectral_graphlet_candidate_states(
        graph,
        candidates,
        candidate_graphs=candidate_graphs,
        current_graphlet_counts=counts,
        graphlet_basis=basis,
        config=cfg,
    )
    assert calls == len(candidates)
    spectrum = laplacian_eigenvalues(graph)
    for mix in (0.2, 0.4, 0.7, 1.0):
        spectral_graphlet_refiner_module.score_prepared_spectral_graphlet_candidates(
            prepared,
            graphlet_basis=basis,
            clean_spectrum=spectrum,
            next_spectrum_target=spectrum,
            clean_graphlet_logits=logits,
            next_graphlet_logits_target=logits * mix,
            graphlet_coordinate_mask=mask,
            spectral_weight=1.0,
            graphlet_weight=1.0,
            config=cfg,
        )
    assert calls == len(candidates)

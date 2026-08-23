from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
import torch

from grapher.rewiring_mlp.evaluation.studies import aggregate_pipeline_diagnostics
from grapher.rewiring_mlp.generic.rewiring import propose_valid_topology_swaps
from grapher.rewiring_mlp.generic.spectral import (
    laplacian_eigenvalues,
    spectrum_moments,
)
from grapher.rewiring_mlp.generic.spectral_data import (
    TopologySpectralExample,
    assert_same_degree_fibre,
    build_spectral_examples,
    collate_spectral_examples,
)
from grapher.rewiring_mlp.generic.spectral_model import (
    TOPOLOGY_SPECTRAL_CHECKPOINT_FORMAT,
    TopologySpectralTransformerPredictor,
    load_topology_spectral_checkpoint,
    save_topology_spectral_checkpoint,
)
from grapher.rewiring_mlp.generic.spectral_refiner import (
    SpectralPrediction,
    SpectralRefinerConfig,
    refine_graph_with_spectral_predictions,
)


def _small_model() -> TopologySpectralTransformerPredictor:
    return TopologySpectralTransformerPredictor(
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


def _nontrivial_graph() -> nx.Graph:
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


def test_spectral_transformer_variable_size_joint_output_constraints() -> None:
    graphs = [nx.path_graph(5), nx.cycle_graph(7)]
    examples = [
        TopologySpectralExample(
            current_graph=graph,
            time=0.4,
            clean_spectrum_target=laplacian_eigenvalues(graph),
        )
        for graph in graphs
    ]
    batch = collate_spectral_examples(examples)
    model = _small_model().eval()

    with torch.no_grad():
        output = model(batch)["clean_spectrum"]

    assert output.shape == (2, 7)
    for row, graph in zip(output, graphs):
        n = graph.number_of_nodes()
        valid = row[:n]
        assert float(valid[0]) == pytest.approx(0.0, abs=1.0e-8)
        assert torch.all(valid[1:] >= valid[:-1] - 1.0e-7)
        assert float(valid.sum()) == pytest.approx(
            2.0 * graph.number_of_edges(), rel=1.0e-5, abs=1.0e-5
        )
        assert torch.all(row[n:] == 0.0)


def test_spectral_transformer_is_permutation_invariant() -> None:
    graph = _nontrivial_graph()
    permutation = {0: 4, 1: 0, 2: 5, 3: 2, 4: 1, 5: 3}
    relabelled = nx.relabel_nodes(graph, permutation, copy=True)
    relabelled = nx.convert_node_labels_to_integers(relabelled, ordering="sorted")
    target = laplacian_eigenvalues(graph)
    batch = collate_spectral_examples(
        [
            TopologySpectralExample(graph, 0.25, target),
            TopologySpectralExample(relabelled, 0.25, target),
        ]
    )
    model = _small_model().eval()
    with torch.no_grad():
        outputs = model(batch)["clean_spectrum"]
    torch.testing.assert_close(outputs[0], outputs[1], rtol=1.0e-5, atol=1.0e-5)


def test_spectral_loss_backpropagates_through_transformer() -> None:
    graph = _nontrivial_graph()
    batch = collate_spectral_examples(
        [TopologySpectralExample(graph, 0.5, laplacian_eigenvalues(graph))]
    )
    model = _small_model()
    loss, metrics = model.loss(batch, loss_weights={"spectrum": 1.0, "moment2": 0.1})
    loss.backward()
    assert metrics["spectral_trace_mae"] < 1.0e-4
    assert model.gap_head[-1].weight.grad is not None
    transformer_grad = next(model.spectral_transformer.parameters()).grad
    assert transformer_grad is not None


def test_spectral_checkpoint_roundtrip(tmp_path) -> None:
    model = _small_model().eval()
    path = tmp_path / "spectral.pt"
    save_topology_spectral_checkpoint(
        model,
        path,
        report={"val_spectral_normalized_rmse": 0.123},
    )
    loaded, _summary, checkpoint = load_topology_spectral_checkpoint(path, device="cpu")
    assert checkpoint["format"] == TOPOLOGY_SPECTRAL_CHECKPOINT_FORMAT
    assert isinstance(loaded, TopologySpectralTransformerPredictor)
    assert checkpoint["report"]["val_spectral_normalized_rmse"] == pytest.approx(0.123)


def test_same_degree_fibre_guard() -> None:
    assert_same_degree_fibre(nx.path_graph(5), nx.path_graph(5))
    with pytest.raises(ValueError, match="same degree sequence"):
        assert_same_degree_fibre(nx.path_graph(5), nx.cycle_graph(5))


def test_spectral_teacher_states_are_real_degree_preserving_graphs() -> None:
    target = _nontrivial_graph()
    examples, report = build_spectral_examples(
        [target],
        trajectory_config={
            "steps": 3,
            "proposal_budget": -1,
            "valid_candidate_budget": -1,
            "states_per_graph": 3,
            "paths_per_graph": 1,
            "preserve_connectivity": True,
            "ensure_connected_source": True,
            "random_relabel_source": False,
            "source_randomization_steps": 0,
            "teacher_mode": "hard",
            "teacher_temperature": 0.2,
            "teacher_top_k": 0,
            "teacher_sample_actions": False,
        },
        spectral_config={"require_same_degree_sequence": True},
        seed=3,
    )
    target_degree = sorted(dict(target.degree()).values())
    assert examples
    for example in examples:
        assert sorted(dict(example.current_graph.degree()).values()) == target_degree
        assert nx.is_connected(example.current_graph)
        assert len(example.clean_spectrum_target) == target.number_of_nodes()
    assert report["mean_final_teacher_spectral_discrepancy"] <= report[
        "mean_initial_spectral_discrepancy"
    ] + 1.0e-12


def test_spectral_refiner_projects_to_valid_swap_and_prints_debug(capsys) -> None:
    source = _nontrivial_graph()
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
    target_spectrum = laplacian_eigenvalues(candidate_graphs[target_action])

    def fake_predictor(_model, graph, *, time, device):
        del time, device
        current = laplacian_eigenvalues(graph)
        first, second = spectrum_moments(target_spectrum)
        return SpectralPrediction(
            clean_spectrum=target_spectrum,
            current_spectrum=current,
            trace=first,
            second_moment=second,
        )

    config = SpectralRefinerConfig.from_dict(
        {
            "mode": "spectral",
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
            "debug": {
                "enabled": True,
                "print_every": 1,
                "top_candidates": 2,
                "spectrum_values": 8,
            },
        }
    )
    model = _small_model()
    refined, trace = refine_graph_with_spectral_predictions(
        source,
        model=model,
        refiner_config=config,
        device="cpu",
        rng=np.random.default_rng(5),
        return_trace=True,
        prediction_fn=fake_predictor,
        debug_context="unit-test",
    )
    captured = capsys.readouterr().out
    assert "[GraphER/Spectral]" in captured
    assert "lambda_current=" in captured
    assert "lambda_next_target=" in captured
    assert "clean_hat_moments" in captured
    assert "candidate_rank=" in captured
    assert "ACCEPT" in captured
    assert "lambda_after_accept=" in captured
    assert sorted(dict(refined.degree()).values()) == sorted(dict(source.degree()).values())
    assert nx.is_connected(refined)
    accepted = [row for row in trace if row.get("accepted")]
    assert len(accepted) == 1
    assert accepted[0]["spectral_gain"] > 0.0


def test_pipeline_diagnostics_accept_spectral_guidance() -> None:
    result = aggregate_pipeline_diagnostics(
        [
            {
                "pipeline_mode": "topology",
                "guidance_mode": "spectral",
                "spectral_error": 0.2,
                "invariant_feasible": 1.0,
                "constructor_success": 1.0,
                "accepted_swaps": 2,
                "runtime_seconds": 1.5,
                "fallback_used": 0.0,
                "candidate_proposals": 4,
                "candidate_passes": 3,
                "candidate_pass_rate": 0.75,
                "proposals_per_accepted_swap": 2.0,
                "stopped": 1.0,
                "stop_opportunities": 1,
                "stop_rate": 1.0,
                "generation_attempts": 1,
                "generation_successes": 1,
                "end_to_end_yield": 1.0,
                "rejection_reasons": {},
            }
        ],
        require_complete=True,
        allow_fallback=False,
    )
    assert result["pipeline_mode"] == "topology"
    assert result["metrics"]["spectral_error"]["mean"] == pytest.approx(0.2)

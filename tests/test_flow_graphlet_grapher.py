from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import torch

from grapher.rewiring_mlp.generic.basis import TopologyGraphletBasis
from grapher.rewiring_mlp.generic.flow_data import (
    build_flow_graphlet_examples,
    collate_flow_graphlet_examples,
)
from grapher.rewiring_mlp.generic.flow_graphlet_refiner import (
    refine_graph_with_flow_graphlet_predictions,
)
from grapher.rewiring_mlp.generic.flow_model import (
    TOPOLOGY_FLOW_GRAPHLET_CHECKPOINT_FORMAT,
    TopologyFlowGraphletPredictor,
    load_topology_flow_graphlet_checkpoint,
    save_topology_flow_graphlet_checkpoint,
)


def _basis() -> TopologyGraphletBasis:
    return TopologyGraphletBasis.from_config(
        {
            "graphlet_k_min": 3,
            "graphlet_k_max": 4,
            "graphlet_connected_only": True,
        }
    )


def _model(basis: TopologyGraphletBasis) -> TopologyFlowGraphletPredictor:
    return TopologyFlowGraphletPredictor(
        graphlet_block_widths=basis.simplex_block_widths,
        hidden_dim=16,
        edge_dim=8,
        graph_dim=16,
        num_layers=1,
        graphlet_dim=16,
        graphlet_context_dim=8,
        pair_dim=16,
        dropout=0.0,
        project_degree_tangent=True,
    )


def test_linear_flow_path_preserves_indexed_degrees_without_rewiring() -> None:
    target = nx.cycle_graph(6)
    basis = _basis()
    examples, report = build_flow_graphlet_examples(
        [target],
        flow_config={
            "storage": "eager",
            "path": "linear",
            "samples_per_graph": 3,
            "paths_per_graph": 1,
            "time_sampling": "grid",
            "require_same_degree_sequence": True,
            "align_nodes_by_degree": True,
            "randomize_equal_degree_alignment": True,
            "joint_random_relabel": True,
        },
        source_config={
            "ensure_connected_source": True,
            "random_relabel_source": False,
            "max_repair_trials": 1000,
            "source_randomization_steps": 0,
        },
        graphlet_basis=basis,
        seed=7,
    )

    assert report["training_state_source"] == "continuous_edge_probability_flow_matching"
    assert report["rewiring_used_for_training_states"] is False
    assert report["indexed_degree_alignment"] is True
    assert report["max_target_degree_tangent_residual"] < 1.0e-10
    assert len(examples) == 3

    for example in examples:
        source_degree = np.asarray(
            [example.source_graph.degree(node) for node in range(6)], dtype=np.float64
        )
        assert np.allclose(example.flow_target.sum(axis=1), 0.0, atol=1.0e-7)
        assert np.allclose(
            example.current_edge_probabilities.sum(axis=1),
            source_degree,
            atol=1.0e-7,
        )
        assert np.allclose(
            example.current_edge_probabilities,
            example.current_edge_probabilities.T,
        )


def test_joint_model_outputs_symmetric_degree_tangent_flow_and_backpropagates() -> None:
    basis = _basis()
    examples, _ = build_flow_graphlet_examples(
        [nx.cycle_graph(6)],
        flow_config={
            "storage": "eager",
            "samples_per_graph": 2,
            "paths_per_graph": 1,
            "time_sampling": "grid",
            "require_same_degree_sequence": True,
            "align_nodes_by_degree": True,
        },
        source_config={
            "ensure_connected_source": True,
            "random_relabel_source": False,
            "max_repair_trials": 1000,
            "source_randomization_steps": 0,
        },
        graphlet_basis=basis,
        seed=3,
    )
    batch = collate_flow_graphlet_examples(examples)
    model = _model(basis)
    outputs = model(batch)
    velocity = outputs["flow_velocity"]

    assert velocity.shape == batch.flow_target.shape
    assert torch.allclose(velocity, velocity.transpose(1, 2), atol=1.0e-6)
    assert torch.max(torch.abs(velocity.sum(dim=2))).item() < 1.0e-5
    assert outputs["clean_graphlet_logits"].shape[1] == basis.simplex_width

    loss, metrics = model.loss(batch)
    assert torch.isfinite(loss)
    assert metrics["flow_degree_tangent_mae"] < 1.0e-5
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_flow_graphlet_checkpoint_round_trip(tmp_path: Path) -> None:
    basis = _basis()
    model = _model(basis)
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_topology_flow_graphlet_checkpoint(
        model,
        checkpoint_path,
        graphlet_basis=basis,
        config={"topology_predictor": {"type": "flow_graphlet"}},
        report={"val_flow_rmse": 0.2, "val_graphlet_logit_rmse": 0.3},
    )
    loaded, loaded_basis, _summary, checkpoint = load_topology_flow_graphlet_checkpoint(
        checkpoint_path,
        device="cpu",
    )
    assert checkpoint["format"] == TOPOLOGY_FLOW_GRAPHLET_CHECKPOINT_FORMAT
    assert checkpoint["guidance_mode"] == "flow_graphlet"
    assert loaded_basis.simplex_block_widths == basis.simplex_block_widths
    assert isinstance(loaded, TopologyFlowGraphletPredictor)


def test_flow_projection_generation_preserves_degree_sequence() -> None:
    basis = _basis()
    model = _model(basis)
    graph = nx.cycle_graph(8)
    refined, _trace = refine_graph_with_flow_graphlet_predictions(
        graph,
        model=model,
        graphlet_basis=basis,
        refiner_config={
            "mode": "flow_graphlet",
            "steps": 2,
            "proposal_budget": 32,
            "valid_candidate_budget": 8,
            "preserve_connectivity": True,
            "min_improvement": 0.0,
            "min_relative_improvement": 0.0,
            "prediction_horizon": {"mode": "fixed", "k": 1},
            "flow_guidance": {"normalize_per_swap": True},
            "graphlet_guidance": {
                "distance": "clr_rmse",
                "logit_epsilon": 1.0e-4,
                "size_weights": {"3": 1.0, "4": 1.0},
            },
            "global_to_local": {
                "flow_initial": 1.0,
                "flow_final": 1.0,
                "graphlet_initial": 1.0,
                "graphlet_final": 1.0,
            },
        },
        device="cpu",
        rng=np.random.default_rng(0),
        return_trace=True,
    )
    assert refined.number_of_edges() == graph.number_of_edges()
    assert [refined.degree(node) for node in range(8)] == [
        graph.degree(node) for node in range(8)
    ]
    assert nx.is_connected(refined)

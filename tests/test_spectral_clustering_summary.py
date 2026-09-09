from __future__ import annotations

import networkx as nx
import pytest
import torch

from grapher.rewiring_mlp.generic.spectral_data import (
    TopologySpectralExample,
    build_spectral_diffusion_examples,
    collate_spectral_examples,
)
from grapher.rewiring_mlp.generic.spectral_model import (
    TopologySpectralTransformerPredictor,
    load_topology_spectral_checkpoint,
    save_topology_spectral_checkpoint,
)
from grapher.rewiring_mlp.generic.spectral import laplacian_eigenvalues


def _model() -> TopologySpectralTransformerPredictor:
    return TopologySpectralTransformerPredictor(
        hidden_dim=8,
        edge_dim=8,
        graph_dim=8,
        num_layers=1,
        spectral_dim=16,
        spectral_layers=1,
        spectral_heads=4,
        spectral_ff_dim=32,
        dropout=0.0,
        use_graph_context=False,
        predict_clustering_coefficient=True,
    )


def test_diffusion_examples_carry_clean_average_clustering_target() -> None:
    graph = nx.complete_graph(5)
    examples, report = build_spectral_diffusion_examples(
        [graph],
        diffusion_config={
            "samples_per_graph": 2,
            "paths_per_graph": 1,
            "spectral_sigma": 0.1,
            "preserve_spectral_trace": True,
            "fix_spectral_lambda1": True,
        },
        source_config={
            "ensure_connected_source": True,
            "random_relabel_source": False,
            "max_repair_trials": 100,
            "source_randomization_steps": 0,
        },
        spectral_config={"require_same_degree_sequence": True},
        seed=3,
    )
    assert report["rewiring_used_for_training_states"] is False
    assert examples
    assert all(
        example.clean_clustering_coefficient_target == pytest.approx(1.0)
        for example in examples
    )


def test_spectral_model_predicts_and_learns_clustering_scalar() -> None:
    graph = nx.cycle_graph(6)
    spectrum = laplacian_eigenvalues(graph)
    batch = collate_spectral_examples(
        [
            TopologySpectralExample(
                current_graph=graph,
                time=0.5,
                current_spectrum=spectrum,
                source_spectrum=spectrum,
                clean_spectrum_target=spectrum,
                clean_clustering_coefficient_target=0.0,
            )
        ]
    )
    model = _model()
    outputs = model(batch)
    predicted = outputs["clean_clustering_coefficient"]
    assert predicted.shape == (1,)
    assert torch.all((predicted >= 0.0) & (predicted <= 1.0))

    loss, metrics = model.loss(
        batch,
        loss_weights={
            "spectrum": 1.0,
            "moment2": 0.0,
            "low_frequency": 0.0,
            "clustering_coefficient": 1.0,
        },
    )
    loss.backward()
    assert "clustering_coefficient_loss" in metrics
    assert "clustering_coefficient_mae" in metrics
    assert model.clustering_coefficient_head[-1].weight.grad is not None


def test_clustering_head_checkpoint_roundtrip(tmp_path) -> None:
    model = _model().eval()
    path = tmp_path / "spectral_clustering.pt"
    save_topology_spectral_checkpoint(model, path)
    loaded, _summary, checkpoint = load_topology_spectral_checkpoint(path, device="cpu")
    assert checkpoint["model_config"]["predict_clustering_coefficient"] is True
    assert loaded.predict_clustering_coefficient is True
    assert loaded.clustering_coefficient_head is not None

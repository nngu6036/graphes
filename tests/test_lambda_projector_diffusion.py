from pathlib import Path

import networkx as nx
import numpy as np
import torch
import yaml

from grapher.rewiring_mlp.generic.eigenspace import (
    eigenspace_projector_distance,
    laplacian_eigenspace_projector,
)
from grapher.rewiring_mlp.generic.joint_edge_spectral_generation import (
    JointEdgeSpectralRefinerConfig,
    refine_graph,
    sample_soft_endpoint,
)
from grapher.rewiring_mlp.generic.spectral_data import (
    build_spectral_diffusion_examples,
    collate_spectral_examples,
)
from grapher.rewiring_mlp.generic.spectral_model import TopologySpectralTransformerPredictor
from grapher.rewiring_mlp.generic.summary_diffusion import (
    SummaryDiffusionConfig,
    sample_eigenspace_projector_bridge_marginal,
)


def _small_model(*, predict_edge_state: bool = False):
    return TopologySpectralTransformerPredictor(
        hidden_dim=16,
        edge_dim=8,
        graph_dim=16,
        num_layers=1,
        spectral_dim=32,
        spectral_layers=1,
        spectral_heads=4,
        spectral_ff_dim=64,
        predict_edge_state=predict_edge_state,
        spectral_representation="lambda_projector",
        projector_rank=3,
    )


def test_projector_is_valid_and_permutation_equivariant():
    graph = nx.path_graph(9)
    projector = laplacian_eigenspace_projector(graph, rank=3)
    assert np.allclose(projector, projector.T, atol=1e-10)
    assert np.allclose(projector @ projector, projector, atol=1e-9)
    assert np.isclose(np.trace(projector), 3.0, atol=1e-9)
    assert np.allclose(projector @ np.ones(len(graph)), 0.0, atol=1e-9)

    permutation = [3, 0, 8, 1, 6, 4, 7, 2, 5]
    mapping = {old: new for new, old in enumerate(permutation)}
    permuted = nx.relabel_nodes(graph, mapping, copy=True)
    P = np.zeros((len(graph), len(graph)))
    for old, new in mapping.items():
        P[new, old] = 1.0
    projected = laplacian_eigenspace_projector(permuted, rank=3)
    assert np.allclose(projected, P @ projector @ P.T, atol=1e-8)


def test_projector_bridge_has_exact_endpoints_and_constant_nullspace():
    source = laplacian_eigenspace_projector(nx.path_graph(7), rank=3)
    clean = laplacian_eigenspace_projector(nx.cycle_graph(7), rank=3)
    schedule = SummaryDiffusionConfig.from_dict({
        "bridge": "brownian",
        "schedule": "linear",
        "projector_sigma": 0.2,
    })
    rng = np.random.default_rng(4)
    at_source, _ = sample_eigenspace_projector_bridge_marginal(
        source, clean, progress=0.0, sigma=schedule.projector_sigma,
        schedule=schedule, rng=rng,
    )
    at_clean, _ = sample_eigenspace_projector_bridge_marginal(
        source, clean, progress=1.0, sigma=schedule.projector_sigma,
        schedule=schedule, rng=rng,
    )
    middle, _ = sample_eigenspace_projector_bridge_marginal(
        source, clean, progress=0.5, sigma=schedule.projector_sigma,
        schedule=schedule, rng=rng,
    )
    assert np.allclose(at_source, source, atol=1e-12)
    assert np.allclose(at_clean, clean, atol=1e-12)
    assert np.allclose(middle, middle.T, atol=1e-12)
    assert np.allclose(middle @ np.ones(7), 0.0, atol=1e-10)


def test_lambda_projector_training_batch_loss_and_backward_are_finite():
    graph = nx.barbell_graph(3, 1)
    examples, report = build_spectral_diffusion_examples(
        [graph],
        diffusion_config={
            "bridge": "brownian",
            "schedule": "linear",
            "samples_per_graph": 2,
            "paths_per_graph": 1,
            "spectral_sigma": 0.05,
            "projector_sigma": 0.05,
        },
        source_config={
            "ensure_connected_source": True,
            "random_relabel_source": False,
            "source_randomization_steps": 0,
        },
        spectral_config={
            "representation": "lambda_projector",
            "projector_rank": 3,
            "projector_distance": "chordal",
            "normalization": "mean_degree",
            "lambda_weight": 1.0,
            "projector_weight": 1.0,
        },
        edge_diffusion_config={"enabled": False},
        seed=7,
    )
    assert report["spectral_representation"] == "lambda_projector"
    batch = collate_spectral_examples(examples)
    assert batch.current_projector.shape == (2, len(graph), len(graph))
    model = _small_model()
    outputs = model(batch)
    assert outputs["clean_spectrum"].shape == (2, len(graph))
    assert outputs["clean_projector"].shape == (2, len(graph), len(graph))
    loss, metrics = model.loss(
        batch,
        loss_weights={
            "spectrum": 1.0,
            "projector": 1.0,
            "moment2": 0.1,
            "low_frequency": 0.0,
        },
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert metrics["projector_loss"] >= 0.0
    assert metrics["projector_chordal"] >= 0.0
    assert metrics["projector_idempotence_rmse"] < 1e-4
    assert metrics["projector_nullspace_rmse"] < 1e-4
    assert metrics["projector_trace_mae"] < 1e-4


def test_lambda_projector_generation_and_refinement_preserve_degree():
    graph = nx.barbell_graph(3, 1)
    model = _small_model()
    config = {
        "edge_diffusion": {"enabled": False, "sampling_steps": 3, "sigma": 0.2, "smoothing": 0.01},
        "summary_diffusion": {"spectral_sigma": 0.05, "projector_sigma": 0.05},
        "spectral_prediction": {
            "representation": "lambda_projector",
            "projector_rank": 3,
            "projector_distance": "chordal",
            "lambda_weight": 1.0,
            "projector_weight": 1.0,
            "normalization": "mean_degree",
        },
        "topology_refiner": {
            "mode": "joint_edge_spectral",
            "steps": 2,
            "proposal_budget": 64,
            "valid_candidate_budget": 16,
            "preserve_connectivity": True,
            "distance": "rmse",
            "weights": {"edge": 0.0, "spectral": 1.0, "clustering": 0.0, "orbit": 0.0, "graphlet": 0.0},
        },
    }
    refiner_cfg = JointEdgeSpectralRefinerConfig.from_dict(config["topology_refiner"], model=model)
    assert refiner_cfg.guidance_mode == "spectral"
    targets, report = sample_soft_endpoint(model, graph, config, seed=11)
    assert report["spectral_representation"] == "lambda_projector"
    assert report["joint_laplacian_lambda_projector_diffusion"] is True
    assert targets["spectrum"].shape == (len(graph),)
    assert targets["projector"].shape == (len(graph), len(graph))
    assert eigenspace_projector_distance(
        targets["projector"], targets["projector"], rank=3
    ) < 1e-12
    refined, trace = refine_graph(
        graph, targets, model, config, rng=np.random.default_rng(11),
        prediction_calls=report["prediction_calls"],
    )
    assert dict(refined.degree()) == dict(graph.degree())
    assert nx.is_connected(refined)
    assert isinstance(trace, list)


def test_lambda_projector_configs_select_structured_spectral_space():
    root = Path(__file__).resolve().parents[1]
    for filename in (
        "community_small_lambda_projector_only_learned.yaml",
        "community_small_joint_edge_lambda_projector_graphlets345_learned.yaml",
    ):
        cfg = yaml.safe_load((root / "configs/experiments/grapher" / filename).read_text())
        assert cfg["spectral_prediction"]["representation"] == "lambda_projector"
        assert cfg["spectral_prediction"]["projector_rank"] == 4
        assert cfg["topology_predictor"]["loss_weights"]["spectrum"] == 1.0
        assert cfg["topology_predictor"]["loss_weights"]["projector"] == 1.0

from pathlib import Path

import networkx as nx
import numpy as np
import torch
import yaml

from grapher.rewiring_mlp.generic.heat_kernel import (
    heat_kernel_distance,
    heat_kernel_stack,
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
    sample_heat_kernel_bridge_marginal,
)


def _small_heat_model(*, predict_edge_state: bool = False):
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
        spectral_representation="heat_kernel",
        heat_kernel_times=(0.25, 1.0, 4.0),
    )


def test_heat_kernel_is_symmetric_stochastic_and_permutation_equivariant():
    graph = nx.path_graph(6)
    times = (0.25, 1.0, 4.0)
    heat = heat_kernel_stack(graph, times=times, normalization="mean_degree")
    assert heat.shape == (6, 6, 3)
    assert np.allclose(heat, heat.transpose(1, 0, 2), atol=1e-10)
    assert np.allclose(heat.sum(axis=1), 1.0, atol=1e-9)

    permutation = [3, 0, 5, 1, 4, 2]
    mapping = {old: new for new, old in enumerate(permutation)}
    permuted = nx.relabel_nodes(graph, mapping, copy=True)
    permuted = nx.Graph(permuted)
    order = list(range(6))
    # H(P L P^T) = P H(L) P^T. Build the explicit permutation matrix in the
    # same old->new convention as networkx relabeling.
    P = np.zeros((6, 6))
    for old, new in mapping.items():
        P[new, old] = 1.0
    heat_permuted = heat_kernel_stack(permuted, times=times, normalization="mean_degree")
    expected = np.stack([P @ heat[..., i] @ P.T for i in range(3)], axis=-1)
    assert np.allclose(heat_permuted, expected, atol=1e-9)


def test_heat_kernel_bridge_has_exact_endpoints():
    source_graph = nx.path_graph(6)
    clean_graph = nx.cycle_graph(6)
    source = heat_kernel_stack(source_graph, times=(0.5, 2.0))
    clean = heat_kernel_stack(clean_graph, times=(0.5, 2.0))
    schedule = SummaryDiffusionConfig.from_dict({
        "bridge": "brownian",
        "schedule": "linear",
        "heat_kernel_sigma": 0.2,
    })
    rng = np.random.default_rng(3)
    at_source, _ = sample_heat_kernel_bridge_marginal(
        source, clean, progress=0.0, sigma=schedule.heat_kernel_sigma,
        schedule=schedule, rng=rng,
    )
    at_clean, _ = sample_heat_kernel_bridge_marginal(
        source, clean, progress=1.0, sigma=schedule.heat_kernel_sigma,
        schedule=schedule, rng=rng,
    )
    assert np.array_equal(at_source, source)
    assert np.array_equal(at_clean, clean)


def test_heat_kernel_training_batch_and_loss_are_consistent():
    graph = nx.cycle_graph(6)
    examples, report = build_spectral_diffusion_examples(
        [graph],
        diffusion_config={
            "bridge": "brownian",
            "schedule": "linear",
            "samples_per_graph": 2,
            "paths_per_graph": 1,
            "heat_kernel_sigma": 0.05,
        },
        source_config={
            "ensure_connected_source": True,
            "random_relabel_source": False,
            "source_randomization_steps": 0,
        },
        spectral_config={
            "representation": "heat_kernel",
            "heat_kernel_times": [0.25, 1.0, 4.0],
            "normalization": "mean_degree",
            "heat_kernel_normalization": "mean_degree",
        },
        edge_diffusion_config={"enabled": False},
        seed=7,
    )
    assert report["spectral_representation"] == "heat_kernel"
    batch = collate_spectral_examples(examples)
    assert batch.current_heat_kernel.shape == (2, 6, 6, 3)
    model = _small_heat_model(predict_edge_state=False)
    outputs = model(batch)
    assert outputs["clean_heat_kernel"].shape == (2, 6, 6, 3)
    loss, metrics = model.loss(
        batch,
        loss_weights={
            "heat_kernel": 1.0,
            "spectrum": 0.0,
            "moment2": 0.0,
            "low_frequency": 0.0,
        },
    )
    assert torch.isfinite(loss)
    assert metrics["heat_kernel_loss"] >= 0.0
    assert metrics["heat_kernel_symmetry_max_abs"] < 1e-6
    assert metrics["heat_kernel_row_sum_mae"] < 1e-2


def test_heat_kernel_generation_works_without_edge_diffusion_head():
    graph = nx.cycle_graph(6)
    model = _small_heat_model(predict_edge_state=False)
    config = {
        "edge_diffusion": {"enabled": False, "sampling_steps": 3, "sigma": 0.2, "smoothing": 0.01},
        "summary_diffusion": {"heat_kernel_sigma": 0.05, "spectral_sigma": 0.2},
        "spectral_prediction": {
            "representation": "heat_kernel",
            "normalization": "mean_degree",
            "heat_kernel_normalization": "mean_degree",
        },
        "topology_refiner": {
            "mode": "joint_edge_spectral",
            "steps": 2,
            "proposal_budget": 64,
            "valid_candidate_budget": 16,
            "preserve_connectivity": True,
            "distance": "rmse",
            "weights": {
                "edge": 0.0,
                "spectral": 1.0,
                "clustering": 0.0,
                "orbit": 0.0,
                "graphlet": 0.0,
            },
        },
    }
    refiner_cfg = JointEdgeSpectralRefinerConfig.from_dict(config["topology_refiner"], model=model)
    assert refiner_cfg.guidance_mode == "spectral"

    targets, report = sample_soft_endpoint(model, graph, config, seed=11)
    assert report["spectral_representation"] == "heat_kernel"
    assert report["joint_laplacian_heat_kernel_diffusion"] is True
    assert targets["heat_kernel"].shape == (6, 6, 3)
    refined, trace = refine_graph(
        graph, targets, model, config, rng=np.random.default_rng(11),
        prediction_calls=report["prediction_calls"],
    )
    assert sorted(dict(refined.degree()).values()) == sorted(dict(graph.degree()).values())
    assert nx.is_connected(refined)
    assert isinstance(trace, list)


def test_heat_kernel_experiment_configs_select_heat_space():
    root = Path(__file__).resolve().parents[1]
    for filename in (
        "community_small_joint_edge_heat_kernel_graphlets345_learned.yaml",
        "community_small_heat_kernel_only_learned.yaml",
    ):
        cfg = yaml.safe_load((root / "configs/experiments/grapher" / filename).read_text())
        assert cfg["spectral_prediction"]["representation"] == "heat_kernel"
        assert cfg["topology_predictor"]["loss_weights"]["heat_kernel"] == 1.0
        assert cfg["topology_predictor"]["loss_weights"]["spectrum"] == 0.0

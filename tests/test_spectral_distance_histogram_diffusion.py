from pathlib import Path

import networkx as nx
import numpy as np
import torch
import yaml

from grapher.rewiring_mlp.generic.joint_edge_spectral_generation import (
    JointEdgeSpectralRefinerConfig,
    refine_graph,
    sample_soft_endpoint,
)
from grapher.rewiring_mlp.generic.spectral_data import (
    build_spectral_diffusion_examples,
    collate_spectral_examples,
)
from grapher.rewiring_mlp.generic.spectral_distance_histogram import (
    SpectralDistanceHistogramSpec,
    degree_pair_weights,
    extract_degree_conditioned_spectral_histogram,
    normalized_spectral_distance_matrix,
    spectral_histogram_wasserstein,
    validate_histogram,
)
from grapher.rewiring_mlp.generic.spectral_model import TopologySpectralTransformerPredictor
from grapher.rewiring_mlp.generic.summary_diffusion import (
    SummaryDiffusionConfig,
    sample_eigenspace_histogram_bridge_marginal,
)


def _spec() -> SpectralDistanceHistogramSpec:
    return SpectralDistanceHistogramSpec(rank=3, bins=8, degree_max=8, max_normalized_distance=3.0)


def _small_model(*, predict_edge_state: bool = False):
    spec = _spec()
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
        spectral_representation="lambda_eigenspace_histogram",
        eigenspace_rank=spec.rank,
        eigenspace_histogram_bins=spec.bins,
        eigenspace_histogram_degree_max=spec.degree_max,
        eigenspace_histogram_max_distance=spec.max_normalized_distance,
    )


def test_normalized_spectral_distances_have_unit_mean_squared_scale():
    graph = nx.barbell_graph(4, 2)
    distances = normalized_spectral_distance_matrix(graph, rank=3)
    upper = distances[np.triu_indices(len(graph), 1)]
    assert np.isclose(np.mean(upper ** 2), 1.0, atol=1e-8)


def test_degree_conditioned_histogram_is_permutation_invariant_and_block_normalized():
    graph = nx.barbell_graph(4, 2)
    spec = _spec()
    hist, mask, weights = extract_degree_conditioned_spectral_histogram(graph, spec)
    validate_histogram(hist, spec, block_mask=mask)
    assert np.isclose(weights.sum(), 1.0, atol=1e-12)
    assert np.array_equal(mask, weights > 0.0)
    assert np.allclose(weights, degree_pair_weights(graph, spec), atol=1e-12)

    rng = np.random.default_rng(5)
    permutation = rng.permutation(len(graph))
    mapping = {old: int(permutation[old]) for old in range(len(graph))}
    permuted = nx.relabel_nodes(graph, mapping, copy=True)
    permuted = nx.convert_node_labels_to_integers(permuted, ordering="sorted")
    hist2, mask2, weights2 = extract_degree_conditioned_spectral_histogram(permuted, spec)
    assert np.array_equal(mask, mask2)
    assert np.allclose(weights, weights2, atol=1e-12)
    assert np.allclose(hist, hist2, atol=1e-12)
    assert spectral_histogram_wasserstein(hist, hist2, spec, block_weights=weights) < 1e-12



def test_permutation_invariance_when_requested_rank_cuts_degenerate_block():
    # Barbell graphs contain repeated Laplacian modes; rank=4 cuts a repeated
    # block for this graph and therefore exercises the block-completion policy.
    graph = nx.barbell_graph(5, 2)
    spec = SpectralDistanceHistogramSpec(rank=4, bins=16, degree_max=10, max_normalized_distance=3.0)
    hist, mask, weights = extract_degree_conditioned_spectral_histogram(graph, spec)
    permutation = np.random.default_rng(19).permutation(len(graph))
    mapping = {old: int(permutation[old]) for old in range(len(graph))}
    permuted = nx.relabel_nodes(graph, mapping, copy=True)
    hist2, mask2, weights2 = extract_degree_conditioned_spectral_histogram(permuted, spec)
    assert np.array_equal(mask, mask2)
    assert np.allclose(weights, weights2, atol=1e-12)
    assert np.allclose(hist, hist2, atol=1e-12)

def test_histogram_bridge_has_exact_endpoints_and_preserves_block_sums():
    spec = _spec()
    graph_a = nx.barbell_graph(4, 2)
    # Degree-preserving double-edge swaps give a distinct graph on the same degree fibre.
    graph_b = graph_a.copy()
    nx.double_edge_swap(graph_b, nswap=3, max_tries=100, seed=3)
    source, mask, weights = extract_degree_conditioned_spectral_histogram(graph_a, spec)
    clean, mask_b, weights_b = extract_degree_conditioned_spectral_histogram(graph_b, spec)
    assert np.array_equal(mask, mask_b)
    assert np.allclose(weights, weights_b)
    schedule = SummaryDiffusionConfig.from_dict({
        "bridge": "brownian",
        "schedule": "linear",
        "eigenspace_histogram_sigma": 0.2,
    })
    rng = np.random.default_rng(4)
    at_source, _ = sample_eigenspace_histogram_bridge_marginal(
        source, clean, progress=0.0, sigma=schedule.eigenspace_histogram_sigma,
        block_mask=mask, bins=spec.bins, schedule=schedule, rng=rng,
    )
    at_clean, _ = sample_eigenspace_histogram_bridge_marginal(
        source, clean, progress=1.0, sigma=schedule.eigenspace_histogram_sigma,
        block_mask=mask, bins=spec.bins, schedule=schedule, rng=rng,
    )
    middle, _ = sample_eigenspace_histogram_bridge_marginal(
        source, clean, progress=0.5, sigma=schedule.eigenspace_histogram_sigma,
        block_mask=mask, bins=spec.bins, schedule=schedule, rng=rng,
    )
    assert np.allclose(at_source, source, atol=1e-12)
    assert np.allclose(at_clean, clean, atol=1e-12)
    blocks = middle.reshape(spec.num_blocks, spec.bins)
    assert np.allclose(blocks[mask].sum(axis=1), 1.0, atol=1e-10)
    assert np.allclose(blocks[~mask], 0.0, atol=1e-12)


def test_spectral_histogram_training_batch_loss_and_backward_are_finite():
    spec = _spec()
    graph = nx.barbell_graph(4, 2)
    examples, report = build_spectral_diffusion_examples(
        [graph],
        diffusion_config={
            "bridge": "brownian",
            "schedule": "linear",
            "samples_per_graph": 2,
            "paths_per_graph": 1,
            "spectral_sigma": 0.05,
            "eigenspace_histogram_sigma": 0.05,
        },
        source_config={
            "ensure_connected_source": True,
            "random_relabel_source": True,
            "source_randomization_steps": 0,
        },
        spectral_config={
            "representation": "lambda_eigenspace_histogram",
            "eigenspace_rank": spec.rank,
            "eigenspace_histogram_bins": spec.bins,
            "eigenspace_histogram_degree_max": spec.degree_max,
            "eigenspace_histogram_max_distance": spec.max_normalized_distance,
            "normalization": "mean_degree",
            "lambda_weight": 1.0,
            "eigenspace_histogram_weight": 1.0,
        },
        edge_diffusion_config={"enabled": False},
        seed=7,
    )
    assert report["spectral_representation"] == "lambda_eigenspace_histogram"
    batch = collate_spectral_examples(examples)
    assert batch.current_eigenspace_histogram.shape == (2, spec.width)
    assert batch.eigenspace_histogram_block_mask.shape == (2, spec.num_blocks)
    model = _small_model()
    outputs = model(batch)
    assert outputs["clean_spectrum"].shape == (2, len(graph))
    assert outputs["clean_eigenspace_histogram"].shape == (2, spec.width)
    blocks = outputs["clean_eigenspace_histogram"].reshape(2, spec.num_blocks, spec.bins)
    mask = batch.eigenspace_histogram_block_mask.bool()
    assert torch.allclose(blocks.sum(dim=-1)[mask], torch.ones_like(blocks.sum(dim=-1)[mask]), atol=1e-5)
    assert torch.allclose(blocks[~mask], torch.zeros_like(blocks[~mask]), atol=1e-6)
    loss, metrics = model.loss(
        batch,
        loss_weights={
            "spectrum": 1.0,
            "eigenspace_histogram": 1.0,
            "moment2": 0.1,
            "low_frequency": 0.0,
        },
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert metrics["eigenspace_histogram_w1"] >= 0.0
    assert metrics["eigenspace_histogram_tv"] >= 0.0


def test_spectral_histogram_generation_and_rewiring_preserve_degree_and_support():
    spec = _spec()
    graph = nx.barbell_graph(4, 2)
    model = _small_model()
    config = {
        "edge_diffusion": {"enabled": False, "sampling_steps": 3, "sigma": 0.2, "smoothing": 0.01},
        "summary_diffusion": {"spectral_sigma": 0.05, "eigenspace_histogram_sigma": 0.05},
        "spectral_prediction": {
            "representation": "lambda_eigenspace_histogram",
            "eigenspace_rank": spec.rank,
            "eigenspace_histogram_bins": spec.bins,
            "eigenspace_histogram_degree_max": spec.degree_max,
            "eigenspace_histogram_max_distance": spec.max_normalized_distance,
            "lambda_weight": 1.0,
            "eigenspace_histogram_weight": 1.0,
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
    assert report["spectral_representation"] == "lambda_eigenspace_histogram"
    assert report["joint_laplacian_eigenspace_histogram_diffusion"] is True
    assert targets["spectrum"].shape == (len(graph),)
    assert targets["eigenspace_histogram"].shape == (spec.width,)
    validate_histogram(
        targets["eigenspace_histogram"], spec,
        block_mask=targets["eigenspace_histogram_block_mask"],
    )
    refined, trace = refine_graph(
        graph, targets, model, config, rng=np.random.default_rng(11),
        prediction_calls=report["prediction_calls"],
    )
    assert dict(refined.degree()) == dict(graph.degree())
    assert nx.is_connected(refined)
    assert isinstance(trace, list)
    _, mask_after, weights_after = extract_degree_conditioned_spectral_histogram(refined, spec)
    assert np.array_equal(mask_after, targets["eigenspace_histogram_block_mask"])
    assert np.allclose(weights_after, targets["eigenspace_histogram_block_weights"], atol=1e-12)


def test_spectral_histogram_configs_select_permutation_invariant_space():
    root = Path(__file__).resolve().parents[1]
    for filename in (
        "community_small_eigenspace_histogram_only_learned.yaml",
        "community_small_joint_edge_eigenspace_histogram_graphlets345_learned.yaml",
    ):
        cfg = yaml.safe_load((root / "configs/experiments/grapher" / filename).read_text())
        spectral = cfg["spectral_prediction"]
        assert spectral["representation"] == "lambda_eigenspace_histogram"
        assert spectral["eigenspace_rank"] == 4
        assert spectral["eigenspace_histogram_bins"] > 1
        assert spectral["eigenspace_histogram_degree_max"] == 19
        assert cfg["topology_predictor"]["loss_weights"]["spectrum"] == 1.0
        assert cfg["topology_predictor"]["loss_weights"]["eigenspace_histogram"] == 1.0

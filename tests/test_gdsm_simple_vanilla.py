from __future__ import annotations

import json
import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import torch

from grapher.models.base import DatasetReference, GenerateRequest, RunSpec, TrainRequest
from grapher.models.gdsm_simple.vanilla_gsdm import (
    CHECKPOINT_FORMAT,
    GSDMNodeScore,
    GSDMSpectrumScore,
    degree_features,
    eigen_mask_from_flags,
    reconstruct_adjacency,
)
from grapher.models.gdsm_simple.wrapper import GDSMSimpleWrapper


def vanilla_options() -> dict:
    return {
        "variant": "vanilla_gsdm",
        "train": {
            "epochs": 2,
            "batch_size": 4,
            "lr": 1.0e-3,
            "weight_decay": 0.0,
            "grad_norm": 1.0,
            "lr_schedule": False,
            "lr_decay": 1.0,
            "ema": 0.9,
            "validation_every": 1,
            "log_every": 1,
        },
        "model": {
            "max_nodes": 6,
            "max_feat_num": 6,
            "hidden_dim": 8,
            "depth": 2,
        },
        "sde": {
            "x": {"type": "vp", "beta_min": 0.1, "beta_max": 1.0, "num_scales": 4},
            "spectrum": {"type": "vp", "beta_min": 0.1, "beta_max": 1.0, "num_scales": 4},
            "eps": 1.0e-5,
            "eigen_mask": "official_extremes",
        },
        "sample": {
            "predictor": "euler",
            "corrector": "none",
            "snr": 0.05,
            "scale_eps": 0.7,
            "n_steps": 0,
            "noise_removal": True,
            "probability_flow": False,
            "eps": 1.0e-3,
            "threshold": 0.5,
            "use_ema": False,
        },
        "generation_batch_size": 2,
        "runtime": {"device": "cpu"},
        "extensions": {
            "degree_conditioning": False,
            "hh_initialization": False,
            "degree_preserving_rewiring": False,
            "structural_summary": "none",
        },
    }


def test_official_extreme_eigen_mask_keeps_exact_node_count_for_odd_sizes():
    flags = torch.tensor(
        [
            [1, 1, 1, 1, 1, 0, 0],  # n=5 -> first 2 + last 3
            [1, 1, 1, 1, 1, 1, 0],  # n=6 -> first 3 + last 3
        ],
        dtype=torch.float32,
    )
    mask = eigen_mask_from_flags(flags)
    assert mask[0].tolist() == [1, 1, 0, 0, 1, 1, 1]
    assert mask[1].tolist() == [1, 1, 1, 0, 1, 1, 1]
    torch.testing.assert_close(mask.sum(1), flags.sum(1))


def test_reconstruction_and_source_faithful_score_shapes():
    torch.manual_seed(3)
    b, n, f = 2, 6, 6
    raw = torch.randn(b, n, n)
    adj = 0.5 * (raw + raw.transpose(-1, -2))
    lam, u = torch.linalg.eigh(adj)
    rebuilt = reconstruct_adjacency(u, lam)
    torch.testing.assert_close(rebuilt, adj, atol=2e-5, rtol=2e-5)

    flags = torch.ones(b, n)
    x = degree_features((adj > 0.5).float(), flags, f)
    mx = GSDMNodeScore(max_feat_num=f, hidden_dim=8, depth=3)
    ml = GSDMSpectrumScore(max_feat_num=f, max_nodes=n, hidden_dim=8, depth=3)
    assert mx(x, adj, flags, u, lam).shape == x.shape
    assert ml(x, adj, flags, u, lam).shape == lam.shape


def test_degree_features_fail_instead_of_silently_clamping():
    adj = torch.ones(1, 5, 5) - torch.eye(5).unsqueeze(0)
    flags = torch.ones(1, 5)
    with pytest.raises(ValueError, match="degree-one-hot support"):
        degree_features(adj, flags, max_feat_num=4)


def _write_dataset(root: Path) -> tuple[DatasetReference, list[nx.Graph]]:
    folder = root / "datasets" / "toy"
    folder.mkdir(parents=True)
    train = [
        nx.cycle_graph(5),
        nx.path_graph(5),
        nx.star_graph(4),
        nx.complete_bipartite_graph(2, 3),
    ]
    val = [nx.cycle_graph(4), nx.path_graph(4)]
    test = [nx.wheel_graph(5)]
    for split, graphs in (("train", train), ("val", val), ("test", test)):
        with (folder / f"{split}.pkl").open("wb") as handle:
            pickle.dump(graphs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return DatasetReference("community_small", root / "datasets", "toy"), train


@pytest.fixture(scope="module")
def vanilla_run(tmp_path_factory):
    torch.set_num_threads(1)
    root = tmp_path_factory.mktemp("vanilla_gsdm")
    dataset, train_graphs = _write_dataset(root)
    run = RunSpec("gdsm_simple", "community_small", "vanilla", 42, root / "runs")
    wrapper = GDSMSimpleWrapper()
    request = TrainRequest(run, dataset, options=vanilla_options())
    artifacts = wrapper.train(request)
    return root, wrapper, request, artifacts, train_graphs


def test_vanilla_checkpoint_has_only_training_spectral_basis(vanilla_run):
    _, wrapper, request, artifacts, train_graphs = vanilla_run
    state = torch.load(artifacts.checkpoint_path, map_location="cpu", weights_only=False)
    assert state["format"] == CHECKPOINT_FORMAT
    assert state["variant"] == "vanilla_gsdm"
    assert state["basis_source"] == "training_split_only"
    assert len(state["basis_adjacencies"]) == len(train_graphs)
    assert "degree_generator" not in state
    assert "graphlet" not in state
    assert "edge_head" not in state
    manifest = json.loads(artifacts.manifest_path.read_text())
    contract = manifest["reference_contract"]
    assert contract["degree_constraint"] is False
    assert contract["rewiring"] is False
    assert contract["categorical_edge_head"] is False
    assert contract["structural_guidance"] is False
    assert wrapper.train(request).checkpoint_path == artifacts.checkpoint_path


def test_vanilla_generation_is_training_basis_plus_one_final_threshold(vanilla_run):
    _, wrapper, request, artifacts, train_graphs = vanilla_run
    result = wrapper.generate(
        GenerateRequest(
            request.run,
            artifacts.checkpoint_path,
            5,
            91,
            generation_id="samples",
            options={"runtime": {"device": "cpu"}},
        )
    )
    with result.graphs_path.open("rb") as handle:
        graphs = pickle.load(handle)
    with (result.generation_dir / "sampled_basis_indices.pkl").open("rb") as handle:
        indices = pickle.load(handle)
    with (result.generation_dir / "continuous_adjacencies.pkl").open("rb") as handle:
        soft = pickle.load(handle)
    assert len(graphs) == len(indices) == len(soft) == 5
    for graph, idx, matrix in zip(graphs, indices, soft):
        assert 0 <= idx < len(train_graphs)
        assert graph.number_of_nodes() == train_graphs[idx].number_of_nodes()
        assert nx.number_of_selfloops(graph) == 0
        np.testing.assert_allclose(matrix, matrix.T, atol=1e-6)
    record = json.loads((result.generation_dir / "manifest.json").read_text())
    assert record["sampling"]["eigenvectors"] == "training_split_empirical"
    assert record["sampling"]["degree_constraint"] is False
    assert record["sampling"]["rewiring"] is False
    assert record["sampling"]["structural_guidance"] is False
    assert record["posthoc_repair"] is False


def test_vanilla_generation_is_deterministic_for_same_seed(vanilla_run):
    _, wrapper, request, artifacts, _ = vanilla_run
    a = wrapper.generate(GenerateRequest(request.run, artifacts.checkpoint_path, 3, 123, generation_id="a"))
    b = wrapper.generate(GenerateRequest(request.run, artifacts.checkpoint_path, 3, 123, generation_id="b"))
    assert a.graphs_sha256 == b.graphs_sha256


def test_vanilla_config_starts_from_vanilla_defaults_not_legacy_options(tmp_path):
    config = tmp_path / "vanilla.yaml"
    config.write_text(
        """
gdsm_simple:
  variant: vanilla_gsdm
  train: {epochs: 2, batch_size: 2}
  model: {max_nodes: 6, max_feat_num: 6, hidden_dim: 8, depth: 2}
  sde:
    x: {type: vp, beta_min: 0.1, beta_max: 1.0, num_scales: 4}
    spectrum: {type: vp, beta_min: 0.1, beta_max: 1.0, num_scales: 4}
  sample: {predictor: euler, corrector: none, n_steps: 0}
  extensions:
    degree_conditioning: false
    hh_initialization: false
    degree_preserving_rewiring: false
    structural_summary: none
""",
        encoding="utf-8",
    )
    wrapper = GDSMSimpleWrapper()
    req = TrainRequest(
        RunSpec("gdsm_simple", "community_small", "cfg", 42),
        DatasetReference("community_small"),
        config_path=config,
    )
    options = wrapper._options(req)
    assert options["variant"] == "vanilla_gsdm"
    assert "diffusion" not in options
    assert "num_layers" not in options["model"]
    assert "rewiring" not in options["extensions"]

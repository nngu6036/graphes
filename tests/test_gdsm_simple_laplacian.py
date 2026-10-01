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
    _laplacian_eigh_padded,
    _padded_dataset,
    combinatorial_laplacian,
    eigen_mask_from_flags,
    laplacian_to_adjacency_scores,
    reconstruct_adjacency,
)
from grapher.models.gdsm_simple.wrapper import GDSMSimpleWrapper


def laplacian_options() -> dict:
    return {
        "variant": "vanilla_laplacian_gsdm",
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
            "eigen_mask": "laplacian_nonzero_prefix",
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


def test_active_prefix_mask_is_exact_graph_size():
    flags = torch.tensor(
        [[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0]], dtype=torch.float32
    )
    mask = eigen_mask_from_flags(flags, "laplacian_nonzero_prefix")
    assert mask[0].tolist() == [0, 1, 1, 1, 0, 0]
    assert mask[1].tolist() == [0, 1, 1, 1, 1, 0]


def test_clean_laplacian_reconstruction_recovers_adjacency_scores():
    graph = nx.cycle_graph(5)
    a = torch.tensor(nx.to_numpy_array(graph), dtype=torch.float32).unsqueeze(0)
    flags = torch.ones(1, 5)
    lap = combinatorial_laplacian(a, flags)
    lam, u = torch.linalg.eigh(lap)
    rebuilt = reconstruct_adjacency(u, lam)
    scores = laplacian_to_adjacency_scores(rebuilt, flags)
    torch.testing.assert_close(rebuilt, lap, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(scores, a, atol=1e-5, rtol=1e-5)


def test_padded_laplacian_eigh_keeps_padding_out_of_zero_eigenspace():
    a = torch.zeros(2, 6, 6)
    a[0, :4, :4] = torch.tensor(nx.to_numpy_array(nx.path_graph(4)), dtype=torch.float32)
    a[1, :5, :5] = torch.tensor(nx.to_numpy_array(nx.cycle_graph(5)), dtype=torch.float32)
    sizes = torch.tensor([4, 5])
    lam, u = _laplacian_eigh_padded(a, sizes)
    for row, n in enumerate((4, 5)):
        rebuilt = reconstruct_adjacency(u[row:row+1], lam[row:row+1])[0]
        target = combinatorial_laplacian(
            a[row:row+1],
            (torch.arange(6).unsqueeze(0) < n).float(),
        )[0]
        torch.testing.assert_close(rebuilt, target, atol=1e-5, rtol=1e-5)
        assert torch.all(lam[row, n:] == 0)
        torch.testing.assert_close(
            u[row, n:, n:], torch.eye(6 - n), atol=0, rtol=0
        )


def _write_dataset(root: Path) -> DatasetReference:
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
    return DatasetReference("community_small", root / "datasets", "toy")


def test_laplacian_dataset_uses_clean_laplacian_spectrum():
    graphs = [nx.path_graph(4), nx.cycle_graph(5)]
    _, adj, flags, sizes, u, lam = _padded_dataset(
        graphs, max_nodes=6, max_feat_num=6,
        spectral_operator="combinatorial_laplacian",
    )
    rebuilt = reconstruct_adjacency(u, lam)
    target = combinatorial_laplacian(adj, flags)
    torch.testing.assert_close(rebuilt, target, atol=2e-5, rtol=2e-5)
    assert sizes.tolist() == [4, 5]


def test_laplacian_variant_rejects_adjacency_extreme_mask(tmp_path):
    wrapper = GDSMSimpleWrapper()
    options = laplacian_options()
    options["sde"]["eigen_mask"] = "official_extremes"
    dataset = _write_dataset(tmp_path)
    run = RunSpec("gdsm_simple", "community_small", "bad-mask", 42, tmp_path / "runs")
    with pytest.raises(ValueError, match="laplacian_nonzero_prefix"):
        wrapper.train(TrainRequest(run, dataset, options=options))


def test_laplacian_stage2_train_generate_contract(tmp_path):
    torch.set_num_threads(1)
    dataset = _write_dataset(tmp_path)
    run = RunSpec("gdsm_simple", "community_small", "lap-stage2", 42, tmp_path / "runs")
    wrapper = GDSMSimpleWrapper()
    artifacts = wrapper.train(TrainRequest(run, dataset, options=laplacian_options()))
    state = torch.load(artifacts.checkpoint_path, map_location="cpu", weights_only=False)
    assert state["format"] == CHECKPOINT_FORMAT
    assert state["variant"] == "vanilla_laplacian_gsdm"
    assert state["spectral_operator"] == "combinatorial_laplacian"
    assert not state["degree_prior"]["enabled"]
    manifest = json.loads(artifacts.manifest_path.read_text())
    contract = manifest["reference_contract"]
    assert contract["spectral_operator"] == "combinatorial_laplacian"
    assert contract["degree_constraint"] is False
    assert contract["rewiring"] is False
    assert contract["categorical_edge_head"] is False

    result = wrapper.generate(
        GenerateRequest(
            run, artifacts.checkpoint_path, 5, 91,
            generation_id="samples",
            options={"runtime": {"device": "cpu"}},
        )
    )
    with result.graphs_path.open("rb") as handle:
        graphs = pickle.load(handle)
    with (result.generation_dir / "continuous_adjacencies.pkl").open("rb") as handle:
        scores = pickle.load(handle)
    with (result.generation_dir / "continuous_laplacians.pkl").open("rb") as handle:
        laps = pickle.load(handle)
    with (result.generation_dir / "sampled_basis_indices.pkl").open("rb") as handle:
        indices = pickle.load(handle)
    with (result.generation_dir / "sampled_spectra.pkl").open("rb") as handle:
        spectra = pickle.load(handle)
    assert len(graphs) == len(scores) == len(laps) == len(indices) == len(spectra) == 5
    for graph, score, lap, spectrum in zip(graphs, scores, laps, spectra):
        assert nx.number_of_selfloops(graph) == 0
        np.testing.assert_allclose(score, score.T, atol=1e-6)
        np.testing.assert_allclose(lap, lap.T, atol=1e-6)
        assert abs(float(spectrum[0])) < 1e-8
        np.testing.assert_allclose(lap.sum(axis=1), 0.0, atol=2e-5, rtol=0)
        offdiag = ~np.eye(len(score), dtype=bool)
        np.testing.assert_allclose(score[offdiag], -lap[offdiag], atol=2e-5, rtol=2e-5)
        expected = score > 0.5
        np.fill_diagonal(expected, False)
        np.testing.assert_array_equal(nx.to_numpy_array(graph).astype(bool), expected)

    record = json.loads((result.generation_dir / "manifest.json").read_text())
    assert record["sampling"]["spectral_operator"] == "combinatorial_laplacian"
    assert record["sampling"]["degree_constraint"] is False
    assert record["sampling"]["rewiring"] is False
    assert record["continuous_laplacians"] is not None
    assert record["diagnostics"]["valid_laplacian_projection_applied"] is False


def test_stage2_explicit_config_resolves_without_legacy_extensions():
    root = Path(__file__).resolve().parents[1]
    config = root / "configs/experiments/gdsm_vanilla_laplacian_explicit/community_small_seed_42.yaml"
    wrapper = GDSMSimpleWrapper()
    request = TrainRequest(
        RunSpec("gdsm_simple", "community_small", "cfg", 42),
        DatasetReference("community_small"),
        config_path=config,
    )
    options = wrapper._options(request)
    assert options["variant"] == "vanilla_laplacian_gsdm"
    assert options["sde"]["eigen_mask"] == "laplacian_nonzero_prefix"
    assert options["degree_prior"]["enabled"] is False
    assert options["extensions"]["degree_preserving_rewiring"] is False

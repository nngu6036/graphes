from __future__ import annotations

import json
import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import torch

from grapher.models.base import DatasetReference, GenerateRequest, RunSpec, TrainRequest
from grapher.models.gdsm_simple.vanilla_gsdm import (
    _joint_structure_targets,
    _resolved_structure_summary_config,
    validate_options,
)
from grapher.models.gdsm_simple.wrapper import GDSMSimpleWrapper


def _summary(max_degree: int = 6) -> dict:
    return {
        "enabled": True,
        "loss_weight": 0.1,
        "graphlet": {
            "enabled": True,
            "orders": [3, 4, 5, 6],
            "histogram_weight": 1.0,
            "mass_weight": 0.25,
        },
        "clustering": {"enabled": False, "bins": 20},
        "orbit": {
            "enabled": True,
            "width": 15,
            "histogram_weight": 0.25,
            "log_total_weight": 0.1,
        },
        "degree": {
            "enabled": True,
            "max_degree": max_degree,
            "histogram_weight": 0.25,
            "cdf_weight": 1.0,
        },
    }


def test_degree_histogram_targets_are_normalized_and_permutation_invariant() -> None:
    path = nx.path_graph(4)
    star = nx.star_graph(3)
    tensors, meta = _joint_structure_targets([path, star], _summary(max_degree=5))
    graphlet_hist, graphlet_mass, clustering, orbit_hist, orbit_total, degree_hist = tensors

    assert meta["orders"] == [3, 4, 5, 6]
    assert meta["degree_enabled"] is True
    assert meta["degree_max_degree"] == 5
    assert meta["degree_bins"] == 6
    assert graphlet_hist.shape[0] == 2
    assert graphlet_mass.shape == (2, 4)
    assert clustering.shape == (2, 20)
    assert orbit_hist.shape == (2, 15)
    assert orbit_total.shape == (2, 1)
    assert degree_hist.shape == (2, 6)
    assert torch.allclose(degree_hist.sum(dim=-1), torch.ones(2), atol=1e-6)
    assert torch.allclose(
        degree_hist[0],
        torch.tensor([0.0, 0.5, 0.5, 0.0, 0.0, 0.0]),
        atol=1e-6,
    )
    assert torch.allclose(
        degree_hist[1],
        torch.tensor([0.0, 0.75, 0.0, 0.25, 0.0, 0.0]),
        atol=1e-6,
    )

    relabeled = nx.relabel_nodes(path, {node: node + 100 for node in path.nodes()})
    tensors2, _ = _joint_structure_targets([relabeled], _summary(max_degree=5))
    assert torch.allclose(degree_hist[:1], tensors2[-1], atol=1e-6)


def _options() -> dict:
    return {
        "variant": "vanilla_laplacian_loggap_graphlet",
        "train": {
            "epochs": 1,
            "batch_size": 4,
            "lr": 1e-3,
            "node_lr": 1e-3,
            "spectrum_lr": 5e-4,
            "weight_decay": 0.0,
            "grad_norm": 1.0,
            "lr_schedule": False,
            "lr_decay": 1.0,
            "ema": 0.9,
            "validation_every": 1,
            "log_every": 1,
        },
        "model": {
            "max_nodes": 7,
            "max_feat_num": 7,
            "hidden_dim": 8,
            "depth": 2,
            "backbone": "dense_gcn",
            "node_backbone": "dense_gcn",
            "spectrum_backbone": "ppgn",
            "ppgn_hidden_dim": 12,
            "ppgn_depth": 2,
            "ppgn_residual_scale": 0.1,
            "ppgn_norm_eps": 1e-5,
            "ppgn_input_clip": 10.0,
        },
        "structural_features": {
            "enabled": True,
            "binarize_threshold": 0.5,
            "random_walk": {"enabled": True, "steps": 3},
            "shortest_path": {"enabled": True, "max_distance": 4},
        },
        "sde": {
            "x": {"type": "vp", "beta_min": 0.1, "beta_max": 0.2, "num_scales": 2},
            "spectrum": {"type": "vp", "beta_min": 0.1, "beta_max": 0.2, "num_scales": 2},
            "eps": 1e-5,
            "eigen_mask": "laplacian_nonzero_prefix",
            "spectral_parameterization": "laplacian_log_gap",
            "log_gap_epsilon": 1e-6,
            "log_gap_min_std": 1e-3,
            "log_gap_exp_clip": 8.0,
        },
        "graphlet_summary": {"enabled": False},
        "structure_summary": _summary(max_degree=6),
        "graphlet_refinement": {"enabled": False},
        "degree_prior": {"enabled": False},
        "sample": {
            "predictor": "euler",
            "corrector": "none",
            "snr": 0.05,
            "scale_eps": 0.7,
            "n_steps": 0,
            "noise_removal": True,
            "probability_flow": False,
            "eps": 1e-2,
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


def _write_dataset(root: Path) -> DatasetReference:
    folder = root / "datasets" / "toy"
    folder.mkdir(parents=True)
    train = [
        nx.cycle_graph(6),
        nx.path_graph(6),
        nx.star_graph(5),
        nx.complete_bipartite_graph(3, 3),
    ]
    val = [nx.cycle_graph(6), nx.path_graph(6)]
    test = [nx.wheel_graph(7)]
    for split, graphs in (("train", train), ("val", val), ("test", test)):
        with (folder / f"{split}.pkl").open("wb") as handle:
            pickle.dump(graphs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return DatasetReference("community_small", root / "datasets", "toy")


def test_degree_histogram_head_train_generate_smoke(tmp_path: Path) -> None:
    torch.set_num_threads(1)
    options = _options()
    validate_options(options)
    resolved = _resolved_structure_summary_config(options)
    assert resolved["degree"] == {
        "enabled": True,
        "max_degree": 6,
        "histogram_weight": 0.25,
        "cdf_weight": 1.0,
    }

    wrapper = GDSMSimpleWrapper()
    dataset = _write_dataset(tmp_path)
    run = RunSpec("gdsm_simple", "community_small", "degree_g346", 42, tmp_path / "runs")
    artifacts = wrapper.train(TrainRequest(run, dataset, options=options))
    state = torch.load(artifacts.checkpoint_path, map_location="cpu", weights_only=False)
    assert state["structure_summary"]["degree_enabled"] is True
    assert state["structure_summary"]["degree_bins"] == 7
    assert any("degree_histogram_logits" in key for key in state["model_spectrum_state"])
    assert "train_degree_histogram_loss" in state["history"][0]
    assert "val_degree_cdf_mae" in state["history"][0]

    generated = wrapper.generate(
        GenerateRequest(run, artifacts.checkpoint_path, 2, 99, generation_id="degree")
    )
    with (generated.generation_dir / "predicted_structure_summaries.pkl").open("rb") as handle:
        rows = pickle.load(handle)
    assert len(rows) == 2
    for row in rows:
        hist = np.asarray(row["degree_histogram"], dtype=np.float64)
        assert hist.shape == (7,)
        assert np.isfinite(hist).all()
        assert np.isclose(hist.sum(), 1.0)
        assert np.isfinite(np.asarray(row["degree_mean"])).all()

    manifest = json.loads((generated.generation_dir / "manifest.json").read_text())
    summary = manifest["predicted_structure_summaries"]
    assert summary["degree_enabled"] is True
    assert summary["degree_max_degree"] == 6
    assert summary["degree_bins"] == 7
    assert summary["graphlet_orders"] == [3, 4, 5, 6]
    assert manifest["sampling"]["rewiring"] is False


def test_shipped_degree_g346_configs_resolve(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    wrapper = GDSMSimpleWrapper()
    for dataset, max_degree in (("community_small", 19), ("ego_small", 17)):
        for seed in (42, 43, 44):
            config = (
                root
                / "configs/experiments/gdsm_laplacian_loggap_topology_degree_g346_explicit"
                / f"{dataset}_seed_{seed}.yaml"
            )
            request = TrainRequest(
                RunSpec("gdsm_simple", dataset, "cfg", seed, tmp_path / "runs"),
                DatasetReference(dataset, tmp_path / "datasets", "unused"),
                config_path=config,
            )
            options = wrapper._options(request)
            validate_options(options)
            summary = _resolved_structure_summary_config(options)
            assert summary["graphlet"]["orders"] == [3, 4, 5, 6]
            assert summary["degree"]["enabled"] is True
            assert summary["degree"]["max_degree"] == max_degree
            assert summary["orbit"]["enabled"] is True
            assert options["degree_prior"]["enabled"] is False
            assert options["graphlet_refinement"]["enabled"] is False

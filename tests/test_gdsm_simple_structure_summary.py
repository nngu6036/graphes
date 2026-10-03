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
)
from grapher.models.gdsm_simple.wrapper import GDSMSimpleWrapper


def _summary_cfg():
    return {
        "enabled": True,
        "loss_weight": 0.1,
        "graphlet": {"enabled": True, "orders": [3, 4, 5], "histogram_weight": 1.0, "mass_weight": 0.25},
        "clustering": {"enabled": True, "bins": 20, "histogram_weight": 0.25, "cdf_weight": 1.0},
        "orbit": {"enabled": True, "width": 15, "histogram_weight": 0.25, "log_total_weight": 0.1},
    }


def test_structure_targets_have_expected_shapes_and_are_relabel_invariant():
    graphs = [nx.cycle_graph(5), nx.complete_graph(5)]
    tensors, meta = _joint_structure_targets(graphs, _summary_cfg())
    gh, gm, ch, oh, olt = tensors
    assert gh.shape[0] == 2 and gh.shape[1] == meta["width"]
    assert gm.shape == (2, 3)
    assert ch.shape == (2, 20)
    assert oh.shape == (2, 15)
    assert olt.shape == (2, 1)
    assert torch.allclose(ch.sum(-1), torch.ones(2), atol=1e-6)
    assert torch.allclose(oh.sum(-1), torch.ones(2), atol=1e-6)
    assert torch.all(olt >= 0)

    relabeled = nx.relabel_nodes(graphs[0], {i: (3 * i + 7) for i in graphs[0].nodes()})
    tensors2, _ = _joint_structure_targets([relabeled], _summary_cfg())
    for a, b in zip(tensors, tensors2):
        assert torch.allclose(a[:1], b, atol=1e-6)


def _write_dataset(root: Path):
    folder = root / "datasets" / "toy"
    folder.mkdir(parents=True)
    train = [nx.cycle_graph(5), nx.path_graph(5), nx.star_graph(4), nx.complete_bipartite_graph(2, 3)]
    val = [nx.cycle_graph(4), nx.path_graph(4)]
    test = [nx.wheel_graph(5)]
    for split, graphs in (("train", train), ("val", val), ("test", test)):
        with (folder / f"{split}.pkl").open("wb") as handle:
            pickle.dump(graphs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return DatasetReference("community_small", root / "datasets", "toy")


def _options():
    return {
        "variant": "vanilla_laplacian_loggap_graphlet",
        "train": {
            "epochs": 1, "batch_size": 4, "lr": 1e-3, "node_lr": 1e-3, "spectrum_lr": 5e-4,
            "weight_decay": 0.0, "grad_norm": 1.0, "lr_schedule": False, "lr_decay": 1.0,
            "ema": 0.9, "validation_every": 1, "log_every": 1,
        },
        "model": {
            "max_nodes": 6, "max_feat_num": 6, "hidden_dim": 8, "depth": 2,
            "backbone": "dense_gcn", "node_backbone": "dense_gcn", "spectrum_backbone": "ppgn",
            "ppgn_hidden_dim": 12, "ppgn_depth": 2, "ppgn_residual_scale": 0.1,
            "ppgn_norm_eps": 1e-5, "ppgn_input_clip": 10.0,
        },
        "structural_features": {
            "enabled": True, "binarize_threshold": 0.5,
            "random_walk": {"enabled": True, "steps": 3},
            "shortest_path": {"enabled": True, "max_distance": 4},
        },
        "sde": {
            "x": {"type": "vp", "beta_min": 0.1, "beta_max": 0.2, "num_scales": 2},
            "spectrum": {"type": "vp", "beta_min": 0.1, "beta_max": 0.2, "num_scales": 2},
            "eps": 1e-5, "eigen_mask": "laplacian_nonzero_prefix",
            "spectral_parameterization": "laplacian_log_gap", "log_gap_epsilon": 1e-6,
            "log_gap_min_std": 1e-3, "log_gap_exp_clip": 8.0,
        },
        "graphlet_summary": {"enabled": False},
        "structure_summary": _summary_cfg(),
        "graphlet_refinement": {"enabled": False},
        "degree_prior": {"enabled": False},
        "sample": {
            "predictor": "euler", "corrector": "none", "snr": 0.05, "scale_eps": 0.7,
            "n_steps": 0, "noise_removal": True, "probability_flow": False,
            "eps": 1e-2, "threshold": 0.5, "use_ema": False,
        },
        "generation_batch_size": 2,
        "runtime": {"device": "cpu"},
        "extensions": {
            "degree_conditioning": False, "hh_initialization": False,
            "degree_preserving_rewiring": False, "structural_summary": "none",
        },
    }


def test_joint_structure_ppgn_train_generate_smoke(tmp_path):
    torch.set_num_threads(1)
    dataset = _write_dataset(tmp_path)
    wrapper = GDSMSimpleWrapper()
    run = RunSpec("gdsm_simple", "community_small", "structure", 42, tmp_path / "runs")
    artifacts = wrapper.train(TrainRequest(run, dataset, options=_options()))
    state = torch.load(artifacts.checkpoint_path, map_location="cpu", weights_only=False)
    assert state["structure_summary"]["clustering_enabled"] is True
    assert state["structure_summary"]["clustering_bins"] == 20
    assert state["structure_summary"]["orbit_enabled"] is True
    assert state["structure_summary"]["orbit_width"] == 15
    keys = state["model_spectrum_state"]
    assert any("clustering_logits" in k for k in keys)
    assert any("orbit_histogram_logits" in k for k in keys)
    assert any("orbit_log_total_raw" in k for k in keys)

    generated = wrapper.generate(GenerateRequest(run, artifacts.checkpoint_path, 2, 99, generation_id="s"))
    pred_path = generated.generation_dir / "predicted_structure_summaries.pkl"
    assert pred_path.is_file()
    with pred_path.open("rb") as handle:
        rows = pickle.load(handle)
    assert len(rows) == 2
    for row in rows:
        assert np.isfinite(row["clustering_histogram"]).all()
        assert np.isclose(np.asarray(row["clustering_histogram"]).sum(), 1.0)
        assert np.isfinite(row["orbit_histogram"]).all()
        assert np.isclose(np.asarray(row["orbit_histogram"]).sum(), 1.0)
        assert np.isfinite(row["orbit_mean_counts"]).all()
    manifest = json.loads((generated.generation_dir / "manifest.json").read_text())
    assert manifest["sampling"]["rewiring"] is False
    assert manifest["predicted_structure_summaries"]["clustering_enabled"] is True
    assert manifest["predicted_structure_summaries"]["orbit_enabled"] is True


def test_shipped_structure_ppgn_config(tmp_path):
    root = Path(__file__).resolve().parents[1]
    wrapper = GDSMSimpleWrapper()
    config = root / "configs/experiments/gdsm_laplacian_loggap_structure_rwsp_ppgn_explicit/community_small_seed_42.yaml"
    request = TrainRequest(
        RunSpec("gdsm_simple", "community_small", "cfg", 42, tmp_path / "runs"),
        DatasetReference("community_small", tmp_path / "datasets", "unused"),
        config_path=config,
    )
    options = wrapper._options(request)
    resolved = _resolved_structure_summary_config(options)
    assert resolved["clustering"]["enabled"] is True
    assert resolved["clustering"]["bins"] == 100
    assert resolved["orbit"]["enabled"] is True
    assert resolved["orbit"]["width"] == 15
    assert options["model"]["spectrum_backbone"] == "ppgn"
    assert options["graphlet_refinement"]["enabled"] is False

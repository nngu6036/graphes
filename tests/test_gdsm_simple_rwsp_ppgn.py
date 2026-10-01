from __future__ import annotations

import json
import pickle

import networkx as nx
import torch

from grapher.models.base import DatasetReference, GenerateRequest, RunSpec, TrainRequest
from grapher.models.gdsm_simple.vanilla_gsdm import (
    random_walk_landing_features,
    truncated_shortest_path_features,
)
from grapher.models.gdsm_simple.wrapper import GDSMSimpleWrapper


def _write_dataset(root):
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


def _options(backbone: str) -> dict:
    return {
        "variant": "vanilla_laplacian_loggap_graphlet",
        "train": {
            "epochs": 1,
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
            "backbone": backbone,
            "ppgn_hidden_dim": 12,
            "ppgn_depth": 2,
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
            "eps": 1.0e-5,
            "eigen_mask": "laplacian_nonzero_prefix",
            "spectral_parameterization": "laplacian_log_gap",
            "log_gap_epsilon": 1.0e-6,
            "log_gap_min_std": 1.0e-3,
            "log_gap_exp_clip": 8.0,
        },
        "graphlet_summary": {
            "enabled": True,
            "orders": [3, 4, 5],
            "loss_weight": 0.10,
            "histogram_weight": 1.0,
            "mass_weight": 0.25,
        },
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
            "eps": 1.0e-2,
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


def test_rw_and_shortest_path_features_are_permutation_equivariant():
    adj = torch.tensor(
        [[[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]]],
        dtype=torch.float32,
    )
    flags = torch.ones((1, 4), dtype=torch.float32)
    rw = random_walk_landing_features(adj, flags, steps=3)
    sp = truncated_shortest_path_features(adj, flags, max_distance=3)
    assert rw.shape == (1, 4, 3)
    assert sp.shape == (1, 4, 4, 5)
    # A simple loop-free graph has zero one-step return probability.
    assert torch.allclose(rw[:, :, 0], torch.zeros((1, 4)), atol=1.0e-6)

    perm = torch.tensor([2, 0, 3, 1])
    adj_p = adj[:, perm][:, :, perm]
    flags_p = flags[:, perm]
    rw_p = random_walk_landing_features(adj_p, flags_p, steps=3)
    sp_p = truncated_shortest_path_features(adj_p, flags_p, max_distance=3)
    assert torch.allclose(rw_p, rw[:, perm], atol=1.0e-6)
    assert torch.equal(sp_p, sp[:, perm][:, :, perm])


def _train_and_generate(tmp_path, backbone: str):
    torch.set_num_threads(1)
    dataset = _write_dataset(tmp_path)
    wrapper = GDSMSimpleWrapper()
    run = RunSpec("gdsm_simple", "community_small", f"rwsp-{backbone}", 42, tmp_path / "runs")
    artifacts = wrapper.train(TrainRequest(run, dataset, options=_options(backbone)))
    state = torch.load(artifacts.checkpoint_path, map_location="cpu", weights_only=False)
    assert state["variant"] == "vanilla_laplacian_loggap_graphlet"
    assert state["model_config"]["backbone"] == backbone
    assert state["model_config"]["structural_features"]["enabled"] is True
    assert state["model_config"]["structural_features"]["random_walk"]["steps"] == 3
    assert state["model_config"]["structural_features"]["shortest_path"]["max_distance"] == 4
    generated = wrapper.generate(
        GenerateRequest(run, artifacts.checkpoint_path, 2, 99, generation_id="s")
    )
    manifest = json.loads((generated.generation_dir / "manifest.json").read_text())
    assert manifest["sampling"]["rewiring"] is False
    assert (generated.generation_dir / "predicted_graphlet_summaries.pkl").is_file()
    return state


def test_densegcn_rwsp_joint_graphlet_variant(tmp_path):
    state = _train_and_generate(tmp_path, "dense_gcn")
    assert any("edge_message" in key for key in state["model_spectrum_state"])
    assert not any("ppgn_layers" in key for key in state["model_spectrum_state"])


def test_ppgn_rwsp_joint_graphlet_variant(tmp_path):
    state = _train_and_generate(tmp_path, "ppgn")
    assert any("ppgn_layers" in key for key in state["model_spectrum_state"])
    assert any("ppgn_layers" in key for key in state["model_x_state"])


def test_shipped_rwsp_configs_select_backbone_and_keep_rewiring_off(tmp_path):
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    wrapper = GDSMSimpleWrapper()
    expected = {
        "gdsm_laplacian_loggap_graphlet_rwsp_densegcn_explicit": "dense_gcn",
        "gdsm_laplacian_loggap_graphlet_rwsp_ppgn_explicit": "ppgn",
    }
    for folder, backbone in expected.items():
        config = root / "configs/experiments" / folder / "community_small_seed_42.yaml"
        request = TrainRequest(
            RunSpec("gdsm_simple", "community_small", f"cfg-{backbone}", 42, tmp_path / "runs"),
            DatasetReference("community_small", tmp_path / "datasets", "unused"),
            config_path=config,
        )
        options = wrapper._options(request)
        assert options["variant"] == "vanilla_laplacian_loggap_graphlet"
        assert options["model"]["backbone"] == backbone
        assert options["structural_features"]["enabled"] is True
        assert options["structural_features"]["random_walk"]["enabled"] is True
        assert options["structural_features"]["shortest_path"]["enabled"] is True
        assert options["graphlet_refinement"]["enabled"] is False


def test_ppgn_score_is_permutation_consistent():
    from grapher.models.gdsm_simple.vanilla_gsdm import (
        GSDMNodePPGNScore,
        GSDMSpectrumGraphletPPGNScore,
    )

    torch.manual_seed(7)
    b, n, f = 1, 5, 5
    x = torch.randn((b, n, f))
    raw = torch.rand((b, n, n))
    adj = 0.5 * (raw + raw.transpose(-1, -2))
    idx = torch.arange(n)
    adj[:, idx, idx] = 0.0
    flags = torch.ones((b, n))
    u = torch.eye(n).unsqueeze(0)
    z = torch.randn((b, n))
    sf = {
        "enabled": True,
        "binarize_threshold": 0.5,
        "random_walk": {"enabled": True, "steps": 3},
        "shortest_path": {"enabled": True, "max_distance": 3},
    }
    node = GSDMNodePPGNScore(
        max_feat_num=f,
        max_nodes=n,
        structural_features=sf,
        ppgn_hidden_dim=12,
        ppgn_depth=2,
    )
    spec = GSDMSpectrumGraphletPPGNScore(
        max_feat_num=f,
        max_nodes=n,
        hidden_dim=8,
        depth=2,
        graphlet_slices=((0, 2), (2, 8), (8, 29)),
        structural_features=sf,
        ppgn_hidden_dim=12,
        ppgn_depth=2,
    )
    perm = torch.tensor([2, 4, 0, 1, 3])
    x_p = x[:, perm]
    adj_p = adj[:, perm][:, :, perm]
    flags_p = flags[:, perm]
    u_p = u[:, perm][:, :, perm]

    node_out = node(x, adj, flags, u, z)
    node_out_p = node(x_p, adj_p, flags_p, u_p, z)
    assert torch.allclose(node_out_p, node_out[:, perm], atol=1.0e-5, rtol=1.0e-5)

    out = spec.forward_all(x, adj, flags, u, z)
    out_p = spec.forward_all(x_p, adj_p, flags_p, u_p, z)
    for key in ("spectrum", "graphlet_logits", "graphlet_mass_logits"):
        assert torch.allclose(out_p[key], out[key], atol=1.0e-5, rtol=1.0e-5)

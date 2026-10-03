from __future__ import annotations

import networkx as nx
import numpy as np

from grapher.rewiring_mlp.core.rewiring import apply_action, make_action
from grapher.rewiring_mlp.generic.refiner import (
    TopologyRefinerConfig,
    _filter_candidates_by_motif_guard,
    _select_row,
)


def _soft_cfg(**updates):
    raw = {
        "steps": 4,
        "proposal_budget": 32,
        "valid_candidate_budget": 16,
        "preserve_connectivity": True,
        "selection": "softmax_safe",
        "graphlet_weight": 1.0,
        "clustering_weight": 0.25,
        "orbit_weight": 0.25,
        "accept_only_improving": False,
        "soft_selection": {
            "temperature_start": 0.5,
            "temperature_end": 0.05,
            "temperature_schedule": "linear",
            "score_normalization": "std",
            "stop_action": True,
            "max_target_worsening": 0.02,
            "max_relative_target_worsening": 0.10,
        },
        "motif_guard": {
            "enabled": True,
            "max_destroyed_triangles": 0,
            "max_destroyed_cycles": {"4": 1, "5": 2},
        },
    }
    raw.update(updates)
    return TopologyRefinerConfig.from_dict(raw)


def test_soft_rewiring_config_allows_bounded_worsening():
    cfg = _soft_cfg()
    assert cfg.selection == "softmax_safe"
    assert cfg.accept_only_improving is False
    assert cfg.max_target_worsening == 0.02
    assert cfg.max_relative_target_worsening == 0.10
    assert cfg.temperature_at(0.0) == 0.5
    assert np.isclose(cfg.temperature_at(1.0), 0.05)
    assert cfg.motif_guard_enabled is True
    assert cfg.max_destroyed_triangles == 0


def test_softmax_filters_large_worsening_but_keeps_small_worsening_and_stop():
    cfg = _soft_cfg()
    rows = [
        {"energy_improvement": -0.01, "relative_energy_improvement": -0.05},
        {"energy_improvement": -0.05, "relative_energy_improvement": -0.05},
    ]
    selected, p_stop, probs, temperature, n_eligible = _select_row(
        rows,
        config=cfg,
        rng=np.random.default_rng(7),
        progress=0.0,
    )
    assert selected in {None, 0}
    assert probs[0] > 0.0
    assert probs[1] == 0.0
    assert p_stop > 0.0
    assert probs[-1] > 0.0
    assert n_eligible == 1
    assert temperature == 0.5


def test_motif_guard_rejects_swap_that_destroys_triangle():
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 0), (2, 3), (3, 4)])
    action = make_action([(0, 1), (3, 4)], [(0, 3), (1, 4)])
    candidate = apply_action(graph, action)
    cfg = _soft_cfg()
    candidates, graphs, damage, diagnostics = _filter_candidates_by_motif_guard(
        graph,
        [action],
        {action: candidate},
        config=cfg,
    )
    assert candidates == []
    assert graphs == {}
    assert damage == {}
    assert diagnostics["num_motif_guard_rejections"] == 1
    assert diagnostics["motif_guard_rejection_reasons"]["triangle_guard"] == 1


def test_motif_guard_can_be_relaxed_without_disabling_soft_selection():
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 0), (2, 3), (3, 4)])
    action = make_action([(0, 1), (3, 4)], [(0, 3), (1, 4)])
    candidate = apply_action(graph, action)
    cfg = _soft_cfg(
        motif_guard={
            "enabled": True,
            "max_destroyed_triangles": 1,
            "max_destroyed_cycles": {"4": 10, "5": 10},
        }
    )
    candidates, graphs, damage, diagnostics = _filter_candidates_by_motif_guard(
        graph,
        [action],
        {action: candidate},
        config=cfg,
    )
    assert candidates == [action]
    assert action in graphs
    assert damage[action]["destroyed_triangles"] == 1
    assert diagnostics["num_motif_guard_rejections"] == 0


def test_selected_structure_checkpoint_accepts_generation_only_soft_rewiring(tmp_path):
    import pickle
    import torch
    from pathlib import Path

    from grapher.models.base import DatasetReference, GenerateRequest, RunSpec, TrainRequest
    from grapher.models.gdsm_simple.wrapper import GDSMSimpleWrapper

    folder = tmp_path / "datasets" / "toy"
    folder.mkdir(parents=True)
    train = [nx.cycle_graph(5), nx.path_graph(5), nx.star_graph(4), nx.complete_bipartite_graph(2, 3)]
    val = [nx.cycle_graph(4), nx.path_graph(4)]
    test = [nx.wheel_graph(5)]
    for split, graphs in (("train", train), ("val", val), ("test", test)):
        with (folder / f"{split}.pkl").open("wb") as handle:
            pickle.dump(graphs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    dataset = DatasetReference("community_small", tmp_path / "datasets", "toy")

    summary = {
        "enabled": True,
        "loss_weight": 0.1,
        "graphlet": {"enabled": True, "orders": [3, 4, 5], "histogram_weight": 1.0, "mass_weight": 0.25},
        "clustering": {"enabled": True, "bins": 20, "histogram_weight": 0.25, "cdf_weight": 1.0},
        "orbit": {"enabled": True, "width": 15, "histogram_weight": 0.25, "log_total_weight": 0.1},
    }
    options = {
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
        "structure_summary": summary,
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
    torch.set_num_threads(1)
    wrapper = GDSMSimpleWrapper()
    run = RunSpec("gdsm_simple", "community_small", "soft-refine", 42, tmp_path / "runs")
    artifacts = wrapper.train(TrainRequest(run, dataset, options=options))

    refiner = {
        "enabled": True,
        "target_source": "joint_structure_summary_head",
        "graphlet_k_min": 3,
        "graphlet_k_max": 5,
        "graphlet_connected_only": True,
        "graphlet_topology_filter": "all",
        "refiner": {
            "steps": 1,
            "proposal_budget": 16,
            "valid_candidate_budget": 8,
            "preserve_connectivity": True,
            "selection": "softmax_safe",
            "graphlet_weight": 1.0,
            "graphlet_mass_weight": 0.1,
            "clustering_weight": 0.25,
            "orbit_weight": 0.25,
            "accept_only_improving": False,
            "soft_selection": {
                "temperature_start": 0.5,
                "temperature_end": 0.05,
                "temperature_schedule": "linear",
                "score_normalization": "std",
                "stop_action": True,
                "max_target_worsening": 0.02,
                "max_relative_target_worsening": 0.10,
            },
            "motif_guard": {"enabled": False},
            "refresh_prediction_every": 1,
            "reject_revisited_states": True,
        },
        "disconnected_source_policy": "skip",
    }
    generated = wrapper.generate(
        GenerateRequest(
            run,
            artifacts.checkpoint_path,
            2,
            91,
            generation_id="soft",
            options={"graphlet_refinement": refiner},
        )
    )
    import json
    manifest = json.loads(generated.manifest_path.read_text())
    assert manifest["sampling"]["rewiring"] == "soft_structural_summary_guided_degree_preserving_double_edge_swaps"
    assert manifest["diagnostics"]["degree_preservation_rate"] == 1.0
    assert (generated.generation_dir / "pre_refinement_graphs.pkl").is_file()

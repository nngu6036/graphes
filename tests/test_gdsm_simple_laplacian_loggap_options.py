from __future__ import annotations

import json
import pickle

import networkx as nx
import torch

from grapher.models.base import DatasetReference, GenerateRequest, RunSpec, TrainRequest
from grapher.models.gdsm_simple.graphlet_stage3 import (
    default_fixed_target_graphlet_refinement_options,
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


def _base_loggap_options() -> dict:
    return {
        "variant": "vanilla_laplacian_loggap",
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
        "model": {"max_nodes": 6, "max_feat_num": 6, "hidden_dim": 8, "depth": 1},
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
        "graphlet_summary": {"enabled": False},
        "graphlet_refinement": {"enabled": False},
        "degree_prior": {"enabled": False},
        "generation_batch_size": 2,
        "runtime": {"device": "cpu"},
        "extensions": {
            "degree_conditioning": False,
            "hh_initialization": False,
            "degree_preserving_rewiring": False,
            "structural_summary": "none",
        },
    }


def _joint_refine_options() -> dict:
    options = _base_loggap_options()
    options["variant"] = "vanilla_laplacian_loggap_graphlet_refine"
    options["graphlet_summary"] = {
        "enabled": True,
        "orders": [3, 4, 5],
        "loss_weight": 0.10,
        "histogram_weight": 1.0,
        "mass_weight": 0.25,
    }
    refine = default_fixed_target_graphlet_refinement_options()
    refine["refiner"].update(
        {
            "steps": 1,
            "proposal_budget": 16,
            "valid_candidate_budget": 8,
            "refresh_prediction_every": 1,
        }
    )
    options["graphlet_refinement"] = refine
    return options


def test_loggap_without_graphlet_auxiliary_head(tmp_path):
    torch.set_num_threads(1)
    dataset = _write_dataset(tmp_path)
    wrapper = GDSMSimpleWrapper()
    run = RunSpec("gdsm_simple", "community_small", "loggap-only", 41, tmp_path / "runs")
    artifacts = wrapper.train(TrainRequest(run, dataset, options=_base_loggap_options()))
    state = torch.load(artifacts.checkpoint_path, map_location="cpu", weights_only=False)
    assert state["variant"] == "vanilla_laplacian_loggap"
    assert state["spectral_transform"]["kind"] == "laplacian_log_gap"
    assert state["graphlet_summary"]["enabled"] is False
    assert not any("graphlet" in key for key in state["model_spectrum_state"])

    generated = wrapper.generate(
        GenerateRequest(run, artifacts.checkpoint_path, 2, 91, generation_id="s")
    )
    manifest = json.loads((generated.generation_dir / "manifest.json").read_text())
    assert manifest["sampling"]["spectral_parameterization"] == "laplacian_log_gap"
    assert manifest["predicted_graphlet_summaries"] is None
    assert manifest["diagnostics"]["active_negative_eigenvalue_fraction"] == 0.0
    assert manifest["diagnostics"]["nondecreasing_violation_fraction"] == 0.0


def test_joint_auxiliary_head_plus_rewiring_freezes_generated_degrees(tmp_path):
    torch.set_num_threads(1)
    dataset = _write_dataset(tmp_path)
    wrapper = GDSMSimpleWrapper()
    run = RunSpec("gdsm_simple", "community_small", "loggap-joint-refine", 42, tmp_path / "runs")
    artifacts = wrapper.train(TrainRequest(run, dataset, options=_joint_refine_options()))
    state = torch.load(artifacts.checkpoint_path, map_location="cpu", weights_only=False)
    assert state["variant"] == "vanilla_laplacian_loggap_graphlet_refine"
    assert state["graphlet_summary"]["enabled"] is True
    assert any("graphlet" in key for key in state["model_spectrum_state"])

    generated = wrapper.generate(
        GenerateRequest(run, artifacts.checkpoint_path, 3, 92, generation_id="s")
    )
    with (generated.generation_dir / "pre_refinement_graphs.pkl").open("rb") as handle:
        sources = pickle.load(handle)
    with generated.graphs_path.open("rb") as handle:
        finals = pickle.load(handle)
    with (generated.generation_dir / "frozen_degree_sequences.pkl").open("rb") as handle:
        frozen = pickle.load(handle)
    assert len(sources) == len(finals) == len(frozen) == 3
    for source, final, target_degree in zip(sources, finals, frozen):
        source_degree = [source.degree(v) for v in sorted(source.nodes())]
        final_degree = [final.degree(v) for v in sorted(final.nodes())]
        assert source_degree == target_degree
        assert final_degree == target_degree
        if nx.is_connected(source):
            assert nx.is_connected(final)

    manifest = json.loads((generated.generation_dir / "manifest.json").read_text())
    assert manifest["sampling"]["rewiring"] == "graphlet_guided_degree_preserving_double_edge_swaps"
    assert manifest["sampling"]["structural_guidance"] == "fixed_joint_auxiliary_graphlet_summary_k3_k4_k5"
    assert manifest["diagnostics"]["degree_preservation_rate"] == 1.0
    assert (generated.generation_dir / "predicted_graphlet_summaries.pkl").is_file()
    assert (generated.generation_dir / "rewiring_diagnostics.json").is_file()

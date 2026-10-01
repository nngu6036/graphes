from __future__ import annotations

import json
import pickle

import networkx as nx
import torch

from grapher.models.base import GenerateRequest, RunSpec, TrainRequest
from grapher.models.gdsm_simple.graphlet_stage3 import (
    default_graphlet_refinement_options,
    resolve_graphlet_refinement_options,
)
from grapher.models.gdsm_simple.wrapper import GDSMSimpleWrapper


def vanilla_options() -> dict:
    return {
        "variant": "vanilla_gsdm",
        "train": {
            "epochs": 2, "batch_size": 4, "lr": 1.0e-3, "weight_decay": 0.0,
            "grad_norm": 1.0, "lr_schedule": False, "lr_decay": 1.0, "ema": 0.9,
            "validation_every": 1, "log_every": 1,
        },
        "model": {"max_nodes": 6, "max_feat_num": 6, "hidden_dim": 8, "depth": 2},
        "sde": {
            "x": {"type": "vp", "beta_min": 0.1, "beta_max": 1.0, "num_scales": 4},
            "spectrum": {"type": "vp", "beta_min": 0.1, "beta_max": 1.0, "num_scales": 4},
            "eps": 1.0e-5, "eigen_mask": "official_extremes",
        },
        "sample": {
            "predictor": "euler", "corrector": "none", "snr": 0.05, "scale_eps": 0.7,
            "n_steps": 0, "noise_removal": True, "probability_flow": False,
            "eps": 1.0e-3, "threshold": 0.5, "use_ema": False,
        },
        "generation_batch_size": 2,
        "runtime": {"device": "cpu"},
        "extensions": {
            "degree_conditioning": False, "hh_initialization": False,
            "degree_preserving_rewiring": False, "structural_summary": "none",
        },
    }


def _write_dataset(root):
    from grapher.models.base import DatasetReference
    folder = root / "datasets" / "toy"
    folder.mkdir(parents=True)
    train = [
        nx.cycle_graph(5), nx.path_graph(5), nx.star_graph(4),
        nx.complete_bipartite_graph(2, 3),
    ]
    val = [nx.cycle_graph(4), nx.path_graph(4)]
    test = [nx.wheel_graph(5)]
    for split, graphs in (("train", train), ("val", val), ("test", test)):
        with (folder / f"{split}.pkl").open("wb") as handle:
            pickle.dump(graphs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return DatasetReference("community_small", root / "datasets", "toy"), train


def stage3_options(*, steps: int = 2) -> dict:
    options = vanilla_options()
    options["variant"] = "vanilla_gsdm_graphlet_refine"
    cfg = default_graphlet_refinement_options()
    cfg["predictor"].update(
        {
            "epochs": 2,
            "batch_size": 4,
            "validation_every": 1,
            "early_stopping_patience": 0,
            "hidden_dim": 16,
            "edge_dim": 8,
            "graph_dim": 16,
            "num_layers": 2,
        }
    )
    cfg["predictor"]["corruption"].update(
        {
            "trajectories_per_graph": 1,
            "states_per_trajectory": 2,
            "max_swaps": 2,
            "proposal_budget": 8,
            "valid_candidate_budget": 4,
        }
    )
    cfg["refiner"].update(
        {
            "steps": steps,
            "proposal_budget": 16,
            "valid_candidate_budget": 8,
            "refresh_prediction_every": 1,
        }
    )
    options["graphlet_refinement"] = cfg
    return options


def test_stage3_config_is_graphlet_only_and_fixed_to_345():
    cfg = resolve_graphlet_refinement_options(default_graphlet_refinement_options())
    assert (cfg["graphlet_k_min"], cfg["graphlet_k_max"]) == (3, 5)
    assert cfg["refiner"]["graphlet_weight"] > 0
    assert cfg["refiner"]["clustering_weight"] == 0
    assert cfg["refiner"]["orbit_weight"] == 0


def test_stage3_predictor_training_does_not_change_vanilla_gsdm_parameters(tmp_path):
    torch.set_num_threads(1)
    dataset, _ = _write_dataset(tmp_path)
    wrapper = GDSMSimpleWrapper()
    vanilla_run = RunSpec("gdsm_simple", "community_small", "vanilla", 55, tmp_path / "runs")
    stage3_run = RunSpec("gdsm_simple", "community_small", "stage3", 55, tmp_path / "runs")
    vanilla_art = wrapper.train(TrainRequest(vanilla_run, dataset, options=vanilla_options()))
    stage3_art = wrapper.train(TrainRequest(stage3_run, dataset, options=stage3_options(steps=0)))
    vanilla = torch.load(vanilla_art.checkpoint_path, map_location="cpu", weights_only=False)
    stage3 = torch.load(stage3_art.checkpoint_path, map_location="cpu", weights_only=False)
    for key in ("model_x_state", "model_spectrum_state"):
        for name in vanilla[key]:
            torch.testing.assert_close(vanilla[key][name], stage3[key][name], rtol=0, atol=0)
    assert stage3["graphlet_refinement"]["enabled"] is True
    assert stage3["graphlet_refinement"]["graphlet_basis"]["keys_by_k"].keys() == {"3", "4", "5"}


def test_stage3_zero_step_control_reproduces_vanilla_graphs_and_freezes_degrees(tmp_path):
    torch.set_num_threads(1)
    dataset, _ = _write_dataset(tmp_path)
    wrapper = GDSMSimpleWrapper()
    vanilla_run = RunSpec("gdsm_simple", "community_small", "vanilla", 66, tmp_path / "runs")
    stage3_run = RunSpec("gdsm_simple", "community_small", "stage3", 66, tmp_path / "runs")
    vanilla_art = wrapper.train(TrainRequest(vanilla_run, dataset, options=vanilla_options()))
    stage3_art = wrapper.train(TrainRequest(stage3_run, dataset, options=stage3_options(steps=0)))
    vanilla_gen = wrapper.generate(GenerateRequest(vanilla_run, vanilla_art.checkpoint_path, 4, 101, generation_id="s"))
    stage3_gen = wrapper.generate(GenerateRequest(stage3_run, stage3_art.checkpoint_path, 4, 101, generation_id="s"))
    with vanilla_gen.graphs_path.open("rb") as f:
        vanilla_graphs = pickle.load(f)
    with stage3_gen.graphs_path.open("rb") as f:
        stage3_graphs = pickle.load(f)
    assert len(vanilla_graphs) == len(stage3_graphs)
    for a, b in zip(vanilla_graphs, stage3_graphs):
        assert a.number_of_nodes() == b.number_of_nodes()
        assert set(a.edges()) == set(b.edges())
    with (stage3_gen.generation_dir / "frozen_degree_sequences.pkl").open("rb") as f:
        frozen = pickle.load(f)
    with stage3_gen.graphs_path.open("rb") as f:
        graphs = pickle.load(f)
    assert [list(dict(g.degree()).values()) for g in graphs] == frozen
    manifest = json.loads((stage3_gen.generation_dir / "manifest.json").read_text())
    assert manifest["diagnostics"]["degree_preservation_rate"] == 1.0
    assert manifest["diagnostics"]["changed_rate"] == 0.0


def test_stage3_rewiring_preserves_indexed_degrees(tmp_path):
    torch.set_num_threads(1)
    dataset, _ = _write_dataset(tmp_path)
    wrapper = GDSMSimpleWrapper()
    run = RunSpec("gdsm_simple", "community_small", "stage3-refine", 77, tmp_path / "runs")
    artifacts = wrapper.train(TrainRequest(run, dataset, options=stage3_options(steps=2)))
    generated = wrapper.generate(GenerateRequest(run, artifacts.checkpoint_path, 5, 202, generation_id="s"))
    with (generated.generation_dir / "vanilla_graphs.pkl").open("rb") as f:
        sources = pickle.load(f)
    with generated.graphs_path.open("rb") as f:
        finals = pickle.load(f)
    assert len(sources) == len(finals) == 5
    for source, final in zip(sources, finals):
        assert [source.degree(v) for v in sorted(source.nodes())] == [final.degree(v) for v in sorted(final.nodes())]
        if nx.is_connected(source):
            assert nx.is_connected(final)

from __future__ import annotations

import json
import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import torch

from grapher.models.base import DatasetReference, GenerateRequest, RunSpec, TrainRequest
from grapher.models.grapher.degree_constrained_realization import (
    default_config,
    realize_graph,
)
from grapher.models.grapher.wrapper import GraphERWrapper


def _laplacian(graph: nx.Graph) -> np.ndarray:
    adjacency = nx.to_numpy_array(graph, nodelist=range(graph.number_of_nodes()))
    return np.diag(adjacency.sum(axis=1)) - adjacency


def test_gumbel_milp_realization_preserves_exact_degrees_and_connectivity() -> None:
    source = nx.cycle_graph(8)
    lap = _laplacian(source)
    scores = -lap.copy()
    np.fill_diagonal(scores, 0.0)

    config = default_config()
    config.update(
        enabled=True,
        gumbel_temperature=2.0,
        solver_time_limit_seconds=10.0,
        require_connected=True,
    )

    realized, threshold, target_degree, diagnostics = realize_graph(
        lap,
        scores,
        threshold=0.5,
        rng=np.random.default_rng(3),
        config=config,
    )

    assert target_degree == [2] * 8
    assert [realized.degree(i) for i in range(8)] == target_degree
    assert nx.is_connected(realized)
    assert diagnostics["degree_exact"] is True
    assert diagnostics["connected"] is True
    assert diagnostics["fallback_used"] is False
    assert realized.number_of_edges() == threshold.number_of_edges() == 8
    # At high temperature the fixed-degree decoder can select an alternative
    # realization while retaining the same indexed degrees.
    assert diagnostics["changed_from_threshold"] is True


def test_laplacian_diagonal_projection_has_trace_consistent_degree_sum() -> None:
    source = nx.path_graph(7)
    lap = _laplacian(source)
    # Continuous perturbation keeps a valid row-sum-zero symmetric operator but
    # makes the diagonal nonintegral.
    noise = np.array([0.12, -0.08, 0.05, -0.04, 0.03, -0.02, -0.06])
    lap = lap + np.diag(noise)
    lap[0, 1] -= noise[0]
    lap[1, 0] -= noise[0]
    scores = -0.5 * (lap + lap.T)
    np.fill_diagonal(scores, 0.0)

    config = default_config()
    config.update(enabled=True, gumbel_temperature=0.25, solver_time_limit_seconds=10.0)
    realized, _, target_degree, diagnostics = realize_graph(
        lap,
        scores,
        threshold=0.5,
        rng=np.random.default_rng(19),
        config=config,
    )

    assert sum(target_degree) == 2 * realized.number_of_edges()
    assert diagnostics["degree_exact"] is True
    assert diagnostics["projected_degree_graphical"] is True
    assert nx.is_connected(realized)


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
            "max_nodes": 6,
            "max_feat_num": 6,
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
            "random_walk": {"enabled": True, "steps": 2},
            "shortest_path": {"enabled": True, "max_distance": 3},
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
        "structure_summary": {
            "enabled": True,
            "loss_weight": 0.1,
            "graphlet": {
                "enabled": True,
                "orders": [3, 4, 5],
                "histogram_weight": 1.0,
                "mass_weight": 0.25,
            },
            "clustering": {
                "enabled": False,
                "bins": 20,
                "histogram_weight": 0.25,
                "cdf_weight": 1.0,
            },
            "orbit": {
                "enabled": True,
                "width": 15,
                "histogram_weight": 0.25,
                "log_total_weight": 0.1,
            },
            "degree": {
                "enabled": True,
                "max_degree": 5,
                "histogram_weight": 0.25,
                "cdf_weight": 1.0,
            },
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
            "eps": 1e-2,
            "threshold": 0.5,
            "use_ema": False,
            "degree_constrained_realization": {
                "enabled": True,
                "gumbel_temperature": 0.5,
                "solver_time_limit_seconds": 10.0,
                "failure_policy": "error",
            },
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


def _load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def test_generation_writes_paired_threshold_control_and_realization_diagnostics(tmp_path: Path) -> None:
    torch.set_num_threads(1)
    dataset = _write_dataset(tmp_path)
    wrapper = GraphERWrapper()
    run = RunSpec("grapher", "community_small", "degree-realization", 42, tmp_path / "runs")
    trained = wrapper.train(TrainRequest(run, dataset, options=_options()))
    generated = wrapper.generate(
        GenerateRequest(
            run,
            trained.checkpoint_path,
            4,
            99,
            generation_id="stochastic-realization",
            options={"graphlet_refinement": {"enabled": False}},
        )
    )

    final_graphs = _load_pickle(generated.generation_dir / "base_graphs.pkl")
    threshold_graphs = _load_pickle(
        generated.generation_dir / "pre_realization_graphs.pkl"
    )
    target_degrees = _load_pickle(
        generated.generation_dir / "projected_realization_degree_sequences.pkl"
    )
    assert len(final_graphs) == len(threshold_graphs) == len(target_degrees) == 4
    for graph, degree in zip(final_graphs, target_degrees):
        assert [graph.degree(i) for i in range(graph.number_of_nodes())] == degree
        assert nx.is_connected(graph)

    report_path = generated.generation_dir / "degree_constrained_realization_diagnostics.json"
    report = json.loads(report_path.read_text())
    assert report["aggregate"]["enabled"] is True
    assert report["aggregate"]["degree_exact_rate"] == 1.0
    assert report["aggregate"]["connectedness_rate"] == 1.0
    assert len(report["per_graph"]) == 4

    manifest = json.loads(generated.manifest_path.read_text())
    assert manifest["sampling"]["topology_decoder"] == (
        "stochastic_degree_constrained_spectral_realization"
    )
    assert manifest["sampling"]["degree_constrained_realization"]["enabled"] is True
    assert manifest["sampling"]["rewiring"] is False


def test_generation_can_select_among_multiple_stochastic_realizations_with_summary_heads(tmp_path: Path) -> None:
    torch.set_num_threads(1)
    dataset = _write_dataset(tmp_path)
    options = _options()
    options["sample"]["degree_constrained_realization"]["candidate_selection"] = {
        "enabled": True,
        "num_candidates": 3,
        "selection": "best",
        "selection_temperature": 0.10,
        "graphlet_weight": 1.0,
        "graphlet_mass_weight": 0.25,
        "clustering_weight": 0.0,
        "orbit_weight": 0.25,
    }

    wrapper = GraphERWrapper()
    run = RunSpec("grapher", "community_small", "summary-selection", 42, tmp_path / "runs")
    trained = wrapper.train(TrainRequest(run, dataset, options=options))
    generated = wrapper.generate(
        GenerateRequest(
            run,
            trained.checkpoint_path,
            3,
            123,
            generation_id="summary-selected-realization",
            options={"graphlet_refinement": {"enabled": False}},
        )
    )

    report = json.loads(
        (generated.generation_dir / "degree_constrained_realization_diagnostics.json").read_text()
    )
    aggregate = report["aggregate"]
    assert aggregate["candidate_selection_enabled"] is True
    assert aggregate["candidate_selection_num_candidates"] == 3
    assert aggregate["candidate_selection_mode"] == "best"
    assert aggregate["candidate_selection_selected_best_rate"] == 1.0
    assert aggregate["candidate_selection_mean_selected_summary_energy"] is not None
    assert len(report["per_graph"]) == 3
    for row in report["per_graph"]:
        selection = row["candidate_selection"]
        assert selection["enabled"] is True
        assert selection["num_candidates"] == 3
        assert selection["selected_is_best"] is True
        assert len(selection["candidates"]) == 3

    manifest = json.loads(generated.manifest_path.read_text())
    assert manifest["sampling"]["topology_decoder"] == (
        "stochastic_degree_constrained_summary_selected_realization"
    )
    assert manifest["sampling"]["degree_constrained_realization"][
        "candidate_selection_enabled"
    ] is True
    assert manifest["sampling"]["rewiring"] is False

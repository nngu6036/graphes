from __future__ import annotations

import json
import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import torch

from grapher.models.base import DatasetReference, GenerateRequest, RunSpec, TrainRequest
from grapher.models.gdsm_simple.vanilla_gsdm import (
    CHECKPOINT_FORMAT,
    _effective_eigen_mask,
    _fit_laplacian_log_gap_stats,
    _laplacian_eigenvalues_to_log_gap_state,
    _log_gap_state_to_laplacian_eigenvalues,
    _sample_fixed_k_topology,
    validate_options,
)
from grapher.models.gdsm_simple.wrapper import GDSMSimpleWrapper


def test_fixed_k_mask_and_loggap_tail_zero() -> None:
    flags = torch.tensor(
        [[1, 1, 1, 1, 1, 0], [1, 1, 1, 0, 0, 0]], dtype=torch.float32
    )
    transform = {
        "kind": "laplacian_log_gap",
        "spectral_cutoff": {
            "enabled": True,
            "k": 2,
            "selection": "low_frequency_nonzero",
            "tail_policy": "zero",
        },
    }
    mask = _effective_eigen_mask(flags, "laplacian_nonzero_prefix", transform)
    torch.testing.assert_close(
        mask,
        torch.tensor(
            [[0, 1, 1, 0, 0, 0], [0, 1, 1, 0, 0, 0]], dtype=torch.float32
        ),
    )

    eig = torch.tensor(
        [[0.0, 0.4, 1.0, 2.2, 3.0, 0.0], [0.0, 0.2, 1.7, 0.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    stats = _fit_laplacian_log_gap_stats(eig, flags, epsilon=1e-6, min_std=1e-3)
    cutoff = transform["spectral_cutoff"]
    state = _laplacian_eigenvalues_to_log_gap_state(
        eig,
        flags,
        mean=stats["mean"],
        std=stats["std"],
        epsilon=1e-6,
        spectral_cutoff=cutoff,
    )
    assert torch.equal(state[:, 3:], torch.zeros_like(state[:, 3:]))
    recovered = _log_gap_state_to_laplacian_eigenvalues(
        state,
        flags,
        mean=stats["mean"],
        std=stats["std"],
        epsilon=1e-6,
        exp_clip=20.0,
        spectral_cutoff=cutoff,
    )
    torch.testing.assert_close(recovered[0, 1:3], eig[0, 1:3], atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(recovered[1, 1:3], eig[1, 1:3], atol=2e-5, rtol=2e-5)
    assert torch.equal(recovered[:, 3:], torch.zeros_like(recovered[:, 3:]))


def test_stochastic_degree_budget_decoder_is_connected_and_reproducible() -> None:
    scores = np.array(
        [
            [0.0, 0.9, 0.2, 0.1, 0.3],
            [0.9, 0.0, 0.8, 0.2, 0.1],
            [0.2, 0.8, 0.0, 0.7, 0.2],
            [0.1, 0.2, 0.7, 0.0, 0.6],
            [0.3, 0.1, 0.2, 0.6, 0.0],
        ],
        dtype=np.float64,
    )
    # Mean degree 2.0 -> five edges for five nodes.
    degree_hist = np.array([0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float64)
    graph1, diag1 = _sample_fixed_k_topology(
        scores,
        degree_hist,
        seed=123,
        temperature=0.5,
        require_connected=True,
    )
    graph2, diag2 = _sample_fixed_k_topology(
        scores,
        degree_hist,
        seed=123,
        temperature=0.5,
        require_connected=True,
    )
    assert nx.is_connected(graph1)
    assert graph1.number_of_edges() == 5
    assert set(graph1.edges()) == set(graph2.edges())
    assert diag1 == diag2


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
            "spectral_cutoff": {
                "enabled": True,
                "k": 2,
                "selection": "low_frequency_nonzero",
                "tail_policy": "zero",
            },
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
            "clustering": {"enabled": False, "bins": 20},
            "orbit": {
                "enabled": True,
                "width": 15,
                "histogram_weight": 0.25,
                "log_total_weight": 0.1,
            },
            "degree": {
                "enabled": True,
                "max_degree": 6,
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
            "fixed_k_decoder": {
                "mode": "stochastic_degree_budget",
                "temperature": 0.5,
                "require_connected": True,
                "edge_count_source": "predicted_degree_histogram",
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


def test_fixed_k_train_generate_contract(tmp_path: Path) -> None:
    torch.set_num_threads(1)
    options = _options()
    validate_options(options)
    wrapper = GDSMSimpleWrapper()
    dataset = _write_dataset(tmp_path)
    run = RunSpec("gdsm_simple", "community_small", "fixed-k", 42, tmp_path / "runs")
    artifacts = wrapper.train(TrainRequest(run, dataset, options=options))
    state = torch.load(artifacts.checkpoint_path, map_location="cpu", weights_only=False)
    assert state["format"] == CHECKPOINT_FORMAT
    assert state["spectral_cutoff"] == {
        "enabled": True,
        "k": 2,
        "selection": "low_frequency_nonzero",
        "tail_policy": "zero",
    }

    generated = wrapper.generate(
        GenerateRequest(run, artifacts.checkpoint_path, 3, 99, generation_id="fixed-k")
    )
    with generated.graphs_path.open("rb") as handle:
        graphs = pickle.load(handle)
    with (generated.generation_dir / "sampled_spectra.pkl").open("rb") as handle:
        spectra = pickle.load(handle)
    assert len(graphs) == len(spectra) == 3
    assert all(nx.is_connected(graph) for graph in graphs)
    for graph, spectrum in zip(graphs, spectra):
        n = graph.number_of_nodes()
        values = np.asarray(spectrum, dtype=np.float64)
        assert np.allclose(values[3:], 0.0)
        assert np.all(values[1 : min(3, n)] >= 0.0)

    diagnostics = json.loads(
        (generated.generation_dir / "fixed_k_decoder_diagnostics.json").read_text()
    )
    assert diagnostics["aggregate"]["requested_k"] == 2
    assert diagnostics["aggregate"]["connectedness_rate"] == 1.0
    assert len(diagnostics["per_graph"]) == 3
    manifest = json.loads((generated.generation_dir / "manifest.json").read_text())
    assert manifest["sampling"]["spectral_cutoff"]["k"] == 2
    assert manifest["sampling"]["topology_decoder"]["mode"] == "stochastic_degree_budget"
    assert manifest["sampling"]["rewiring"] is False


def test_shipped_fixed_k_configs_resolve(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    wrapper = GDSMSimpleWrapper()
    for dataset in ("community_small", "ego_small"):
        for k in (2, 4, 6, 8, 12, "full"):
            for seed in (42, 43, 44):
                config = (
                    root
                    / "configs/experiments/gdsm_laplacian_loggap_topology_degree_g345_fixed_k_explicit"
                    / f"{dataset}_k{k}_seed_{seed}.yaml"
                )
                request = TrainRequest(
                    RunSpec("gdsm_simple", dataset, "cfg", seed, tmp_path / "runs"),
                    DatasetReference(dataset, tmp_path / "datasets", "unused"),
                    config_path=config,
                )
                options = wrapper._options(request)
                validate_options(options)
                assert options["sde"]["spectral_cutoff"]["enabled"] is True
                expected_k = ({"community_small": 19, "ego_small": 17}[dataset] if k == "full" else k)
                assert options["sde"]["spectral_cutoff"]["k"] == expected_k
                assert options["sample"]["fixed_k_decoder"]["mode"] == "stochastic_degree_budget"
                assert options["structure_summary"]["graphlet"]["orders"] == [3, 4, 5]
                assert options["structure_summary"]["degree"]["enabled"] is True
                assert options["graphlet_refinement"]["enabled"] is False

from __future__ import annotations

import json
import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import torch

from grapher.models.base import DatasetReference, GenerateRequest, RunSpec, TrainRequest
from grapher.models.gdsm_simple.vanilla_gsdm import (
    _complete_conditional_spectral_tail,
    _fit_conditional_spectral_tail_model,
    _fit_laplacian_log_gap_stats,
    _operator_eigenvalues_to_spectral_state,
    validate_options,
)
from grapher.models.gdsm_simple.wrapper import GDSMSimpleWrapper


def _toy_spectra() -> tuple[torch.Tensor, torch.Tensor]:
    # Monotone connected-Laplacian spectra with padded final coordinates.
    eig = torch.tensor(
        [
            [0.0, 0.2, 0.7, 1.3, 2.2, 3.1, 0.0],
            [0.0, 0.3, 0.9, 1.7, 2.8, 3.8, 0.0],
            [0.0, 0.15, 0.65, 1.4, 2.5, 3.6, 4.2],
            [0.0, 0.25, 0.8, 1.6, 2.6, 3.7, 4.4],
        ],
        dtype=torch.float32,
    )
    flags = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.float32,
    )
    return eig, flags


def test_conditional_tail_fit_and_completion_is_nonzero_and_ordered() -> None:
    eig, flags = _toy_spectra()
    stats = _fit_laplacian_log_gap_stats(eig, flags, epsilon=1e-6, min_std=1e-3)
    transform = {
        "kind": "laplacian_log_gap",
        "epsilon": 1e-6,
        "exp_clip": 20.0,
        "mean": stats["mean"],
        "std": stats["std"],
        "count": stats["count"],
        "spectral_cutoff": {
            "enabled": True,
            "k": 2,
            "selection": "low_frequency_nonzero",
            "tail_policy": "conditional_ridge_gaussian",
            "tail_ridge": 1e-2,
            "tail_min_std": 0.05,
            "tail_temperature": 1.0,
            "tail_condition_on_size": True,
        },
    }
    payload = _fit_conditional_spectral_tail_model(eig, flags, transform)
    assert payload["enabled"] is True
    assert payload["prefix_width"] == 2
    transform["spectral_tail_model"] = payload

    prefix_state = _operator_eigenvalues_to_spectral_state(
        eig,
        flags,
        spectral_operator="combinatorial_laplacian",
        spectral_transform=transform,
    )
    assert torch.count_nonzero(prefix_state[:, 3:]) == 0
    gen = torch.Generator(device="cpu").manual_seed(123)
    full_state, completed, diag = _complete_conditional_spectral_tail(
        prefix_state, flags, transform, generator=gen
    )
    assert diag["enabled"] is True
    assert torch.count_nonzero(full_state[:, 3:]) > 0
    for row, n_raw in enumerate(flags.sum(dim=1).tolist()):
        n = int(n_raw)
        active = completed[row, :n]
        assert abs(float(active[0])) < 1e-6
        assert torch.all(active[1:] >= -1e-7)
        assert torch.all(torch.diff(active) >= -1e-7)


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
                "tail_policy": "conditional_ridge_gaussian",
                "tail_ridge": 1e-2,
                "tail_min_std": 0.05,
                "tail_temperature": 1.0,
                "tail_condition_on_size": True,
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
                "mode": "threshold",
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


def test_conditional_tail_train_generate_uses_threshold_decoder(tmp_path: Path) -> None:
    torch.set_num_threads(1)
    options = _options()
    validate_options(options)
    wrapper = GDSMSimpleWrapper()
    dataset = _write_dataset(tmp_path)
    run = RunSpec("gdsm_simple", "community_small", "tail", 42, tmp_path / "runs")
    artifacts = wrapper.train(TrainRequest(run, dataset, options=options))
    state = torch.load(artifacts.checkpoint_path, map_location="cpu", weights_only=False)
    tail = state["spectral_transform"]["spectral_tail_model"]
    assert tail["enabled"] is True
    assert tail["format"] == "conditional_ridge_gaussian_loggap_tail_v1"

    generated = wrapper.generate(
        GenerateRequest(run, artifacts.checkpoint_path, 3, 99, generation_id="tail")
    )
    with (generated.generation_dir / "sampled_spectra.pkl").open("rb") as handle:
        spectra = pickle.load(handle)
    # At least one generated graph has a non-zero tail coordinate beyond k=2.
    assert any(np.any(np.asarray(values)[3:] > 1e-8) for values in spectra)
    manifest = json.loads((generated.generation_dir / "manifest.json").read_text())
    assert manifest["sampling"]["spectral_cutoff"]["tail_policy"] == "conditional_ridge_gaussian"
    assert manifest["sampling"]["topology_decoder"]["mode"] == "threshold"
    assert "conditional_ridge_gaussian" in manifest["sampling"]["high_frequency_tail"]


def test_shipped_conditional_tail_configs_resolve(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    wrapper = GDSMSimpleWrapper()
    folder = root / "configs/experiments/gdsm_laplacian_loggap_topology_degree_g345_fixed_k_tail_explicit"
    for dataset in ("community_small", "ego_small"):
        for k in (2, 4, 6, 8, 12, "full"):
            for seed in (42, 43, 44):
                config = folder / f"{dataset}_k{k}_seed_{seed}.yaml"
                request = TrainRequest(
                    RunSpec("gdsm_simple", dataset, "cfg", seed, tmp_path / "runs"),
                    DatasetReference(dataset, tmp_path / "datasets", "unused"),
                    config_path=config,
                )
                options = wrapper._options(request)
                validate_options(options)
                cutoff = options["sde"]["spectral_cutoff"]
                assert cutoff["tail_policy"] == "conditional_ridge_gaussian"
                assert options["sample"]["fixed_k_decoder"]["mode"] == "threshold"
                assert options["structure_summary"]["graphlet"]["orders"] == [3, 4, 5]


def test_full_k_tail_completion_is_noop_and_does_not_consume_rng() -> None:
    eig, flags = _toy_spectra()
    # Use max_nodes-1 so every active nontrivial coordinate is in the prefix.
    stats = _fit_laplacian_log_gap_stats(eig, flags, epsilon=1e-6, min_std=1e-3)
    transform = {
        "kind": "laplacian_log_gap",
        "epsilon": 1e-6,
        "exp_clip": 20.0,
        "mean": stats["mean"],
        "std": stats["std"],
        "count": stats["count"],
        "spectral_cutoff": {
            "enabled": True,
            "k": eig.size(1) - 1,
            "selection": "low_frequency_nonzero",
            "tail_policy": "conditional_ridge_gaussian",
            "tail_ridge": 1e-2,
            "tail_min_std": 0.05,
            "tail_temperature": 1.0,
            "tail_condition_on_size": True,
        },
    }
    transform["spectral_tail_model"] = _fit_conditional_spectral_tail_model(eig, flags, transform)
    prefix = _operator_eigenvalues_to_spectral_state(
        eig,
        flags,
        spectral_operator="combinatorial_laplacian",
        spectral_transform=transform,
    )
    gen = torch.Generator(device="cpu").manual_seed(999)
    before = gen.get_state().clone()
    full_state, completed, diag = _complete_conditional_spectral_tail(
        prefix, flags, transform, generator=gen
    )
    after = gen.get_state().clone()
    assert torch.equal(before, after)
    assert diag["active_tail_coordinates"] == 0
    torch.testing.assert_close(full_state, prefix)
    for row, n_raw in enumerate(flags.sum(dim=1).tolist()):
        n = int(n_raw)
        torch.testing.assert_close(completed[row, :n], eig[row, :n], atol=2e-5, rtol=2e-5)

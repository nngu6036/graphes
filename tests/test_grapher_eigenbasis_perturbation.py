from __future__ import annotations

import json
import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import torch

from grapher.models.base import DatasetReference, GenerateRequest, RunSpec, TrainRequest
from grapher.models.grapher.eigenbasis_perturbation import (
    perturb_laplacian_basis,
    resolve_config,
)
from grapher.models.grapher.vanilla_gsdm import _laplacian_eigh_padded, reconstruct_adjacency
from grapher.models.grapher.wrapper import GraphERWrapper


def test_givens_perturbation_preserves_zero_mode_orthogonality_and_spectrum():
    donor = torch.zeros((1, 6, 6), dtype=torch.float64)
    for i in range(6):
        donor[0, i, (i + 1) % 6] = 1.0
        donor[0, (i + 1) % 6, i] = 1.0
    sizes = torch.tensor([6], dtype=torch.long)
    eigenvalues, basis = _laplacian_eigh_padded(donor.float(), sizes)
    basis = basis.double()
    cfg = resolve_config({
        "eigenbasis_perturbation": {
            "enabled": True,
            "stage": "decode",
            "sigma": 0.08,
            "num_rotations": 4,
            "pairing": "adjacent",
            "max_abs_angle": 0.2,
        }
    })
    perturbed, rows = perturb_laplacian_basis(
        basis, sizes, config=cfg, rng=np.random.default_rng(123)
    )
    assert rows[0]["applied_rotations"] == 4
    assert rows[0]["active_frobenius_basis_change"] > 0
    assert torch.allclose(perturbed[:, :, 0], basis[:, :, 0], atol=1e-12)
    ident = torch.eye(6, dtype=torch.float64)
    assert torch.allclose(perturbed[0].T @ perturbed[0], ident, atol=2e-6)

    # Rotating eigenvectors changes the matrix but not its eigenvalue multiset.
    lam = torch.tensor([[0.0, 0.3, 0.8, 1.4, 2.2, 3.1]], dtype=torch.float64)
    before = reconstruct_adjacency(basis, lam)[0]
    after = reconstruct_adjacency(perturbed, lam)[0]
    assert not torch.allclose(before, after)
    assert torch.allclose(
        torch.linalg.eigvalsh(after), torch.sort(lam[0]).values, atol=2e-6
    )
    assert torch.allclose(after.sum(dim=-1), torch.zeros(6, dtype=torch.float64), atol=2e-6)


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
            "graphlet": {"enabled": True, "orders": [3, 4, 5], "histogram_weight": 1.0, "mass_weight": 0.25},
            "clustering": {"enabled": False, "bins": 20, "histogram_weight": 0.25, "cdf_weight": 1.0},
            "orbit": {"enabled": True, "width": 15, "histogram_weight": 0.25, "log_total_weight": 0.1},
            "degree": {"enabled": True, "max_degree": 5, "histogram_weight": 0.25, "cdf_weight": 1.0},
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


def test_reverse_and_decode_variants_generate_and_decode_is_paired(tmp_path):
    torch.set_num_threads(1)
    dataset = _write_dataset(tmp_path)
    wrapper = GraphERWrapper()
    run = RunSpec("grapher", "community_small", "basis", 42, tmp_path / "runs")
    artifacts = wrapper.train(TrainRequest(run, dataset, options=_options()))

    base = wrapper.generate(
        GenerateRequest(run, artifacts.checkpoint_path, 4, 99, generation_id="base")
    )
    perturb_common = {
        "enabled": True,
        "sigma": 0.08,
        "num_rotations": 3,
        "pairing": "adjacent",
        "max_abs_angle": 0.2,
    }
    decode = wrapper.generate(
        GenerateRequest(
            run,
            artifacts.checkpoint_path,
            4,
            99,
            generation_id="decode",
            options={
                "sample": {"eigenbasis_perturbation": {**perturb_common, "stage": "decode"}},
                "graphlet_refinement": {"enabled": False},
            },
        )
    )
    reverse = wrapper.generate(
        GenerateRequest(
            run,
            artifacts.checkpoint_path,
            4,
            99,
            generation_id="reverse",
            options={
                "sample": {"eigenbasis_perturbation": {**perturb_common, "stage": "reverse"}},
                "graphlet_refinement": {"enabled": False},
            },
        )
    )

    # Decode-only perturbation must preserve donor choices and the generated
    # spectrum exactly because the reverse trajectory is untouched.
    assert _load_pickle(base.generation_dir / "sampled_basis_indices.pkl") == _load_pickle(
        decode.generation_dir / "sampled_basis_indices.pkl"
    )
    base_spectra = _load_pickle(base.generation_dir / "sampled_spectra.pkl")
    decode_spectra = _load_pickle(decode.generation_dir / "sampled_spectra.pkl")
    for left, right in zip(base_spectra, decode_spectra):
        assert np.array_equal(left, right)

    for generated, stage in ((decode, "decode"), (reverse, "reverse")):
        path = generated.generation_dir / "eigenbasis_perturbation_diagnostics.json"
        assert path.is_file()
        report = json.loads(path.read_text())
        assert report["aggregate"]["enabled"] is True
        assert report["aggregate"]["stage"] == stage
        assert report["aggregate"]["max_orthogonality_error_after"] < 1e-4
        assert report["aggregate"]["max_zero_mode_error_after"] < 1e-4
        assert len(report["per_graph"]) == 4
        assert all(row["active_frobenius_basis_change"] > 0 for row in report["per_graph"])
        manifest = json.loads(generated.manifest_path.read_text())
        assert manifest["sampling"]["eigenbasis_perturbation"]["stage"] == stage
        assert manifest["sampling"]["rewiring"] is False

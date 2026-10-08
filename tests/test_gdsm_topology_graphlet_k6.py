from __future__ import annotations

from pathlib import Path

import networkx as nx
import torch

from grapher.models.base import DatasetReference, RunSpec, TrainRequest
from grapher.models.gdsm_simple.vanilla_gsdm import (
    _joint_structure_targets,
    _resolved_structure_summary_config,
    validate_options,
)
from grapher.models.gdsm_simple.wrapper import GDSMSimpleWrapper


def _summary_k6() -> dict:
    return {
        "enabled": True,
        "loss_weight": 0.1,
        "graphlet": {
            "enabled": True,
            "orders": [3, 4, 5, 6],
            "histogram_weight": 1.0,
            "mass_weight": 0.25,
        },
        "clustering": {"enabled": False, "bins": 100},
        "orbit": {
            "enabled": True,
            "width": 15,
            "histogram_weight": 0.25,
            "log_total_weight": 0.1,
        },
    }


def test_k6_topology_targets_have_complete_basis() -> None:
    tensors, meta = _joint_structure_targets(
        [nx.cycle_graph(6), nx.complete_graph(6)],
        _summary_k6(),
    )
    histogram, mass, clustering, orbit_hist, orbit_total = tensors
    # 2 + 6 + 21 + 112 connected unlabeled graphlets for k=3..6.
    assert meta["width"] == 141
    assert meta["orders"] == [3, 4, 5, 6]
    assert histogram.shape == (2, 141)
    assert mass.shape == (2, 4)
    assert clustering.shape == (2, 100)
    assert orbit_hist.shape == (2, 15)
    assert orbit_total.shape == (2, 1)
    for start, stop in meta["graphlet_slices"]:
        assert torch.allclose(
            histogram[:, start:stop].sum(dim=-1),
            torch.ones(2),
            atol=1e-6,
        )


def test_k6_options_are_accepted() -> None:
    options = {
        "variant": "vanilla_laplacian_loggap_graphlet",
        "train": {},
        "model": {
            "max_nodes": 20,
            "max_feat_num": 20,
            "node_backbone": "dense_gcn",
            "spectrum_backbone": "ppgn",
        },
        "sde": {
            "x": {"type": "vp", "beta_min": 0.1, "beta_max": 0.2, "num_scales": 2},
            "spectrum": {"type": "vp", "beta_min": 0.1, "beta_max": 0.2, "num_scales": 2},
            "eigen_mask": "laplacian_nonzero_prefix",
            "spectral_parameterization": "laplacian_log_gap",
            "log_gap_epsilon": 1e-6,
            "log_gap_min_std": 1e-3,
            "log_gap_exp_clip": 8.0,
        },
        "sample": {},
        "degree_prior": {"enabled": False},
        "graphlet_refinement": {"enabled": False},
        "graphlet_summary": {"enabled": False},
        "structure_summary": _summary_k6(),
        "structural_features": {"enabled": True},
        "generation_batch_size": 8,
        "runtime": {"device": "cpu"},
        "extensions": {},
    }
    validate_options(options)
    resolved = _resolved_structure_summary_config(options)
    assert resolved["graphlet"]["orders"] == [3, 4, 5, 6]


def test_shipped_k6_configs_resolve() -> None:
    root = Path(__file__).resolve().parents[1]
    wrapper = GDSMSimpleWrapper()
    for dataset in ("community_small", "ego_small"):
        config = (
            root
            / "configs/experiments/gdsm_laplacian_loggap_topology_g346_explicit"
            / f"{dataset}_seed_42.yaml"
        )
        request = TrainRequest(
            RunSpec("gdsm_simple", dataset, "cfg", 42, root / "tmp_runs"),
            DatasetReference(dataset, root / "tmp_datasets", "unused"),
            config_path=config,
        )
        options = wrapper._options(request)
        assert options["structure_summary"]["graphlet"]["orders"] == [3, 4, 5, 6]
        assert options["structure_summary"]["clustering"]["enabled"] is False
        assert options["structure_summary"]["orbit"]["enabled"] is True
        assert options["graphlet_refinement"]["enabled"] is False

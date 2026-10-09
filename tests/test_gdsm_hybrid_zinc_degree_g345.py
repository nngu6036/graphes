from __future__ import annotations

from pathlib import Path
import pickle

import networkx as nx
import torch
import yaml

from grapher.models.gdsm_simple import hierarchical_attributed_loggap as hierarchical
from grapher.models.gdsm_simple.hybrid_attributed_loggap import (
    HybridAttributedSpectrumPPGNScore,
    validate_options,
)
from grapher.models.base import DatasetReference, GenerateRequest, RunSpec, TrainRequest
from grapher.models.gdsm_simple.wrapper import GDSMSimpleWrapper
from grapher.rewiring_mlp.attributed.data import GraphCategoryVocabulary


def _molecule(atoms, edges):
    graph = nx.Graph()
    for index, atomic_num in enumerate(atoms):
        graph.add_node(index, atomic_num=int(atomic_num))
    for u, v, bond in edges:
        graph.add_edge(int(u), int(v), bond_type=int(bond))
    return graph


def _tiny_options():
    root = Path(__file__).resolve().parents[1]
    config = root / (
        "configs/experiments/"
        "gdsm_laplacian_loggap_hybrid_degree_g345_zinc_explicit/"
        "zinc_seed_42.yaml"
    )
    data = yaml.safe_load(config.read_text())
    options = data["gdsm_simple"]
    options["model"]["max_nodes"] = 6
    options["model"]["max_feat_num"] = 6
    options["topology_summary"]["degree"]["max_degree"] = 5
    options["topology_summary"]["graphlet"]["num_samples"] = 8
    options["attribute_summary"]["typed_graphlet"]["num_samples"] = 8
    options["attribute_summary"]["typed_graphlet"]["max_basis_graphs"] = 3
    options["preprocess"]["num_workers"] = 0
    return options


def test_zinc_config_selects_degree_g345_hybrid():
    options = _tiny_options()
    validate_options(options)
    assert options["topology_summary"]["degree"]["enabled"] is True
    assert options["topology_summary"]["graphlet"]["orders"] == [3, 4, 5]
    assert options["topology_summary"]["graphlet"]["backend"] == "sampled"
    assert options["attribute_summary"]["typed_graphlet"]["orders"] == [3, 4, 5]
    assert options["attribute_summary"]["typed_graphlet"]["backend"] == "sampled"
    assert options["attributed"]["decode"]["constraint_mode"] == "none"


def test_degree_target_and_head_are_present():
    options = _tiny_options()
    graphs = [
        _molecule([6, 6, 8], [(0, 1, 1), (1, 2, 2)]),
        _molecule([6, 7, 16, 17], [(0, 1, 1), (1, 2, 1), (2, 3, 1)]),
        _molecule([6, 6, 6, 6], [(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 0, 1)]),
    ]
    vocab = GraphCategoryVocabulary.from_graphs(graphs, options["attributed"])
    targets, topology_basis, typed_basis, topology_meta, attribute_meta = (
        hierarchical._hierarchical_structure_targets(
            graphs, options, vocab, seed=42
        )
    )
    assert len(targets) == 9
    degree_target = targets[5]
    assert degree_target.shape == (3, 6)
    assert torch.allclose(degree_target.sum(-1), torch.ones(3))
    assert topology_meta["degree_enabled"] is True
    assert topology_meta["degree_bins"] == 6

    model = HybridAttributedSpectrumPPGNScore(
        max_feat_num=6,
        max_nodes=6,
        node_classes=vocab.num_node_categories,
        edge_classes=len(vocab.edge_values),
        topology_graphlet_slices=tuple(topology_basis.slices),
        typed_graphlet_slices=tuple(typed_basis.slices),
        structural_features=options["structural_features"],
        topology_clustering_bins=0,
        topology_orbit_width=15,
        topology_degree_bins=6,
        ppgn_hidden_dim=12,
        ppgn_depth=2,
    )
    x = torch.randn(2, 6, 6)
    adj = torch.zeros(2, 6, 6)
    adj[:, 0, 1] = adj[:, 1, 0] = 1
    flags = torch.ones(2, 6)
    eigenvectors = torch.eye(6).repeat(2, 1, 1)
    eigenvalues = torch.randn(2, 6)
    out = model.forward_topology(x, adj, flags, eigenvectors, eigenvalues)
    assert out["topology_degree_histogram_logits"].shape == (2, 6)
    means = model.topology_structure_means_from_outputs(out)
    assert means["degree_histogram"].shape == (2, 6)
    assert torch.allclose(means["degree_histogram"].sum(-1), torch.ones(2), atol=1e-6)


def test_zinc_like_train_generate_smoke(tmp_path):
    torch.set_num_threads(1)
    graphs = [
        _molecule([6, 6, 8], [(0, 1, 1), (1, 2, 2)]),
        _molecule([6, 7, 16, 17], [(0, 1, 1), (1, 2, 1), (2, 3, 1)]),
        _molecule([6, 15, 8, 6], [(0, 1, 1), (1, 2, 2), (1, 3, 1)]),
        _molecule([6, 6, 35], [(0, 1, 1), (1, 2, 1)]),
    ]
    folder = tmp_path / "datasets" / "toy_zinc"
    folder.mkdir(parents=True)
    for split, rows in (("train", graphs), ("val", graphs[:3]), ("test", graphs[1:])):
        with (folder / f"{split}.pkl").open("wb") as handle:
            pickle.dump(rows, handle, protocol=pickle.HIGHEST_PROTOCOL)

    options = _tiny_options()
    options["train"].update({
        "epochs": 1,
        "batch_size": 2,
        "lr": 1.0e-3,
        "node_lr": 1.0e-3,
        "spectrum_lr": 1.0e-3,
        "weight_decay": 0.0,
        "grad_norm": 1.0,
        "lr_schedule": False,
        "lr_decay": 1.0,
        "ema": 0.9,
        "validation_every": 1,
        "log_every": 1,
    })
    options["model"].update({
        "hidden_dim": 8,
        "depth": 2,
        "ppgn_hidden_dim": 12,
        "ppgn_depth": 2,
    })
    options["sde"] = {
        "x": {"type": "vp", "beta_min": 0.1, "beta_max": 0.2, "num_scales": 2},
        "spectrum": {"type": "vp", "beta_min": 0.1, "beta_max": 0.2, "num_scales": 2},
        "eps": 1.0e-5,
        "eigen_mask": "laplacian_nonzero_prefix",
        "spectral_parameterization": "laplacian_log_gap",
        "log_gap_epsilon": 1.0e-6,
        "log_gap_min_std": 1.0e-3,
        "log_gap_exp_clip": 8.0,
    }
    options["sample"].update({"corrector": "none", "n_steps": 0, "eps": 1.0e-2})
    options["attribute_diffusion"].update({"steps": 4, "sample_steps": 4})
    options["generation_batch_size"] = 2
    options["runtime"] = {"device": "cpu"}

    wrapper = GDSMSimpleWrapper()
    run = RunSpec("gdsm_simple", "zinc", "hybrid_zinc", 42, tmp_path / "runs")
    dataset = DatasetReference("zinc", tmp_path / "datasets", "toy_zinc")
    artifacts = wrapper.train(TrainRequest(run, dataset, options=options))
    generated = wrapper.generate(
        GenerateRequest(run, artifacts.checkpoint_path, 4, 43, generation_id="sample")
    )
    with generated.graphs_path.open("rb") as handle:
        rows = pickle.load(handle)
    assert len(rows) == 4
    assert all(
        data["atomic_num"] in {6, 7, 8, 9, 15, 16, 17, 35, 53}
        for graph in rows for _, data in graph.nodes(data=True)
    )

from __future__ import annotations

import json
import pickle
from pathlib import Path

import networkx as nx
import torch

from grapher.models.base import DatasetReference, GenerateRequest, RunSpec, TrainRequest
from grapher.models.gdsm_simple.hierarchical_attributed_loggap import (
    AttributedSpectrumPPGNScore,
    CHECKPOINT_FORMAT,
    _topology_structure_targets,
    _typed_structure_targets,
    default_options,
)
from grapher.models.gdsm_simple.wrapper import GDSMSimpleWrapper
from grapher.rewiring_mlp.attributed.data import GraphCategoryVocabulary


def _molecule(edges, atoms):
    graph = nx.Graph()
    for index, atomic_num in enumerate(atoms):
        graph.add_node(index, atomic_num=int(atomic_num))
    for u, v, bond_type in edges:
        graph.add_edge(int(u), int(v), bond_type=int(bond_type))
    return graph


def _graphs():
    return [
        _molecule([], [6]),
        _molecule([(0, 1, 1), (1, 2, 1), (2, 3, 2), (3, 4, 1), (4, 0, 1)], [6, 6, 7, 6, 8]),
        _molecule([(0, 1, 1), (1, 2, 2), (2, 3, 1), (3, 4, 1)], [6, 7, 6, 8, 9]),
        _molecule([(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 0, 1), (0, 4, 2)], [6, 6, 6, 7, 8]),
        _molecule([(0, 1, 1), (0, 2, 1), (0, 3, 2), (0, 4, 1)], [6, 7, 8, 6, 9]),
    ]


def _write_dataset(root: Path) -> DatasetReference:
    folder = root / "datasets" / "toy"
    folder.mkdir(parents=True)
    graphs = _graphs()
    for split, rows in (("train", graphs), ("val", graphs[:4]), ("test", graphs[1:])):
        with (folder / f"{split}.pkl").open("wb") as handle:
            pickle.dump(rows, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return DatasetReference("qm9", root / "datasets", "toy")


def _options():
    options = default_options()
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
        "max_nodes": 5,
        "max_feat_num": 5,
        "hidden_dim": 8,
        "depth": 2,
        "ppgn_hidden_dim": 12,
        "ppgn_depth": 2,
    })
    options["structural_features"] = {
        "enabled": True,
        "binarize_threshold": 0.5,
        "random_walk": {"enabled": True, "steps": 2},
        "shortest_path": {"enabled": True, "max_distance": 3},
    }
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
    options["sample"].update({"corrector": "none", "n_steps": 0, "eps": 1.0e-2, "use_ema": False})
    options["topology_summary"]["clustering"]["bins"] = 20
    options["attribute_summary"]["typed_graphlet"]["max_basis_graphs"] = 5
    options["attributed"]["decode"].update({
        "node_mode": "sample",
        "edge_mode": "sample",
        "edge_temperature": 0.6,
        "constraint_mode": "atom_bond_valence",
    })
    options["preprocess"].update({"num_workers": 0, "cache_structure_targets": False})
    options["generation_batch_size"] = 2
    options["runtime"] = {"device": "cpu"}
    return options


def _model() -> AttributedSpectrumPPGNScore:
    return AttributedSpectrumPPGNScore(
        max_feat_num=5,
        max_nodes=5,
        node_classes=4,
        edge_classes=3,
        topology_graphlet_slices=((0, 2), (2, 8), (8, 29)),
        typed_graphlet_slices=((0, 3), (3, 7), (7, 12)),
        structural_features={
            "enabled": True,
            "binarize_threshold": 0.5,
            "random_walk": {"enabled": True, "steps": 2},
            "shortest_path": {"enabled": True, "max_distance": 3},
        },
        topology_clustering_bins=0,
        topology_orbit_width=15,
        ppgn_hidden_dim=12,
        ppgn_depth=2,
    )


def _inputs():
    torch.manual_seed(4)
    x = torch.randn((2, 5, 5))
    adjacency = torch.rand((2, 5, 5))
    adjacency = 0.5 * (adjacency + adjacency.transpose(1, 2))
    idx = torch.arange(5)
    adjacency[:, idx, idx] = 0.0
    flags = torch.ones((2, 5))
    eigenvectors = torch.eye(5).unsqueeze(0).repeat(2, 1, 1)
    eigenvalues = torch.randn((2, 5))
    return x, adjacency, flags, eigenvectors, eigenvalues


def test_topology_and_typed_graphlet_targets_are_distinct():
    graphs = _graphs()[1:]
    options = _options()
    vocabulary = GraphCategoryVocabulary.from_graphs(graphs, options["attributed"])
    topology_targets, topology_basis, topology_meta = _topology_structure_targets(graphs, options)
    typed_targets, typed_basis, typed_meta = _typed_structure_targets(
        graphs, options, vocabulary, seed=42
    )
    assert topology_basis.attributed is False
    assert topology_meta["topology_only_graphlets"] is True
    assert topology_targets[0].shape[1] == topology_basis.width
    assert typed_basis.attributed is True
    assert typed_meta["typed_graphlets"] is True
    assert typed_meta["graphlet_vocabulary"] == "training_only_plus_overflow"
    assert typed_targets[0].shape[1] == typed_basis.width
    assert topology_basis.width != typed_basis.width


def test_branches_have_disjoint_gradients_and_expected_outputs():
    model = _model()
    x, adjacency, flags, eigenvectors, eigenvalues = _inputs()
    topology = model.forward_topology(x, adjacency, flags, eigenvectors, eigenvalues)
    attributes = model.forward_attributes(x, adjacency, flags, eigenvectors, eigenvalues)
    assert topology["spectrum"].shape == (2, 5)
    assert topology["topology_graphlet_logits"].shape == (2, 29)
    assert topology["topology_orbit_histogram_logits"].shape == (2, 15)
    assert attributes["node_logits"].shape == (2, 5, 4)
    assert attributes["edge_logits"].shape == (2, 5, 5, 3)
    assert attributes["typed_graphlet_logits"].shape == (2, 12)
    assert torch.allclose(
        attributes["edge_logits"], attributes["edge_logits"].transpose(1, 2), atol=1.0e-6
    )

    model.zero_grad(set_to_none=True)
    sum(value.sum() for value in attributes.values()).backward()
    assert any(parameter.grad is not None for parameter in model.attribute_parameters())
    assert all(parameter.grad is None for parameter in model.topology_parameters())

    model.zero_grad(set_to_none=True)
    topology = model.forward_topology(x, adjacency, flags, eigenvectors, eigenvalues)
    sum(value.sum() for value in topology.values()).backward()
    assert any(parameter.grad is not None for parameter in model.topology_parameters())
    assert all(parameter.grad is None for parameter in model.attribute_parameters())


def test_hierarchical_train_generate_smoke(tmp_path):
    torch.set_num_threads(1)
    dataset = _write_dataset(tmp_path)
    wrapper = GDSMSimpleWrapper()
    run = RunSpec("gdsm_simple", "qm9", "hierarchical", 42, tmp_path / "runs")
    artifacts = wrapper.train(TrainRequest(run, dataset, options=_options()))
    state = torch.load(artifacts.checkpoint_path, map_location="cpu", weights_only=False)
    assert state["format"] == CHECKPOINT_FORMAT
    assert state["architecture_contract"] == "hierarchical_topology_then_attributes_separate_ppgn_v1"
    assert state["topology_summary"]["topology_only_graphlets"] is True
    assert state["attribute_summary"]["typed_graphlets"] is True
    assert state["reference_contract"]["rewiring"] is False
    assert any(key.startswith("topology_ppgn_layers") for key in state["model_spectrum_state"])
    assert any(key.startswith("attribute_ppgn_layers") for key in state["model_spectrum_state"])

    generated = wrapper.generate(
        GenerateRequest(run, artifacts.checkpoint_path, 12, 99, generation_id="sample")
    )
    manifest = json.loads(generated.manifest_path.read_text())
    assert manifest["sampling"]["rewiring"] is False
    assert manifest["sampling"]["posthoc_repair"] is False
    assert manifest["sampling"]["topology_attribute_encoder_sharing"] is False
    assert manifest["sampling"]["topology_graphlet_supervision"] is True
    assert manifest["sampling"]["typed_graphlet_supervision"] is True
    assert (generated.generation_dir / "predicted_topology_summaries.pkl").is_file()
    assert (generated.generation_dir / "predicted_typed_graphlet_summaries.pkl").is_file()
    with generated.graphs_path.open("rb") as handle:
        rows = pickle.load(handle)
    assert len(rows) == 12
    for graph in rows:
        assert all(data["atomic_num"] in {6, 7, 8, 9} for _, data in graph.nodes(data=True))
        assert all(data["bond_type"] in {1, 2, 3} for _, _, data in graph.edges(data=True))


def test_explicit_qm9_configs_select_hierarchical_no_rewiring(tmp_path):
    root = Path(__file__).resolve().parents[1]
    for seed in (42, 43, 44):
        config = root / (
            "configs/experiments/gdsm_laplacian_loggap_hierarchical_attributed_explicit/"
            f"qm9_seed_{seed}.yaml"
        )
        request = TrainRequest(
            RunSpec("gdsm_simple", "qm9", "cfg", seed, tmp_path / "runs"),
            DatasetReference("qm9", tmp_path / "datasets", "unused"),
            config_path=config,
        )
        options = GDSMSimpleWrapper()._options(request)
        assert options["variant"] == "vanilla_laplacian_loggap_hierarchical_attributed_ppgn"
        assert options["topology_summary"]["graphlet"]["orders"] == [3, 4, 5]
        assert options["topology_summary"]["orbit"]["enabled"] is True
        assert options["attribute_summary"]["typed_graphlet"]["orders"] == [3, 4, 5]
        assert options["attributed"]["decode"]["constraint_mode"] == "atom_bond_valence"
        assert options["extensions"]["degree_preserving_rewiring"] is False

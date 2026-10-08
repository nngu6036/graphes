from __future__ import annotations

import json
import pickle
from pathlib import Path

import networkx as nx
import torch

from grapher.models.base import DatasetReference, GenerateRequest, RunSpec, TrainRequest
from grapher.models.gdsm_simple.hybrid_attributed_loggap import (
    ATTRIBUTE_DIFFUSION_TRAINING_CONTRACT,
    CHECKPOINT_FORMAT,
    HybridAttributedSpectrumPPGNScore,
    _attribute_noise_models,
    _draw_noisy_attributes,
    _fit_attribute_diffusion_schema,
    default_options,
)
from grapher.models.gdsm_simple.wrapper import GDSMSimpleWrapper


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
    options["attribute_diffusion"].update({"steps": 4, "sample_steps": 4})
    options["preprocess"].update({"num_workers": 0, "cache_structure_targets": False})
    options["generation_batch_size"] = 2
    options["runtime"] = {"device": "cpu"}
    return options


def _model() -> HybridAttributedSpectrumPPGNScore:
    return HybridAttributedSpectrumPPGNScore(
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


def test_hybrid_attribute_branch_is_time_conditioned_and_parameter_disjoint():
    torch.manual_seed(3)
    model = _model()
    x = torch.randn((2, 5, 5))
    adjacency = torch.zeros((2, 5, 5))
    adjacency[:, 0, 1] = adjacency[:, 1, 0] = 1
    adjacency[:, 1, 2] = adjacency[:, 2, 1] = 1
    flags = torch.ones((2, 5))
    support = adjacency.bool()
    node_state = torch.tensor([[0, 1, 2, 3, 0], [1, 1, 2, 0, 3]])
    edge_state = torch.zeros((2, 5, 5), dtype=torch.long)
    edge_state[:, 1, 2] = edge_state[:, 2, 1] = 1
    node_input, edge_input = model.categorical_attribute_inputs(
        flags, support, node_state, edge_state
    )
    eigenvectors = torch.eye(5).unsqueeze(0).repeat(2, 1, 1)
    eigenvalues = torch.randn((2, 5))
    early = model.forward_attributes(
        x, adjacency, flags, eigenvectors, eigenvalues,
        node_attr=node_input, edge_attr=edge_input,
        attribute_time=torch.ones(2),
    )
    late = model.forward_attributes(
        x, adjacency, flags, eigenvectors, eigenvalues,
        node_attr=node_input, edge_attr=edge_input,
        attribute_time=torch.full((2,), 4.0),
    )
    assert not torch.allclose(early["node_logits"], late["node_logits"])
    assert edge_input.shape[-1] == 3
    assert torch.count_nonzero(edge_input[~support]) == 0

    model.zero_grad(set_to_none=True)
    sum(value.sum() for value in early.values()).backward()
    assert any(parameter.grad is not None for parameter in model.attribute_parameters())
    assert all(parameter.grad is None for parameter in model.topology_parameters())


def test_marginal_noise_uses_real_bond_classes_only_and_fixed_support():
    node_labels = torch.tensor([[0, 1, 2], [3, 0, -1]])
    edge_labels = torch.full((2, 3, 3), -1, dtype=torch.long)
    edge_labels[0, 0, 1] = edge_labels[0, 1, 0] = 0
    edge_labels[0, 1, 2] = edge_labels[0, 2, 1] = 1
    edge_labels[1, 0, 1] = edge_labels[1, 1, 0] = 2
    flags = node_labels.ge(0).float()
    support = edge_labels.ge(0)
    schema = _fit_attribute_diffusion_schema(
        node_labels, edge_labels, node_classes=4, edge_classes=3, pseudocount=1e-3
    )
    assert len(schema["bond_marginal"]) == 3
    assert schema["bond_head_includes_no_edge"] is False
    cfg = {"steps": 4}
    _, node_noise, bond_noise = _attribute_noise_models(schema, cfg, torch.device("cpu"))
    generator = torch.Generator().manual_seed(9)
    t = torch.tensor([4, 2])
    noisy_nodes, noisy_bonds = _draw_noisy_attributes(
        node_labels, edge_labels, flags, support, t,
        node_noise, bond_noise, generator,
    )
    assert noisy_nodes.shape == node_labels.shape
    assert noisy_bonds.shape == edge_labels.shape
    assert torch.equal(noisy_bonds, noisy_bonds.transpose(1, 2))
    assert torch.count_nonzero(noisy_bonds[~support]) == 0
    assert int(noisy_bonds[support].max()) < 3


def test_hybrid_train_generate_smoke(tmp_path):
    torch.set_num_threads(1)
    dataset = _write_dataset(tmp_path)
    wrapper = GDSMSimpleWrapper()
    run = RunSpec("gdsm_simple", "qm9", "hybrid", 42, tmp_path / "runs")
    artifacts = wrapper.train(TrainRequest(run, dataset, options=_options()))
    state = torch.load(artifacts.checkpoint_path, map_location="cpu", weights_only=False)
    assert state["format"] == CHECKPOINT_FORMAT
    assert state["attribute_diffusion_training_contract"] == ATTRIBUTE_DIFFUSION_TRAINING_CONTRACT
    assert state["architecture_contract"] == "hierarchical_topology_then_fixed_topology_categorical_attributes_v1"
    assert state["attribute_diffusion_schema"]["bond_head_includes_no_edge"] is False
    assert any(key.startswith("attribute_time_mlp") for key in state["model_spectrum_state"])

    generated = wrapper.generate(
        GenerateRequest(run, artifacts.checkpoint_path, 8, 99, generation_id="sample")
    )
    manifest = json.loads(generated.manifest_path.read_text())
    sampling = manifest["sampling"]
    assert sampling["rewiring"] is False
    assert sampling["posthoc_repair"] is False
    assert sampling["edge_head_includes_no_edge"] is False
    assert sampling["topology_preserved_during_attribute_diffusion"] is True
    assert sampling["attribute_sampling_steps"] == 4
    assert (generated.generation_dir / "topology_graphs.pkl").is_file()
    assert (generated.generation_dir / "molecular_graphs.pkl").is_file()
    assert (generated.generation_dir / "attribute_diffusion_diagnostics.json").is_file()

    with generated.graphs_path.open("rb") as handle:
        molecules = pickle.load(handle)
    with (generated.generation_dir / "topology_graphs.pkl").open("rb") as handle:
        topologies = pickle.load(handle)
    assert len(molecules) == len(topologies) == 8
    for molecule, topology in zip(molecules, topologies):
        assert set(molecule.edges()) == set(topology.edges())
        assert all(data["atomic_num"] in {6, 7, 8, 9} for _, data in molecule.nodes(data=True))
        assert all(data["bond_type"] in {1, 2, 3} for _, _, data in molecule.edges(data=True))


def test_explicit_qm9_configs_select_hybrid_model(tmp_path):
    root = Path(__file__).resolve().parents[1]
    for seed in (42, 43, 44):
        config = root / (
            "configs/experiments/gdsm_laplacian_loggap_hybrid_categorical_attributed_explicit/"
            f"qm9_seed_{seed}.yaml"
        )
        request = TrainRequest(
            RunSpec("gdsm_simple", "qm9", "cfg", seed, tmp_path / "runs"),
            DatasetReference("qm9", tmp_path / "datasets", "unused"),
            config_path=config,
        )
        options = GDSMSimpleWrapper()._options(request)
        assert options["variant"] == "vanilla_laplacian_loggap_hybrid_categorical_attributed_ppgn"
        assert options["attribute_diffusion"]["steps"] == 100
        assert options["attribute_diffusion"]["sample_steps"] == 100
        assert options["topology_summary"]["graphlet"]["orders"] == [3, 4, 5]
        assert options["attribute_summary"]["typed_graphlet"]["orders"] == [3, 4, 5]
        assert options["attributed"]["decode"]["constraint_mode"] == "atom_bond_valence"
        assert options["extensions"]["degree_preserving_rewiring"] is False

from __future__ import annotations

import json
import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import torch

from grapher.models.base import DatasetReference, GenerateRequest, RunSpec, TrainRequest
from grapher.models.gdsm_simple.attributed_loggap import (
    AttributedSpectrumPPGNScore,
    CHECKPOINT_FORMAT,
    _attributed_labels,
    _mask_categorical_inputs,
    _typed_structure_targets,
    default_options,
)
from grapher.models.gdsm_simple.wrapper import GDSMSimpleWrapper
from grapher.rewiring_mlp.attributed.data import GraphCategoryVocabulary


def _mol_graph(edges, atoms):
    g = nx.Graph()
    for i, atomic_num in enumerate(atoms):
        g.add_node(i, atomic_num=int(atomic_num))
    for u, v, bond_type in edges:
        g.add_edge(int(u), int(v), bond_type=int(bond_type))
    return g


def _graphs():
    return [
        _mol_graph([(0,1,1),(1,2,1),(2,3,2),(3,4,1),(4,0,1)], [6,6,7,6,8]),
        _mol_graph([(0,1,1),(1,2,2),(2,3,1),(3,4,1)], [6,7,6,8,9]),
        _mol_graph([(0,1,1),(1,2,1),(2,3,1),(3,0,1),(0,4,2)], [6,6,6,7,8]),
        _mol_graph([(0,1,1),(0,2,1),(0,3,2),(0,4,1)], [6,7,8,6,9]),
    ]


def _write_dataset(root: Path):
    folder = root / "datasets" / "toy"
    folder.mkdir(parents=True)
    singleton = _mol_graph([], [6])
    train = [singleton] + _graphs()
    val = [singleton, _graphs()[0], _graphs()[1]]
    test = [_graphs()[2]]
    for split, graphs in (("train", train), ("val", val), ("test", test)):
        with (folder / f"{split}.pkl").open("wb") as handle:
            pickle.dump(graphs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return DatasetReference("qm9", root / "datasets", "toy")


def _options():
    o = default_options()
    o["train"].update({
        "epochs": 1,
        "batch_size": 2,
        "lr": 1e-3,
        "node_lr": 1e-3,
        "spectrum_lr": 1e-3,
        "weight_decay": 0.0,
        "grad_norm": 1.0,
        "lr_schedule": False,
        "lr_decay": 1.0,
        "ema": 0.9,
        "validation_every": 1,
        "log_every": 1,
    })
    o["model"].update({
        "max_nodes": 5,
        "max_feat_num": 5,
        "hidden_dim": 8,
        "depth": 2,
        "ppgn_hidden_dim": 12,
        "ppgn_depth": 2,
    })
    o["structural_features"] = {
        "enabled": True,
        "binarize_threshold": 0.5,
        "random_walk": {"enabled": True, "steps": 2},
        "shortest_path": {"enabled": True, "max_distance": 3},
    }
    o["sde"] = {
        "x": {"type": "vp", "beta_min": 0.1, "beta_max": 0.2, "num_scales": 2},
        "spectrum": {"type": "vp", "beta_min": 0.1, "beta_max": 0.2, "num_scales": 2},
        "eps": 1e-5,
        "eigen_mask": "laplacian_nonzero_prefix",
        "spectral_parameterization": "laplacian_log_gap",
        "log_gap_epsilon": 1e-6,
        "log_gap_min_std": 1e-3,
        "log_gap_exp_clip": 8.0,
    }
    o["sample"].update({"corrector": "none", "n_steps": 0, "eps": 1e-2, "use_ema": False})
    o["structure_summary"]["clustering"]["bins"] = 20
    o["structure_summary"]["graphlet"]["max_basis_graphs"] = 4
    o["generation_batch_size"] = 2
    o["runtime"] = {"device": "cpu"}
    return o


def test_attributed_ppgn_edge_head_excludes_no_edge_and_is_symmetric():
    torch.manual_seed(3)
    n = 5
    model = AttributedSpectrumPPGNScore(
        max_feat_num=n,
        max_nodes=n,
        node_classes=4,
        edge_classes=3,
        graphlet_slices=((0, 3), (3, 7), (7, 12)),
        structural_features={
            "enabled": True,
            "binarize_threshold": 0.5,
            "random_walk": {"enabled": True, "steps": 2},
            "shortest_path": {"enabled": True, "max_distance": 3},
        },
        clustering_bins=20,
        orbit_width=15,
        ppgn_hidden_dim=12,
        ppgn_depth=2,
    )
    x = torch.randn((2, n, n))
    a = torch.rand((2, n, n)); a = 0.5 * (a + a.transpose(1,2))
    idx = torch.arange(n); a[:, idx, idx] = 0
    flags = torch.ones((2, n))
    u = torch.eye(n).unsqueeze(0).repeat(2,1,1)
    z = torch.randn((2,n))
    out = model.forward_all(x,a,flags,u,z)
    assert out["node_logits"].shape == (2,n,4)
    # Exactly three real QM9 bond classes; there is no fourth no-edge class.
    assert out["edge_logits"].shape == (2,n,n,3)
    assert torch.allclose(out["edge_logits"], out["edge_logits"].transpose(1,2), atol=1e-6)
    assert all(torch.isfinite(v).all() for v in out.values())


def test_masked_edge_supervision_inputs_do_not_use_no_edge_category():
    graphs = _graphs()[:2]
    vocab = GraphCategoryVocabulary.from_graphs(graphs, {
        "node_attribute": "atomic_num", "node_categories": [6,7,8,9],
        "edge_attribute": "bond_type", "edge_categories": [1,2,3],
    })
    nl, el = _attributed_labels(graphs, vocab, 5)
    model = AttributedSpectrumPPGNScore(
        max_feat_num=5,max_nodes=5,node_classes=4,edge_classes=3,
        graphlet_slices=((0,2),(2,4),(4,6)),
        structural_features={"enabled": False},clustering_bins=0,orbit_width=0,
        ppgn_hidden_dim=8,ppgn_depth=1,
    )
    flags = torch.ones((2,5))
    current = torch.tensor(np.stack([nx.to_numpy_array(g) for g in graphs]), dtype=torch.float32)
    gen = torch.Generator().manual_seed(4)
    node_in, edge_in, node_mask, edge_mask = _mask_categorical_inputs(
        nl, el, flags, current, torch.tensor([0.5,0.5]), model,
        {"corruption": {"mask_probability_min": 1.0, "mask_probability_max": 1.0, "full_mask_probability": 0.0}}, gen,
    )
    assert node_in.shape[-1] == 5  # 4 categories + MASK
    assert edge_in.shape[-1] == 4  # 3 real bond types + MASK, never no-edge
    assert node_mask.any() and edge_mask.any()
    # Nonedges carry an all-zero categorical vector; existence is a separate topology channel.
    assert torch.all(edge_in[current == 0] == 0)


def test_typed_graphlet_targets_use_attributed_training_vocabulary():
    graphs = _graphs()
    options = _options()
    vocab = GraphCategoryVocabulary.from_graphs(graphs, options["attributed"])
    tensors, basis, meta = _typed_structure_targets(graphs, options, vocab, seed=42)
    gh, gm, ch, oh, ot = tensors
    assert meta["typed_graphlets"] is True
    assert meta["graphlet_vocabulary"] == "training_only_plus_overflow"
    assert basis.attributed is True
    assert gh.shape[0] == len(graphs)
    assert gm.shape == (len(graphs), 3)
    assert ch.shape == (len(graphs), 20)
    assert oh.shape == (len(graphs), 15)
    assert ot.shape == (len(graphs), 1)


def test_attributed_loggap_train_generate_smoke(tmp_path):
    torch.set_num_threads(1)
    dataset = _write_dataset(tmp_path)
    wrapper = GDSMSimpleWrapper()
    run = RunSpec("gdsm_simple", "qm9", "attr-loggap", 42, tmp_path / "runs")
    artifacts = wrapper.train(TrainRequest(run, dataset, options=_options()))
    state = torch.load(artifacts.checkpoint_path, map_location="cpu", weights_only=False)
    assert state["format"] == CHECKPOINT_FORMAT
    assert state["reference_contract"]["categorical_flow_matching"] is False if "reference_contract" in state else True
    assert state["model_config"]["edge_classes"] == 3
    assert state["structure_summary"]["typed_graphlets"] is True
    assert any("node_category_head" in k for k in state["model_spectrum_state"])
    assert any("edge_category_head" in k for k in state["model_spectrum_state"])

    generated = wrapper.generate(GenerateRequest(run, artifacts.checkpoint_path, 20, 99, generation_id="s"))
    with generated.graphs_path.open("rb") as handle:
        rows = pickle.load(handle)
    assert len(rows) == 20
    assert any(g.number_of_nodes() == 1 for g in rows)
    for g in rows:
        assert all("atomic_num" in d for _, d in g.nodes(data=True))
        assert all(d["atomic_num"] in {6,7,8,9} for _, d in g.nodes(data=True))
        assert all("bond_type" in d for _,_,d in g.edges(data=True))
        assert all(d["bond_type"] in {1,2,3} for _,_,d in g.edges(data=True))
    manifest = json.loads((generated.generation_dir / "manifest.json").read_text())
    assert manifest["sampling"]["edge_head_includes_no_edge"] is False
    assert manifest["sampling"]["categorical_flow_matching"] is False
    assert manifest["sampling"]["rewiring"] is False


def test_shipped_qm9_attributed_config_resolves(tmp_path):
    root = Path(__file__).resolve().parents[1]
    config = root / "configs/experiments/gdsm_laplacian_loggap_attributed_rwsp_ppgn_explicit/qm9_seed_42.yaml"
    wrapper = GDSMSimpleWrapper()
    request = TrainRequest(
        RunSpec("gdsm_simple", "qm9", "cfg", 42, tmp_path / "runs"),
        DatasetReference("qm9", tmp_path / "datasets", "unused"),
        config_path=config,
    )
    options = wrapper._options(request)
    assert options["variant"] == "vanilla_laplacian_loggap_attributed_ppgn"
    assert options["model"]["spectrum_backbone"] == "ppgn"
    assert options["attributed"]["node_categories"] == [6,7,8,9]
    assert options["attributed"]["edge_categories"] == [1,2,3]
    assert options["structure_summary"]["graphlet"]["attributed"] is True
    assert options["graphlet_refinement"]["enabled"] is False

from __future__ import annotations

import json
import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import torch
import yaml

from grapher.models.base import DatasetReference, GenerateRequest, RunSpec, TrainRequest
from grapher.models.gdsm_simple import hierarchical_attributed_loggap as hierarchical
from grapher.models.gdsm_simple.hybrid_attributed_loggap import (
    HybridAttributedSpectrumPPGNScore,
    validate_options,
)
from grapher.models.gdsm_simple.typed_degree_summary import (
    TypedDegreeHistogramBasis,
)
from grapher.models.gdsm_simple.wrapper import GDSMSimpleWrapper
from grapher.rewiring_mlp.attributed.data import GraphCategoryVocabulary


def _molecule(atoms, edges):
    graph = nx.Graph()
    for index, atomic_num in enumerate(atoms):
        graph.add_node(index, atomic_num=int(atomic_num))
    for u, v, bond in edges:
        graph.add_edge(int(u), int(v), bond_type=int(bond))
    return graph


def _qm9_options() -> dict:
    root = Path(__file__).resolve().parents[1]
    config = root / (
        "configs/experiments/"
        "gdsm_laplacian_loggap_hybrid_degree_typed_degree_g345_explicit/"
        "qm9_seed_42.yaml"
    )
    options = yaml.safe_load(config.read_text())["gdsm_simple"]
    options["model"]["max_nodes"] = 5
    options["model"]["max_feat_num"] = 5
    options["topology_summary"]["degree"]["max_degree"] = 4
    options["attribute_summary"]["typed_graphlet"]["max_basis_graphs"] = 4
    options["preprocess"]["num_workers"] = 0
    return options


def test_typed_degree_basis_is_permutation_invariant_and_uses_overflow() -> None:
    graphs = [
        _molecule([6, 6, 8], [(0, 1, 1), (1, 2, 2)]),
        _molecule([6, 7, 9], [(0, 1, 1), (1, 2, 1)]),
    ]
    vocab = GraphCategoryVocabulary.from_graphs(
        graphs,
        {
            "node_attribute": "atomic_num",
            "node_categories": [6, 7, 8, 9],
            "edge_attribute": "bond_type",
            "edge_categories": [1, 2, 3],
        },
    )
    basis = TypedDegreeHistogramBasis.fit_from_graphs(
        graphs, vocab, min_count=1, max_signatures=3
    )
    assert basis.width == 4
    histogram = basis.histogram_for_graph(graphs[0], vocab)
    relabelled = nx.relabel_nodes(graphs[0], {0: 2, 1: 0, 2: 1})
    relabelled_histogram = basis.histogram_for_graph(relabelled, vocab)
    assert np.allclose(histogram, relabelled_histogram)
    assert np.isclose(histogram.sum(), 1.0)

    unseen = _molecule([6, 8, 8], [(0, 1, 2), (0, 2, 2)])
    unseen_histogram = basis.histogram_for_graph(unseen, vocab)
    assert unseen_histogram[basis.overflow_index] > 0.0
    roundtrip = TypedDegreeHistogramBasis.from_dict(basis.to_dict())
    assert roundtrip == basis


def test_typed_degree_targets_head_and_loss_are_present() -> None:
    options = _qm9_options()
    validate_options(options)
    graphs = [
        _molecule([6, 6, 8], [(0, 1, 1), (1, 2, 2)]),
        _molecule([6, 7, 9], [(0, 1, 1), (1, 2, 1)]),
        _molecule([6, 6, 6, 6], [(0, 1, 1), (1, 2, 1), (2, 3, 1)]),
    ]
    vocab = GraphCategoryVocabulary.from_graphs(graphs, options["attributed"])
    targets, topology_basis, typed_basis, topology_meta, attribute_meta = (
        hierarchical._hierarchical_structure_targets(
            graphs, options, vocab, seed=42
        )
    )
    assert len(targets) == 9
    typed_degree_target = targets[-1]
    assert attribute_meta["typed_degree_enabled"] is True
    assert typed_degree_target.shape[1] == attribute_meta["typed_degree_bins"]
    assert torch.allclose(typed_degree_target.sum(-1), torch.ones(len(graphs)))

    model = HybridAttributedSpectrumPPGNScore(
        max_feat_num=5,
        max_nodes=5,
        node_classes=vocab.num_node_categories,
        edge_classes=len(vocab.edge_values),
        topology_graphlet_slices=tuple(topology_basis.slices),
        typed_graphlet_slices=tuple(typed_basis.slices),
        structural_features=options["structural_features"],
        topology_clustering_bins=0,
        topology_orbit_width=15,
        topology_degree_bins=int(topology_meta["degree_bins"]),
        typed_degree_bins=int(attribute_meta["typed_degree_bins"]),
        ppgn_hidden_dim=12,
        ppgn_depth=2,
    )
    flags = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.float32)
    support = torch.zeros((1, 5, 5), dtype=torch.bool)
    support[:, 0, 1] = support[:, 1, 0] = True
    support[:, 1, 2] = support[:, 2, 1] = True
    node_state = torch.tensor([[0, 0, 2, 0, 0]], dtype=torch.long)
    edge_state = torch.zeros((1, 5, 5), dtype=torch.long)
    edge_state[:, 1, 2] = edge_state[:, 2, 1] = 1
    node_attr, edge_attr = model.categorical_attribute_inputs(
        flags, support, node_state, edge_state
    )
    x = torch.randn(1, 5, 5)
    eigenvalues = torch.randn(1, 5)
    out = model.forward_attributes(
        x,
        support.float(),
        flags,
        torch.eye(5).unsqueeze(0),
        eigenvalues,
        node_attr=node_attr,
        edge_attr=edge_attr,
        attribute_time=torch.tensor([0.5]),
    )
    assert out["typed_degree_histogram_logits"].shape == (
        1,
        attribute_meta["typed_degree_bins"],
    )
    means = model.attribute_structure_means_from_outputs(out)
    assert torch.allclose(
        means["typed_degree_histogram"].sum(-1), torch.ones(1), atol=1.0e-6
    )

    loss, metrics = hierarchical._attribute_structure_loss(
        model,
        out,
        targets[-3][:1],
        targets[-2][:1],
        typed_degree_target[:1],
        options["attribute_summary"],
    )
    assert torch.isfinite(loss)
    assert "typed_degree_histogram_loss" in metrics


def test_hybrid_typed_degree_train_generate_smoke(tmp_path: Path) -> None:
    torch.set_num_threads(1)
    graphs = [
        _molecule([6, 6, 8], [(0, 1, 1), (1, 2, 2)]),
        _molecule([6, 7, 9], [(0, 1, 1), (1, 2, 1)]),
        _molecule([6, 6, 6, 8], [(0, 1, 1), (1, 2, 1), (2, 3, 2)]),
        _molecule([6, 8, 6], [(0, 1, 2), (1, 2, 1)]),
    ]
    folder = tmp_path / "datasets" / "toy_qm9_typed_degree"
    folder.mkdir(parents=True)
    for split, rows in (("train", graphs), ("val", graphs[:3]), ("test", graphs[1:])):
        with (folder / f"{split}.pkl").open("wb") as handle:
            pickle.dump(rows, handle, protocol=pickle.HIGHEST_PROTOCOL)

    options = _qm9_options()
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
    run = RunSpec("gdsm_simple", "qm9", "hybrid_typed_degree", 42, tmp_path / "runs")
    dataset = DatasetReference("qm9", tmp_path / "datasets", "toy_qm9_typed_degree")
    artifacts = wrapper.train(TrainRequest(run, dataset, options=options))
    try:
        state = torch.load(artifacts.checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(artifacts.checkpoint_path, map_location="cpu")
    assert state["attribute_summary"]["typed_degree_enabled"] is True
    assert any(
        "typed_degree_histogram_logits" in key
        for key in state["model_spectrum_state"]
    )
    assert "train_typed_degree_histogram_loss" in state["history"][0]

    generated = wrapper.generate(
        GenerateRequest(run, artifacts.checkpoint_path, 4, 43, generation_id="sample")
    )
    predictions_path = generated.generation_dir / "predicted_typed_graphlet_summaries.pkl"
    with predictions_path.open("rb") as handle:
        predictions = pickle.load(handle)
    assert len(predictions) == 4
    for row in predictions:
        histogram = np.asarray(row["typed_degree_histogram"], dtype=np.float64)
        assert np.isclose(histogram.sum(), 1.0)

    manifest = json.loads(generated.manifest_path.read_text())
    assert manifest["sampling"]["typed_degree_histogram_supervision"] is True
    assert manifest["predicted_typed_graphlet_summaries"]["typed_degree_histogram"] is True

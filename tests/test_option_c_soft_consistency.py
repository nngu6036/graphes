"""Option-C schema-v2 soft degree + normalized-Laplacian consistency."""
from __future__ import annotations

import copy
import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import torch
import yaml
import pytest

from grapher.models.option_c.config import load_config, validate_config
from grapher.models.option_c.data import prepare_records, collate, TypedGraphletsMulti
from grapher.models.option_c.diffusion import WeightedEdgeCodec, pair_mask, symmetrize
from grapher.models.option_c.losses import (
    soft_presence, soft_degree_consistency_loss,
    normalized_laplacian_spectral_loss, joint_loss,
)
from grapher.models.option_c.model import OptionCDenoiser, model_config
from grapher.models.option_c.runtime import load_checkpoint
from grapher.models.option_c.training import train
from grapher.models.option_c.sampling import generate
from grapher.models.option_c.data import GraphCategoryVocabulary

ROOT = Path(__file__).resolve().parents[1]


def cfg_v2(dataset="community_small"):
    cfg = load_config(ROOT / f"configs/experiments/option_c_soft_consistency/{dataset}.yaml")
    cfg["model"].update(hidden_dim=16, num_layers=1, num_heads=2, ff_dim=32, dropout=0.)
    cfg["training"].update(epochs=1, batch_size=3, validation_every=1, checkpoint_every=1,
                            log_every=1, log_every_batches=0)
    cfg["diffusion"]["steps"] = 8
    cfg["sampling"].update(steps=2, batch_size=3)
    cfg["graphlets"]["clustering_bins"] = 10
    cfg["refinement"].update(enabled=False, proposal_budget=16, valid_candidate_budget=8, max_steps=1)
    return validate_config(cfg)


def graphs(cfg):
    out = [nx.path_graph(3), nx.cycle_graph(4), nx.path_graph(6), nx.star_graph(4), nx.cycle_graph(5), nx.path_graph(4)]
    attr, edge_attr = cfg["categories"]["node_attribute"], cfg["categories"]["edge_attribute"]
    if attr:
        values = cfg["categories"]["node_categories"]
        for g in out:
            nx.set_node_attributes(g, {i: values[i % min(3, len(values))] for i in g}, attr)
            nx.set_edge_attributes(g, 1, edge_attr)
        out[0].edges[0, 1][edge_attr] = 2
        out[2].edges[0, 1][edge_attr] = 3
    return out


def prepared(tmp_path, dataset="community_small"):
    cfg = cfg_v2(dataset)
    root = tmp_path / "datasets"
    directory = root / cfg["dataset"]["name"]
    directory.mkdir(parents=True)
    source = tmp_path / "dataset.yaml"
    source.write_text(yaml.safe_dump({"name": cfg["dataset"]["name"], "protocol_id": "soft-consistency-unit"}))
    (directory / "resolved_dataset_config.yaml").write_text(source.read_text())
    cfg["dataset"].update(root=str(root), config_path=str(source))
    cfg["training"]["cache_dir"] = str(tmp_path / "cache")
    gs = graphs(cfg)
    for split, subset in (("train", gs), ("val", gs[1:4])):
        with (directory / f"{split}.pkl").open("wb") as f:
            pickle.dump(subset, f)
    return cfg


def test_v2_profiles_declare_new_consistency_objective():
    for dataset in ("community_small", "ego_small", "qm9", "zinc"):
        cfg = load_config(ROOT / f"configs/experiments/option_c_soft_consistency/{dataset}.yaml")
        assert cfg["schema_version"] == 2
        assert cfg["spectral"]["normalization"] == "normalized_laplacian"
        assert cfg["consistency"]["soft_threshold_physical"] == .5
        assert cfg["consistency"]["temperature_physical"] == .1
        assert cfg["loss_weights"]["degree"] > 0


def test_soft_presence_uses_physical_units_under_molecular_scaling():
    mask = torch.ones(1, 2, dtype=torch.bool)
    cfg = {"soft_threshold_physical": .5, "temperature_physical": .1,
           "degree_normalization": "n_minus_one", "normalized_laplacian_epsilon": 1e-8}
    # Scaled 1/6 corresponds to physical 0.5 with edge_scale=3 -> sigmoid(0)=0.5.
    w = torch.tensor([[[0., 1/6], [1/6, 0.]]])
    soft = soft_presence(w, mask, cfg, edge_scale=3.)
    torch.testing.assert_close(soft[0, 0, 1], torch.tensor(.5))
    assert soft[0, 0, 0] == 0


def test_degree_consistency_prefers_clean_topology():
    mask = torch.ones(1, 4, dtype=torch.bool)
    cfg = {"soft_threshold_physical": .5, "temperature_physical": .05,
           "degree_normalization": "n_minus_one", "normalized_laplacian_epsilon": 1e-8}
    clean = torch.tensor([[[0.,1.,0.,0.],[1.,0.,1.,0.],[0.,1.,0.,1.],[0.,0.,1.,0.]]])
    good = clean.clone() * .9 + (1-clean) * .1
    good[:, torch.arange(4), torch.arange(4)] = 0
    bad = torch.full_like(clean, .1)
    bad[:, torch.arange(4), torch.arange(4)] = 0
    gl = soft_degree_consistency_loss(good, clean, mask, cfg, edge_scale=1.)
    bl = soft_degree_consistency_loss(bad, clean, mask, cfg, edge_scale=1.)
    assert gl < bl


def test_normalized_laplacian_target_path3_is_0_1_2():
    cfg = cfg_v2("community_small")
    rows, _, schema = prepare_records([nx.path_graph(3)], [nx.path_graph(3)], cfg)
    np.testing.assert_allclose(rows[0]["spectrum"], np.array([0., 1., 2.], np.float32), atol=1e-6)


def test_normalized_laplacian_loss_is_padding_safe_permutation_invariant_and_differentiable():
    mask = torch.tensor([[1,1,1,1,0]], dtype=torch.bool)
    clean = torch.tensor([[[0.,1.,0.,0.,0.], [1.,0.,1.,0.,0.], [0.,1.,0.,1.,0.],
                           [0.,0.,1.,0.,0.], [0.,0.,0.,0.,0.]]], dtype=torch.float64)
    # Exact target spectrum of P4 normalized Laplacian.
    d = clean[0,:4,:4].sum(-1)
    inv = d.rsqrt()
    lap = torch.eye(4, dtype=torch.float64) - inv[:,None]*clean[0,:4,:4]*inv[None,:]
    target = torch.zeros(1,5,dtype=torch.float64)
    target[0,:4] = torch.linalg.eigvalsh(lap)
    cfg = {"soft_threshold_physical": .5, "temperature_physical": .12,
           "degree_normalization": "n_minus_one", "normalized_laplacian_epsilon": 1e-8}
    raw = torch.tensor([[[0.,.88,.18,.09,0.], [.88,0.,.74,.11,0.], [.18,.74,0.,.81,0.],
                         [.09,.11,.81,0.,0.], [0.,0.,0.,0.,0.]]], dtype=torch.float64, requires_grad=True)
    pred = symmetrize(raw, mask)
    loss = normalized_laplacian_spectral_loss(pred, target, mask, cfg, edge_scale=1.)
    dirty = pred.detach().clone(); dirty[0,4,4] = 999.; dirty.requires_grad_(True)
    torch.testing.assert_close(normalized_laplacian_spectral_loss(dirty, target, mask, cfg, edge_scale=1.), loss.detach())
    p = torch.tensor([2,0,3,1])
    perm = pred[:, :4, :4][:,p][:,:,p]
    pmask = torch.ones(1,4,dtype=torch.bool)
    ptarget = target[:,:4]
    torch.testing.assert_close(normalized_laplacian_spectral_loss(perm, ptarget, pmask, cfg, edge_scale=1.), loss)
    loss.backward()
    assert raw.grad is not None and torch.isfinite(raw.grad).all() and raw.grad.abs().sum() > 0


def test_v2_joint_loss_updates_adjacency_from_degree_and_normalized_laplacian():
    cfg = cfg_v2("community_small")
    gs = graphs(cfg)[:3]
    rows, _, schema = prepare_records(gs, gs, cfg)
    basis = TypedGraphletsMulti.from_schema(schema)
    batch = collate(rows, basis, device="cpu")
    model = OptionCDenoiser(**model_config(cfg, schema, basis))
    pred = model(batch["x"], batch["w"], torch.tensor([3,4,5]), batch["mask"], 8)
    loss, parts = joint_loss(pred, batch, basis, cfg["loss_weights"], cfg["spectral"],
                             schema_version=2, consistency_cfg=cfg["consistency"],
                             edge_scale=cfg["edge_representation"]["scale"])
    assert "degree" in parts and torch.isfinite(parts["degree"]) and torch.isfinite(parts["spectral"])
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.adjacency_head.parameters())



@pytest.mark.parametrize("dataset", ["community_small", "ego_small", "qm9", "zinc"])
def test_v2_small_training_generation_and_checkpoint_contract(tmp_path, dataset):
    cfg = prepared(tmp_path, dataset)
    train_dir = tmp_path / "train"
    manifest = train(cfg, train_dir, seed=42, device="cpu")
    assert manifest["status"] == "completed"
    assert manifest["contract"]["degree_consistency"].startswith("soft_threshold")
    state = load_checkpoint(train_dir / "checkpoints/best.pt")
    assert state["generation_contract"]["schema_version"] == 2
    assert state["generation_contract"]["spectral"]["normalization"] == "normalized_laplacian"
    out = tmp_path / "gen"
    result = generate(cfg, train_dir / "checkpoints/best.pt", out, num_graphs=3, seed=7, device="cpu")
    assert result["status"] == "completed"
    assert result["contract"]["soft_degree_consistency_trained"]
    assert result["contract"]["normalized_laplacian_consistency_trained"]


def test_v1_and_v2_configs_cannot_share_checkpoint_contract(tmp_path):
    cfg = prepared(tmp_path)
    train(cfg, tmp_path / "train", seed=1, device="cpu")
    old = load_config(ROOT / "configs/experiments/option_c/community_small.yaml")
    old["model"].update(hidden_dim=16, num_layers=1, num_heads=2, ff_dim=32, dropout=0.)
    old["diffusion"]["steps"] = 8
    old["sampling"].update(steps=2, batch_size=3)
    old["graphlets"]["clustering_bins"] = 10
    old["refinement"]["enabled"] = False
    old = validate_config(old)
    with pytest.raises(ValueError, match="trained"):
        generate(old, tmp_path/"train/checkpoints/best.pt", tmp_path/"bad", num_graphs=1, seed=1, device="cpu")

"""Option C: kernel mathematics, symmetry, losses, cache provenance and full CLI IO."""
from __future__ import annotations

import copy
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import torch
import yaml

from grapher.models.option_c.config import load_config, validate_config
from grapher.models.option_c.data import (
    GraphCategoryVocabulary, TypedGraphletsMulti, collate, load_data, prepare_records,
)
from grapher.models.option_c.diffusion import (
    MarginalNoise, WeightedEdgeCodec, cosine_alpha_bar, draw_categories, pair_mask,
    q_sample, reverse_weighted, sampling_grid, symmetric_noise, symmetrize,
)
from grapher.models.option_c.losses import joint_loss, weighted_spectral_loss
from grapher.models.option_c.model import OptionCDenoiser, model_config, prediction_targets
from grapher.models.option_c.refinement import refine, energy
from grapher.models.option_c.runtime import load_checkpoint, resolve_device
from grapher.models.option_c.sampling import generate
from grapher.models.option_c.training import train
from grapher.models.gdsm_simple.categorical.data import topology_summary, encode_graph
from grapher.models.gdsm_simple.categorical.multiscale import fit_basis
from grapher.models.gdsm_simple.categorical.refiner import candidates

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def small_cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def config(dataset="qm9"):
    cfg = load_config(ROOT / f"configs/experiments/option_c/{dataset}.yaml")
    cfg["model"].update(hidden_dim=16, num_layers=1, num_heads=2, ff_dim=32, dropout=0.)
    cfg["training"].update(epochs=2, batch_size=3, validation_every=1, checkpoint_every=1,
                            log_every=1, log_every_batches=0)
    cfg["diffusion"]["steps"] = 8
    cfg["sampling"].update(steps=4, batch_size=3)
    cfg["graphlets"]["clustering_bins"] = 10
    cfg["refinement"].update(proposal_budget=16, valid_candidate_budget=8, max_steps=1)
    return validate_config(cfg)


def graphs(cfg):
    out = [nx.empty_graph(1), nx.path_graph(3), nx.cycle_graph(4), nx.path_graph(6), nx.star_graph(4), nx.cycle_graph(5)]
    attr = cfg["categories"]["node_attribute"]
    edge_attr = cfg["categories"]["edge_attribute"]
    if attr:
        values = cfg["categories"]["node_categories"]
        for g in out:
            nx.set_node_attributes(g, {i: values[i % min(3, len(values))] for i in g}, attr)
            nx.set_edge_attributes(g, 1, edge_attr)
        out[1].edges[0, 1][edge_attr] = 2
        out[3].edges[0, 1][edge_attr] = 3
    return out


def prepared(tmp_path, dataset="qm9"):
    cfg = config(dataset)
    root = tmp_path / "datasets"
    directory = root / cfg["dataset"]["name"]
    directory.mkdir(parents=True)
    source = tmp_path / "dataset.yaml"
    source.write_text(yaml.safe_dump({"name": cfg["dataset"]["name"], "protocol_id": "unit-test-fixture"}))
    (directory / "resolved_dataset_config.yaml").write_text(source.read_text())
    cfg["dataset"].update(root=str(root), config_path=str(source))
    cfg["training"]["cache_dir"] = str(tmp_path / "cache")
    gs = graphs(cfg)
    for split, subset in (("train", gs), ("val", gs[1:4])):
        with (directory / f"{split}.pkl").open("wb") as f:
            pickle.dump(subset, f)
    # Deliberately NO test.pkl: training and generation must not open it.
    return cfg


def same_graphs(left, right):
    assert len(left) == len(right)
    for a, b in zip(left, right):
        assert dict(a.nodes(data=True)) == dict(b.nodes(data=True))
        assert {tuple(sorted((i, j))): d for i, j, d in a.edges(data=True)} == {tuple(sorted((i, j))): d for i, j, d in b.edges(data=True)}


def read_graphs(path):
    with Path(path).open("rb") as f:
        return pickle.load(f)


@pytest.mark.parametrize("dataset,n,epochs", [("community_small", 20, 10000), ("ego_small", 18, 5000),
                                              ("qm9", 9, 200), ("zinc", 38, 200)])
def test_full_profiles(dataset, n, epochs):
    cfg = load_config(ROOT / f"configs/experiments/option_c/{dataset}.yaml")
    assert cfg["model"]["max_nodes"] == n
    assert cfg["training"]["epochs"] == epochs
    assert cfg["graphlets"]["sizes"] == [3, 4, 5]
    assert cfg["protocol"]["full_training_split"]
    assert cfg["protocol"]["seeds"] == [42, 43, 44]
    assert cfg["sampling"]["sampler"] == "ddpm"


@pytest.mark.parametrize("change", ["unknown", "prediction", "negative", "scale", "duplicate", "steps", "refinement", "graphlet"])
def test_config_rejects_inconsistent_semantics(change):
    cfg = config()
    if change == "unknown": cfg["model"]["magic"] = True
    if change == "prediction": cfg["diffusion"]["prediction"] = "epsilon"
    if change == "negative": cfg["loss_weights"]["adjacency"] = -1
    if change == "scale": cfg["edge_representation"]["scale"] = .5
    if change == "duplicate": cfg["edge_representation"]["weights"]["3"] = 1.
    if change == "steps": cfg["sampling"]["steps"] = 100
    if change == "refinement": cfg["refinement"]["timing"] = "every_step"
    if change == "graphlet": cfg["graphlets"]["sizes"] = [5, 3]
    with pytest.raises(ValueError): validate_config(cfg)


def test_codec_physical_mapping_and_threshold_ties():
    # Deliberately reordered category indexes and non-unit scale.
    vocab = GraphCategoryVocabulary(node_values=(6,), edge_values=(3, 1, 2), node_attribute="atomic_num", edge_attribute="bond_type")
    codec = WeightedEdgeCodec(vocab, {"weights": {1: 1., 2: 2., 3: 3.}, "scale": 3.})
    e = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]])
    np.testing.assert_array_equal(codec.decode(codec.encode(e)), e)
    for physical, category in [(-7., 0), (.49, 0), (.5, 2), (1.49, 2), (1.5, 3), (2.49, 3), (2.5, 1), (8., 1)]:
        w = np.array([[0, physical/3], [physical/3, 0]])
        assert codec.decode(w)[0, 1] == category
    assert codec.decode(np.ones((3, 3))).diagonal().sum() == 0
    with pytest.raises(ValueError): codec.decode(np.full((2, 2), np.nan))


def test_symmetric_noise_has_unit_variance_and_masks():
    mask = torch.ones(40000, 2, dtype=torch.bool)
    mask[-1, 1] = False
    noise = symmetric_noise(mask, torch.Generator().manual_seed(11))
    assert abs(noise[:-1, 0, 1].var().item()-1.) < .025
    assert abs(noise[:-1, 0, 1].mean().item()) < .02
    assert torch.equal(noise, noise.transpose(1, 2))
    assert not noise[~pair_mask(mask)].any()


@pytest.mark.parametrize("steps", [2, 8, 500])
def test_forward_exact_terminal_and_sampling_grid(steps):
    a = cosine_alpha_bar(steps)
    assert a[0] == 1 and a[-1] == 0
    mask = torch.tensor([[1, 1, 1], [1, 0, 0]], dtype=torch.bool)
    clean = symmetrize(torch.randn(2, 3, 3), mask)
    eps = symmetric_noise(mask, torch.Generator().manual_seed(22))
    t = torch.full((2,), steps, dtype=torch.long)
    torch.testing.assert_close(q_sample(clean, eps, a, t, mask), eps)
    grid = sampling_grid(steps, min(steps, 4))
    assert grid[0] == steps and grid[-1] == 0 and all(x > y for x, y in zip(grid, grid[1:]))
    with pytest.raises(ValueError): sampling_grid(steps, steps+1)


@pytest.mark.parametrize("t,u", [(8, 5), (7, 2), (1, 0), (8, 0)])
def test_gaussian_posterior_and_ddim_closed_form(t, u):
    a = cosine_alpha_bar(8)
    mask = torch.tensor([[1, 1, 1], [1, 0, 0]], dtype=torch.bool)
    clean = symmetrize(torch.randn(2, 3, 3), mask)
    eps = symmetric_noise(mask, torch.Generator().manual_seed(7))
    tv, uv = torch.full((2,), t), torch.full((2,), u)
    noisy = q_sample(clean, eps, a, tv, mask)
    deterministic = reverse_weighted(noisy, clean, a, tv, uv, mask, torch.Generator(), sampler="ddim")
    torch.testing.assert_close(deterministic, q_sample(clean, eps, a, uv, mask), atol=2e-6, rtol=2e-6)
    result = reverse_weighted(noisy, clean, a, tv, uv, mask, torch.Generator().manual_seed(34), sampler="ddpm")
    ratio = a[t]/a[u]
    mean = a[u].sqrt()*(1-ratio)/(1-a[t])*clean + ratio.sqrt()*(1-a[u])/(1-a[t])*noisy
    variance = (1-a[u])*(1-ratio)/(1-a[t])
    expected = mean + variance.sqrt()*symmetric_noise(mask, torch.Generator().manual_seed(34))
    torch.testing.assert_close(result, expected)
    if u == 0: torch.testing.assert_close(result, clean)


def test_node_prior_is_not_the_learned_final_node_distribution():
    noise = MarginalNoise([.8, .15, .05], cosine_alpha_bar(8))
    clean = torch.tensor([[0, 1, 2]])
    torch.testing.assert_close(noise.forward_probs(clean, torch.tensor([8])), noise.marginal.expand(1, 3, 3))
    pred = torch.tensor([[[.02, .03, .95]]]).expand(1, 12000, 3)
    current = torch.zeros(1, 12000, dtype=torch.long)
    reverse = noise.reverse_probs(pred, current, torch.tensor([1]), torch.tensor([0]))
    torch.testing.assert_close(reverse, pred)
    samples = draw_categories(reverse, torch.Generator().manual_seed(77))
    rate = (samples == 2).float().mean().item()
    assert .94 < rate < .96  # Samples the learned conditional, not mX and not argmax.


def test_spectral_padding_permutation_and_gradient():
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.bool)
    raw = torch.randn(2, 4, 4, dtype=torch.float64, requires_grad=True)
    w = symmetrize(raw, mask)
    target = torch.zeros(2, 4, dtype=torch.float64)
    loss = weighted_spectral_loss(w, target, mask)
    # For zero target, sum eigenvalues^2/n^2 equals Frobenius squared/n^2.
    expected = ((w[0, :2, :2].square().sum()/4) + (w[1].square().sum()/16))/2
    torch.testing.assert_close(loss, expected)
    dirty = w.clone(); dirty[0, 2:, 2:] = 99.
    torch.testing.assert_close(weighted_spectral_loss(dirty, target, mask), loss)
    loss.backward()
    assert raw.grad is not None and torch.isfinite(raw.grad).all() and raw.grad.abs().sum() > 0
    degenerate = torch.zeros(1, 4, 4, requires_grad=True)
    tied_loss = weighted_spectral_loss(degenerate, torch.tensor([[-1., -.2, .2, 1.]]), torch.ones(1, 4, dtype=torch.bool))
    tied_loss.backward()
    assert torch.isfinite(degenerate.grad).all()


def test_train_only_schema_masks_and_multiscale_targets():
    cfg = config()
    gs = graphs(cfg)
    rows, val, schema = prepare_records(gs[:2], [gs[2]], cfg)
    _, _, schema2 = prepare_records(gs[:2], [gs[3], gs[4]], cfg)
    assert schema == schema2
    assert 4 not in schema["graph_sizes"]
    basis = TypedGraphletsMulti.from_schema(schema)
    batch = collate(rows, basis, device="cpu")
    assert batch["graphlet_order_mask"][0].sum() == 0
    model = OptionCDenoiser(**model_config(cfg, schema, basis))
    pred = model(batch["x"], batch["w"], torch.ones(len(rows), dtype=torch.long), batch["mask"], 8)
    assert "edge_logits" not in pred and "clean_spectrum" not in pred
    loss, parts = joint_loss(pred, batch, basis, cfg["loss_weights"], cfg["spectral"])
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    corrupt = {**batch, "mass": batch["mass"].clone()}
    corrupt["mass"][~batch["graphlet_order_mask"]] = 100.
    loss2 = joint_loss(pred, corrupt, basis, cfg["loss_weights"], cfg["spectral"])[0]
    torch.testing.assert_close(loss2, loss)
    assert all(p.grad is not None for p in model.adjacency_head.parameters())


def test_shared_model_permutation_equivariance_and_padding():
    cfg = config(); gs = graphs(cfg)[1:3]
    rows, _, schema = prepare_records(gs, gs, cfg)
    basis = TypedGraphletsMulti.from_schema(schema)
    batch = collate(rows, basis, device="cpu")
    model = OptionCDenoiser(**model_config(cfg, schema, basis)).eval()
    mask = batch["mask"]
    w = symmetric_noise(mask, torch.Generator().manual_seed(12))
    t = torch.tensor([4, 7])
    pred = model(batch["x"], w, t, mask, 8)
    x2, w2 = batch["x"].clone(), w.clone()
    permutations = [torch.tensor([2, 0, 1, 3]), torch.tensor([3, 1, 0, 2])]
    for i, p in enumerate(permutations):
        x2[i] = x2[i, p]; w2[i] = w2[i][p][:, p]
    out = model(x2, w2, t, mask, 8)
    for i, p in enumerate(permutations):
        torch.testing.assert_close(out["node_logits"][i], pred["node_logits"][i][p], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(out["clean_adjacency"][i], pred["clean_adjacency"][i][p][:, p], atol=1e-6, rtol=1e-5)
    for key in ("graphlet_logits", "graphlet_mass", "clustering_logits", "orbit_log_mean"):
        torch.testing.assert_close(out[key], pred[key], atol=1e-6, rtol=1e-5)
    noisy_pad = w.clone(); noisy_pad[~pair_mask(mask)] = 1000.
    masked = model(batch["x"], noisy_pad, t, mask, 8)
    torch.testing.assert_close(masked["clean_adjacency"], pred["clean_adjacency"])
    assert not pred["clean_adjacency"][~pair_mask(mask)].any()


def test_spectral_loss_backpropagates_through_adjacency_and_soft_node_heads():
    cfg = config(); gs = graphs(cfg)[1:3]
    rows, _, schema = prepare_records(gs, gs, cfg)
    basis = TypedGraphletsMulti.from_schema(schema); batch = collate(rows, basis, device="cpu")
    model = OptionCDenoiser(**model_config(cfg, schema, basis))
    pred = model(batch["x"], batch["w"], torch.tensor([3, 4]), batch["mask"], 8)
    weighted_spectral_loss(pred["clean_adjacency"], batch["spectrum"], batch["mask"]).backward()
    for module in (model.adjacency_head, model.node_head, model.blocks):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())


def test_cache_reuse_invalidation_and_protocol_mismatch(tmp_path):
    cfg = prepared(tmp_path)
    a = load_data(cfg)
    b = load_data(cfg)
    assert a[-1]["preprocessing_cache_key"] == b[-1]["preprocessing_cache_key"]
    assert len(a[0]) == 6 and len(a[1]) == 3
    train_path = Path(cfg["dataset"]["root"]) / cfg["dataset"]["name"] / "train.pkl"
    gs = read_graphs(train_path); gs.append(gs[1].copy())
    train_path.write_bytes(pickle.dumps(gs))
    changed = load_data(cfg)
    assert changed[-1]["preprocessing_cache_key"] != a[-1]["preprocessing_cache_key"]
    prepared_cfg = train_path.parent / "resolved_dataset_config.yaml"
    prepared_cfg.write_text("protocol_id: wrong\n")
    with pytest.raises(ValueError, match="protocol"): load_data(cfg)


@pytest.mark.parametrize("dataset", ["community_small", "ego_small", "qm9", "zinc"])
def test_training_generation_and_paired_refinement_roundtrip(tmp_path, dataset):
    cfg = prepared(tmp_path, dataset)
    out = tmp_path / "train"
    manifest = train(cfg, out, seed=42, device="cpu")
    assert manifest["status"] == "completed" and manifest["epochs_completed"] == 2
    assert manifest["dataset"]["num_train_graphs_used"] == 6
    state = load_checkpoint(out / "checkpoints/best.pt")
    assert state["schema"]["graphlet_orders"] == [3, 4, 5]
    # Model inference must work after data files are removed.
    shutil_root = Path(cfg["dataset"]["root"])
    import shutil
    shutil.rmtree(shutil_root)
    gen = tmp_path / "gen"
    generation = generate(cfg, out / "checkpoints/best.pt", gen, num_graphs=5, seed=71, device="cpu")
    assert generation["num_generated"] == 5
    g = read_graphs(gen / "base_graphs.pkl"); before = read_graphs(gen / "pre_rewire_graphs.pkl")
    for initial, final in zip(before, g):
        assert len(initial) == len(final) and dict(initial.degree()) == dict(final.degree())
        assert dict(initial.nodes(data=True)) == dict(final.nodes(data=True))
        assert nx.number_of_selfloops(final) == 0
    none = copy.deepcopy(cfg); none["refinement"]["enabled"] = False
    generate(none, out / "checkpoints/best.pt", tmp_path / "plain", num_graphs=5, seed=71, device="cpu")
    same_graphs(before, read_graphs(tmp_path / "plain/base_graphs.pkl"))
    assert generation["contract"]["categorical_edge_head"] is False
    assert generation["contract"]["spectral_stochastic_state"] is False
    saved = torch.load(gen / "continuous_adjacencies.pt", weights_only=True)
    assert len(saved["pre_threshold_weighted_adjacencies"]) == 5
    codec = WeightedEdgeCodec(GraphCategoryVocabulary.from_dict(state["schema"]["category_vocabulary"]), cfg["edge_representation"])
    for matrix, graph in zip(saved["pre_threshold_weighted_adjacencies"], before):
        _, edges = encode_graph(graph, GraphCategoryVocabulary.from_dict(state["schema"]["category_vocabulary"]), cfg["model"]["max_nodes"])
        np.testing.assert_array_equal(codec.decode(matrix.numpy()/codec.scale), edges)
    if dataset in ("qm9", "zinc"):
        same_graphs(g, read_graphs(gen / "molecular_graphs.pkl"))
    with pytest.raises(FileExistsError): generate(cfg, out / "checkpoints/best.pt", gen, num_graphs=5, seed=71, device="cpu")
    mismatch = copy.deepcopy(cfg); mismatch["diffusion"]["steps"] += 1
    with pytest.raises(ValueError, match="trained"): generate(mismatch, out / "checkpoints/best.pt", tmp_path/"bad", num_graphs=1, seed=1, device="cpu")


def test_resume_exact_cpu_weights_and_optimizer(tmp_path):
    cfg = prepared(tmp_path)
    train(cfg, tmp_path/"full", seed=44, device="cpu")
    short = copy.deepcopy(cfg); short["training"]["epochs"] = 1
    train(short, tmp_path/"resumed", seed=44, device="cpu")
    train(cfg, tmp_path/"resumed", seed=44, device="cpu", resume=True)
    a, b = [load_checkpoint(tmp_path/path/"checkpoints/last.pt") for path in ("full", "resumed")]
    assert a["epoch"] == b["epoch"] == 2 and a["optimizer_steps"] == b["optimizer_steps"] == 4
    for key in a["model_state"]:
        torch.testing.assert_close(a["model_state"][key], b["model_state"][key], rtol=0, atol=0)
    with pytest.raises(ValueError, match="seed"): train(cfg, tmp_path/"resumed", seed=45, device="cpu", resume=True)


def test_optional_ema_and_ddim(tmp_path):
    cfg = prepared(tmp_path)
    cfg["training"].update(epochs=1, ema_decay=.9)
    cfg["sampling"].update(use_ema=True, sampler="ddim")
    train(cfg, tmp_path/"train", seed=1, device="cpu")
    state = load_checkpoint(tmp_path/"train/checkpoints/best.pt")
    assert state["ema_state"] is not None
    generate(cfg, tmp_path/"train/checkpoints/best.pt", tmp_path/"gen", num_graphs=2, seed=2, device="cpu")


def test_oracle_refiner_lowers_energy_and_preserves_typed_degrees():
    cfg = config("community_small")
    cfg["refinement"].update(proposal_budget=500, valid_candidate_budget=128, max_steps=2)
    vocab = GraphCategoryVocabulary.topology_only(); codec = WeightedEdgeCodec(vocab, cfg["edge_representation"])
    graph = nx.cycle_graph(7); graph.add_edges_from([(0, 2), (2, 5)])
    x = np.zeros(7, np.int16); before = nx.to_numpy_array(graph, dtype=np.int16)
    proposals = [e for e in candidates(before, cfg["refinement"], np.random.default_rng(9)) if nx.is_connected(nx.from_numpy_array(e))]
    from grapher.models.gdsm_simple.categorical.multiscale import count_multi
    source_counts = count_multi(x, before)
    destination = next(e for e in proposals if count_multi(x, e) != source_counts)
    dest_counts = count_multi(x, destination)
    combined = {k: source_counts[k]+dest_counts[k] for k in (3, 4, 5)}
    basis, _ = fit_basis(combined, cfg["graphlets"])
    hist, mass = basis.summary(x, destination)
    clustering, orbit = topology_summary(destination, cfg["graphlets"]["clustering_bins"])
    target = {"histogram": hist, "mass": mass, "clustering": clustering, "orbit": orbit,
              "weighted_adjacency": codec.encode(destination)}
    after, diag = refine(x, before, target, basis, codec, cfg["graphlets"]["clustering_bins"], cfg["refinement"], np.random.default_rng(9))
    assert diag["accepted_steps"] > 0
    assert diag["final"]["total"] < diag["initial"]["total"]
    assert diag["final"]["structure"] < diag["initial"]["structure"]
    assert diag["degree_preserved"] and diag["typed_degrees_preserved"]
    assert nx.is_connected(nx.from_numpy_array(after))


def test_real_cli_entrypoints_do_not_delegate_to_gdsm(tmp_path):
    cfg = prepared(tmp_path)
    p = tmp_path/"fixture.yaml"; p.write_text(yaml.safe_dump({"option_c": cfg}))
    env = {**os.environ, "PYTHONPATH": str(ROOT/"src"), "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    out = tmp_path/"train_cli"
    run = subprocess.run([sys.executable, str(ROOT/"scripts/train_option_c.py"), "--config", str(p),
                          "--output-dir", str(out), "--seed", "42", "--device", "cpu", "--epochs", "1",
                          "--cpu-threads", "1"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stdout+run.stderr
    run = subprocess.run([sys.executable, str(ROOT/"scripts/generate_option_c.py"), "--config", str(p),
                          "--checkpoint", str(out/"checkpoints/best.pt"), "--output-dir", str(tmp_path/"gen_cli"),
                          "--seed", "42", "--device", "cpu", "--num-generate", "3", "--steps", "2", "--no-refine",
                          "--cpu-threads", "1"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stdout+run.stderr
    assert len(read_graphs(tmp_path/"gen_cli/base_graphs.pkl")) == 3
    for script in ("train_option_c.py", "generate_option_c.py"):
        source = (ROOT/"scripts"/script).read_text()
        assert "external_cli" not in source and "run_gdsm_simple_baseline" not in source


def test_no_implicit_dataset_build_and_safe_overwrite(tmp_path):
    cfg = prepared(tmp_path)
    foreign = tmp_path/"foreign"; foreign.mkdir(); (foreign/"keep.txt").write_text("keep")
    with pytest.raises(FileExistsError): train(cfg, foreign, seed=42, device="cpu", overwrite=True)
    assert (foreign/"keep.txt").read_text() == "keep"
    val = Path(cfg["dataset"]["root"])/cfg["dataset"]["name"]/"val.pkl"
    val.unlink()
    with pytest.raises(FileNotFoundError, match="Missing prepared"): train(cfg, tmp_path/"failed", seed=42, device="cpu")
    assert not val.exists()
    assert json.loads((tmp_path/"failed/manifest.json").read_text())["status"] == "failed"
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="GPU requested"): resolve_device("gpu")


def test_spectral_float64_gradient_check():
    mask = torch.ones(1, 3, dtype=torch.bool)
    raw = torch.tensor([[[0., .7, .13], [.7, 0., .4], [.13, .4, 0.]]], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([[-.9, -.1, 1.]], dtype=torch.float64)
    assert torch.autograd.gradcheck(lambda v: weighted_spectral_loss(symmetrize(v, mask), target, mask),
                                   (raw,), atol=1e-5, rtol=1e-4)


def test_zero_spectral_weight_does_not_call_eigensolver(monkeypatch):
    cfg = config(); gs = graphs(cfg)[1:3]
    rows, _, schema = prepare_records(gs, gs, cfg)
    basis = TypedGraphletsMulti.from_schema(schema); batch = collate(rows, basis, device="cpu")
    model = OptionCDenoiser(**model_config(cfg, schema, basis))
    pred = model(batch["x"], batch["w"], torch.tensor([3, 4]), batch["mask"], 8)
    weights = dict(cfg["loss_weights"]); weights["spectral"] = 0.
    def forbidden(*args, **kwargs):
        raise AssertionError("Unexpected eigensolve")
    monkeypatch.setattr(torch.linalg, "eigvalsh", forbidden)
    loss, parts = joint_loss(pred, batch, basis, weights, cfg["spectral"])
    assert torch.isfinite(loss) and parts["spectral"] == 0.


@pytest.mark.parametrize("dataset", ["community_small", "ego_small", "qm9", "zinc"])
def test_full_width_model_at_dataset_max_nodes(dataset):
    cfg = load_config(ROOT / f"configs/experiments/option_c/{dataset}.yaml")
    n = cfg["model"]["max_nodes"]
    small = graphs(cfg)[1:3]
    _, _, schema = prepare_records(small, small, cfg)
    basis = TypedGraphletsMulti.from_schema(schema)
    model = OptionCDenoiser(**model_config(cfg, schema, basis))
    mask = torch.ones(1, n, dtype=torch.bool)
    x = torch.zeros(1, n, dtype=torch.long)
    w = symmetric_noise(mask, torch.Generator().manual_seed(4))
    pred = model(x, w, torch.tensor([250]), mask, 500)
    assert pred["clean_adjacency"].shape == (1, n, n)
    (pred["clean_adjacency"].square().mean()+pred["node_logits"].square().mean()).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())

"""Probability-path and checkpoint regression tests using the supplied transformer."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
WORKERS = ROOT / "src/grapher/models/external_workers"
SOURCE = ROOT / "external/catflow"


def load_local(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def worker(monkeypatch):
    monkeypatch.syspath_prepend(str(SOURCE))
    monkeypatch.syspath_prepend(str(WORKERS))
    # The native transformer imports 'utils'; do not retain a different wrapper's module.
    for key in list(sys.modules):
        if key in {"utils", "flow_matching", "models"} or key.startswith("models."):
            monkeypatch.delitem(sys.modules, key, raising=False)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    module = load_local("_catflow_worker_test", WORKERS / "catflow.py")
    yield module
    torch.set_num_threads(old_threads)


@pytest.fixture
def path_module(worker):
    return load_local("_catflow_path_test", WORKERS / "catflow_path.py")


@pytest.mark.parametrize("edge", [False, True])
def test_linear_path_exact_endpoints_and_identity(path_module, edge):
    target = torch.zeros((5, 4, 4, 2) if edge else (5, 4, 2))
    target[..., 0] = 1
    t_shape = (5, 1, 1, 1) if edge else (5, 1, 1)
    assert torch.equal(path_module.linear_interpolant(target, torch.ones(t_shape), edge), target)
    torch.manual_seed(41)
    z = path_module.linear_interpolant(target, torch.zeros(t_shape), edge)
    torch.manual_seed(41)
    xt = path_module.linear_interpolant(target, torch.full(t_shape, .95), edge)
    assert torch.allclose(xt, .05 * z + .95 * target)
    assert torch.allclose((target - xt) / .05, target - z, atol=4e-6)
    if edge:
        assert torch.equal(xt, xt.transpose(1, 2))


def test_legacy_probe_reproduces_mismatch_without_consuming_rng(worker, path_module):
    _, flow = worker.upstream({"source_root": str(SOURCE)})
    before = torch.get_rng_state().clone()
    probe = path_module.probe_upstream_path(flow)
    assert torch.equal(before, torch.get_rng_state())
    assert .49 < probe["node_residual_std_at_t1"] < .51
    assert .34 < probe["edge_residual_std_at_t1"] < .37


def test_probability_path_choice_is_explicit(worker, path_module):
    _, flow = worker.upstream({"source_root": str(SOURCE)})
    x = torch.zeros(2, 4, 2)
    t = torch.ones(2, 1, 1)
    assert torch.equal(path_module.training_interpolant(x, t, "linear", flow), x)
    assert not torch.equal(path_module.training_interpolant(x, t, "upstream", flow), x)
    with pytest.raises(ValueError):
        path_module.training_interpolant(x, t, "typo", flow)


def test_no_edge_and_padding_are_distinct(worker):
    data = {"adjacency": np.zeros((2, 5, 5), dtype=np.int8),
            "node_types": np.asarray([[0, 0, 0, -1, -1], [0, -1, -1, -1, -1]]),
            "num_nodes": np.asarray([3, 1])}
    data["adjacency"][0, 0, 1] = data["adjacency"][0, 1, 0] = 1
    x, e, mask, pair = worker.dense_batch(data, [0, 1], torch.device("cpu"), 1, 2)
    assert e[0, 0, 2].tolist() == [1, 0]  # valid no-edge
    assert e[0, 0, 1].tolist() == [0, 1]
    assert e[0, 0, 4].tolist() == [0, 0]  # padding
    assert e[0, 0, 0].tolist() == [0, 0]  # diagonal
    assert pair.sum().item() == 6
    assert mask.sum().item() == 4
    assert x[0, 3:].sum() == 0


def tiny_data():
    adjacency = np.zeros((3, 5, 5), dtype=np.int8)
    for i, n in enumerate((3, 4, 5)):
        for u in range(n):
            adjacency[i, u, (u + 1) % n] = adjacency[i, (u + 1) % n, u] = 1
    types = np.asarray([[0, 0, 0, -1, -1], [0, 0, 0, 0, -1], [0, 0, 0, 0, 0]])
    return {"adjacency": adjacency, "node_types": types, "num_nodes": np.asarray([3, 4, 5])}


def test_native_network_gradients_and_validation_restore(worker):
    from types import SimpleNamespace
    utils, flow = worker.upstream({"source_root": str(SOURCE)})
    model = utils.get_GT_model(SimpleNamespace(num_layers=1, small_model=1, task="abstract"), 1, 2)
    cls, _ = worker.ema_class("bundled")
    ema = cls(model.parameters(), decay=.999)
    data = tiny_data()
    loss, lx, le = worker.batch_loss(model, data, np.asarray([0, 1]), torch.device("cpu"), 1, 2, flow, "linear")
    loss.backward()
    assert lx == 0 and le > 0 and torch.isfinite(loss)
    assert any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in model.parameters())
    # Make EMA and raw weights different, then ensure temporary evaluation restores raw weights.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(.001)
    before = [p.detach().clone() for p in model.parameters()]
    rng_before = torch.get_rng_state().clone()
    a = worker.validation_loss(model, ema, data, torch.device("cpu"), 1, 2, flow, "linear", 2, 2, 123)
    assert model.training
    assert all(torch.equal(p, old) for p, old in zip(model.parameters(), before))
    assert torch.equal(rng_before, torch.get_rng_state())
    b = worker.validation_loss(model, ema, data, torch.device("cpu"), 1, 2, flow, "linear", 2, 2, 123)
    assert a == b


def test_snapshot_and_best_last_selection(worker):
    value = {"weights": torch.tensor([1.])}
    saved = worker.cpu_snapshot(value)
    value["weights"].fill_(2.)
    assert saved["weights"].item() == 1.
    state = {"epoch": 10, "best": {"epoch": 3}}
    assert worker.select_weights(state, {}) == ({"epoch": 3}, "best")
    assert worker.select_weights(state, {"checkpoint_selection": "last"})[0]["epoch"] == 10
    assert worker.select_weights({"epoch": 4}, {}) == ({"epoch": 4}, "last")
    with pytest.raises(ValueError):
        worker.select_weights({"epoch": 4}, {"checkpoint_selection": "best"})


def test_native_training_reload_generation_subprocess(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for split in ("train", "val"):
        np.savez_compressed(data_dir / (split + ".npz"), **tiny_data())
    options = {"train": {"epochs": 3, "batch_size": 2, "path": "linear", "validation_every": 1,
                         "validation_repeats": 2, "checkpoint_every": 1, "log_every": 1},
               "model": {"num_layers": 1, "small_model": 1},
               "sample": {"method": "euler", "steps": 8, "use_ema": True},
               "runtime": {"device": "cpu"}, "generation_batch_size": 2}
    job = {"stage": "train", "model": "catflow", "seed": 42, "source_root": str(SOURCE),
           "dataset_dir": str(data_dir), "checkpoint": str(tmp_path / "model.pt"),
           "worker_manifest": str(tmp_path / "train.json"), "options": options,
           "profile": {"atomic_numbers": [], "bond_types": [], "domain": "generic", "max_nodes": 5}}
    job_path = tmp_path / "job.json"
    job_path.write_text(json.dumps(job))
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(SOURCE))
    def run():
        p = subprocess.run([sys.executable, str(WORKERS / "catflow.py"), str(job_path)],
                           cwd=tmp_path, env=env, text=True, capture_output=True)
        assert p.returncode == 0, p.stdout + p.stderr
    run()
    state = torch.load(tmp_path / "model.pt", weights_only=False)
    assert state["epoch"] == 3 and state["optimizer_steps"] == 6
    assert state["best"]["epoch"] == min(state["history"], key=lambda r: r["val_ema"])["epoch"]
    assert state["probability_path"]["name"] == "linear"
    job.update(stage="generate", seed=7, num_graphs=5, output=str(tmp_path / "samples.npz"),
               worker_manifest=str(tmp_path / "generate.json"))
    job_path.write_text(json.dumps(job)); run()
    with np.load(tmp_path / "samples.npz") as out:
        assert out["adjacency"].shape == (5, 5, 5)
        assert np.array_equal(out["adjacency"], out["adjacency"].transpose(0, 2, 1))
        assert not np.diagonal(out["adjacency"], axis1=1, axis2=2).any()
        for x, n in zip(out["node_types"], out["num_nodes"]):
            assert np.all(x[n:] == -1)
    rec = json.loads((tmp_path / "generate.json").read_text())
    assert rec["checkpoint_selection"] == "best"
    assert rec["sampled_epoch"] == state["best"]["epoch"]
    assert rec["total_nfe"] == 24  # 3 batches x 8 Euler evaluations

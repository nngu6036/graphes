"""Contract tests do not substitute mock neural models in production workers."""
from __future__ import annotations
import copy
import json
import pickle
import subprocess
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import yaml

from grapher.models import (ArtifactCollisionError, DatasetReference, GenerateRequest,
                            RunSpec, TrainRequest, create_baseline, normalize_baseline_id)
from grapher.models.external_codec import GraphProfile, PROFILES, encode_graphs, save_graphs, decode_graphs
from grapher.models.external_cli import build_parser


@pytest.fixture
def profile():
    return GraphProfile("community_small", "toy", 6)


def test_roundtrip_retains_isolates_and_node_count(tmp_path, profile):
    graph = nx.path_graph(3)
    graph.add_node("isolated")
    path = tmp_path / "graphs.npz"
    save_graphs(path, [graph, nx.empty_graph(1)], profile)
    out = decode_graphs(path, profile, 2)
    assert [len(g) for g in out] == [4, 1]
    assert [g.number_of_edges() for g in out] == [2, 0]
    assert sum(d == 0 for _, d in out[0].degree()) == 1


@pytest.mark.parametrize("kind", ["count", "symmetry", "diagonal", "padding", "float", "category", "size"])
def test_bad_exports_are_rejected(tmp_path, profile, kind):
    arrays = encode_graphs([nx.path_graph(3)], profile)
    if kind == "symmetry": arrays["adjacency"][0, 0, 1] = 0
    if kind == "diagonal": arrays["adjacency"][0, 0, 0] = 1
    if kind == "padding": arrays["node_types"][0, 5] = 0
    if kind == "float": arrays["adjacency"] = arrays["adjacency"].astype(float)
    if kind == "category": arrays["adjacency"][0, 0, 1] = arrays["adjacency"][0, 1, 0] = 3
    if kind == "size": arrays["num_nodes"][0] = 7
    path = tmp_path / "bad.npz"
    np.savez_compressed(path, **arrays)
    with pytest.raises((ValueError, TypeError)):
        decode_graphs(path, profile, 2 if kind == "count" else 1)


@pytest.mark.parametrize("kind", ["directed", "multigraph", "selfloop", "empty", "large"])
def test_input_corrections_are_not_silent(profile, kind):
    graph = {"directed": nx.DiGraph([(0, 1)]), "multigraph": nx.MultiGraph([(0, 1)]),
             "selfloop": nx.Graph([(0, 0)]), "empty": nx.Graph(), "large": nx.path_graph(7)}[kind]
    with pytest.raises(ValueError): encode_graphs([graph], profile)


@pytest.mark.parametrize("dataset", ["qm9", "zinc"])
def test_molecule_categories_roundtrip(tmp_path, dataset):
    p = PROFILES[dataset]
    graph = nx.Graph()
    graph.add_node("C", atomic_num=6)
    graph.add_node("O", atomic_num=8)
    graph.add_edge("C", "O", bond_type=2)
    path = tmp_path / "molecule.npz"
    save_graphs(path, [graph], p)
    result = decode_graphs(path, p, 1)[0]
    assert result.nodes[0]["atomic_num"] == 6
    assert result.nodes[1]["atomic_num"] == 8
    assert result.edges[0, 1]["bond_type"] == 2
    assert all(a["formal_charge"] == 0 for _, a in result.nodes(data=True))
    graph.edges["C", "O"]["bond_type"] = 4
    with pytest.raises(ValueError, match="unsupported bond"):
        encode_graphs([graph], p)


@pytest.mark.parametrize("model", ["catflow", "gdsm", "edge", "spectre"])
def test_cli_and_datasets(model):
    parser = build_parser(model)
    args = parser.parse_args(["--stage", "train", "--dataset", "community_small", "--n-epochs", "3"])
    assert args.epochs == 3
    assert parser.parse_args(["--stage", "generate", "--dataset", "ego_small"]).stage == "generate"
    for name in create_baseline(model).supported_datasets:
        path = Path(__file__).parents[1] / "configs" / "baselines" / f"{model}_{name}.yaml"
        assert yaml.safe_load(path.read_text())[model]["train"]["epochs"] > 0


def test_gsdm_alias_and_lazy_exports():
    assert normalize_baseline_id("GSDM") == "gdsm"
    assert normalize_baseline_id("GDSM") == "gdsm"
    script = '''import sys
from grapher.models import CatFlowWrapper, GDSMWrapper, GSDMWrapper, EDGEWrapper, SPECTREWrapper
assert GDSMWrapper is GSDMWrapper
assert "torch" not in sys.modules
assert "torch_geometric" not in sys.modules
'''
    subprocess.run([sys.executable, "-c", script], check=True)


def prepared(tmp_path, model):
    wrapper = create_baseline(model)
    source = tmp_path / "source"
    for filename in wrapper.source_markers:
        path = source / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Unit-test source identity fixture, not an executable backend.\n")
    data = tmp_path / "datasets" / "toy"
    data.mkdir(parents=True)
    for split in ("train", "val", "test"):
        (data / (split + ".pkl")).write_bytes(pickle.dumps([nx.path_graph(4), nx.cycle_graph(5)]))
    run = RunSpec.for_seed(model_id=model, dataset_id="community_small", seed=42, output_root=tmp_path / "out")
    options = {"source_root": str(source), "python": sys.executable, "max_nodes": 6,
               "train": {"epochs": 1, "batch_size": 2},
               "runtime": {"device": "cpu", "progress": {"enabled": False}}}
    request = TrainRequest(run=run, dataset=DatasetReference("community_small", root=tmp_path / "datasets", serialized_id="toy"), options=options)
    return wrapper, request, source


def install_worker_contract_double(monkeypatch, wrapper, calls, wrong_count=False):
    # Unit-test subprocess completion, NOT a fallback in the real wrappers.
    def fake_worker(job, python, work, log):
        calls.append(copy.deepcopy(job))
        log.write_text("contract test only\n")
        if job["stage"] == "train":
            path = Path(job["checkpoint"]); path.parent.mkdir(parents=True)
            path.write_bytes(b"contract-test-checkpoint")
        else:
            p = GraphProfile("community_small", "toy", job["profile"]["max_nodes"])
            count = job["num_graphs"] - int(wrong_count)
            save_graphs(Path(job["output"]), [nx.empty_graph(4) for _ in range(count)], p)
        Path(job["worker_manifest"]).write_text(json.dumps({"format": "grapher_external_worker_v1", "model": wrapper.model_id, "stage": job["stage"]}))
    monkeypatch.setattr(wrapper, "_run_worker", fake_worker)


@pytest.mark.parametrize("model", ["catflow", "gdsm", "edge", "spectre"])
def test_train_generate_contract_and_reuse(tmp_path, monkeypatch, model):
    wrapper, request, source = prepared(tmp_path, model)
    calls = []
    install_worker_contract_double(monkeypatch, wrapper, calls)
    trained = wrapper.train(request)
    assert len(calls) == 1
    manifest = json.loads(trained.manifest_path.read_text())
    assert manifest["dataset"]["fingerprint"] == request.dataset.fingerprint()
    assert not (trained.manifest_path.parent / "native_dataset" / "test.npz").exists()
    assert set(manifest["data_conversion"]) == {"train", "val"}
    assert wrapper.train(request).checkpoint_path == trained.checkpoint_path
    assert len(calls) == 1
    generate = GenerateRequest(run=request.run, checkpoint_path=trained.checkpoint_path, num_graphs=3, generation_seed=7)
    result = wrapper.generate(generate)
    assert result.num_generated == 3
    graphs = pickle.loads(result.graphs_path.read_bytes())
    assert all(len(g) == 4 and g.number_of_edges() == 0 for g in graphs)
    assert json.loads(result.manifest_path.read_text())["posthoc_repair"] is False
    with pytest.raises(ArtifactCollisionError): wrapper.generate(generate)
    # Invariants protect existing artifacts against changed training/source data.
    (source / wrapper.source_markers[0]).write_text("# Changed source\n")
    with pytest.raises(RuntimeError, match="source changed"):
        wrapper.generate(GenerateRequest(run=request.run, checkpoint_path=trained.checkpoint_path, num_graphs=1, generation_seed=8))


@pytest.mark.parametrize("tamper", ["checkpoint", "data", "count"])
def test_no_publication_of_corrupt_generation(tmp_path, monkeypatch, tamper):
    wrapper, request, _ = prepared(tmp_path, "catflow")
    calls = []
    install_worker_contract_double(monkeypatch, wrapper, calls, wrong_count=tamper == "count")
    trained = wrapper.train(request)
    if tamper == "checkpoint": trained.checkpoint_path.write_bytes(b"changed")
    if tamper == "data": (trained.manifest_path.parent / "native_dataset/train.npz").write_bytes(b"changed")
    gen = GenerateRequest(run=request.run, checkpoint_path=trained.checkpoint_path, num_graphs=3, generation_seed=8)
    with pytest.raises((RuntimeError, ValueError)): wrapper.generate(gen)
    assert not request.run.layout.generation_dir(gen.resolved_generation_id).exists()


def test_domain_and_options_fail_loudly(tmp_path):
    for name in ("edge", "gdsm"):
        with pytest.raises(ValueError): create_baseline(name)._profile("qm9", {})
    with pytest.raises(ValueError): create_baseline("spectre")._profile("zinc", {})
    wrapper, request, _ = prepared(tmp_path, "catflow")
    with pytest.raises(ValueError, match="corrector-training"):
        wrapper._options(TrainRequest(run=request.run, dataset=request.dataset, options={"training_estimates": {"enabled": True}}))
    with pytest.raises(ValueError, match="Unknown"):
        wrapper._options(TrainRequest(run=request.run, dataset=request.dataset, options={"num_epohcs": 1}))


def test_ema_fallback_matches_bundled_recurrence():
    import importlib.util
    import torch
    path = Path(__file__).parents[1] / "src/grapher/models/external_workers/ema_compat.py"
    spec = importlib.util.spec_from_file_location("grapher_test_ema", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    value = torch.nn.Parameter(torch.tensor([1., 2.]))
    ema = module.ExponentialMovingAverage([value], .999)
    with torch.no_grad(): value.copy_(torch.tensor([3., 4.]))
    ema.update()
    expected = torch.tensor([1., 2.]) * (2 / 11) + torch.tensor([3., 4.]) * (9 / 11)
    assert torch.allclose(ema.shadow_params[0], expected)
    with ema.average_parameters(): assert torch.allclose(value, expected)
    assert torch.equal(value, torch.tensor([3., 4.]))
    cloned = module.ExponentialMovingAverage([value], .9)
    cloned.load_state_dict(ema.state_dict()); cloned.to(dtype=torch.float64)
    assert cloned.shadow_params[0].dtype == torch.float64
    assert cloned.num_updates == 1

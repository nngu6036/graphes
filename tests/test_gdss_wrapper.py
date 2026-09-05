from __future__ import annotations

import hashlib
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import numpy as np
import yaml

from grapher.models.gdss.codec import profile_for
from grapher.models.gdss.wrapper import GDSSWrapper, _environment, _resolved_gdss_config
from grapher.models.gdss.workers.train import _build_loader


def _generic_source_config(path: Path) -> Path:
    config = {
        "data": {
            "data": "community_small", "dir": "./data", "batch_size": 128,
            "test_split": 0.2, "max_node_num": 20, "max_feat_num": 10, "init": "deg",
        },
        "sde": {"x": {"type": "VP", "beta_min": 0.1, "beta_max": 1.0, "num_scales": 10},
                "adj": {"type": "VP", "beta_min": 0.1, "beta_max": 1.0, "num_scales": 10}},
        "model": {"x": "ScoreNetworkX", "adj": "ScoreNetworkA", "conv": "GCN", "num_heads": 4,
                  "depth": 2, "adim": 16, "nhid": 16, "num_layers": 3, "num_linears": 2,
                  "c_init": 2, "c_hid": 8, "c_final": 4},
        "train": {"name": "test", "num_epochs": 50, "save_interval": 10, "print_interval": 10,
                  "reduce_mean": False, "lr": 0.01, "lr_schedule": True, "ema": 0.999,
                  "weight_decay": 0.0001, "grad_norm": 1.0, "lr_decay": 0.999, "eps": 1e-5},
        "sampler": {"predictor": "Euler", "corrector": "None", "snr": 0.0, "scale_eps": 0.0, "n_steps": 1},
        "sample": {"use_ema": False, "noise_removal": True, "probability_flow": False, "eps": 1e-4, "seed": 1},
    }
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_capabilities_are_ready_without_importing_upstream() -> None:
    wrapper = GDSSWrapper()
    assert wrapper.capabilities.status == "ready"
    assert wrapper.capabilities.isolation == "subprocess"
    assert wrapper.capabilities.domains == frozenset({"generic", "attributed"})


def test_resolved_generic_config_uses_final_epoch_and_runtime_overrides(tmp_path: Path) -> None:
    source = _generic_source_config(tmp_path / "community_small.yaml")
    resolved = _resolved_gdss_config(
        source_path=source,
        sampling_path=None,
        options={"batch_size": 32, "num_workers": 2, "train": {"num_epochs": 123}},
        profile=profile_for("community_small"),
        run_id="seed_42",
        seed=42,
        progress={"epoch_interval": 17},
    )
    assert resolved["data"]["data"] == "community_small"
    assert resolved["data"]["batch_size"] == 32
    assert resolved["data"]["num_workers"] == 2
    assert resolved["train"]["num_epochs"] == 123
    assert resolved["train"]["save_interval"] == 123
    assert resolved["train"]["print_interval"] == 17
    assert resolved["sample"]["seed"] == 42


def test_molecular_sampling_config_is_injected_without_changing_model_shape(tmp_path: Path) -> None:
    source = {
        "data": {"data": "QM9", "dir": "./data", "batch_size": 1024, "max_node_num": 9, "max_feat_num": 4, "init": "atom"},
        "sde": {"x": {"type": "VE", "beta_min": 0.1, "beta_max": 1.0, "num_scales": 10},
                "adj": {"type": "VE", "beta_min": 0.1, "beta_max": 1.0, "num_scales": 10}},
        "model": {},
        "train": {"num_epochs": 3, "save_interval": 1, "print_interval": 1},
    }
    source_path = tmp_path / "qm9.yaml"
    source_path.write_text(yaml.safe_dump(source), encoding="utf-8")
    sample_path = tmp_path / "sample_qm9.yaml"
    sample_path.write_text(yaml.safe_dump({
        "sampler": {"predictor": "Reverse", "corrector": "Langevin", "snr": 0.2, "scale_eps": 0.7, "n_steps": 1},
        "sample": {"use_ema": False, "noise_removal": True, "probability_flow": False, "eps": 1e-4, "seed": 99},
    }), encoding="utf-8")
    resolved = _resolved_gdss_config(
        source_path=source_path, sampling_path=sample_path, options={}, profile=profile_for("qm9"),
        run_id="seed_7", seed=7, progress={"epoch_interval": None},
    )
    assert resolved["data"]["data"] == "QM9"
    assert resolved["data"]["max_node_num"] == 9
    assert resolved["data"]["max_feat_num"] == 4
    assert resolved["sampler"]["predictor"] == "Reverse"
    assert resolved["sample"]["seed"] == 7


def test_ego_wrapper_expands_upstream_degree_feature_space(tmp_path: Path) -> None:
    source = yaml.safe_load(
        _generic_source_config(tmp_path / "ego_small.yaml").read_text(
            encoding="utf-8"
        )
    )
    source["data"].update(
        data="ego_small",
        max_node_num=18,
        max_feat_num=17,
    )
    source_path = tmp_path / "ego_small.yaml"
    source_path.write_text(yaml.safe_dump(source), encoding="utf-8")
    repository_root = Path(__file__).resolve().parents[1]
    wrapper_options = yaml.safe_load(
        (repository_root / "configs/baselines/gdss_ego_small.yaml").read_text(
            encoding="utf-8"
        )
    )["gdss"]

    resolved = _resolved_gdss_config(
        source_path=source_path,
        sampling_path=None,
        options=wrapper_options,
        profile=profile_for("ego_small"),
        run_id="seed_42",
        seed=42,
        progress={"epoch_interval": None},
    )

    assert resolved["data"]["max_node_num"] == 18
    assert resolved["data"]["max_feat_num"] == 18


def test_cpu_environment_hides_cuda() -> None:
    env, require_cuda = _environment(
        Path("/tmp/gdss"), seed=9, device="cpu", cuda_visible_devices="3"
    )
    assert require_cuda is False
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert env["PYTHONHASHSEED"] == "9"


def test_ego_degree_17_uses_the_eighteenth_feature_channel(tmp_path: Path) -> None:
    adjacency = np.zeros((1, 18, 18), dtype=np.int8)
    adjacency[0] = nx.to_numpy_array(nx.star_graph(17), dtype=np.int8)
    np.savez_compressed(
        tmp_path / "ego.npz",
        adjacency=adjacency,
        num_nodes=np.asarray([18], dtype=np.int64),
        sample_index=np.asarray([0], dtype=np.int64),
    )
    config = SimpleNamespace(
        data=SimpleNamespace(max_feat_num=18, batch_size=1, num_workers=0)
    )

    features, _adjacency = next(
        iter(_build_loader(tmp_path / "ego.npz", config, domain="generic", shuffle=False))
    )

    assert tuple(features.shape) == (1, 18, 18)
    assert features[0, 0, 17].item() == 1.0


def test_wrapper_train_generate_transaction(tmp_path: Path, monkeypatch) -> None:
    import grapher.models.gdss.wrapper as wrapper_module
    from grapher.models import DatasetReference, GenerateRequest, RunSpec, TrainRequest

    source_root = tmp_path / "gdss"
    (source_root / "config").mkdir(parents=True)
    _generic_source_config(source_root / "config" / "community_small.yaml")
    python_path = Path(sys.executable).resolve()
    monkeypatch.setattr(wrapper_module, "resolve_gdss_root", lambda *_args, **_kwargs: source_root)
    monkeypatch.setattr(wrapper_module, "resolve_gdss_python", lambda **_kwargs: python_path)
    monkeypatch.setattr(wrapper_module, "_source_identity", lambda _root: {
        "source_root": str(source_root), "revision": None, "files": {},
        "source_fingerprint": "fake-source", "integration_mode": "test",
    })
    monkeypatch.setattr(wrapper_module, "_python_identity", lambda _python: {
        "python_executable": str(python_path), "python_version": "test",
    })

    dataset_root = tmp_path / "datasets"
    dataset_dir = dataset_root / "sbm"
    dataset_dir.mkdir(parents=True)
    for split, graphs in {
        "train": [nx.path_graph(4), nx.cycle_graph(5)],
        "val": [nx.path_graph(5)],
        "test": [nx.complete_graph(3)],
    }.items():
        with (dataset_dir / f"{split}.pkl").open("wb") as handle:
            pickle.dump(graphs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    wrapper = GDSSWrapper()

    def fake_run_external(command, **_kwargs):
        command = list(command)
        if command[1].endswith("train.py"):
            checkpoint = Path(command[command.index("--checkpoint") + 1])
            manifest = Path(command[command.index("--manifest") + 1])
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(b"gdss-checkpoint")
            manifest.write_text(json.dumps({
                "format": "grapher_gdss_training_worker_v1",
                "configured_num_epochs": 50,
                "configured_batch_size": 128,
                "optimizer_split": "train", "monitor_split": "val",
                "test_used_during_training": False, "device": "cpu",
                "checkpoint": {"path": str(checkpoint), "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()},
            }), encoding="utf-8")
            return
        if command[1].endswith("generate.py"):
            output = Path(command[command.index("--output") + 1])
            manifest = Path(command[command.index("--manifest") + 1])
            count = int(command[command.index("--num-graphs") + 1])
            adjacency = np.zeros((count, 20, 20), dtype=np.int8)
            adjacency[:, 0, 1] = 1
            adjacency[:, 1, 0] = 1
            output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(output, adjacency=adjacency, num_nodes=np.full((count,), 2, dtype=np.int64), sample_index=np.arange(count, dtype=np.int64))
            manifest.write_text(json.dumps({
                "format": "grapher_gdss_export_v1", "num_generated": count,
                "batch_size": count, "sampling_rounds": 1, "device": "cpu",
                "postprocessing": "test",
                "output": {"path": str(output), "sha256": hashlib.sha256(output.read_bytes()).hexdigest()},
            }), encoding="utf-8")
            return
        raise AssertionError(command)

    monkeypatch.setattr(wrapper, "_run_external", fake_run_external)
    run = RunSpec.for_seed(model_id="gdss", dataset_id="community_small", seed=42, output_root=tmp_path / "baselines")
    dataset = DatasetReference("community_small", root=dataset_root, serialized_id="sbm", native_id="community_small")
    training = wrapper.train(TrainRequest(
        run=run, dataset=dataset,
        options={"runtime": {"device": "cpu", "progress": {"enabled": False}}, "training_estimates": {"enabled": False}},
    ))
    assert training.checkpoint_path.is_file()
    assert (training.run_dir / "train" / "native_dataset" / "train.npz").is_file()
    generation = wrapper.generate(GenerateRequest(
        run=run, checkpoint_path=training.checkpoint_path, num_graphs=3, generation_seed=9,
        options={"runtime": {"device": "cpu", "progress": {"enabled": False}}},
    ))
    assert generation.num_generated == 3
    assert all(path.is_file() for path in generation.native_artifacts)
    with generation.graphs_path.open("rb") as handle:
        graphs = pickle.load(handle)
    assert len(graphs) == 3
    assert all(graph.number_of_edges() == 1 for graph in graphs)

from __future__ import annotations

import hashlib
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import pytest

from grapher.models import DatasetReference, GenerateRequest, RunSpec, TrainRequest
from grapher.models.graphrnn.wrapper import (
    GENERATION_MANIFEST_FORMAT,
    TRAINING_ESTIMATES_MANIFEST_FORMAT,
    TRAINING_MANIFEST_FORMAT,
    GraphRNNWrapper,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fake_root(tmp_path: Path) -> Path:
    root = tmp_path / "GraphRNN"
    root.mkdir()
    for name in ("model.py", "data.py", "train.py", "args.py", "README.md"):
        (root / name).write_text(f"# {name} fixture\n", encoding="utf-8")
    return root


def _prepared_dataset(tmp_path: Path) -> DatasetReference:
    directory = tmp_path / "datasets" / "sbm"
    directory.mkdir(parents=True)
    values = {
        "train": [nx.path_graph(3), nx.cycle_graph(4)],
        "val": [nx.path_graph(4)],
        "test": [nx.cycle_graph(5)],
    }
    for split, graphs in values.items():
        with (directory / f"{split}.pkl").open("wb") as handle:
            pickle.dump(graphs, handle)
    return DatasetReference(
        "community_small",
        root=tmp_path / "datasets",
        serialized_id="sbm",
        native_id="community_small",
    )


def _fake_generation(output_dir: Path, count: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    export = output_dir / "graphrnn_samples.npz"
    export.write_bytes(b"neutral")
    manifest = output_dir / "graphrnn_manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    log = output_dir / "graphrnn.log"
    log.write_text("generated\n", encoding="utf-8")
    graphs = [nx.path_graph(3 + (index % 2)) for index in range(count)]
    for index, graph in enumerate(graphs):
        graph.graph["graphrnn_sample_index"] = index
    return SimpleNamespace(
        graphs=graphs,
        export_path=export,
        manifest_path=manifest,
        log_path=log,
        export_sha256=_sha(export),
        manifest={
            "format": "grapher_graphrnn_export_v1",
            "num_generated": count,
            "statistics": {
                "variant": "GraphRNN_RNN",
                "empty_graph_count": 0,
            },
        },
    )


def test_train_and_generate_publish_common_artifacts(monkeypatch, tmp_path: Path) -> None:
    root = _fake_root(tmp_path)
    monkeypatch.setenv("GRAPHRNN", str(root))
    monkeypatch.setenv("GRAPHRNN_PYTHON", sys.executable)
    dataset = _prepared_dataset(tmp_path)
    run = RunSpec.for_seed(
        model_id="graphrnn",
        dataset_id="community_small",
        seed=42,
        output_root=tmp_path / "outputs" / "baselines",
    )
    wrapper = GraphRNNWrapper()

    def fake_external(command, **kwargs):
        assert kwargs["label"] == "GraphRNN training"
        output = Path(command[command.index("--output-dir") + 1])
        worker_manifest = Path(command[command.index("--manifest") + 1])
        checkpoints = output / "checkpoints"
        checkpoints.mkdir(parents=True, exist_ok=True)
        checkpoint = checkpoints / "graphrnn.pt"
        checkpoint.write_bytes(b"checkpoint")
        (checkpoints / "graphrnn_epoch_3.pt").write_bytes(b"epoch")
        (output / "loss_history.jsonl").write_text(
            '{"epoch": 3, "loss": 0.5}\n', encoding="utf-8"
        )
        worker_manifest.write_text(
            json.dumps(
                {
                    "format": "grapher_graphrnn_training_worker_v1",
                    "checkpoint": str(checkpoint),
                    "checkpoint_epoch": 3,
                    "configured_epochs": 3,
                    "resumed_from_epoch": None,
                    "last_loss": 0.5,
                    "device": "cpu",
                    "resolved_config": {
                        "variant": "GraphRNN_RNN",
                        "max_num_node": 5,
                        "max_prev_node": 4,
                        "hidden_size_rnn": 16,
                        "hidden_size_rnn_output": 8,
                        "embedding_size_rnn": 8,
                        "embedding_size_rnn_output": 4,
                        "embedding_size_output": 8,
                        "num_layers": 2,
                        "batch_size": 2,
                        "batch_ratio": 1,
                        "epochs": 3,
                        "learning_rate": 0.003,
                        "milestones": [2],
                        "lr_rate": 0.3,
                        "scheduler_step_unit": "batch",
                        "num_workers": 0,
                        "save_every_epochs": 1,
                        "log_every_epochs": 1,
                        "gradient_clip_norm": None,
                        "sample_time": 1,
                        "generation_batch_size": 2,
                        "deterministic": False,
                        "torch_num_threads": None,
                    },
                }
            ),
            encoding="utf-8",
        )
        Path(kwargs["log_path"]).write_text("trained\n", encoding="utf-8")

    monkeypatch.setattr(wrapper, "_run_external", fake_external)

    def fake_generate(**kwargs):
        return _fake_generation(Path(kwargs["output_dir"]), kwargs["num_graphs"])

    monkeypatch.setattr(
        "grapher.models.graphrnn.backend.generate_graphrnn_graphs",
        fake_generate,
    )
    training = wrapper.train(
        TrainRequest(
            run=run,
            dataset=dataset,
            options={
                "variant": "GraphRNN_RNN",
                "max_num_node": 5,
                "max_prev_node": 4,
                "hidden_size_rnn": 16,
                "hidden_size_rnn_output": 8,
                "embedding_size_rnn": 8,
                "embedding_size_rnn_output": 4,
                "embedding_size_output": 8,
                "num_layers": 2,
                "batch_size": 2,
                "batch_ratio": 1,
                "epochs": 3,
                "milestones": [2],
                "save_every_epochs": 1,
                "log_every_epochs": 1,
                "generation_batch_size": 2,
                "training_estimates": {"enabled": True, "num_graphs": 2},
                "runtime": {"device": "cpu"},
            },
        )
    )

    assert training.checkpoint_path.is_file()
    worker_manifest = json.loads(
        (training.run_dir / "train" / "worker_manifest.json").read_text()
    )
    assert worker_manifest["checkpoint"] == "checkpoints/graphrnn.pt"
    assert ".staging" not in json.dumps(worker_manifest)
    train_manifest = json.loads(training.manifest_path.read_text())
    assert train_manifest["format"] == TRAINING_MANIFEST_FORMAT
    assert train_manifest["dataset"]["benchmark_id"] == "community_small"
    resolved = training.run_dir / "train" / "resolved_config.yaml"
    assert "scheduler_step_unit: batch" in resolved.read_text()

    estimates_manifest = json.loads(
        training.training_estimates_manifest_path.read_text()
    )
    assert estimates_manifest["format"] == TRAINING_ESTIMATES_MANIFEST_FORMAT
    assert estimates_manifest["pairing"]["status"] == "unpaired"
    with training.estimated_graphs_path.open("rb") as handle:
        assert len(pickle.load(handle)) == 2

    generation = wrapper.generate(
        GenerateRequest(
            run=run,
            checkpoint_path=training.checkpoint_path,
            num_graphs=3,
            generation_seed=9,
        )
    )
    assert generation.num_generated == 3
    manifest = json.loads(generation.manifest_path.read_text())
    assert manifest["format"] == GENERATION_MANIFEST_FORMAT
    assert manifest["returned_count"] == 3
    assert manifest["raw_diagnostics"]["empty_graph_count"] == 0
    with generation.graphs_path.open("rb") as handle:
        graphs = pickle.load(handle)
    assert [graph.graph["graphrnn_sample_index"] for graph in graphs] == [0, 1, 2]


def test_wrapper_rejects_molecular_benchmark_before_external_launch(
    monkeypatch, tmp_path: Path
) -> None:
    root = _fake_root(tmp_path)
    monkeypatch.setenv("GRAPHRNN", str(root))
    monkeypatch.setenv("GRAPHRNN_PYTHON", sys.executable)
    directory = tmp_path / "datasets" / "qm9_attributed"
    directory.mkdir(parents=True)
    for split in ("train", "val", "test"):
        with (directory / f"{split}.pkl").open("wb") as handle:
            pickle.dump([nx.path_graph(3)], handle)
    dataset = DatasetReference(
        "qm9",
        root=tmp_path / "datasets",
        serialized_id="qm9_attributed",
    )
    run = RunSpec.for_seed(
        model_id="graphrnn",
        dataset_id="qm9",
        seed=1,
        output_root=tmp_path / "outputs",
    )
    with pytest.raises(ValueError, match="supports"):
        GraphRNNWrapper().train(TrainRequest(run=run, dataset=dataset))
    assert not run.layout.run_dir.exists()

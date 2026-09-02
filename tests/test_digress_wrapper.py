from __future__ import annotations

import hashlib
import json
import pickle
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import networkx as nx

from grapher.models import DatasetReference, GenerateRequest, RunSpec, TrainRequest
from grapher.models.digress.wrapper import (
    GENERATION_MANIFEST_FORMAT,
    TRAINING_ESTIMATES_MANIFEST_FORMAT,
    TRAINING_MANIFEST_FORMAT,
    DiGressWrapper,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fake_root(tmp_path: Path) -> Path:
    root = tmp_path / "DiGress"
    (root / "src" / "datasets").mkdir(parents=True)
    (root / "configs" / "experiment").mkdir(parents=True)
    (root / "configs" / "dataset").mkdir(parents=True)
    (root / "src" / "diffusion_model_discrete.py").write_text("# fake\n")
    (root / "src" / "datasets" / "spectre_dataset.py").write_text("# fake\n")
    (root / "src" / "datasets" / "qm9_dataset.py").write_text("# fake\n")
    (root / "configs" / "config.yaml").write_text("defaults: []\n")
    (root / "configs" / "experiment" / "comm20.yaml").write_text("{}\n")
    (root / "configs" / "experiment" / "qm9_no_h.yaml").write_text("{}\n")
    (root / "configs" / "dataset" / "comm20.yaml").write_text("{}\n")
    (root / "configs" / "dataset" / "qm9.yaml").write_text("{}\n")
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
        native_id="comm20",
    )


def _prepared_zinc_dataset(tmp_path: Path) -> DatasetReference:
    directory = tmp_path / "datasets" / "zinc"
    directory.mkdir(parents=True)
    graph = nx.Graph()
    graph.add_node(0, atomic_num=6, atom_type=6)
    graph.add_node(1, atomic_num=17, atom_type=17)
    graph.add_edge(0, 1, bond_type=1, bond_order=1.0)
    for split in ("train", "val", "test"):
        with (directory / f"{split}.pkl").open("wb") as handle:
            pickle.dump([graph], handle)
    return DatasetReference(
        "zinc",
        root=tmp_path / "datasets",
        serialized_id="zinc",
        native_id="zinc",
    )


def _fake_generation(output_dir: Path, count: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    export = output_dir / "digress_samples.npz"
    export.write_bytes(b"neutral")
    manifest = output_dir / "digress_manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    log = output_dir / "digress.log"
    log.write_text("generated\n", encoding="utf-8")
    graphs = [nx.path_graph(3 + (index % 2)) for index in range(count)]
    for index, graph in enumerate(graphs):
        graph.graph["digress_sample_index"] = index
    return SimpleNamespace(
        graphs=graphs,
        export_path=export,
        manifest_path=manifest,
        log_path=log,
        export_sha256=_sha(export),
        manifest={"format": "grapher_digress_export_v1", "num_generated": count},
    )


def test_train_and_generate_publish_common_artifacts(monkeypatch, tmp_path: Path) -> None:
    root = _fake_root(tmp_path)
    monkeypatch.setenv("DIGRESS", str(root))
    monkeypatch.setenv("DIGRESS_PYTHON", sys.executable)
    dataset = _prepared_dataset(tmp_path)
    run = RunSpec.for_seed(
        model_id="digress",
        dataset_id="community_small",
        seed=42,
        output_root=tmp_path / "outputs" / "baselines",
    )
    wrapper = DiGressWrapper()

    def fake_external(command, **kwargs):
        label = kwargs["label"]
        if label == "DiGress dataset preparation":
            output_root = Path(command[command.index("--output-root") + 1])
            manifest_path = Path(command[command.index("--manifest") + 1])
            records = {}
            for split in ("train", "val", "test"):
                source = Path(command[command.index(f"--{split}") + 1])
                raw = output_root / "raw" / f"{split}.pt"
                processed = output_root / "processed" / f"{split}.pt"
                raw.parent.mkdir(parents=True, exist_ok=True)
                processed.parent.mkdir(parents=True, exist_ok=True)
                raw.write_bytes(split.encode())
                processed.write_bytes((split + "p").encode())
                with source.open("rb") as handle:
                    count = len(pickle.load(handle))
                records[split] = {
                    "graph_count": count,
                    "source": {"path": str(source), "sha256": _sha(source)},
                    "raw": {"path": str(raw), "sha256": _sha(raw)},
                    "processed": {
                        "path": str(processed),
                        "sha256": _sha(processed),
                    },
                }
            manifest_path.write_text(
                json.dumps(
                    {
                        "format": "grapher_to_digress_generic_dataset_v1",
                        "dataset": "comm20",
                        "split_order_preserved": True,
                        "graphs_dropped": 0,
                        "splits": records,
                    }
                ),
                encoding="utf-8",
            )
        elif label == "DiGress training":
            output = Path(command[command.index("--output-dir") + 1])
            manifest_path = Path(command[command.index("--manifest") + 1])
            checkpoint = output / "checkpoints" / "model.ckpt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(b"checkpoint")
            config = output / "resolved_config.yaml"
            config.write_text(
                "dataset:\n  name: comm20\n  datadir: temporary\n"
                "train:\n  n_epochs: 3\n  batch_size: 2\n"
                "model:\n  diffusion_steps: 5\n",
                encoding="utf-8",
            )
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps(
                    {
                        "format": "grapher_digress_training_worker_v1",
                        "status": "complete",
                        "dataset": "comm20",
                        "experiment": "comm20",
                        "checkpoint": str(checkpoint),
                        "resolved_config": str(config),
                        "molecular_statistics": None,
                        "configured_n_epochs": 3,
                        "completed_epochs": 3,
                        "global_step": 3,
                        "batch_size": 2,
                        "diffusion_steps": 5,
                    }
                ),
                encoding="utf-8",
            )
        else:
            raise AssertionError(label)

    monkeypatch.setattr(wrapper, "_run_external", fake_external)

    def fake_generate(**kwargs):
        return _fake_generation(Path(kwargs["output_dir"]), kwargs["num_graphs"])

    monkeypatch.setattr(
        "grapher.models.digress.backend.generate_digress_graphs", fake_generate
    )
    training = wrapper.train(
        TrainRequest(
            run=run,
            dataset=dataset,
            options={
                "n_epochs": 3,
                "batch_size": 2,
                "training_estimates": {"enabled": True, "num_graphs": 2},
            },
        )
    )

    assert training.checkpoint_path.is_file()
    train_manifest = json.loads(training.manifest_path.read_text())
    assert train_manifest["format"] == TRAINING_MANIFEST_FORMAT
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
    with generation.graphs_path.open("rb") as handle:
        graphs = pickle.load(handle)
    assert [graph.graph["digress_sample_index"] for graph in graphs] == [0, 1, 2]


def test_zinc_training_uses_managed_molecular_profile_and_stock_templates(
    monkeypatch, tmp_path: Path
) -> None:
    root = _fake_root(tmp_path)
    monkeypatch.setenv("DIGRESS", str(root))
    monkeypatch.setenv("DIGRESS_PYTHON", sys.executable)
    dataset = _prepared_zinc_dataset(tmp_path)
    run = RunSpec.for_seed(
        model_id="digress",
        dataset_id="zinc",
        seed=42,
        output_root=tmp_path / "outputs" / "baselines",
    )
    wrapper = DiGressWrapper()

    def fake_external(command, **kwargs):
        label = kwargs["label"]
        if label == "DiGress dataset preparation":
            assert Path(command[1]).name == "prepare_molecular_dataset.py"
            assert command[command.index("--dataset") + 1] == "zinc"
            output_root = Path(command[command.index("--output-root") + 1])
            manifest_path = Path(command[command.index("--manifest") + 1])
            records = {}
            for split in ("train", "val", "test"):
                source = Path(command[command.index(f"--{split}") + 1])
                processed = output_root / "processed" / f"{split}.pt"
                processed.parent.mkdir(parents=True, exist_ok=True)
                processed.write_bytes(split.encode())
                model_view = output_root / "model_view" / f"{split}.pkl"
                model_view.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, model_view)
                records[split] = {
                    "graph_count": 1,
                    "source": {"path": str(source), "sha256": _sha(source)},
                    "processed": {
                        "path": str(processed),
                        "sha256": _sha(processed),
                    },
                }
            manifest_path.write_text(
                json.dumps(
                    {
                        "format": "grapher_to_digress_molecular_dataset_v1",
                        "dataset": "zinc",
                        "split_order_preserved": True,
                        "graphs_dropped": 0,
                        "vocabulary": {
                            "atom_class_to_atomic_number": [
                                6,
                                7,
                                8,
                                9,
                                15,
                                16,
                                17,
                                35,
                                53,
                            ],
                            "present_edge_classes": [1, 2, 3],
                        },
                        "splits": records,
                    }
                ),
                encoding="utf-8",
            )
            return
        if label == "DiGress training":
            assert command[command.index("--dataset") + 1] == "zinc"
            assert command[command.index("--experiment") + 1] == "zinc_no_h"
            output = Path(command[command.index("--output-dir") + 1])
            manifest_path = Path(command[command.index("--manifest") + 1])
            checkpoint = output / "checkpoints" / "model.ckpt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(b"zinc checkpoint")
            config = output / "resolved_config.yaml"
            config.write_text(
                "dataset:\n  name: zinc\n  datadir: temporary\n"
                "train:\n  n_epochs: 1\n  batch_size: 2\n"
                "model:\n  diffusion_steps: 5\n",
                encoding="utf-8",
            )
            statistics = output / "molecular_statistics.json"
            statistics.write_text("{}\n", encoding="utf-8")
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps(
                    {
                        "format": "grapher_digress_training_worker_v1",
                        "status": "complete",
                        "dataset": "zinc",
                        "experiment": "zinc_no_h",
                        "checkpoint": str(checkpoint),
                        "resolved_config": str(config),
                        "molecular_statistics": str(statistics),
                        "configured_n_epochs": 1,
                        "completed_epochs": 1,
                        "global_step": 1,
                        "batch_size": 2,
                        "diffusion_steps": 5,
                    }
                ),
                encoding="utf-8",
            )
            return
        raise AssertionError(label)

    monkeypatch.setattr(wrapper, "_run_external", fake_external)
    artifacts = wrapper.train(
        TrainRequest(
            run=run,
            dataset=dataset,
            options={
                "native_dataset": "zinc",
                "experiment": "zinc_no_h",
                "n_epochs": 1,
                "training_estimates": {"enabled": False},
            },
        )
    )

    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    assert manifest["dataset"]["native_id"] == "zinc"
    assert manifest["dataset"]["domain"] == "molecular"
    assert (
        manifest["dataset"]["model_representation"]
        == "heavy_atom_kekulized_categorical"
    )
    assert manifest["upstream"]["experiment"] == "zinc_no_h"
    assert manifest["upstream"]["config_template"] == {
        "dataset": "qm9",
        "experiment": "qm9_no_h",
    }
    assert artifacts.checkpoint_path.read_bytes() == b"zinc checkpoint"

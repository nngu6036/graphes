from __future__ import annotations

import json
import hashlib
import os
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import pytest

from grapher.models import DatasetReference, GenerateRequest, RunSpec, TrainRequest
from grapher.models.defog import (
    GENERATION_MANIFEST_FORMAT,
    TRAINING_MANIFEST_FORMAT,
    TRAINING_ESTIMATES_MANIFEST_FORMAT,
    DeFoGWrapper,
    _default_experiment,
    _find_checkpoint,
    _native_dataset,
    _prepare_worker_path,
    _verify_managed_generation_assets,
)


def _fake_defog_root(tmp_path: Path) -> Path:
    root = tmp_path / "defog"
    (root / "src").mkdir(parents=True)
    (root / "configs" / "experiment").mkdir(parents=True)
    (root / "configs" / "dataset").mkdir()
    (root / "src" / "main.py").write_text("# fixture\n", encoding="utf-8")
    (root / "configs" / "config.yaml").write_text("defaults: []\n", encoding="utf-8")
    (root / "configs" / "experiment" / "comm20.yaml").write_text("{}\n", encoding="utf-8")
    (root / "configs" / "dataset" / "comm20.yaml").write_text(
        "name: comm20\n", encoding="utf-8"
    )
    for native, experiment in (("qm9", "qm9_no_h"), ("zinc", "zinc")):
        (root / "configs" / "experiment" / f"{experiment}.yaml").write_text(
            "{}\n", encoding="utf-8"
        )
        (root / "configs" / "dataset" / f"{native}.yaml").write_text(
            f"name: {native}\n", encoding="utf-8"
        )
    return root


def _prepared_dataset(
    tmp_path: Path,
    *,
    benchmark_id: str = "community_small",
    serialized_id: str = "sbm",
) -> DatasetReference:
    dataset_dir = tmp_path / "datasets" / serialized_id
    dataset_dir.mkdir(parents=True)
    for split in ("train", "val", "test"):
        # The isolated converter owns deserialization. This mocked test only
        # needs stable source artifacts for the wrapper's hashes.
        (dataset_dir / f"{split}.pkl").write_bytes(f"fixture-{split}".encode())
    return DatasetReference(
        benchmark_id,
        root=tmp_path / "datasets",
        serialized_id=serialized_id,
        native_id="comm20",
    )


def _write_fake_runtime_diagnostics(environment: dict[str, str]) -> None:
    path = Path(environment["GRAPHER_DEFOG_DIAGNOSTICS_PATH"])
    path.write_text(
        json.dumps(
            {
                "format": "grapher_defog_runtime_diagnostics_v1",
                "dataset": environment["GRAPHER_DEFOG_DATASET"],
                "requested_gpus": int(
                    environment["GRAPHER_DEFOG_REQUESTED_GPUS"]
                ),
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_wrapper_declares_implemented_generic_and_attributed_scope() -> None:
    wrapper = DeFoGWrapper()
    assert wrapper.capabilities.status == "ready"
    assert wrapper.capabilities.domains == frozenset({"generic", "attributed"})
    assert wrapper.capabilities.isolation == "subprocess"


@pytest.mark.parametrize(
    ("benchmark_id", "explicit_profile"),
    [
        ("community_small", None),
        # The attached DeFoG revision has no native Ego-small configuration.
        # Its comm20 configuration is the declared compatibility profile for
        # prepared, unlabeled Ego-small topology splits.
        ("ego_small", None),
        ("ego_small", "comm20"),
    ],
)
def test_generic_benchmark_resolves_to_declared_defog_profile(
    benchmark_id: str,
    explicit_profile: str | None,
) -> None:
    assert _native_dataset(benchmark_id, explicit_profile) == "comm20"


@pytest.mark.parametrize(
    ("benchmark_id", "serialized_id", "profile_role"),
    [
        ("community_small", "sbm", "declared_native_alias"),
        ("ego_small", "ego_small", "generic_loader_compatibility_profile"),
    ],
)
def test_training_publishes_common_artifacts(
    monkeypatch,
    tmp_path,
    benchmark_id: str,
    serialized_id: str,
    profile_role: str,
) -> None:
    root = _fake_defog_root(tmp_path)
    monkeypatch.setenv("DEFOG", str(root))
    monkeypatch.setenv("DEFOG_PYTHON", sys.executable)
    run = RunSpec.for_seed(
        model_id="defog",
        dataset_id=benchmark_id,
        seed=42,
        output_root=tmp_path / "outputs" / "baselines",
    )
    request = TrainRequest(
        run=run,
        dataset=_prepared_dataset(
            tmp_path,
            benchmark_id=benchmark_id,
            serialized_id=serialized_id,
        ),
        options={
            "n_epochs": 3,
            "runtime": {"gpus": 0},
            "hydra_overrides": ["model.n_layers=2"],
        },
    )
    wrapper = DeFoGWrapper()
    calls = []

    def fake_external(command, **kwargs):
        calls.append((list(command), kwargs))
        assert kwargs["cwd"] == root / "src"
        assert kwargs["environment"]["PYTHONPATH"].split(os.pathsep) == [
            str(root),
            str(root / "src"),
        ]
        if kwargs["label"] == "DeFoG dataset preparation":
            manifest = Path(command[command.index("--manifest") + 1])
            output_root = Path(command[command.index("--output-root") + 1])
            (output_root / "raw").mkdir(parents=True)
            split_records = {}
            for split, count in (("train", 2), ("val", 1), ("test", 1)):
                native_path = output_root / "raw" / f"{split}.pt"
                native_path.write_bytes(f"native-{split}".encode())
                split_records[split] = {
                    "graph_count": count,
                    "source": {
                        "path": str(request.dataset.split_paths[split]),
                        "sha256": hashlib.sha256(
                            request.dataset.split_paths[split].read_bytes()
                        ).hexdigest(),
                    },
                    "output": {"path": str(native_path), "sha256": "fixture"},
                }
            manifest.write_text(
                json.dumps({"splits": split_records}) + "\n",
                encoding="utf-8",
            )
            return
        _write_fake_runtime_diagnostics(kwargs["environment"])
        run_override = next(value for value in command if value.startswith("hydra.run.dir="))
        native_run = Path(run_override.split("=", 1)[1])
        (native_run / ".hydra").mkdir(parents=True)
        (native_run / ".hydra" / "config.yaml").write_text(
            "dataset:\n  name: comm20\ntrain:\n  n_epochs: 3\n",
            encoding="utf-8",
        )
        checkpoint_dir = native_run / "checkpoints" / "grapher_seed_42"
        checkpoint_dir.mkdir(parents=True)
        (checkpoint_dir / "epoch=2.ckpt").write_bytes(b"defog checkpoint")

    def fake_generate(config, *, num_graphs, seed, output_dir):
        assert num_graphs == 2
        assert seed == 42
        assert config.dataset_datadir is not None
        assert config.dataset_datadir.name == "native_dataset"
        native = Path(output_dir)
        export = native / "defog_samples.npz"
        export.write_bytes(b"training neutral export")
        native_manifest = native / "defog_manifest.json"
        native_manifest.write_text("{}\n", encoding="utf-8")
        log = native / "defog.log"
        log.write_text("generated training estimates\n", encoding="utf-8")
        first, second = nx.path_graph(3), nx.cycle_graph(4)
        first.graph["defog_raw_index"] = 0
        second.graph["defog_raw_index"] = 1
        return SimpleNamespace(
            graphs=[first, second],
            export_path=export,
            manifest_path=native_manifest,
            log_path=log,
            manifest={"format": "defog_generic_topology_v1", "exported_samples": 2},
        )

    monkeypatch.setattr(wrapper, "_run_external", fake_external)
    monkeypatch.setattr("grapher.models.defog.generate_defog_graphs", fake_generate)
    artifacts = wrapper.train(request)

    assert len(calls) == 2
    assert artifacts.checkpoint_path.read_bytes() == b"defog checkpoint"
    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    assert manifest["format"] == TRAINING_MANIFEST_FORMAT
    assert manifest["dataset"]["benchmark_id"] == benchmark_id
    assert manifest["dataset"]["serialized_id"] == serialized_id
    assert manifest["dataset"]["native_id"] == "comm20"
    assert manifest["dataset"]["profile_role"] == profile_role
    assert manifest["run_id"] == "seed_42"
    assert manifest["train_seed"] == 42
    assert manifest["checkpoint"]["selected_epoch"] == 2
    assert manifest["checkpoint"]["final_epoch_verified"] is True
    assert manifest["runtime"]["requested_gpus"] == 0
    assert manifest["runtime"]["single_device_strategy_policy"] == (
        "disable_ddp_use_auto"
    )
    diagnostics_path = artifacts.run_dir / "train" / "runtime_diagnostics.json"
    assert diagnostics_path.is_file()
    assert json.loads(diagnostics_path.read_text(encoding="utf-8"))["format"] == (
        "grapher_defog_runtime_diagnostics_v1"
    )
    assert manifest["commands"]["shell"] is False
    assert "train.n_epochs=3" in manifest["commands"]["train"]
    assert "model.n_layers=2" in manifest["commands"]["train"]
    assert "general.name=grapher_seed_42" in manifest["commands"]["train"]
    assert manifest["training_estimates"]["pairing_status"] == "unpaired"
    assert run.layout.run_manifest_path.is_file()
    assert artifacts.estimated_graphs_path is not None
    assert artifacts.ground_truth_graphs_path is not None
    assert artifacts.training_estimates_manifest_path is not None
    with artifacts.estimated_graphs_path.open("rb") as handle:
        estimates = pickle.load(handle)
    assert [graph.graph["defog_raw_index"] for graph in estimates] == [0, 1]
    assert (
        artifacts.ground_truth_graphs_path.read_bytes()
        == request.dataset.split_paths["train"].read_bytes()
    )
    estimates_manifest = json.loads(
        artifacts.training_estimates_manifest_path.read_text(encoding="utf-8")
    )
    assert estimates_manifest["format"] == TRAINING_ESTIMATES_MANIFEST_FORMAT
    assert estimates_manifest["estimated_graphs"]["count"] == 2
    assert estimates_manifest["ground_truth_graphs"]["count"] == 2
    assert estimates_manifest["pairing"]["status"] == "unpaired"
    assert estimates_manifest["pairing"]["pair_count"] == 0


@pytest.mark.parametrize(
    "override",
    ["general.gpus=2", "+general.gpus=2", "++general.gpus=2"],
)
def test_training_rejects_hydra_override_of_single_gpu_boundary(
    monkeypatch,
    tmp_path,
    override: str,
) -> None:
    root = _fake_defog_root(tmp_path)
    monkeypatch.setenv("DEFOG", str(root))
    monkeypatch.setenv("DEFOG_PYTHON", sys.executable)
    request = TrainRequest(
        run=RunSpec.for_seed(
            model_id="defog",
            dataset_id="community_small",
            seed=42,
            output_root=tmp_path / "outputs" / "baselines",
        ),
        dataset=_prepared_dataset(tmp_path),
        options={
            "runtime": {"gpus": 1},
            "hydra_overrides": [override],
            "training_estimates": {"enabled": False},
        },
    )
    wrapper = DeFoGWrapper()
    launched_labels: list[str] = []

    def fake_external(command, **kwargs):
        launched_labels.append(kwargs["label"])
        if kwargs["label"] != "DeFoG dataset preparation":
            raise AssertionError("A protected GPU override reached DeFoG training.")
        manifest = Path(command[command.index("--manifest") + 1])
        split_records = {}
        for split, source_path in request.dataset.split_paths.items():
            split_records[split] = {
                "graph_count": 1,
                "source": {
                    "path": str(source_path),
                    "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                },
            }
        manifest.write_text(
            json.dumps({"splits": split_records}) + "\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(wrapper, "_run_external", fake_external)

    with pytest.raises(
        (ValueError, RuntimeError),
        match=r"controlled by the wrapper.*general\.gpus",
    ):
        wrapper.train(request)

    assert "DeFoG training" not in launched_labels


def test_generation_serializes_exact_ordered_batch(monkeypatch, tmp_path) -> None:
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    run = RunSpec.for_seed(
        model_id="defog",
        dataset_id="community_small",
        seed=42,
        output_root=tmp_path / "outputs" / "baselines",
    )
    request = GenerateRequest(
        run=run,
        checkpoint_path=checkpoint,
        num_graphs=2,
        generation_seed=9,
    )

    def fake_generate(config, *, num_graphs, seed, output_dir):
        assert config.dataset == "comm20"
        assert (num_graphs, seed) == (2, 9)
        native = Path(output_dir)
        export = native / "defog_samples.npz"
        export.write_bytes(b"neutral export")
        native_manifest = native / "defog_manifest.json"
        native_manifest.write_text("{}\n", encoding="utf-8")
        log = native / "defog.log"
        log.write_text("generated\n", encoding="utf-8")
        first, second = nx.path_graph(3), nx.cycle_graph(4)
        first.graph["defog_raw_index"] = 0
        second.graph["defog_raw_index"] = 1
        return SimpleNamespace(
            graphs=[first, second],
            export_path=export,
            manifest_path=native_manifest,
            log_path=log,
            manifest={"format": "defog_generic_topology_v1", "exported_samples": 2},
        )

    monkeypatch.setattr("grapher.models.defog.generate_defog_graphs", fake_generate)
    artifacts = DeFoGWrapper().generate(request)

    with artifacts.graphs_path.open("rb") as handle:
        graphs = pickle.load(handle)
    assert [graph.graph["defog_raw_index"] for graph in graphs] == [0, 1]
    assert artifacts.num_requested == artifacts.num_generated == 2
    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    assert manifest["format"] == GENERATION_MANIFEST_FORMAT
    assert manifest["run_id"] == "seed_42"
    assert manifest["requested_count"] == manifest["returned_count"] == 2
    assert manifest["base_graphs"]["sha256"] == artifacts.graphs_sha256
    assert (artifacts.generation_dir / "native" / "defog_samples.npz").is_file()


def test_same_seed_distinct_run_ids_publish_independent_batches(
    monkeypatch,
    tmp_path,
) -> None:
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"shared fixture checkpoint")
    output_root = tmp_path / "outputs" / "baselines"

    def fake_generate(config, *, num_graphs, seed, output_dir):
        assert config.dataset == "comm20"
        assert (num_graphs, seed) == (1, 7)
        native = Path(output_dir)
        export = native / "defog_samples.npz"
        export.write_bytes(b"neutral export")
        native_manifest = native / "defog_manifest.json"
        native_manifest.write_text("{}\n", encoding="utf-8")
        log = native / "defog.log"
        log.write_text("generated\n", encoding="utf-8")
        graph = nx.path_graph(3)
        graph.graph["defog_raw_index"] = 0
        return SimpleNamespace(
            graphs=[graph],
            export_path=export,
            manifest_path=native_manifest,
            log_path=log,
            manifest={"format": "defog_generic_topology_v1", "exported_samples": 1},
        )

    monkeypatch.setattr("grapher.models.defog.generate_defog_graphs", fake_generate)
    artifacts = []
    for run_id in ("replicate_a", "replicate_b"):
        run = RunSpec.for_seed(
            model_id="defog",
            dataset_id="community_small",
            seed=42,
            run_id=run_id,
            output_root=output_root,
        )
        artifacts.append(
            DeFoGWrapper().generate(
                GenerateRequest(
                    run=run,
                    checkpoint_path=checkpoint,
                    num_graphs=1,
                    generation_seed=7,
                    generation_id="evaluation_batch",
                )
            )
        )

    assert artifacts[0].generation_dir != artifacts[1].generation_dir
    assert all(item.graphs_path.is_file() for item in artifacts)
    assert [
        json.loads(item.manifest_path.read_text(encoding="utf-8"))["run_id"]
        for item in artifacts
    ] == ["replicate_a", "replicate_b"]
    assert artifacts[0].generation_dir == (
        output_root
        / "defog"
        / "community_small"
        / "replicate_a"
        / "generations"
        / "evaluation_batch"
    )
    assert artifacts[1].generation_dir == (
        output_root
        / "defog"
        / "community_small"
        / "replicate_b"
        / "generations"
        / "evaluation_batch"
    )


def test_generation_recovers_training_configuration(monkeypatch, tmp_path) -> None:
    run = RunSpec.for_seed(
        model_id="defog",
        dataset_id="custom_generic",
        seed=5,
        output_root=tmp_path / "outputs" / "baselines",
    )
    checkpoint = run.layout.checkpoints_dir / "model.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    run.layout.native_training_dataset_dir.mkdir(parents=True)
    run.layout.resolved_training_config_path.write_text(
        "dataset:\n  name: comm20\nmodel:\n  n_layers: 3\n",
        encoding="utf-8",
    )
    run.layout.training_manifest_path.write_text(
        json.dumps(
            {
                "dataset": {"native_id": "comm20"},
                "upstream": {"experiment": "comm20"},
            }
        ),
        encoding="utf-8",
    )

    def fake_generate(config, *, num_graphs, seed, output_dir):
        assert config.dataset == "comm20"
        assert config.experiment == "comm20"
        assert config.resolved_config_path == run.layout.resolved_training_config_path
        assert config.dataset_datadir == run.layout.native_training_dataset_dir
        native = Path(output_dir)
        export = native / "defog_samples.npz"
        export.write_bytes(b"neutral export")
        native_manifest = native / "defog_manifest.json"
        native_manifest.write_text("{}\n", encoding="utf-8")
        log = native / "defog.log"
        log.write_text("generated\n", encoding="utf-8")
        graph = nx.path_graph(3)
        graph.graph["defog_raw_index"] = 0
        return SimpleNamespace(
            graphs=[graph],
            export_path=export,
            manifest_path=native_manifest,
            log_path=log,
            manifest={"format": "defog_generic_topology_v1", "exported_samples": 1},
        )

    monkeypatch.setattr("grapher.models.defog.generate_defog_graphs", fake_generate)
    artifacts = DeFoGWrapper().generate(
        GenerateRequest(
            run=run,
            checkpoint_path=checkpoint,
            num_graphs=1,
            generation_seed=11,
        )
    )
    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    assert manifest["dataset"]["native_id"] == "comm20"
    assert manifest["resolved_training_config"]["sha256"]


def test_managed_generation_rejects_mutated_native_dataset(tmp_path: Path) -> None:
    run = RunSpec.for_seed(
        model_id="defog",
        dataset_id="community_small",
        seed=3,
        output_root=tmp_path / "outputs" / "baselines",
    )
    native_file = run.layout.native_training_dataset_dir / "raw" / "train.pt"
    native_file.parent.mkdir(parents=True)
    native_file.write_bytes(b"original native split")
    run.layout.resolved_training_config_path.parent.mkdir(parents=True, exist_ok=True)
    run.layout.resolved_training_config_path.write_text(
        "dataset:\n  name: comm20\n", encoding="utf-8"
    )
    (run.layout.train_dir / "dataset_conversion.json").write_text(
        json.dumps(
            {
                "splits": {
                    "train": {
                        "output": {
                            "path": "native_dataset/raw/train.pt",
                            "sha256": hashlib.sha256(
                                native_file.read_bytes()
                            ).hexdigest(),
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    training_manifest = {
        "resolved_config": {
            "sha256": hashlib.sha256(
                run.layout.resolved_training_config_path.read_bytes()
            ).hexdigest()
        }
    }

    _verify_managed_generation_assets(run.layout, training_manifest)
    native_file.write_bytes(b"mutated native split")
    with pytest.raises(RuntimeError, match="changed after training"):
        _verify_managed_generation_assets(run.layout, training_manifest)


def test_explicit_post_fit_checkpoint_precedes_periodic_checkpoint(
    tmp_path: Path,
) -> None:
    periodic = tmp_path / "checkpoints" / "epoch=999.ckpt"
    explicit_final = tmp_path / "checkpoints" / "grapher_final.ckpt"
    periodic.parent.mkdir(parents=True)
    periodic.write_bytes(b"periodic")
    explicit_final.write_bytes(b"post-fit final")

    assert _find_checkpoint(tmp_path) == explicit_final


@pytest.mark.parametrize(
    ("benchmark_id", "serialized_id", "experiment", "representation", "run_id"),
    [
        (
            "qm9",
            "qm9_attributed",
            "qm9_no_h",
            "heavy_atom_categorical",
            "qm9_replicate_a",
        ),
        (
            "zinc",
            "zinc_attributed",
            "zinc",
            "kekule_no_aromatic_class",
            "zinc_replicate_b",
        ),
    ],
)
def test_molecular_train_and_generate_dispatch_and_manifest_semantics(
    monkeypatch,
    tmp_path: Path,
    benchmark_id: str,
    serialized_id: str,
    experiment: str,
    representation: str,
    run_id: str,
) -> None:
    root = _fake_defog_root(tmp_path)
    monkeypatch.setenv("DEFOG", str(root))
    monkeypatch.setenv("DEFOG_PYTHON", sys.executable)

    dataset_dir = tmp_path / "datasets" / serialized_id
    dataset_dir.mkdir(parents=True)
    for split in ("train", "val", "test"):
        (dataset_dir / f"{split}.pkl").write_bytes(
            f"{benchmark_id}-{split}".encode()
        )
    dataset = DatasetReference(
        benchmark_id,
        root=tmp_path / "datasets",
        serialized_id=serialized_id,
    )
    run = RunSpec.for_seed(
        model_id="defog",
        dataset_id=benchmark_id,
        seed=17,
        run_id=run_id,
        output_root=tmp_path / "outputs" / "baselines",
    )
    request = TrainRequest(
        run=run,
        dataset=dataset,
        options={
            "n_epochs": 2,
            "runtime": {"gpus": 0},
            # Keep this test focused on train/generate dispatch; the independent
            # training-estimate contract is covered by the generic wrapper test.
            "training_estimates": {"enabled": False},
        },
    )
    wrapper = DeFoGWrapper()
    calls: list[tuple[list[str], dict]] = []
    molecular_statistics = {
        "format": "grapher_defog_molecular_statistics_v1",
        "dataset": benchmark_id,
        "distribution_sha256": f"{benchmark_id}-statistics-digest",
        "distributions": {},
    }

    def fake_external(command, **kwargs):
        command = list(command)
        calls.append((command, kwargs))
        if kwargs["label"] == "DeFoG dataset preparation":
            assert Path(command[1]).name == "defog_prepare_molecular_dataset_worker.py"
            assert command[command.index("--dataset") + 1] == benchmark_id
            manifest_path = Path(command[command.index("--manifest") + 1])
            native_root = Path(command[command.index("--output-root") + 1])
            (native_root / "raw").mkdir(parents=True)
            split_records = {}
            for split, count in (("train", 2), ("val", 1), ("test", 1)):
                output_path = native_root / "raw" / f"{split}.pt"
                output_path.write_bytes(f"native-{benchmark_id}-{split}".encode())
                source_path = dataset.split_paths[split]
                split_records[split] = {
                    "graph_count": count,
                    "source": {
                        "path": str(source_path),
                        "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                    },
                    "output": {
                        "path": str(output_path),
                        "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
                    },
                }
            manifest_path.write_text(
                json.dumps(
                    {
                        "format": "grapher_to_defog_molecular_dataset_v1",
                        "dataset": benchmark_id,
                        "representation": representation,
                        "splits": split_records,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            return

        _write_fake_runtime_diagnostics(kwargs["environment"])
        assert f"+experiment={experiment}" in command
        assert f"dataset={benchmark_id}" in command
        assert f"general.name=grapher_{run_id}" in command
        statistics_path = Path(
            kwargs["environment"]["GRAPHER_DEFOG_STATISTICS_PATH"]
        )
        statistics_path.write_text(
            json.dumps(molecular_statistics) + "\n",
            encoding="utf-8",
        )
        native_run = Path(
            next(value for value in command if value.startswith("hydra.run.dir="))
            .split("=", 1)[1]
        )
        (native_run / ".hydra").mkdir(parents=True)
        (native_run / ".hydra" / "config.yaml").write_text(
            (
                f"dataset:\n  name: {benchmark_id}\n"
                f"train:\n  n_epochs: 2\n"
            ),
            encoding="utf-8",
        )
        checkpoint_dir = native_run / "checkpoints" / f"grapher_{run_id}"
        checkpoint_dir.mkdir(parents=True)
        (checkpoint_dir / "epoch=1.ckpt").write_bytes(
            f"{benchmark_id}-checkpoint".encode()
        )

    monkeypatch.setattr(wrapper, "_run_external", fake_external)
    training = wrapper.train(request)

    assert len(calls) == 2
    train_manifest = json.loads(training.manifest_path.read_text(encoding="utf-8"))
    assert train_manifest["dataset"]["benchmark_id"] == benchmark_id
    assert train_manifest["dataset"]["serialized_id"] == serialized_id
    assert train_manifest["dataset"]["native_id"] == benchmark_id
    assert train_manifest["dataset"]["domain"] == "molecular"
    assert train_manifest["dataset"]["profile_role"] == "native"
    assert train_manifest["dataset"]["model_representation"] == representation
    assert train_manifest["upstream"]["experiment"] == experiment
    assert train_manifest["run_id"] == run_id
    assert train_manifest["training_estimates"] == {"enabled": False}
    assert Path(train_manifest["commands"]["prepare"][1]).name == (
        "defog_prepare_molecular_dataset_worker.py"
    )
    assert training.run_dir == (
        tmp_path
        / "outputs"
        / "baselines"
        / "defog"
        / benchmark_id
        / run_id
    )
    assert training.estimated_graphs_path is None
    assert training.ground_truth_graphs_path is None

    def fake_generate(config, *, num_graphs, seed, output_dir):
        assert config.dataset == benchmark_id
        assert config.experiment == experiment
        assert config.dataset_datadir == run.layout.native_training_dataset_dir
        assert config.resolved_config_path == run.layout.resolved_training_config_path
        assert (num_graphs, seed) == (1, 23)
        native = Path(output_dir)
        export = native / "defog_samples.npz"
        export.write_bytes(f"{benchmark_id}-neutral-export".encode())
        native_manifest = native / "defog_manifest.json"
        native_manifest.write_text("{}\n", encoding="utf-8")
        log = native / "defog.log"
        log.write_text("generated molecular graph\n", encoding="utf-8")
        graph = nx.Graph()
        graph.add_node(0, atomic_num=6, atom_type=6)
        graph.add_node(1, atomic_num=7, atom_type=7)
        graph.add_edge(0, 1, bond_type=1, bond_order=1.0)
        graph.graph.update(base_generator="defog", defog_raw_index=0)
        return SimpleNamespace(
            graphs=[graph],
            export_path=export,
            manifest_path=native_manifest,
            log_path=log,
            manifest={
                "format": "defog_graph_batch_v2",
                "dataset": benchmark_id,
                "exported_samples": 1,
                "schema": {"domain": "molecular"},
                "runtime": {"molecular_statistics": molecular_statistics},
            },
        )

    monkeypatch.setattr("grapher.models.defog.generate_defog_graphs", fake_generate)
    generation = wrapper.generate(
        GenerateRequest(
            run=run,
            checkpoint_path=training.checkpoint_path,
            num_graphs=1,
            generation_seed=23,
            generation_id="evaluation_batch",
        )
    )

    generation_manifest = json.loads(
        generation.manifest_path.read_text(encoding="utf-8")
    )
    assert generation_manifest["dataset"]["benchmark_id"] == benchmark_id
    assert generation_manifest["dataset"]["native_id"] == benchmark_id
    assert generation_manifest["dataset"]["domain"] == "molecular"
    assert generation_manifest["dataset"]["model_representation"] == representation
    assert generation_manifest["run_id"] == run_id
    assert generation_manifest["generation_id"] == "evaluation_batch"
    assert generation_manifest["requested_count"] == 1
    assert generation_manifest["returned_count"] == 1
    with generation.graphs_path.open("rb") as handle:
        generated_graphs = pickle.load(handle)
    assert generated_graphs[0].nodes[0]["atomic_num"] == 6
    assert generated_graphs[0].edges[0, 1]["bond_type"] == 1


@pytest.mark.parametrize(
    ("native_dataset", "experiment"),
    [
        ("qm9", "qm9_no_h"),
        ("qm9_attributed", "qm9_no_h"),
        ("zinc", "zinc"),
        ("zinc_attributed", "zinc"),
    ],
)
def test_molecular_defaults_and_preparation_worker(
    native_dataset: str,
    experiment: str,
) -> None:
    resolved = _native_dataset(native_dataset)
    assert resolved in {"qm9", "zinc"}
    assert _default_experiment(resolved) == experiment
    assert _prepare_worker_path(resolved).name == (
        "defog_prepare_molecular_dataset_worker.py"
    )

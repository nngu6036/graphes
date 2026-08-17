from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

from grapher.models.defog_backend import (
    DEFOG_EXPORT_FORMAT,
    DeFoGGeneratorConfig,
    _worker_environment,
    build_defog_worker_command,
    generate_defog_graphs,
    load_defog_export,
    resolve_defog_root,
)
from scripts import run_defog_grapher
from scripts.defog_export_worker import encode_generic_samples
from scripts.evaluate_graph_generation_report import resolve_base_graph_path


def _sample(n: int, edges: list[tuple[int, int]]) -> list[np.ndarray]:
    node_labels = np.zeros(n, dtype=np.int64)
    edge_labels = np.zeros((n, n), dtype=np.int64)
    for source, target in edges:
        edge_labels[source, target] = 1
        edge_labels[target, source] = 1
    return [node_labels, edge_labels]


def _write_export(path: Path, samples: list[list[np.ndarray]]) -> None:
    arrays = encode_generic_samples(samples)
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)


def _fake_defog_root(tmp_path: Path) -> Path:
    root = tmp_path / "defog"
    (root / "src").mkdir(parents=True)
    (root / "configs").mkdir()
    (root / "src" / "main.py").write_text("# fixture\n", encoding="utf-8")
    (root / "configs" / "config.yaml").write_text("defaults: []\n", encoding="utf-8")
    return root


def test_resolve_defog_root_requires_valid_environment(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("DEFOG", raising=False)
    with pytest.raises(EnvironmentError, match="DEFOG"):
        resolve_defog_root()

    bad_root = tmp_path / "not-defog"
    bad_root.mkdir()
    monkeypatch.setenv("DEFOG", str(bad_root))
    with pytest.raises(FileNotFoundError, match="not a DeFoG source root"):
        resolve_defog_root()

    root = _fake_defog_root(tmp_path)
    monkeypatch.setenv("DEFOG", str(root))
    assert resolve_defog_root() == root.resolve()


def test_worker_encoding_and_parent_loader_preserve_isolates(tmp_path) -> None:
    export = tmp_path / "samples.npz"
    _write_export(
        export,
        [
            _sample(4, [(0, 1), (1, 2)]),
            _sample(3, [(0, 2)]),
        ],
    )

    graphs = load_defog_export(export, expected_count=2)

    assert [graph.number_of_nodes() for graph in graphs] == [4, 3]
    assert [set(graph.edges()) for graph in graphs] == [
        {(0, 1), (1, 2)},
        {(0, 2)},
    ]
    assert 3 in graphs[0]
    assert all(graph.graph["base_generator"] == "defog" for graph in graphs)


@pytest.mark.parametrize(
    "sample, message",
    [
        ([np.zeros(3), np.zeros((2, 2))], "expected"),
        (
            [np.zeros(2), np.asarray([[0, 1], [0, 0]])],
            "not symmetric",
        ),
        (
            [np.zeros(2), np.asarray([[1, 0], [0, 0]])],
            "self-loop",
        ),
        (
            [np.zeros(2), np.asarray([[0, 2], [2, 0]])],
            "outside 0/1",
        ),
        (
            [np.asarray([0, 1]), np.zeros((2, 2))],
            "multiple node classes",
        ),
    ],
)
def test_worker_rejects_malformed_generic_samples(sample, message) -> None:
    with pytest.raises(ValueError, match=message):
        encode_generic_samples([sample])


def test_loader_rejects_wrong_requested_count(tmp_path) -> None:
    export = tmp_path / "samples.npz"
    _write_export(export, [_sample(3, [(0, 1)])])
    with pytest.raises(ValueError, match="expected 2"):
        load_defog_export(export, expected_count=2)


def test_worker_command_and_environment_are_shell_free(tmp_path) -> None:
    root = _fake_defog_root(tmp_path)
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    resolved_config = tmp_path / "resolved_config.yaml"
    resolved_config.write_text("dataset:\n  name: comm20\n", encoding="utf-8")
    dataset_datadir = tmp_path / "native_dataset"
    dataset_datadir.mkdir()
    config = DeFoGGeneratorConfig.from_dict(
        {
            "type": "defog",
            "dataset": "comm20",
            "experiment": "comm20",
            "checkpoint_path": str(checkpoint),
            "dataset_datadir": str(dataset_datadir),
            "resolved_config_path": str(resolved_config),
        },
        python_executable=sys.executable,
    )
    command = build_defog_worker_command(
        config,
        defog_root=root,
        python_executable=sys.executable,
        export_path=tmp_path / "out.npz",
        manifest_path=tmp_path / "manifest.json",
        num_graphs=7,
        seed=42,
    )
    environment = _worker_environment(config, defog_root=root, seed=42)

    assert isinstance(command, list)
    assert command[0] == sys.executable
    assert "--checkpoint" in command
    assert command[command.index("--resolved-config") + 1] == str(
        resolved_config.resolve()
    )
    assert command[command.index("--dataset-datadir") + 1] == str(
        dataset_datadir.resolve()
    )
    assert command[command.index("--num-samples") + 1] == "7"
    # Omitted sampling options must inherit the saved DeFoG configuration.
    # Passing adapter-wide defaults here would silently replace QM9/ZINC's
    # experiment-specific schedule.
    assert "--sample-steps" not in command
    assert "--time-distortion" not in command
    assert "--eta" not in command
    assert "--omega" not in command
    assert environment["PYTHONPATH"].split(os.pathsep) == [
        str(root),
        str(root / "src"),
    ]
    assert environment["WANDB_MODE"] == "disabled"


def test_neutral_export_reuse_skips_defog_process(monkeypatch, tmp_path) -> None:
    export = tmp_path / "defog_samples.npz"
    _write_export(export, [_sample(3, [(0, 1), (1, 2)])])
    config = DeFoGGeneratorConfig.from_dict(
        {
            "type": "defog",
            "dataset": "comm20",
            "experiment": "comm20",
            "generated_path": str(export),
        }
    )

    def unexpected_run(*args, **kwargs):
        raise AssertionError("A neutral export must not launch DeFoG.")

    monkeypatch.setattr(subprocess, "run", unexpected_run)
    result = generate_defog_graphs(
        config,
        num_graphs=1,
        seed=42,
        output_dir=tmp_path / "output",
    )

    assert len(result.graphs) == 1
    assert result.export_path == export.resolve()
    assert result.log_path is None


def test_neutral_export_reuse_verifies_manifest_checksum(tmp_path) -> None:
    export = tmp_path / "defog_samples.npz"
    manifest = tmp_path / "defog_manifest.json"
    _write_export(export, [_sample(3, [(0, 1)])])
    manifest.write_text(
        json.dumps(
            {
                "format": DEFOG_EXPORT_FORMAT,
                "exported_samples": 1,
                "export": {"sha256": "0" * 64},
            }
        ),
        encoding="utf-8",
    )
    config = DeFoGGeneratorConfig.from_dict(
        {
            "type": "defog",
            "dataset": "comm20",
            "experiment": "comm20",
            "generated_path": str(export),
            "manifest_path": str(manifest),
        }
    )

    with pytest.raises(ValueError, match="checksum"):
        generate_defog_graphs(
            config,
            num_graphs=1,
            seed=42,
            output_dir=tmp_path / "output",
        )


def test_subprocess_generation_publishes_validated_export(
    monkeypatch,
    tmp_path,
) -> None:
    root = _fake_defog_root(tmp_path)
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setenv("DEFOG", str(root))
    config = DeFoGGeneratorConfig.from_dict(
        {
            "type": "defog",
            "dataset": "comm20",
            "experiment": "comm20",
            "checkpoint_path": str(checkpoint),
            "runtime": {"python_executable": sys.executable},
        }
    )

    def fake_run(command, **kwargs):
        assert kwargs["shell"] is False
        assert kwargs["cwd"] == str(root / "src")
        export_path = Path(command[command.index("--output") + 1])
        manifest_path = Path(command[command.index("--manifest") + 1])
        _write_export(export_path, [_sample(4, [(0, 1), (2, 3)])])
        manifest_path.write_text(
            json.dumps(
                {
                    "format": DEFOG_EXPORT_FORMAT,
                    "exported_samples": 1,
                }
            ),
            encoding="utf-8",
        )
        kwargs["stdout"].write("fake worker completed\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = generate_defog_graphs(
        config,
        num_graphs=1,
        seed=42,
        output_dir=tmp_path / "output",
    )

    assert len(result.graphs) == 1
    assert result.export_path.name == "defog_samples.npz"
    assert result.log_path is not None and result.log_path.is_file()
    assert result.manifest["format"] == DEFOG_EXPORT_FORMAT


def test_trusted_pickle_reuse_does_not_require_configured_checkpoint(
    monkeypatch,
    tmp_path,
) -> None:
    root = _fake_defog_root(tmp_path)
    raw_pickle = tmp_path / "generated_samples_rank0.pkl"
    with raw_pickle.open("wb") as handle:
        pickle.dump([_sample(3, [(0, 1)])], handle)
    monkeypatch.setenv("DEFOG", str(root))
    config = DeFoGGeneratorConfig.from_dict(
        {
            "type": "defog",
            "dataset": "comm20",
            "experiment": "comm20",
            "checkpoint_path": str(tmp_path / "missing.ckpt"),
            "generated_path": str(raw_pickle),
            "runtime": {"python_executable": sys.executable},
        }
    )

    def fake_run(command, **kwargs):
        assert "--input-pickle" in command
        assert "--checkpoint" not in command
        export_path = Path(command[command.index("--output") + 1])
        manifest_path = Path(command[command.index("--manifest") + 1])
        _write_export(export_path, [_sample(3, [(0, 1)])])
        manifest_path.write_text(
            json.dumps(
                {
                    "format": DEFOG_EXPORT_FORMAT,
                    "exported_samples": 1,
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = generate_defog_graphs(
        config,
        num_graphs=1,
        seed=42,
        output_dir=tmp_path / "output",
    )

    assert len(result.graphs) == 1


def test_correction_keeps_disconnected_base_as_reported_noop(monkeypatch) -> None:
    connected = nx.cycle_graph(4)
    disconnected = nx.disjoint_union(nx.path_graph(2), nx.path_graph(2))
    calls: list[nx.Graph] = []

    def fake_refiner(graph, **kwargs):
        calls.append(graph)
        return graph.copy(), [
            {
                "step": 0,
                "accepted": False,
                "reason": "explicit_stop_no_candidates",
                "num_proposals": 0,
                "num_valid_candidates": 0,
                "candidate_rejection_reasons": {},
            }
        ]

    monkeypatch.setattr(
        run_defog_grapher,
        "refine_graph_with_topology_predictions",
        fake_refiner,
    )
    refined, traces, records, _ = run_defog_grapher.correct_defog_base_graphs(
        [connected, disconnected],
        model=None,
        graphlet_basis=None,
        summary_config=None,
        refiner_config={},
        device="cpu",
        rng=np.random.default_rng(0),
        predictor_graphlet_error=0.1,
        disconnected_policy="no_op_and_report",
        show_progress=False,
    )

    assert len(calls) == 1
    assert set(refined[1].nodes()) == set(disconnected.nodes())
    assert set(refined[1].edges()) == set(disconnected.edges())
    assert traces[1][0]["reason"] == "source_disconnected_noop"
    assert records[1]["correction_attempted"] == 0.0
    assert records[1]["rejection_reasons"] == {"source_disconnected": 1}
    assert [dict(graph.degree()) for graph in refined] == [
        dict(connected.degree()),
        dict(disconnected.degree()),
    ]


def test_report_discovers_defog_base_before_legacy_hh_source(tmp_path) -> None:
    defog = tmp_path / "defog_base_graphs.pkl"
    coarse = tmp_path / "coarse_graphs.pkl"
    defog.touch()
    coarse.touch()

    path, stage = resolve_base_graph_path(tmp_path)

    assert path == defog
    assert stage == "defog_base"

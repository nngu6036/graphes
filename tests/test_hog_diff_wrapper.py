from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from grapher.models.hog_diff.codec import profile_for
from grapher.models.hog_diff.wrapper import (
    HOGDiffWrapper,
    _environment,
    _resolved_hog_config,
)


class HOGDiffWrapperConfigTests(unittest.TestCase):
    def _source_config(self, root: Path, *, attributed: bool = False) -> Path:
        config = {
            "data": {
                "name": "placeholder",
                "max_node": 9 if attributed else 20,
                "atom_channels": 4 if attributed else 21,
                "num_workers": 7,
                "test_split": 0.2,
            },
            "training": {
                "n_iters": 10,
                "batch_size": 8,
                "seed": 1,
                "snapshot_sampling": True,
                "log_freq": 50,
            },
            "OUtraining": {
                "n_iters": 20,
                "batch_size": 16,
                "seed": 2,
                "snapshot_sampling": True,
                "log_freq": 60,
            },
            "optim": {
                "optimizer": "Adam",
                "lr": "2e-4",
                "beta1": 0.9,
                "eps": "1e-8",
                "weight_decay": 0,
                "warmup": 0,
                "grad_clip": 1.0,
            },
            "OUoptim": {
                "optimizer": "Adam",
                "lr": "1e-4",
                "beta1": 0.9,
                "eps": "1e-8",
                "weight_decay": 0,
                "warmup": 0,
                "grad_clip": 1.0,
            },
            "eval": {"seed": 3, "batch_size": 4, "save_graph": True},
            "exp": {"plot": True},
        }
        path = root / "source.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        return path

    def test_capabilities_are_ready_without_importing_upstream(self) -> None:
        wrapper = HOGDiffWrapper()
        self.assertEqual(wrapper.capabilities.status, "ready")
        self.assertEqual(wrapper.capabilities.isolation, "subprocess")
        self.assertEqual(wrapper.capabilities.domains, frozenset({"generic", "attributed"}))

    def test_generic_config_uses_exact_projected_test_split_and_disables_snapshot_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self._source_config(root)
            profile = profile_for("community_small")
            resolved = _resolved_hog_config(
                source_path=source,
                options={
                    "num_workers": 0,
                    "higher_order": {"n_iters": 123},
                    "ou": {"batch_size": 5},
                },
                profile=profile,
                dataset_manifest={
                    "upstream_training_projection": {"test_split": 0.2380952381}
                },
                run_id="seed_42",
                seed=42,
                progress={"iteration_interval": 17},
            )
            self.assertEqual(resolved["data"]["name"], "community_small")
            self.assertAlmostEqual(resolved["data"]["test_split"], 0.2380952381)
            self.assertEqual(resolved["data"]["num_workers"], 0)
            self.assertEqual(resolved["training"]["n_iters"], 123)
            self.assertEqual(resolved["OUtraining"]["batch_size"], 5)
            self.assertEqual(resolved["training"]["seed"], 42)
            self.assertEqual(resolved["OUtraining"]["seed"], 42)
            self.assertFalse(resolved["training"]["snapshot_sampling"])
            self.assertFalse(resolved["OUtraining"]["snapshot_sampling"])
            self.assertEqual(resolved["training"]["log_freq"], 17)
            self.assertEqual(resolved["OUtraining"]["log_freq"], 17)
            self.assertIsInstance(resolved["optim"]["eps"], float)
            self.assertEqual(resolved["optim"]["eps"], 1.0e-8)
            self.assertIsInstance(resolved["optim"]["lr"], float)
            self.assertEqual(resolved["optim"]["lr"], 2.0e-4)
            self.assertIsInstance(resolved["OUoptim"]["eps"], float)
            self.assertEqual(resolved["OUoptim"]["eps"], 1.0e-8)
            self.assertFalse(resolved["eval"]["save_graph"])
            self.assertFalse(resolved["exp"]["plot"])

    def test_attributed_config_preserves_native_shape_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self._source_config(root, attributed=True)
            profile = profile_for("qm9")
            resolved = _resolved_hog_config(
                source_path=source,
                options={},
                profile=profile,
                dataset_manifest={},
                run_id="seed_7",
                seed=7,
                progress={"iteration_interval": None},
            )
            self.assertEqual(resolved["data"]["name"], "qm9")
            self.assertEqual(resolved["data"]["max_node"], 9)
            self.assertEqual(resolved["data"]["atom_channels"], 4)
            self.assertFalse(resolved["training"]["snapshot_sampling"])
            self.assertFalse(resolved["OUtraining"]["snapshot_sampling"])

    def test_cpu_environment_isolates_cuda_and_hogdiff_artifact_roots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "hog"
            data = Path(tmp) / "data"
            runtime = Path(tmp) / "runtime"
            root.mkdir()
            data.mkdir()
            runtime.mkdir()
            env, require_cuda = _environment(
                root,
                data_root=data,
                runtime_root=runtime,
                seed=9,
                device="cpu",
                cuda_visible_devices="3",
            )
            self.assertFalse(require_cuda)
            self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "")
            self.assertEqual(env["DATA_ROOT"], str(data.resolve()))
            self.assertEqual(env["CKPT_ROOT"], str((runtime / "checkpoint_runtime").resolve()))
            self.assertEqual(env["WANDB_MODE"], "disabled")
            self.assertEqual(env["PYTHONHASHSEED"], "9")


if __name__ == "__main__":
    unittest.main()


def test_wrapper_train_and_generate_transaction_with_isolated_worker_contract(
    tmp_path: Path, monkeypatch
) -> None:
    """Exercise GraphER artifact publication without importing HOG-Diff itself."""
    import hashlib
    import json
    import pickle
    import sys

    import networkx as nx
    import numpy as np

    import grapher.models.hog_diff.wrapper as wrapper_module
    from grapher.models import DatasetReference, GenerateRequest, RunSpec, TrainRequest

    source_root = tmp_path / "hogdiff"
    (source_root / "configs").mkdir(parents=True)
    source_config = {
        "data": {
            "name": "community_small",
            "max_node": 20,
            "atom_channels": 21,
            "num_workers": 0,
            "test_split": 0.2,
        },
        "training": {"n_iters": 2, "batch_size": 2, "snapshot_sampling": False},
        "OUtraining": {"n_iters": 3, "batch_size": 2, "snapshot_sampling": False},
        "eval": {"batch_size": 2, "num_samples": 2, "seed": 0},
        "exp": {"plot": False},
    }
    (source_root / "configs" / "cs.yaml").write_text(
        yaml.safe_dump(source_config), encoding="utf-8"
    )

    dataset_root = tmp_path / "datasets"
    dataset_dir = dataset_root / "sbm"
    dataset_dir.mkdir(parents=True)
    splits = {
        "train": [nx.path_graph(4), nx.cycle_graph(5)],
        "val": [nx.path_graph(5)],
        "test": [nx.complete_graph(3)],
    }
    for split, graphs in splits.items():
        with (dataset_dir / f"{split}.pkl").open("wb") as handle:
            pickle.dump(graphs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    python_path = Path(sys.executable).resolve()
    monkeypatch.setattr(wrapper_module, "resolve_hogdiff_root", lambda *_args, **_kwargs: source_root)
    monkeypatch.setattr(
        wrapper_module,
        "resolve_hogdiff_python",
        lambda **_kwargs: python_path,
    )
    monkeypatch.setattr(
        wrapper_module,
        "_source_identity",
        lambda _root: {
            "source_root": str(source_root),
            "revision": None,
            "files": {},
            "source_fingerprint": "fake-source",
            "integration_mode": "test",
            "compatibility_shim": "test",
        },
    )
    monkeypatch.setattr(
        wrapper_module,
        "_python_identity",
        lambda _python: {"python_executable": str(python_path), "python_version": "test"},
    )

    wrapper = HOGDiffWrapper()

    def fake_run_external(command, **_kwargs):
        command = list(command)
        if command[1].endswith("train.py"):
            checkpoint = Path(command[command.index("--checkpoint") + 1])
            manifest = Path(command[command.index("--manifest") + 1])
            mode = command[command.index("--mode") + 1]
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            with checkpoint.open("ab") as handle:
                handle.write((mode + "\n").encode())
            manifest.write_text(
                json.dumps(
                    {
                        "format": "grapher_hogdiff_training_stage_v1",
                        "mode": mode,
                        "configured_n_iters": 2 if mode == "higher-order" else 3,
                        "configured_batch_size": 2,
                        "initial_step": 0,
                        "checkpoint_step": 2 if mode == "higher-order" else 3,
                        "device": "cpu",
                    }
                ),
                encoding="utf-8",
            )
            return
        if command[1].endswith("generate.py"):
            assert command[command.index("--max-numerical-retries") + 1] == "8"
            output = Path(command[command.index("--output") + 1])
            manifest = Path(command[command.index("--manifest") + 1])
            count = int(command[command.index("--num-graphs") + 1])
            adjacency = np.zeros((count, 20, 20), dtype=np.int8)
            adjacency[:, 0, 1] = 1
            adjacency[:, 1, 0] = 1
            output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                output,
                adjacency=adjacency,
                num_nodes=np.full((count,), 2, dtype=np.int64),
                sample_index=np.arange(count, dtype=np.int64),
            )
            digest = hashlib.sha256(output.read_bytes()).hexdigest()
            manifest.write_text(
                json.dumps(
                    {
                        "format": "grapher_hogdiff_export_v1",
                        "num_generated": count,
                        "batch_size": count,
                        "sampling_rounds": 1,
                        "device": "cpu",
                        "postprocessing": "test",
                        "max_numerical_retries_per_batch": 8,
                        "numerical_retry_count": 0,
                        "numerical_retries": [],
                        "output": {"path": str(output), "sha256": digest},
                    }
                ),
                encoding="utf-8",
            )
            return
        raise AssertionError(command)

    monkeypatch.setattr(wrapper, "_run_external", fake_run_external)

    run = RunSpec.for_seed(
        model_id="hog_diff",
        dataset_id="community_small",
        seed=42,
        output_root=tmp_path / "baselines",
    )
    dataset = DatasetReference(
        benchmark_id="community_small",
        root=dataset_root,
        serialized_id="sbm",
        native_id="community_small",
    )
    training = wrapper.train(
        TrainRequest(
            run=run,
            dataset=dataset,
            options={
                "training_estimates": {"enabled": False},
                "runtime": {"device": "cpu", "progress": {"enabled": False}},
            },
        )
    )
    assert training.checkpoint_path.is_file()
    assert training.manifest_path.is_file()
    assert (training.run_dir / "train" / "native_dataset" / "community_small" / "community_small.pkl").is_file()

    generation = wrapper.generate(
        GenerateRequest(
            run=run,
            checkpoint_path=training.checkpoint_path,
            num_graphs=3,
            generation_seed=9,
            options={"runtime": {"device": "cpu", "progress": {"enabled": False}}},
        )
    )
    assert generation.num_generated == 3
    assert generation.graphs_path.is_file()
    assert all(path.is_file() for path in generation.native_artifacts)
    with generation.graphs_path.open("rb") as handle:
        graphs = pickle.load(handle)
    assert len(graphs) == 3
    assert all(graph.number_of_edges() == 1 for graph in graphs)
    generated_manifest = json.loads(generation.manifest_path.read_text(encoding="utf-8"))
    assert generated_manifest["native_diagnostics"]["numerical_retry_count"] == 0

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from grapher.models import (
    ArtifactCollisionError,
    ArtifactLayout,
    BaselineNotImplementedError,
    DatasetReference,
    GenerateRequest,
    RunSpec,
    TrainRequest,
    available_baselines,
    create_baseline,
    normalize_baseline_id,
)

EXPECTED_BASELINES = (
    "dhvae_hh",
    "digress",
    "catflow",
    "defog",
    "hog_diff",
    "flagg",
)


class BaselineRegistryTests(unittest.TestCase):
    def test_registry_ids_and_aliases(self) -> None:
        self.assertEqual(available_baselines(), EXPECTED_BASELINES)
        self.assertEqual(normalize_baseline_id("DH-VAE+HH"), "dhvae_hh")
        self.assertEqual(normalize_baseline_id("HOG-Diff"), "hog_diff")
        self.assertEqual(normalize_baseline_id("DeFoG"), "defog")

    def test_wrapper_statuses(self) -> None:
        expected = {model_id: "placeholder" for model_id in EXPECTED_BASELINES}
        expected.update({"dhvae_hh": "partial", "defog": "ready"})
        for model_id, status in expected.items():
            wrapper = create_baseline(model_id)
            self.assertEqual(wrapper.model_id, model_id)
            self.assertEqual(wrapper.capabilities.status, status)


class ArtifactLayoutTests(unittest.TestCase):
    def test_layout_separates_training_and_generation_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            layout = ArtifactLayout(
                "defog",
                "community_small",
                "seed_42",
                output_root=Path(tmp) / "outputs" / "baselines",
            )
            expected_run = (
                Path(tmp)
                / "outputs"
                / "baselines"
                / "defog"
                / "community_small"
                / "seed_42"
            )
            self.assertEqual(layout.run_dir, expected_run)
            self.assertEqual(
                layout.generated_graphs_path("seed_7_n_1024"),
                expected_run / "generations" / "seed_7_n_1024" / "base_graphs.pkl",
            )
            self.assertEqual(
                layout.estimated_training_graphs_path,
                expected_run / "train" / "training_estimates" / "estimated_graphs.pkl",
            )
            self.assertEqual(
                layout.ground_truth_training_graphs_path,
                expected_run
                / "train"
                / "training_estimates"
                / "ground_truth_graphs.pkl",
            )
            ids = {
                layout.default_generation_id(seed=1, num_graphs=1024),
                layout.default_generation_id(seed=2, num_graphs=1024),
                layout.default_generation_id(seed=1, num_graphs=2048),
            }
            self.assertEqual(len(ids), 3)

    def test_distinct_run_ids_isolate_replicates_with_the_same_seed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "outputs" / "baselines"
            first = RunSpec.for_seed(
                model_id="defog",
                dataset_id="community_small",
                seed=42,
                run_id="replicate_a",
                output_root=output_root,
            )
            second = RunSpec.for_seed(
                model_id="defog",
                dataset_id="community_small",
                seed=42,
                run_id="replicate_b",
                output_root=output_root,
            )

            self.assertEqual(first.train_seed, second.train_seed)
            self.assertNotEqual(first.run_id, second.run_id)
            self.assertNotEqual(first.layout.run_dir, second.layout.run_dir)
            self.assertNotEqual(first.layout.train_dir, second.layout.train_dir)
            self.assertNotEqual(
                first.layout.generation_dir("evaluation_batch"),
                second.layout.generation_dir("evaluation_batch"),
            )
            self.assertEqual(
                first.layout.run_dir,
                output_root / "defog" / "community_small" / "replicate_a",
            )
            self.assertEqual(
                second.layout.run_dir,
                output_root / "defog" / "community_small" / "replicate_b",
            )

    def test_unsafe_identifiers_and_collisions_are_rejected(self) -> None:
        for unsafe in ("../defog", "defog/run", "/absolute", "", ".."):
            with self.subTest(unsafe=unsafe), self.assertRaises(ValueError):
                ArtifactLayout(unsafe, "community_small", "seed_42")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "existing"
            path.mkdir()
            with self.assertRaises(ArtifactCollisionError):
                ArtifactLayout.require_available(path)


class DatasetAndRequestTests(unittest.TestCase):
    def test_dataset_aliases_and_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = Path(tmp) / "sbm"
            dataset_dir.mkdir()
            for split in ("train", "val", "test"):
                (dataset_dir / f"{split}.pkl").write_bytes(split.encode())
            dataset = DatasetReference(
                "community_small",
                root=Path(tmp),
                serialized_id="sbm",
                native_id="comm20",
            )
            self.assertEqual(dataset.dataset_dir, dataset_dir)
            self.assertEqual(dataset.native_id, "comm20")
            first = dataset.fingerprint()
            (dataset_dir / "test.pkl").write_bytes(b"changed")
            self.assertNotEqual(first, dataset.fingerprint())

    def test_requests_validate_identity_and_count(self) -> None:
        run = RunSpec.for_seed(model_id="defog", dataset_id="grid", seed=42)
        with self.assertRaises(ValueError):
            TrainRequest(run=run, dataset=DatasetReference("community_small"))
        with self.assertRaises(ValueError):
            GenerateRequest(
                run=run,
                checkpoint_path=Path("checkpoint.pt"),
                num_graphs=0,
                generation_seed=7,
            )


class PlaceholderBehaviorTests(unittest.TestCase):
    def test_unimplemented_wrapper_leaves_no_partial_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "baselines"
            run = RunSpec.for_seed(
                model_id="digress",
                dataset_id="community_small",
                seed=42,
                output_root=output_root,
            )
            wrapper = create_baseline("digress")
            train_request = TrainRequest(
                run=run,
                dataset=DatasetReference(
                    "community_small",
                    root=Path(tmp) / "datasets",
                    serialized_id="sbm",
                    native_id="comm20",
                ),
            )
            with self.assertRaises(BaselineNotImplementedError):
                wrapper.train(train_request)
            generation_request = GenerateRequest(
                run=run,
                checkpoint_path=Path(tmp) / "missing.pt",
                num_graphs=16,
                generation_seed=9,
            )
            with self.assertRaises(BaselineNotImplementedError):
                wrapper.generate(generation_request)
            self.assertFalse(output_root.exists())


if __name__ == "__main__":
    unittest.main()

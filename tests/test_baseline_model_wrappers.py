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
    def test_all_planned_baselines_are_registered(self) -> None:
        self.assertEqual(available_baselines(), EXPECTED_BASELINES)
        self.assertEqual(len(set(available_baselines())), len(EXPECTED_BASELINES))

    def test_display_name_aliases_resolve_to_canonical_ids(self) -> None:
        self.assertEqual(normalize_baseline_id("DH-VAE+HH"), "dhvae_hh")
        self.assertEqual(normalize_baseline_id("HOG-Diff"), "hog_diff")
        self.assertEqual(normalize_baseline_id("DeFoG"), "defog")

    def test_registry_builds_only_grapher_facing_wrappers(self) -> None:
        expected_status = {model_id: "placeholder" for model_id in EXPECTED_BASELINES}
        expected_status["dhvae_hh"] = "partial"
        for model_id in EXPECTED_BASELINES:
            wrapper = create_baseline(model_id)
            self.assertEqual(wrapper.model_id, model_id)
            self.assertEqual(
                wrapper.capabilities.status,
                expected_status[model_id],
            )


class ArtifactLayoutTests(unittest.TestCase):
    def test_run_location_contains_model_dataset_and_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            layout = ArtifactLayout(
                "defog",
                "community_small",
                "seed_42",
                output_root=Path(tmp) / "outputs" / "baselines",
            )
            self.assertEqual(
                layout.run_dir,
                Path(tmp)
                / "outputs"
                / "baselines"
                / "defog"
                / "community_small"
                / "seed_42",
            )
            self.assertEqual(
                layout.generated_graphs_path("seed_7_n_1024"),
                layout.run_dir
                / "generations"
                / "seed_7_n_1024"
                / "base_graphs.pkl",
            )

    def test_generation_seed_and_size_get_distinct_locations(self) -> None:
        layout = ArtifactLayout("defog", "community_small", "seed_42")
        first = layout.default_generation_id(seed=1, num_graphs=1024)
        second = layout.default_generation_id(seed=2, num_graphs=1024)
        third = layout.default_generation_id(seed=1, num_graphs=2048)
        self.assertEqual(len({first, second, third}), 3)

    def test_unsafe_identifiers_are_rejected(self) -> None:
        for unsafe in ("../defog", "defog/run", "/absolute", "", ".."):
            with self.subTest(unsafe=unsafe):
                with self.assertRaises(ValueError):
                    ArtifactLayout(unsafe, "community_small", "seed_42")

    def test_existing_artifact_is_not_silently_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "existing"
            path.mkdir()
            with self.assertRaises(ArtifactCollisionError):
                ArtifactLayout.require_available(path)
            self.assertEqual(ArtifactLayout.require_available(path, overwrite=True), path)


class DatasetAndRequestTests(unittest.TestCase):
    def test_dataset_reference_keeps_benchmark_storage_and_native_ids(self) -> None:
        dataset = DatasetReference(
            benchmark_id="community_small",
            serialized_id="sbm",
            native_id="comm20",
        )
        self.assertEqual(dataset.benchmark_id, "community_small")
        self.assertEqual(dataset.serialized_id, "sbm")
        self.assertEqual(dataset.native_id, "comm20")
        self.assertEqual(dataset.dataset_dir, Path("outputs/datasets/sbm"))

    def test_fingerprint_covers_all_project_splits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = Path(tmp) / "community_small"
            dataset_dir.mkdir()
            for split in ("train", "val", "test"):
                (dataset_dir / f"{split}.pkl").write_bytes(split.encode("utf-8"))
            dataset = DatasetReference("community_small", root=Path(tmp))
            first = dataset.fingerprint()
            (dataset_dir / "test.pkl").write_bytes(b"changed")
            second = dataset.fingerprint()
            self.assertNotEqual(first, second)

    def test_train_request_requires_matching_benchmark_id(self) -> None:
        run = RunSpec.for_seed(model_id="defog", dataset_id="grid", seed=42)
        with self.assertRaises(ValueError):
            TrainRequest(run=run, dataset=DatasetReference("community_small"))

    def test_num_graphs_must_be_positive(self) -> None:
        run = RunSpec.for_seed(
            model_id="defog", dataset_id="community_small", seed=42
        )
        with self.assertRaises(ValueError):
            GenerateRequest(
                run=run,
                checkpoint_path=Path("checkpoint.pt"),
                num_graphs=0,
                generation_seed=7,
            )


class PlaceholderBehaviorTests(unittest.TestCase):
    def test_train_and_generate_raise_without_creating_partial_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "baselines"
            run = RunSpec.for_seed(
                model_id="defog",
                dataset_id="community_small",
                seed=42,
                output_root=output_root,
            )
            wrapper = create_baseline("defog")
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
            self.assertFalse(output_root.exists())

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

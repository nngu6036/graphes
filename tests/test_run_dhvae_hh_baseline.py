from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from grapher.models import GenerationArtifacts, TrainingArtifacts
from scripts.run_dhvae_hh_baseline import DATASET_PROFILES, run_pipeline


class _WrapperStub:
    def __init__(self) -> None:
        self.training_request = None
        self.generation_request = None

    def train(self, request):
        self.training_request = request
        layout = request.run.layout
        checkpoint = layout.checkpoints_dir / "checkpoint.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"checkpoint")
        return TrainingArtifacts(
            run_dir=layout.run_dir,
            checkpoint_path=checkpoint,
            manifest_path=layout.training_manifest_path,
        )

    def generate(self, request):
        self.generation_request = request
        layout = request.run.layout
        generation_dir = layout.generation_dir(request.resolved_generation_id)
        graphs = layout.generated_graphs_path(request.resolved_generation_id)
        graphs.parent.mkdir(parents=True, exist_ok=True)
        graphs.write_bytes(b"graphs")
        return GenerationArtifacts(
            run_dir=layout.run_dir,
            generation_dir=generation_dir,
            graphs_path=graphs,
            manifest_path=layout.generation_manifest_path(
                request.resolved_generation_id
            ),
            num_requested=request.num_graphs,
            num_generated=request.num_graphs,
            graphs_sha256="abc",
        )


class RunDHVAEHHBaselineTests(unittest.TestCase):
    def test_all_dataset_profiles_and_request_wiring(self) -> None:
        for dataset_id, profile in DATASET_PROFILES.items():
            with self.subTest(dataset=dataset_id), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                prepared = root / "datasets" / profile.serialized_id
                prepared.mkdir(parents=True)
                for split in ("train", "val", "test"):
                    (prepared / f"{split}.pkl").write_bytes(b"prepared")
                stub = _WrapperStub()
                args = argparse.Namespace(
                    dataset=dataset_id,
                    num_samples=7,
                    seed_id=13,
                    dataset_root=root / "datasets",
                    output_root=root / "baselines",
                    experiment_config=None,
                    device="cpu",
                    run_id=None,
                    generation_id=None,
                    training_estimate_count=3,
                    disable_training_estimates=False,
                    overwrite=False,
                )
                with patch(
                    "scripts.run_dhvae_hh_baseline.create_baseline",
                    return_value=stub,
                ):
                    summary = run_pipeline(args)

                self.assertEqual(stub.training_request.run, stub.generation_request.run)
                self.assertEqual(
                    stub.training_request.dataset.serialized_id,
                    profile.serialized_id,
                )
                self.assertEqual(stub.training_request.run.train_seed, 13)
                self.assertEqual(stub.generation_request.generation_seed, 13)
                self.assertEqual(stub.generation_request.num_graphs, 7)
                self.assertEqual(
                    stub.generation_request.checkpoint_path,
                    stub.training_request.run.layout.checkpoints_dir
                    / "checkpoint.pt",
                )
                self.assertEqual(summary["generation_id"], "seed_13_n_7")


if __name__ == "__main__":
    unittest.main()

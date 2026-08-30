from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from grapher.models import GenerationArtifacts, TrainingArtifacts
from scripts.run_defog_baseline import DATASET_PROFILES, run_pipeline


class _WrapperStub:
    def __init__(self) -> None:
        self.training_request = None
        self.generation_request = None

    def train(self, request):
        self.training_request = request
        layout = request.run.layout
        checkpoint = layout.checkpoints_dir / "model.ckpt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"checkpoint")
        log_path = layout.training_log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("trained\n", encoding="utf-8")
        return TrainingArtifacts(
            run_dir=layout.run_dir,
            checkpoint_path=checkpoint,
            manifest_path=layout.training_manifest_path,
            log_path=log_path,
        )

    def generate(self, request):
        self.generation_request = request
        layout = request.run.layout
        generation_dir = layout.generation_dir(request.resolved_generation_id)
        generation_dir.mkdir(parents=True, exist_ok=True)
        graphs = layout.generated_graphs_path(request.resolved_generation_id)
        graphs.write_bytes(b"graphs")
        log_path = layout.generation_log_path(request.resolved_generation_id)
        log_path.write_text("generated\n", encoding="utf-8")
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
            log_path=log_path,
        )


class RunDeFoGBaselineTests(unittest.TestCase):
    def test_progress_options_are_forwarded_to_training_and_generation(self) -> None:
        profile = DATASET_PROFILES["community_small"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared = root / "datasets" / profile.serialized_id
            prepared.mkdir(parents=True)
            for split in ("train", "val", "test"):
                (prepared / f"{split}.pkl").write_bytes(b"prepared")
            stub = _WrapperStub()
            args = argparse.Namespace(
                dataset="community_small",
                num_samples=7,
                seed_id=13,
                dataset_root=root / "datasets",
                output_root=root / "baselines",
                run_id=None,
                generation_id=None,
                wrapper_config=None,
                resume_from=None,
                defog_root=None,
                defog_python=None,
                device="gpu",
                gpu_id=3,
                overwrite=False,
                progress_interval_seconds=4.0,
                epoch_progress_interval=2,
                generation_progress_every_batches=3,
                no_stream_subprocess_output=False,
                quiet=False,
            )
            with patch(
                "scripts.run_defog_baseline.create_baseline",
                return_value=stub,
            ):
                summary = run_pipeline(args)

            expected_progress = {
                "enabled": True,
                "stream_output": True,
                "interval_seconds": 4.0,
                "generation_batch_interval": 3,
                "epoch_interval": 2,
            }
            self.assertEqual(
                stub.training_request.options["runtime"]["progress"],
                expected_progress,
            )
            self.assertEqual(stub.training_request.options["runtime"]["gpus"], 1)
            self.assertEqual(stub.training_request.options["runtime"]["device"], "cuda")
            self.assertEqual(
                stub.training_request.options["runtime"]["cuda_visible_devices"], "3"
            )
            self.assertTrue(stub.training_request.options["runtime"]["require_cuda"])
            self.assertEqual(
                stub.generation_request.options["runtime"]["progress"],
                expected_progress,
            )
            self.assertEqual(stub.generation_request.options["runtime"]["device"], "cuda")
            self.assertEqual(
                stub.generation_request.options["runtime"]["cuda_visible_devices"], "3"
            )
            self.assertTrue(
                stub.training_request.options["training_estimates"]["enabled"]
            )
            self.assertEqual(summary["generation_id"], "seed_13_n_7")
            self.assertEqual(summary["num_samples"], 7)


if __name__ == "__main__":
    unittest.main()

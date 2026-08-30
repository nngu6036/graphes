from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

from grapher.models import GenerationArtifacts, TrainingArtifacts
from scripts.run_digress_baseline import DATASET_PROFILES, run_pipeline


class WrapperStub:
    def __init__(self) -> None:
        self.training_request = None
        self.generation_request = None

    def train(self, request):
        self.training_request = request
        layout = request.run.layout
        checkpoint = layout.checkpoints_dir / "model.ckpt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"checkpoint")
        layout.training_log_path.parent.mkdir(parents=True, exist_ok=True)
        layout.training_log_path.write_text("trained\n", encoding="utf-8")
        return TrainingArtifacts(
            run_dir=layout.run_dir,
            checkpoint_path=checkpoint,
            manifest_path=layout.training_manifest_path,
            log_path=layout.training_log_path,
        )

    def generate(self, request):
        self.generation_request = request
        layout = request.run.layout
        generation_dir = layout.generation_dir(request.resolved_generation_id)
        generation_dir.mkdir(parents=True, exist_ok=True)
        graphs = layout.generated_graphs_path(request.resolved_generation_id)
        graphs.write_bytes(b"graphs")
        log = layout.generation_log_path(request.resolved_generation_id)
        log.write_text("generated\n", encoding="utf-8")
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
            log_path=log,
        )


def test_runner_forwards_training_and_progress_options(tmp_path: Path) -> None:
    profile = DATASET_PROFILES["community_small"]
    prepared = tmp_path / "datasets" / profile.serialized_id
    prepared.mkdir(parents=True)
    for split in ("train", "val", "test"):
        (prepared / f"{split}.pkl").write_bytes(b"prepared")
    stub = WrapperStub()
    args = argparse.Namespace(
        dataset="community_small",
        num_samples=7,
        seed_id=13,
        dataset_root=tmp_path / "datasets",
        output_root=tmp_path / "baselines",
        run_id=None,
        generation_id=None,
        wrapper_config=None,
        resume_from=None,
        digress_root=None,
        digress_python=None,
        device="gpu",
        gpu_id=2,
        overwrite=False,
        n_epochs=123,
        batch_size=16,
        generation_batch_size=8,
        num_workers=0,
        check_val_every_n_epochs=5,
        save_every_n_epochs=10,
        skip_training_estimates=False,
        training_estimate_count=4,
        progress_interval_seconds=4.0,
        epoch_progress_interval=2,
        generation_progress_every_batches=3,
        no_stream_subprocess_output=False,
        quiet=False,
    )
    with patch(
        "scripts.run_digress_baseline.create_baseline", return_value=stub
    ):
        summary = run_pipeline(args)

    assert stub.training_request.options["n_epochs"] == 123
    assert stub.training_request.options["batch_size"] == 16
    assert stub.training_request.options["generation_batch_size"] == 8
    assert stub.training_request.options["training_estimates"] == {
        "enabled": True,
        "num_graphs": 4,
    }
    expected_progress = {
        "enabled": True,
        "stream_output": True,
        "interval_seconds": 4.0,
        "generation_batch_interval": 3,
        "epoch_interval": 2,
    }
    assert stub.training_request.options["runtime"]["progress"] == expected_progress
    assert stub.training_request.options["runtime"]["gpus"] == 1
    assert stub.training_request.options["runtime"]["device"] == "gpu"
    assert stub.training_request.options["runtime"]["cuda_visible_devices"] == "2"
    assert stub.training_request.options["runtime"]["require_cuda"] is True
    assert stub.generation_request.options["runtime"]["progress"] == expected_progress
    assert stub.generation_request.options["runtime"]["device"] == "gpu"
    assert stub.generation_request.options["runtime"]["cuda_visible_devices"] == "2"
    assert summary["generation_id"] == "seed_13_n_7"

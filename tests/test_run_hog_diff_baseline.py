from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

from grapher.models import GenerationArtifacts, TrainingArtifacts
from scripts.run_hog_diff_baseline import DATASET_PROFILES, run_pipeline


class WrapperStub:
    def __init__(self) -> None:
        self.training_request = None
        self.generation_request = None

    def train(self, request):
        self.training_request = request
        layout = request.run.layout
        checkpoint = layout.checkpoints_dir / "hog_diff.pth"
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
        generation_dir.mkdir(parents=True, exist_ok=True)
        graphs = layout.generated_graphs_path(request.resolved_generation_id)
        graphs.write_bytes(b"graphs")
        return GenerationArtifacts(
            run_dir=layout.run_dir,
            generation_dir=generation_dir,
            graphs_path=graphs,
            manifest_path=layout.generation_manifest_path(request.resolved_generation_id),
            num_requested=request.num_graphs,
            num_generated=request.num_graphs,
            graphs_sha256="abc",
        )


def test_runner_forwards_two_stage_overrides_and_dataset_aliases(tmp_path: Path) -> None:
    profile = DATASET_PROFILES["qm9"]
    prepared = tmp_path / "datasets" / profile.serialized_id
    prepared.mkdir(parents=True)
    for split in ("train", "val", "test"):
        (prepared / f"{split}.pkl").write_bytes(b"prepared")
    stub = WrapperStub()
    args = argparse.Namespace(
        dataset="qm9",
        num_samples=17,
        seed_id=42,
        dataset_root=tmp_path / "datasets",
        output_root=tmp_path / "baselines",
        run_id=None,
        generation_id=None,
        wrapper_config=None,
        resume_from=None,
        hogdiff_root=None,
        hogdiff_python=None,
        overwrite=False,
        ho_iters=123,
        ou_iters=456,
        ho_batch_size=8,
        ou_batch_size=16,
        num_workers=0,
        generation_batch_size=11,
        device="cpu",
        cuda_visible_devices="1",
        timeout_seconds=120.0,
        skip_training_estimates=False,
        training_estimate_count=4,
        progress_interval_seconds=3.0,
        iteration_progress_interval=7,
        generation_progress_every_batches=2,
        no_stream_subprocess_output=False,
        quiet=False,
    )
    with patch("scripts.run_hog_diff_baseline.create_baseline", return_value=stub):
        summary = run_pipeline(args)

    request = stub.training_request
    assert request.dataset.serialized_id == "qm9_attributed"
    assert request.dataset.native_id == "qm9"
    assert request.options["higher_order"] == {"n_iters": 123, "batch_size": 8}
    assert request.options["ou"] == {"n_iters": 456, "batch_size": 16}
    assert request.options["generation_batch_size"] == 11
    assert request.options["training_estimates"] == {"enabled": True, "num_graphs": 4}
    assert request.options["runtime"]["device"] == "cpu"
    assert request.options["runtime"]["progress"]["iteration_interval"] == 7
    assert stub.generation_request.options["generation_batch_size"] == 11
    assert summary["model"] == "hog_diff"
    assert summary["native_dataset"] == "qm9"

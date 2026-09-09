from __future__ import annotations

import json
import sys
from pathlib import Path

import networkx as nx
import pytest

from grapher.data.io import save_dataset_splits
from grapher.rewiring_mlp.generic.spectral_model import load_topology_spectral_checkpoint
from grapher.utils.io import load_yaml, save_yaml
from scripts import train_topology_grapher as train


@pytest.mark.parametrize("override_flag", [None, "--set", "--override"])
def test_training_applies_ablation_overrides_to_model_loss_and_checkpoint(
    tmp_path: Path, monkeypatch, override_flag: str | None,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    config = load_yaml(
        repository / "configs/experiments/grapher/community_small_topology_spectral_debug.yaml"
    )
    root = tmp_path / "datasets"
    save_dataset_splits(
        "tiny", {split: [nx.cycle_graph(5)] for split in ("train", "val", "test")},
        {}, root,
    )
    config["dataset"] = {"name": "tiny", "root": str(root), "build_if_missing": False}
    config["summary_diffusion"].update({
        "storage": "eager", "samples_per_graph": 2, "paths_per_graph": 1,
    })
    config["topology_predictor"].update({
        "hidden_dim": 8, "edge_dim": 8, "graph_dim": 8,
        "spectral_dim": 8, "spectral_layers": 1, "spectral_ff_dim": 16,
    })
    config["topology_predictor"]["loss_weights"].update({
        "spectrum": 0.5, "moment2": 0.1, "low_frequency": 0.2,
    })
    config_path = tmp_path / "config.yaml"
    save_yaml(config, config_path)
    original_yaml = config_path.read_bytes()
    output = tmp_path / "training"
    argv = [
        "train_topology_grapher.py", "--config", str(config_path),
        "--output-dir", str(output), "--epochs", "1", "--batch-size", "2",
        "--seed", "42", "--device", "cpu",
    ]
    overrides = [
        "structure_summary_prediction.clustering_coefficient=false",
        "topology_predictor.loss_weights.spectrum=1.0",
        "topology_predictor.loss_weights.moment2=0.0",
        "topology_predictor.loss_weights.low_frequency=0.0",
        "topology_predictor.loss_weights.clustering_coefficient=0.0",
    ] if override_flag else []
    for expression in overrides:
        argv.extend([override_flag, expression])
    monkeypatch.setattr(sys, "argv", argv)
    train.main()

    report = json.loads((output / "training_report.json").read_text())
    model, _, checkpoint = load_topology_spectral_checkpoint(
        output / "checkpoint.pt", device="cpu",
    )
    clustering_enabled = override_flag is None
    assert model.predict_clustering_coefficient is clustering_enabled
    assert (model.clustering_coefficient_head is not None) is clustering_enabled
    assert report["predictor_targets"]["clean_average_clustering_coefficient"] is clustering_enabled
    assert report["config_overrides"] == overrides
    assert report["active_losses"] == (
        ["clustering_coefficient", "low_frequency", "moment2", "spectrum"]
        if clustering_enabled else ["spectrum"]
    )
    resolved = checkpoint["config"]
    assert resolved["structure_summary_prediction"]["clustering_coefficient"] is clustering_enabled
    if overrides:
        weights = resolved["topology_predictor"]["loss_weights"]
        assert weights["spectrum"] == 1.0
        assert all(weights[key] == 0.0 for key in (
            "moment2", "low_frequency", "clustering_coefficient",
        ))
        for phase in ("train", "val"):
            metrics = report["history"][0]
            assert metrics[f"{phase}_loss"] == pytest.approx(metrics[f"{phase}_spectral_loss"])
    assert config_path.read_bytes() == original_yaml

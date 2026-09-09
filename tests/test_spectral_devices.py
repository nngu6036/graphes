from __future__ import annotations

import json
import sys

import networkx as nx
import numpy as np
import pytest
import torch

from grapher.data.io import save_dataset_splits
from grapher.rewiring_mlp.attributed.spectral import (
    attributed_laplacian_spectra,
    batched_attributed_laplacian_spectra,
)
from grapher.rewiring_mlp.generic.spectral import (
    batched_laplacian_eigenvalues,
    laplacian_eigenvalues,
)
from grapher.utils.io import load_pickle, save_yaml


@pytest.mark.parametrize("device", ["gpu", "auto"])
@pytest.mark.parametrize("cuda_available", [False, True])
@pytest.mark.parametrize("batched,scalar", [
    (batched_laplacian_eigenvalues, laplacian_eigenvalues),
    (batched_attributed_laplacian_spectra, attributed_laplacian_spectra),
])
def test_batched_spectra_resolve_device_aliases(
    monkeypatch, device, cuda_available, batched, scalar,
) -> None:
    graphs = [nx.path_graph(4), nx.cycle_graph(4)]
    for graph in graphs:
        nx.set_edge_attributes(graph, 2, "bond_type")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    as_tensor = torch.as_tensor
    requested_devices = []

    def capture_tensor_device(data, *, dtype, device):
        requested_devices.append(device)
        # Exercise CUDA selection on CPU-only test hosts as well.
        return as_tensor(data, dtype=dtype, device="cpu")

    monkeypatch.setattr(torch, "as_tensor", capture_tensor_device)
    spectra = batched(graphs, device=device, backend="auto", batch_size=1)
    assert requested_devices == ([torch.device("cuda")] * 2 if cuda_available else [])
    for graph, spectrum in zip(graphs, spectra):
        np.testing.assert_allclose(spectrum, scalar(graph), atol=1.0e-10)


def test_oracle_projection_cli_accepts_gpu_alias(tmp_path, monkeypatch) -> None:
    from scripts import diagnose_spectral_oracle_projection as diagnose

    # The shared resolver uses CPU when CUDA is unavailable.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    graph = nx.Graph([
        (0, 1), (0, 2), (0, 3), (1, 2), (2, 4), (3, 5), (4, 5),
    ])
    root = tmp_path / "datasets"
    save_dataset_splits(
        "oracle_test", {split: [graph] for split in ("train", "val", "test")},
        {}, root,
    )
    config = tmp_path / "config.yaml"
    save_yaml({
        "dataset": {"name": "oracle_test", "root": str(root), "build_if_missing": False},
        "topology_refiner": {
            "steps": 2,
            "proposal_budget": -1,
            "valid_candidate_budget": -1,
            "spectral_guidance": {"min_clean_mix": 1.0, "max_clean_mix": 1.0},
        },
    }, config)
    output = tmp_path / "oracle"
    monkeypatch.setattr(sys, "argv", [
        "diagnose_spectral_oracle_projection.py", "--config", str(config),
        "--split", "val", "--max-graphs", "1", "--seed", "42",
        "--device", "gpu", "--output-dir", str(output),
    ])
    diagnose.main()
    report = json.loads((output / "oracle_projection_report.json").read_text())
    assert report["num_graphs"] == 1
    assert report["degree_preservation_rate"] == 1.0
    assert report["connectedness_rate"] == 1.0
    assert report["mean_final_spectral_distance"] <= report["mean_initial_spectral_distance"]
    assert len(load_pickle(output / "oracle_spectral_refined_graphs.pkl")) == 1

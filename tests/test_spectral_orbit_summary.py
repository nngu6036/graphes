from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
import torch

from grapher.properties.summary import python_orbit_count_vector
from grapher.rewiring_mlp.generic.orbit import (
    TOPOLOGY_ORBIT_SUMMARY_WIDTH,
    extract_orbit_summary,
    orbit_summary_distance,
    orbit_summary_width,
    validate_orbit_summary,
)
from grapher.rewiring_mlp.generic.rewiring import propose_valid_topology_swaps
from grapher.rewiring_mlp.generic.spectral import laplacian_eigenvalues
from grapher.rewiring_mlp.generic.spectral_data import (
    TopologySpectralExample,
    build_spectral_diffusion_examples,
    collate_spectral_examples,
)
from grapher.rewiring_mlp.generic.spectral_model import (
    TopologySpectralTransformerPredictor,
    load_topology_spectral_checkpoint,
    save_topology_spectral_checkpoint,
)
from grapher.rewiring_mlp.generic.spectral_refiner import (
    SpectralPrediction,
    SpectralRefinerConfig,
    predict_clean_spectrum,
    refine_graph_with_spectral_predictions,
    score_spectral_candidates,
)


def graph() -> nx.Graph:
    return nx.Graph(
        [(0, 1), (0, 2), (0, 3), (1, 2), (2, 4), (3, 5), (4, 5)]
    )


def model(*, orbit: bool = True) -> TopologySpectralTransformerPredictor:
    return TopologySpectralTransformerPredictor(
        hidden_dim=8,
        edge_dim=8,
        graph_dim=8,
        num_layers=1,
        spectral_dim=16,
        spectral_layers=1,
        spectral_heads=4,
        spectral_ff_dim=32,
        use_graph_context=False,
        predict_orbit_summary=orbit,
        orbit_summary_width=15,
    )


def example(g: nx.Graph) -> TopologySpectralExample:
    spectrum = laplacian_eigenvalues(g)
    return TopologySpectralExample(
        current_graph=g,
        time=0.4,
        current_spectrum=spectrum,
        source_spectrum=spectrum,
        clean_spectrum_target=spectrum,
        clean_orbit_summary_target=extract_orbit_summary(g),
    )


def test_orbit_summary_matches_python_evaluator_and_is_permutation_invariant():
    g = graph()
    expected = python_orbit_count_vector(g)
    observed = extract_orbit_summary(g)
    assert observed.shape == (TOPOLOGY_ORBIT_SUMMARY_WIDTH,)
    np.testing.assert_allclose(observed, expected)
    permuted = nx.relabel_nodes(g, {node: node + 17 for node in g})
    np.testing.assert_allclose(extract_orbit_summary(permuted), observed)


def test_orbit_summary_config_and_validation():
    assert orbit_summary_width({}) is None
    assert orbit_summary_width({"orbit_summary": True}) == 15
    with pytest.raises(ValueError, match="orbit_width"):
        orbit_summary_width({"orbit_summary": True, "orbit_width": 14})
    with pytest.raises(ValueError, match="width 15"):
        validate_orbit_summary(np.zeros(14))
    with pytest.raises(ValueError, match="negative"):
        validate_orbit_summary(np.r_[-1.0, np.zeros(14)])


def test_log_orbit_distance_is_zero_only_for_equal_vectors():
    a = extract_orbit_summary(nx.path_graph(5))
    b = extract_orbit_summary(nx.cycle_graph(5))
    assert orbit_summary_distance(a, a) == pytest.approx(0.0)
    assert orbit_summary_distance(a, b) > 0.0
    assert orbit_summary_distance(a, b, distance="raw_rmse") > 0.0


def test_diffusion_examples_carry_fixed_clean_orbit_target():
    g = graph()
    examples, _ = build_spectral_diffusion_examples(
        [g],
        diffusion_config={"samples_per_graph": 3, "paths_per_graph": 1},
        source_config={"ensure_connected_source": True, "random_relabel_source": False},
        structure_summary_config={"orbit_summary": True, "orbit_width": 15},
        seed=7,
    )
    expected = extract_orbit_summary(g)
    assert len(examples) == 3
    for row in examples:
        np.testing.assert_allclose(row.clean_orbit_summary_target, expected)
    batch = collate_spectral_examples(examples)
    assert batch.clean_orbit_summary_target.shape == (3, 15)


def test_orbit_head_loss_and_shared_representation_have_gradients():
    batch = collate_spectral_examples([example(graph()), example(nx.cycle_graph(6))])
    m = model()
    outputs = m(batch)
    assert outputs["clean_orbit_summary"].shape == (2, 15)
    assert torch.all(outputs["clean_orbit_summary"] >= 0)
    loss, metrics = m.loss(
        batch,
        loss_weights={"spectrum": 1.0, "moment2": 0.0, "orbit_summary": 1.0},
    )
    assert np.isfinite(metrics["orbit_summary_log_rmse"])
    loss.backward()
    for parameter in (m.orbit_summary_head[-1].weight, m.spectral_token_in[0].weight):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_checkpoint_roundtrip_and_legacy_checkpoint_compatibility(tmp_path):
    path = tmp_path / "orbit.pt"
    m = model().eval()
    save_topology_spectral_checkpoint(m, path)
    loaded, _, checkpoint = load_topology_spectral_checkpoint(path, device="cpu")
    assert loaded.predict_orbit_summary
    assert loaded.orbit_summary_width == 15
    assert checkpoint["model_config"]["predict_orbit_summary"] is True

    legacy = model(orbit=False)
    save_topology_spectral_checkpoint(legacy, path)
    state = torch.load(path, map_location="cpu", weights_only=False)
    state["model_config"].pop("predict_orbit_summary")
    state["model_config"].pop("orbit_summary_width")
    torch.save(state, path)
    loaded, _, _ = load_topology_spectral_checkpoint(path, device="cpu")
    assert loaded.orbit_summary_head is None


def test_predict_clean_spectrum_returns_orbit_summary_without_target_input():
    prediction = predict_clean_spectrum(model(), graph(), time=0.0, device="cpu")
    assert prediction.clean_orbit_summary is not None
    assert prediction.clean_orbit_summary.shape == (15,)
    assert np.all(prediction.clean_orbit_summary >= 0.0)


def test_orbit_only_guidance_improves_evaluator_compatible_summary():
    g = graph()
    candidates, candidate_graphs, _ = propose_valid_topology_swaps(
        g,
        proposal_budget=-1,
        valid_candidate_budget=-1,
        preserve_connectivity=True,
        rng=np.random.default_rng(0),
    )
    source_orbit = extract_orbit_summary(g)
    target_action = next(
        action
        for action, candidate in candidate_graphs.items()
        if orbit_summary_distance(source_orbit, extract_orbit_summary(candidate)) > 1.0e-8
    )
    target = candidate_graphs[target_action]
    target_orbit = extract_orbit_summary(target)
    target_spectrum = laplacian_eigenvalues(target)
    cfg = SpectralRefinerConfig.from_dict(
        {
            "steps": 1,
            "guidance_mode": "orbit",
            "proposal_budget": -1,
            "valid_candidate_budget": -1,
            "min_improvement": 1.0e-10,
            "spectral_guidance": {"weight": 0.0, "min_clean_mix": 1.0, "max_clean_mix": 1.0},
            "orbit_guidance": {"weight": 1.0, "distance": "log_rmse"},
        }
    )
    rows = score_spectral_candidates(
        g,
        candidates,
        clean_spectrum=target_spectrum,
        next_spectrum_target=target_spectrum,
        clean_orbit_summary=target_orbit,
        config=cfg,
        candidate_graphs=candidate_graphs,
    )
    assert min(row["candidate_orbit_discrepancy"] for row in rows) == pytest.approx(0.0)

    def oracle(_model, current, **kwargs):
        return SpectralPrediction(
            target_spectrum,
            laplacian_eigenvalues(current),
            float(target_spectrum.sum()),
            float(target_spectrum @ target_spectrum),
            clean_orbit_summary=target_orbit,
        )

    final, trace = refine_graph_with_spectral_predictions(
        g,
        model=None,
        refiner_config=cfg,
        prediction_fn=oracle,
        return_trace=True,
    )
    assert dict(final.degree()) == dict(g.degree())
    assert nx.is_connected(final)
    assert orbit_summary_distance(extract_orbit_summary(final), target_orbit) < orbit_summary_distance(source_orbit, target_orbit)
    accepted = next(row for row in trace if row["accepted"])
    assert accepted["orbit_gain"] > 0.0


def test_train_generate_evaluate_orbit_smoke(tmp_path, monkeypatch):
    import json
    import sys
    from pathlib import Path

    from grapher.data.io import save_dataset_splits
    from grapher.utils.io import load_pickle, load_yaml, save_yaml
    from scripts import evaluate_graph_generation_report as evaluate
    from scripts import run_topology_grapher as generate
    from scripts import train_topology_grapher as train

    repo = Path(__file__).resolve().parents[1]
    cfg = load_yaml(
        repo
        / "configs/experiments/grapher/community_small_topology_spectral_clustering_histogram_orbit.yaml"
    )
    g1 = graph()
    g2 = nx.cycle_graph(6)
    root = tmp_path / "datasets"
    save_dataset_splits(
        "tiny",
        {split: [g1, g2] for split in ("train", "val", "test")},
        {},
        root,
    )
    cfg["dataset"] = {"name": "tiny", "root": str(root), "build_if_missing": False}
    cfg["benchmark"] = "tiny"
    cfg["summary_diffusion"].update(storage="streaming", samples_per_graph=2, paths_per_graph=1)
    cfg["topology_predictor"].update(
        hidden_dim=8,
        edge_dim=8,
        graph_dim=8,
        spectral_dim=16,
        spectral_layers=1,
        spectral_heads=4,
        spectral_ff_dim=32,
        epochs=1,
        batch_size=2,
    )
    cfg["topology_refiner"].update(steps=2, proposal_budget=32, valid_candidate_budget=8)
    cfg["topology_refiner"]["candidate_search"]["spectrum_batch_size"] = 8
    config = tmp_path / "config.yaml"
    save_yaml(cfg, config)

    training = tmp_path / "train"
    monkeypatch.setattr(
        sys,
        "argv",
        ["train", "--config", str(config), "--output-dir", str(training), "--device", "cpu"],
    )
    train.main()
    report = json.loads((training / "training_report.json").read_text())
    assert "orbit_summary" in report["active_losses"]
    assert report["predictor_targets"]["orbit_summary_width"] == 15
    assert np.isfinite(report["history"][0]["val_orbit_summary_log_rmse"])

    generated = tmp_path / "generated"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate",
            "--config",
            str(config),
            "--checkpoint",
            str(training / "checkpoint.pt"),
            "--output-dir",
            str(generated),
            "--num-generate",
            "2",
            "--device",
            "cpu",
        ],
    )
    generate.main()
    generation_report = json.loads((generated / "report.json").read_text())
    assert generation_report["diagnostics"]["rewiring_guidance_mode"] == "clustering_orbit"
    assert generation_report["diagnostics"]["predictor_orbit_summary_enabled"] is True
    source = load_pickle(generated / "coarse_graphs.pkl")
    final = load_pickle(generated / "topology_refined_graphs.pkl")
    assert all(dict(a.degree()) == dict(b.degree()) for a, b in zip(source, final))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate",
            "--config",
            str(config),
            "--generated-dir",
            str(generated),
            "--output-dir",
            str(tmp_path / "eval"),
            "--num-samples",
            "2",
            "--dpi",
            "40",
        ],
    )
    evaluate.main()
    assert (tmp_path / "eval" / "graph_mmd_metrics.csv").exists()

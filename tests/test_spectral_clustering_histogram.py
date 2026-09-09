from __future__ import annotations

import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import torch

from grapher.data.io import save_dataset_splits
from grapher.properties.summary import clustering_histogram
from grapher.rewiring_mlp.generic.clustering import (
    clustering_histogram_bins, clustering_histogram_wasserstein,
    extract_clustering_histogram, validate_clustering_histogram,
)
from grapher.rewiring_mlp.generic.rewiring import propose_valid_topology_swaps
from grapher.rewiring_mlp.generic.spectral import laplacian_eigenvalues
from grapher.rewiring_mlp.generic.spectral_data import (
    TopologySpectralExample, TopologySpectralDiffusionIterableDataset,
    build_spectral_diffusion_examples, collate_spectral_examples,
)
from grapher.rewiring_mlp.generic.spectral_model import (
    TopologySpectralTransformerPredictor, load_topology_spectral_checkpoint,
    save_topology_spectral_checkpoint,
)
from grapher.rewiring_mlp.generic.spectral_refiner import (
    SpectralPrediction, SpectralRefinerConfig, predict_clean_spectrum,
    score_spectral_candidates, refine_graph_with_spectral_predictions,
)
from grapher.utils.io import load_pickle, load_yaml, save_yaml


def model(*, histogram=True, bins=10):
    return TopologySpectralTransformerPredictor(
        hidden_dim=8, edge_dim=8, graph_dim=8, num_layers=1,
        spectral_dim=16, spectral_layers=1, spectral_heads=4,
        spectral_ff_dim=32, use_graph_context=False,
        predict_clustering_histogram=histogram, clustering_histogram_bins=bins,
    )


def graph():
    return nx.Graph([(0, 1), (0, 2), (0, 3), (1, 2), (2, 4), (3, 5), (4, 5)])


def example(g, bins=10):
    spec = laplacian_eigenvalues(g)
    return TopologySpectralExample(
        current_graph=g, time=0.4, current_spectrum=spec, source_spectrum=spec,
        clean_spectrum_target=spec,
        clean_clustering_histogram_target=extract_clustering_histogram(g, bins),
    )


@pytest.mark.parametrize('bins', [2, 10, 100])
def test_extractor_is_identical_to_evaluator_and_handles_boundaries(bins):
    # Triangle + leaf: [1/3, 1, 1, 0]; tests both endpoint bins and isolates.
    g = nx.Graph([(0, 1), (1, 2), (2, 0), (0, 3)])
    expected = clustering_histogram(g, bins)
    observed = extract_clustering_histogram(g, bins)
    np.testing.assert_array_equal(observed, expected)
    assert observed[0] >= 0.25
    assert observed[-1] >= 0.5
    assert observed.sum() == pytest.approx(1.0)
    permuted = nx.relabel_nodes(g, {i: 10-i for i in g})
    np.testing.assert_array_equal(extract_clustering_histogram(permuted, bins), observed)
    assert extract_clustering_histogram(nx.empty_graph(1), bins)[0] == 1.0
    assert extract_clustering_histogram(nx.empty_graph(0), bins)[0] == 1.0


@pytest.mark.parametrize('values', [[0.2, 0.2], [np.nan, 0.0], [-0.1, 1.1], [1.0], [[1.0, 0.0]]])
def test_invalid_histograms_fail(values):
    with pytest.raises(ValueError):
        validate_clustering_histogram(np.asarray(values))


def test_bins_and_w1_are_order_aware():
    assert clustering_histogram_bins({}) is None
    assert clustering_histogram_bins({'clustering_histogram': True}) == 100
    for invalid in (True, 1, 2.5):
        with pytest.raises(ValueError):
            clustering_histogram_bins({'clustering_histogram': True, 'clustering_bins': invalid})
    a = np.array([1.0, 0.0, 0.0, 0.0])
    near = np.array([0.0, 1.0, 0.0, 0.0])
    far = np.array([0.0, 0.0, 0.0, 1.0])
    assert clustering_histogram_wasserstein(a, a) == 0.0
    assert clustering_histogram_wasserstein(a, near) == pytest.approx(0.25)
    assert clustering_histogram_wasserstein(a, far) == pytest.approx(0.75)
    # Same binned mean, but different distributions.
    assert clustering_histogram_wasserstein(np.array([0.5, 0, 0, 0.5]), np.array([0, 0.5, 0.5, 0])) > 0


@pytest.mark.parametrize('storage', ['eager', 'cached', 'uncached'])
def test_clean_histogram_is_constant_across_diffusion_times(storage):
    g = graph()
    kwargs = dict(
        diffusion_config={'samples_per_graph': 3, 'paths_per_graph': 1, 'spectral_sigma': 0.4, 'cache_endpoints': storage != 'uncached'},
        source_config={'ensure_connected_source': True, 'random_relabel_source': False},
        structure_summary_config={'clustering_histogram': True, 'clustering_bins': 10},
        seed=3,
    )
    if storage == 'eager':
        examples, _ = build_spectral_diffusion_examples([g], **kwargs)
    else:
        ds = TopologySpectralDiffusionIterableDataset([g], **kwargs)
        examples = list(ds)
        ds.set_epoch(1)
        second = list(ds)
        assert not np.allclose(examples[0].current_spectrum, second[0].current_spectrum)
        np.testing.assert_array_equal(examples[0].clean_clustering_histogram_target, second[0].clean_clustering_histogram_target)
    assert len(examples) == 3
    expected = extract_clustering_histogram(g, 10)
    for row in examples:
        np.testing.assert_allclose(row.clean_clustering_histogram_target, expected, atol=1e-7)
    batch = collate_spectral_examples(examples).to('cpu')
    assert batch.clean_clustering_histogram_target.shape == (3, 10)


def test_batch_validates_missing_or_inconsistent_width():
    a, b = example(graph(), 10), example(graph(), 20)
    with pytest.raises(ValueError, match='bin count'):
        collate_spectral_examples([a, b])
    b.clean_clustering_histogram_target = None
    with pytest.raises(ValueError, match='with and without'):
        collate_spectral_examples([a, b])


def test_head_normalization_gradients_and_no_target_leakage():
    examples = [example(nx.cycle_graph(5)), example(nx.complete_graph(4))]
    batch = collate_spectral_examples(examples)
    m = model().eval()
    output = m(batch)
    hist = output['clean_clustering_histogram']
    assert hist.shape == (2, 10)
    assert torch.all(hist > 0)
    torch.testing.assert_close(hist.sum(-1), torch.ones(2))
    loss, metrics = m.loss(batch, loss_weights={'spectrum': 1, 'moment2': 0, 'clustering_histogram': 1})
    assert all(np.isfinite(x) for x in metrics.values())
    loss.backward()
    for param in (m.clustering_histogram_head[-1].weight, m.spectral_token_in[0].weight, m.gap_head[-1].weight):
        assert param.grad is not None and torch.isfinite(param.grad).all()
        assert param.grad.abs().sum() > 0
    # Targets are supervision, never forward inputs.
    batch.clean_clustering_histogram_target = torch.flip(batch.clean_clustering_histogram_target, dims=[1])
    batch.clean_spectrum_target = torch.zeros_like(batch.clean_spectrum_target)
    torch.testing.assert_close(m(batch)['clean_clustering_histogram'], hist)
    # Adjacency arrangement is not encoded when graph context is off.
    perm = torch.arange(batch.adjacency.shape[1]-1, -1, -1)
    batch.adjacency = batch.adjacency[:, perm, :][:, :, perm]
    torch.testing.assert_close(m(batch)['clean_clustering_histogram'], hist)


def test_histogram_loss_uses_cdf_and_optional_cross_entropy():
    b = collate_spectral_examples([example(nx.path_graph(5), 4)])
    m = model(bins=4)
    def result(p):
        probs = torch.tensor([p], dtype=torch.float32, requires_grad=True)
        out = {'clean_spectrum': b.clean_spectrum_target, 'clean_clustering_histogram': probs,
               'clean_clustering_histogram_logits': probs.clamp_min(1e-8).log()}
        return m._spectral_loss_from_outputs(b, out, loss_weights={'spectrum': 0, 'moment2': 0, 'clustering_histogram': 1})[0]
    assert result([0, 1, 0, 0]).item() == pytest.approx(0.25)
    assert result([0, 0, 0, 1]).item() == pytest.approx(0.75)


def test_checkpoint_roundtrip_and_old_model_compatibility(tmp_path):
    m = model().eval()
    path = tmp_path/'hist.pt'
    save_topology_spectral_checkpoint(m, path)
    loaded, _, ckpt = load_topology_spectral_checkpoint(path, device='cpu')
    assert loaded.predict_clustering_histogram and loaded.clustering_histogram_bins == 10
    assert ckpt['model_config']['clustering_histogram_bins'] == 10
    batch = collate_spectral_examples([example(graph())])
    torch.testing.assert_close(m(batch)['clean_clustering_histogram'], loaded(batch)['clean_clustering_histogram'])
    # Simulate an actual legacy checkpoint with no new model_config keys.
    old = model(histogram=False)
    save_topology_spectral_checkpoint(old, path)
    state = torch.load(path, map_location='cpu', weights_only=False)
    state['model_config'].pop('predict_clustering_histogram')
    state['model_config'].pop('clustering_histogram_bins')
    torch.save(state, path)
    loaded, _, _ = load_topology_spectral_checkpoint(path, device='cpu')
    assert loaded.clustering_histogram_head is None
    assert 'clean_clustering_histogram' not in loaded(batch)


@pytest.mark.parametrize('mode', ['clustering', 'spectral_clustering'])
def test_histogram_guides_actual_degree_preserving_swaps(mode):
    g = graph()
    candidates, candidate_graphs, _ = propose_valid_topology_swaps(
        g, proposal_budget=-1, valid_candidate_budget=-1,
        preserve_connectivity=True, rng=np.random.default_rng(0),
    )
    h0 = extract_clustering_histogram(g, 10)
    target = next(c for c in candidate_graphs.values() if clustering_histogram_wasserstein(h0, extract_clustering_histogram(c, 10)) > 0.0)
    target_hist = extract_clustering_histogram(target, 10)
    target_spec = laplacian_eigenvalues(target)
    cfg = SpectralRefinerConfig.from_dict({
        'steps': 1, 'guidance_mode': mode, 'proposal_budget': -1, 'valid_candidate_budget': -1,
        'min_improvement': 1e-10,
        'spectral_guidance': {'weight': 1, 'min_clean_mix': 1, 'max_clean_mix': 1, 'expand_on_plateau': False},
        'clustering_guidance': {'statistic': 'histogram', 'histogram_bins': 10, 'weight': 1},
    })
    rows = score_spectral_candidates(
        g, candidates, clean_spectrum=target_spec, next_spectrum_target=target_spec,
        clean_clustering_histogram=target_hist, config=cfg, candidate_graphs=candidate_graphs,
    )
    for row in rows:
        expected = clustering_histogram_wasserstein(extract_clustering_histogram(row['candidate_graph'], 10), target_hist)
        assert row['candidate_clustering_discrepancy'] == pytest.approx(expected)
    def oracle(_model, current, **kwargs):
        return SpectralPrediction(target_spec, laplacian_eigenvalues(current), float(target_spec.sum()), float(target_spec@target_spec), clean_clustering_histogram=target_hist)
    final, trace = refine_graph_with_spectral_predictions(g, model=None, refiner_config=cfg, prediction_fn=oracle, return_trace=True)
    assert dict(g.degree()) == dict(final.degree())
    assert nx.is_connected(final)
    assert clustering_histogram_wasserstein(extract_clustering_histogram(final, 10), target_hist) < clustering_histogram_wasserstein(h0, target_hist)
    row = next(r for r in trace if r['accepted'])
    assert row['clustering_gain'] > 0 and row['clustering_statistic'] == 'histogram'
    assert len(row['target_clustering_histogram']) == 10
    with pytest.raises(ValueError, match='checkpoint'):
        score_spectral_candidates(g, candidates, clean_spectrum=target_spec, next_spectrum_target=target_spec, config=cfg, candidate_graphs=candidate_graphs)


def test_predict_clean_spectrum_returns_histogram_without_target():
    p = predict_clean_spectrum(model(), graph(), time=0.0, device='cpu')
    assert p.clean_clustering_histogram.shape == (10,)
    assert p.clean_clustering_histogram.sum() == pytest.approx(1.0)
    assert p.clean_clustering_coefficient is None


def test_cross_entropy_only_histogram_training_has_gradients():
    m = model()
    batch = collate_spectral_examples([example(graph())])
    loss, metrics = m.loss(batch, loss_weights={
        'spectrum': 0, 'moment2': 0, 'low_frequency': 0,
        'clustering_histogram': 0, 'clustering_histogram_ce': 1,
    })
    assert loss.item() == pytest.approx(metrics['clustering_histogram_ce'])
    loss.backward()
    assert m.clustering_histogram_head[-1].weight.grad.abs().sum() > 0


@pytest.mark.parametrize('storage', ['eager', 'streaming'])
def test_train_generate_evaluate_histogram_smoke(tmp_path, monkeypatch, storage):
    from scripts import train_topology_grapher as train
    from scripts import run_topology_grapher as generate
    from scripts import diagnose_spectral_denoiser as diagnose
    from scripts import evaluate_graph_generation_report as evaluate
    repo = Path(__file__).resolve().parents[1]
    cfg = load_yaml(repo/'configs/experiments/grapher/community_small_topology_spectral_clustering_histogram.yaml')
    root = tmp_path/'datasets'
    save_dataset_splits('tiny', {s: [graph(), nx.cycle_graph(5)] for s in ('train', 'val', 'test')}, {}, root)
    cfg['dataset'] = {'name': 'tiny', 'root': str(root), 'build_if_missing': False}
    cfg['benchmark'] = 'tiny'
    cfg['summary_diffusion'].update(storage=storage, samples_per_graph=2, paths_per_graph=1)
    cfg['topology_predictor'].update(hidden_dim=8, edge_dim=8, graph_dim=8, spectral_dim=16, spectral_layers=1, spectral_ff_dim=32, epochs=1)
    cfg['topology_refiner'].update(steps=2, proposal_budget=64, valid_candidate_budget=16)
    cfg['evaluation']['compute_orbit'] = False  # ORCA is not required for a tiny CLI smoke test.
    config = tmp_path/'config.yaml'
    save_yaml(cfg, config)
    training = tmp_path/'train'
    monkeypatch.setattr(sys, 'argv', ['train', '--config', str(config), '--output-dir', str(training), '--device', 'cpu'])
    train.main()
    report = json.loads((training/'training_report.json').read_text())
    assert 'clustering_histogram' in report['active_losses']
    assert report['predictor_targets']['clustering_histogram_bins'] == 100
    assert np.isfinite(report['history'][0]['val_clustering_histogram_w1'])
    checkpoint = training/'checkpoint.pt'
    generated = tmp_path/'generated'
    monkeypatch.setattr(sys, 'argv', ['generate', '--config', str(config), '--checkpoint', str(checkpoint), '--output-dir', str(generated), '--num-generate', '3', '--device', 'cpu'])
    generate.main()
    report = json.loads((generated/'report.json').read_text())
    assert report['diagnostics']['clustering_guidance_statistic'] == 'histogram'
    assert report['diagnostics']['rewiring_guidance_mode'] == 'spectral_clustering'
    assert report['diagnostics']['degree_preservation_rate'] == 1
    source = load_pickle(generated/'coarse_graphs.pkl')
    final = load_pickle(generated/'topology_refined_graphs.pkl')
    assert len(source) == len(final) == 3
    assert all(dict(a.degree()) == dict(b.degree()) for a,b in zip(source, final))
    for mode in ('spectral', 'clustering'):
        variant = tmp_path/mode
        monkeypatch.setattr(sys, 'argv', [
            'generate', '--config', str(config), '--checkpoint', str(checkpoint),
            '--output-dir', str(variant), '--num-generate', '3', '--device', 'cpu',
            '--set', f'topology_refiner.guidance_mode={mode}',
        ])
        generate.main()
        variant_sources = load_pickle(variant/'coarse_graphs.pkl')
        assert all(nx.utils.graphs_equal(a,b) for a,b in zip(source, variant_sources))
        variant_final = load_pickle(variant/'topology_refined_graphs.pkl')
        assert all(dict(a.degree()) == dict(b.degree()) for a,b in zip(source, variant_final))
    result = tmp_path/'denoiser.json'
    monkeypatch.setattr(sys, 'argv', ['diagnose', '--config', str(config), '--checkpoint', str(checkpoint), '--device', 'cpu', '--samples-per-graph', '2', '--paths-per-graph', '1', '--json-out', str(result), '--source-endpoint-only'])
    diagnose.main()
    metrics = json.loads(result.read_text())
    assert metrics['source_endpoint_only'] is True
    assert 'clustering_histogram_w1' in metrics['overall']
    assert list(metrics['by_time']) == ['[0.00,0.25)']
    monkeypatch.setattr(sys, 'argv', ['evaluate', '--config', str(config), '--generated-dir', str(generated), '--output-dir', str(tmp_path/'eval'), '--num-samples', '2', '--dpi', '40'])
    evaluate.main()
    assert (tmp_path/'eval'/'graph_mmd_metrics.csv').exists()

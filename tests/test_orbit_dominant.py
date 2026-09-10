from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from grapher.data.io import save_dataset_splits
from grapher.rewiring_mlp.generic.clustering import extract_clustering_histogram
from grapher.rewiring_mlp.generic.orbit import extract_orbit_summary, orbit_summary_distance
from grapher.rewiring_mlp.generic.rewiring import propose_valid_topology_swaps
from grapher.rewiring_mlp.generic.spectral import laplacian_eigenvalues
from grapher.rewiring_mlp.generic import spectral_refiner as refiner
from grapher.utils.io import load_yaml, load_pickle, save_yaml, save_pickle
from scripts import evaluate_graph_generation_report as evaluate
from scripts.run_topology_grapher import _guidance_diagnostic_summary

REPO = Path(__file__).resolve().parents[1]
OLD_CFG = REPO / 'configs/experiments/grapher/community_small_topology_spectral_clustering_histogram_orbit.yaml'
NEW_CFG = REPO / 'configs/experiments/grapher/community_small_topology_spectral_clustering_histogram_orbit_dominant.yaml'


def source_graph():
    return nx.Graph([(0,1), (0,2), (0,3), (1,2), (2,4), (3,5), (4,5)])


def case(mode='clustering_orbit'):
    graph = source_graph()
    actions, graphs, _ = propose_valid_topology_swaps(
        graph, proposal_budget=-1, valid_candidate_budget=-1,
        preserve_connectivity=True, rng=np.random.default_rng(17),
    )
    original = extract_orbit_summary(graph)
    target = next(g for g in graphs.values() if orbit_summary_distance(original, extract_orbit_summary(g)) > 1e-8)
    cfg = refiner.SpectralRefinerConfig.from_dict(load_yaml(NEW_CFG)['topology_refiner'])
    cfg = replace(cfg, steps=3, guidance_mode=mode, proposal_budget=-1, valid_candidate_budget=-1)
    spectrum = laplacian_eigenvalues(target)
    kwargs = dict(
        clean_spectrum=spectrum, next_spectrum_target=spectrum,
        clean_clustering_histogram=extract_clustering_histogram(target, 100),
        clean_orbit_summary=extract_orbit_summary(target), candidate_graphs=graphs,
    )
    return graph, actions, cfg, spectrum, kwargs


def test_new_configuration_preserves_model_training_and_old_ablation():
    old, new = load_yaml(OLD_CFG), load_yaml(NEW_CFG)
    model_old, model_new = dict(old['topology_predictor']), dict(new['topology_predictor'])
    model_old.pop('checkpoint_path'); model_new.pop('checkpoint_path')
    assert model_old == model_new
    for field in ('summary_diffusion', 'structure_summary_prediction', 'generation', 'constructor', 'degree_generator', 'source_enrichment'):
        assert old[field] == new[field]
    assert old['topology_refiner']['clustering_guidance']['weight'] == 1.0
    assert new['topology_refiner']['clustering_guidance']['weight'] == 0.25
    assert new['topology_refiner']['orbit_guidance']['weight'] == 1.0
    assert new['topology_refiner']['spectral_guidance']['weight'] == 0.0
    assert new['evaluation']['reference_split'] == 'val'


@pytest.mark.parametrize('mode', ['clustering', 'orbit', 'clustering_orbit'])
def test_lazy_spectral_measurements_do_not_change_summary_scores(mode, monkeypatch):
    graph, actions, cfg, _, kwargs = case(mode)
    eager = refiner.score_spectral_candidates(graph, actions, config=replace(cfg, compute_candidate_spectral_diagnostics=True), **kwargs)
    def forbidden(*args, **kwargs):
        raise AssertionError('Candidate eigensolver must not run in lazy summary scoring')
    monkeypatch.setattr(refiner, 'batched_laplacian_eigenvalues', forbidden)
    lazy = refiner.score_spectral_candidates(graph, actions, config=cfg, **kwargs)
    assert [r['action'] for r in eager] == [r['action'] for r in lazy]
    for before, after in zip(eager, lazy):
        assert before['energy_improvement'] == after['energy_improvement']
        assert before['objective_residual'] == after['objective_residual']
        assert before['relative_energy_improvement'] == after['relative_energy_improvement']
        assert after['candidate_spectrum'] is None
        assert after['candidate_spectral_discrepancy'] is None
    if mode == 'clustering_orbit':
        for row in lazy:
            assert row['energy_improvement'] == pytest.approx(
                .25 * row['clustering_relative_improvement'] + row['orbit_relative_improvement'])


@pytest.mark.parametrize('mode', ['clustering', 'orbit', 'clustering_orbit'])
def test_lazy_mode_same_actions_and_preserves_constraints_and_trace(mode):
    graph, _, cfg, spectrum, kwargs = case(mode)
    def oracle(_model, current, **extra):
        return refiner.SpectralPrediction(
            spectrum, laplacian_eigenvalues(current), float(spectrum.sum()),
            float(spectrum @ spectrum),
            clean_clustering_histogram=kwargs['clean_clustering_histogram'],
            clean_orbit_summary=kwargs['clean_orbit_summary'],
        )
    results=[]
    for eager in (True,False):
        final, trace = refiner.refine_graph_with_spectral_predictions(
            graph, model=None, refiner_config=replace(cfg, compute_candidate_spectral_diagnostics=eager),
            prediction_fn=oracle, rng=np.random.default_rng(27), return_trace=True,
        )
        assert dict(final.degree()) == dict(graph.degree())
        assert nx.is_connected(final)
        accepted=[r for r in trace if r['accepted']]
        assert accepted
        assert all(np.isfinite(r['projection_residual']) for r in accepted)
        assert all(r['energy_improvement'] > cfg.min_improvement for r in accepted)
        assert all('scoring_components' in r for r in accepted)
        assert all(r['candidate_spectral_diagnostics_eager'] == eager for r in accepted)
        results.append((final, trace, accepted))
    assert set(results[0][0].edges()) == set(results[1][0].edges())
    assert [r['action'] for r in results[0][2]] == [r['action'] for r in results[1][2]]
    np.testing.assert_allclose([r['spectral_gain'] for r in results[0][2]], [r['spectral_gain'] for r in results[1][2]], atol=1e-12)
    stops=[r for r in results[1][1] if r.get('terminal_stop')]
    assert stops
    assert stops[-1]['reason'] == 'explicit_stop_below_guidance_improvement_threshold'


@pytest.mark.parametrize('mode,debug', [('spectral',False), ('clustering_orbit',True)])
def test_required_or_debug_spectra_are_not_skipped(mode, debug, monkeypatch):
    graph, actions, cfg, _, kwargs = case(mode)
    cfg=replace(cfg, spectral_weight=1.0 if mode=='spectral' else 0.0, debug_enabled=debug)
    original=refiner.batched_laplacian_eigenvalues
    calls=[]
    def count(graphs, **options):
        calls.append(len(graphs))
        return original(graphs, **options)
    monkeypatch.setattr(refiner, 'batched_laplacian_eigenvalues',count)
    rows=refiner.score_spectral_candidates(graph, actions, config=cfg, **kwargs)
    assert calls == [len(actions)]
    assert all(r['candidate_spectrum'] is not None for r in rows)


def test_inactive_diagnostic_is_null_not_zero():
    _, _, cfg, _, _ = case('orbit')
    rows=[dict(current_orbit_discrepancy=.3,candidate_orbit_discrepancy=.2,orbit_gain=.1,
               current_clustering_discrepancy=None,clustering_gain=0.0)]
    result=_guidance_diagnostic_summary(cfg, rows)
    assert result['mean_accepted_clustering_gain'] is None
    assert result['clustering_diagnostics_computed'] is False
    assert result['mean_accepted_orbit_gain'] == .1
    assert result['accepted_orbit_improved_fraction'] == 1.0
    assert result['spectral_guidance_weight'] == 0.0
    assert _guidance_diagnostic_summary(cfg, [])['mean_accepted_orbit_gain'] is None


def test_reference_split_resolution_and_overwrite_guard(tmp_path):
    assert evaluate.resolve_reference_split({}) == 'test'
    assert evaluate.resolve_reference_split({'protocol':{'tune_on_split':'val'}}) == 'test'
    cfg={'evaluation':{'reference_split':'val'}}
    assert evaluate.resolve_reference_split(cfg) == 'val'
    assert evaluate.resolve_reference_split(cfg, 'test') == 'test'
    with pytest.raises(ValueError, match='reference_split'):
        evaluate.resolve_reference_split({'evaluation':{'reference_split':'train'}})
    p=tmp_path / 'graph_evaluation_report.json'
    p.write_text(json.dumps({'reference_split':'val'}))
    evaluate.validate_report_reference_split(tmp_path,'val')
    with pytest.raises(ValueError, match='different reference split'):
        evaluate.validate_report_reference_split(tmp_path,'test')
    p.write_text('{}') # legacy report had test hard-coded
    evaluate.validate_report_reference_split(tmp_path,'test')
    with pytest.raises(ValueError):
        evaluate.validate_report_reference_split(tmp_path,'val')


@pytest.mark.parametrize('split', ['val', 'test'])
def test_evaluator_really_uses_requested_split(tmp_path, monkeypatch, capsys, split):
    root=tmp_path/'datasets'
    graph=source_graph()
    splits={'train':[nx.path_graph(6)],'val':[graph], 'test':[nx.complete_graph(6),nx.cycle_graph(6)]}
    save_dataset_splits('tiny',splits,{},root)
    generated=tmp_path/'generated';generated.mkdir()
    save_pickle([graph],generated/'topology_refined_graphs.pkl')
    save_pickle([graph],generated/'coarse_graphs.pkl')
    cfg={'dataset':{'name':'tiny','root':str(root)},'evaluation':{'reference_split':'val','compute_orbit':True}}
    path=tmp_path/'config.yaml'; save_yaml(cfg,path)
    out=tmp_path/f'eval_{split}'
    # Skip figure rendering here (the end-to-end smoke test exercises it).
    monkeypatch.setattr(evaluate,'plot_generated_graphs',lambda *a, **kw:None)
    monkeypatch.setattr(sys,'argv',['evaluate','--config',str(path),'--generated-dir',str(generated),
                                  '--reference-split',split,'--output-dir',str(out)])
    evaluate.main()
    report=json.loads((out/'graph_evaluation_report.json').read_text())
    assert report['reference_split'] == split
    assert report['num_reference_graphs'] == len(splits[split])
    assert report['reference_split_sha256'] == report['dataset_provenance']['split_sha256'][split]
    assert all(r['comparison'].endswith('_to_'+split) for r in report['metrics'])
    final=report['metrics'][-1]
    expected=evaluate._paper_mmd(splits[split],[graph],compute_orbit=True,metric_protocol='graphrnn',clustering_bins=100)
    for k in expected: assert final[k] == pytest.approx(expected[k])
    assert f'held-out {split} graphs' in capsys.readouterr().out


def test_new_combination_reuses_legacy_joint_checkpoint_end_to_end(tmp_path,monkeypatch):
    from scripts import train_topology_grapher as train
    from scripts import run_topology_grapher as generate
    cfg=load_yaml(OLD_CFG)
    root=tmp_path/'datasets';g=source_graph()
    save_dataset_splits('tiny',{s:[g,nx.cycle_graph(6)] for s in ('train','val','test')},{},root)
    cfg['dataset']={'name':'tiny','root':str(root),'build_if_missing':False}
    cfg['benchmark']='tiny'
    cfg['summary_diffusion'].update(samples_per_graph=2,paths_per_graph=1)
    cfg['topology_predictor'].update(hidden_dim=8,edge_dim=8,graph_dim=8,spectral_dim=16,
        spectral_layers=1,spectral_heads=4,spectral_ff_dim=32,epochs=1,batch_size=2)
    path=tmp_path/'old_cfg.yaml';save_yaml(cfg,path)
    trained=tmp_path/'train'
    monkeypatch.setattr(sys,'argv',['train','--config',str(path),'--output-dir',str(trained),'--device','cpu'])
    train.main()
    # Only generation settings change. Load the previously trained joint head as-is.
    cfg['topology_refiner']=load_yaml(NEW_CFG)['topology_refiner']
    cfg['topology_refiner'].update(steps=2,proposal_budget=48,valid_candidate_budget=12)
    cfg['evaluation']['reference_split']='val'
    path=tmp_path/'new_cfg.yaml';save_yaml(cfg,path)
    out=tmp_path/'generated'
    monkeypatch.setattr(sys,'argv',['generate','--config',str(path),'--checkpoint',str(trained/'checkpoint.pt'),
                                  '--output-dir',str(out),'--num-generate','2','--device','cpu'])
    generate.main()
    report=json.loads((out/'report.json').read_text())
    d=report['diagnostics']
    assert d['guidance_mode'] == report['guidance_mode'] == 'clustering_orbit'
    assert d['predictor_family'] == 'spectral'
    assert d['clustering_guidance_weight'] == .25
    assert d['orbit_guidance_weight'] == 1.
    assert d['predictor_error_scope'] == 'checkpoint_validation_report'
    assert d['degree_preservation_rate'] == 1.0
    assert d['connectedness_rate'] == 1.0
    a,b=load_pickle(out/'coarse_graphs.pkl'),load_pickle(out/'topology_refined_graphs.pkl')
    assert all(dict(x.degree())==dict(y.degree()) for x,y in zip(a,b))
    monkeypatch.setattr(sys,'argv',['evaluate','--config',str(path),'--generated-dir',str(out),
        '--output-dir',str(tmp_path/'eval_val'),'--num-samples','2','--dpi','40'])
    evaluate.main()
    eval_report=json.loads((tmp_path/'eval_val/graph_evaluation_report.json').read_text())
    assert eval_report['reference_split']=='val'
    assert eval_report['metrics'][-1]['comparison']=='topology_final_to_val'

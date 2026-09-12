from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import networkx as nx
import numpy as np
import pytest
import torch

from grapher.data.io import save_dataset_splits
from grapher.models.dhvae_hh.degree_perturbation import METHODS, DegreePerturbationError
from grapher.rewiring_mlp.generic.joint_degree_training import build_joint_model, graph_fingerprint
from grapher.rewiring_mlp.generic.spectral_model import save_topology_spectral_checkpoint
from grapher.utils.io import load_yaml, save_yaml, load_pickle

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def fixture_run(tmp_path):
    old = torch.get_num_threads()
    torch_rng_state = torch.random.get_rng_state()
    torch.set_num_threads(1)
    cfg = load_yaml(ROOT / 'configs/experiments/grapher/community_small_degree_perturb_empirical.yaml')
    parents = [[4,4,2,2,2,2], [3,3,3,3,2,2], [3,3,3,2,2,1], [4,2,2,2,2,2]]
    graphs = [nx.havel_hakimi_graph(d) for d in parents]
    assert all(nx.is_connected(g) for g in graphs)
    splits = {'train':graphs, 'val':[nx.cycle_graph(6),nx.wheel_graph(6)], 'test':[nx.path_graph(6)]}
    root=tmp_path/'data'
    save_dataset_splits('perturb_fixture', splits, {}, root)
    cfg['dataset'].update(name='perturb_fixture', root=str(root), config_path=None)
    cfg['joint_degree'].update(initialize_degree_checkpoint=None, initialize_topology_checkpoint=None,
                               conditioning_dim=8, max_degree=5)
    cfg['joint_degree']['degree_model'].update(latent_dim=4, hidden_dim=8, size_condition_dim=4,
        edge_condition_dim=4, prior_hidden_dim=8, num_layers=1)
    cfg['topology_predictor'].update(hidden_dim=8, edge_dim=8, graph_dim=8, spectral_dim=16,
        spectral_layers=1, spectral_heads=4, spectral_ff_dim=32, dropout=0.)
    cfg['topology_refiner'].update(steps=2, proposal_budget=16, valid_candidate_budget=4)
    cfg['generation'].update(num_generate=4, degree_rng_mode='independent')
    cfg['evaluation'].update(compute_graphlet_history=False, compute_orbit=True)
    # A RANDOM neural checkpoint is sufficient to test CLI integration, not quality.
    torch.manual_seed(55)
    model,_ = build_joint_model(cfg, graphs)
    ckpt=tmp_path/'fixture.pt'
    report={'epoch':0, 'val_spectral_normalized_rmse':.1, 'val_clustering_histogram_w1':.1,
            'val_orbit_summary_log_rmse':.1,
            'dataset_graph_fingerprints':{k:graph_fingerprint(v) for k,v in splits.items()}}
    save_topology_spectral_checkpoint(model, ckpt, config=cfg, report=report)
    cfg['topology_predictor']['checkpoint_path']=str(ckpt)
    yield cfg, ckpt
    torch.set_num_threads(old)
    torch.random.set_rng_state(torch_rng_state)


def test_all_modes_cli_generate_evaluate_and_parent_pairing(tmp_path, monkeypatch, fixture_run):
    from scripts import run_topology_grapher as generate
    from scripts import evaluate_graph_generation_report as evaluate
    cfg,ckpt = fixture_run
    parents, sources = [], []
    for method in ('empirical', *METHODS, 'zero_probability'):
        settings=deepcopy(cfg)
        if method!='empirical':
            settings['generation'].update(degree_source='train_empirical_perturbed', degree_perturbation={
                'method':'unit_transfer' if method=='zero_probability' else method,
                'probability':0.0 if method=='zero_probability' else 1.0,
                'failure_policy':'keep_original', 'steps':1})
        config=tmp_path/f'{method}.yaml';save_yaml(settings,config)
        out=tmp_path/"generation_seed_42"/method
        monkeypatch.setattr(sys,'argv',['generate','--config',str(config),'--checkpoint',str(ckpt),
            '--output-dir',str(out),'--num-generate','4','--seed','42','--device','cpu'])
        generate.main()
        report=json.loads((out/'report.json').read_text())
        prior=json.loads((out/'degree_prior_report.json').read_text())
        assert report['diagnostics']['degree_preservation_rate']==1
        assert report['diagnostics']['connectedness_rate']==1
        assert report['rng_streams']['degree_sampling_independent']
        parents.append(report['parent_degree_fingerprint'])
        sources.append(report['source_graph_fingerprint'])
        coarse=load_pickle(out/'coarse_graphs.pkl')
        final=load_pickle(out/'topology_refined_graphs.pkl')
        degrees=json.loads((out/'sampled_degree_sequences.json').read_text())
        for a,b,d in zip(coarse,final,degrees):
            assert sorted(dict(a.degree()).values(),reverse=True)==d
            assert sorted(dict(b.degree()).values(),reverse=True)==d
        assert prior['num_returned']==4
        if method not in ('empirical','zero_probability'):
            assert prior['all_connected_feasible'] and prior['all_preserve_n_m']
            assert prior['num_changed']>0
        monkeypatch.setattr(sys,'argv',['evaluate','--config',str(config),'--generated-dir',str(out),
            '--reference-split','val','--output-dir',str(out/'evaluation_val'),
            '--num-samples','1','--dpi','50'])
        evaluate.main()
        ev=json.loads((out/'evaluation_val/graph_evaluation_report.json').read_text())
        rows={r['comparison']:r for r in ev['metrics']}
        assert rows['hh_source_to_val']['degree_mmd']==pytest.approx(rows['topology_final_to_val']['degree_mmd'],abs=1e-12)
        assert ev['num_graphs_evaluated']==4
        assert (out/'evaluation_val/graph_mmd_metrics.csv').is_file()
    assert len(set(parents))==1
    # Disabling perturbation must reproduce the NEW independent empirical source
    # and final graphs exactly, not merely their degree distributions.
    assert sources[0]==sources[-1]
    a=load_pickle(tmp_path/'generation_seed_42/empirical/topology_refined_graphs.pkl')
    b=load_pickle(tmp_path/'generation_seed_42/zero_probability/topology_refined_graphs.pkl')
    assert graph_fingerprint(a)==graph_fingerprint(b)
    from scripts import summarize_degree_perturbation_evaluations as summary
    result=summary.summarize(tmp_path,seeds=(42,))
    assert len(result['rows'])==5
    assert result['matched_parent_degrees_per_seed']
    assert all(r['degree_mmd_sample_std'] is None for r in result['aggregates'])
    monkeypatch.setattr(sys,'argv',['summarize','--generation-root',str(tmp_path),'--seeds','42'])
    summary.main()
    assert (tmp_path/'summary_val/rows.csv').exists()
    # Parent mismatch must not silently become a paired prior comparison.
    path=tmp_path/'generation_seed_42/unit_transfer/report.json'
    report=json.loads(path.read_text());report['parent_degree_fingerprint']='mismatch'
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError,match='Parent degrees differ'):
        summary.summarize(tmp_path,seeds=(42,))


def test_prior_diagnostic_cli_no_checkpoint(tmp_path, monkeypatch, fixture_run):
    from scripts import diagnose_degree_perturbations as diagnose
    cfg,ckpt=fixture_run
    ckpt.unlink()
    path=tmp_path/'cfg.yaml';save_yaml(cfg,path)
    out=tmp_path/'coverage'
    monkeypatch.setattr(sys,'argv',['diagnose','--config',str(path),'--output-dir',str(out),
        '--num-samples','16','--probability','1','--seed','42'])
    diagnose.main()
    result=json.loads((out/'report.json').read_text())
    assert not result['checkpoint_loaded']
    assert result['parents_identical_across_methods']
    assert set(result['reports'])==set(METHODS)
    assert len({r['parent_degree_fingerprint'] for r in result['reports'].values()})==1


def test_sampler_dispatch_never_reads_heldout_or_external_checkpoint():
    from scripts import run_topology_grapher as generate
    s=generate._build_generation_degree_sampler('train_empirical_perturbed',
        {'type':'degree_histogram_vae','checkpoint_path':'DOES_NOT_EXIST'},
        train_graphs=[nx.cycle_graph(6)],reference_graphs=[nx.complete_graph(11)], seed=42,
        perturbation_cfg={'method':'interpolation','probability':1.,'failure_policy':'keep_original'})
    row=s.sample()
    assert row['degree_sequence']==[2]*6
    assert row['sampling_diagnostics']['failure_reason']=='no_distinct_training_partner_same_n_m'
    assert len(s.degree_sequences)==1


def test_new_configs_only_change_prior_and_declared_fixed_search_settings():
    base=load_yaml(ROOT/'configs/experiments/grapher/community_small_topology_joint_degree_multicheckpoint.yaml')
    for method in ('empirical',*METHODS):
        cfg=load_yaml(ROOT/f'configs/experiments/grapher/community_small_degree_perturb_{method}.yaml')
        assert cfg['dataset']==base['dataset']
        assert cfg['joint_degree']==base['joint_degree']
        assert cfg['topology_predictor']['loss_weights']==base['topology_predictor']['loss_weights']
        assert cfg['generation']['degree_rng_mode']=='independent'
        assert cfg['topology_refiner']['steps']==32
        assert cfg['topology_refiner']['proposal_budget']==1024
        assert cfg['topology_refiner']['valid_candidate_budget']==-1
        assert cfg['topology_refiner']['clustering_guidance']['weight']==.25
        assert not cfg['source_enrichment']['enabled']
        if method!='empirical':
            assert cfg['generation']['degree_perturbation']['method']==method
            assert cfg['generation']['degree_perturbation']['probability']==.25
            assert cfg['generation']['degree_perturbation']['failure_policy']=='keep_original'


def test_strict_failure_writes_audit_without_resampling_parent(tmp_path,monkeypatch,fixture_run):
    from scripts import run_topology_grapher as generate
    cfg,ckpt=fixture_run
    cfg['generation'].update(degree_source='train_empirical_perturbed', degree_perturbation={
        'method':'unit_transfer','probability':1.0,'max_distance':.1,'failure_policy':'error'})
    path=tmp_path/'strict.yaml';save_yaml(cfg,path)
    out=tmp_path/'failed'
    monkeypatch.setattr(sys,'argv',['generate','--config',str(path),'--output-dir',str(out),'--device','cpu'])
    with pytest.raises(DegreePerturbationError):
        generate.main()
    report=json.loads((out/'degree_prior_report.json').read_text())
    assert report['generation_aborted']
    assert report['num_samples']==1
    assert not (out/'topology_refined_graphs.pkl').exists()

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import random
import sys

import networkx as nx
import numpy as np
import pytest
import torch

from grapher.data.io import save_dataset_splits
from grapher.models.dhvae_hh.degree_vae import load_degree_vae_checkpoint
from grapher.rewiring_mlp.generic.joint_checkpointing import (
    JointCheckpointManager, ensure_fresh_joint_output, file_sha256,
    resolve_checkpoint_policy, state_dict_sha256, verify_checkpoint_registry,
)
from grapher.rewiring_mlp.generic.joint_degree_training import build_joint_model
from grapher.rewiring_mlp.generic.spectral_model import load_topology_spectral_checkpoint
from grapher.utils.io import load_yaml, save_yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / 'configs/experiments/grapher/community_small_topology_joint_degree_multicheckpoint.yaml'


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def tiny_config():
    cfg = load_yaml(CFG)
    cfg['joint_degree'].update(
        initialize_degree_checkpoint=None, initialize_topology_checkpoint=None,
        freeze_epochs=1, graphs_per_batch=2, conditioning_dim=8, learning_rate=1e-3,
    )
    cfg['joint_degree']['degree_model'].update(
        latent_dim=4, hidden_dim=8, size_condition_dim=4, edge_condition_dim=4,
        prior_hidden_dim=8, num_layers=1,
    )
    cfg['topology_predictor'].update(
        hidden_dim=8, edge_dim=8, graph_dim=8, spectral_dim=16,
        spectral_layers=1, spectral_heads=4, spectral_ff_dim=32,
        epochs=5, learning_rate=1e-3, progress_interval=5,
    )
    cfg['summary_diffusion'].update(samples_per_graph=1, paths_per_graph=1)
    cfg['topology_refiner'].update(steps=1, proposal_budget=16, valid_candidate_budget=4)
    cfg['generation'].update(num_generate=2)
    cfg['evaluation'].update(compute_graphlet_history=False)
    return cfg


def toy_graphs():
    return [nx.cycle_graph(5), nx.path_graph(5), nx.cycle_graph(6), nx.path_graph(6)]


def validation_report(epoch, joint, hist, orbit):
    metrics = dict(joint_loss=joint, loss=joint, structure_loss=joint / 2,
                   degree_loss=joint * 50, clustering_histogram_w1=hist,
                   orbit_summary_log_rmse=orbit, spectral_normalized_rmse=0.1)
    return {'epoch': epoch, **{'val_' + key: value for key, value in metrics.items()}}, metrics


def test_independent_metrics_warmup_ties_last_and_exact_pairs(tmp_path):
    cfg = tiny_config()
    model, _ = build_joint_model(cfg, toy_graphs())
    manager = JointCheckpointManager(tmp_path, config=cfg, histogram_enabled=True, orbit_enabled=True)
    # Warmup is numerically best but must not win any best selection.
    rows = [(1, .001, .001, .001), (2, .10, .08, .09),
            (3, .12, .04, .07), (4, .13, .05, .02), (5, .14, .04, .02)]
    snapshots = {}
    for epoch, j, h, o in rows:
        with torch.no_grad():
            next(model.degree_model.parameters()).add_(0.001)
        snapshots[epoch] = state_dict_sha256(model.degree_model.state_dict())
        report, val = validation_report(epoch, j, h, o)
        rng = torch.random.get_rng_state().clone()
        py_rng = random.getstate()
        np_rng = np.random.get_state()
        manager.update(model, epoch=epoch, eligible=epoch > 1, report=report, val_metrics=val)
        assert torch.equal(rng, torch.random.get_rng_state())
        assert py_rng == random.getstate()
        assert np.array_equal(np_rng[1], np.random.get_state()[1])
        if epoch == 1:
            assert set(manager.records) == {'last'}
            assert not manager.records['last']['eligible_for_best']
            assert not (tmp_path / 'checkpoint.pt').exists()
    manager.finish()
    registry = verify_checkpoint_registry(tmp_path)
    assert registry['training_complete']
    assert {k: r['epoch'] for k, r in registry['selections'].items()} == {
        'best_joint': 2, 'best_histogram': 3, 'best_orbit': 4, 'last': 5,
    }
    for kind, record in registry['selections'].items():
        joint_model, _, payload = load_topology_spectral_checkpoint(tmp_path / record['checkpoint'], device='cpu')
        degree, vectorizer, degree_payload = load_degree_vae_checkpoint(tmp_path / record['degree_checkpoint'], device='cpu')
        a = payload['report']['checkpoint_selection']
        b = degree_payload['metrics']['checkpoint_selection']
        assert a == b
        assert a['epoch'] == record['epoch']
        assert state_dict_sha256(joint_model.state_dict()) == record['model_state_sha256']
        assert state_dict_sha256(degree.state_dict()) == snapshots[record['epoch']]
        assert state_dict_sha256(degree.state_dict()) == state_dict_sha256(joint_model.degree_model.state_dict())
        assert vectorizer.__dict__ == joint_model.degree_vectorizer.__dict__
    assert file_sha256(tmp_path / 'checkpoint.pt') == registry['selections']['best_joint']['checkpoint_sha256']
    assert file_sha256(tmp_path / 'degree_checkpoint.pt') == registry['selections']['best_joint']['degree_checkpoint_sha256']


@pytest.mark.parametrize('metric', ['val_joint_loss', 'val_clustering_histogram_w1', 'val_orbit_summary_log_rmse'])
@pytest.mark.parametrize('bad', [None, float('nan'), float('inf')])
def test_bad_selection_metric_fails_before_writing(tmp_path, metric, bad):
    manager = JointCheckpointManager(tmp_path, config=tiny_config(), histogram_enabled=True, orbit_enabled=True)
    report, val = validation_report(1, .1, .2, .3)
    report[metric] = bad
    with pytest.raises(ValueError, match='Missing or nonfinite'):
        manager.update(None, epoch=1, eligible=True, report=report, val_metrics=val)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('hist,orbit', [(False, False), (True, False), (False, True)])
def test_disabled_head_is_not_a_zero_metric_selection(tmp_path, hist, orbit):
    cfg = tiny_config()
    model, _ = build_joint_model(cfg, toy_graphs())
    manager = JointCheckpointManager(tmp_path, config=cfg, histogram_enabled=hist, orbit_enabled=orbit)
    report, val = validation_report(1, .1, .2, .3)
    if not hist:
        report.pop('val_clustering_histogram_w1')
    if not orbit:
        report.pop('val_orbit_summary_log_rmse')
    manager.update(model, epoch=1, eligible=True, report=report, val_metrics=val)
    manager.finish()
    result = verify_checkpoint_registry(tmp_path)
    assert ('best_histogram' in result['selections']) == hist
    assert ('best_orbit' in result['selections']) == orbit
    assert ('best_histogram' in result['unavailable']) == (not hist)
    assert ('best_orbit' in result['unavailable']) == (not orbit)


@pytest.mark.parametrize('artifact', ['history.json', 'report.json', 'checkpoint.pt', 'degree_checkpoint.pt', 'checkpoint_registry.json', 'checkpoints'])
def test_refuse_old_or_partial_output(tmp_path, artifact):
    (tmp_path / artifact).touch()
    with pytest.raises(FileExistsError, match='new output directory'):
        ensure_fresh_joint_output(tmp_path)


@pytest.mark.parametrize('policy', [{'enabled':'false'}, {'save_last':1}, {'save_best':'test_mmd'}])
def test_invalid_policy_rejected(policy):
    with pytest.raises(ValueError):
        resolve_checkpoint_policy({'joint_degree': {'checkpointing': policy}})


def test_disabled_multisave_preserves_root_best_joint(tmp_path):
    cfg = tiny_config()
    cfg['joint_degree']['checkpointing']['enabled'] = False
    model, _ = build_joint_model(cfg, toy_graphs())
    manager = JointCheckpointManager(tmp_path, config=cfg, histogram_enabled=True, orbit_enabled=True)
    for epoch, j in enumerate([.3, .1, .2], start=1):
        report, val = validation_report(epoch, j, .1, .2)
        manager.update(model, epoch=epoch, eligible=True, report=report, val_metrics=val)
    manager.finish()
    registry = verify_checkpoint_registry(tmp_path)
    assert set(registry['selections']) == {'best_joint'}
    assert registry['selections']['best_joint']['epoch'] == 2
    assert registry['selections']['best_joint']['checkpoint'] == 'checkpoint.pt'
    assert not (tmp_path / 'checkpoints').exists()


@pytest.mark.parametrize('tamper', ['export', 'marker', 'alias'])
def test_integrity_verifier_detects_wrong_export_or_marker(tmp_path, tamper):
    cfg = tiny_config()
    model, _ = build_joint_model(cfg, toy_graphs())
    manager = JointCheckpointManager(tmp_path, config=cfg, histogram_enabled=True, orbit_enabled=True)
    report, val = validation_report(1, .1, .2, .3)
    manager.update(model, epoch=1, eligible=True, report=report, val_metrics=val)
    manager.finish()
    if tamper == 'export':
        path = tmp_path / 'checkpoints/best_histogram/degree_checkpoint.pt'
        path.write_bytes(b'wrong epoch export')
    elif tamper == 'marker':
        path = tmp_path / 'checkpoints/best_orbit/selection.json'
        data = json.loads(path.read_text()); data['epoch'] = 2
        path.write_text(json.dumps(data))
    else:
        (tmp_path / 'checkpoint.pt').write_bytes(b'stale root alias')
    with pytest.raises(ValueError):
        verify_checkpoint_registry(tmp_path)


def test_export_serialization_failure_does_not_publish_partial_pair(tmp_path, monkeypatch):
    import grapher.rewiring_mlp.generic.joint_checkpointing as saves
    cfg = tiny_config()
    model, _ = build_joint_model(cfg, toy_graphs())
    manager = JointCheckpointManager(tmp_path, config=cfg, histogram_enabled=True, orbit_enabled=True)
    report, val = validation_report(1, .3, .3, .3)
    manager.update(model, epoch=1, eligible=True, report=report, val_metrics=val)
    before = verify_checkpoint_registry(tmp_path)
    def fail(*args, **kwargs):
        raise OSError('simulated failed degree serialization')
    monkeypatch.setattr(saves, 'save_degree_vae_checkpoint', fail)
    report, val = validation_report(2, .2, .2, .2)
    with pytest.raises(OSError):
        manager.update(model, epoch=2, eligible=True, report=report, val_metrics=val)
    assert verify_checkpoint_registry(tmp_path) == before
    assert not list(tmp_path.rglob('*.tmp'))


def test_hash_supports_empty_scalar_and_bfloat_tensors():
    state = {'a': torch.tensor(2), 'b': torch.empty(0), 'c': torch.ones(3, dtype=torch.bfloat16)}
    assert state_dict_sha256(state) == state_dict_sha256(deepcopy(state))
    state['a'] += 1
    assert state_dict_sha256(state) != state_dict_sha256({'a': torch.tensor(2), 'b': state['b'], 'c': state['c']})


def test_actual_trainer_multi_saving_does_not_change_training_or_best_selection(tmp_path, monkeypatch):
    from scripts import train_topology_grapher as train
    cfg = tiny_config()
    cfg['topology_predictor']['epochs'] = 3
    root = tmp_path / 'data'
    save_dataset_splits('joint_fixture', {'train': toy_graphs(), 'val': [nx.wheel_graph(6)], 'test': [nx.path_graph(5)]}, {}, root)
    cfg['dataset'].update(name='joint_fixture', root=str(root), config_path=None)
    runs = []
    for enabled in (False, True):
        out = tmp_path / str(enabled)
        cfg['joint_degree']['checkpointing']['enabled'] = enabled
        yaml_path = tmp_path / f'{enabled}.yaml'; save_yaml(cfg, yaml_path)
        monkeypatch.setattr(sys, 'argv', ['train', '--config', str(yaml_path), '--output-dir', str(out), '--seed', '42', '--device', 'cpu'])
        train.main()
        runs.append(out)
    assert json.loads((runs[0] / 'history.json').read_text()) == json.loads((runs[1] / 'history.json').read_text())
    a = verify_checkpoint_registry(runs[0])['selections']['best_joint']
    b = verify_checkpoint_registry(runs[1])['selections']['best_joint']
    assert a['epoch'] == b['epoch']
    assert a['model_state_sha256'] == b['model_state_sha256']


def test_end_to_end_each_selection_loads_generates_and_evaluates(tmp_path, monkeypatch, capsys):
    from scripts import train_topology_grapher as train
    from scripts import run_topology_grapher as generate
    from scripts import evaluate_graph_generation_report as evaluate
    from scripts import inspect_joint_checkpoints as inspect
    import grapher.rewiring_mlp.generic.joint_degree_training as jt
    cfg = tiny_config()
    root = tmp_path / 'data'
    save_dataset_splits('joint_fixture', {'train': toy_graphs(), 'val': [nx.wheel_graph(6)], 'test': [nx.path_graph(5)]}, {}, root)
    cfg['dataset'].update(name='joint_fixture', root=str(root), config_path=None)
    yaml_path = tmp_path / 'cfg.yaml'; save_yaml(cfg, yaml_path)
    out = tmp_path / 'train'
    actual_run = jt.run_joint_epoch
    counter = 0
    def controlled_validation(*args, **kwargs):
        nonlocal counter
        result = actual_run(*args, **kwargs)
        if kwargs.get('optimizer') is None:
            counter += 1
            j, h, o = [(0.01, .01, .01), (.1,.08,.09), (.12,.04,.07), (.13,.05,.02), (.14,.06,.03)][counter-1]
            result.update(joint_loss=j, loss=j, structure_loss=j/2, degree_loss=j*50,
                          clustering_histogram_w1=h, orbit_summary_log_rmse=o)
        return result
    monkeypatch.setattr(jt, 'run_joint_epoch', controlled_validation)
    monkeypatch.setattr(sys, 'argv', ['train','--config',str(yaml_path),'--output-dir',str(out),'--seed','42','--device','cpu'])
    train.main()
    registry = verify_checkpoint_registry(out)
    assert {k:r['epoch'] for k,r in registry['selections'].items()} == {'best_joint':2,'best_histogram':3,'best_orbit':4,'last':5}
    monkeypatch.setattr(sys,'argv',['inspect','--training-dir',str(out),'--verify'])
    inspect.main()
    assert 'File integrity and paired selection markers: PASS' in capsys.readouterr().out
    # Generation never needs the standalone export, but it records the selected model.
    for kind, record in registry['selections'].items():
        (out / record['degree_checkpoint']).rename(out / f'unused_{kind}_export.pt')
        gdir = tmp_path / f'learned_{kind}'
        path = out / record['checkpoint']
        monkeypatch.setattr(sys,'argv',['generate','--config',str(yaml_path),'--checkpoint',str(path),'--output-dir',str(gdir),'--num-generate','2','--seed','42','--device','cpu'])
        generate.main()
        report = json.loads((gdir/'report.json').read_text())
        assert report['checkpoint_selection']['kind'] == kind
        assert report['checkpoint_epoch'] == record['epoch']
        assert report['checkpoint_sha256'] == record['checkpoint_sha256']
        assert report['degree_sampler_source'] == 'joint_checkpoint_embedded'
        assert report['diagnostics']['degree_preservation_rate'] == 1
        assert report['diagnostics']['connectedness_rate'] == 1
        assert len(report['source_graph_fingerprint']) == 64
        monkeypatch.setattr(sys,'argv',['evaluate','--config',str(yaml_path),'--generated-dir',str(gdir),'--reference-split','val','--output-dir',str(gdir/'evaluation_val')])
        evaluate.main()
        assert (gdir/'evaluation_val/graph_mmd_metrics.csv').is_file()
    # Empirical conditions can be exactly paired across the independently selected checkpoints.
    fingerprints = []
    for kind in registry['selections']:
        gdir = tmp_path / f'empirical_{kind}'
        path = out / registry['selections'][kind]['checkpoint']
        monkeypatch.setattr(sys,'argv',['generate','--config',str(yaml_path),'--checkpoint',str(path),'--output-dir',str(gdir),'--num-generate','2','--seed','42','--device','cpu','--set','generation.degree_source=train_empirical'])
        generate.main()
        fingerprints.append(json.loads((gdir/'report.json').read_text())['source_graph_fingerprint'])
    assert len(set(fingerprints)) == 1


def test_new_config_changes_retention_names_not_losses_or_guidance():
    old = load_yaml(ROOT / 'configs/experiments/grapher/community_small_topology_joint_degree.yaml')
    new = load_yaml(CFG)
    assert new['topology_predictor']['loss_weights'] == old['topology_predictor']['loss_weights']
    assert new['joint_degree'] == old['joint_degree']
    assert new['topology_refiner'] == old['topology_refiner']
    assert 'multicheckpoint' in new['topology_predictor']['checkpoint_path']


def evaluation_fixture(root, kind, *, source='same'):
    folder = root / kind
    (folder / 'evaluation_val').mkdir(parents=True)
    gen = dict(checkpoint_selection={'kind':kind, 'epoch':2}, seed=42, degree_source='train_empirical',
               num_generated=2, source_graph_fingerprint=source, config={'topology_refiner': {'steps':32}})
    ev = dict(reference_split='val', reference_split_sha256='known', reference_graph_indices_zero_based=[0],
              num_reference_graphs=1, num_graphs_evaluated=2, generic_mmd_protocol='graphrnn',
              generic_clustering_bins=100, compute_orbit=True, generated_stage='topology_final',
              metrics=[dict(comparison='topology_final_to_val', degree_mmd=.1, clustering_mmd=.2, orbit_mmd=.3)])
    (folder/'report.json').write_text(json.dumps(gen))
    path=folder/'evaluation_val/graph_evaluation_report.json'
    path.write_text(json.dumps(ev))
    return path


def test_evaluation_comparison_reports_matched_sources(tmp_path):
    from scripts.summarize_joint_checkpoint_evaluations import summarize
    evaluation_fixture(tmp_path, 'best_joint'); evaluation_fixture(tmp_path, 'best_orbit')
    report = summarize(tmp_path, selections=['best_joint','best_orbit'], require_same_sources=True)
    assert report['identical_source_graphs']
    assert len(report['results']) == 2


def test_evaluation_comparison_different_priors_are_not_paired(tmp_path):
    from scripts.summarize_joint_checkpoint_evaluations import summarize
    evaluation_fixture(tmp_path, 'best_joint'); evaluation_fixture(tmp_path, 'best_orbit', source='different')
    assert not summarize(tmp_path, selections=['best_joint','best_orbit'])['identical_source_graphs']
    with pytest.raises(ValueError, match='not exactly matched'):
        summarize(tmp_path, selections=['best_joint','best_orbit'], require_same_sources=True)


@pytest.mark.parametrize('field,value', [('reference_split','test'), ('reference_split_sha256','different'),
                                       ('num_graphs_evaluated',1),('generic_clustering_bins',20)])
def test_evaluation_comparison_refuses_protocol_mix(tmp_path, field, value):
    from scripts.summarize_joint_checkpoint_evaluations import summarize
    evaluation_fixture(tmp_path, 'best_joint')
    path = evaluation_fixture(tmp_path, 'best_orbit')
    ev = json.loads(path.read_text()); ev[field] = value; path.write_text(json.dumps(ev))
    with pytest.raises(ValueError):
        summarize(tmp_path, selections=['best_joint','best_orbit'])

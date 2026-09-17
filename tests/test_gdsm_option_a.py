from __future__ import annotations

import copy
import json
import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import torch

from grapher.models.base import DatasetReference, RunSpec, TrainRequest, GenerateRequest
from grapher.models.gdsm_simple.spectral_decode import (
    decode_spectral_batch, isotonic_nondecreasing, project_rewiring_feedback,
)
from grapher.models.gdsm_simple.structure3 import validate_structure_options, reconcile_summary3
from grapher.models.gdsm_simple.structured_pipeline import make_condition
from grapher.models.gdsm_simple.wrapper import GDSMSimpleWrapper


def load(path):
    with Path(path).open('rb') as handle:
        return pickle.load(handle)


def options(conditioning='initialization_only'):
    return {'train': {'epochs': 2, 'batch_size': 4, 'validation_every': 1, 'log_every': 1},
            'model': {'max_nodes': 6, 'hidden_dim': 16, 'num_layers': 1, 'num_heads': 4, 'ff_dim': 32},
            'diffusion': {'steps': 30, 'beta_start': .01, 'beta_end': .3},
            'sample': {'steps': 8, 'threshold': .5}, 'generation_batch_size': 2,
            'runtime': {'device': 'cpu'},
            'extensions': {'degree_conditioning': True, 'degree_preserving_rewiring': True,
                'generation_mode': 'spectral_decode',
                'structural_summary': {'enabled': True, 'graphlet_size': 3,
                    'clustering_bins': 10, 'basis_pairings_per_graph': 2},
                'initialization': {'mode': 'degree_basis', 'conditioning': conditioning,
                                   'degree_generator': {'type': 'empirical'}},
                'spectral_decode': {'feedback_mode': 'fixed_basis_delta',
                                    'connectivity': 'unconstrained', 'save_degree_trajectory': True},
                'structure_guidance': {'every': 1, 'start_fraction': 1., 'max_steps_per_event': 2,
                    'proposal_budget': 32, 'valid_candidate_budget': 16,
                    'spectrum_feedback': .5, 'save_intermediate_graphs': True}}}


def test_decode_changes_degrees_with_same_basis():
    n = 5
    # Uniform dominant eigenvector: changing its coefficient changes all degrees.
    raw = np.column_stack([np.ones(n), np.eye(n)[:, 1:]])
    u, _ = np.linalg.qr(raw)
    u = np.roll(u, -1, axis=1)  # dominant uniform column last
    bases = torch.tensor(np.stack([u, u]), dtype=torch.float64)
    values = torch.zeros((2,n), dtype=torch.float64)
    values[1,-1] = n/np.sqrt(n)
    binary, _, _ = decode_spectral_batch(values, bases, torch.ones((2,n), dtype=torch.bool), torch.tensor([n,n]))
    assert binary[0].sum() == 0
    assert torch.all(binary[1].sum(-1) == n-1)
    assert torch.equal(binary, binary.transpose(1,2))
    assert not binary.diagonal(dim1=1,dim2=2).any()


def test_decode_padding_does_not_mix_zeros_into_valid_eigenvalues():
    path = nx.path_graph(3)
    eigenvalues, u = np.linalg.eigh(nx.to_numpy_array(path))
    x = torch.zeros((1,6), dtype=torch.float64)
    x[0,:3] = torch.tensor(eigenvalues/np.sqrt(3))
    basis = torch.zeros((1,6,6), dtype=torch.float64); basis[0,:3,:3] = torch.tensor(u)
    mask = torch.tensor([[True,True,True,False,False,False]])
    binary, values, order = decode_spectral_batch(x,basis,mask,torch.tensor([3]))
    np.testing.assert_array_equal(binary[0,:3,:3], nx.to_numpy_array(path, dtype=bool))
    assert not binary[0,3:].any()
    assert values[0,3:].sum() == 0
    assert order[0,:3].tolist() == [0,1,2]


def test_decode_uses_strict_threshold_and_retains_all_nodes():
    # Build a valid symmetric coefficient representation with entries exactly .5.
    u = torch.tensor([[[1.,1.],[-1.,1.]]], dtype=torch.float64)/np.sqrt(2)
    vals = torch.tensor([[-.5,.5]], dtype=torch.float64)/np.sqrt(2)
    binary,_,_ = decode_spectral_batch(vals,u,torch.ones((1,2),dtype=torch.bool),torch.tensor([2]),threshold=.5)
    assert binary.shape == (1,2,2)
    assert not binary.any()


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -.1, 1.1])
def test_decode_rejects_invalid_thresholds(bad):
    with pytest.raises(ValueError):
        decode_spectral_batch(torch.zeros((1,2)),torch.eye(2).unsqueeze(0),
                              torch.ones((1,2),dtype=torch.bool),torch.tensor([2]),bad)


def test_isotonic_projection_does_not_permute_basis_coordinates():
    np.testing.assert_allclose(isotonic_nondecreasing(np.array([0.,3.,1.,4.])), [0.,2.,2.,4.])
    assert np.isclose(isotonic_nondecreasing(np.array([3.,2.,1.])).sum(), 6.)
    np.testing.assert_array_equal(isotonic_nondecreasing(np.array([1.,2.,3.])), [1.,2.,3.])


def test_fixed_basis_delta_feedback_is_not_sorted_graph_eigenvalues():
    # A relabelled path is cospectral but its edge delta has nonzero projection
    # in this deliberately unrelated fixed basis.
    before = nx.path_graph(5)
    after = nx.relabel_nodes(before, {0:2,2:0}, copy=True)
    rng = np.random.default_rng(7)
    u, _ = np.linalg.qr(rng.normal(size=(5,5)))
    z = np.linspace(-10,10,5)
    updated, diag = project_rewiring_feedback(z,u,before,after,.3)
    delta = nx.to_numpy_array(after,nodelist=range(5))-nx.to_numpy_array(before,nodelist=range(5))
    expected = z + .3*np.diag(u.T@delta@u)/np.sqrt(5)
    np.testing.assert_allclose(updated,expected,atol=1e-12)
    assert diag['representable_feedback_nonzero']
    assert 0 <= diag['basis_projection_relative_residual'] <= 1.00001
    np.testing.assert_allclose(np.linalg.eigvalsh(nx.to_numpy_array(before)),
                               np.linalg.eigvalsh(nx.to_numpy_array(after)), atol=1e-12)


def test_no_swap_produces_no_spectral_feedback():
    g = nx.cycle_graph(5); u=np.eye(5); z=np.arange(5.)
    updated,diag=project_rewiring_feedback(z,u,g,g,1.)
    np.testing.assert_array_equal(updated,z)
    assert not diag['representable_feedback_nonzero']


def test_strict_initialization_only_hides_degree_and_anchor_blocks():
    basis=np.eye(5); d=np.array([2]*5); anchor=np.arange(5.)
    strict=make_condition(basis,d,anchor,6,{'conditioning':'initialization_only'})
    other=make_condition(basis,np.array([4]*5),anchor+100,6,{'conditioning':'initialization_only'})
    np.testing.assert_array_equal(strict,other)
    assert np.all(strict[:12] == 0)
    assert np.any(strict[12:] != 0)
    legacy=make_condition(basis,d,anchor,6,{})
    assert np.any(legacy[:12] != 0)


@pytest.fixture(scope='module')
def trained_option_a(tmp_path_factory):
    torch.set_num_threads(1)
    root=tmp_path_factory.mktemp('option_a')
    data=root/'data'/'toy'; data.mkdir(parents=True)
    train=[nx.cycle_graph(6),nx.complete_bipartite_graph(3,3),nx.wheel_graph(6),nx.path_graph(6)]
    for split,graphs in (('train',train),('val',[nx.star_graph(5),nx.complete_graph(6)]),('test',[nx.ladder_graph(3)])):
        with (data/(split+'.pkl')).open('wb') as h: pickle.dump(graphs,h)
    run=RunSpec('gdsm_simple','community_small','option_a',42,root/'runs')
    req=TrainRequest(run,DatasetReference('community_small',root/'data','toy'),options=options())
    w=GDSMSimpleWrapper(); art=w.train(req)
    return w,req,art,root


def test_option_a_roundtrip_invariants_and_actual_degree_changes(trained_option_a):
    w,req,art,root=trained_option_a
    gen=w.generate(GenerateRequest(req.run,art.checkpoint_path,6,45,generation_id='normal'))
    manifest=json.loads(gen.manifest_path.read_text())
    agg=manifest['structural_rewiring']
    assert manifest['generation_mode']=='spectral_decode'
    assert agg['global_degree_constraint'] is False
    assert agg['local_rewiring_degree_preservation_rate']==1.
    assert agg['mean_spectral_degree_change_steps'] > 0
    assert agg['prior_to_final_indexed_degree_preservation_rate'] < 1.
    assert manifest['decode']['degree_projection'] is False
    final=load(gen.graphs_path); pre=load(gen.generation_dir/'final_pre_rewire_graphs.pkl')
    assert len(final)==6
    for g,h in zip(final,pre):
        assert len(g)==6 and nx.number_of_selfloops(g)==0
        assert dict(g.degree()) == dict(h.degree())
    records=json.loads((gen.generation_dir/'rewiring_diagnostics.json').read_text())
    for row in records['per_graph']:
        assert row['num_spectral_decodes']==8
        for event in row['events']:
            assert event['degrees_before_rewiring']==event['degrees_after_rewiring']
            assert event['reconciled_against']=='current_decoded_degrees_not_sampled_prior'
            target=reconcile_summary3(event['raw_prediction'], np.asarray(event['degrees_before_rewiring']))
            np.testing.assert_allclose(target['orbit_log_mean'], event['reconciled_target']['orbit_log_mean'])
    snapshots=load(gen.generation_dir/'intermediate_graphs.pkl')
    for g,history in zip(final,snapshots):
        np.testing.assert_array_equal(nx.to_numpy_array(g),nx.to_numpy_array(history[-1]['graph']))
    trajectory=np.load(gen.generation_dir/'degree_trajectories.npz')
    assert trajectory['degrees'].shape==(6,8,6)
    for i,g in enumerate(final):
        assert trajectory['degrees'][i,-1].tolist()==[g.degree(v) for v in range(6)]
    repeated=w.generate(GenerateRequest(req.run,art.checkpoint_path,6,45,generation_id='repeat'))
    assert repeated.graphs_sha256==gen.graphs_sha256


def test_strict_checkpoint_cannot_silently_restore_legacy_degree_condition(trained_option_a):
    w,req,art,_=trained_option_a
    with pytest.raises(ValueError,match='initialization.conditioning'):
        w.generate(GenerateRequest(req.run,art.checkpoint_path,1,42,generation_id='wrong_condition',
            options={'extensions':{'initialization':{'conditioning':'degree_basis'}}}))


def test_option_a_never_calls_hh_and_no_rewire_is_plain_spectral_decode(trained_option_a,monkeypatch):
    import grapher.models.gdsm_simple.structured_pipeline as legacy
    import grapher.models.gdsm_simple.degree_initialization as init
    def fail(*a,**kw): raise AssertionError('HH must not run in Option A')
    monkeypatch.setattr(legacy,'realize_degrees',fail)
    monkeypatch.setattr(init,'construct_indexed_havel_hakimi',fail)
    w,req,art,_=trained_option_a
    gen=w.generate(GenerateRequest(req.run,art.checkpoint_path,4,43,generation_id='no_swaps',
        options={'extensions':{'degree_preserving_rewiring':False}}))
    manifest=json.loads(gen.manifest_path.read_text())
    assert manifest['structural_rewiring']['mean_accepted_steps']==0
    assert manifest['structural_rewiring']['mean_feedback_events']==0
    final=load(gen.graphs_path); threshold=load(gen.generation_dir/'threshold_graphs.pkl')
    for g,h in zip(final,threshold):
        np.testing.assert_array_equal(nx.to_numpy_array(g),nx.to_numpy_array(h))


def test_zero_feedback_makes_intermediate_swaps_irrelevant_to_later_decode(trained_option_a):
    w,req,art,_=trained_option_a
    paths=[]
    for gid,enabled in (('zero_feedback_swaps',True),('zero_feedback_no_swaps',False)):
        gen=w.generate(GenerateRequest(req.run,art.checkpoint_path,2,44,generation_id=gid,
            options={'extensions':{'degree_preserving_rewiring':enabled,'structure_guidance':{'spectrum_feedback':0.}}}))
        paths.append(load(gen.generation_dir/'target_adjacency_eigenvalues.pkl'))
    for a,b in zip(*paths):
        np.testing.assert_array_equal(a,b)


def test_feedback_changes_later_spectral_trajectory_when_swap_is_representable(trained_option_a,monkeypatch):
    import grapher.models.gdsm_simple.spectral_decode_pipeline as pipeline
    real_refine=pipeline.refine_structure3
    # Oracle search provides an admissible swap independent of a tiny model's
    # untrained summary quality. This test isolates the sampler feedback wiring.
    from grapher.rewiring_mlp.generic.rewiring import propose_valid_topology_swaps
    def controlled_refine(graph,target,summary,*,source,config,rng,visited):
        actions,candidates,_=propose_valid_topology_swaps(graph,proposal_budget=-1,valid_candidate_budget=-1,
                                                       preserve_connectivity=nx.is_connected(graph),rng=rng)
        candidate=candidates[actions[0]] if actions else graph.copy()
        return candidate, {'accepted_steps':int(bool(actions)), 'degree_preserved':True,
                           'all_accepted_steps_improve_energy':False,'all_accepted_steps_improve_structure':False}
    monkeypatch.setattr(pipeline,'refine_structure3',controlled_refine)
    w,req,art,_=trained_option_a
    paths=[]; aggregates=[]
    for gid,weight in (('controlled_feedback',.7),('controlled_no_feedback',0.)):
        gen=w.generate(GenerateRequest(req.run,art.checkpoint_path,6,45,generation_id=gid,
            options={'extensions':{'structure_guidance':{'spectrum_feedback':weight}}}))
        paths.append(load(gen.generation_dir/'target_adjacency_eigenvalues.pkl'))
        aggregates.append(json.loads(gen.manifest_path.read_text())['structural_rewiring'])
    assert aggregates[0]['mean_feedback_events'] > 0
    assert any(not np.allclose(a,b) for a,b in zip(*paths))


def test_legacy_structure3_checkpoint_can_use_free_degree_decoder(tmp_path):
    opt=options('degree_basis'); opt['extensions'].pop('generation_mode')
    data=tmp_path/'data'/'toy'; data.mkdir(parents=True)
    for split in ('train','val','test'):
        with (data/(split+'.pkl')).open('wb') as h: pickle.dump([nx.cycle_graph(6),nx.path_graph(6)],h)
    run=RunSpec('gdsm_simple','community_small','legacy',42,tmp_path/'runs')
    w=GDSMSimpleWrapper(); req=TrainRequest(run,DatasetReference('community_small',tmp_path/'data','toy'),options=opt)
    art=w.train(req)
    gen=w.generate(GenerateRequest(run,art.checkpoint_path,2,42,generation_id='free_degree_legacy',
        options={'extensions':{'generation_mode':'spectral_decode'}}))
    manifest=json.loads(gen.manifest_path.read_text())
    assert manifest['degree_prior']['explicit_denoiser_conditioning']=='degree_basis'
    assert manifest['structural_rewiring']['global_degree_constraint'] is False
    with pytest.raises(ValueError,match='initialization.conditioning'):
        w.generate(GenerateRequest(run,art.checkpoint_path,1,42,generation_id='strict_wrong',
            options={'extensions':{'initialization':{'conditioning':'initialization_only'},'generation_mode':'spectral_decode'}}))


@pytest.mark.parametrize('ext', [
    {'generation_mode':'typo'}, {'initialization':{'conditioning':'typo'}},
    {'spectral_decode':{'feedback_mode':'eigenvalues_only'}},
    {'spectral_decode':{'connectivity':'repair'}},
])
def test_invalid_option_a_config_is_rejected(ext):
    opt=options(); opt['extensions'].update(ext)
    with pytest.raises(ValueError): validate_structure_options(opt)


@pytest.mark.parametrize('dataset',['community_small','ego_small'])
@pytest.mark.parametrize('suffix',['','_legacy_conditioning'])
def test_option_a_configs_and_generation_cli(dataset,suffix):
    from grapher.models.external_cli import _generation_options,build_parser
    import yaml
    path=Path(__file__).resolve().parents[1]/'configs'/'baselines'/f'gdsm_simple_{dataset}_structure3_option_a{suffix}.yaml'
    raw=yaml.safe_load(path.read_text())['gdsm_simple']
    validate_structure_options(raw)
    args=build_parser('gdsm_simple').parse_args(['--dataset',dataset,'--stage','generate','--wrapper-config',str(path),'--device','cpu'])
    generated=_generation_options('gdsm_simple',args,wrapper_config=path)
    assert generated['extensions']['generation_mode']=='spectral_decode'
    expected='degree_basis' if suffix else 'initialization_only'
    assert generated['extensions']['initialization']['conditioning']==expected


def test_option_a_trained_dhvae_prior_integration(trained_option_a):
    from grapher.models.dhvae_hh.degree_vae import (
        DegreeVectorizer, build_degree_vae, degree_vae_loss, save_degree_vae_checkpoint,
    )
    w,req,art,root=trained_option_a
    graphs=load(req.dataset.split_paths['train'])
    torch.manual_seed(73)
    vectorizer=DegreeVectorizer.fit(graphs,require_connected=True)
    x,targets=vectorizer.to_training_arrays(graphs)
    x=torch.from_numpy(x); targets={k:torch.from_numpy(v) for k,v in targets.items()}
    model=build_degree_vae(vectorizer,latent_dim=4,hidden_dim=16,size_condition_dim=4,edge_condition_dim=4,
        use_edge_count_conditioning=True,prior_condition_on_edges=True,
        prior_type='conditional_gaussian',prior_components=1,num_layers=1)
    optimizer=torch.optim.Adam(model.parameters(),lr=.015)
    for _ in range(40):
        optimizer.zero_grad()
        output,mu,logvar=model(x,targets['num_nodes_count'],targets['num_edges_count'])
        loss,_=degree_vae_loss(output,targets,mu,logvar,weights={'num_edges':2.,'degree':5.})
        loss.backward(); optimizer.step()
    path=root/'option_a_degree.pt'; save_degree_vae_checkpoint(path,model,vectorizer)
    prior={'type':'dhvae','checkpoint_path':str(path),'sample_num_nodes':'empirical','sample_num_edges':'model',
           'fallback':'error','postprocess_policy':'reject_only','max_resample':100,'model_resample_attempts':8,
           'exact_degree_sum_conditioning':True,'parity_conditioned':False}
    gen=w.generate(GenerateRequest(req.run,art.checkpoint_path,4,79,generation_id='learned_degree_option_a',
        options={'extensions':{'initialization':{'degree_generator':prior}}}))
    manifest=json.loads(gen.manifest_path.read_text())
    assert manifest['degree_prior']['learned']
    assert manifest['degree_prior']['training_degree_multiset_verified']
    assert manifest['degree_prior']['explicit_denoiser_conditioning']=='initialization_only'
    assert not manifest['structural_rewiring']['global_degree_constraint']
    assert manifest['structural_rewiring']['local_rewiring_degree_preservation_rate']==1.


def test_audit_cli_runs_on_generated_option_a_batch(trained_option_a):
    import subprocess
    import sys
    w,req,art,_=trained_option_a
    gen=w.generate(GenerateRequest(req.run,art.checkpoint_path,2,91,generation_id='audit_cli'))
    root=Path(__file__).resolve().parents[1]
    result=subprocess.run([sys.executable,str(root/'scripts'/'audit_gdsm_option_a.py'),
        '--generated-dir',str(gen.generation_dir)],check=True,capture_output=True,text=True)
    audit=json.loads(result.stdout)
    assert audit['num_graphs']==2
    assert audit['all_local_rewiring_events_preserve_degrees']
    assert audit['final_event_degree_preservation_rate']==1.
    assert audit['final_graphs_simple_rate']==1.


def test_local_summary_reconciliation_does_not_restore_original_degrees():
    raw={'clustering_histogram':np.ones(10)/10,
         'orbit_log_mean':np.log1p([4.,2.,1.,1.]),'graphlet_histogram':np.array([.4,.6])}
    current=np.array([1,1,0,0,0,0]); sampled=np.array([4,4,4,4,4,4])
    actual=reconcile_summary3(raw,current)
    prior=reconcile_summary3(raw,sampled)
    assert actual['orbit_log_mean'][0] == pytest.approx(np.log1p(current.mean()))
    assert actual['clustering_histogram'][0] == pytest.approx(1.)
    assert not np.allclose(actual['orbit_log_mean'],prior['orbit_log_mean'])


def test_option_a_handles_edgeless_decodes_without_repair(trained_option_a,monkeypatch):
    import grapher.models.gdsm_simple.spectral_decode_pipeline as pipeline
    # Exact zero clean spectra exercise empty edge sets with all n nodes kept.
    monkeypatch.setattr(pipeline,'centered_x0',lambda x,*args: torch.zeros_like(x))
    w,req,art,_=trained_option_a
    gen=w.generate(GenerateRequest(req.run,art.checkpoint_path,2,97,generation_id='edgeless'))
    graphs=load(gen.graphs_path)
    assert all(len(g)==6 and g.number_of_edges()==0 for g in graphs)
    aggregate=json.loads(gen.manifest_path.read_text())['structural_rewiring']
    assert aggregate['final_connected_rate']==0.
    assert aggregate['mean_final_isolates']==6.
    assert aggregate['mean_accepted_steps']==0.
    assert aggregate['mean_feedback_events']==0.

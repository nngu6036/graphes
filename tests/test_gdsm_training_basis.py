"""Fixed-training-basis sampling, exact degrees, compatibility and audit regressions."""
from __future__ import annotations

import copy
import json
import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import torch
import yaml

from grapher.models.base import DatasetReference, GenerateRequest, RunSpec, TrainRequest
from grapher.models.gdsm_simple.wrapper import GDSMSimpleWrapper, _sha256
from grapher.models.gdsm_simple.categorical.config import resolve
from grapher.models.gdsm_simple.categorical.evaluation import audit
from grapher.models.gdsm_simple.categorical.pipeline import prepare_data, validate_generation
from grapher.models.gdsm_simple.categorical.training_basis import (
    basis_digest, fixed_basis_proposal, sample_training_basis,
)
from test_gdsm_spectral_topology import options, labelled, small_model


def read_pickle(path):
    with Path(path).open('rb') as f:
        return pickle.load(f)


def fixed_overrides(full_trace=False):
    return {'extensions': {'attributed_categorical': {
        'topology': {'basis_source': 'training_bank'},
        'spectrum_feedback': 0., 'save_degree_trajectory': True,
        'save_trajectory': full_trace,
    }}}


def test_basis_sampling_is_same_size_deterministic_and_does_not_mutate_bank():
    rng = np.random.default_rng(42)
    bank = {n: [np.linalg.qr(rng.normal(size=(n,n)))[0].astype(np.float32) for _ in range(3)]
            for n in [3,5]}
    a, j, record = sample_training_basis(bank,5,np.random.default_rng(7))
    b, k, repeat = sample_training_basis(bank,5,np.random.default_rng(7))
    assert j == k and record == repeat and record['bank_size_for_n'] == 3
    np.testing.assert_array_equal(a,bank[5][j])
    np.testing.assert_array_equal(a,b)
    assert record['basis_sha256'] == basis_digest(a)
    a[:] = 0
    assert np.linalg.norm(bank[5][j]) > 1
    assert not record['fallback'] and record['fixed_for_entire_trajectory']


def test_missing_size_cannot_silently_use_haar_or_a_different_size():
    with pytest.raises(ValueError,match='No training eigenbasis.*n=4'):
        sample_training_basis({3:[np.eye(3)]},4,np.random.default_rng(42))


@pytest.mark.parametrize('matrix',[np.ones((3,3)),np.eye(4),np.full((3,3),np.nan)])
def test_invalid_donor_basis_is_rejected(matrix):
    with pytest.raises(ValueError):
        sample_training_basis({3:[matrix]},3,np.random.default_rng(42))


def test_fixed_proposal_matches_matrix_product_handles_padding_and_backpropagates():
    torch.manual_seed(12)
    mask=torch.tensor([[1,1,1,1],[1,1,0,0]],dtype=torch.bool)
    u=torch.zeros(2,4,4)
    u[0]=torch.linalg.qr(torch.randn(4,4)).Q
    u[1,:2,:2]=torch.linalg.qr(torch.randn(2,2)).Q
    values=torch.randn(2,4,requires_grad=True)
    out=fixed_basis_proposal(values,u,mask)
    for i,n in enumerate([4,2]):
        expected=u[i,:n,:n]@torch.diag(values[i,:n]*n**.5)@u[i,:n,:n].T
        expected=expected-torch.diag_embed(expected.diag())
        torch.testing.assert_close(out[i,:n,:n],expected)
    assert not out.diagonal(dim1=-2,dim2=-1).any()
    assert not out[1,2:].any() and not out[1,:,2:].any()
    out.square().sum().backward()
    assert torch.isfinite(values.grad).all() and values.grad.abs().sum()>0
    assert not values.grad[1,2:].any()


def test_fixed_proposal_is_sign_invariant_and_row_permutation_equivariant():
    u=torch.linalg.qr(torch.randn(1,5,5)).Q
    z=torch.tensor([[-2.,-1.,0.,1.,2.]])
    mask=torch.ones(1,5,dtype=torch.bool)
    a=fixed_basis_proposal(z,u,mask)
    signs=torch.tensor([1.,-1.,-1.,1.,-1.])
    torch.testing.assert_close(a,fixed_basis_proposal(z,u*signs,mask))
    p=torch.tensor([3,2,4,0,1])
    torch.testing.assert_close(a[:,p][:,:,p],fixed_basis_proposal(z,u[:,p],mask))


@pytest.mark.parametrize('case',['nan','shape','mask'])
def test_fixed_proposal_rejects_invalid_inputs(case):
    u=torch.eye(3)[None]; z=torch.ones(1,3); mask=torch.ones(1,3,dtype=torch.bool)
    if case=='nan':z[0,0]=float('nan')
    if case=='shape':u=u[:,:2,:2]
    if case=='mask':mask[0,1]=False
    with pytest.raises((ValueError,FloatingPointError)):
        fixed_basis_proposal(z,u,mask)


@pytest.mark.parametrize('attributed',[False,True])
def test_denoiser_uses_donor_not_current_eigenpairs(monkeypatch,attributed):
    from grapher.models.gdsm_simple.categorical import model as module
    model,batch,cfg,_=small_model(attributed);model.eval()
    b,n=batch['mask'].shape
    u=torch.zeros(b,n,n)
    for i,count in enumerate(batch['mask'].sum(1)):
        count=int(count);u[i,:count,:count]=torch.linalg.qr(torch.randn(count,count)).Q
    def forbidden(*args,**kwargs):
        raise AssertionError('Current graph eigenpairs must not define the fixed proposal')
    monkeypatch.setattr(module,'eigenpairs',forbidden)
    args=(batch['x'],batch['e'],batch['z'],torch.tensor([5]*b),batch['anchor'],batch['mask'],10)
    pred=model(*args,edge_support=batch['e']>0,proposal_basis=u)
    other=model(*args,edge_support=batch['e']>0,proposal_basis=u,
                current_pairs=(torch.zeros(b,n),torch.zeros(b,n,n)))
    expected=fixed_basis_proposal(pred['clean_spectrum'],u,batch['mask'])
    torch.testing.assert_close(pred['spectral_scores'],expected)
    torch.testing.assert_close(pred['spectral_scores'],other['spectral_scores'])
    assert pred['fixed_training_basis'] and pred['bond_only']
    assert (model.edge_head is None)==(not attributed)


def test_training_bank_provenance_excludes_validation_and_survives_reservoir_replacement():
    op=options(False);cfg=resolve(op);cfg['initialization']['basis_max_per_size']=2
    train=[nx.path_graph(6),nx.cycle_graph(6),nx.complete_graph(6),nx.star_graph(5),nx.wheel_graph(6)]
    val=[nx.path_graph(5)]
    _,_,_,_,bank,schema=prepare_data(train,val,6,cfg,42)
    assert set(bank)=={6} and len(bank[6])==2
    records=schema['basis_bank_provenance']['entries'][6]
    for u,record in zip(bank[6],records):
        assert record['basis_sha256']==basis_digest(u)
        g=train[record['training_graph_index']]
        coeff=np.array(record['normalized_adjacency_eigenvalues'])*len(g)**.5
        np.testing.assert_allclose((u*coeff[None])@u.T,nx.to_numpy_array(g),atol=1e-6)
        assert record['indexed_degrees']==[g.degree(i) for i in g]
    assert schema['basis_bank_provenance']['source_split']=='train'


@pytest.mark.parametrize('key,value',[('basis_source','unknown'),('mode','categorical'),('decoder','threshold')])
def test_incompatible_basis_controls_are_rejected(key,value):
    op=options(False);cat=op['extensions']['attributed_categorical']
    cat['topology']['basis_source']='training_bank';cat['spectrum_feedback']=0.
    cat['topology'][key]=value
    with pytest.raises(ValueError):resolve(op)


def test_fixed_basis_rejects_sorted_current_spectrum_feedback():
    op=options(False)
    op['extensions']['attributed_categorical']['topology']['basis_source']='training_bank'
    with pytest.raises(ValueError,match='spectrum_feedback: 0.0'):resolve(op)


@pytest.fixture(scope='module',params=[False,True],ids=['generic','attributed'])
def managed(request,tmp_path_factory):
    attributed=request.param;root=tmp_path_factory.mktemp('fixed_training_basis')
    folder=root/'data/toy';folder.mkdir(parents=True)
    graphs=[labelled(nx.cycle_graph(6)),labelled(nx.path_graph(6)),labelled(nx.complete_graph(4)),labelled(nx.path_graph(3))]
    for split,values in [('train',graphs),('val',[labelled(nx.wheel_graph(6)),labelled(nx.path_graph(4))]),('test',[labelled(nx.ladder_graph(3))])]:
        with (folder/f'{split}.pkl').open('wb') as f:pickle.dump(values,f)
    run=RunSpec('gdsm_simple','community_small','reuse_v2',42,root/'runs')
    wrapper=GDSMSimpleWrapper()
    req=TrainRequest(run,DatasetReference('community_small',root/'data','toy'),options=options(attributed))
    artifacts=wrapper.train(req)
    result=wrapper.generate(GenerateRequest(run,artifacts.checkpoint_path,6,73,
                           generation_id='fixed_basis',options=fixed_overrides()))
    return wrapper,req,artifacts,result


def test_reuse_existing_bond_only_checkpoint_and_audit(managed):
    _,_,art,gen=managed
    report=audit(gen.generation_dir)
    assert report['status']=='passed' and report['fixed_training_eigenbasis']
    assert report['sampled_basis_matches_checkpoint'] is True
    assert report['final_spectral_proposal_matches_saved_training_basis']
    assert report['recorded_decoder_basis_updates_after_initialization']==0
    assert report['recorded_basis_updates_per_graph']==1
    assert report['prior_indexed_degree_preservation_rate']==1.
    assert report['connectedness_rate']==1.
    assert report['recorded_categorical_degree_change_steps_mean']==0.
    assert report['saved_intermediate_degree_vectors_verified']
    assert not report['saved_full_intermediate_graphs_verified']
    manifest=json.loads(gen.manifest_path.read_text())
    assert manifest['decode']['generation_basis_differs_from_training']
    assert not manifest['decode']['global_projection_optimality_guarantee']
    assert manifest['diagnostics']['graph_eigenpair_recomputations_per_graph']==1
    state=torch.load(art.checkpoint_path,map_location='cpu',weights_only=False)
    donors=read_pickle(gen.generation_dir/'sampled_training_eigenvectors.pkl')
    records=read_pickle(gen.generation_dir/'sampled_training_basis_records.pkl')
    for u,record in zip(donors,records):
        np.testing.assert_array_equal(u,state['basis_bank'][len(u)][record['bank_index_within_size']])


def test_fixed_generation_only_decomposes_final_graphs_and_is_reproducible(managed,monkeypatch):
    from grapher.models.gdsm_simple.categorical import pipeline,model as module
    wrapper,req,art,first=managed; calls=[];original=pipeline.eigenpairs
    def counted(e,mask):
        calls.append(tuple(e.shape));return original(e,mask)
    def forbidden(*args,**kwargs):
        raise AssertionError('Model must not recompute current-graph proposal eigenbasis')
    monkeypatch.setattr(pipeline,'eigenpairs',counted)
    monkeypatch.setattr(module,'eigenpairs',forbidden)
    second=wrapper.generate(GenerateRequest(req.run,art.checkpoint_path,6,73,
                            generation_id='repeat_fixed',options=fixed_overrides()))
    assert len(calls)==2 # Six graphs, batches of three: final artifacts only.
    assert _sha256(first.graphs_path)==_sha256(second.graphs_path)
    assert _sha256(first.generation_dir/'sampled_training_eigenvectors.pkl')==_sha256(second.generation_dir/'sampled_training_eigenvectors.pkl')
    assert audit(second.generation_dir)['status']=='passed'


def test_full_intermediate_graphs_match_saved_degree_vectors(managed):
    wrapper,req,art,_=managed
    gen=wrapper.generate(GenerateRequest(req.run,art.checkpoint_path,3,75,
                         generation_id='full_trace_fixed',options=fixed_overrides(True)))
    report=audit(gen.generation_dir)
    assert report['saved_full_intermediate_graphs_verified']
    assert report['saved_intermediate_degree_vectors_verified']


def test_old_v2_configs_without_new_default_keys_remain_compatible(managed):
    _,req,art,_=managed
    state=torch.load(art.checkpoint_path,map_location='cpu',weights_only=False)
    state['categorical_config']['topology'].pop('basis_source',None)
    state['categorical_config'].pop('save_degree_trajectory',None)
    state['schema'].pop('basis_bank_provenance',None)
    op=copy.deepcopy(req.options)
    op['extensions']['attributed_categorical'].update(fixed_overrides()['extensions']['attributed_categorical'])
    # update nested topology without deleting required old controls
    op['extensions']['attributed_categorical']['topology']={
        **state['categorical_config']['topology'],'basis_source':'training_bank'}
    cfg=validate_generation(state,op)
    assert cfg['topology']['basis_source']=='training_bank'


def replace_hashed_pickle(root,name,fn):
    path=root/name;obj=read_pickle(path);fn(obj)
    with path.open('wb') as f:pickle.dump(obj,f)
    manifest_path=root/'manifest.json';manifest=json.loads(manifest_path.read_text())
    manifest['artifact_sha256'][name]=_sha256(path)
    manifest_path.write_text(json.dumps(manifest))


@pytest.mark.parametrize('kind',['degree','proposal','donor'])
def test_audit_catches_semantic_corruption_after_checksums_are_updated(managed,tmp_path,kind):
    import shutil
    _,_,_,gen=managed;root=tmp_path/'corrupt';shutil.copytree(gen.generation_dir,root)
    if kind=='degree':
        def change(obj):obj['indexed_degrees'][0][1,0]+=1
        replace_hashed_pickle(root,'degree_trajectories.pkl',change)
    elif kind=='proposal':
        def change(obj):obj[0]['spectral_scores'][0,1]+=.3
        replace_hashed_pickle(root,'predicted_summaries.pkl',change)
    else:
        # Orthonormal but not the claimed checkpoint donor, including digest rewrite.
        def change(obj):obj[0]=np.eye(len(obj[0]),dtype=np.float32)
        replace_hashed_pickle(root,'sampled_training_eigenvectors.pkl',change)
        changed=read_pickle(root/'sampled_training_eigenvectors.pkl')[0]
        def change_record(obj):obj[0]['basis_sha256']=basis_digest(changed)
        replace_hashed_pickle(root,'sampled_training_basis_records.pkl',change_record)
    with pytest.raises((AssertionError,ValueError)):audit(root)


@pytest.mark.parametrize('dataset',['community_small','ego_small','qm9','zinc'])
@pytest.mark.parametrize('seed',[42,43,44])
def test_new_explicit_configs_preserve_training_and_degree_prior_settings(dataset,seed):
    root=Path('configs/experiments')
    new=yaml.safe_load((root/f'gdsm_training_basis_degree_explicit/{dataset}_seed_{seed}.yaml').read_text())['gdsm_simple']
    old=yaml.safe_load((root/f'gdsm_spectral_degree_explicit/{dataset}_seed_{seed}.yaml').read_text())['gdsm_simple']
    cfg=resolve(new)
    assert cfg['topology']['basis_source']=='training_bank'
    assert cfg['topology']['decoder']=='degree_preserving' and cfg['topology']['preserve_connectivity']
    assert cfg['spectrum_feedback']==0. and cfg['save_degree_trajectory']
    assert new['train']==old['train'] and new['model']==old['model']
    assert cfg['graphlets']['sizes']==[3,4,5]
    assert cfg['initialization']==resolve(old)['initialization']


def test_control_differs_only_in_basis_source():
    root=Path('configs/experiments/gdsm_training_basis_degree_explicit')
    base=yaml.safe_load((root/'community_small_seed_42.yaml').read_text())
    control=yaml.safe_load((root/'community_small_seed_42_current_basis_control.yaml').read_text())
    control['gdsm_simple']['extensions']['attributed_categorical']['topology']['basis_source']='training_bank'
    assert base==control


def test_generate_from_old_v2_without_new_fields_or_bank_provenance(managed,tmp_path):
    import shutil
    wrapper,req,art,_=managed
    run=RunSpec('gdsm_simple','community_small','legacy_v2',42,tmp_path/'runs')
    shutil.copytree(req.run.layout.train_dir,run.layout.train_dir)
    checkpoint=run.layout.train_dir/'checkpoints/gdsm_simple.pt'
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    state['categorical_config']['topology'].pop('basis_source',None)
    state['categorical_config'].pop('save_degree_trajectory',None)
    state['schema'].pop('basis_bank_provenance',None)
    torch.save(state,checkpoint)
    mpath=run.layout.training_manifest_path
    manifest=json.loads(mpath.read_text());manifest['run_id']=run.run_id
    manifest['checkpoint']['sha256']=_sha256(checkpoint)
    cat=manifest['options']['extensions']['attributed_categorical']
    cat['topology'].pop('basis_source',None);cat.pop('save_degree_trajectory',None)
    mpath.write_text(json.dumps(manifest))
    gen=wrapper.generate(GenerateRequest(run,checkpoint,3,77,options=fixed_overrides()))
    records=read_pickle(gen.generation_dir/'sampled_training_basis_records.pkl')
    assert all(row['training_graph_index'] is None for row in records)
    assert audit(gen.generation_dir)['sampled_basis_matches_checkpoint'] is True


def test_fresh_training_with_generation_profile_keeps_explicit_training_contract(managed,tmp_path):
    wrapper,req,_,_=managed
    op=copy.deepcopy(req.options)
    cat=op['extensions']['attributed_categorical']
    cat['topology']['basis_source']='training_bank'
    cat['spectrum_feedback']=0.;cat['save_degree_trajectory']=True;cat['save_trajectory']=False
    run=RunSpec('gdsm_simple','community_small','fresh_training_basis',42,tmp_path/'runs')
    art=wrapper.train(TrainRequest(run,req.dataset,options=op))
    manifest=json.loads(run.layout.training_manifest_path.read_text())
    assert manifest['contract']['training_proposal_basis']=='current_corrupted_graph'
    assert manifest['contract']['configured_generation_basis']=='training_bank'
    gen=wrapper.generate(GenerateRequest(run,art.checkpoint_path,3,79))
    report=audit(gen.generation_dir)
    assert report['status']=='passed' and report['sampled_basis_matches_checkpoint'] is True

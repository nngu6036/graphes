"""Regression tests for one-state adjacency diffusion and molecular integration."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import os
import subprocess
import sys

import networkx as nx
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from grapher.rewiring_mlp.attributed.adjacency_diffusion import (
    ADJACENCY_MODE, ADJACENCY_TYPE, LEGACY_MODE, adjacency_views,
    masked_eigenvalues, spectral_features, settings, validate_model_config,
)
from grapher.rewiring_mlp.attributed.soft_edge_bridge import (
    labels_to_logits, edge_probabilities, bridge_edges, pair_mask,
)
from grapher.rewiring_mlp.attributed.joint_typed_edge_training import build_model, validate_config
from grapher.rewiring_mlp.attributed.joint_typed_edge_model import (
    noisy_batch, structural_loss, save_checkpoint, load_checkpoint,
    ADJACENCY_FORMAT, FORMAT,
)
from grapher.rewiring_mlp.attributed.joint_typed_edge_data import EndpointStore, collate
from grapher.rewiring_mlp.attributed.joint_typed_edge_generation import sample_soft_endpoint, refine_typed_graph
from grapher.utils.io import save_yaml
from test_joint_typed_edge import tiny_config, carbon_cycle, prepare_dataset


@pytest.fixture(autouse=True)
def small_threads():
    old=torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def adjacency_config(base=None, *, spectral=True, loss=0.0, graphlets=False, views=None):
    cfg=deepcopy(base or tiny_config())
    cfg['attributed_predictor']['type']=ADJACENCY_TYPE
    cfg['edge_diffusion'].update(spectral_mode=ADJACENCY_MODE, spectral_sigma=0., spectral_enabled=spectral)
    cfg['adjacency_spectrum']={'views':views or ['topology','bond_weighted'],
        'bond_weights':{1:1.,2:2.,3:3.},'normalization':'size_bound'}
    cfg['attributed_predictor']['loss_weights'].update(spectrum=0.,adjacency_spectrum=loss)
    if graphlets:
        cfg['structure_summary_prediction'].update(induced_graphlet_histogram=True,
            induced_graphlet_attributed=True,induced_graphlet_k=5,induced_graphlet_scope='all')
        cfg['attributed_predictor']['loss_weights']['induced_graphlet_histogram']=1.
        cfg['attributed_refiner']['weights']['graphlet']=.1
    return cfg


def model_batch(cfg=None):
    cfg=cfg or adjacency_config()
    graphs=[carbon_cycle(5),carbon_cycle(6)]
    graphs[1].nodes[0]['atomic_num']=8
    model=build_model(cfg,graphs,torch.device('cpu'))
    store=EndpointStore(graphs,model.vectorizer,model.atom_types,cfg,seed=7,
                        graphlet_basis=model.induced_graphlet_basis)
    items=[store[i] for i in range(len(graphs))]
    batch=collate(items,model.vectorizer,model.atom_types)
    return cfg,model,items,batch


def hard_prob(graph, categories=2, pad=0):
    n=len(graph); labels=torch.zeros(1,n+pad,n+pad,dtype=torch.long)
    for u,v,d in graph.edges(data=True): labels[0,u,v]=labels[0,v,u]=d.get('bond_type',1)
    mask=torch.arange(n+pad)[None]<n
    return F.one_hot(labels,categories).float(), mask


@pytest.mark.parametrize('graph,expected',[
    (nx.path_graph(2),[-1.,1.]),
    (nx.complete_graph(3),[-1.,-1.,2.]),
    (nx.cycle_graph(4),[-2.,0.,0.,2.]),
])
def test_signed_adjacency_spectra_no_zero_mode(graph,expected):
    p,mask=hard_prob(graph,pad=3)
    f=spectral_features(p,mask,[1.],['topology'])
    assert torch.allclose(f['spectra'][0,0,:len(graph)],torch.tensor(expected),atol=1e-6)
    assert torch.count_nonzero(f['spectra'][0,0,len(graph):])==0
    assert f['spectra'].sum().abs()<1e-5
    assert f['spectra'].min()<0
    assert f['normalized'].abs().max()<=1+1e-6


def test_padding_arbitrary_node_mask_and_permutation():
    p,mask=hard_prob(nx.path_graph(3),pad=3)
    perm=torch.tensor([3,0,4,2,1,5])
    q=p[:,perm][:,:,perm]; qm=mask[:,perm]
    a=spectral_features(p,mask,[1],['topology'])
    b=spectral_features(q,qm,[1],['topology'])
    assert torch.allclose(a['spectra'],b['spectra'],atol=1e-6)
    assert torch.equal(a['mask'],b['mask'])
    # Invalid-pair values must not contribute to spectra.
    q[~pair_mask(qm)] = torch.tensor([0.,999.])
    c=spectral_features(q,qm,[1],['topology'])
    assert torch.allclose(a['spectra'],c['spectra'],atol=1e-6)


def test_weights_and_per_bond_views_not_category_codes():
    g=nx.path_graph(3);g.edges[0,1]['bond_type']=2;g.edges[1,2]['bond_type']=1
    p,mask=hard_prob(g,categories=4)
    a=adjacency_views(p,mask,[1.,1.5,3.],['topology','bond_weighted','per_bond'])
    assert a.shape==(1,5,3,3)
    assert a[0,1,0,1]==1.5 and a[0,1,1,2]==1
    assert a[0,2,1,2]==1 and a[0,3,0,1]==1
    assert torch.equal(a[:,:,range(3),range(3)],torch.zeros(1,5,3))
    f=spectral_features(p,mask,[1.,1.5,3.],['topology','bond_weighted','per_bond'])
    assert torch.equal(f['scale'],torch.tensor([[2.,6.,2.,2.,2.]]))
    cfg=adjacency_config();cfg['adjacency_spectrum']['bond_weights']={10:1.,20:1.5,30:3.}
    assert settings(cfg,(10,20,30))['bond_weights']==[1.,1.5,3.]


def test_same_expected_weight_different_categories_are_retained():
    mask=torch.ones(1,2,dtype=torch.bool)
    a=torch.zeros(1,2,2,4);b=a.clone()
    a[0,0,1,2]=a[0,1,0,2]=1.
    b[0,0,1,1]=b[0,1,0,1]=.5;b[0,0,1,3]=b[0,1,0,3]=.5
    av=adjacency_views(a,mask,[1,2,3],['topology','bond_weighted','per_bond'])
    bv=adjacency_views(b,mask,[1,2,3],['topology','bond_weighted','per_bond'])
    assert torch.equal(av[:,:2],bv[:,:2])
    assert not torch.equal(av[:,2:],bv[:,2:])


@pytest.mark.parametrize('n',[0,1])
def test_empty_or_single_node_helper_no_nan(n):
    p=torch.randn(1,4,4,2).softmax(-1).requires_grad_(True)
    mask=torch.arange(4)[None]<n
    f=spectral_features(p,mask,[1.],['topology'])
    assert torch.isfinite(f['spectra']).all() and torch.count_nonzero(f['spectra'])==0
    f['spectra'].square().sum().backward()
    assert torch.isfinite(p.grad).all()


def test_spectral_autograd_repeated_modes_and_float64():
    # Uniform off-diagonal probabilities give repeated eigenvalues.
    z=torch.zeros(2,5,5,4,requires_grad=True,dtype=torch.float64)
    mask=torch.ones(2,5,dtype=torch.bool)
    f=spectral_features(edge_probabilities(z,mask),mask,[1.,2.,3.],['topology','bond_weighted'])
    assert f['spectra'].dtype==torch.float64
    f['spectra'].square().sum().backward()
    assert torch.isfinite(z.grad).all() and z.grad.abs().sum()>0


def test_eigenvalues_gradcheck():
    z=torch.randn(1,3,3,2,dtype=torch.double,requires_grad=True)
    mask=torch.ones(1,3,dtype=torch.bool)
    def fn(x):
        p=edge_probabilities((x+x.transpose(1,2))/2,mask)
        return spectral_features(p,mask,[1.],['topology'])['spectra']
    assert torch.autograd.gradcheck(fn,(z,),eps=1e-6,atol=1e-4)


@pytest.mark.parametrize('bad',[
    {'views':['bogus']},{'views':['topology','topology']},{'normalization':'mean_degree'},
    {'bond_weights':{1:1,2:2}},{'bond_weights':{1:1,2:2,3:float('nan')}},
    {'bond_weights':{1:1,2:2,3:-3}}, {'unused_key': True},
])
def test_bad_adjacency_settings_rejected(bad):
    cfg=adjacency_config();cfg['adjacency_spectrum'].update(bad)
    with pytest.raises(ValueError):validate_config(cfg)


def test_missing_weight_mapping_not_guessed():
    cfg=adjacency_config();del cfg['adjacency_spectrum']['bond_weights']
    with pytest.raises(ValueError,match='Specify'):validate_config(cfg)


@pytest.mark.parametrize('key,value',[
    ('spectral_sigma',.15), ('spectral_mode',LEGACY_MODE),
])
def test_no_accidental_independent_process(key,value):
    cfg=adjacency_config();cfg['edge_diffusion'][key]=value
    with pytest.raises(ValueError):validate_config(cfg)


def test_wrong_loss_family_rejected():
    cfg=adjacency_config();cfg['attributed_predictor']['loss_weights']['spectrum']=.5
    with pytest.raises(ValueError,match='adjacency_spectrum'):validate_config(cfg)
    cfg=tiny_config();cfg['attributed_predictor']['loss_weights']['adjacency_spectrum']=.1
    with pytest.raises(ValueError,match='requires'):validate_config(cfg)


def test_no_laplacian_in_data_noisy_input_or_rollout(monkeypatch):
    def forbidden(*args,**kw):raise AssertionError('Independent Laplacian path executed')
    import grapher.rewiring_mlp.attributed.joint_typed_edge_data as data
    import grapher.rewiring_mlp.attributed.joint_typed_edge_model as mod
    import grapher.rewiring_mlp.attributed.joint_typed_edge_generation as gen
    monkeypatch.setattr(data,'attributed_laplacian_spectra',forbidden)
    monkeypatch.setattr(mod,'bridge_spectra',forbidden)
    monkeypatch.setattr(mod,'project_spectra',forbidden)
    monkeypatch.setattr(gen,'spectral_noise',forbidden)
    cfg,m,items,b=model_batch()
    assert 'source_spectra' not in b and 'target_spectra' not in b
    noisy=noisy_batch(b,m,cfg,endpoint_only=True)
    assert 'spectral_state' not in noisy
    pred=m(noisy)
    assert 'clean_spectra' not in pred and not hasattr(m,'spectrum_head')
    assert 'input_adjacency_spectra' in pred
    targets,stats=sample_soft_endpoint(m,items[0]['source'],cfg,seed=4)
    assert 'spectra' not in targets and stats['diffusion']['independent_spectral_diffusion'] is False
    assert stats['prediction_calls']==cfg['edge_diffusion']['sampling_steps']+1


def test_noise_stream_contains_only_edge_noise():
    cfg,m,_,b=model_batch()
    gen=torch.Generator().manual_seed(53); expected=torch.Generator().manual_seed(53)
    inp=noisy_batch(b,m,cfg,generator=gen,endpoint_only=True)
    s=labels_to_logits(b['source_labels'],m.categories,b['mask'],m.smoothing)
    t=labels_to_logits(b['target_labels'],m.categories,b['mask'],m.smoothing)
    value=bridge_edges(s,t,torch.zeros(len(b['n'])),b['mask'],cfg['edge_diffusion']['sigma'],generator=expected)
    assert torch.equal(inp['edge_state'],value)
    assert torch.equal(gen.get_state(),expected.get_state())


def test_spectra_are_derived_not_supplied():
    cfg,m,_,b=model_batch();m.eval()
    inp=noisy_batch(b,m,cfg,endpoint_only=True);out=m(inp)
    wrong={**inp,'spectral_state':torch.full((2,2,6),float('nan')),
           'source_spectra':torch.full((2,2,6),123456.)}
    out2=m(wrong)
    assert torch.equal(out['clean_edge_logits'],out2['clean_edge_logits'])
    expected=m.adjacency_features(out['clean_edge_probabilities'],b['mask'])['spectra']
    assert torch.equal(expected,out['clean_adjacency_spectra'])
    changed={**inp,'edge_state':torch.zeros_like(inp['edge_state'])}
    assert not torch.allclose(out['input_adjacency_spectra'],m(changed)['input_adjacency_spectra'])


def test_joint_gradients_optional_spectral_loss():
    cfg,m,_,b=model_batch(adjacency_config(loss=.1))
    inp=noisy_batch(b,m,cfg,endpoint_only=True); out=m(inp)
    weights={k:0. for k in cfg['attributed_predictor']['loss_weights']};weights['adjacency_spectrum']=1.
    loss,metrics=structural_loss(out,inp,m,weights); loss.backward()
    assert 'adjacency_spectral_nrmse' in metrics and 'spectral_nrmse' not in metrics
    assert metrics['adjacency_spectral_trace_max_abs']<1e-4
    for prefix in ('degree_model.encoder','degree_model.signature_decoder','edge_head'):
        ps=[p for n,p in m.named_parameters() if n.startswith(prefix) and p.grad is not None]
        assert ps and all(torch.isfinite(p.grad).all() for p in ps)
        assert sum(float(p.grad.abs().sum()) for p in ps)>0


@pytest.mark.parametrize('features,loss',[(False,0.),(False,.1),(True,0.),(True,.1)])
def test_feature_and_spectral_loss_ablations(features,loss):
    cfg,m,_,b=model_batch(adjacency_config(spectral=features,loss=loss))
    inp=noisy_batch(b,m,cfg,endpoint_only=True);out=m(inp)
    value,_=structural_loss(out,inp,m,cfg['attributed_predictor']['loss_weights'])
    value.backward();assert torch.isfinite(value)
    assert ('input_adjacency_spectra' in out)==features
    assert ('clean_adjacency_spectra' in out)==(features or loss>0)
    assert not hasattr(m,'spectrum_head')


def test_per_bond_model_and_permutation_equivariance():
    cfg,m,_,b=model_batch(adjacency_config(views=['topology','per_bond']))
    inp=noisy_batch(b,m,cfg,endpoint_only=True);m.eval();out=m(inp)
    assert out['input_adjacency_spectra'].shape==(2,4,6)
    perm=torch.tensor([3,0,5,2,1,4]);x=dict(inp)
    for key in ('mask','atom','typed_degrees'):x[key]=inp[key][:,perm]
    for key in ('source_labels','target_labels','edge_state'):x[key]=inp[key][:,perm][:,:,perm]
    moved=m(x)
    assert torch.allclose(moved['clean_edge_logits'],out['clean_edge_logits'][:,perm][:,:,perm],atol=5e-6)
    assert torch.allclose(moved['clean_clustering_histogram'],out['clean_clustering_histogram'],atol=5e-6)
    assert torch.allclose(moved['input_adjacency_spectra'],out['input_adjacency_spectra'],atol=5e-6)


def test_matching_smoothed_target_zero_spectral_error():
    cfg,m,_,b=model_batch();inp=noisy_batch(b,m,cfg,endpoint_only=True);out=m(inp)
    target=labels_to_logits(b['target_labels'],m.categories,b['mask'],m.smoothing)
    out['clean_edge_logits']=target;out['clean_edge_probabilities']=edge_probabilities(target,b['mask'])
    out['clean_adjacency_spectra']=m.adjacency_features(out['clean_edge_probabilities'],b['mask'])['spectra']
    _,metrics=structural_loss(out,inp,m,cfg['attributed_predictor']['loss_weights'])
    assert metrics['adjacency_spectral_nrmse']==0.


def test_rollout_spectral_consistency_and_rewiring_invariants():
    cfg,m,items,_=model_batch(adjacency_config(graphlets=True))
    source=items[0]['source']
    targets,stats=sample_soft_endpoint(m,source,cfg,seed=45)
    probs=torch.tensor(targets['edge_probabilities'])[None]
    mask=torch.ones(1,len(source),dtype=torch.bool)
    expected=m.adjacency_features(probs,mask)['spectra'][0].numpy()
    assert np.allclose(expected,targets['adjacency_spectra'],atol=1e-6)
    assert targets['induced_graphlet_metadata']['attributed'] is True
    final,r=refine_typed_graph(source,targets,m,cfg,seed=45)
    assert r['typed_degree_preserved'] and r['connected']
    assert dict(final.degree())==dict(source.degree())
    again,stats2=sample_soft_endpoint(m,source,cfg,seed=45)
    assert np.array_equal(again['edge_probabilities'],targets['edge_probabilities'])
    changed,_=sample_soft_endpoint(m,source,cfg,seed=46)
    assert not np.array_equal(changed['edge_probabilities'],targets['edge_probabilities'])


def test_checkpoint_semantics_graphlet_basis_and_prior_export(tmp_path):
    cfg,m,_,b=model_batch(adjacency_config(graphlets=True));m.eval()
    inp=noisy_batch(b,m,cfg,endpoint_only=True);expected=m(inp)
    cp=tmp_path/'adj.pt';save_checkpoint(cp,m,cfg,{'toy':1})
    loaded,raw=load_checkpoint(cp)
    assert raw['format']==ADJACENCY_FORMAT
    assert loaded.diffusion_metadata()==m.diffusion_metadata()
    assert loaded.induced_graphlet_metadata()==m.induced_graphlet_metadata()
    assert torch.equal(loaded(inp)['clean_edge_logits'],expected['clean_edge_logits'])
    bad=deepcopy(cfg);bad['adjacency_spectrum']['bond_weights'][2]=1.5
    with pytest.raises(ValueError,match='weights'):validate_model_config(loaded,bad)
    with pytest.raises(ValueError,match='mode differs'):validate_model_config(loaded,tiny_config())
    raw['format']=FORMAT;torch.save(raw,tmp_path/'bad.pt')
    with pytest.raises(ValueError,match='semantics'):load_checkpoint(tmp_path/'bad.pt')


def test_legacy_checkpoints_still_load(tmp_path):
    cfg=tiny_config();m=build_model(cfg,[carbon_cycle(5),carbon_cycle(6)],torch.device('cpu'))
    save_checkpoint(tmp_path/'old.pt',m,cfg,{})
    restored,raw=load_checkpoint(tmp_path/'old.pt')
    assert raw['format']==FORMAT and 'spectral_mode' not in raw['model_config']
    assert restored.spectral_mode==LEGACY_MODE
    with pytest.raises(ValueError,match='mode differs'):validate_model_config(restored,adjacency_config())


def test_endpoint_cache_namespaces(tmp_path):
    cfg,m,_,_=model_batch();legacy=tiny_config();g=carbon_cycle(5);cache=tmp_path/'cache.sqlite'
    a=EndpointStore([g],m.vectorizer,m.atom_types,legacy,seed=9,cache_path=cache)
    assert 'source_spectra' in a[0];a.close()
    b=EndpointStore([g],m.vectorizer,m.atom_types,cfg,seed=9,cache_path=cache)
    assert 'source_spectra' not in b[0];b.close()


@pytest.mark.parametrize('dataset',['qm9','zinc'])
def test_supplied_configs(dataset):
    from grapher.utils.io import load_yaml
    cfg=load_yaml(f'configs/experiments/grapher/{dataset}_attributed_joint_adjacency_graphlets5.yaml')
    validate_config(cfg)
    assert cfg['structure_summary_prediction']['induced_graphlet_attributed'] is True
    assert cfg['degree_generator']['checkpoint_path'] is None
    assert cfg['edge_diffusion']['spectral_sigma']==0


def test_cli_train_all_checkpoints_diagnose_generate_evaluate(tmp_path):
    cfg=adjacency_config(prepare_dataset(tmp_path),graphlets=True,loss=.1)
    path=tmp_path/'cfg.yaml';save_yaml(cfg,path)
    train=tmp_path/'train'
    env={**os.environ,'PYTHONPATH':'src:.','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONHASHSEED':'0'}
    def run(script,*args,has_config=True):
        cmd=[sys.executable,'scripts/'+script]
        if has_config:cmd+=['--config',str(path)]
        result=subprocess.run(cmd+list(map(str,args)),env=env,text=True,capture_output=True,timeout=90)
        assert result.returncode==0,result.stdout+'\n'+result.stderr
        return result.stdout
    run('train_attributed_grapher.py','--output-dir',train,'--epochs',2,'--device','cpu')
    log=run('inspect_joint_typed_checkpoints.py','--training-dir',train,'--verify',has_config=False)
    assert ADJACENCY_MODE in log
    registry=json.loads((train/'checkpoint_registry.json').read_text())
    for kind in registry['selections']:
        cp=train/'checkpoints'/kind/'checkpoint.pt';gen=tmp_path/'generation'/kind
        run('run_attributed_grapher.py','--checkpoint',cp,'--output-dir',gen,'--num-generate',2,'--device','cpu')
        report=json.loads((gen/'report.json').read_text())
        assert report['diffusion']['mode']==ADJACENCY_MODE
        assert report['diffusion']['independent_spectral_diffusion'] is False
        assert report['induced_graphlet_metadata']['attributed'] is True
        assert report['diagnostics']['typed_degree_preservation_rate']==1
    for mode in ('source','bridge','rollout'):
        extra=['--rewire'] if mode=='rollout' else []
        run('diagnose_joint_typed_edge.py','--checkpoint',train/'checkpoint.pt','--mode',mode,
            '--max-graphs',2,'--device','cpu','--json-out',tmp_path/(mode+'.json'),*extra)
        result=json.loads((tmp_path/(mode+'.json')).read_text())
        assert result['diffusion']['mode']==ADJACENCY_MODE
        assert result['maxima']['adjacency_prediction_spectrum_consistency_max_abs']<1e-5
    run('evaluate_induced_graphlets.py','--checkpoint',train/'checkpoint.pt',
        '--generated-graphs',gen/'molecular_graphs.pkl','--reference-split','val','--json-out',gen/'graphlets.json')
    assert json.loads((gen/'graphlets.json').read_text())['catalogue']['attributed'] is True


def test_binary_edge_model_inside_typed_pipeline():
    cfg=adjacency_config(views=['topology'])
    cfg['categorical_state']['edge_categories']=[1]
    cfg['adjacency_spectrum']['bond_weights']={1:1.}
    cfg,m,items,b=model_batch(cfg)
    pred=m(noisy_batch(b,m,cfg,endpoint_only=True))
    assert pred['clean_edge_probabilities'].shape[-1]==2
    assert pred['input_adjacency_spectra'].shape[1]==1
    targets,_=sample_soft_endpoint(m,items[0]['source'],cfg,seed=4)
    _,result=refine_typed_graph(items[0]['source'],targets,m,cfg,seed=4)
    assert result['typed_degree_preserved']


def test_frozen_prior_is_not_updated_by_adjacency_losses():
    cfg,m,_,b=model_batch(adjacency_config(loss=.1))
    m.set_degree_trainable(False)
    inp=noisy_batch(b,m,cfg,endpoint_only=True)
    loss,_=structural_loss(m(inp),inp,m,cfg['attributed_predictor']['loss_weights'])
    loss.backward()
    assert all(p.grad is None for p in m.degree_model.parameters())
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.edge_head.parameters())
    m.zero_grad(set_to_none=True);m.set_degree_trainable(True)
    loss,_=structural_loss(m(inp),inp,m,cfg['attributed_predictor']['loss_weights']);loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.degree_model.parameters())


def test_low_precision_eigensolver_promotes_to_float32():
    p,mask=hard_prob(nx.cycle_graph(4))
    with torch.autocast('cpu',dtype=torch.bfloat16):
        result=spectral_features(p.to(torch.bfloat16),mask,[1.],['topology'])
    assert result['spectra'].dtype==torch.float32
    assert torch.isfinite(result['spectra']).all()

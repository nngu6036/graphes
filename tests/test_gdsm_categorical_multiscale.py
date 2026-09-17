"""Research-profile graphlets: exact counts, losses, local deltas, and managed IO."""
from __future__ import annotations
import copy
import itertools
import json
import pickle
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import torch

from grapher.models.base import DatasetReference,RunSpec,TrainRequest,GenerateRequest
from grapher.models.gdsm_simple.wrapper import GDSMSimpleWrapper
from grapher.models.gdsm_simple.categorical.config import resolve
from grapher.models.gdsm_simple.categorical.data import collate,typed_counts,topology_summary
from grapher.models.gdsm_simple.categorical.multiscale import (
    canonical_pattern,count_multi,update_counts,connected_subsets,TypedGraphletsMulti,
    IndexedCounts,pack_basis_counts)
from grapher.models.gdsm_simple.categorical.pipeline import prepare_data,_make_model_config,validate_generation
from grapher.models.gdsm_simple.categorical.model import SpectralCategoricalDenoiser,losses,predictions_numpy
from grapher.models.gdsm_simple.categorical.refiner import refine,candidates
from grapher.models.gdsm_simple.categorical.evaluation import evaluate,audit,rbf_mmd2
from grapher.models.gdsm_simple.categorical.multiscale_evaluation import sparse_rbf_mmd2


def opts(sizes=(3,4,5)):
    out=copy.deepcopy(GDSMSimpleWrapper.default_options)
    out['train'].update(epochs=2,batch_size=3,validation_every=1,log_every=1)
    out['model'].update(max_nodes=8,hidden_dim=16,num_layers=1,num_heads=2,ff_dim=32)
    out['diffusion']['steps']=8;out['sample']['steps']=4
    out['runtime']['device']='cpu';out['generation_batch_size']=3
    out['extensions']['attributed_categorical']={
        'enabled':True,'categories':{'node_attribute':'atom','edge_attribute':'bond'},
        'graphlets':{'sizes':list(sizes),'size_weights':[1.]*len(sizes),'clustering_bins':10},
        'guidance':{'start_fraction':1.,'every':1,'max_steps_per_event':1,'proposal_budget':16,'valid_candidate_budget':8}}
    return out


def labelled(g):
    g=g.copy()
    nx.set_node_attributes(g,{i:('C' if i%2 else 'N') for i in g},'atom')
    nx.set_edge_attributes(g,1,'bond')
    return g


def brute(x,e,orders):
    out={k:Counter() for k in orders}
    for k in orders:
        for nodes in itertools.combinations(range(len(x)),k):
            sub=e[np.ix_(nodes,nodes)]
            if nx.is_connected(nx.from_numpy_array(sub>0)):
                raw=tuple(int(x[i]) for i in nodes)+tuple(int(e[i,j]) for i,j in itertools.combinations(nodes,2))
                out[k][canonical_pattern(k,raw)]+=1
    return out


@pytest.mark.parametrize('n',[1,3,4,5,6,8])
def test_exact_counts_and_permutation_invariance(n):
    rng=np.random.default_rng(103+n)
    for prob in (0.,.18,.5,1.):
        g=nx.gnp_random_graph(n,prob,seed=n)
        x=rng.integers(0,3,n);e=nx.to_numpy_array(g,dtype=int)
        for i,j in g.edges():e[i,j]=e[j,i]=int(rng.integers(1,4))
        counts=count_multi(x,e)
        assert counts==brute(x,e,(3,4,5))
        assert counts[3]==typed_counts(x,e)
        p=rng.permutation(n)
        assert counts==count_multi(x[p],e[p][:,p])


@pytest.mark.parametrize('k,classes',[(3,2),(4,6),(5,21)])
def test_all_connected_untyped_isomorphism_classes(k,classes):
    keys=set()
    for g in nx.graph_atlas_g():
        if len(g)==k and nx.is_connected(g):
            counts=count_multi(np.zeros(k,int),nx.to_numpy_array(g,dtype=int),(k,))
            assert sum(counts[k].values())==1;keys.update(counts[k])
    assert len(keys)==classes


@pytest.mark.parametrize('k',[4,5])
def test_canonicalization_matches_full_permutations(k):
    rng=np.random.default_rng(5*k)
    for _ in range(20):
        x=rng.integers(0,3,k);e=rng.integers(0,4,(k,k));e=np.triu(e,1);e+=e.T
        raw=tuple(map(int,x))+tuple(int(e[i,j]) for i,j in itertools.combinations(range(k),2))
        expected=min(tuple(int(x[i]) for i in p)+tuple(int(e[p[i],p[j]]) for i,j in itertools.combinations(range(k),2)) for p in itertools.permutations(range(k)))
        assert canonical_pattern(k,raw)==expected


@pytest.mark.parametrize('n',[5,6,9])
def test_exact_local_deltas_include_disconnections_insertions_and_recolors(n):
    rng=np.random.default_rng(n+61)
    for _ in range(15):
        e=rng.integers(0,3,(n,n));e=np.triu(e,1);e+=e.T;x=rng.integers(0,3,n)
        before=count_multi(x,e);after=e.copy()
        for __ in range(4):
            i,j=rng.choice(n,2,replace=False);after[i,j]=after[j,i]=rng.integers(0,3)
        result=update_counts(x,e,after,before)
        assert result==count_multi(x,after)
        assert before==count_multi(x,e)  # Input cache not mutated.


def test_count_budget_fails_loud_instead_of_truncating():
    e=nx.to_numpy_array(nx.complete_graph(8),dtype=int)
    with pytest.raises(RuntimeError,match='No approximate'):
        count_multi(np.zeros(8,int),e,limit=3)


@pytest.mark.parametrize('orders',[(3,),(3,4),(3,4,5)])
def test_multi_head_shapes_normalization_gradients_and_masks(orders):
    cfg=resolve(opts(orders));gs=[labelled(nx.path_graph(3)),labelled(nx.cycle_graph(6)),labelled(nx.empty_graph(1))]
    rows,_,v,b,_,_=prepare_data(gs,gs,8,cfg,42)
    batch=collate(rows,b,device='cpu');mc=_make_model_config(opts(orders),cfg,v,b,8)
    model=SpectralCategoricalDenoiser(**mc)
    out=model(batch['x'],batch['e'],batch['z'],torch.ones(3,dtype=torch.long),batch['anchor'],batch['mask'],8)
    loss,parts=losses(out,batch,cfg['loss_weights']);loss.backward()
    assert torch.isfinite(loss) and out['graphlet_mass'].shape==(3,len(orders))
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.graphlet_head.parameters())
    h=predictions_numpy(out,0,3)['histogram']
    for k in orders:np.testing.assert_allclose(h[b.slices[k]].sum(),1.,atol=1e-6)
    target_mass=batch['mass'].clone();target_mass[~batch['graphlet_order_mask']]=100.
    other=losses(out,{**batch,'mass':target_mass},cfg['loss_weights'])[1]['mass']
    torch.testing.assert_close(other,parts['mass'])


def test_vocabulary_train_only_caps_and_compact_targets():
    cfg=resolve(opts());cfg['graphlets']['max_vocab_per_size']=1
    train=[labelled(nx.path_graph(6)),labelled(nx.star_graph(5))];val=[labelled(nx.complete_graph(6))]
    rows,vals,v,b,_,schema=prepare_data(train,val,8,cfg,42)
    assert all(len(b.keys_by_order[k])<=1 for k in b.orders)
    assert all(isinstance(row['counts'],IndexedCounts) for row in rows+vals)
    for row in rows+vals:
        a=b.encode_counts(row['counts'],len(row['x']));expected=b.summary(row['x'],row['e'])
        for av,ev in zip(a,expected):np.testing.assert_allclose(av,ev)
    h,m=b.encode_counts(vals[0]['counts'],6)
    assert all(h[b.slices[k]][-1]==1. for k in b.orders)
    for k in b.orders:assert schema['training_graphlet_vocabulary_coverage'][str(k)]['retained_classes']==1
    other=TypedGraphletsMulti.from_schema(schema)
    assert other.keys_by_order==b.keys_by_order


def test_oracle_refinement_uses_multi_order_exact_deltas():
    e=nx.to_numpy_array(nx.cycle_graph(6),dtype=int);x=np.zeros(6,int)
    cfg=resolve(opts())['guidance'];cfg.update(proposal_budget=-1,valid_candidate_budget=-1,max_steps_per_event=1)
    cand=list(candidates(e,cfg,np.random.default_rng(0)))
    target_e=next(a for a in cand if count_multi(x,a)!=count_multi(x,e))
    counts=[count_multi(x,a) for a in (e,target_e)]
    basis=TypedGraphletsMulti({k:sorted(set(counts[0][k])|set(counts[1][k])) for k in (3,4,5)})
    h,m=basis.summary(x,target_e);c,o=topology_summary(target_e,10)
    target={'histogram':h,'mass':m,'clustering':c,'orbit':o,'spectrum':np.linalg.eigvalsh(target_e)/6**.5,'edge_probs':np.ones((6,6,2))*.5}
    cfg['preserve_connectivity_if_connected']=False
    out,diag=refine(x,e,target,basis,10,cfg,np.random.default_rng(0))
    assert diag['accepted_steps']==1
    assert diag['final']['total']<diag['initial']['total'] and 'by_order' in diag['final']
    np.testing.assert_array_equal((out>0).sum(1),(e>0).sum(1))


@pytest.fixture(scope='module')
def run_multi(tmp_path_factory):
    root=tmp_path_factory.mktemp('multi');folder=root/'data/toy';folder.mkdir(parents=True)
    gs=[labelled(nx.path_graph(5)),labelled(nx.cycle_graph(6)),labelled(nx.star_graph(5)),labelled(nx.complete_graph(4))]
    for split,graphs in [('train',gs),('val',[labelled(nx.wheel_graph(6))]),('test',[labelled(nx.path_graph(7))])]:
        with (folder/f'{split}.pkl').open('wb') as f:pickle.dump(graphs,f)
    run=RunSpec('gdsm_simple','community_small','multiscale',42,root/'runs')
    req=TrainRequest(run,DatasetReference('community_small',root/'data','toy'),options=opts())
    wrapper=GDSMSimpleWrapper();art=wrapper.train(req)
    gen=wrapper.generate(GenerateRequest(run,art.checkpoint_path,6,42,generation_id='main'))
    return wrapper,req,art,gen


def test_managed_train_generate_audit_and_per_order_evaluation(run_multi):
    _,req,art,gen=run_multi
    state=torch.load(art.checkpoint_path,map_location='cpu',weights_only=False)
    assert state['schema']['graphlet_orders']==[3,4,5]
    assert state['model_config']['graphlet_block_sizes']==list(TypedGraphletsMulti.from_schema(state['schema']).block_sizes)
    assert audit(gen.generation_dir)['status']=='passed'
    result=evaluate(gen.generation_dir,req.dataset.split_paths['test'])
    assert set(result['graphlet_metrics_by_order'])=={'3','4','5'}
    assert all(v>=0 for v in result['mmd2'].values())
    result=evaluate(gen.generation_dir,gen.graphs_path)
    assert all(abs(v)<1e-12 for v in result['mmd2'].values())


def test_multi_no_guidance_reuses_checkpoint_but_changing_order_rejects(run_multi):
    wrapper,req,art,_=run_multi
    gen=wrapper.generate(GenerateRequest(req.run,art.checkpoint_path,3,42,generation_id='no_guidance',options={'extensions':{'attributed_categorical':{'guidance':{'enabled':False}}}}))
    diag=json.loads((gen.generation_dir/'rewiring_diagnostics.json').read_text())
    assert all(not r['guidance_events'] for r in diag['graphs'])
    state=torch.load(art.checkpoint_path,map_location='cpu',weights_only=False)
    other=opts((3,4))
    with pytest.raises(ValueError,match='retrain'):validate_generation(state,other)


def test_sparse_mmd_matches_dense():
    from scipy.sparse import csr_matrix
    rng=np.random.default_rng(0);a=rng.random((17,25));b=rng.random((13,25))
    a[a<.8]=0;b[b<.8]=0
    assert abs(sparse_rbf_mmd2(csr_matrix(a),csr_matrix(b),block=4)-rbf_mmd2(a,b,block=4))<1e-12


@pytest.mark.parametrize('settings',[{'sizes':[5,3]},{'sizes':[3,3]},{'sizes':[3,6]},
                                    {'sizes':[3,4,5],'size_weights':[1.,0.,1.]},
                                    {'sizes':[3,4,5],'max_vocab_per_size':0},
                                    {'sizes':[3,4,5],'counting':'sampled'}])
def test_config_rejects_ambiguous_or_unsupported_multiscale_settings(settings):
    op=opts();op['extensions']['attributed_categorical']['graphlets'].update(settings)
    with pytest.raises(ValueError):resolve(op)

@pytest.mark.parametrize('dataset',['community_small','ego_small','qm9','zinc'])
@pytest.mark.parametrize('variant',['main','no_guidance','final_only','backbone_only','categorical_only','k3','k34'])
def test_research_configs_are_consumed_and_budget_preserved(dataset,variant,tmp_path):
    suffix='' if variant=='main' else '_'+variant
    path=Path(f'configs/experiments/grapher_research/{dataset}_g345{suffix}.yaml')
    req=TrainRequest(RunSpec('gdsm_simple',dataset,'config',42),DatasetReference(dataset,tmp_path),config_path=path)
    op=GDSMSimpleWrapper()._options(req);cfg=resolve(op)
    orders=[3] if variant=='k3' else [3,4] if variant=='k34' else [3,4,5]
    assert cfg['graphlets']['sizes']==orders
    assert cfg['graphlets']['counting']=='exact_connected' and cfg['graphlets']['connected_only']
    assert op['diffusion']['steps']==op['sample']['steps']==500
    assert op['model']['hidden_dim']==128 and op['model']['num_layers']==3
    assert op['train']['max_train_graphs'] is None
    if variant in ('no_guidance','backbone_only','categorical_only'):assert not cfg['guidance']['enabled']
    if variant=='final_only':assert cfg['guidance']['start_fraction']==0.
    if variant=='categorical_only':
        assert not cfg['spectral_conditioning'] and cfg['loss_weights']['spectral']==0
    if dataset in ('community_small','ego_small'):
        ntrain=64 if dataset=='community_small' else 128
        assert op['train']['epochs']*((ntrain+op['train']['batch_size']-1)//op['train']['batch_size'])==20000

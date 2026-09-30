"""Regression tests for spectral support and real-bond-only diffusion."""
from __future__ import annotations

import copy
import json
import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import torch

from grapher.models.base import DatasetReference, GenerateRequest, RunSpec, TrainRequest
from grapher.models.gdsm_simple.wrapper import GDSMSimpleWrapper
from grapher.models.gdsm_simple.categorical.config import DEFAULTS, resolve
from grapher.models.gdsm_simple.categorical.data import collate
from grapher.models.gdsm_simple.categorical.evaluation import audit
from grapher.models.gdsm_simple.categorical.model import SpectralCategoricalDenoiser, losses
from grapher.models.gdsm_simple.categorical.noise import (
    MarginalNoise, cosine_alpha_bar, draw_bonds, forward_bond_graph,
    pair_mask, reverse_bonds,
)
from grapher.models.gdsm_simple.categorical.pipeline import (
    SPECTRAL_TOPOLOGY_FORMAT, prepare_data, validate_generation, _make_model_config,
)
from grapher.models.gdsm_simple.categorical.spectral import eigenpairs
from grapher.models.gdsm_simple.categorical.topology import decode_topology, initial_support


torch.set_num_threads(1)


def options(attributed=True):
    cfg=copy.deepcopy(DEFAULTS)
    cfg['topology'].update(mode='spectral',proposal_budget=32,preserve_connectivity=True)
    cfg['categories']={'node_attribute':'atom','edge_attribute':'bond'} if attributed else {'node_attribute':None,'edge_attribute':None}
    cfg['graphlets'].update(sizes=[3,4,5],clustering_bins=10)
    cfg['guidance'].update(every=1,start_fraction=1.,proposal_budget=12,valid_candidate_budget=8,max_steps_per_event=1)
    cfg['guidance']['weights']['edge']=0.
    cfg['save_trajectory']=True
    return {'train':{'epochs':2,'batch_size':3,'validation_every':1,'log_every':1,
                     'lr':.001,'weight_decay':1e-6,'grad_norm':1.,'max_train_graphs':None},
            'model':{'max_nodes':6,'hidden_dim':16,'num_layers':1,'num_heads':2,'ff_dim':32,'dropout':0.},
            'diffusion':{'steps':10},'sample':{'steps':4},'runtime':{'device':'cpu'},'generation_batch_size':3,
            'extensions':{'attributed_categorical':cfg}}


def labelled(g):
    g=g.copy()
    for i in g: g.nodes[i]['atom']=6+i%2
    for i,(u,v) in enumerate(g.edges()): g.edges[u,v]['bond']=1+i%3
    return g


def small_model(attributed=True):
    op=options(attributed);cfg=resolve(op)
    graphs=[labelled(nx.cycle_graph(6)),labelled(nx.path_graph(4)),labelled(nx.empty_graph(1))]
    train,_,v,b,_,schema=prepare_data(graphs,graphs,6,cfg,42)
    batch=collate(train,b,device='cpu')
    model=SpectralCategoricalDenoiser(**_make_model_config(op,cfg,v,b,6))
    return model,batch,cfg,schema


@pytest.mark.parametrize('attributed',[False,True])
def test_head_vocabulary_mask_and_gradients(attributed):
    model,batch,cfg,schema=small_model(attributed)
    calls=[]
    if attributed:
        model.edge_head.register_forward_pre_hook(lambda module,args: calls.append(args[0].shape[0]))
    else:
        assert model.edge_head is None
    support=batch['e']>0
    pred=model(batch['x'],batch['e'],batch['z'],torch.tensor([8,8,8]),batch['anchor'],batch['mask'],10,edge_support=support)
    assert pred['edge_logits'].shape[-1]==model.edge_classes-1
    assert not schema['bond_head_includes_no_edge']
    assert np.isclose(sum(schema['bond_marginal']),1.)
    assert torch.equal(pred['edge_logits'],pred['edge_logits'].transpose(1,2))
    if attributed: assert calls==[int(support.triu(1).sum())]
    assert torch.count_nonzero(pred['edge_logits'][~support])==0
    loss,parts=losses(pred,batch,cfg['loss_weights']);loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(v) for v in parts.values())
    if attributed:
        assert model.edge_head[-1].weight.grad.abs().sum()>0
        assert model.spec_out[-1].weight.grad is not None
    else:
        assert parts['edge']==0


def test_clean_support_does_not_leak_to_summary_or_spectral_heads():
    model,batch,_,_=small_model();model.eval()
    args=(batch['x'],batch['e'],batch['z'],torch.tensor([8,8,8]),batch['anchor'],batch['mask'],10)
    a=model(*args,edge_support=batch['e']>0)
    b=model(*args,edge_support=torch.zeros_like(batch['e'],dtype=torch.bool))
    for key in ('clean_spectrum','node_logits','graphlet_logits','clustering_logits','graphlet_mass','orbit_log_mean'):
        torch.testing.assert_close(a[key],b[key])


def test_bond_loss_ignores_absence_padding_and_diagonal():
    model,batch,cfg,_=small_model()
    pred=model(batch['x'],batch['e'],batch['z'],torch.tensor([5,5,5]),batch['anchor'],batch['mask'],10,edge_support=batch['e']>0)
    _,before=losses(pred,batch,cfg['loss_weights'])
    altered={**pred,'edge_logits':pred['edge_logits'].clone()}
    altered['edge_logits'][batch['e']==0]=torch.tensor([1e5,-1e5,2e5])
    _,after=losses(altered,batch,cfg['loss_weights'])
    torch.testing.assert_close(before['edge'],after['edge'])


def test_bond_sampling_skips_all_absent_pairs(monkeypatch):
    mask=torch.tensor([[1,1,1,0],[1,0,0,0]],dtype=torch.bool)
    support=torch.zeros(2,4,4,dtype=torch.bool);support[0,0,2]=support[0,2,0]=True
    probs=torch.full((2,4,4,3),float('nan'))
    probs[support]=torch.tensor([0.,1.,0.])
    seen=[];original=torch.multinomial
    def checked(value,*args,**kwargs):
        seen.append(value.shape);return original(value,*args,**kwargs)
    monkeypatch.setattr(torch,'multinomial',checked)
    e=draw_bonds(probs,support,mask,torch.Generator().manual_seed(1))
    assert seen==[torch.Size([1,3])]
    assert e[0,0,2]==2 and torch.equal(e>0,support)
    probs.fill_(float('nan'));support.zero_();seen.clear()
    assert not draw_bonds(probs,support,mask,torch.Generator().manual_seed(2)).any()
    assert not seen


def test_generic_edges_do_not_sample_categories(monkeypatch):
    def forbidden(*args,**kwargs): raise AssertionError('Generic bonds must not be sampled')
    monkeypatch.setattr(torch,'multinomial',forbidden)
    mask=torch.ones((1,3),dtype=torch.bool);support=pair_mask(mask)
    e=draw_bonds(torch.ones(1,3,3,1),support,mask,torch.Generator())
    assert torch.equal(e,support.long())


def test_new_edges_do_not_receive_a_fake_previous_bond(monkeypatch):
    from grapher.models.gdsm_simple.categorical import noise as module
    schedule=cosine_alpha_bar(10);q=MarginalNoise([.2,.3,.5],schedule)
    current=torch.zeros(1,3,3,dtype=torch.long);current[0,0,1]=current[0,1,0]=2
    mask=torch.ones((1,3),dtype=torch.bool);support=pair_mask(mask)
    p=torch.tensor([.8,.1,.1]).expand(1,3,3,3).clone()
    t=torch.tensor([8]);s=torch.tensor([3]);captured=[]
    monkeypatch.setattr(module,'draw_bonds',lambda probs,*args: captured.append(probs.clone()) or support.long())
    reverse_bonds(p,current,support,q,t,s,mask,torch.Generator())
    expected=schedule[3]*p[0,0,2]+(1-schedule[3])*q.marginal
    torch.testing.assert_close(captured[0][0,0,2],expected)
    retained=q.reverse_probs(p[0,0,1][None],torch.tensor([1]),t,s)[0]
    torch.testing.assert_close(captured[0][0,0,1],retained)
    assert not torch.allclose(expected,retained)
    captured.clear()
    reverse_bonds(p,current,support,q,t,torch.tensor([0]),mask,torch.Generator())
    torch.testing.assert_close(captured[0][support],p[support])


def test_factorized_forward_has_correct_clean_endpoint_and_storage_labels():
    a=cosine_alpha_bar(10);xn=MarginalNoise([.5,.5],a);bn=MarginalNoise([.2,.3,.5],a);tn=MarginalNoise([.7,.3],a)
    x=torch.tensor([[0,1,0]]);e=torch.tensor([[[0,2,0],[2,0,3],[0,3,0]]]);mask=torch.ones_like(x,dtype=torch.bool)
    xx,ee=forward_bond_graph(xn,bn,tn,x,e,torch.tensor([0]),mask,torch.Generator().manual_seed(4))
    assert torch.equal(xx,x) and torch.equal(ee,e)
    xx,ee=forward_bond_graph(xn,bn,tn,x,e,torch.tensor([10]),mask,torch.Generator().manual_seed(5))
    assert (ee>=0).all() and (ee<=3).all() and torch.equal(ee,ee.transpose(1,2))


def test_spectral_scores_can_change_topology_without_changing_degrees():
    e=torch.zeros(1,4,4,dtype=torch.long)
    e[0,0,1]=e[0,1,0]=2;e[0,2,3]=e[0,3,2]=3
    score=torch.zeros_like(e,dtype=torch.float)
    score[0,0,2]=score[0,2,0]=score[0,1,3]=score[0,3,1]=10
    cfg={**DEFAULTS['topology'],'mode':'spectral','proposal_budget':64}
    mask=torch.ones(1,4,dtype=torch.bool);d=(e>0).sum(-1)
    out,diag=decode_topology(score,e,mask,d,cfg,np.random.default_rng(42))
    assert not torch.equal(out,e>0)
    assert torch.equal(out.sum(-1),d)
    assert diag[0]['accepted_swaps']==1 and diag[0]['spectral_score_gain']==20.
    assert not out.diagonal(dim1=-2,dim2=-1).any()
    # The result depends on spectral scores, not current bond colour equality.
    recolored=e.clone();recolored[e>0]=1
    same,_=decode_topology(score,recolored,mask,d,cfg,np.random.default_rng(42))
    assert torch.equal(out,same)


@pytest.mark.parametrize('graph',[nx.path_graph(6),nx.cycle_graph(6),nx.empty_graph(1),nx.empty_graph(5),nx.complete_graph(5)])
def test_feasible_initialization_and_many_random_spectral_steps(graph):
    d=np.array([graph.degree(i) for i in range(len(graph))]);n=len(d)
    cfg={**DEFAULTS['topology'],'mode':'spectral','proposal_budget':24}
    a,_=initial_support(d,cfg,np.random.default_rng(12));np.testing.assert_array_equal(a.sum(1),d)
    mask=torch.ones(1,n,dtype=torch.bool);degrees=torch.as_tensor(d)[None];e=torch.as_tensor(a)[None].long()
    rng=np.random.default_rng(13);torch_rng=torch.Generator().manual_seed(14)
    for _ in range(12):
        score=torch.randn(1,n,n,generator=torch_rng)
        support,_=decode_topology(score,e,mask,degrees,cfg,rng)
        probs=torch.rand(1,n,n,3,generator=torch_rng)
        e=draw_bonds(probs,support,mask,torch_rng)
        assert torch.equal((e>0).sum(-1),degrees)


def test_threshold_mode_only_guarantees_bond_support():
    e=torch.ones(1,4,4,dtype=torch.long)-torch.eye(4,dtype=torch.long)[None]
    cfg={**DEFAULTS['topology'],'mode':'spectral','decoder':'threshold'}
    mask=torch.ones(1,4,dtype=torch.bool);d=(e>0).sum(-1)
    support,diags=decode_topology(torch.zeros_like(e,dtype=torch.float),e,mask,d,cfg,np.random.default_rng(0))
    assert not support.any() and not torch.equal(support.sum(-1),d)
    assert not diags[0]['degree_guarantee']


def test_degree_decoder_fails_loudly_on_invalid_input():
    e=torch.zeros(1,3,3,dtype=torch.long);mask=torch.ones(1,3,dtype=torch.bool)
    with pytest.raises(AssertionError,match='fixed indexed'):
        decode_topology(e.float(),e,mask,torch.ones(1,3,dtype=torch.long),DEFAULTS['topology'],np.random.default_rng(0))
    with pytest.raises(FloatingPointError):
        decode_topology(e.float()+float('nan'),e,mask,e.sum(-1),DEFAULTS['topology'],np.random.default_rng(0))


@pytest.mark.parametrize('d',[[2]*6,[3,3,2,2,2,2],[1,1,1,1],[0,1,1]])
def test_connected_initialization_or_explicit_impossibility(d):
    cfg={**DEFAULTS['topology'],'preserve_connectivity':True}
    if min(d)==0 or sum(d)<2*(len(d)-1):
        with pytest.raises(ValueError,match='connected realization'):initial_support(d,cfg,np.random.default_rng(42))
    else:
        a,_=initial_support(d,cfg,np.random.default_rng(42))
        assert nx.is_connected(nx.from_numpy_array(a))
        np.testing.assert_array_equal(a.sum(1),d)


@pytest.fixture(scope='module',params=[False,True],ids=['generic','attributed'])
def trained(request,tmp_path_factory):
    attributed=request.param;root=tmp_path_factory.mktemp('spectral_topology')
    folder=root/'data/toy';folder.mkdir(parents=True)
    train=[labelled(nx.cycle_graph(6)),labelled(nx.path_graph(6)),labelled(nx.complete_graph(4)),labelled(nx.path_graph(3))]
    for split,graphs in [('train',train),('val',[labelled(nx.wheel_graph(6)),labelled(nx.path_graph(4))]),('test',[labelled(nx.ladder_graph(3))])]:
        with (folder/(split+'.pkl')).open('wb') as f:pickle.dump(graphs,f)
    run=RunSpec('gdsm_simple','community_small','spectral_topology',42,root/'runs')
    wrapper=GDSMSimpleWrapper();req=TrainRequest(run,DatasetReference('community_small',root/'data','toy'),options=options(attributed))
    art=wrapper.train(req)
    gen=wrapper.generate(GenerateRequest(run,art.checkpoint_path,6,73))
    return wrapper,req,art,gen


def load(path):
    with open(path,'rb') as f:return pickle.load(f)


def test_managed_train_generate_audit_and_all_intermediate_degrees(trained):
    _,req,art,gen=trained
    state=torch.load(art.checkpoint_path,map_location='cpu',weights_only=False)
    assert state['format']==SPECTRAL_TOPOLOGY_FORMAT
    assert state['model_config']['bond_only']
    assert len(state['schema']['bond_marginal'])==state['model_config']['edge_classes']-1
    assert all(np.isfinite(h['train_loss']) and np.isfinite(h['val_loss']) for h in state['history'])
    m=json.loads(gen.manifest_path.read_text());a=audit(gen.generation_dir)
    assert m['decode']['exact_indexed_degree_guarantee']
    assert not m['decode']['bond_head_includes_no_edge']
    assert m['diagnostics']['prior_indexed_degree_preservation_rate']==1.
    assert m['diagnostics']['categorical_degree_change_steps_mean']==0
    assert m['diagnostics']['spectral_degree_change_steps_mean']==0
    assert a['status']=='passed' and a['saved_intermediate_degrees_verified']
    assert a['bond_support_verified_against_final_pre_rewire_graph']
    assert a['connectedness_rate']==1.
    ds=load(gen.generation_dir/'sampled_degree_sequences.pkl')
    traces=load(gen.generation_dir/'categorical_trajectories.pkl')
    assert len(traces)==6
    for d,trace in zip(ds,traces):
        assert len(trace)==4
        for step in trace:np.testing.assert_array_equal((step['edge_categories']>0).sum(1),d)


def test_managed_spectral_only_without_structural_guidance(trained):
    wrapper,req,art,_=trained
    gen=wrapper.generate(GenerateRequest(req.run,art.checkpoint_path,2,74,generation_id='no_summary_swaps',options={
        'extensions':{'attributed_categorical':{'guidance':{'enabled':False}}}}))
    assert audit(gen.generation_dir)['prior_indexed_degree_preservation_rate']==1.


def test_generation_can_choose_threshold_but_not_legacy_categorical_head(trained):
    wrapper,req,art,_=trained
    gen=wrapper.generate(GenerateRequest(req.run,art.checkpoint_path,3,74,generation_id='threshold',options={
        'extensions':{'attributed_categorical':{'topology':{'decoder':'threshold','preserve_connectivity':False},'guidance':{'enabled':False}}}}))
    a=audit(gen.generation_dir)
    assert a['status']=='passed' and not a['indexed_degree_guaranteed']
    assert a['bond_support_verified_against_final_pre_rewire_graph']
    state=torch.load(art.checkpoint_path,map_location='cpu',weights_only=False)
    op=wrapper._options(req);op['extensions']['attributed_categorical']['topology']['mode']='categorical'
    with pytest.raises(ValueError,match='retraining'):validate_generation(state,op)


def test_old_checkpoints_cannot_silently_enter_spectral_mode():
    state={'format':'gdsm_spectral_categorical_checkpoint_v1'}
    with pytest.raises(ValueError,match='retraining'):validate_generation(state,options())


@pytest.mark.parametrize('key,value',[('mode','invalid'),('decoder','round'),('max_swaps_per_step',-1),
                                    ('proposal_budget',0),('threshold',float('nan')),('preserve_connectivity',1),
                                    ('initial_random_swaps_per_edge',-1)])
def test_invalid_topology_configuration_rejected(key,value):
    op=options();op['extensions']['attributed_categorical']['topology'][key]=value
    with pytest.raises(ValueError):resolve(op)


@pytest.mark.parametrize('dataset',['community_small','ego_small','qm9','zinc'])
@pytest.mark.parametrize('seed',[41,42,43,44,45])
def test_explicit_profiles_resolve_and_keep_seed_specific_priors(dataset,seed):
    path=Path(f'configs/experiments/gdsm_spectral_degree_explicit/{dataset}_seed_{seed}.yaml')
    req=TrainRequest(RunSpec('gdsm_simple',dataset,'check',seed),DatasetReference(dataset),config_path=path)
    op=GDSMSimpleWrapper()._options(req);cfg=resolve(op)
    assert cfg['topology']['mode']=='spectral' and cfg['topology']['decoder']=='degree_preserving'
    assert cfg['topology']['preserve_connectivity'] and cfg['graphlets']['sizes']==[3,4,5]
    assert f'/seed_{seed}/' in cfg['initialization']['degree_generator']['checkpoint_path']
    assert op['train']['max_train_graphs'] is None

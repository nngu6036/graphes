"""Invariant, masking, training, integration and compatibility regression tests."""
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
from grapher.models.gdsm_simple.wrapper import GDSMSimpleWrapper
from grapher.models.gdsm_simple.categorical.config import DEFAULTS, resolve
from grapher.models.gdsm_simple.categorical.topology import (initial_topology, random_swaps,
    spectral_topology_step, training_topology)
from grapher.models.gdsm_simple.categorical.noise import (MarginalNoise, cosine_alpha_bar,
    pair_mask, draw_bonds, bond_forward_probs, bond_reverse_probs)
from grapher.models.gdsm_simple.categorical.model import SpectralCategoricalDenoiser, losses
from grapher.models.gdsm_simple.categorical.pipeline import (prepare_data, _make_model_config,
    _loss_batch, validate_generation, FORMAT)
from grapher.models.gdsm_simple.categorical.data import collate
from grapher.models.gdsm_simple.categorical.evaluation import audit

torch.set_num_threads(1)


def opts(typed=True):
    cfg=copy.deepcopy(DEFAULTS)
    cfg['categories']={'node_attribute':'atom' if typed else None,'edge_attribute':'bond' if typed else None}
    cfg['topology'].update(mode='spectral_degree',require_connected=True,every=1,
                           train_swap_attempts_per_edge=5.,initial_swap_attempts_per_edge=5.,
                           proposal_budget=12,valid_candidate_budget=8,max_steps_per_event=1)
    cfg['guidance'].update(every=1,start_fraction=1.,proposal_budget=12,valid_candidate_budget=8,max_steps_per_event=1)
    cfg['guidance']['weights']['edge']=0.
    cfg['graphlets'].update(sizes=[3,4,5],size_weights=[1.,1.,1.],clustering_bins=10)
    cfg['save_trajectory']=True
    return {'train':{'epochs':2,'batch_size':3,'validation_every':1,'log_every':1},
            'model':{'max_nodes':6,'hidden_dim':16,'num_layers':1,'num_heads':2,'ff_dim':32,'dropout':0.},
            'diffusion':{'steps':10},'sample':{'steps':4},'runtime':{'device':'cpu'},
            'generation_batch_size':3,'extensions':{'attributed_categorical':cfg}}


def graph_list(typed=True):
    graphs=[nx.cycle_graph(6),nx.path_graph(6),nx.star_graph(5),nx.complete_graph(4),nx.path_graph(3)]
    if typed:
        for g in graphs:
            nx.set_node_attributes(g,{v:int(v%2) for v in g},'atom')
            nx.set_edge_attributes(g,{uv:1+i%2 for i,uv in enumerate(g.edges())},'bond')
    return graphs


def load(root,name):
    with (root/name).open('rb') as f: return pickle.load(f)


@pytest.mark.parametrize('degrees',[[0],[0,0,0],[2,2,2,2,2,2],[1,3,2,2,2],[4,1,1,1,1],[4,4,4,4,4]])
def test_initial_topology_exact_even_for_rigid_degrees_and_isolates(degrees):
    cfg=copy.deepcopy(DEFAULTS['topology'])
    a,_=initial_topology(degrees,cfg,np.random.default_rng(13))
    np.testing.assert_array_equal(a.sum(1),degrees)
    np.testing.assert_array_equal(a,a.T)
    assert not a.diagonal().any()


def test_connected_realizations_over_small_graph_atlas():
    cfg={**DEFAULTS['topology'],'require_connected':True,'initial_swap_attempts_per_edge':0.}
    for g in nx.graph_atlas_g():
        if not 1<=len(g)<=6 or not nx.is_connected(g): continue
        d=np.array([g.degree(v) for v in g])[::-1]
        a,_=initial_topology(d,cfg,np.random.default_rng(1))
        np.testing.assert_array_equal(a.sum(1),d)
        assert nx.is_connected(nx.from_numpy_array(a))


@pytest.mark.parametrize('d',[[1,1,1,1],[2,2,2,0],[-1,1],[1.5,1.5],[3,1,0]])
def test_impossible_connected_or_invalid_sequences_fail_without_repair(d):
    with pytest.raises(ValueError):
        initial_topology(d,{**DEFAULTS['topology'],'require_connected':True},np.random.default_rng(4))


def test_spectral_decoder_changes_edges_without_changing_indexed_degrees():
    a=nx.to_numpy_array(nx.cycle_graph(6),dtype=np.int64)
    target=a.copy();target[0,1]=target[1,0]=target[3,4]=target[4,3]=0
    target[0,3]=target[3,0]=target[1,4]=target[4,1]=1
    cfg={**DEFAULTS['topology'],'proposal_budget':-1,'valid_candidate_budget':-1,'max_steps_per_event':2}
    out,diag=spectral_topology_step(a,target,cfg,np.random.default_rng(7))
    assert diag['accepted_steps']>0 and diag['final_pair_score']>diag['initial_pair_score']
    np.testing.assert_array_equal(out.sum(1),a.sum(1))
    np.testing.assert_array_equal(out,target)
    assert nx.is_connected(nx.from_numpy_array(out))


def test_equal_adjacency_eigenvalues_do_not_determine_degree_sequence():
    star=nx.to_numpy_array(nx.star_graph(4))
    cycle=nx.to_numpy_array(nx.disjoint_union(nx.cycle_graph(4),nx.empty_graph(1)))
    np.testing.assert_allclose(np.linalg.eigvalsh(star),np.linalg.eigvalsh(cycle),atol=1e-14)
    assert not np.array_equal(np.sort(star.sum(1)),np.sort(cycle.sum(1)))


@pytest.mark.parametrize('classes',[1,2,4])
def test_bond_sampler_only_visits_existing_unordered_edges(classes):
    mask=torch.tensor([[1,1,1,1],[1,0,0,0]],dtype=torch.bool)
    support=torch.zeros((2,4,4),dtype=torch.bool)
    support[0,0,1]=support[0,1,0]=support[0,1,2]=support[0,2,1]=True
    # NaNs on absent pairs prove they are never sent to categorical sampling.
    probs=torch.full((2,4,4,classes),float('nan'))
    probs[support]=1/classes
    out=draw_bonds(probs,support,mask,torch.Generator().manual_seed(5))
    assert torch.equal(out>0,support) and torch.equal(out,out.transpose(1,2))
    assert ((out[support]>=1)&(out[support]<=classes)).all()
    again=draw_bonds(probs,support,mask,torch.Generator().manual_seed(5))
    assert torch.equal(out,again)


def test_bond_birth_has_no_fictitious_posterior_observation():
    schedule=cosine_alpha_bar(10);noise=MarginalNoise([.3,.7],schedule)
    p=torch.tensor([[[[.8,.2],[.25,.75]],[[.25,.75],[.8,.2]]]])
    e=torch.tensor([[[0,2],[2,0]]]);t=torch.tensor([10]);s=torch.tensor([3])
    actual=bond_reverse_probs(noise,p,e,t,s)
    born=schedule[3]*p+(1-schedule[3])*noise.marginal
    torch.testing.assert_close(actual[e==0],born[e==0])
    expected=noise.reverse_probs(p,(e-1).clamp_min(0),t,s)
    torch.testing.assert_close(actual[e>0],expected[e>0])
    torch.testing.assert_close(bond_reverse_probs(noise,p,e,t,torch.tensor([0])),p)


def test_training_support_is_corrupted_but_degree_exact():
    e=torch.tensor(nx.to_numpy_array(nx.cycle_graph(6),dtype=np.int64))[None]
    mask=torch.ones((1,6),dtype=torch.bool);cfg=opts()['extensions']['attributed_categorical']['topology']
    changed=False
    for seed in range(10):
        a=training_topology(e,mask,torch.tensor([10]),10,cfg,torch.Generator().manual_seed(seed))
        assert torch.equal(a.sum(2),(e>0).sum(2))
        changed|=not torch.equal(a,e>0)
    assert changed


@pytest.mark.parametrize('typed',[True,False])
def test_joint_training_gradients_bond_dimensions_and_no_mask_leakage(typed):
    op=opts(typed);cfg=resolve(op);graphs=graph_list(typed)
    records,_,vocab,basis,_,meta=prepare_data(graphs,graphs,6,cfg,42)
    model=SpectralCategoricalDenoiser(**_make_model_config(op,cfg,vocab,basis,6))
    assert model.output_edge_classes==vocab.num_edge_categories-1
    assert (model.edge_head is None)==(not typed)
    assert len(meta['bond_marginal'])==vocab.num_edge_categories-1
    np.testing.assert_allclose(sum(meta['bond_marginal']),1.)
    schedule=cosine_alpha_bar(10);xn=MarginalNoise(meta['node_marginal'],schedule);en=MarginalNoise(meta['bond_marginal'],schedule)
    loss,parts=_loss_batch(model,records,basis,cfg,schedule,xn,en,torch.Generator().manual_seed(2),'cpu',permutations=True)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.spec_out.parameters())
    if typed: assert model.edge_head[-1].weight.grad.abs().sum()>0
    else: assert float(parts['edge'])==0.
    model.eval();batch=collate(records,basis,device='cpu');t=torch.full((len(records),),5)
    args=(batch['x'],batch['e'],batch['z'],t,batch['anchor'],batch['mask'],10)
    pred=model(*args);query=pair_mask(batch['mask'])
    expanded=model(*args,edge_query_mask=query)
    for key in ('clean_spectrum','spectral_scores','node_logits','graphlet_logits','graphlet_mass','clustering_logits','orbit_log_mean'):
        torch.testing.assert_close(pred[key],expanded[key],atol=2e-6,rtol=2e-6)
    active=pred['edge_prediction_mask']
    torch.testing.assert_close(pred['edge_logits'][active],expanded['edge_logits'][active])


def test_bond_loss_is_only_clean_real_edges_and_permutation_equivariant():
    op=opts();cfg=resolve(op);graphs=graph_list()
    rows,_,vocab,basis,_,_=prepare_data(graphs,graphs,6,cfg,42)
    model=SpectralCategoricalDenoiser(**_make_model_config(op,cfg,vocab,basis,6)).eval()
    batch=collate(rows[:1],basis,device='cpu');t=torch.tensor([5])
    pred=model(batch['x'],batch['e'],batch['z'],t,batch['anchor'],batch['mask'],10)
    _,parts=losses(pred,batch,cfg['loss_weights'])
    valid=torch.triu(batch['e']>0,diagonal=1)
    expected=torch.nn.functional.cross_entropy(pred['edge_logits'][valid],batch['e'][valid]-1)
    torch.testing.assert_close(parts['edge'],expected)
    pred2=dict(pred);pred2['edge_logits']=pred['edge_logits'].clone()
    pred2['edge_logits'][batch['e']==0]=torch.tensor([1e5,-1e5])
    torch.testing.assert_close(losses(pred2,batch,cfg['loss_weights'])[1]['edge'],expected)
    perm=torch.tensor([4,1,5,0,3,2]);e=batch['e'][:,perm][:,:,perm]
    other=model(batch['x'][:,perm],e,batch['z'],t,batch['anchor'],batch['mask'],10)
    torch.testing.assert_close(other['edge_logits'],pred['edge_logits'][:,perm][:,:,perm],atol=2e-5,rtol=2e-5)


@pytest.fixture(scope='module',params=[False,True],ids=['generic','attributed'])
def trained_degree(request,tmp_path_factory):
    typed=request.param;root=tmp_path_factory.mktemp('degree_typed' if typed else 'degree_generic')
    folder=root/'data/toy';folder.mkdir(parents=True);graphs=graph_list(typed)
    for split in ('train','val','test'):
        with (folder/f'{split}.pkl').open('wb') as f:pickle.dump(graphs,f)
    wrapper=GDSMSimpleWrapper();run=RunSpec('gdsm_simple','community_small','degree',42,root/'runs')
    req=TrainRequest(run,DatasetReference('community_small',root/'data','toy'),options=opts(typed))
    art=wrapper.train(req)
    gen=wrapper.generate(GenerateRequest(run,art.checkpoint_path,9,42,generation_id='first'))
    return wrapper,req,art,gen


def test_end_to_end_degrees_all_timesteps_and_audit(trained_degree):
    _,_,art,gen=trained_degree;root=gen.generation_dir
    d=load(root,'sampled_degree_sequences.pkl');initial=load(root,'initial_graphs.pkl');final=load(root,'base_graphs.pkl')
    trajectories=load(root,'categorical_trajectories.pkl')
    for seq,a,g,tr in zip(d,initial,final,trajectories):
        np.testing.assert_array_equal([a.degree(i) for i in range(len(a))],seq)
        np.testing.assert_array_equal([g.degree(i) for i in range(len(g))],seq)
        assert len(tr)==4
        for step in tr:np.testing.assert_array_equal((step['edge_categories']>0).sum(1),seq)
        assert nx.is_connected(g)
    report=audit(root)
    assert report['indexed_prior_degree_preservation_rate']==1.
    assert report['all_saved_trajectory_degrees_verified']
    assert not report['degree_changes_are_allowed_not_required']
    assert report['recorded_categorical_degree_change_steps_mean']==0.
    train_manifest=json.loads(art.manifest_path.read_text())
    assert train_manifest['contract']['original_degrees_enforced']
    assert not train_manifest['contract']['edge_query_mask_is_encoder_input']
    manifest=json.loads(gen.manifest_path.read_text())
    assert manifest['decode']['ordinary_indexed_degrees_preserved_every_step']
    assert not manifest['decode']['bond_output_includes_no_edge']
    assert all('bond_probs' in p and 'edge_probs' not in p for p in load(root,'predicted_summaries.pkl'))


def test_reproducible_and_degree_exact_without_structural_guidance(trained_degree):
    wrapper,req,art,gen=trained_degree
    repeat=wrapper.generate(GenerateRequest(req.run,art.checkpoint_path,9,42,generation_id='repeat'))
    for a,b in zip(load(gen.generation_dir,'base_graphs.pkl'),load(repeat.generation_dir,'base_graphs.pkl')):
        assert nx.utils.graphs_equal(a,b)
    other=wrapper.generate(GenerateRequest(req.run,art.checkpoint_path,3,45,generation_id='no_structure',
        options={'extensions':{'attributed_categorical':{'guidance':{'enabled':False}}}}))
    assert audit(other.generation_dir)['indexed_prior_degree_preservation_rate']==1.
    diag=json.loads((other.generation_dir/'rewiring_diagnostics.json').read_text())
    assert all(not g['guidance_events'] for g in diag['graphs'])
    assert all(g['topology_events'] for g in diag['graphs'])


def test_incompatible_checkpoint_modes_rejected(trained_degree):
    wrapper,req,art,_=trained_degree
    state=torch.load(art.checkpoint_path,map_location='cpu',weights_only=False)
    op=wrapper._options(req)
    op['extensions']['attributed_categorical']['topology']['mode']='categorical'
    with pytest.raises(ValueError,match='retrain'):validate_generation(state,op)
    legacy=copy.deepcopy(state);legacy['categorical_config'].pop('topology');legacy['model_config'].pop('bond_only')
    with pytest.raises(ValueError,match='retrain'):validate_generation(legacy,wrapper._options(req))


@pytest.mark.parametrize('change',[
    {'mode':'invalid'},{'train_swap_attempts_per_edge':-1},{'initial_swap_attempts_per_edge':float('nan')},
    {'every':0},{'max_steps_per_event':-1},{'proposal_budget':0},{'start_fraction':2},
    {'require_connected':True,'preserve_connectivity_if_connected':False}])
def test_config_rejects_bad_hard_degree_settings(change):
    op=opts();op['extensions']['attributed_categorical']['topology'].update(change)
    with pytest.raises(ValueError):resolve(op)


def test_cannot_use_bond_probabilities_as_edge_existence_scores():
    op=opts();op['extensions']['attributed_categorical']['guidance']['weights']['edge']=.1
    with pytest.raises(ValueError,match='edge-existence'):resolve(op)


def test_new_support_queries_reuse_backbone_and_never_classify_non_edges():
    op=opts();cfg=resolve(op);graphs=graph_list()
    rows,_,vocab,basis,_,_=prepare_data(graphs,graphs,6,cfg,42)
    model=SpectralCategoricalDenoiser(**_make_model_config(op,cfg,vocab,basis,6)).eval()
    batch=collate(rows[:1],basis,device='cpu');counts=[]
    hook=model.edge_head.register_forward_pre_hook(lambda module,args: counts.append(len(args[0])))
    pred=model(batch['x'],batch['e'],batch['z'],torch.tensor([5]),batch['anchor'],batch['mask'],10)
    support=batch['e']>0
    proposal=support.clone()
    proposal[0,0,1]=proposal[0,1,0]=proposal[0,3,4]=proposal[0,4,3]=False
    proposal[0,0,3]=proposal[0,3,0]=proposal[0,1,4]=proposal[0,4,1]=True
    requery=model.query_bonds(pred,proposal)
    hook.remove()
    assert counts==[int(torch.triu(support,1).sum()),int(torch.triu(proposal,1).sum())]
    assert torch.equal(requery['edge_prediction_mask'],proposal)
    assert not requery['edge_logits'][~proposal].any()
    assert requery['clean_spectrum'] is pred['clean_spectrum']
    assert requery['graphlet_logits'] is pred['graphlet_logits']


@pytest.mark.parametrize('dataset',['community_small','ego_small','qm9','zinc'])
@pytest.mark.parametrize('seed',[41,42,43,44,45])
def test_explicit_new_configs_preserve_existing_experiment_settings(dataset,seed,tmp_path):
    import yaml
    old_path=Path(f'configs/experiments/gdsm_final_explicit/{dataset}_seed_{seed}.yaml')
    new_path=Path(f'configs/experiments/gdsm_spectral_degree_explicit/{dataset}_seed_{seed}.yaml')
    old=yaml.safe_load(old_path.read_text())['gdsm_simple'];new=yaml.safe_load(new_path.read_text())['gdsm_simple']
    for key in ('train','model','diffusion','sample','generation_batch_size'):
        assert new[key]==old[key]
    before=old['extensions']['attributed_categorical'];after=new['extensions']['attributed_categorical']
    assert after['graphlets']==before['graphlets'] and after['initialization']==before['initialization']
    request=TrainRequest(RunSpec('gdsm_simple',dataset,'cfg',seed),DatasetReference(dataset,tmp_path),config_path=new_path)
    cfg=resolve(GDSMSimpleWrapper()._options(request))
    assert cfg['topology']['mode']=='spectral_degree'
    assert cfg['initialization']['degree_generator']['checkpoint_path'].endswith(f'seed_{seed}/checkpoint.pt')

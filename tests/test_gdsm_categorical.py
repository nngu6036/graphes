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

from grapher.models.base import DatasetReference, RunSpec, TrainRequest, GenerateRequest
from grapher.models.gdsm_simple.wrapper import GDSMSimpleWrapper
from grapher.models.gdsm_simple.categorical.config import DEFAULTS, resolve
from grapher.models.gdsm_simple.categorical.noise import (MarginalNoise,cosine_alpha_bar,pair_mask,draw_graph,spectral_q_sample,spectral_reverse)
from grapher.models.gdsm_simple.categorical.data import (TypedGraphlets3,typed_counts,canonical_triple,encode_graph,decode_graph,collate,permute_aligned)
from grapher.models.gdsm_simple.categorical.model import SpectralCategoricalDenoiser,losses
from grapher.models.gdsm_simple.categorical.pipeline import prepare_data,validate_generation,FORMAT
from grapher.models.gdsm_simple.categorical.spectral import degree_anchor,eigenpairs,spectral_proposal
from grapher.models.gdsm_simple.categorical.refiner import refine
from grapher.rewiring_mlp.attributed.data import GraphCategoryVocabulary

torch.set_num_threads(1)


def labelled(g):
    g=g.copy()
    nx.set_node_attributes(g,{v:'C' if i%2 else 'N' for i,v in enumerate(g.nodes)},'atom')
    nx.set_edge_attributes(g,{uv:1 if i%2 else 2 for i,uv in enumerate(g.edges)},'bond')
    return g


def config():
    cfg=copy.deepcopy(DEFAULTS)
    cfg['categories']={'node_attribute':'atom','edge_attribute':'bond'}
    cfg['graphlets']['clustering_bins']=10
    return cfg


def options():
    return {'train':{'epochs':2,'batch_size':3,'validation_every':1,'log_every':1},
            'model':{'max_nodes':6,'hidden_dim':16,'num_layers':1,'num_heads':2,'ff_dim':32},
            'diffusion':{'steps':10},'sample':{'steps':4},'runtime':{'device':'cpu'},'generation_batch_size':2,
            'extensions':{'attributed_categorical':{**config(),'save_trajectory':True,
                'guidance':{**config()['guidance'],'every':1,'start_fraction':1.,'proposal_budget':12,'valid_candidate_budget':8,'max_steps_per_event':1}}}}


@pytest.mark.parametrize('steps',[2,10,1000])
def test_marginal_schedule_exact_endpoints_and_products(steps):
    a=cosine_alpha_bar(steps); q=MarginalNoise([.7,.2,.1],a)
    assert a[0]==1 and a[-1]==0 and (a[:-1]>a[1:]).all()
    acc=torch.eye(3)
    for t in range(1,len(a)):
        qt=q.matrix(a[t]/a[t-1]); acc=acc@qt
        torch.testing.assert_close(qt.sum(1),torch.ones(3))
        torch.testing.assert_close(q.marginal@qt,q.marginal)
        torch.testing.assert_close(acc,q.matrix(a[t]),atol=2e-5,rtol=2e-5)


@pytest.mark.parametrize('t,s',[(10,9),(10,0),(9,3),(7,6),(1,0)])
def test_exact_posterior_mixture_agrees_with_bruteforce(t,s):
    a=cosine_alpha_bar(10).double();q=MarginalNoise([.7,.2,.1],a)
    p=torch.tensor([[.17,.23,.6],[.8,.12,.08],[.03,.2,.77]],dtype=torch.float64)
    current=torch.tensor([0,1,2]);tv=torch.tensor(t);sv=torch.tensor(s)
    actual=q.reverse_probs(p,current,tv,sv)
    qs,qt,qst=q.matrix(a[s]),q.matrix(a[t]),q.matrix(a[t]/a[s])
    expected=torch.zeros_like(p)
    for row,i in enumerate(current):
        for k,j in itertools.product(range(3),repeat=2):
            expected[row,j]+=p[row,k]*qs[k,j]*qst[j,i]/qt[k,i]
    torch.testing.assert_close(actual,expected,atol=1e-10,rtol=1e-10)
    if s==0:torch.testing.assert_close(actual,p)


def test_forward_marginal_support_and_terminal_independence():
    a=cosine_alpha_bar(10);q=MarginalNoise([.8,.15,.05],a)
    x=torch.arange(3)[None];p=q.forward_probs(x,torch.tensor([5]))
    assert (p>0).all() # insertion, deletion, and category changes all possible
    torch.testing.assert_close(q.forward_probs(x,torch.tensor([10])),q.marginal.expand(1,3,3))
    torch.testing.assert_close(q.forward_probs(x,torch.tensor([0])),torch.eye(3)[None])


def test_pair_sampling_symmetry_masks_and_reproducibility():
    mask=torch.tensor([[1,1,1,1],[1,0,0,0]],dtype=torch.bool)
    px=torch.ones(2,4,2)/2;pe=torch.ones(2,4,4,3)/3
    x,e=draw_graph(px,pe,mask,torch.Generator().manual_seed(22))
    x2,e2=draw_graph(px,pe,mask,torch.Generator().manual_seed(22))
    assert torch.equal(x,x2) and torch.equal(e,e2)
    assert torch.equal(e,e.transpose(1,2)) and (e[~pair_mask(mask)]==0).all()
    assert (x[~mask]==0).all() and not e[1].any()


def test_noise_broadcast_single_category_and_invalid_parameters():
    a=cosine_alpha_bar(5);q=MarginalNoise([1.],a)
    x=torch.zeros(2,3,3,dtype=torch.long);t=torch.tensor([1,5]);s=torch.tensor([0,2])
    p=q.forward_probs(x,t);assert p.shape==(2,3,3,1)
    torch.testing.assert_close(q.reverse_probs(p,x,t,s),torch.ones_like(p))
    with pytest.raises(ValueError):MarginalNoise([0.,1.],a)
    with pytest.raises(ValueError):q.reverse_probs(p,x,s,t)


def test_source_centred_spectral_skip_algebra():
    a=cosine_alpha_bar(10);rng=torch.Generator().manual_seed(17)
    clean=torch.randn(3,5,generator=rng);anchor=torch.randn(3,5,generator=rng);eps=torch.randn(3,5,generator=rng)
    mask=torch.ones(3,5,dtype=torch.bool);mask[0,3:]=False
    t=torch.tensor([10,7,1]);s=torch.tensor([5,2,0])
    noisy=spectral_q_sample(clean,eps,anchor,a,t,mask)
    actual=spectral_reverse(noisy,clean,anchor,a,t,s,mask)
    expected=spectral_q_sample(clean,eps,anchor,a,s,mask)
    torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-6)
    torch.testing.assert_close(noisy[0],(anchor[0]+eps[0])*mask[0])


def brute_counts(x,e):
    out=Counter()
    for i,j,k in itertools.combinations(range(len(x)),3):
        if (e[i,j]>0)+(int(e[i,k]>0))+(int(e[j,k]>0))>=2:
            out[canonical_triple(tuple(map(int,(x[i],x[j],x[k],e[i,j],e[i,k],e[j,k]))))]+=1
    return out


def test_typed_graphlets_exact_on_graph_atlas_and_node_permutations():
    rng=np.random.default_rng(71)
    for g in nx.graph_atlas_g():
        if not 1<=len(g)<=5:continue
        e=nx.to_numpy_array(g,dtype=np.int64);x=rng.integers(0,3,len(g))
        for i,j in g.edges:e[i,j]=e[j,i]=rng.integers(1,3)
        count=typed_counts(x,e)
        assert count==brute_counts(x,e)
        p=rng.permutation(len(g));assert count==typed_counts(x[p],e[p][:,p])


def test_typed_paths_distinguish_centre_and_edge_types_mass_overflow():
    e=nx.to_numpy_array(nx.path_graph(3),dtype=np.int64)
    c1=typed_counts(np.array([0,1,0]),e);c2=typed_counts(np.array([1,0,0]),e)
    assert c1!=c2
    b=TypedGraphlets3(list(c1));h,m=b.summary(np.array([0,1,0]),e)
    assert h[0]==1 and m==1
    h,m=b.summary(np.array([1,0,0]),e);assert h[-1]==1 and m==1
    h,m=b.summary(np.array([0,1,0]),np.zeros_like(e));assert h.sum()==0 and m==0
    single=TypedGraphlets3([]).summary(np.array([0]),np.zeros((1,1),int))
    assert single[1]==0 and single[0].sum()==0


def test_train_only_marginals_vocabulary_and_validation_overflow():
    g=labelled(nx.path_graph(4));v=labelled(nx.complete_graph(4))
    rows,vals,vocab,basis,bank,meta=prepare_data([g],[v],4,config(),42)
    assert len(basis.keys)==len(typed_counts(rows[0]['x'],rows[0]['e']))
    assert basis.encode_counts(vals[0]['counts'],4)[0][-1]==1
    assert sum(meta['node_category_counts'])==4 and sum(meta['edge_category_counts'])==6
    assert meta['edge_category_counts'][0]==3
    assert set(bank)=={4}
    v.nodes[0]['atom']='unseen'
    with pytest.raises(ValueError):prepare_data([g],[v],4,config(),42)


def test_encoding_strict_and_does_not_discard_isolates():
    g=labelled(nx.path_graph(3));g.add_node(99,atom='C')
    vocab=GraphCategoryVocabulary.from_graphs([g],config()['categories'])
    x,e=encode_graph(g,vocab,4);out=decode_graph(x,e,vocab)
    assert len(out)==4 and out.number_of_edges()==2
    assert [out.nodes[i]['atom'] for i in out]==[g.nodes[i]['atom'] for i in g]
    g.add_edge(0,0,bond=1)
    with pytest.raises(ValueError):encode_graph(g,vocab,4)


def test_anchor_is_finite_sorted_soft_and_degree_sensitive():
    u=np.linalg.qr(np.random.default_rng(4).normal(size=(5,5)))[0]
    a,diag=degree_anchor(u,[2,2,2,1,1]);other,_=degree_anchor(u,[4,1,1,1,1])
    assert np.isfinite(a).all() and (np.diff(a)>=0).all() and not np.allclose(a,other)
    assert abs(a.sum())<1e-5 and diag['row_sum_rmse']>0
    np.testing.assert_allclose(degree_anchor(u*np.array([1,-1,1,-1,1]),[2,2,2,1,1])[0],a,atol=1e-6)
    assert degree_anchor(np.eye(1),[0])[0][0]==0


@pytest.mark.parametrize('kind',['empty','cycle','path'])
def test_spectral_proposal_basis_invariance_and_reconstruction(kind):
    g={'empty':nx.empty_graph,'cycle':nx.cycle_graph,'path':nx.path_graph}[kind](6)
    e=torch.tensor(nx.to_numpy_array(g,dtype=np.int64))[None];mask=torch.ones(1,6,dtype=torch.bool)
    values,u=eigenpairs(e,mask);candidate=torch.linspace(-1,1,6)[None]
    score=spectral_proposal(candidate,values,u,mask)
    signs=torch.tensor([1.,-1.,1.,-1.,1.,-1.]);us=u*signs[None,None]
    torch.testing.assert_close(score,spectral_proposal(candidate,values,us,mask),atol=1e-6,rtol=1e-6)
    # Rotate only eigenspaces with numerically equal eigenvalues.
    vals=values[0].numpy();rot=u.clone();start=0
    while start<6:
        end=start+1
        while end<6 and abs(vals[end]-vals[start])<1e-6:end+=1
        q=torch.linalg.qr(torch.randn(end-start,end-start,generator=torch.Generator().manual_seed(start+5)))[0]
        rot[0,:,start:end]=rot[0,:,start:end]@q;start=end
    torch.testing.assert_close(score,spectral_proposal(candidate,values,rot,mask),atol=2e-6,rtol=2e-6)
    torch.testing.assert_close(spectral_proposal(values,values,u,mask),e.float(),atol=2e-6,rtol=2e-6)


def prepared_model():
    graphs=[labelled(nx.path_graph(5)),labelled(nx.cycle_graph(6)),labelled(nx.empty_graph(1))]
    rows,_,v,b,_,_=prepare_data(graphs,graphs,6,config(),42)
    batch=collate(rows,b,device='cpu')
    model=SpectralCategoricalDenoiser(node_classes=v.num_node_categories,edge_classes=v.num_edge_categories,graphlet_classes=b.dimension,
        hidden_dim=16,num_layers=1,num_heads=2,ff_dim=32,clustering_bins=10,max_nodes=6)
    return model,batch,b


def test_all_prediction_heads_have_gradients_and_finite_losses():
    model,batch,_=prepared_model();t=torch.tensor([2,6,10])
    pred=model(batch['x'],batch['e'],batch['z']+.2,t,batch['anchor'],batch['mask'],10)
    loss,parts=losses(pred,batch,config()['loss_weights']);loss.backward()
    assert torch.isfinite(loss) and len(parts)==7
    for head in (model.node_head,model.edge_head,model.spec_out,model.graphlet_head,model.mass_head,model.clustering_head,model.orbit_head):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
    torch.testing.assert_close(pred['edge_logits'],pred['edge_logits'].transpose(1,2))
    torch.testing.assert_close(pred['clean_spectrum'].sum(1),torch.zeros(3),atol=2e-6,rtol=2e-6)


def test_node_permutation_equivariance_no_spectral_rank_node_confusion():
    model,batch,_=prepared_model();model.eval();t=torch.tensor([5,5,5])
    pred=model(batch['x'],batch['e'],batch['z'],t,batch['anchor'],batch['mask'],10)
    altered={k:v.clone() for k,v in batch.items()};perms=[]
    for i,n in enumerate(batch['mask'].sum(1).tolist()):
        p=torch.randperm(n,generator=torch.Generator().manual_seed(19+i));perms.append(p)
        altered['x'][i,:n]=batch['x'][i,:n][p];altered['e'][i,:n,:n]=batch['e'][i,:n,:n][p][:,p]
    out=model(altered['x'],altered['e'],batch['z'],t,batch['anchor'],batch['mask'],10)
    for key in ('clean_spectrum','graphlet_logits','clustering_logits','orbit_log_mean','graphlet_mass'):
        torch.testing.assert_close(out[key],pred[key],atol=2e-5,rtol=2e-5)
    for i,p in enumerate(perms):
        n=len(p);torch.testing.assert_close(out['node_logits'][i,:n],pred['node_logits'][i,:n][p],atol=2e-5,rtol=2e-5)
        torch.testing.assert_close(out['edge_logits'][i,:n,:n],pred['edge_logits'][i,:n,:n][p][:,p],atol=2e-5,rtol=2e-5)


def test_padding_does_not_affect_active_predictions():
    model,batch,_=prepared_model();model.eval()
    pred=model(batch['x'],batch['e'],batch['z'],torch.tensor([5,5,5]),batch['anchor'],batch['mask'],10)
    n=5
    out=model(batch['x'][:1,:n],batch['e'][:1,:n,:n],batch['z'][:1,:n],torch.tensor([5]),batch['anchor'][:1,:n],batch['mask'][:1,:n],10)
    for k in ('graphlet_logits','clustering_logits','orbit_log_mean'):
        torch.testing.assert_close(out[k][0],pred[k][0],atol=2e-5,rtol=2e-5)
    torch.testing.assert_close(out['node_logits'][0],pred['node_logits'][0,:n],atol=2e-5,rtol=2e-5)


def test_categorical_only_ablation_removes_spectral_dependence():
    model,batch,_=prepared_model();model.eval();model.spectral_conditioning=False
    def call(z,a):return model(batch['x'],batch['e'],z,torch.tensor([5,5,5]),a,batch['mask'],10)
    p=call(batch['z'],batch['anchor']);q=call(batch['z']+3,batch['anchor']-10)
    for k in ('node_logits','edge_logits','graphlet_logits','graphlet_mass','clustering_logits','orbit_log_mean'):
        torch.testing.assert_close(p[k],q[k])


def test_oracle_refinement_improves_typed_structure_event_local_invariants():
    g=nx.Graph();g.add_nodes_from(range(8));g.add_edges_from([(0,4),(1,2),(1,4),(1,5),(1,7),(2,4),(3,5),(4,7),(5,6),(6,7)])
    e=nx.to_numpy_array(g,dtype=np.int64);target=e.copy()
    for u,v in [(0,4),(5,6)]:target[u,v]=target[v,u]=0
    for u,v in [(0,5),(4,6)]:target[u,v]=target[v,u]=1
    x=np.arange(8)%2;b=TypedGraphlets3(sorted(set(typed_counts(x,e))|set(typed_counts(x,target))))
    from grapher.models.gdsm_simple.categorical.data import topology_summary
    h,m=b.summary(x,target);c,o=topology_summary(target,10)
    goal={'histogram':h,'mass':m,'clustering':c,'orbit':o,'spectrum':np.linalg.eigvalsh(target)/8**.5,'edge_probs':np.eye(2)[target]}
    cfg=copy.deepcopy(config()['guidance']);cfg.update(proposal_budget=-1,valid_candidate_budget=-1,max_steps_per_event=1)
    cfg['weights'].update(spectral=0.,edge=0.)
    out,diag=refine(x,e,goal,b,10,cfg,np.random.default_rng(5))
    assert diag['accepted_steps']==1
    np.testing.assert_array_equal((out>0).sum(1),(e>0).sum(1))
    assert diag['degree_preserved'] and diag['typed_degrees_preserved']


@pytest.mark.parametrize('g',[nx.empty_graph(1),nx.empty_graph(5),nx.complete_graph(5)])
def test_refinement_no_action_safe(g):
    from grapher.models.gdsm_simple.categorical.data import topology_summary
    e=nx.to_numpy_array(g,dtype=int);x=np.zeros(len(g),int);b=TypedGraphlets3(sorted(typed_counts(x,e)))
    h,m=b.summary(x,e);c,o=topology_summary(e,10)
    target={'histogram':h,'mass':m,'clustering':c,'orbit':o,'spectrum':np.linalg.eigvalsh(e)/len(g)**.5,'edge_probs':np.ones((*e.shape,2))*.5}
    out,diag=refine(x,e,target,b,10,config()['guidance'],np.random.default_rng(42))
    assert diag['accepted_steps']==0;np.testing.assert_array_equal(out,e)


@pytest.fixture(scope='module')
def trained_categorical(tmp_path_factory):
    root=tmp_path_factory.mktemp('categorical');folder=root/'data/toy';folder.mkdir(parents=True)
    train=[labelled(nx.cycle_graph(6)),labelled(nx.path_graph(6)),labelled(nx.star_graph(5)),labelled(nx.complete_graph(4)),labelled(nx.path_graph(3))]
    for split,graphs in [('train',train),('val',[labelled(nx.wheel_graph(6)),labelled(nx.path_graph(4))]),('test',[labelled(nx.ladder_graph(3))])]:
        with (folder/(split+'.pkl')).open('wb') as f:pickle.dump(graphs,f)
    run=RunSpec('gdsm_simple','community_small','categorical',42,root/'runs')
    wrapper=GDSMSimpleWrapper();request=TrainRequest(run,DatasetReference('community_small',root/'data','toy'),options=options())
    artifacts=wrapper.train(request)
    generation=wrapper.generate(GenerateRequest(run,artifacts.checkpoint_path,6,42,generation_id='first',options={'runtime':{'device':'cpu'}}))
    return wrapper,request,artifacts,generation


def test_training_checkpoint_and_marginal_contract(trained_categorical):
    wrapper,request,artifacts,gen=trained_categorical
    state=torch.load(artifacts.checkpoint_path,map_location='cpu',weights_only=False)
    assert state['format']==FORMAT
    assert state['schema']['edge_category_counts'][0]>0
    assert len(state['history'])==2
    assert artifacts.checkpoint_path==wrapper.train(request).checkpoint_path
    manifest=json.loads(artifacts.manifest_path.read_text())
    assert not manifest['test_used_for_training'] and not manifest['contract']['original_degrees_enforced']
    with pytest.raises(ValueError):validate_generation({'format':'legacy'},wrapper._options(request))
    changed=copy.deepcopy(manifest['options']);changed['extensions']['attributed_categorical']['noise']['pseudocount']=.4
    with pytest.raises(ValueError,match='retrain'):validate_generation(state,changed)


def readp(folder,name):
    with (folder/name).open('rb') as f:return pickle.load(f)


def test_generation_recomputes_eigenvectors_after_every_categorical_update(trained_categorical):
    _,_,_,gen=trained_categorical;p=gen.generation_dir
    graphs=readp(p,'base_graphs.pkl');vals=readp(p,'final_adjacency_eigenvalues.pkl');us=readp(p,'final_eigenvectors.pkl')
    diag=json.loads((p/'rewiring_diagnostics.json').read_text())
    assert len(graphs)==6 and diag['aggregate']['basis_updates_per_graph']==5
    assert any(row['categorical_degree_change_steps']>0 for row in diag['graphs'])
    assert any(row['node_category_change_steps']>0 for row in diag['graphs'])
    for g,z,u in zip(graphs,vals,us):
        np.testing.assert_allclose((u*(z*np.sqrt(len(g)))[None])@u.T,nx.to_numpy_array(g),atol=3e-6)
        assert all('atom' in g.nodes[v] for v in g)
        assert all('bond' in a for _,_,a in g.edges(data=True))
    trajectories=readp(p,'categorical_trajectories.pkl')
    assert all([step['t'] for step in tr]==[8,5,2,0] for tr in trajectories)
    for g,tr in zip(graphs,trajectories):
        np.testing.assert_array_equal(nx.to_numpy_array(g),(tr[-1]['edge_categories']>0))


def test_final_local_swaps_preserve_only_final_step_degrees(trained_categorical):
    _,_,_,gen=trained_categorical;p=gen.generation_dir
    before=readp(p,'final_pre_rewire_graphs.pkl');after=readp(p,'base_graphs.pkl')
    for g,h in zip(before,after):
        assert dict(g.degree())==dict(h.degree())
        assert nx.get_node_attributes(g,'atom')==nx.get_node_attributes(h,'atom')
        for v in g:
            assert Counter(a['bond'] for _,_,a in g.edges(v,data=True))==Counter(a['bond'] for _,_,a in h.edges(v,data=True))
    manifest=json.loads(gen.manifest_path.read_text())
    assert manifest['decode']['categorical_edge_state_is_authoritative'] and not manifest['decode']['fixed_initial_eigenbasis']
    assert not manifest['posthoc_repair'] and manifest['rejected_final_graphs']==0


def test_generation_reproducible_and_no_guidance_is_runtime_ablation(trained_categorical):
    wrapper,request,artifacts,old=trained_categorical
    new=wrapper.generate(GenerateRequest(request.run,artifacts.checkpoint_path,6,42,generation_id='repeat',options={'runtime':{'device':'cpu'}}))
    a,b=readp(old.generation_dir,'base_graphs.pkl'),readp(new.generation_dir,'base_graphs.pkl')
    for g,h in zip(a,b):assert nx.utils.graphs_equal(g,h)
    new=wrapper.generate(GenerateRequest(request.run,artifacts.checkpoint_path,2,42,generation_id='no_guidance',options={'extensions':{'attributed_categorical':{'guidance':{'enabled':False}}}}))
    diag=json.loads((new.generation_dir/'rewiring_diagnostics.json').read_text())
    assert all(not row['guidance_events'] for row in diag['graphs'])


def test_final_connected_acceptance_rejects_and_resamples_without_retraining(trained_categorical,monkeypatch):
    import grapher.models.gdsm_simple.categorical.pipeline as pipeline
    wrapper,request,artifacts,_=trained_categorical
    real_nx=pipeline.nx
    calls={'connected':0}
    class NXProxy:
        @staticmethod
        def is_graphical(*args,**kwargs): return real_nx.is_graphical(*args,**kwargs)
        @staticmethod
        def number_connected_components(*args,**kwargs): return real_nx.number_connected_components(*args,**kwargs)
        @staticmethod
        def connected_components(*args,**kwargs): return real_nx.connected_components(*args,**kwargs)
        @staticmethod
        def is_connected(graph):
            calls['connected']+=1
            # Force one sampler-level rejection, then use real connectivity.
            if calls['connected']==1: return False
            return real_nx.is_connected(graph)
    monkeypatch.setattr(pipeline,'nx',NXProxy)
    gen=wrapper.generate(GenerateRequest(
        request.run,artifacts.checkpoint_path,2,31415,generation_id='connected_acceptance',
        options={'runtime':{'device':'cpu'},'extensions':{'attributed_categorical':{
            'guidance':{'enabled':False},
            'final_acceptance':{'require_connected':True,'max_attempt_multiplier':50.0},
        }}}
    ))
    graphs=readp(gen.generation_dir,'base_graphs.pkl')
    initials=readp(gen.generation_dir,'initial_graphs.pkl')
    pre=readp(gen.generation_dir,'final_pre_rewire_graphs.pkl')
    manifest=json.loads(gen.manifest_path.read_text())
    diag=json.loads((gen.generation_dir/'final_acceptance_diagnostics.json').read_text())
    assert len(graphs)==len(initials)==len(pre)==2
    assert all(real_nx.is_connected(g) for g in graphs)
    assert manifest['final_sample_acceptance']['mode']=='reject_and_resample'
    assert manifest['rejected_final_graphs']>=1 and manifest['num_attempted']>manifest['num_generated']
    assert diag['num_rejected_disconnected']>=1 and diag['generation_yield']<1.0
    assert (gen.generation_dir/'rejected_disconnected_graphs.pkl').is_file()
    assert len(readp(gen.generation_dir,'rejected_disconnected_graphs.pkl'))==manifest['rejected_final_graphs']
    from grapher.models.gdsm_simple.categorical.evaluation import audit
    audited=audit(gen.generation_dir)
    assert audited['connectedness_rate']==1.0 and audited['rejected_disconnected_final_graphs']>=1


def test_dynamic_basis_is_passed_to_next_network_call(trained_categorical,monkeypatch):
    wrapper,request,artifacts,_=trained_categorical
    original=SpectralCategoricalDenoiser.forward;seen=[]
    def checked(self,x,e,z,t,anchor,mask,total,**kwargs):
        values,vectors=kwargs['current_pairs']
        for i,n in enumerate(mask.sum(1).tolist()):
            recon=(vectors[i,:n,:n]*(values[i,:n]*n**.5)[None])@vectors[i,:n,:n].T
            torch.testing.assert_close(recon,(e[i,:n,:n]>0).float(),atol=2e-5,rtol=2e-5)
        seen.append(e.clone())
        return original(self,x,e,z,t,anchor,mask,total,**kwargs)
    monkeypatch.setattr(SpectralCategoricalDenoiser,'forward',checked)
    wrapper.generate(GenerateRequest(request.run,artifacts.checkpoint_path,2,87,generation_id='basis_hook'))
    assert len(seen)==4 and any(not torch.equal(seen[i],seen[i-1]) for i in range(1,len(seen)))


def test_spectral_and_graph_branches_really_influence_each_other():
    model,batch,_=prepared_model();model.eval()
    pred=model(batch['x'],batch['e'],batch['z'],torch.tensor([5,5,5]),batch['anchor'],batch['mask'],10)
    pred['edge_logits'].square().sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.spec_out.parameters())
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.graph_to_spectral.parameters())


def test_audit_actual_outputs_and_separate_attributed_mmd(trained_categorical,tmp_path):
    from grapher.models.gdsm_simple.categorical.evaluation import audit,evaluate,rbf_mmd2
    _,request,_,gen=trained_categorical
    result=audit(gen.generation_dir)
    assert result['status']=='passed' and result['max_eigenpair_reconstruction_error']<1e-5
    result=evaluate(gen.generation_dir,request.dataset.split_paths['test'],max_reference=1)
    assert result['num_generated']==6 and result['num_reference']==1 and len(result['mmd2'])==4
    assert all(v>=0 for v in result['mmd2'].values())
    result=evaluate(gen.generation_dir,gen.graphs_path)
    assert all(abs(v)<1e-12 for v in result['mmd2'].values())
    assert rbf_mmd2([[1,0]],[[0,1]])>0


@pytest.mark.parametrize('dataset',['qm9','zinc','community_small','ego_small','attributed'])
def test_categorical_configs_and_cli_profiles(dataset,tmp_path):
    from grapher.models.external_cli import _MODEL_DATASETS
    path=Path(f'configs/baselines/gdsm_simple_{dataset}_categorical.yaml')
    request=TrainRequest(RunSpec('gdsm_simple',dataset,'cfg',42),DatasetReference(dataset,tmp_path),config_path=path)
    opt=GDSMSimpleWrapper()._options(request);cfg=resolve(opt)
    assert cfg['enabled'] and cfg['noise']['type']=='marginal'
    assert cfg['graphlets']['size']==3 and opt['sample']['steps']==500
    assert dataset in _MODEL_DATASETS['gdsm_simple']
    if dataset in ('qm9','zinc'):assert cfg['initialization']['degree_generator']['type']=='dhvae'


@pytest.mark.parametrize('key,value',[
    ('noise',{'type':'absorbing'}),('graphlets',{'size':4}),('initialization',{'mode':'hh'}),
    ('guidance',{'same_edge_type':False}),('spectrum_feedback',1.5),('feedback_start_fraction',-1.),
    ('noise',{'pseudocount':float('nan')}),('loss_weights',{'node':float('inf')}),
])
def test_new_config_rejects_unsupported_or_invalid_settings(key,value):
    opt=GDSMSimpleWrapper()._options(TrainRequest(RunSpec('gdsm_simple','qm9','bad',42),DatasetReference('qm9'),options=options()))
    cfg=opt['extensions']['attributed_categorical']
    if isinstance(value,dict):cfg[key].update(value)
    else:cfg[key]=value
    with pytest.raises(ValueError):resolve(opt)


@pytest.mark.parametrize('value',[0.5,float('inf'),float('nan')])
def test_final_acceptance_rejects_bad_attempt_multiplier(value):
    op=options();op['extensions']['attributed_categorical']['final_acceptance']['max_attempt_multiplier']=value
    with pytest.raises(ValueError,match='max_attempt_multiplier'): resolve(op)


def test_final_acceptance_is_generation_only_checkpoint_compatible(trained_categorical):
    wrapper,request,artifacts,_=trained_categorical
    state=torch.load(artifacts.checkpoint_path,map_location='cpu',weights_only=False)
    op=wrapper._options(request)
    op['extensions']['attributed_categorical']['final_acceptance']={'require_connected':True,'max_attempt_multiplier':7.0}
    cfg=validate_generation(state,op)
    assert cfg['final_acceptance']['require_connected'] and cfg['final_acceptance']['max_attempt_multiplier']==7.0


def test_legacy_degree_flags_cannot_leak_into_new_sampler():
    op=options();op['extensions']['degree_preserving_rewiring']=True
    with pytest.raises(ValueError,match='legacy'):
        GDSMSimpleWrapper()._options(TrainRequest(RunSpec('gdsm_simple','qm9','bad',42),DatasetReference('qm9'),options=op))


def test_real_learned_ordinary_degree_prior_and_provenance(trained_categorical):
    from grapher.models.dhvae_hh.degree_vae import DegreeVectorizer,build_degree_vae,degree_vae_loss,save_degree_vae_checkpoint
    wrapper,request,_,_=trained_categorical
    train=readp(request.dataset.dataset_dir,'train.pkl')
    torch.manual_seed(73)
    vectorizer=DegreeVectorizer.fit(train,require_connected=True)
    x,targets=vectorizer.to_training_arrays(train);x=torch.from_numpy(x);targets={k:torch.from_numpy(v) for k,v in targets.items()}
    model=build_degree_vae(vectorizer,latent_dim=4,hidden_dim=16,size_condition_dim=4,edge_condition_dim=4,
        use_edge_count_conditioning=True,prior_condition_on_edges=True,prior_type='conditional_gaussian',prior_components=1,num_layers=1)
    opt=torch.optim.Adam(model.parameters(),lr=.015)
    for _ in range(40):
        opt.zero_grad();out,mu,lv=model(x,targets['num_nodes_count'],targets['num_edges_count'])
        loss,_=degree_vae_loss(out,targets,mu,lv,weights={'num_edges':2.,'degree':5.})
        loss.backward();opt.step()
    path=request.run.output_root/'degree.pt';save_degree_vae_checkpoint(path,model,vectorizer)
    op=options();op['train']['epochs']=1
    op['extensions']['attributed_categorical']['initialization']['degree_generator']={
        'type':'dhvae','checkpoint_path':str(path),'sample_num_nodes':'empirical','sample_num_edges':'model',
        'fallback':'error','postprocess_policy':'reject_only','max_resample':100,'model_resample_attempts':8,
        'exact_degree_sum_conditioning':True,'parity_conditioned':False}
    run=RunSpec('gdsm_simple','community_small','learned',42,request.run.output_root)
    art=wrapper.train(TrainRequest(run,request.dataset,options=op))
    gen=wrapper.generate(GenerateRequest(run,art.checkpoint_path,3,79))
    manifest=json.loads(gen.manifest_path.read_text())
    assert manifest['degree_prior']['learned'] and manifest['degree_prior']['training_degree_multiset_verified']
    assert not manifest['decode']['initial_degree_projection']
    bad=DegreeVectorizer.fit([train[0]],require_connected=True)
    badmodel=build_degree_vae(bad,latent_dim=4,hidden_dim=16,size_condition_dim=4,num_layers=1)
    save_degree_vae_checkpoint(path,badmodel,bad)
    with pytest.raises(ValueError,match='multiset'):
        wrapper.generate(GenerateRequest(run,art.checkpoint_path,1,80,generation_id='stale_prior'))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Real CUDA device unavailable')
def test_categorical_cuda_smoke():
    model,batch,_=prepared_model();model=model.cuda();batch={k:v.cuda() for k,v in batch.items()}
    a=cosine_alpha_bar(10,device='cuda');q=MarginalNoise([.8,.2],a)
    p=q.forward_probs(torch.zeros(3,6,dtype=torch.long,device='cuda'),torch.tensor([2,3,4],device='cuda'))
    assert torch.isfinite(p).all()
    pred=model(batch['x'],batch['e'],batch['z'],torch.tensor([2,4,9],device='cuda'),batch['anchor'],batch['mask'],10)
    loss,_=losses(pred,batch,config()['loss_weights']);loss.backward();assert torch.isfinite(loss)

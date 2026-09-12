from __future__ import annotations

from dataclasses import replace
from math import comb
from pathlib import Path
from argparse import Namespace
from copy import deepcopy
import json
import os
import subprocess
import sys

import networkx as nx
import numpy as np
import pytest
import torch

from grapher.rewiring_mlp.generic.induced_graphlets import (
    InducedGraphletSpec, catalogue, _canonical_codes, _graph, extract_counts,
    extract_histogram, histogram_distance, histogram_from_counts,
    InducedGraphletCounter, prediction_and_loss, mask_prediction,
)
from grapher.rewiring_mlp.generic.spectral import laplacian_eigenvalues
from grapher.rewiring_mlp.generic.spectral_data import (
    TopologySpectralExample, collate_spectral_examples, build_spectral_diffusion_examples,
    TopologySpectralDiffusionIterableDataset,
)
from grapher.rewiring_mlp.generic.spectral_model import (
    TopologySpectralTransformerPredictor, save_topology_spectral_checkpoint, load_topology_spectral_checkpoint,
)
from grapher.rewiring_mlp.generic.spectral_refiner import (
    SpectralPrediction, SpectralRefinerConfig, score_spectral_candidates, refine_graph_with_spectral_predictions,
)
from grapher.rewiring_mlp.generic.rewiring import propose_valid_topology_swaps
from grapher.utils.io import load_yaml, save_yaml, load_pickle, save_pickle


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize('k,all_count,connected_count', [(3,4,2),(4,11,6),(5,34,21)])
def test_catalogue_matches_independent_atlas_and_every_labeled_pattern(k, all_count, connected_count):
    cat = catalogue(k, 'all')
    assert len(cat['bins']) == all_count
    assert sum(b['connected'] for b in cat['bins']) == connected_count
    assert len(catalogue(k, 'connected')['bins']) == connected_count + 1
    assert len(set(b['id'] for b in cat['bins'])) == all_count
    atlas = [g for g in nx.graph_atlas_g() if len(g)==k]
    bins = [int(extract_histogram(g, InducedGraphletSpec(k)).argmax()) for g in atlas]
    assert sorted(bins) == list(range(all_count))
    for code, canonical in enumerate(_canonical_codes(k)):
        assert nx.is_isomorphic(_graph(k, code), _graph(k, int(canonical)))


@pytest.mark.parametrize('k', [3,4,5])
@pytest.mark.parametrize('scope', ['all','connected'])
def test_counts_normalization_permutation_and_attribute_independence(k, scope):
    spec=InducedGraphletSpec(k,scope);g=nx.gnp_random_graph(9,.35,seed=12)
    counts=extract_counts(g,spec)
    assert counts.sum()==comb(len(g),k)
    assert np.issubdtype(counts.dtype,np.integer)
    h=extract_histogram(g,spec)
    assert h.sum()==pytest.approx(1)
    names={i:f'node_{len(g)-i}' for i in g}
    other=nx.relabel_nodes(g,names)
    nx.set_node_attributes(other,8,'atomic_num');nx.set_edge_attributes(other,3,'bond_type')
    np.testing.assert_array_equal(h,extract_histogram(other,spec))
    if scope=='connected':
        disconnected=sum(not nx.is_connected(g.subgraph(nodes)) for nodes in __import__('itertools').combinations(g,k))
        assert counts[-1]==disconnected


@pytest.mark.parametrize('k', [3,4,5])
def test_noncycles_are_separate_bins_and_disconnected_types_not_lost(k):
    spec=InducedGraphletSpec(k)
    graphs=[nx.empty_graph(k),nx.path_graph(k),nx.complete_graph(k)]
    assert len({int(extract_histogram(g,spec).argmax()) for g in graphs})==3
    if k>=4:
        a=nx.cycle_graph(k);b=a.copy();b.add_edge(0,2)
        assert extract_histogram(a,spec).argmax()!=extract_histogram(b,spec).argmax()
    a=nx.empty_graph(k);a.add_edge(0,1)
    assert extract_histogram(a,spec).argmax()!=extract_histogram(nx.empty_graph(k),spec).argmax()


@pytest.mark.parametrize('k',[3,4,5])
@pytest.mark.parametrize('scope',['all','connected'])
def test_local_delta_equals_full_recount(k,scope):
    rng=np.random.default_rng(45);spec=InducedGraphletSpec(k,scope)
    for seed in range(4):
        g=nx.gnp_random_graph(9,.4,seed=seed)
        engine=InducedGraphletCounter(g,spec)
        for _ in range(6):
            candidate=g.copy()
            for a,b in rng.integers(0,9,size=(4,2)):
                if a!=b:
                    if candidate.has_edge(int(a),int(b)): candidate.remove_edge(int(a),int(b))
                    else: candidate.add_edge(int(a),int(b))
            np.testing.assert_array_equal(engine.candidate_counts(candidate),extract_counts(candidate,spec))
            np.testing.assert_array_equal(engine.candidate_histogram(candidate),extract_histogram(candidate,spec))
        np.testing.assert_array_equal(engine.counts,extract_counts(g,spec))


@pytest.mark.parametrize('value',[True,2,6,5.0,'5',None])
def test_unsupported_k_rejected(value):
    with pytest.raises(ValueError): InducedGraphletSpec(value)


def test_invalid_graph_scope_and_histograms_rejected():
    with pytest.raises(ValueError): InducedGraphletSpec(5,'cycles')
    for g in [nx.DiGraph([(0,1)]),nx.MultiGraph([(0,1)]),nx.Graph([(0,0)])]:
        with pytest.raises(ValueError): extract_histogram(g,InducedGraphletSpec())
    with pytest.raises(ValueError): histogram_distance([.2,.8],[.5,.3])
    with pytest.raises(ValueError): histogram_distance([.2,.8],[.1,.2,.7])
    with pytest.raises(ValueError): histogram_from_counts([1]*34,5,InducedGraphletSpec())


def test_masked_small_graphs_have_zero_loss_and_gradients():
    spec=InducedGraphletSpec(5);sizes=torch.tensor([2,4])
    target=torch.tensor(np.stack([extract_histogram(nx.path_graph(n),spec) for n in sizes.tolist()])).float()
    logits=torch.randn(2,spec.width,requires_grad=True)
    brier,ce,m=prediction_and_loss(logits,target,sizes,spec)
    assert brier.item()==0 and ce.item()==0 and m['induced_graphlet_valid_fraction'].item()==0
    (brier+ce).backward();assert torch.count_nonzero(logits.grad)==0
    np.testing.assert_array_equal(mask_prediction(logits.softmax(-1),sizes,spec).detach(),target)


def test_unordered_loss_invariant_to_joint_bin_permutation():
    spec=InducedGraphletSpec(5);logits=torch.randn(2,34);target=torch.softmax(torch.randn(2,34),-1);n=torch.tensor([6,7])
    a=prediction_and_loss(logits,target,n,spec)
    perm=torch.randperm(34);b=prediction_and_loss(logits[:,perm],target[:,perm],n,spec)
    for x,y in zip(a[:2],b[:2]):assert torch.allclose(x,y)
    assert torch.allclose(a[2]['induced_graphlet_histogram_tv'],b[2]['induced_graphlet_histogram_tv'])


def settings(spec):
    return {'induced_graphlet_histogram':True,'induced_graphlet_k':spec.k,'induced_graphlet_scope':spec.scope}


def graph():
    return nx.Graph([(0,1),(0,2),(0,3),(1,2),(2,4),(3,5),(4,5)])


def example(g,spec):
    lam=laplacian_eigenvalues(g)
    return TopologySpectralExample(current_graph=g,time=.2,current_spectrum=lam,source_spectrum=lam,
        clean_spectrum_target=lam,clean_induced_graphlet_histogram_target=extract_histogram(g,spec).astype(np.float32))


def predictor(spec):
    return TopologySpectralTransformerPredictor(hidden_dim=8,edge_dim=8,graph_dim=8,num_layers=1,
       spectral_dim=16,spectral_layers=1,spectral_heads=4,spectral_ff_dim=32,use_graph_context=False,
       predict_induced_graphlet_histogram=True,induced_graphlet_k=spec.k,induced_graphlet_scope=spec.scope)


@pytest.mark.parametrize('scope',['all','connected'])
def test_spectral_targets_head_gradients_and_checkpoint(scope,tmp_path):
    spec=InducedGraphletSpec(5,scope);m=predictor(spec);b=collate_spectral_examples([example(graph(),spec),example(nx.path_graph(3),spec)])
    out=m(b);assert out['clean_induced_graphlet_histogram'].shape==(2,spec.width)
    loss,metrics=m.loss(b,loss_weights={'spectrum':0,'moment2':0,'low_frequency':0,'induced_graphlet_histogram':1})
    loss.backward()
    assert m.induced_graphlet_histogram_head[-1].weight.grad.abs().sum()>0
    assert m.spectral_token_in[0].weight.grad.abs().sum()>0
    assert metrics['induced_graphlet_valid_fraction']==.5
    m.eval();out=m(b)
    f=tmp_path/'model.pt';save_topology_spectral_checkpoint(m,f,summary_config=None,config={})
    restored,_,raw=load_topology_spectral_checkpoint(f,device='cpu')
    assert restored.induced_graphlet_spec==spec
    assert torch.equal(out['clean_induced_graphlet_histogram'],restored(b)['clean_induced_graphlet_histogram'])
    raw['model_config']['induced_graphlet_catalogue_fingerprint']='wrong'
    torch.save(raw,f)
    with pytest.raises(ValueError,match='fingerprint'):load_topology_spectral_checkpoint(f,device='cpu')


@pytest.mark.parametrize('cache',[True,False])
def test_streaming_and_eager_targets(cache):
    spec=InducedGraphletSpec(5,'all');gs=[graph(),nx.path_graph(3)]
    cfg={'samples_per_graph':2,'paths_per_graph':1,'cache_endpoints':cache}
    data=TopologySpectralDiffusionIterableDataset(gs,diffusion_config=cfg,structure_summary_config=settings(spec))
    examples=list(data);assert len(examples)==4
    assert all(x.clean_induced_graphlet_histogram_target.shape==(34,) for x in examples)
    e,_=build_spectral_diffusion_examples(gs,diffusion_config=cfg,structure_summary_config=settings(spec))
    assert len(e)==4 and all(x.clean_induced_graphlet_histogram_target is not None for x in e)


def test_graphlet_guidance_oracle_local_delta_preserves_degrees():
    spec=InducedGraphletSpec();g=graph()
    actions,candidates,_=propose_valid_topology_swaps(g,proposal_budget=-1,valid_candidate_budget=-1,
        preserve_connectivity=True,rng=np.random.default_rng(4))
    target=next(c for c in candidates.values() if histogram_distance(extract_histogram(g,spec),extract_histogram(c,spec))>0)
    lam=laplacian_eigenvalues(target);h=extract_histogram(target,spec)
    cfg=SpectralRefinerConfig.from_dict({'steps':2,'guidance_mode':'graphlet','proposal_budget':-1,'valid_candidate_budget':-1,
       'induced_graphlet_guidance':{'k':5,'scope':'all','weight':.1},
       'spectral_guidance':{'weight':0,'min_clean_mix':1,'max_clean_mix':1,'expand_on_plateau':False},
       'candidate_search':{'compute_spectral_diagnostics':False}})
    rows=score_spectral_candidates(g,actions,clean_spectrum=lam,next_spectrum_target=lam,
          clean_induced_graphlet_histogram=h,config=cfg,candidate_graphs=candidates)
    for row in rows:
        assert row['candidate_graphlet_discrepancy']==pytest.approx(histogram_distance(extract_histogram(row['candidate_graph'],spec),h))
    def oracle(*args,**kwargs):
        return SpectralPrediction(lam,laplacian_eigenvalues(args[1]),float(lam.sum()),float(np.square(lam).sum()),clean_induced_graphlet_histogram=h)
    final,trace=refine_graph_with_spectral_predictions(g,model=predictor(spec),prediction_fn=oracle,refiner_config=cfg,rng=np.random.default_rng(4),return_trace=True)
    assert dict(final.degree())==dict(g.degree()) and nx.is_connected(final)
    assert histogram_distance(extract_histogram(final,spec),h)==pytest.approx(0)
    accepted=[r for r in trace if r.get('accepted')]
    assert accepted and all(r['graphlet_gain']>0 and r['graphlet_diagnostics_computed'] for r in accepted)


def test_wrong_guidance_weight_or_catalogue_fails():
    with pytest.raises(ValueError):SpectralRefinerConfig.from_dict({'guidance_mode':'graphlet','induced_graphlet_guidance':{'weight':0}})
    with pytest.raises(ValueError):SpectralRefinerConfig.from_dict({'guidance_mode':'graphlet','induced_graphlet_guidance':{'distance':'wasserstein1'}})
    with pytest.raises(ValueError):SpectralRefinerConfig.from_dict({'guidance_mode':'graphlet','induced_graphlet_guidance':{'k':6}})


def test_joint_degree_graphlet_gradients():
    from test_joint_degree_grapher import configuration, graphs
    from grapher.rewiring_mlp.generic.joint_degree_training import build_joint_model
    cfg=configuration();cfg['structure_summary_prediction'].update(settings(InducedGraphletSpec()))
    m,_=build_joint_model(cfg,graphs());b=collate_spectral_examples([example(g,InducedGraphletSpec()) for g in graphs()])
    weights={'spectrum':0,'moment2':0,'low_frequency':0,'clustering_histogram':0,'orbit_summary':0,'induced_graphlet_histogram':1}
    # Supply targets for other enabled heads; their losses have zero weight.
    from grapher.rewiring_mlp.generic.clustering import extract_clustering_histogram
    from grapher.rewiring_mlp.generic.orbit import extract_orbit_summary
    b.clean_clustering_histogram_target=torch.tensor(np.stack([extract_clustering_histogram(g,100) for g in graphs()])).float()
    b.clean_orbit_summary_target=torch.tensor(np.stack([extract_orbit_summary(g) for g in graphs()])).float()
    loss,_=m.loss(b,loss_weights=weights);loss.backward()
    for prefix in ('degree_model.encoder','degree_model.degree_decoder','induced_graphlet_histogram_head'):
        assert sum(p.grad.abs().sum().item() for name,p in m.named_parameters() if name.startswith(prefix) and p.grad is not None)>0


def test_joint_typed_head_gradients_cache_and_rollout(tmp_path):
    from test_joint_typed_edge import tiny_config, model_and_batch, carbon_cycle
    from grapher.rewiring_mlp.attributed.joint_typed_edge_model import noisy_batch, structural_loss, save_checkpoint, load_checkpoint
    from grapher.rewiring_mlp.attributed.joint_typed_edge_generation import sample_soft_endpoint, refine_typed_graph
    cfg=tiny_config();cfg['structure_summary_prediction'].update(settings(InducedGraphletSpec()))
    cfg['attributed_predictor']['loss_weights']['induced_graphlet_histogram']=1.0
    cfg['attributed_refiner']['weights']['graphlet']=.1
    cfg,m,items,b=model_and_batch(cfg)
    assert m.induced_graphlet_basis is not None, 'Molecular trainer must fit a LABELED vocabulary.'
    assert m.induced_graphlet_spec is None
    assert m.induced_graphlet_metadata()['attributed'] is True
    expected_width = m.induced_graphlet_basis.width
    assert b['induced_histogram'].shape==(2, expected_width)
    out=m(noisy_batch(b,m,cfg,endpoint_only=True))
    weights={k:0. for k in cfg['attributed_predictor']['loss_weights']};weights['induced_graphlet_histogram']=1
    loss,metrics=structural_loss(out,b,m,weights);loss.backward()
    for prefix in ('degree_model.encoder','degree_model.signature_decoder','induced_graphlet_head'):
        assert sum(p.grad.abs().sum().item() for name,p in m.named_parameters() if name.startswith(prefix) and p.grad is not None)>0
    path=tmp_path/'typed.pt';save_checkpoint(path,m,cfg,metrics);restored,_=load_checkpoint(path)
    assert restored.induced_graphlet_metadata()==m.induced_graphlet_metadata()
    targets,_=sample_soft_endpoint(restored,items[0]['source'],cfg,seed=4)
    assert targets['induced_graphlet_metadata']['width']==expected_width
    final,report=refine_typed_graph(items[0]['source'],targets,restored,cfg,seed=4)
    assert report['typed_degree_preserved'] and report['connected']
    assert 'graphlet' in report['initial_discrepancies']
    del targets['induced_graphlet_metadata']
    with pytest.raises(ValueError,match='catalogue'):refine_typed_graph(items[0]['source'],targets,restored,cfg,seed=4)


def test_joint_typed_disabled_head_rejects_guidance():
    from test_joint_typed_edge import model_and_batch
    from grapher.rewiring_mlp.attributed.joint_typed_edge_generation import validate_refiner
    cfg,m,_,_=model_and_batch();cfg['attributed_refiner']['weights']['graphlet']=.1
    with pytest.raises(ValueError,match='head'):validate_refiner(cfg['attributed_refiner'],m)


@pytest.mark.parametrize('joint',[False,True])
def test_topology_cli_train_generate_diagnose_evaluate(joint,tmp_path,monkeypatch):
    from scripts import train_topology_grapher as train, run_topology_grapher as generate
    from scripts import diagnose_spectral_denoiser as diagnose, evaluate_graph_generation_report as evaluate
    from scripts import evaluate_induced_graphlets as evaluate_graphlets
    from grapher.data.io import save_dataset_splits
    from test_joint_degree_grapher import configuration, graphs
    cfg=configuration();gs=graphs();root=tmp_path/'data'
    save_dataset_splits('induced_fixture',{'train':gs,'val':[gs[1],gs[2]],'test':[gs[3]]},{},root)
    cfg['benchmark']='induced_fixture';cfg['dataset'].update(name='induced_fixture',root=str(root),config_path=None)
    if not joint:
        cfg['joint_degree']['enabled']=False;cfg['generation']['degree_source']='train_empirical'
    cfg['joint_degree'].setdefault('checkpointing',{}).update(enabled=True,save_last=True)
    cfg['topology_predictor'].update(epochs=2)
    cfg['structure_summary_prediction'].update(settings(InducedGraphletSpec()))
    cfg['topology_predictor']['loss_weights']['induced_graphlet_histogram']=1
    cfg['topology_refiner']['guidance_mode']='clustering_orbit_graphlet'
    cfg['topology_refiner']['induced_graphlet_guidance']={'k':5,'scope':'all','weight':.1}
    train_dir=tmp_path/'train';gen_dir=tmp_path/'gen';cp=train_dir/'checkpoint.pt'
    path=tmp_path/'config.yaml';save_yaml(cfg,path)
    monkeypatch.setattr(sys,'argv',['train','--config',str(path),'--output-dir',str(train_dir),'--seed','42','--device','cpu']);train.main()
    if joint:
        reg=json.loads((train_dir/'checkpoint_registry.json').read_text())
        assert 'best_graphlet' in reg['selections']
    monkeypatch.setattr(sys,'argv',['gen','--config',str(path),'--checkpoint',str(cp),'--output-dir',str(gen_dir),'--num-generate','2','--seed','42','--device','cpu']);generate.main()
    d=json.loads((gen_dir/'report.json').read_text())['diagnostics']
    assert d['induced_graphlet_metadata']['width']==34
    assert 'graphlet' in d['scoring_components'] and d['degree_preservation_rate']==1
    monkeypatch.setattr(sys,'argv',['diag','--config',str(path),'--checkpoint',str(cp),'--source-endpoint-only','--samples-per-graph','1','--paths-per-graph','1','--device','cpu','--json-out',str(tmp_path/'diag.json')]);diagnose.main()
    assert 'induced_graphlet_histogram_tv' in json.loads((tmp_path/'diag.json').read_text())['overall']
    monkeypatch.setattr(sys,'argv',['eval','--config',str(path),'--generated-dir',str(gen_dir),'--reference-split','val','--output-dir',str(gen_dir/'eval')]);evaluate.main()
    monkeypatch.setattr(sys,'argv',['graphlets','--config',str(path),'--generated-graphs',str(gen_dir/'topology_refined_graphs.pkl'),'--reference-split','val','--json-out',str(gen_dir/'graphlets.json')]);evaluate_graphlets.main()
    assert json.loads((gen_dir/'graphlets.json').read_text())['catalogue']['width']==34
    # Wrong vocab is rejected even when the k and checkpoint family look plausible.
    cfg['topology_refiner']['induced_graphlet_guidance']['scope']='connected';save_yaml(cfg,path)
    monkeypatch.setattr(sys,'argv',['gen','--config',str(path),'--checkpoint',str(cp),'--output-dir',str(tmp_path/'bad'),'--num-generate','1','--device','cpu'])
    with pytest.raises(ValueError,match='catalogue'):generate.main()


def test_joint_typed_cli_smoke(tmp_path):
    from test_joint_typed_edge import prepare_dataset
    cfg=prepare_dataset(tmp_path);cfg['structure_summary_prediction'].update(settings(InducedGraphletSpec()))
    cfg['attributed_predictor']['loss_weights']['induced_graphlet_histogram']=1
    cfg['attributed_refiner']['weights']['graphlet']=.1
    path=tmp_path/'cfg.yaml';save_yaml(cfg,path);train=tmp_path/'train';gen=tmp_path/'gen'
    env={**os.environ,'PYTHONPATH':'src','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    def run(script,*args):
        p=subprocess.run([sys.executable,'scripts/'+script,'--config',str(path),*map(str,args)],env=env,capture_output=True,text=True)
        assert p.returncode==0,p.stdout+'\n'+p.stderr
    run('train_attributed_grapher.py','--output-dir',train,'--epochs',2,'--device','cpu')
    reg=json.loads((train/'checkpoint_registry.json').read_text());assert 'best_graphlet' in reg['selections']
    run('run_attributed_grapher.py','--checkpoint',train/'checkpoint.pt','--output-dir',gen,'--num-generate',2,'--device','cpu')
    report_width = json.loads((gen/'report.json').read_text())['induced_graphlet_metadata']['width']
    assert report_width > 0
    assert json.loads((gen/'report.json').read_text())['induced_graphlet_metadata']['attributed'] is True
    for mode in ('source','bridge','rollout'):
        run('diagnose_joint_typed_edge.py','--checkpoint',train/'checkpoint.pt','--mode',mode,'--max-graphs',2,'--device','cpu','--json-out',tmp_path/(mode+'.json'))
        assert 'induced_graphlet_histogram_tv' in json.loads((tmp_path/(mode+'.json')).read_text())['means']
    run('evaluate_induced_graphlets.py','--checkpoint',train/'checkpoint.pt','--generated-graphs',gen/'molecular_graphs.pkl','--reference-split','val','--json-out',gen/'graphlets.json')

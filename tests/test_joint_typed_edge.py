from __future__ import annotations
from copy import deepcopy
from argparse import Namespace
import json
from pathlib import Path
import subprocess
import sys
import os

import networkx as nx
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from grapher.rewiring_mlp.attributed.soft_edge_bridge import (
    labels_to_logits,center_edges,edge_noise,bridge_edges,advance_edges,pair_mask,bridge_spectra,
)
from grapher.models.dhvae_hh.typed_degree_vae import (
    TypedSignatureVectorizer, build_typed_signature_vae,
    load_typed_signature_checkpoint, save_typed_signature_checkpoint,
)
from grapher.models.dhvae_hh.typed_constructor import TypedConstructionError, construct_typed_graph
from grapher.rewiring_mlp.attributed.joint_typed_edge_data import (
    ENDPOINT_VALENCE_POLICY, EndpointStore, collate, graph_record, graph_from_record, load_splits,
)
from grapher.rewiring_mlp.attributed.joint_typed_edge_model import (
    JointTypedEdgePredictor,noisy_batch,structural_loss,save_checkpoint,load_checkpoint,
)
from grapher.rewiring_mlp.attributed.joint_typed_edge_training import train_joint_typed_edge,validate_config,build_model
from grapher.rewiring_mlp.attributed.joint_typed_edge_generation import (
    sample_soft_endpoint,refine_typed_graph,edge_energy,generation_sources,generate_joint_typed_edge,
)
from grapher.rewiring_mlp.molecular.typed_invariants import extract_typed_invariant,typed_invariant_matches_graph
from grapher.utils.io import save_pickle,save_yaml


@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def carbon_cycle(n=6):
    g=nx.cycle_graph(n);nx.set_node_attributes(g,6,'atomic_num');nx.set_edge_attributes(g,1,'bond_type');return g


def tiny_config(root=None):
    return {'experiment':'tiny','benchmark':'toy','seed':42,'pipeline':{'stage':'attributed'},
     'dataset':{'name':'toy','root':str(root or 'unused')},
     'categorical_state':{'node_attribute':'atomic_num','edge_attribute':'bond_type','node_categories':[6,8],'edge_categories':[1,2,3]},
     'typed_signature':{'max_ordinary_degree':4},'degree_generator':{'checkpoint_path':None},
     'source_enrichment':{'enabled':False},
     'joint_typed_degree':{'enabled':True,'initialize_degree_checkpoint':None,'trainable':True,'freeze_epochs':0,
       'latent_dim':4,'hidden_dim':8,'num_layers':1,'size_condition_dim':4,'prior_type':'conditional_gaussian',
       'prior_components':1,'learning_rate':1e-3,'loss_weight':0.01},
     'attributed_predictor':{'type':'joint_typed_soft_edge','hidden_dim':16,'num_layers':1,'spectral_layers':1,'spectral_heads':2,
       'batch_size':2,'epochs':2,'learning_rate':1e-3,
       'loss_weights':{'edge_ce':1.,'edge_logit':0.1,'typed_consistency':0.1,'spectrum':0.5,'clustering_histogram':0.25,'orbit_summary':0.5}},
     'edge_diffusion':{'enabled':True,'bridge':'centered_logit_brownian','smoothing':0.01,'sigma':1.0,'spectral_enabled':True,
       'spectral_sigma':0.1,'endpoint_fraction':0.1,'sampling_steps':3,'views_per_graph':2,'generation_strategy':'bridge_then_rewire'},
     'structure_summary_prediction':{'clustering_histogram':True,'clustering_bins':10,'orbit_summary':True},
     'training_sources':{'memory_cache_graphs':1},
     'constructor':{'ensure_connected':True,'randomize_assignment':True,'max_restarts':2,'max_backtracks':1000,'candidate_ranking':'uniform'},
     'generation':{'invariant_source':'learned','num_generate':2,'max_attempts_per_graph':4,'max_invariant_resample':100,
       'require_rdkit_source_validity':True,'checkpoint_every':1},
     'attributed_refiner':{'steps':3,'proposal_budget':100,'valid_candidate_budget':50,'strict_same_bond':True,
       'preserve_connectivity':True,'rdkit_candidate_filter':True,'normalization':'initial',
       'weights':{'edge':1.,'clustering':0.25,'orbit':1.}}}


def model_and_batch(config=None):
    cfg=config or tiny_config();graphs=[carbon_cycle(5),carbon_cycle(6)]
    model=build_model(cfg,graphs,torch.device('cpu'))
    store=EndpointStore(graphs,model.vectorizer,model.atom_types,cfg,seed=7,graphlet_basis=model.induced_graphlet_basis)
    items=[store[0],store[1]]
    return cfg,model,items,collate(items,model.vectorizer,model.atom_types)


@pytest.mark.parametrize('n,k',[(1,4),(3,2),(6,4)])
def test_bridge_masks_symmetry_endpoints(n,k):
    mask=torch.tensor([[True]*n+[False]*2]);labels=torch.zeros(1,n+2,n+2,dtype=torch.long)
    if n>1: labels[0,0,1]=labels[0,1,0]=1
    a=labels_to_logits(labels,k,mask);b=-a
    for t in (0.,0.3,1.):
        z=bridge_edges(a,b,torch.tensor([t]),mask,generator=torch.Generator().manual_seed(4))
        assert torch.allclose(z,z.transpose(1,2))
        assert torch.allclose(z.sum(-1),torch.zeros_like(z[...,0]),atol=2e-6)
        assert torch.count_nonzero(z[~pair_mask(mask)])==0
        if t==0: assert torch.allclose(z,a)
        if t==1: assert torch.allclose(z,b)


def test_bridge_conditional_mean_variance():
    mask=torch.ones(20000,2,dtype=torch.bool);z=torch.zeros(20000,2,2,3);end=z+torch.tensor([-1.,0.,1.])
    result=advance_edges(z,end,0.2,0.6,mask,2.,generator=torch.Generator().manual_seed(3))[:,0,1,:]
    assert torch.allclose(result.mean(0),torch.tensor([-.5,0,.5]),atol=0.025)
    variance=4*(.6-.2)*(1-.6)/(1-.2)*(1-1/3)
    assert torch.allclose(result.var(0),torch.full((3,),variance),atol=0.03)


@pytest.mark.parametrize('t,s',[(0.3,0.3),(.5,.1),(1.,1.),(-.1,.2)])
def test_bad_times(t,s):
    with pytest.raises(ValueError): advance_edges(torch.zeros(1,2,2,4),torch.zeros(1,2,2,4),t,s,torch.ones(1,2,dtype=torch.bool),1.)


def test_spectral_bridge_preserves_trace_and_zero_mode():
    mask=torch.tensor([[True,True,True,False]])
    a=torch.tensor([[[0.,2.,4.,0.],[0.,3.,5.,0.]]]);b=torch.tensor([[[0.,1.,5.,0.],[0.,4.,4.,0.]]])
    z=bridge_spectra(a,b,torch.tensor([.6]),mask,.3,generator=torch.Generator().manual_seed(4))
    assert torch.allclose(z.sum(-1),a.sum(-1));assert torch.count_nonzero(z[:,:,0])==0;assert torch.count_nonzero(z[:,:,-1])==0


def test_joint_endpoint_alignment_cache_and_relabel(tmp_path):
    cfg,model,items,batch=model_and_batch()
    store=EndpointStore([carbon_cycle(5)],model.vectorizer,model.atom_types,cfg,seed=6,cache_path=tmp_path/'cache.sqlite',graphlet_basis=model.induced_graphlet_basis)
    x=store[0];store.close()
    store2=EndpointStore([carbon_cycle(5)],model.vectorizer,model.atom_types,cfg,seed=6,cache_path=tmp_path/'cache.sqlite',graphlet_basis=model.induced_graphlet_basis)
    assert graph_record(x['source'])==graph_record(store2[0]['source']);store2.close()
    b=collate(items,model.vectorizer,model.atom_types,rng=np.random.default_rng(5))
    assert torch.equal(b['typed_features'],batch['typed_features'])
    assert torch.equal((b['source_labels'][...,None]==torch.tensor([1,2,3])).sum(2),b['typed_degrees'])
    assert torch.equal((b['target_labels'][...,None]==torch.tensor([1,2,3])).sum(2),b['typed_degrees'])


def raw_qm9_target(case='bond_order_sum'):
    if case == 'ordinary_degree':
        graph = nx.empty_graph(6)
        graph.add_edges_from((4, node) for node in (0, 1, 2, 3, 5))
        nx.set_node_attributes(graph, 6, 'atomic_num')
        graph.nodes[4]['atomic_num'] = 7
        nx.set_edge_attributes(graph, 1, 'bond_type')
        return graph
    # Raw SDF nitro representation: C-N(=O)=O. Sanitization would change bonds.
    graph = nx.Graph()
    graph.add_nodes_from((i, {'atomic_num': z}) for i, z in enumerate([6, 7, 8, 8]))
    graph.add_edges_from([(0, 1, {'bond_type': 1}), (1, 2, {'bond_type': 2}),
                          (1, 3, {'bond_type': 2})])
    return graph


def raw_qm9_config(root=None):
    cfg = tiny_config(root)
    cfg['categorical_state']['node_categories'] = [6, 7, 8]
    for section in ('typed_signature', 'constructor'):
        cfg[section]['max_weighted_valence'] = {6: 4., 7: 4., 8: 3.}
        cfg[section]['max_ordinary_degree'] = 4
    return cfg


@pytest.mark.parametrize('case', ['bond_order_sum', 'ordinary_degree'])
def test_raw_target_endpoints_preserve_signatures_and_keep_generation_valence_caps(tmp_path, case):
    cfg = raw_qm9_config()
    before_cfg = deepcopy(cfg)
    target = raw_qm9_target(case)
    before_target = graph_record(target)
    invariant = extract_typed_invariant(target, edge_types=(1, 2, 3))
    strict_constructor = deepcopy(cfg['constructor'])
    if case == 'ordinary_degree':
        # Isolate the ordinary-degree cap from the weighted-valence cap.
        strict_constructor['max_weighted_valence'] = None
    error = 'node 4: degree 5 exceeds 4' if case == 'ordinary_degree' else 'weighted degree 5 exceeds 4'
    with pytest.raises(TypedConstructionError, match=error):
        construct_typed_graph(invariant, strict_constructor)
    model = build_model(cfg, [target], torch.device('cpu'))
    cache = tmp_path / 'endpoints.sqlite'
    for _ in range(2):  # Exercise construction and SQLite replay.
        store = EndpointStore([target], model.vectorizer, model.atom_types, cfg, seed=42, cache_path=cache)
        try:
            item = store[0]
            assert typed_invariant_matches_graph(item['source'], invariant)
            assert graph_record(item['target']) == before_target
            assert item['constructor']['valence_policy'] == ENDPOINT_VALENCE_POLICY
            batch = collate([item], model.vectorizer, model.atom_types)
            if case == 'ordinary_degree':
                assert batch['typed_degrees'][0, 4].tolist() == [5., 0., 0.]
                assert batch['spectral_trace'][0].tolist() == [10., 10.]
            else:
                assert batch['typed_degrees'][0, 1].tolist() == [1., 2., 0.]
                assert batch['spectral_trace'][0].tolist() == [6., 10.]
        finally:
            store.close()
    assert cfg == before_cfg
    assert graph_record(target) == before_target
    assert model.vectorizer.max_weighted_valence[7] == 4.
    assert model.vectorizer.max_ordinary_degree == 4
    # Unconditional empirical generation must still reject this stored encoding.
    cfg['generation'].update(invariant_source='train_empirical', max_attempts_per_graph=1,
                             require_rdkit_source_validity=False)
    cfg['attributed_refiner']['rdkit_candidate_filter'] = False
    cfg['constructor'] = strict_constructor
    with pytest.raises(RuntimeError, match="constructor_failures.*1"):
        list(generation_sources(model, [target], cfg, seed=42, num_generate=1))


@pytest.mark.parametrize('warm_start', [False, True])
@pytest.mark.parametrize('case', ['bond_order_sum', 'ordinary_degree'])
def test_raw_qm9_targets_train_and_diagnose(tmp_path, monkeypatch, warm_start, case):
    from scripts import diagnose_joint_typed_edge as diagnose

    cfg = raw_qm9_config(tmp_path / 'data')
    dataset = tmp_path / 'data' / 'toy'
    dataset.mkdir(parents=True)
    for split in ('train', 'val', 'test'):
        save_pickle([raw_qm9_target(case)] * 2, dataset / f'{split}.pkl')
    if warm_start:
        initial = build_model(cfg, [raw_qm9_target(case)], torch.device('cpu'))
        component = tmp_path / 'typed_degree.pt'
        save_typed_signature_checkpoint(component, initial.degree_model, initial.vectorizer)
        cfg['joint_typed_degree']['initialize_degree_checkpoint'] = str(component)
    output = tmp_path / 'training'
    args = Namespace(seed=42, epochs=1, batch_size=2, device='cpu', output_dir=str(output),
                     max_train_graphs=None, max_val_graphs=None)
    train_joint_typed_edge(cfg, args)
    report = json.loads((output / 'report.json').read_text())
    assert report['train_graphs'] == report['val_graphs'] == 2
    assert report['endpoint_valence_policy'] == ENDPOINT_VALENCE_POLICY
    assert json.loads((output / 'run_config.json').read_text())['endpoint_valence_policy'] == ENDPOINT_VALENCE_POLICY
    model, _ = load_checkpoint(output / 'checkpoint.pt', 'cpu')
    assert model.vectorizer.max_weighted_valence[7] == 4.
    assert model.vectorizer.max_ordinary_degree == 4
    config_path = tmp_path / 'config.yaml'
    save_yaml(cfg, config_path)
    diagnostic = tmp_path / 'diagnostic.json'
    monkeypatch.setattr(sys, 'argv', ['diagnose', '--config', str(config_path),
        '--checkpoint', str(output / 'checkpoint.pt'), '--split', 'val', '--device', 'cpu',
        '--max-graphs', '2', '--json-out', str(diagnostic)])
    diagnose.main()
    result = json.loads(diagnostic.read_text())
    assert result['graphs'] == result['examples'] == 2
    assert all(np.isfinite(value) for value in result['means'].values())


def test_structure_gradients_reach_typed_encoder_and_decoder():
    cfg=tiny_config();graphs=[carbon_cycle(5),carbon_cycle(6)]
    graphs[1].nodes[0]['atomic_num']=8
    model=build_model(cfg,graphs,torch.device('cpu'))
    store=EndpointStore(graphs,model.vectorizer,model.atom_types,cfg,seed=7,graphlet_basis=model.induced_graphlet_basis)
    batch=collate([store[0],store[1]],model.vectorizer,model.atom_types)
    model.train();model.set_degree_trainable(True)
    noisy=noisy_batch(batch,model,cfg,generator=torch.Generator().manual_seed(6));pred=model(noisy)
    loss,metrics=structural_loss(pred,noisy,model,cfg['attributed_predictor']['loss_weights']);loss.backward()
    for module in [model.degree_model.encoder,model.degree_model.signature_decoder]:
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in module.parameters())
    assert model.degree_model.signature_decoder.net[-1].weight.grad.abs().sum()>0
    assert metrics['orbit_identity_max_abs']<1e-4
    assert torch.allclose(pred['clean_spectra'].sum(-1),batch['spectral_trace'],atol=1e-5)


def test_freeze_means_no_typed_gradients():
    cfg,model,_,b=model_and_batch();model.set_degree_trainable(False);model.train()
    assert not model.degree_model.training
    noisy=noisy_batch(b,model,cfg);loss,_=structural_loss(model(noisy),noisy,model,cfg['attributed_predictor']['loss_weights']);loss.backward()
    assert all(p.grad is None for p in model.degree_model.parameters())
    assert any(p.grad is not None for p in model.edge_head.parameters())


def test_node_permutation_equivariance():
    cfg,model,items,b=model_and_batch();model.eval();noisy=noisy_batch(b,model,cfg,endpoint_only=True)
    p=torch.tensor([2,0,4,1,3,5]);bp={k:v.clone() for k,v in noisy.items()}
    for k in ('mask','atom','typed_degrees'):bp[k]=bp[k][:,p]
    for k in ('source_labels','target_labels','edge_state'):bp[k]=bp[k][:,p][:,:,p]
    # Spectral ranks describe eigenvalue positions, not node labels: do not permute spectra/masks there.
    # Test equal-sized graph separately so spectral padding mask remains unchanged.
    items=[items[1]];b=collate(items,model.vectorizer,model.atom_types);noisy=noisy_batch(b,model,cfg,endpoint_only=True)
    bp={k:v.clone() for k,v in noisy.items()}
    for k in ('atom','typed_degrees'):bp[k]=bp[k][:,p]
    for k in ('source_labels','target_labels','edge_state'):bp[k]=bp[k][:,p][:,:,p]
    a=model(noisy);ap=model(bp)
    assert torch.allclose(ap['clean_edge_logits'],a['clean_edge_logits'][:,p][:,:,p],atol=3e-6)
    assert torch.allclose(ap['clean_clustering_histogram'],a['clean_clustering_histogram'],atol=3e-6)
    assert torch.allclose(ap['clean_orbit_summary'],a['clean_orbit_summary'],atol=3e-6)


def test_padding_does_not_change_valid_predictions():
    cfg,model,items,b=model_and_batch();model.eval()
    single=collate([items[0]],model.vectorizer,model.atom_types)
    pa=model(noisy_batch(single,model,cfg,endpoint_only=True));pb=model(noisy_batch(b,model,cfg,endpoint_only=True))
    assert torch.allclose(pa['clean_edge_logits'][0],pb['clean_edge_logits'][0,:5,:5],atol=3e-6)
    assert torch.allclose(pa['clean_clustering_histogram'][0],pb['clean_clustering_histogram'][0],atol=3e-6)


def test_edge_nll_includes_no_bond():
    g=carbon_cycle(5);p=np.zeros((5,5,4))+0.1;p[...,0]=.7
    assert edge_energy(g,p,(1,2,3))==pytest.approx((5*-np.log(.1)+5*-np.log(.7))/10)


def test_oracle_edge_projection_preserves_typed_graph():
    cfg,model,items,b=model_and_batch();source=carbon_cycle(6)
    # A legal same-type switch provides a reachable oracle.
    target=source.copy();target.remove_edges_from([(0,1),(3,4)]);target.add_edge(0,3,bond_type=1);target.add_edge(1,4,bond_type=1)
    labels=torch.zeros(1,6,6,dtype=torch.long)
    for u,v in target.edges():labels[0,u,v]=labels[0,v,u]=1
    p=labels_to_logits(labels,4,torch.ones(1,6,dtype=torch.bool)).softmax(-1)[0].numpy()
    cfg['attributed_refiner']['weights']={'edge':1.};cfg['attributed_refiner']['proposal_budget']=10000;cfg['attributed_refiner']['valid_candidate_budget']=10000
    final,report=refine_typed_graph(source,{'edge_probabilities':p},model,cfg,seed=5)
    assert report['accepted_steps']>0
    assert typed_invariant_matches_graph(final,extract_typed_invariant(source,edge_types=model.edge_types))
    assert edge_energy(final,p,model.edge_types)<edge_energy(source,p,model.edge_types)
    assert all(row['energy_after']<row['energy_before'] for row in report['trace'])


def test_soft_rollout_seed_and_mask():
    cfg,model,items,b=model_and_batch();source=items[0]['source']
    t1,d1=sample_soft_endpoint(model,source,cfg,seed=1);t2,d2=sample_soft_endpoint(model,source,cfg,seed=1)
    assert np.array_equal(t1['edge_probabilities'],t2['edge_probabilities'])
    assert np.allclose(t1['edge_probabilities'],t1['edge_probabilities'].transpose(1,0,2))
    assert np.allclose(t1['edge_probabilities'].sum(-1),1.)
    assert d1['prediction_calls']==4
    t3,_=sample_soft_endpoint(model,source,cfg,seed=2)
    assert not np.array_equal(t1['edge_probabilities'],t3['edge_probabilities'])


def test_checkpoint_roundtrip(tmp_path):
    cfg,model,items,b=model_and_batch();model.eval();noisy=noisy_batch(b,model,cfg,endpoint_only=True)
    before=model(noisy)['clean_edge_logits'];p=tmp_path/'joint.pt';save_checkpoint(p,model,cfg,{})
    restored,c=load_checkpoint(p);after=restored(noisy)['clean_edge_logits'];assert torch.equal(before,after)


def prepare_dataset(tmp_path):
    cfg=tiny_config(tmp_path/'data');root=tmp_path/'data'/'toy';root.mkdir(parents=True)
    for split,n in [('train',4),('val',2),('test',2)]:save_pickle([carbon_cycle(6) for _ in range(n)],root/f'{split}.pkl')
    return cfg


def test_random_training_subset_reaches_basis_endpoints_and_generation(tmp_path, monkeypatch):
    from grapher.rewiring_mlp.attributed import joint_typed_edge_training as training
    from grapher.rewiring_mlp.attributed import joint_typed_edge_generation as generation
    from grapher.utils.io import load_pickle

    cfg = prepare_dataset(tmp_path)
    cfg['dataset']['max_train_graphs'] = 1  # The CLI count must take precedence.
    cfg['structure_summary_prediction'].update(
        induced_graphlet_histogram=True, induced_graphlet_attributed=True,
        induced_graphlet_k=3, induced_graphlet_scope='all')
    pool = [carbon_cycle(6) for _ in range(8)]
    for index, graph in enumerate(pool):
        graph.graph['prepared_index'] = index
    split_path = tmp_path/'data'/'toy'/'train.pkl'
    save_pickle(pool, split_path)
    observed = {}
    original_fit = training.fit_training_basis
    original_store = training.EndpointStore

    def fit(config, graphs):
        observed['basis'] = [g.graph['prepared_index'] for g in graphs]
        return original_fit(config, graphs)

    def store(graphs, *args, **kwargs):
        observed.setdefault('stores', []).append([g.graph.get('prepared_index') for g in graphs])
        return original_store(graphs, *args, **kwargs)

    monkeypatch.setattr(training, 'fit_training_basis', fit)
    monkeypatch.setattr(training, 'EndpointStore', store)
    output = tmp_path/'training_subset'
    args = Namespace(seed=42, epochs=1, batch_size=2, device='cpu', output_dir=str(output),
                     max_train_graphs=3, max_val_graphs=None)
    train_joint_typed_edge(cfg, args)
    selection = json.loads((output/'training_subset.json').read_text())
    indices = selection['indices']
    assert len(indices) == 3 and indices != [0, 1, 2]
    assert observed['basis'] == observed['stores'][0] == indices
    assert observed['stores'][1] == [None, None]
    assert [g.graph['prepared_index'] for g in load_pickle(split_path)] == list(range(8))
    report = json.loads((output/'report.json').read_text())
    assert report['training_subset'] == selection
    _, checkpoint = load_checkpoint(output/'checkpoint.pt')
    assert checkpoint['config']['dataset']['training_subset'] == selection

    original_sources = generation.generation_sources

    def sources(model, graphs, *args, **kwargs):
        observed['generation'] = [g.graph['prepared_index'] for g in graphs]
        return original_sources(model, graphs, *args, **kwargs)

    monkeypatch.setattr(generation, 'generation_sources', sources)
    cfg['generation']['invariant_source'] = 'train_empirical'
    generate_joint_typed_edge(cfg, Namespace(seed=99, device='cpu',
        output_dir=str(tmp_path/'generation_subset'), num_generate=1,
        checkpoint=str(output/'checkpoint.pt')))
    assert observed['generation'] == indices


def test_train_generate_export_diagnose_smoke(tmp_path):
    cfg=prepare_dataset(tmp_path);out=tmp_path/'training'
    args=Namespace(seed=42,epochs=2,batch_size=2,device='cpu',output_dir=str(out),max_train_graphs=None,max_val_graphs=None)
    train_joint_typed_edge(cfg,args)
    reg=json.loads((out/'checkpoint_registry.json').read_text());assert len(reg['selections'])==5
    for kind in reg['selections']:
        m,ckpt=load_checkpoint(out/'checkpoints'/kind/'checkpoint.pt')
        prior,v,_=load_typed_signature_checkpoint(out/'checkpoints'/kind/'degree_checkpoint.pt',device='cpu')
        for k,t in m.degree_model.state_dict().items():assert torch.equal(t,prior.state_dict()[k])
    gdir=tmp_path/'generation';ga=Namespace(seed=42,device='cpu',output_dir=str(gdir),num_generate=2,checkpoint=str(out/'checkpoint.pt'))
    generate_joint_typed_edge(cfg,ga)
    report=json.loads((gdir/'report.json').read_text())
    assert report['complete'] and report['num_generated']==2
    assert report['degree_sampler_source']=='joint_checkpoint_embedded'
    assert report['diagnostics']['typed_degree_preservation_rate']==1
    path=tmp_path/'cfg.yaml';save_yaml(cfg,path)
    env=dict(os.environ,PYTHONPATH='src',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    for mode in ('source','bridge','rollout'):
        subprocess.run([sys.executable,'scripts/diagnose_joint_typed_edge.py','--config',str(path),
         '--checkpoint',str(out/'checkpoint.pt'),'--device','cpu','--max-graphs','2','--mode',mode,
         '--json-out',str(tmp_path/(mode+'.json'))],check=True,env=env,capture_output=True,text=True)
    # Existing molecule evaluator accepts the new serialized output unchanged.
    subprocess.run([sys.executable,'scripts/evaluate_generated_molecules.py','--generated-graphs',str(gdir/'molecular_graphs.pkl'),
      '--dataset-root',str(tmp_path/'data'),'--dataset','toy','--reference-split','val','--skip-fcd','--nspdk-backend','proxy',
      '--output-dir',str(gdir/'eval')],check=True,env=env,capture_output=True,text=True)
    # No stale data or overwrites.
    with pytest.raises(FileExistsError):generate_joint_typed_edge(cfg,ga)
    save_pickle([carbon_cycle(5)],tmp_path/'data'/'toy'/'test.pkl')
    ga.output_dir=str(tmp_path/'mismatch')
    with pytest.raises(ValueError,match='fingerprint'):generate_joint_typed_edge(cfg,ga)


def test_unsupported_config_rejected():
    cfg=tiny_config();cfg['source_enrichment']['enabled']=True
    with pytest.raises(ValueError):validate_config(cfg)
    cfg=tiny_config();cfg['degree_generator']['checkpoint_path']='old.pt'
    with pytest.raises(ValueError):validate_config(cfg)


def test_missing_typed_signature_is_not_silently_remapped():
    cfg,model,items,b=model_and_batch();g=carbon_cycle(5);g.nodes[0]['atomic_num']=8
    store=EndpointStore([g],model.vectorizer,model.atom_types,cfg,seed=1)
    with pytest.raises(ValueError,match='Unseen typed signature'):store[0]


def test_mixed_bond_source_and_hard_swaps():
    cfg=tiny_config();g=carbon_cycle(6)
    for u,v in [(0,1),(2,3),(4,5)]:g.edges[u,v]['bond_type']=2
    model=build_model(cfg,[g],torch.device('cpu'))
    store=EndpointStore([g],model.vectorizer,model.atom_types,cfg,seed=5)
    item=store[0]
    assert typed_invariant_matches_graph(item['source'],extract_typed_invariant(g,edge_types=model.edge_types))
    target=g.copy();target.remove_edges_from([(0,1),(2,3)]);target.add_edge(0,2,bond_type=2);target.add_edge(1,3,bond_type=2)
    labels=torch.zeros(1,6,6,dtype=torch.long)
    for u,v,d in target.edges(data=True):labels[0,u,v]=labels[0,v,u]=d['bond_type']
    probs=labels_to_logits(labels,4,torch.ones(1,6,dtype=torch.bool)).softmax(-1)[0].numpy()
    cfg['attributed_refiner'].update(weights={'edge':1.},proposal_budget=10000,valid_candidate_budget=10000)
    final,rep=refine_typed_graph(g,{'edge_probabilities':probs},model,cfg,seed=7)
    assert rep['accepted_steps']>0
    assert typed_invariant_matches_graph(final,extract_typed_invariant(g,edge_types=model.edge_types))
    assert rep['candidate_search_totals']['valid_candidates_scored']>0


def test_diagonal_probabilities_are_no_bond():
    cfg,m,_,b=model_and_batch();m.eval();pred=m(noisy_batch(b,m,cfg,endpoint_only=True))
    probs=pred['clean_edge_probabilities'];invalid=~pair_mask(b['mask'])
    assert torch.all(probs[...,0][invalid]==1)
    assert torch.count_nonzero(probs[...,1:][invalid])==0


@pytest.mark.parametrize('hist,orbit,spectra',[(False,False,False),(True,False,False),(False,True,False),(False,False,True)])
def test_optional_heads_and_edge_only_model(hist,orbit,spectra):
    cfg=tiny_config();cfg['structure_summary_prediction'].update(clustering_histogram=hist,orbit_summary=orbit)
    cfg['edge_diffusion']['spectral_enabled']=spectra
    weights=cfg['attributed_predictor']['loss_weights']
    if not hist:weights['clustering_histogram']=0.
    if not orbit:weights['orbit_summary']=0.
    if not spectra:weights['spectrum']=0.
    cfg['attributed_refiner']['weights']={'edge':1.}
    validate_config(cfg)
    _,m,items,b=model_and_batch(cfg);noisy=noisy_batch(b,m,cfg);pred=m(noisy)
    loss,metrics=structural_loss(pred,noisy,m,weights)
    loss.backward();assert torch.isfinite(loss)
    targets,_=sample_soft_endpoint(m,items[0]['source'],cfg,seed=4)
    final,r=refine_typed_graph(items[0]['source'],targets,m,cfg,seed=5)
    assert r['typed_degree_preserved']


def test_warmstart_frozen_and_registry_cli(tmp_path):
    cfg=prepare_dataset(tmp_path);m=build_model(cfg,[carbon_cycle(6)]*4,torch.device('cpu'))
    from grapher.models.dhvae_hh.typed_degree_vae import save_typed_signature_checkpoint
    init=tmp_path/'prior.pt';save_typed_signature_checkpoint(init,m.degree_model,m.vectorizer)
    cfg['joint_typed_degree'].update(initialize_degree_checkpoint=str(init),freeze_epochs=1,trainable=False)
    out=tmp_path/'frozen';args=Namespace(seed=42,epochs=1,batch_size=2,device='cpu',output_dir=str(out),max_train_graphs=None,max_val_graphs=None)
    train_joint_typed_edge(cfg,args);loaded,_=load_checkpoint(out/'checkpoint.pt')
    for k,v in loaded.degree_model.state_dict().items():assert torch.equal(v,m.degree_model.state_dict()[k])
    env=dict(os.environ,PYTHONPATH=os.pathsep.join(('src', '.')),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    result=subprocess.run([sys.executable,'scripts/inspect_joint_typed_checkpoints.py','--training-dir',str(out),'--verify'],check=True,env=env,capture_output=True,text=True)
    assert 'verified' in result.stdout


def test_original_cli_dispatches_new_family(tmp_path):
    cfg=prepare_dataset(tmp_path);path=tmp_path/'cfg.yaml';save_yaml(cfg,path)
    env=dict(os.environ,PYTHONPATH=os.pathsep.join(('src', '.')),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    train=tmp_path/'cli_train';gen=tmp_path/'cli_gen'
    subprocess.run([sys.executable,'scripts/train_attributed_grapher.py','--config',str(path),'--output-dir',str(train),
        '--epochs','1','--batch-size','2','--device','cpu','--seed','42'],check=True,env=env,capture_output=True,text=True)
    subprocess.run([sys.executable,'scripts/run_attributed_grapher.py','--config',str(path),'--output-dir',str(gen),
        '--checkpoint',str(train/'checkpoint.pt'),'--num-generate','1','--device','cpu','--seed','42'],check=True,env=env,capture_output=True,text=True)
    assert (gen/'molecular_graphs.pkl').is_file()


def test_source_rng_independent_of_refinement_and_soft_rollout():
    cfg,m,items,b=model_and_batch();graphs=[carbon_cycle(5),carbon_cycle(6)]
    a=list(generation_sources(m,graphs,cfg,seed=42,num_generate=3))
    rng=np.random.default_rng(0)
    for i,(g,_,_) in enumerate(a):
        sample_soft_endpoint(m,g,cfg,seed=i)
        torch.randn(100);rng.normal(size=100)
    other=deepcopy(cfg);other['attributed_refiner']['weights']={'edge':0.,'orbit':1.}
    other['attributed_refiner']['steps']=0
    b=list(generation_sources(m,graphs,other,seed=42,num_generate=3))
    assert [graph_record(x[0]) for x in a]==[graph_record(x[0]) for x in b]


@pytest.mark.parametrize('change',[
    ('edge_diffusion.sigma',-1),('edge_diffusion.smoothing',0),('edge_diffusion.sampling_steps',1),
    ('joint_typed_degree.loss_weight',-1),('generation.checkpoint_every',0),
    ('generation.degree_source','learned'),('structure_summary_prediction.orbit_width',73),
    ('structure_summary_prediction.cycle_graphlet_histogram',True),
])
def test_semantic_config_guards(change):
    cfg=tiny_config();key,value=change;top,field=key.split('.');cfg[top][field]=value
    with pytest.raises(ValueError):validate_config(cfg)


def test_full_loss_counts_each_graph_equally():
    # Loss averaging must not overweight the larger molecular graph's O(n^2) pairs.
    cfg,m,items,b=model_and_batch();m.eval()
    noisy=noisy_batch(b,m,cfg,endpoint_only=True);out=m(noisy)
    loss,metrics=structural_loss(out,noisy,m,cfg['attributed_predictor']['loss_weights'])
    ce=[]
    for item in items:
        one=collate([item],m.vectorizer,m.atom_types);one=noisy_batch(one,m,cfg,endpoint_only=True)
        _,x=structural_loss(m(one),one,m,cfg['attributed_predictor']['loss_weights']);ce.append(x['edge_ce_loss'])
    assert metrics['edge_ce_loss']==pytest.approx(float(np.mean(ce)),abs=1e-6)

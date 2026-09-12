from argparse import Namespace
from copy import deepcopy
import json
from pathlib import Path
import sys

import networkx as nx
import numpy as np
import pytest
import torch

from test_joint_typed_edge import tiny_config
from test_typed_degree_perturbation import parents
from grapher.data.io import save_dataset_splits
from grapher.models.dhvae_hh.degree_perturbation import METHODS, DegreePerturbationError
from grapher.models.dhvae_hh.typed_constructor import construct_typed_graph
from grapher.models.dhvae_hh.typed_degree_perturbation import typed_key,typed_totals
from grapher.rewiring_mlp.molecular.typed_invariants import TypedInvariant,extract_typed_invariant,typed_invariant_matches_graph
from grapher.rewiring_mlp.attributed.joint_typed_edge_training import train_joint_typed_edge
from grapher.rewiring_mlp.attributed.joint_typed_edge_generation import generation_sources
from grapher.rewiring_mlp.attributed.joint_typed_edge_model import load_checkpoint
from grapher.utils.io import save_yaml,load_pickle

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def trained_typed(tmp_path):
    old=torch.get_num_threads();state=torch.random.get_rng_state();torch.set_num_threads(1)
    graphs=[construct_typed_graph(inv,{'max_ordinary_degree':4},np.random.default_rng(i))[0] for i,inv in enumerate(parents(False))]
    data=tmp_path/'data';save_dataset_splits('toy',{'train':graphs,'val':graphs[:2],'test':graphs[2:]},{},data)
    cfg=tiny_config(data);cfg['constructor'].update(max_ordinary_degree=4,max_weighted_valence={6:4,8:2})
    cfg['generation'].update(invariant_source='train_empirical',invariant_rng_mode='independent')
    cfg['attributed_refiner'].update(steps=2,proposal_budget=32,valid_candidate_budget=12)
    cfg['attributed_predictor'].update(epochs=1)
    train=tmp_path/'train'
    args=Namespace(seed=42,epochs=1,batch_size=2,device='cpu',output_dir=str(train),max_train_graphs=None,max_val_graphs=None)
    train_joint_typed_edge(cfg,args)
    yield cfg,train/'checkpoint.pt',graphs
    torch.set_num_threads(old);torch.random.set_rng_state(state)


def test_train_then_all_typed_options_generate_evaluate(tmp_path,monkeypatch,trained_typed):
    from scripts import run_attributed_grapher as generate
    from scripts import evaluate_generated_molecules as evaluate
    cfg,ckpt,graphs=trained_typed
    parent_hashes=[];source_hashes=[];final_hashes=[]
    from grapher.rewiring_mlp.attributed.joint_typed_edge_data import record_hash,graph_record
    for method in ('empirical',*METHODS,'zero_probability'):
        settings=deepcopy(cfg)
        if method!='empirical':
            settings['generation'].update(invariant_source='train_empirical_perturbed',degree_perturbation={
                'method':'unit_transfer' if method=='zero_probability' else method,
                'probability':0. if method=='zero_probability' else 1.,'failure_policy':'keep_original'})
        path=tmp_path/(method+'.yaml');save_yaml(settings,path);out=tmp_path/method
        monkeypatch.setattr(sys,'argv',['generate','--config',str(path),'--checkpoint',str(ckpt),
             '--output-dir',str(out),'--num-generate','4','--seed','42','--device','cpu'])
        generate.main()
        prior=json.loads((out/'typed_degree_prior_report.json').read_text())
        report=json.loads((out/'report.json').read_text())
        assert report['complete'] and report['diagnostics']['typed_degree_preservation_rate']==1
        assert report['degree_sampler_source']!='joint_checkpoint_embedded'
        assert prior['num_returned']==4 and prior['checkpoint_signature_mask_enabled']
        assert prior['all_preserve_edge_type_counts'] and prior['all_joint_realizations_verified']
        parent_hashes.append(prior['parent_typed_fingerprint']);source_hashes.append(report['source_graphs_sha256'])
        sources=load_pickle(out/'coarse_graphs.pkl');finals=load_pickle(out/'molecular_graphs.pkl')
        final_hashes.append(record_hash([graph_record(g) for g in finals]))
        for source,final,row in zip(sources,finals,prior['records']):
            target=TypedInvariant.from_dict(row['typed_invariant'])
            realized=extract_typed_invariant(source,edge_types=(1,2,3))
            assert typed_key(realized)==typed_key(target)
            assert typed_invariant_matches_graph(final,realized)
            assert typed_totals(target)==tuple(row['edge_type_totals'])
        if method not in ('empirical','zero_probability'):assert prior['num_changed']>0
        monkeypatch.setattr(sys,'argv',['evaluate','--generated-graphs',str(out/'molecular_graphs.pkl'),
            '--dataset','toy','--dataset-root',cfg['dataset']['root'],'--reference-split','val',
            '--skip-fcd','--nspdk-backend','proxy','--output-dir',str(out/'evaluation_val')])
        evaluate.main()
        assert (out/'evaluation_val/molecular_evaluation_metrics.json').is_file()
    assert len(set(parent_hashes))==1
    assert source_hashes[0]==source_hashes[-1] and final_hashes[0]==final_hashes[-1]


def test_typed_diagnostic_masks_and_pairing(tmp_path,monkeypatch,trained_typed):
    from scripts import diagnose_typed_degree_perturbations as diagnose
    cfg,ckpt,graphs=trained_typed;path=tmp_path/'diag.yaml';save_yaml(cfg,path)
    out=tmp_path/'diagnose'
    monkeypatch.setattr(sys,'argv',['diagnose','--config',str(path),'--checkpoint',str(ckpt),
        '--output-dir',str(out),'--num-samples','6','--probability','1','--seed','42'])
    diagnose.main()
    result=json.loads((out/'report.json').read_text())
    assert result['parents_identical_across_methods'] and result['checkpoint_loaded']
    assert all(r['checkpoint_signature_mask_enabled'] for r in result['reports'].values())
    assert all(r['num_changed']>0 for r in result['reports'].values())


def test_strict_typed_failure_is_persisted_not_redrawn(tmp_path,monkeypatch,trained_typed):
    from scripts import run_attributed_grapher as generate
    cfg,ckpt,graphs=trained_typed
    cfg['generation'].update(invariant_source='train_empirical_perturbed',degree_perturbation={
        'method':'unit_transfer','probability':1.,'max_distance':.1,'failure_policy':'error'})
    path=tmp_path/'fail.yaml';save_yaml(cfg,path);out=tmp_path/'fail'
    monkeypatch.setattr(sys,'argv',['generate','--config',str(path),'--checkpoint',str(ckpt),
        '--output-dir',str(out),'--num-generate','4','--seed','42','--device','cpu'])
    with pytest.raises(DegreePerturbationError):generate.main()
    prior=json.loads((out/'typed_degree_prior_report.json').read_text())
    assert prior['num_samples']==1 and prior['num_returned']==0
    assert len(prior['records'])==1
    assert not json.loads((out/'report.json').read_text())['complete']


def test_actual_source_constructor_retries_do_not_change_parent(monkeypatch,trained_typed):
    import grapher.rewiring_mlp.attributed.joint_typed_edge_generation as module
    from grapher.models.dhvae_hh.typed_constructor import TypedConstructionError
    cfg,ckpt,graphs=trained_typed;model,_=load_checkpoint(ckpt,'cpu')
    cfg['generation'].update(invariant_source='train_empirical_perturbed',degree_perturbation={
        'method':'unit_transfer','probability':1.,'failure_policy':'keep_original'})
    original=module.construct_typed_graph;attempted=[]
    def sometimes(inv,*args,**kwargs):
        attempted.append(typed_key(inv))
        if len(attempted)==1:raise TypedConstructionError('test rejection',{'failure_reason':'budget_exhausted'})
        return original(inv,*args,**kwargs)
    monkeypatch.setattr(module,'construct_typed_graph',sometimes)
    rows=list(generation_sources(model,graphs,cfg,seed=42,num_generate=2))
    assert attempted[0]==attempted[1]
    assert rows[0][1]['attempts']==2
    assert rows[-1][2]['invariant_proposals']==2



"""Command compatibility with the September 12 163051 archive; not quality benchmarks."""
from copy import deepcopy
from argparse import Namespace
from pathlib import Path
import json
import sys

import networkx as nx
import numpy as np
import pytest
import torch

from grapher.data.io import save_dataset_splits
from grapher.data.sampling import sample_training_graphs
from grapher.utils.io import load_yaml, save_yaml, load_pickle
from grapher.models.dhvae_hh.typed_constructor import construct_typed_graph
from grapher.models.dhvae_hh.degree_perturbation import METHODS
from grapher.rewiring_mlp.attributed.joint_typed_edge_generation import generation_sources, sample_soft_endpoint, refine_typed_graph
from grapher.rewiring_mlp.attributed.joint_typed_edge_model import save_checkpoint, load_checkpoint
from grapher.rewiring_mlp.attributed.joint_typed_edge_data import graph_record, record_hash
from grapher.rewiring_mlp.attributed.joint_typed_edge_training import build_model
from test_typed_degree_perturbation import parents
from test_typed_degree_perturbation_cli import trained_typed
from test_unified_adjacency_diffusion import adjacency_config

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture(autouse=True)
def one_thread():
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_empirical_generation_restores_random_training_subset(tmp_path, monkeypatch, trained_typed):
    from scripts import run_attributed_grapher
    from grapher.rewiring_mlp.attributed import joint_typed_edge_generation as generation
    cfg,path,graphs=trained_typed
    model,checkpoint=load_checkpoint(path,'cpu')
    selected,selection=sample_training_graphs(graphs,2,seed=77)
    assert selection['indices'] != [0,1]
    trained_config=deepcopy(checkpoint['config'])
    trained_config['dataset'].update(training_subset=selection,max_train_graphs=2)
    ckpt=tmp_path/'subset.pt'
    save_checkpoint(ckpt,model,trained_config,checkpoint.get('report',{}),
                    dataset_provenance=checkpoint['dataset_provenance'])
    expected=record_hash([graph_record(g) for g in selected]);calls=[]
    original=generation.build_typed_empirical_sampler
    def capture(settings,pool,**kw):
        calls.append(record_hash([graph_record(g) for g in pool]))
        return original(settings,pool,**kw)
    monkeypatch.setattr(generation,'build_typed_empirical_sampler',capture)
    config=tmp_path/'subset.yaml';save_yaml(cfg,config)
    monkeypatch.setattr(sys,'argv',['generate','--config',str(config),'--checkpoint',str(ckpt),
        '--output-dir',str(tmp_path/'subset_out'),'--num-generate','1','--device','cpu','--seed','42'])
    run_attributed_grapher.main()
    assert calls and set(calls)=={expected}


@pytest.mark.parametrize('method',METHODS)
def test_typed_perturbations_retain_new_adjacency_diffusion(method):
    graphs=[construct_typed_graph(inv,{'max_ordinary_degree':4},np.random.default_rng(i))[0]
            for i,inv in enumerate(parents(False))]
    cfg=adjacency_config()
    cfg['generation'].update(invariant_source='train_empirical_perturbed',invariant_rng_mode='independent',
        degree_perturbation={'method':method,'probability':1.,'failure_policy':'keep_original'})
    model=build_model(cfg,graphs,torch.device('cpu'))
    source,_,_=next(generation_sources(model,graphs,cfg,seed=42,num_generate=1))
    targets,diag=sample_soft_endpoint(model,source,cfg,seed=19)
    assert 'adjacency_metadata' in targets
    assert 'spectra' not in targets
    assert 'diffusion' in diag
    _,report=refine_typed_graph(source,targets,model,cfg,seed=7)
    assert report['typed_preserved'] if 'typed_preserved' in report else report['typed_degree_preserved']


def test_multisize_train_generate_all_options_evaluate(tmp_path,monkeypatch):
    from scripts import train_topology_grapher,run_topology_grapher,evaluate_graph_generation_report
    cfg=load_yaml(ROOT/'configs/experiments/grapher/community_small_spectral_graphlet35_prior_options.yaml')
    degrees=([4,4,2,2,2,2],[3,3,3,3,2,2],[3,3,3,2,2,1],[4,2,2,2,2,2])
    graphs=[nx.havel_hakimi_graph(d) for d in degrees]
    data=tmp_path/'data'
    save_dataset_splits('multi_toy',{'train':graphs,'val':[nx.cycle_graph(6),nx.wheel_graph(6)],'test':[nx.path_graph(6)]},{},data)
    cfg['dataset'].update(name='multi_toy',root=str(data),config_path=None)
    cfg['topology_predictor'].update(hidden_dim=8,edge_dim=8,graph_dim=8,num_layers=1,
        spectral_dim=16,spectral_layers=1,spectral_heads=4,spectral_ff_dim=32,graphlet_dim=16,
        graphlet_dropout=0.,dropout=0.,epochs=1,batch_size=2)
    cfg['summary_diffusion'].update(samples_per_graph=2,paths_per_graph=1)
    cfg['topology_refiner'].update(steps=2,proposal_budget=16,valid_candidate_budget=4)
    cfg['topology_refiner']['prediction_horizon']={'mode':'fixed','k':1,'refresh_on_plateau':False}
    cfg['topology_refiner']['spectral_guidance'].update(expand_on_plateau=False)
    cfg['evaluation']['compute_graphlet_history']=True
    path=tmp_path/'train.yaml';save_yaml(cfg,path);train=tmp_path/'trained'
    monkeypatch.setattr(sys,'argv',['train','--config',str(path),'--output-dir',str(train),'--seed','42','--device','cpu'])
    train_topology_grapher.main()
    ckpt=torch.load(train/'checkpoint.pt',map_location='cpu',weights_only=False)
    assert ckpt['config']['graphlet_prediction']['graphlet_k_min']==3
    assert ckpt['config']['graphlet_prediction']['graphlet_k_max']==5
    fingerprints=[]
    for method in ('empirical',*METHODS):
        variant=deepcopy(cfg)
        if method!='empirical':
            variant['generation'].update(degree_source='train_empirical_perturbed',degree_perturbation={
                'method':method,'probability':1.,'failure_policy':'keep_original'})
        config=tmp_path/(method+'.yaml');save_yaml(variant,config);out=tmp_path/'generation_seed_42'/method
        monkeypatch.setattr(sys,'argv',['generate','--config',str(config),'--checkpoint',str(train/'checkpoint.pt'),
            '--output-dir',str(out),'--num-generate','4','--seed','42','--device','cpu'])
        run_topology_grapher.main()
        report=json.loads((out/'report.json').read_text())
        fingerprints.append(report['parent_degree_fingerprint'])
        assert report['diagnostics']['degree_preservation_rate']==1.
        a=load_pickle(out/'coarse_graphs.pkl');b=load_pickle(out/'topology_refined_graphs.pkl')
        assert all(sorted(dict(x.degree()).values())==sorted(dict(y.degree()).values()) for x,y in zip(a,b))
        monkeypatch.setattr(sys,'argv',['evaluate','--config',str(config),'--generated-dir',str(out),
            '--reference-split','val','--output-dir',str(out/'evaluation_val'),'--num-samples','1','--dpi','50'])
        evaluate_graph_generation_report.main()
        assert (out/'evaluation_val/graph_mmd_metrics.csv').exists()
    assert len(set(fingerprints))==1
    from scripts.summarize_degree_perturbation_evaluations import summarize
    result=summarize(tmp_path,seeds=(42,))
    assert len(result['rows'])==5

@pytest.mark.parametrize('dataset',('qm9','community_small'))
def test_runner_train_once_generate_evaluate_every_prior(tmp_path,dataset):
    import os,subprocess
    fake=tmp_path/'fake_python'
    fake.write_text('''#!/usr/bin/env python3
import json,os,sys
from pathlib import Path
args=sys.argv[1:]
with open(os.environ['COMMAND_LOG'],'a') as f:f.write(json.dumps(args)+'\\n')
if args[0] in ('scripts/train_attributed_grapher.py','scripts/train_topology_grapher.py'):
    out=Path(args[args.index('--output-dir')+1]);out.mkdir(parents=True,exist_ok=True)
    (out/'checkpoint.pt').write_bytes(b'test fixture, not a real model')
''')
    fake.chmod(0o755)
    log=tmp_path/'calls.jsonl';train=tmp_path/'train'
    env={**os.environ,'PYTHON':str(fake),'TRAIN':str(train),'CKPT':str(train/'checkpoint.pt'),
         'OUT_ROOT':str(tmp_path/'outputs'),'SEEDS':'42 43','DEVICE':'cpu','NGEN':'4','NTRAIN':'20',
         'COMMAND_LOG':str(log)}
    env.pop('CFG',None)
    subprocess.run(['bash','scripts/run_prior_options.sh',dataset,'all','all'],cwd=ROOT,env=env,check=True,capture_output=True,text=True)
    calls=[json.loads(row) for row in log.read_text().splitlines()]
    training=[a for a in calls if a[0].startswith('scripts/train_')]
    assert len(training)==1
    if dataset=='qm9':assert training[0].count('--num-train-graphs')==1
    generated=[a for a in calls if a[0].startswith('scripts/run_')]
    evaluated=[a for a in calls if a[0].startswith('scripts/evaluate_')]
    assert len(generated)==10 and len(evaluated)==10
    key='invariant_source' if dataset=='qm9' else 'degree_source'
    assert sum(f'generation.{key}=train_empirical' in a for a in generated)==2
    assert sum(f'generation.{key}=train_empirical_perturbed' in a for a in generated)==8
    for method in METHODS:assert sum(f'generation.degree_perturbation.method={method}' in a for a in generated)==2
    if dataset=='qm9':
        assert all('--require-fcd' in a and '--nspdk-backend' in a and 'raw_valid' in a for a in evaluated)
    else:
        assert calls[-1][0]=='scripts/summarize_degree_perturbation_evaluations.py'

"""Local tests of this add-on; these do NOT train or test the real GDSM model."""
from __future__ import annotations
import copy
import importlib.util
import json
import math
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import yaml
from rdkit import Chem
from rdkit.Chem import QED

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import gdsm_final_suite as suite
import evaluate_qed_research as qe

@pytest.mark.parametrize('dataset,maxnodes,epochs,batch,count',[
 ('community_small',20,10000,32,1024),('ego_small',18,5000,32,1024),
 ('qm9',9,200,64,10000),('zinc',38,200,32,10000)])
def test_profiles(dataset,maxnodes,epochs,batch,count):
    p=yaml.safe_load((ROOT/'configs/experiments/gdsm_final/protocol.yaml').read_text())
    c=yaml.safe_load((ROOT/p['datasets'][dataset]['model_config']).read_text())['gdsm_simple']
    assert c['train']['max_train_graphs'] is None
    assert (c['model']['max_nodes'],c['train']['epochs'],c['train']['batch_size'])==(maxnodes,epochs,batch)
    assert p['datasets'][dataset]['num_generate']==count
    ac=c['extensions']['attributed_categorical']
    assert ac['graphlets']['sizes']==[3,4,5]
    assert ac['graphlets']['counting']=='exact_connected'
    assert ac['graphlets']['size_weights']==[1,1,1]
    assert c['diffusion']['steps']==c['sample']['steps']==500
    assert ac['guidance']['enabled'] is True
    assert ac['guidance']['max_steps_per_event']==2
    assert c['extensions']['degree_preserving_rewiring'] is False

@pytest.fixture
def prepared(tmp_path,monkeypatch):
    shutil.copytree(ROOT/'configs',tmp_path/'configs')
    protocol=yaml.safe_load((tmp_path/suite.PROTOCOL).read_text())
    for ds,info in protocol['datasets'].items():
        prior={'seed':42,'dataset':{'name':info['serialized_dataset'],'seed':42,'root':'outputs/datasets',
                                   'max_train_graphs':8,'max_val_graphs':2,'max_graphs':100000,'build_if_missing':True},
               'degree_generator':{'type':'degree_histogram_vae','checkpoint_path':'pilot/seed_42/checkpoint.pt','device':'auto'},
               'degree_evaluation':{'output_dir':'pilot/evaluation','max_reference_sequences':1024}}
        path=tmp_path/info['degree_prior_base_config'];path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(yaml.safe_dump(prior))
    monkeypatch.chdir(tmp_path)
    args=SimpleNamespace(dataset='all',seeds=[42,43,44],run_tag='gdsm_final_g345',device='cpu',
                         dataset_root=Path('outputs/datasets'),output_root=Path('outputs/baselines'))
    return protocol,args

def test_twelve_distinct_full_split_configs(prepared):
    protocol,args=prepared;ctxs=suite.contexts(args,protocol)
    assert len(ctxs)==12 and len({c['checkpoint'] for c in ctxs})==12
    for c in ctxs:
        d=c['prior'];assert d['seed']==c['seed'];assert d['dataset']['seed']==42
        assert d['dataset']['max_train_graphs'] is None and d['dataset']['max_val_graphs'] is None
        assert d['dataset']['max_graphs']==100000 and not d['dataset']['build_if_missing']
        assert f'seed_{c["seed"]}' in str(c['checkpoint'])
        assert c['model']['gdsm_simple']['extensions']['attributed_categorical']['initialization']['degree_generator']['checkpoint_path']==str(c['checkpoint'])
        if c['seed']!=42:assert '/seed_42/' not in str(c['checkpoint'])

def test_model_config_collision_fails(prepared):
    protocol,args=prepared;c=suite.contexts(args,protocol)[0]
    obj=yaml.safe_load(c['cfg'].read_text());obj['gdsm_simple']['train']['epochs']=1;c['cfg'].write_text(yaml.safe_dump(obj))
    with pytest.raises(RuntimeError,match='collision'):suite.contexts(args,protocol)

def test_typed_prior_is_rejected(prepared):
    protocol,args=prepared;path=Path(protocol['datasets']['qm9']['degree_prior_base_config'])
    obj=yaml.safe_load(path.read_text());obj['degree_generator']['type']='typed_signature';path.write_text(yaml.safe_dump(obj))
    with pytest.raises(ValueError,match='ORDINARY'):suite.contexts(args,protocol)

def test_actual_serialized_dataset_mismatch_rejected(prepared):
    protocol,args=prepared;path=Path(protocol['datasets']['community_small']['degree_prior_base_config'])
    obj=yaml.safe_load(path.read_text());obj['dataset']['name']='wrong';path.write_text(yaml.safe_dump(obj))
    with pytest.raises(ValueError,match='frozen serialized'):suite.contexts(args,protocol)

def test_cli_counts_seeds_and_no_correction(prepared):
    protocol,args=prepared
    for ctx in suite.contexts(args,protocol):
        commands=suite.command_plans(ctx,args,protocol)
        gen=commands['generate'][0];train=commands['train'][0]
        assert '--no-common-config' in train
        assert gen[gen.index('--num-samples')+1]==str(ctx['n'])
        assert gen[gen.index('--generation-seed')+1]==str(ctx['seed'])
        assert '--fcd-use-corrected' not in str(commands) and '--hogdiff-compatible-metrics' not in str(commands)
        evaluation=commands['evaluate'][-1]
        assert '--max-reference' not in evaluation and '--max-train' not in evaluation
        if ctx['molecular']:
            assert evaluation[evaluation.index('--metric-molecule-source')+1]=='raw_valid'
            assert '--require-fcd' in evaluation and '--nspdk-backend' in evaluation
            assert len(commands['qed'])==1
        else:assert evaluation[evaluation.index('--num-samples')+1]=='1024'

def converter(smiles,*,infer_projected_formal_charges=False):
    assert not infer_projected_formal_charges
    if smiles is None:return None,'invalid'
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles),canonical=True,isomericSmiles=False),None

def test_qed_duplicates_and_invalid_denominators():
    ethanol=QED.qed(Chem.MolFromSmiles('CCO'),w=QED.WEIGHT_MEAN)
    benzene=QED.qed(Chem.MolFromSmiles('c1ccccc1'),w=QED.WEIGHT_MEAN)
    m,rows,values,errors=qe.score_graphs(['CCO','CCO','c1ccccc1',None],converter,[.5,.7])
    assert m['num_raw_valid']==3 and m['validity_without_correction']==.75
    assert m['qed_mean']==pytest.approx((2*ethanol+benzene)/3)
    assert m['qed_yield']==pytest.approx((2*ethanol+benzene)/4)
    assert m['qed_unique_valid_mean']==pytest.approx((ethanol+benzene)/2)
    assert rows[-1]['qed'] is None and errors=={'invalid':1}

def test_all_invalid_is_missing_mean_not_zero():
    m,rows,values,errors=qe.score_graphs([None,None],converter,[.5])
    assert m['qed_mean'] is None and m['qed_yield']==0
    assert m['qed_ge_0p5_fraction_valid'] is None and m['qed_ge_0p5_yield_all']==0

def test_single_valid_within_std_missing():
    m,*_=qe.score_graphs(['CCO'],converter,[.5])
    assert m['qed_std_within_run'] is None

def test_disconnected_retained_and_separately_counted():
    m,rows,*_=qe.score_graphs(['CCO.C','CCO'],converter,[.5])
    assert m['num_raw_valid']==2 and m['num_raw_valid_single_component']==1
    assert rows[0]['num_fragments']==2

def test_empty_cohort_rejected():
    with pytest.raises(ValueError,match='Empty'):qe.score_graphs([],converter,[.5])

def test_numeric_leaves_no_per_graph_pseudoreplicates():
    d=suite.numeric_leaves({'metrics':[{'comparison':'model_to_test','degree_mmd':.2}],
                           'per_graph':[.1,.2,.3],'flag':True,'missing':None})
    assert d=={'metrics.model_to_test.degree_mmd':.2,'missing':None}

@pytest.fixture
def qed_integration(tmp_path):
    # Explicit test adapter using SMILES-in-JSON rather than a real repository.
    adapter=tmp_path/'adapter.py'
    adapter.write_text('''import json\nfrom pathlib import Path\nfrom rdkit import Chem\ndef _load_graphs_from_path(path):return json.loads(Path(path).read_text())\ndef _graph_to_canonical_smiles_and_error(s,*,infer_projected_formal_charges=False):\n    if s is None:return None,"invalid"\n    return Chem.MolToSmiles(Chem.MolFromSmiles(s),canonical=True,isomericSmiles=False),None\n''')
    gen=tmp_path/'generated.json';gen.write_text(json.dumps(['CCO','c1ccccc1',None]))
    ref=tmp_path/'reference.json';ref.write_text(json.dumps(['CCO','CCN']))
    report=tmp_path/'molecular_evaluation_metrics.json'
    report.write_text(json.dumps({'metrics':{'metric_molecule_source':'raw_valid','num_generated_graphs':3,
                     'num_valid_generated_molecules':2,'validity_without_correction':2/3}}))
    return SimpleNamespace(generated_graphs=gen,reference_graphs=ref,molecular_report=report,evaluator=adapter,
                           expected_num_generated=3,output_dir=tmp_path/'qed',thresholds=[.5,.7])

def test_qed_end_to_end_adapter(qed_integration):
    out=qe.run(qed_integration)
    assert out['metrics']['reference_num_raw_valid']==2 and out['metrics']['qed_wasserstein_1']>=0
    assert (qed_integration.output_dir/'qed_per_graph.csv').exists()
    assert len(out['source_files']['generated_graphs']['sha256'])==64

def test_qed_incomplete_generation_fails(qed_integration):
    qed_integration.expected_num_generated=10000
    with pytest.raises(ValueError,match='Expected 10000'):qe.run(qed_integration)

def test_qed_corrected_source_fails(qed_integration):
    data=json.loads(qed_integration.molecular_report.read_text());data['metrics']['metric_molecule_source']='corrected_valid'
    qed_integration.molecular_report.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='raw_valid'):qe.run(qed_integration)

def test_qed_valid_count_mismatch_fails(qed_integration):
    data=json.loads(qed_integration.molecular_report.read_text());data['metrics']['num_valid_generated_molecules']=3
    qed_integration.molecular_report.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='cohort mismatch'):qe.run(qed_integration)

def test_qed_saved_cohort_mismatch_fails(qed_integration):
    (qed_integration.molecular_report.parent/'valid_generated.smi').write_text('CCO\n')
    with pytest.raises(ValueError,match='cohort/order'):qe.run(qed_integration)

@pytest.mark.parametrize('threshold',[float('nan'),float('inf'),-1,2])
def test_invalid_threshold_fails(qed_integration,threshold):
    qed_integration.thresholds=[threshold]
    with pytest.raises(ValueError,match='threshold'):qe.run(qed_integration)

@pytest.fixture
def aggregation_fixture(prepared):
    protocol,args=prepared;args.dataset='qm9';ctxs=suite.contexts(args,protocol)
    for ctx in ctxs:
        manifest={'splits':{s:{'sha256':s+'_fixed'} for s in ('train','val','test')},
                  'source_sha256':'same_source','protocol_sha256':'same_protocol'}
        suite.write_json(ctx['resolved']/'manifest.json',manifest)
        main=ctx['gen']/'evaluation_molecules/molecular_evaluation_metrics.json'
        qpath=ctx['gen']/'evaluation_qed/qed_metrics.json'
        suite.write_json(main,{'metrics':{'fcd':ctx['seed']-41.,'num_generated_graphs':10000}})
        suite.write_json(qpath,{'metrics':{'qed_mean':.5+(ctx['seed']-42)*.1,'sometimes_missing':None if ctx['seed']==44 else .2}})
        for stage in ('degree','train','generate','audit','evaluate','qed'):
            files=[main] if stage=='evaluate' else ([qpath] if stage=='qed' else [])
            suite.write_json(ctx['resolved']/f'stage_{stage}.json',{'status':'complete','outputs':{str(p):suite.digest(p) for p in files}})
    return ctxs,args

def test_aggregate_sample_std_and_no_missing_seed_cherry_pick(aggregation_fixture):
    ctxs,args=aggregation_fixture;suite.aggregate(ctxs,args)
    data=json.loads((Path('outputs/final_results')/args.run_tag/'qm9_summary.json').read_text())
    fcd=next(x for x in data if x['metric'].endswith('.fcd'))
    assert fcd['mean']==2. and fcd['sample_std']==1.
    missing=next(x for x in data if x['metric'].endswith('.sometimes_missing'))
    assert missing['n_available']==2 and missing['mean'] is None and missing['sample_std'] is None

def test_aggregate_rejects_inconsistent_splits(aggregation_fixture):
    ctxs,args=aggregation_fixture;p=ctxs[-1]['resolved']/'manifest.json'
    obj=json.loads(p.read_text());obj['splits']['test']['sha256']='different';suite.write_json(p,obj)
    with pytest.raises(RuntimeError,match='differs across seeds'):suite.aggregate(ctxs,args)

def test_aggregate_rejects_modified_report(aggregation_fixture):
    ctxs,args=aggregation_fixture;p=ctxs[0]['gen']/'evaluation_qed/qed_metrics.json'
    p.write_text('{}')
    with pytest.raises(RuntimeError,match='Modified metric file'):suite.aggregate(ctxs,args)

def test_aggregate_requires_all_three_seeds(aggregation_fixture):
    ctxs,args=aggregation_fixture;args.seeds=[42,43]
    with pytest.raises(ValueError,match='exactly seeds'):suite.aggregate(ctxs,args)


def test_source_lock_includes_evaluation_yaml(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    p=Path('configs/experiments/baselines/test.yaml');p.parent.mkdir(parents=True)
    p.write_text('seed: 42\n');first=suite.source_signature()
    p.write_text('seed: 43\n');second=suite.source_signature()
    assert first[0]!=second[0] and first[1]==second[1]==1

def test_required_graphlet_report_and_qed_exports(prepared):
    protocol,args=prepared
    ctx=next(c for c in suite.contexts(args,protocol) if c['molecular'])
    outputs=suite.stage_outputs(ctx,'evaluate')
    assert ctx['gen']/'categorical_metrics.json' in outputs
    assert ctx['gen']/'evaluation_molecules/molecular_evaluation_metrics.json' in outputs
    assert len(suite.stage_outputs(ctx,'qed'))==3

#!/usr/bin/env python3
"""Frozen four-dataset, three-seed experiment launcher for multiscale gdsm_simple.

Run from the existing GraphES project root. This is an orchestration/configuration
add-on, NOT an implementation of the model. Requires the installed categorical
multiscale branch. No data generation, download, split reseeding or output repair.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
import yaml

PROTOCOL=Path('configs/experiments/gdsm_final/protocol.yaml')
STAGES=('resolve','check','degree','train','generate','audit','evaluate','qed','all','aggregate')


def digest(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n');temp.replace(path)


def locked_yaml(path: Path, obj: dict):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if yaml.safe_load(path.read_text())!=obj:
            raise RuntimeError(f'Configuration collision: {path}. Use a NEW --run-tag; do not overwrite old results.')
    else:path.write_text(yaml.safe_dump(obj,sort_keys=False))


def remove_training_caps(obj):
    # Only training/validation caps; do not change preparation settings such as
    # dataset.max_graphs, node limits, source paths, split fractions or data seeds.
    if isinstance(obj,dict):
        return {k:(None if k in {'max_train_graphs','max_val_graphs'} else remove_training_caps(v)) for k,v in obj.items()}
    if isinstance(obj,list):return [remove_training_caps(v) for v in obj]
    return obj


def load_graphs(path: Path):
    # This loader is the same trusted-local-pickle helper used by the project.
    from grapher.utils.networkx_pickle import load_trusted_networkx_pickle
    obj=load_trusted_networkx_pickle(path)
    if isinstance(obj,dict):
        for key in ('graphs','molecular_graphs','generated_graphs'):
            if key in obj:obj=obj[key];break
    graphs=list(obj)
    import networkx as nx
    if not all(isinstance(g,nx.Graph) for g in graphs):raise TypeError(f'Not a graph collection: {path}')
    return graphs


def contexts(args,protocol):
    datasets=list(protocol['datasets']) if args.dataset=='all' else [args.dataset]
    result=[]
    for ds in datasets:
        info=protocol['datasets'][ds]
        for seed in args.seeds:
            run=f'seed_{seed}_{args.run_tag}'
            resolved=Path('outputs/final_configs')/ds/run
            model=yaml.safe_load(Path(info['model_config']).read_text())
            base_prior=Path(info['degree_prior_base_config'])
            if not base_prior.is_file():raise FileNotFoundError(f'Missing existing ordinary DH-VAE config: {base_prior}')
            prior=remove_training_caps(yaml.safe_load(base_prior.read_text()))
            if prior.get('degree_generator',{}).get('type')!='degree_histogram_vae':
                raise ValueError(f'{base_prior}: categorical spectral initialization requires the ORDINARY degree_histogram_vae, not a typed-signature prior')
            serial=info['serialized_dataset']
            if prior.get('dataset',{}).get('name')!=serial:
                raise ValueError(f'{base_prior}: dataset.name is not the frozen serialized dataset {serial}')
            prior['seed']=seed
            prior['dataset'].update(root=str(args.dataset_root),build_if_missing=False,max_train_graphs=None)
            checkpoint=Path('outputs/degree_generators')/args.run_tag/ds/f'seed_{seed}'/'checkpoint.pt'
            prior['degree_generator']['checkpoint_path']=str(checkpoint)
            prior['degree_generator']['device']=args.device
            if 'degree_evaluation' in prior:
                prior['degree_evaluation']['output_dir']=str(checkpoint.parent/'evaluation')
                prior['degree_evaluation']['max_reference_sequences']=None
            opt=model['gdsm_simple'];opt['train']['max_train_graphs']=None
            opt['extensions']['attributed_categorical']['initialization']['degree_generator']['checkpoint_path']=str(checkpoint)
            locked_yaml(resolved/'model.yaml',model);locked_yaml(resolved/'degree_prior.yaml',prior)
            n=info['num_generate'];gid=f'seed_{seed}_n_{n}'
            run_dir=args.output_root/'gdsm_simple'/ds/run
            result.append(dict(dataset=ds,seed=seed,serial=serial,n=n,run=run,gid=gid,model=model,prior=prior,
                               cfg=resolved/'model.yaml',dcfg=resolved/'degree_prior.yaml',resolved=resolved,
                               checkpoint=checkpoint,run_dir=run_dir,gen=run_dir/'generations'/gid,
                               molecular=info['molecular'],info=info))
    return result


def source_signature():
    paths=sorted(set(Path('src').rglob('*.py')) | set(Path('scripts').glob('*.py'))
                 | set(Path('scripts').glob('*.sh')) | set(Path('configs').rglob('*.yaml'))
                 | set(Path('configs').rglob('*.yml')))
    h=hashlib.sha256()
    for path in paths:
        h.update(str(path).encode());h.update(digest(path).encode())
    return h.hexdigest(),len(paths)


def split_signature(ctx,args,with_counts=False):
    result={}
    for split in ('train','val','test'):
        path=args.dataset_root/ctx['serial']/f'{split}.pkl'
        if not path.is_file():raise FileNotFoundError(f'Prepared split required; it will NOT be rebuilt: {path}')
        item={'path':str(path),'sha256':digest(path),'bytes':path.stat().st_size}
        if with_counts:
            graphs=load_graphs(path)
            if not graphs:raise ValueError(f'Empty prepared split: {path}')
            item['num_graphs']=len(graphs)
            item['max_nodes']=max(g.number_of_nodes() for g in graphs)
            if item['max_nodes']>ctx['info']['max_nodes']:
                raise ValueError(f'{path}: max nodes {item["max_nodes"]} exceeds fixed model limit {ctx["info"]["max_nodes"]}; refusing truncation')
            if any(g.is_directed() or g.is_multigraph() for g in graphs):raise ValueError(f'{path}: expected undirected simple graphs')
            del graphs
        result[split]=item
    return result


def environment_info():
    info={'python':sys.version,'platform':platform.platform(),'device_visible':os.environ.get('CUDA_VISIBLE_DEVICES'),
          'eigh_backend':os.environ.get('GDSM_EIGH_BACKEND'),'pythonhashseed':os.environ.get('PYTHONHASHSEED')}
    try:
        import torch
        info.update(torch=torch.__version__,cuda_runtime=torch.version.cuda,cuda_available=torch.cuda.is_available())
        if torch.cuda.is_available():info['gpu']=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except ImportError:pass
    try:
        from rdkit import rdBase
        info['rdkit']=rdBase.rdkitVersion
    except ImportError:pass
    for name,cmd in [('git_commit',['git','rev-parse','HEAD']),('git_status',['git','status','--porcelain']),
                     ('pip_freeze',[sys.executable,'-m','pip','freeze']),('gpu_status',['nvidia-smi'])]:
        try:
            completed=subprocess.run(cmd,text=True,capture_output=True,timeout=30)
            info[name]=completed.stdout.strip() if completed.returncode==0 else None
        except (FileNotFoundError,subprocess.TimeoutExpired):info[name]=None
    return info


def establish_lock(ctx,args,source,env_info):
    manifest=ctx['resolved']/'manifest.json'
    if manifest.exists():
        old=json.loads(manifest.read_text());splits=split_signature(ctx,args)
        for key in splits:
            if splits[key]['sha256']!=old['splits'][key]['sha256']:raise RuntimeError(f'{ctx["dataset"]}: {key} split changed')
        for key,path in [('model_sha256',ctx['cfg']),('prior_config_sha256',ctx['dcfg'])]:
            if digest(path)!=old[key]:raise RuntimeError(f'Changed {key}')
        if old['source_sha256']!=source[0]:raise RuntimeError('Source code changed within this run; use a new --run-tag')
        if old['protocol_sha256']!=digest(PROTOCOL):raise RuntimeError('Final protocol changed within this run; use a new --run-tag')
        # Explicit version/device lock; GPU hardware is recorded but not required
        # to be bit-identical across machines. Exact cross-device reproducibility is not claimed.
        for key in ('python','torch','cuda_runtime','rdkit','eigh_backend'):
            if old['environment'].get(key)!=env_info.get(key):raise RuntimeError(f'Environment changed ({key}); use a new --run-tag')
        if old['device']!=args.device:raise RuntimeError('Requested neural device changed within this run')
        ctx['manifest']=old;return
    splits=split_signature(ctx,args,with_counts=True)
    updates=ctx['model']['gdsm_simple']['train']['epochs']*math.ceil(splits['train']['num_graphs']/ctx['model']['gdsm_simple']['train']['batch_size'])
    obj={'dataset':ctx['dataset'],'serialized_dataset':ctx['serial'],'seed':ctx['seed'],'generation_seed':ctx['seed'],
         'requested_samples':ctx['n'],'run_id':ctx['run'],'splits':splits,'model_sha256':digest(ctx['cfg']),
         'prior_config_sha256':digest(ctx['dcfg']),'source_sha256':source[0],'source_files_hashed':source[1],
         'protocol_sha256':digest(PROTOCOL),'environment':env_info,'device':args.device,
         'expected_model_updates_no_drop_last':updates,'train_full_split':True,'validation_for_model_selection_only':True}
    write_json(manifest,obj);ctx['manifest']=obj
    print(f"{ctx['dataset']} seed={ctx['seed']}: train={splits['train']['num_graphs']}, val={splits['val']['num_graphs']}, test={splits['test']['num_graphs']}, requested={ctx['n']}",flush=True)


def command_plans(ctx,args,protocol):
    py=sys.executable;ds=ctx['dataset'];s=str(ctx['seed']);g=str(ctx['gen']);n=str(ctx['n']);serial=ctx['serial']
    base=[py,'scripts/run_gdsm_simple_baseline.py','--dataset',ds,'--no-common-config','--wrapper-config',str(ctx['cfg']),
          '--dataset-root',str(args.dataset_root),'--serialized-dataset',serial,'--output-root',str(args.output_root),
          '--seed-id',s,'--run-id',ctx['run'],'--device',args.device]
    reference=str(args.dataset_root/serial/'test.pkl')
    plans={
      'check':[[py,'scripts/check_gdsm_categorical_data.py','--wrapper-config',str(ctx['cfg']),'--dataset-root',str(args.dataset_root),'--serialized-dataset',serial]],
      'degree':[[py,'scripts/train_degree_generator.py','--config',str(ctx['dcfg'])]],
      'train':[base+['--stage','train']],
      'generate':[base+['--stage','generate','--generation-seed',s,'--generation-id',ctx['gid'],'--num-samples',n]],
      'audit':[[py,'scripts/audit_gdsm_categorical.py','--generated-dir',g]],
      'evaluate':[[py,'scripts/evaluate_gdsm_categorical.py','--generated-dir',g,'--reference-graphs',reference,'--seed',str(protocol['evaluation_seed'])]],
      'qed':[]}
    if ctx['molecular']:
        plans['evaluate'].append([py,'scripts/evaluate_generated_molecules.py','--generated-graphs',g+'/molecular_graphs.pkl',
          '--dataset-root',str(args.dataset_root),'--dataset',serial,'--reference-split','test','--train-split','train',
          '--require-fcd','--nspdk-backend','eden','--metric-molecule-source','raw_valid','--output-dir',g+'/evaluation_molecules'])
        plans['qed']=[[py,'scripts/evaluate_qed_research.py','--generated-graphs',g+'/molecular_graphs.pkl',
          '--reference-graphs',reference,'--molecular-report',g+'/evaluation_molecules/molecular_evaluation_metrics.json',
          '--expected-num-generated',n,'--output-dir',g+'/evaluation_qed','--thresholds']+list(map(str,protocol['qed_thresholds']))]
    else:
        # Retain the frozen topology evaluator and explicitly supply full graph
        # files. The native evaluator owns the semantics of --num-samples.
        plans['evaluate'].append([py,'scripts/evaluate_graph_generation_report.py','--config',f'configs/experiments/baselines/{ds}_evaluation.yaml',
          '--generated-dir',g,'--generated-graphs',g+'/base_graphs.pkl','--base-graphs',g+'/initial_graphs.pkl',
          '--generated-stage','gdsm_final','--reference-split','test','--generic-mmd-protocol','graphrnn',
          '--num-samples',n,'--output-dir',g+'/evaluation_topology'])
    return plans


def validate_implementation(ctx,args):
    from grapher.models.base import DatasetReference,RunSpec,TrainRequest
    from grapher.models.gdsm_simple.wrapper import GDSMSimpleWrapper
    from grapher.models.gdsm_simple.categorical.config import resolve
    if not Path('src/grapher/models/gdsm_simple/categorical/multiscale.py').is_file():
        raise RuntimeError('Install the existing graphlet-3/4/5 categorical support branch first. This add-on does not implement it.')
    request=TrainRequest(RunSpec('gdsm_simple',ctx['dataset'],ctx['run'],ctx['seed']),
                         DatasetReference(ctx['serial'],args.dataset_root),config_path=ctx['cfg'])
    cfg=resolve(GDSMSimpleWrapper()._options(request))
    if list(cfg['graphlets']['sizes'])!=[3,4,5]:raise RuntimeError('Resolved model is NOT using all graphlet orders 3,4,5')
    import torch
    if args.device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA was requested but is unavailable')
    if ctx['molecular']:
        from rdkit.Chem import QED
        if not hasattr(QED,'WEIGHT_MEAN'):raise RuntimeError('RDKit QED average weights unavailable')
        if importlib.util.find_spec('fcd_torch') is None:raise RuntimeError('fcd_torch is required for final molecular evaluation')
        if importlib.util.find_spec('eden') is None:raise RuntimeError('EDeN NSPDK is required; no proxy fallback is allowed')
    else:
        evaluation_path=Path(f'configs/experiments/baselines/{ctx["dataset"]}_evaluation.yaml')
        evaluation=yaml.safe_load(evaluation_path.read_text())
        data=evaluation.get('dataset',{})
        if isinstance(data,dict):
            if 'root' in data and Path(data['root']).resolve()!=args.dataset_root.resolve():
                raise RuntimeError(f'{evaluation_path}: evaluation data root differs from training data root')
            if 'name' in data and data['name']!=ctx['serial']:
                raise RuntimeError(f'{evaluation_path}: evaluation serialized dataset differs from training')
        orca=Path(os.environ.get('ORCA_EXEC','/home/quang/orca/orca.out'))
        if not orca.is_file() or not os.access(orca,os.X_OK):raise RuntimeError(f'ORCA executable not found/executable: {orca}')


def verify_output_count(ctx):
    path=ctx['gen']/('molecular_graphs.pkl' if ctx['molecular'] else 'base_graphs.pkl')
    graphs=load_graphs(path)
    if len(graphs)!=ctx['n']:raise RuntimeError(f'{path}: expected {ctx["n"]} samples, found {len(graphs)}; refusing incomplete final evaluation')


def stage_outputs(ctx,stage):
    if stage=='degree':return [ctx['checkpoint']]
    if stage=='train':
        return [p for ext in ('*.pt','*.pth') for p in ctx['run_dir'].rglob(ext) if 'generations' not in p.parts]
    if stage=='generate':return [ctx['gen']/('molecular_graphs.pkl' if ctx['molecular'] else 'base_graphs.pkl')]
    if stage=='evaluate':
        folder=ctx['gen']/('evaluation_molecules' if ctx['molecular'] else 'evaluation_topology')
        main=([folder/'molecular_evaluation_metrics.json'] if ctx['molecular'] else list(folder.glob('*.json')))
        if not main:raise RuntimeError('Native evaluator produced no JSON report')
        return main+[ctx['gen']/'categorical_metrics.json']
    if stage=='qed' and ctx['molecular']:
        return [ctx['gen']/'evaluation_qed'/name for name in ('qed_metrics.json','qed_per_graph.csv','qed_reference_per_graph.csv')]
    return []


def run_stage(ctx,stage,commands,args):
    if not commands:return
    stamp=ctx['resolved']/f'stage_{stage}.json'
    if stage!='check' and stamp.exists():
        old=json.loads(stamp.read_text())
        if old.get('status')=='complete':
            for name,h in old.get('outputs',{}).items():
                if not Path(name).exists() or digest(Path(name))!=h:raise RuntimeError(f'Completed-stage artifact changed: {name}')
            print(f"SKIP verified completed {ctx['dataset']}/{ctx['seed']}/{stage}",flush=True);return
    if stage=='degree' and ctx['checkpoint'].exists():raise RuntimeError(f'Prior exists without a matching final-suite completion stamp: {ctx["checkpoint"]}; do not silently reuse a pilot prior')
    if stage=='train' and stage_outputs(ctx,'train'):raise RuntimeError('Model checkpoint exists without a matching completion stamp; use a new --run-tag for interrupted/untracked training')
    if stage in ('audit','evaluate','qed'):verify_output_count(ctx)
    log_dir=ctx['resolved']/'logs';log_dir.mkdir(exist_ok=True)
    started=time.time();record={'stage':stage,'status':'running','commands':commands,'started_unix':started,'outputs':{}}
    write_json(stamp,record)
    env=os.environ.copy();env['PYTHONHASHSEED']=str(ctx['seed']);env['PYTHONUNBUFFERED']='1'
    try:
        with (log_dir/f'{stage}.log').open('a') as log:
            for command in commands:
                text=shlex.join(command);print(text,flush=True);log.write('\n$ '+text+'\n');log.flush()
                process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,env=env)
                assert process.stdout is not None
                for line in process.stdout:sys.stdout.write(line);sys.stdout.flush();log.write(line);log.flush()
                if process.wait()!=0:raise subprocess.CalledProcessError(process.returncode,command)
        if stage=='generate':verify_output_count(ctx)
        outputs=stage_outputs(ctx,stage)
        if stage in ('degree','train','generate','evaluate','qed') and not outputs:raise RuntimeError(f'{stage} produced no expected artifacts')
        if any(not p.is_file() for p in outputs):raise RuntimeError(f'Missing {stage} output')
        record.update(status='complete',wall_seconds=time.time()-started,outputs={str(p):digest(p) for p in outputs})
        write_json(stamp,record)
    except BaseException as exc:
        record.update(status='failed',wall_seconds=time.time()-started,error=f'{type(exc).__name__}: {exc}')
        write_json(stamp,record);raise


def numeric_leaves(obj,prefix=''):
    # Lists typically contain per-graph data, not run-level metrics. Do not
    # turn them into pseudoreplicates when aggregating training seeds.
    result={}
    if isinstance(obj,dict):
        for key,value in obj.items():result.update(numeric_leaves(value,f'{prefix}.{key}' if prefix else str(key)))
    elif isinstance(obj,list):
        # Some evaluators store a small table of named comparison records.
        # Never aggregate unnamed per-graph samples as independent runs.
        for item in obj:
            if isinstance(item,dict):
                label=next((item.get(k) for k in ('comparison','metric_name') if isinstance(item.get(k),str)),None)
                if label is not None:result.update(numeric_leaves(item,f'{prefix}.{label}'))
    elif obj is None:result[prefix]=None
    elif isinstance(obj,(int,float)) and not isinstance(obj,bool):
        result[prefix]=float(obj) if math.isfinite(float(obj)) else None
    return result


def aggregate(context_list,args):
    if sorted(set(args.seeds))!=[42,43,44]:raise ValueError('Final aggregation requires exactly seeds 42,43,44')
    collected=[]
    fingerprints={}
    for ctx in context_list:
        manifest=json.loads((ctx['resolved']/'manifest.json').read_text())
        signature={'splits':{k:v['sha256'] for k,v in manifest['splits'].items()},
                   'source':manifest['source_sha256'],'protocol':manifest['protocol_sha256']}
        if ctx['dataset'] in fingerprints and fingerprints[ctx['dataset']]!=signature:
            raise RuntimeError(f'{ctx["dataset"]}: data/code/protocol differs across seeds; refusing aggregation')
        fingerprints[ctx['dataset']]=signature
        for stage in ['degree','train','generate','audit','evaluate']+(['qed'] if ctx['molecular'] else []):
            stamp=ctx['resolved']/f'stage_{stage}.json'
            if not stamp.exists() or json.loads(stamp.read_text()).get('status')!='complete':raise RuntimeError(f'Missing completed {stage}: {ctx["dataset"]}, seed {ctx["seed"]}')
            for filename,h in json.loads(stamp.read_text())['outputs'].items():
                if digest(Path(filename))!=h:raise RuntimeError(f'Modified metric file: {filename}')
        reports=list(ctx['gen'].glob('categorical_metrics.json'))
        if ctx['molecular']:
            reports += [ctx['gen']/'evaluation_molecules/molecular_evaluation_metrics.json',ctx['gen']/'evaluation_qed/qed_metrics.json']
        else:reports += sorted((ctx['gen']/'evaluation_topology').glob('*.json'))
        row={'dataset':ctx['dataset'],'seed':ctx['seed'],'metrics':{},'files':[]}
        for path in reports:
            obj=json.loads(path.read_text());part=obj.get('metrics',obj) if isinstance(obj,dict) else {}
            row['metrics'].update(numeric_leaves(part,str(path.relative_to(ctx['gen']))))
            row['files'].append(str(path))
        collected.append(row)
    summaries=[]
    for ds in sorted({c['dataset'] for c in collected}):
        rows=[c for c in collected if c['dataset']==ds]
        if sorted(r['seed'] for r in rows)!=[42,43,44]:raise ValueError(f'Missing/duplicate seed in {ds}')
        keys=sorted(set().union(*(r['metrics'].keys() for r in rows)))
        for key in keys:
            values=[r['metrics'].get(key) for r in rows];present=[v for v in values if v is not None]
            complete=len(present)==3
            summaries.append({'dataset':ds,'metric':key,'n_available':len(present),
                              'mean':statistics.mean(present) if complete else None,
                              'sample_std':statistics.stdev(present) if complete else None,
                              **{f'seed_{r["seed"]}':r['metrics'].get(key) for r in rows}})
    out=Path('outputs/final_results')/args.run_tag
    label=args.dataset
    write_json(out/f'{label}_per_seed.json',collected);write_json(out/f'{label}_summary.json',summaries)
    with (out/f'{label}_summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['dataset','metric','n_available','mean','sample_std','seed_42','seed_43','seed_44'])
        writer.writeheader();writer.writerows(summaries)
    print(f'Saved {out}/{label}_summary.csv')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['all','community_small','ego_small','qm9','zinc'],default='all')
    p.add_argument('--stage',choices=STAGES,default='all');p.add_argument('--seeds',type=int,nargs='+',default=[42,43,44])
    p.add_argument('--device',default='cuda:0');p.add_argument('--dataset-root',type=Path,default=Path('outputs/datasets'))
    p.add_argument('--output-root',type=Path,default=Path('outputs/baselines'));p.add_argument('--run-tag',default='gdsm_final_g345')
    p.add_argument('--dry-run',action='store_true',help='Resolve YAMLs and print commands only; no validation/training/evaluation')
    args=p.parse_args()
    if any(s not in (42,43,44) for s in args.seeds) or len(set(args.seeds))!=len(args.seeds):p.error('Use distinct seeds from 42,43,44')
    if not args.run_tag or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.run_tag):p.error('Use an alphanumeric/underscore/hyphen run tag')
    if not PROTOCOL.is_file():p.error('Run from the existing GraphES project root after installing this add-on')
    protocol=yaml.safe_load(PROTOCOL.read_text());os.environ.setdefault('GDSM_EIGH_BACKEND','cpu')
    ctxs=contexts(args,protocol)
    if args.stage=='resolve':
        for c in ctxs:print(c['cfg'])
        return
    if args.dry_run:
        for c in ctxs:
            plans=command_plans(c,args,protocol)
            stages=['check','degree','train','generate','audit','evaluate','qed'] if args.stage=='all' else [args.stage]
            for stage in stages:
                for command in plans.get(stage,[]):print(shlex.join(command))
        return
    if args.stage=='aggregate':aggregate(ctxs,args);return
    source=source_signature();env_info=environment_info()
    # Audit all selected cases before starting any expensive training.
    for c in ctxs:
        establish_lock(c,args,source,env_info);validate_implementation(c,args)
    for c in ctxs:
        plans=command_plans(c,args,protocol)
        stages=['check','degree','train','generate','audit','evaluate','qed'] if args.stage=='all' else [args.stage]
        for stage in stages:run_stage(c,stage,plans[stage],args)
    if args.stage=='all' and sorted(args.seeds)==[42,43,44]:aggregate(ctxs,args)
if __name__=='__main__':main()

#!/usr/bin/env python3
"""Evaluate QED on exactly the raw-valid molecular cohort; never repair or resample.

Primary score retains duplicate valid draws. Invalid graphs have missing QED, not
QED=0. The separately named qed_yield divides the sum of valid QED by ALL outputs.
Uses the project's existing raw graph-to-SMILES converter and cross-checks the
molecular evaluation report. RDKit default average weights are fixed explicitly.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Callable
import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import QED
from scipy.stats import wasserstein_distance


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def load_converter(path: Path):
    spec=importlib.util.spec_from_file_location('_gdsm_molecular_evaluator',path)
    if spec is None or spec.loader is None: raise RuntimeError(f'Cannot import {path}')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    for name in ('_load_graphs_from_path','_graph_to_canonical_smiles_and_error'):
        if not callable(getattr(module,name,None)):
            raise RuntimeError(f'{path} lacks {name}; update adapter for your evaluator, do not change chemistry silently.')
    return module


def describe(values: list[float]) -> dict[str, Any]:
    if not values:
        return dict(count=0,mean=None,std=None,median=None,q10=None,q90=None,min=None,max=None)
    a=np.asarray(values,dtype=np.float64)
    if not np.isfinite(a).all(): raise ValueError('Non-finite QED')
    return dict(count=len(a),mean=float(a.mean()),std=float(a.std(ddof=1)) if len(a)>1 else None,
                median=float(np.median(a)),q10=float(np.quantile(a,.1)),q90=float(np.quantile(a,.9)),
                min=float(a.min()),max=float(a.max()))


def score_graphs(graphs: list, convert: Callable, thresholds: list[float]):
    rows=[];valid=[];unique={};connected=[];errors=Counter();cache={}
    for index,graph in enumerate(graphs):
        smiles,error=convert(graph,infer_projected_formal_charges=False)
        row={'index':index,'raw_valid':smiles is not None,'canonical_smiles':smiles or '',
             'qed':None,'num_fragments':None,'error':error or ''}
        if smiles is None:
            errors[error or 'UnspecifiedInvalidGraph']+=1;rows.append(row);continue
        if smiles not in cache:
            mol=Chem.MolFromSmiles(smiles)
            if mol is None: raise RuntimeError(f'Raw-valid SMILES failed RDKit parsing at graph {index}: {smiles}')
            value=float(QED.qed(mol,w=QED.WEIGHT_MEAN))
            if not math.isfinite(value) or not 0.<=value<=1.:
                raise RuntimeError(f'Invalid QED at graph {index}: {value}')
            cache[smiles]=(value,len(Chem.GetMolFrags(mol)))
        value,nfrag=cache[smiles];valid.append(value);unique[smiles]=value
        if nfrag==1: connected.append(value)
        row.update(qed=value,num_fragments=nfrag);rows.append(row)
    n=len(graphs);v=len(valid)
    if n==0: raise ValueError('Empty graph cohort')
    stats=describe(valid)
    metrics={'num_generated_graphs':n,'num_raw_valid':v,'validity_without_correction':v/n,
             'qed_mean':stats['mean'],'qed_std_within_run':stats['std'],'qed_median':stats['median'],
             'qed_q10':stats['q10'],'qed_q90':stats['q90'],'qed_unique_valid_mean':describe(list(unique.values()))['mean'],
             'num_unique_raw_valid':len(unique),'qed_yield':float(sum(valid)/n),
             'num_raw_valid_single_component':len(connected),'valid_single_component_yield':len(connected)/n,
             'qed_single_component_mean':describe(connected)['mean']}
    for threshold in thresholds:
        key=f'{threshold:g}'.replace('.','p');count=sum(q>=threshold for q in valid)
        metrics[f'qed_ge_{key}_fraction_valid']=count/v if v else None
        metrics[f'qed_ge_{key}_yield_all']=count/n
    return metrics,rows,valid,dict(errors)


def run(args):
    if any(not math.isfinite(t) or not 0<=t<=1 for t in args.thresholds):
        raise ValueError('QED thresholds must be finite and in [0,1]')
    module=load_converter(args.evaluator)
    graphs=module._load_graphs_from_path(args.generated_graphs)
    if len(graphs)!=args.expected_num_generated:
        raise ValueError(f'Expected {args.expected_num_generated} attempted samples, found {len(graphs)}')
    metrics,rows,values,errors=score_graphs(graphs,module._graph_to_canonical_smiles_and_error,args.thresholds)
    report=json.loads(args.molecular_report.read_text())
    original=report['metrics']
    if original.get('metric_molecule_source')!='raw_valid':
        raise ValueError('The molecular report must explicitly use raw_valid, not corrected molecules.')
    for original_key,new_key in [('num_generated_graphs','num_generated_graphs'),('num_valid_generated_molecules','num_raw_valid')]:
        if int(original[original_key])!=metrics[new_key]:
            raise ValueError(f'QED/molecular-evaluation cohort mismatch for {original_key}')
    if not math.isclose(float(original['validity_without_correction']),metrics['validity_without_correction'],abs_tol=1e-12):
        raise ValueError('Native validity differs from molecular evaluation')
    raw_smiles=args.molecular_report.parent/'valid_generated.smi'
    if raw_smiles.exists():
        saved=[s.strip() for s in raw_smiles.read_text().splitlines() if s.strip()]
        computed=[r['canonical_smiles'] for r in rows if r['raw_valid']]
        if saved!=computed: raise ValueError('QED cohort/order differs from valid_generated.smi')
    del graphs
    refs=module._load_graphs_from_path(args.reference_graphs)
    ref_metrics,ref_rows,ref_values,ref_errors=score_graphs(refs,module._graph_to_canonical_smiles_and_error,args.thresholds)
    metrics.update(reference_num_graphs=len(refs),reference_num_raw_valid=len(ref_values),reference_qed_mean=ref_metrics['qed_mean'],
                   qed_mean_minus_reference=(metrics['qed_mean']-ref_metrics['qed_mean']) if values and ref_values else None,
                   qed_wasserstein_1=float(wasserstein_distance(values,ref_values)) if values and ref_values else None)
    out={'schema_version':1,'metric_molecule_source':'raw_valid','qed_definition':'RDKit QED.qed(mol, w=QED.WEIGHT_MEAN)',
         'rdkit_version':rdBase.rdkitVersion,'weights':list(QED.WEIGHT_MEAN),'thresholds':args.thresholds,
         'threshold_interpretation':'descriptive cutoffs only, not proof of activity, safety, or clinical usefulness',
         'primary_weighting':'all raw-valid generated draws; duplicates retained; invalid graphs excluded from mean',
         'yield_definition':'sum(QED of raw-valid outputs) / total attempted outputs; distinct from mean QED',
         'stereochemistry':'uses existing evaluator canonical isomericSmiles=False convention',
         'disconnected_policy':'primary follows existing raw sanitization cohort; single-component subset reported separately; no largest-fragment extraction',
         'source_files':{k:{'path':str(p),'sha256':sha256(p)} for k,p in [('generated_graphs',args.generated_graphs),('reference_graphs',args.reference_graphs),('molecular_report',args.molecular_report),('converter',args.evaluator)]},
         'metrics':metrics,'reference_metrics':ref_metrics,'generated_invalid_errors':errors,'reference_invalid_errors':ref_errors}
    args.output_dir.mkdir(parents=True,exist_ok=True)
    target=args.output_dir/'qed_metrics.json';temp=target.with_suffix('.json.tmp')
    temp.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');temp.replace(target)
    for filename,records in [('qed_per_graph.csv',rows),('qed_reference_per_graph.csv',ref_rows)]:
        with (args.output_dir/filename).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['index','raw_valid','canonical_smiles','qed','num_fragments','error'])
            writer.writeheader();writer.writerows(records)
    print(json.dumps(metrics,indent=2,allow_nan=False))
    print(f'Saved {target}')
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--generated-graphs',type=Path,required=True)
    p.add_argument('--reference-graphs',type=Path,required=True)
    p.add_argument('--molecular-report',type=Path,required=True)
    p.add_argument('--expected-num-generated',type=int,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--evaluator',type=Path,default=Path('scripts/evaluate_generated_molecules.py'))
    p.add_argument('--thresholds',type=float,nargs='+',default=[.5,.7])
    args=p.parse_args()
    if args.expected_num_generated<1:p.error('--expected-num-generated must be positive')
    run(args)
if __name__=='__main__':main()

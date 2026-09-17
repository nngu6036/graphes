#!/usr/bin/env python
"""Profile exact multi-order counting on a small fixed training subset; no fitting or test data."""
import argparse,json,time
from collections import defaultdict
from pathlib import Path
import numpy as np
from grapher.models.base import DatasetReference,RunSpec,TrainRequest
from grapher.models.gdsm_simple.wrapper import GDSMSimpleWrapper,_graphs
from grapher.models.gdsm_simple.categorical.config import resolve
from grapher.models.gdsm_simple.categorical.data import encode_graph
from grapher.models.gdsm_simple.categorical.multiscale import count_multi
from grapher.rewiring_mlp.attributed.data import GraphCategoryVocabulary

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wrapper-config',type=Path,required=True);p.add_argument('--graphs',type=Path,required=True)
    p.add_argument('--max-graphs',type=int,default=32);p.add_argument('--output-json',type=Path)
    args=p.parse_args()
    if args.max_graphs<1:p.error('--max-graphs must be positive')
    req=TrainRequest(RunSpec('gdsm_simple','attributed','profile',42),DatasetReference('attributed',Path('outputs/datasets')),config_path=args.wrapper_config)
    op=GDSMSimpleWrapper()._options(req);cfg=resolve(op);gc=cfg['graphlets']
    graphs=_graphs(args.graphs)[:args.max_graphs];vocab=GraphCategoryVocabulary.from_graphs(graphs,cfg['categories'])
    times=[];counts=defaultdict(list);patterns=defaultdict(set)
    for g in graphs:
        x,e=encode_graph(g,vocab,op['model']['max_nodes']);start=time.monotonic()
        c=count_multi(x,e,gc['sizes'],limit=gc['max_connected_subsets']);times.append(time.monotonic()-start)
        for k,b in c.items():counts[k].append(sum(b.values()));patterns[k].update(b)
    report={'status':'passed','source_graphs':str(args.graphs),'graphs_profiled':len(graphs),
            'counting':'exact_connected_induced','seconds_total':sum(times),'seconds_mean':float(np.mean(times)),
            'note':'Small prefix diagnostic with canonicalization cache; not an estimate of full preprocessing or GPU runtime',
            'per_order':{str(k):{'mean_connected_subsets':float(np.mean(values)),'max_connected_subsets':int(max(values)),
                                  'classes_in_profile_subset':len(patterns[k])} for k,values in counts.items()}}
    if args.output_json:
        args.output_json.parent.mkdir(parents=True,exist_ok=True);args.output_json.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()

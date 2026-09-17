#!/usr/bin/env python
"""Check frozen splits without preparing data, deriving a validation vocabulary or training."""
import argparse
import json
from pathlib import Path
from collections import Counter
from grapher.models.base import DatasetReference,RunSpec,TrainRequest
from grapher.models.gdsm_simple.wrapper import GDSMSimpleWrapper,_graphs
from grapher.models.gdsm_simple.categorical.config import resolve
from grapher.models.gdsm_simple.categorical.data import encode_graph
from grapher.rewiring_mlp.attributed.data import GraphCategoryVocabulary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wrapper-config',type=Path,required=True)
    p.add_argument('--dataset-root',type=Path,default=Path('outputs/datasets'))
    p.add_argument('--serialized-dataset',required=True)
    args=p.parse_args()
    ref=DatasetReference('attributed',args.dataset_root,args.serialized_dataset).require_prepared()
    request=TrainRequest(RunSpec('gdsm_simple','attributed','check',42),ref,config_path=args.wrapper_config)
    options=GDSMSimpleWrapper()._options(request);cfg=resolve(options)
    train=_graphs(ref.split_paths['train']);v=GraphCategoryVocabulary.from_graphs(train,cfg['categories'])
    max_nodes=options['model']['max_nodes'] or max(map(len,train));counts={}
    for split in ('train','val'):
        graphs=train if split=='train' else _graphs(ref.split_paths[split])
        for g in graphs:encode_graph(g,v,max_nodes)
        counts[split]={'graphs':len(graphs),'sizes':dict(sorted(Counter(map(len,graphs)).items()))}
    # Test file exists and is hashed, but no test samples are examined at preflight.
    print(json.dumps({'status':'passed','split_fingerprint':ref.fingerprint(),'checked':counts,
                      'category_vocabulary':v.to_dict(),'test_content_read':False,'max_nodes':max_nodes,
                      'ordinary_degree_checkpoint':cfg['initialization']['degree_generator'].get('checkpoint_path'),
                      'degree_prior_is_not_a_hard_constraint':True},indent=2))

if __name__=='__main__':main()

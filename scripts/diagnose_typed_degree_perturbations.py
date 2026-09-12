#!/usr/bin/env python
"""Audit all four typed prior kernels; no denoising or rewiring is performed.

An optional joint checkpoint enforces its fixed signature vocabulary. Without a
checkpoint, no claim of joint-encoder support is made. Only training invariants
are used; the existing splits are never rebuilt.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path

from grapher.data.sampling import restore_training_graphs
from grapher.models.dhvae_hh.degree_perturbation import METHODS, DegreePerturbationError
from grapher.rewiring_mlp.attributed.typed_prior import build_typed_empirical_sampler
from grapher.rewiring_mlp.attributed.joint_typed_edge_data import load_splits,graph_from_record,graph_record
from grapher.rewiring_mlp.molecular.graph_io import is_valid_molecular_graph,require_rdkit
from grapher.utils.io import load_yaml,apply_config_overrides,save_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--checkpoint',default=None,help='Optional joint typed checkpoint; enforces its signature support.')
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--methods',nargs='+',choices=METHODS,default=list(METHODS))
    parser.add_argument('--num-samples',type=int,default=64)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--probability',type=float,default=1.)
    parser.add_argument('--set',dest='overrides',action='append',default=[])
    args=parser.parse_args()
    if args.num_samples<1:parser.error('--num-samples must be positive')
    config=load_yaml(args.config);apply_config_overrides(config,args.overrides)
    if config.get('pipeline',{}).get('stage') not in ('attributed','attributed_topology','molecular'):
        raise ValueError('Use an attributed config; Community-small uses diagnose_degree_perturbations.py.')
    splits,provenance=load_splits(config)
    vectorizer=None;checkpoint=None
    if args.checkpoint:
        from grapher.rewiring_mlp.attributed.joint_typed_edge_model import load_checkpoint
        model,checkpoint=load_checkpoint(args.checkpoint,'cpu');vectorizer=model.vectorizer
        if checkpoint.get('dataset_provenance',{}).get('fingerprint') != provenance['fingerprint']:
            raise ValueError('Dataset fingerprint does not match the joint checkpoint.')
    subset_config=checkpoint['config']['dataset'] if checkpoint is not None else config['dataset']
    train=restore_training_graphs(splits['train'],subset_config)
    cat=config.get('categorical_state',{})
    node_attribute=cat.get('node_attribute','atomic_num');edge_attribute=cat.get('edge_attribute','bond_type')
    if vectorizer is not None and tuple(cat['edge_categories'])!=vectorizer.vocabulary.edge_types:
        raise ValueError('Config/checkpoint typed edge vocabulary differs.')
    require_valid=config.get('generation',{}).get('require_rdkit_source_validity',True)
    if require_valid:require_rdkit()
    def valid(graph):
        return is_valid_molecular_graph(graph_from_record(graph_record(graph,node_attribute,edge_attribute)))
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    report={'training_only':True,'checkpoint_loaded':args.checkpoint is not None,'dataset_provenance':provenance,'reports':{}}
    parents=[]
    for method in args.methods:
        cfg=deepcopy(config);gc=cfg.setdefault('generation',{})
        gc.update(invariant_source='train_empirical_perturbed',invariant_rng_mode='independent')
        gc['degree_perturbation']={**gc.get('degree_perturbation',{}),'method':method,
                                 'probability':args.probability,'failure_policy':'keep_original'}
        sampler=build_typed_empirical_sampler(cfg,train,seed=args.seed,edge_types=tuple(cat['edge_categories']),
            node_attribute=node_attribute,edge_attribute=edge_attribute,vectorizer=vectorizer,
            graph_validator=valid if require_valid else None)
        errors=[]
        for _ in range(args.num_samples):
            try:sampler.sample()
            except DegreePerturbationError as exc:
                # Coverage audit counts failed outputs; production generation fails.
                errors.append(str(exc))
        prior=sampler.report();prior['output_failures']=errors
        parents.append(prior['parent_typed_fingerprint'])
        save_json(prior,out/(method+'.json'))
        report['reports'][method]={k:v for k,v in prior.items() if k not in ('records','parent_exclusions')}
        print(f"{method}: changed={prior['changed_fraction']:.4f} novel={prior['novel_typed_fraction']:.4f} "
              f"fallbacks={prior['num_identity_fallbacks']} failed_outputs={len(errors)}",flush=True)
    report['parents_identical_across_methods']=len(set(parents))==1
    if not report['parents_identical_across_methods']:raise AssertionError('Typed parent pairing failed.')
    save_json(report,out/'report.json')


if __name__=='__main__':main()

#!/usr/bin/env python
"""Additional induced-histogram diagnostics. Attributed evaluation uses the saved vocabulary.

Read the actual saved graphs (including raw molecular graphs); do not silently
sanitize, filter, relabel categories, or replace the existing FCD/NSPDK evaluator.
"""
from pathlib import Path
import argparse
import numpy as np
from grapher.utils.io import load_yaml, load_pickle
from grapher.rewiring_mlp.generic.induced_graphlets import InducedGraphletSpec, extract_histogram
from grapher.rewiring_mlp.attributed.data import GraphletBasis
from grapher.rewiring_mlp.attributed.induced_graphlets import (
    extract_histogram as extract_attributed_histogram,
    metadata as attributed_graphlet_metadata,
)
from grapher.rewiring_mlp.generic.joint_checkpointing import atomic_json, file_sha256
from grapher.rewiring_mlp.attributed.induced_graphlets import wants_attributed_histogram, validate_model_graphlets
from scipy.spatial.distance import cdist


def features(graphs, spec, *, attributed_basis=None):
    k = int(attributed_basis.sizes[0]) if attributed_basis is not None else spec.k
    valid = [g for g in graphs if len(g) >= k]
    if not valid:
        raise ValueError(f'No graphs with n>={k}; graphlet comparison is undefined.')
    extractor = (lambda g: extract_attributed_histogram(g, attributed_basis)) if attributed_basis is not None else (lambda g: extract_histogram(g, spec))
    return np.stack([extractor(g) for g in valid]), len(graphs)-len(valid)


def kernel_mean(a, b, sigma):
    # Chunk both axes to bound pairwise storage for molecular datasets.
    total = 0.0
    for i in range(0, len(a), 128):
        for j in range(0, len(b), 128):
            # Avoid an O(batch^2 * vocabulary_width) temporary for large labeled vocabularies.
            tv = 0.5*cdist(a[i:i+128], b[j:j+128], metric='cityblock')
            total += np.exp(-tv/sigma).sum()
    return float(total/(len(a)*len(b)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--generated-graphs', required=True)
    p.add_argument('--checkpoint', help='Required for attributed graphlets; reuse the TRAIN-only fitted vocabulary.')
    p.add_argument('--reference-split', choices=('val','test','train'), default='val')
    p.add_argument('--k', type=int)
    p.add_argument('--scope', choices=('all','connected'))
    p.add_argument('--max-reference-graphs', type=int, default=1024)
    p.add_argument('--max-generated-graphs', type=int, default=1024)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--sigma', type=float, default=1.0)
    p.add_argument('--json-out', required=True)
    args=p.parse_args(); cfg=load_yaml(args.config)
    spec=InducedGraphletSpec.from_config(cfg.get('structure_summary_prediction')) or InducedGraphletSpec()
    spec=InducedGraphletSpec(args.k if args.k is not None else spec.k, args.scope or spec.scope)
    if not np.isfinite(args.sigma) or args.sigma<=0: p.error('sigma must be positive and finite')
    if args.max_reference_graphs<=0 or args.max_generated_graphs<=0: p.error('sample caps must be positive')
    root=Path(cfg['dataset'].get('root','outputs/datasets'))/cfg['dataset']['name']
    refpath=root/(args.reference_split+'.pkl'); genpath=Path(args.generated_graphs)
    references=list(load_pickle(refpath)); generated=list(load_pickle(genpath))
    attributed_basis=None
    checkpoint_info=None
    pipeline_stage = cfg.get('pipeline', {}).get('stage', '')
    if pipeline_stage in ('attributed', 'attributed_topology', 'molecular') and wants_attributed_histogram(cfg):
        if not args.checkpoint:
            p.error('Attributed graphlet evaluation requires --checkpoint; never refit a vocabulary during evaluation.')
        from grapher.rewiring_mlp.attributed.joint_typed_edge_model import load_checkpoint
        model, ckpt = load_checkpoint(args.checkpoint, 'cpu')
        validate_model_graphlets(model, cfg)
        attributed_basis = model.induced_graphlet_basis
        if attributed_basis is None:
            p.error('The selected checkpoint has no attributed graphlet vocabulary.')
        info = attributed_graphlet_metadata(attributed_basis)
        if spec.k != info['k'] or spec.scope != info['scope']:
            p.error('k/scope cannot differ from the checkpoint vocabulary.')
        expected = ckpt.get('dataset_provenance', {}).get('split_sha256', {})
        if not expected or expected.get(args.reference_split) != file_sha256(refpath):
            raise ValueError('Reference dataset fingerprint differs from checkpoint.')
        if expected.get('train') != file_sha256(root/'train.pkl'):
            raise ValueError('Training dataset fingerprint differs from the vocabulary source.')
        checkpoint_info = {'path':str(args.checkpoint), 'sha256':file_sha256(args.checkpoint),
                           'epoch':ckpt.get('epoch'), 'vocabulary_source':'checkpoint_training_only'}
    rng=np.random.default_rng(args.seed)
    # Sampling caps and RNG recorded explicitly; no test reference during tuning.
    r_idx=np.sort(rng.choice(len(references), min(len(references),args.max_reference_graphs), replace=False))
    g_idx=np.sort(rng.choice(len(generated), min(len(generated),args.max_generated_graphs), replace=False))
    h_ref, omitted_ref=features([references[i] for i in r_idx],spec,attributed_basis=attributed_basis)
    h_gen, omitted_gen=features([generated[i] for i in g_idx],spec,attributed_basis=attributed_basis)
    mean_tv=float(.5*np.abs(h_ref.mean(0)-h_gen.mean(0)).sum())
    mmd=kernel_mean(h_ref,h_ref,args.sigma)+kernel_mean(h_gen,h_gen,args.sigma)-2*kernel_mean(h_ref,h_gen,args.sigma)
    report={'format':'induced_graphlet_histogram_comparison_v2','reference_split':args.reference_split,
        'catalogue':(attributed_graphlet_metadata(attributed_basis) if attributed_basis is not None else spec.metadata()),'reference_sha256':file_sha256(refpath),'generated_sha256':file_sha256(genpath),
        'checkpoint':checkpoint_info,
        'reference_indices':r_idx.tolist(),'generated_indices':g_idx.tolist(),'seed':args.seed,
        'reference_valid_graphs':len(h_ref),'generated_valid_graphs':len(h_gen),
        'reference_small_graphs_excluded':omitted_ref,'generated_small_graphs_excluded':omitted_gen,
        'mean_histogram_tv':mean_tv,'laplace_tv_biased_mmd2':float(mmd),'sigma':args.sigma,
        'metric_scope':('additional attributed graphlet diagnostic (node/edge labels included)' if attributed_basis is not None else 'additional topology diagnostic, not GraphRNN clustering/orbit MMD, NSPDK or FCD'),
        'reference_mean_histogram':h_ref.mean(0).tolist(),'generated_mean_histogram':h_gen.mean(0).tolist()}
    if attributed_basis is not None:
        k = attributed_basis.sizes[0]
        j = attributed_basis.keys_by_k[k].index(attributed_basis.overflow_key)
        report['reference_mean_overflow_mass'] = float(h_ref[:,j].mean())
        report['generated_mean_overflow_mass'] = float(h_gen[:,j].mean())
    atomic_json(report,Path(args.json_out))
    width = attributed_basis.width if attributed_basis is not None else spec.width
    mode = 'attributed' if attributed_basis is not None else 'topology'
    print(f'Induced graphlets ({mode}) k={spec.k} scope={spec.scope} bins={width}: reference={len(h_ref)} generated={len(h_gen)}')
    print(f'TV between mean graphlet histograms: {mean_tv:.6f}')
    print(f'Additional Laplace-TV biased MMD^2 (sigma={args.sigma}): {mmd:.6f}')
    print(f'Excluded graphs with n<k: reference={omitted_ref} generated={omitted_gen}')

if __name__=='__main__': main()

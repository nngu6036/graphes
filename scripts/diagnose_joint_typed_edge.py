#!/usr/bin/env python
"""Paired endpoint/bridge/soft-rollout diagnostics. Does not generate fake clean labels."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from grapher.utils.io import load_yaml,apply_config_overrides
from grapher.rewiring_mlp.attributed.joint_typed_edge_model import load_checkpoint,noisy_batch,structural_loss
from grapher.rewiring_mlp.attributed.joint_typed_edge_data import load_splits,EndpointStore,collate
from grapher.rewiring_mlp.attributed.joint_typed_edge_generation import sample_soft_endpoint,refine_typed_graph
from grapher.rewiring_mlp.attributed.soft_edge_bridge import labels_to_logits,center_edges,edge_probabilities
from grapher.rewiring_mlp.attributed.adjacency_diffusion import ADJACENCY_MODE,validate_model_config
from grapher.rewiring_mlp.attributed.joint_typed_edge_training import validate_config
from grapher.rewiring_mlp.generic.joint_checkpointing import atomic_json
from grapher.rewiring_mlp.attributed.induced_graphlets import (
    validate_model_graphlets, extract_histogram as attributed_histogram,
    histogram_distance as attributed_distance,
)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--split',choices=['train','val','test'],default='val')
    p.add_argument('--mode',choices=['source','bridge','rollout'],default='source')
    p.add_argument('--max-graphs',type=int,default=64);p.add_argument('--samples-per-graph',type=int,default=1)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--device',default='auto')
    p.add_argument('--rewire',action='store_true',help='Also measure paired realization; only with --mode rollout.')
    p.add_argument('--json-out',required=True);p.add_argument('--set',action='append',default=[])
    args=p.parse_args();cfg=load_yaml(args.config);apply_config_overrides(cfg,args.set)
    if args.rewire and args.mode!='rollout': p.error('--rewire requires --mode rollout')
    if args.max_graphs<1 or args.samples_per_graph<1: p.error('Counts must be positive.')
    validate_config(cfg)
    model,ckpt=load_checkpoint(args.checkpoint,args.device);validate_model_config(model,cfg)
    splits,prov=load_splits(cfg)
    if ckpt.get('dataset_provenance',{}).get('fingerprint')!=prov['fingerprint']:
        raise ValueError('Dataset/checkpoint provenance mismatch.')
    from grapher.rewiring_mlp.generic.induced_graphlets import InducedGraphletSpec, extract_histogram, histogram_distance
    validate_model_graphlets(model, cfg)
    device=next(model.parameters()).device;graphs=splits[args.split][:args.max_graphs]
    # Match the training validation-source convention; not a held-out source prior.
    store=EndpointStore(graphs,model.vectorizer,model.atom_types,cfg,seed=int(ckpt['config']['seed'])+(args.split!='train'),graphlet_basis=model.induced_graphlet_basis)
    generator=torch.Generator(device=device).manual_seed(args.seed)
    rows=[]
    try:
        with torch.no_grad():
            for i in range(len(store)):
                item=store[i];batch=collate([item],model.vectorizer,model.atom_types,device=device)
                baselogits=labels_to_logits(batch['source_labels'],model.categories,batch['mask'],model.smoothing)
                for sample in range(args.samples_per_graph):
                    if args.mode=='rollout':
                        targets,stats=sample_soft_endpoint(model,item['source'],cfg,seed=args.seed+i*1009+sample)
                        pred={'clean_edge_logits':torch.tensor(np.log(np.maximum(targets['edge_probabilities'],1e-30)),device=device)[None].float()}
                        pred['clean_edge_logits']=center_edges(pred['clean_edge_logits'],batch['mask'])
                        pred['clean_edge_probabilities']=edge_probabilities(pred['clean_edge_logits'],batch['mask'])
                        for src,dst in [('histogram','clean_clustering_histogram'),('orbit','clean_orbit_summary'),('spectra','clean_spectra'),('adjacency_spectra','clean_adjacency_spectra'),('induced_histogram','clean_induced_graphlet_histogram')]:
                            if src in targets: pred[dst]=torch.tensor(targets[src],device=device)[None].float()
                        inp=batch
                    else:
                        inp=noisy_batch(batch,model,cfg,generator=generator,endpoint_only=args.mode=='source');pred=model(inp)
                    _,metrics=structural_loss(pred,inp,model,cfg['attributed_predictor']['loss_weights'])
                    baseline={**pred,'clean_edge_logits':baselogits,'clean_edge_probabilities':edge_probabilities(baselogits,batch['mask'])}
                    if model.spectral_mode == ADJACENCY_MODE and model.adjacency_output_spectra:
                        baseline['clean_adjacency_spectra']=model.adjacency_features(baseline['clean_edge_probabilities'],batch['mask'])['spectra']
                    _,base=structural_loss(baseline,inp,model,cfg['attributed_predictor']['loss_weights'])
                    row={'graph_index':i,'sample':sample,'source_edge_ce':base['edge_ce_loss'],**metrics}
                    if model.spectral_mode == ADJACENCY_MODE and model.adjacency_output_spectra:
                        row['source_adjacency_spectral_nrmse']=base['adjacency_spectral_nrmse']
                        actual=model.adjacency_features(pred['clean_edge_probabilities'],batch['mask'])['spectra']
                        reported=pred.get('clean_adjacency_spectra',actual)
                        row['adjacency_prediction_spectrum_consistency_max_abs']=float((actual-reported).abs().max().cpu())
                    if model.induced_graphlet_basis is not None:
                        basis = model.induced_graphlet_basis
                        row['source_induced_graphlet_histogram_tv'] = (attributed_distance(
                            attributed_histogram(item['source'], basis), item['induced_histogram'], basis)
                            if len(item['source']) >= int(basis.sizes[0]) else 0.0)
                    elif model.induced_graphlet_spec is not None:
                        row['source_induced_graphlet_histogram_tv'] = (histogram_distance(
                            extract_histogram(item['source'], model.induced_graphlet_spec), item['induced_histogram'],
                            model.induced_graphlet_spec) if len(item['source']) >= model.induced_graphlet_spec.k else 0.0)
                    if args.rewire:
                        final,r=refine_typed_graph(item['source'],targets,model,cfg,seed=args.seed+i*1009+sample)
                        from grapher.rewiring_mlp.attributed.joint_typed_edge_generation import edge_energy
                        # Hard adjacency accuracy is label-aligned and NOT an isomorphism metric.
                        target_labels=batch['target_labels'][0].cpu().numpy();n=len(final)
                        labels=np.zeros((n,n),dtype=int)
                        for u,v,d in final.edges(data=True): labels[u,v]=labels[v,u]=model.edge_types.index(d['bond_type'])+1
                        idx=np.triu_indices(n,1)
                        row['paired_final_edge_accuracy']=float(np.mean(labels[idx]==target_labels[idx])) if len(idx[0]) else 1.
                        row['accepted_swaps']=r['accepted_steps'];row['typed_degree_preserved']=r['typed_degree_preserved']
                    rows.append(row)
    finally: store.close()
    means={k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k not in ('graph_index','sample')}
    maxima={k:float(max(r[k] for r in rows)) for k in rows[0] if k.endswith('_max_abs')}
    report={'split':args.split,'mode':args.mode,'graphs':len(graphs),'examples':len(rows),
            'induced_graphlet_metadata':model.induced_graphlet_metadata(),
            'diffusion':model.diffusion_metadata(),
            'scope':'paired_known_targets_not_unconditional_generation','means':means,'maxima':maxima,'rows':rows,'dataset_provenance':prov}
    atomic_json(report,Path(args.json_out))
    print(f'Joint typed edge diagnostic: mode={args.mode} split={args.split} graphs={len(graphs)} examples={len(rows)}')
    print(f'  diffusion: {model.diffusion_metadata()}')
    if model.induced_graphlet_metadata() is not None:
        m = model.induced_graphlet_metadata()
        print(f"  graphlet attributed={m['attributed']} k={m['k']} bins={m['width']}")
    for k,v in means.items(): print(f'  {k}: {v:.6f}')

if __name__=='__main__': main()

"""Graph-balanced joint training, provenance checks and matched typed-VAE exports."""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch

from grapher.models.dhvae_hh.typed_degree_vae import (
    TypedSignatureVectorizer,TypedSignatureHistogramVAE,build_typed_signature_vae,
    typed_signature_vae_loss,save_typed_signature_checkpoint,TYPED_CHECKPOINT_FORMAT,
)
from grapher.rewiring_mlp.attributed.joint_typed_edge_data import (
    ENDPOINT_VALENCE_POLICY, EndpointStore, collate, load_splits, validate_graph,
)
from grapher.rewiring_mlp.attributed.joint_typed_edge_model import (
    JointTypedEdgePredictor,noisy_batch,structural_loss,save_checkpoint,
)
from grapher.rewiring_mlp.generic.joint_checkpointing import atomic_json,file_sha256,state_dict_sha256
from grapher.utils.device import resolve_torch_device
from grapher.rewiring_mlp.generic.induced_graphlets import InducedGraphletSpec
from grapher.rewiring_mlp.attributed.data import GraphletBasis
from grapher.rewiring_mlp.attributed.induced_graphlets import (fit_training_basis, wants_attributed_histogram)


def validate_config(config):
    if config.get('pipeline', {}).get('stage', 'attributed') not in ('attributed', 'attributed_topology', 'molecular'):
        raise ValueError('Joint typed edge training/generation requires an attributed pipeline.')
    if config.get('attributed_predictor', {}).get('type') != 'joint_typed_soft_edge':
        raise ValueError('attributed_predictor.type must be joint_typed_soft_edge.')
    if not config.get('joint_typed_degree',{}).get('enabled',False):
        raise ValueError('joint_typed_degree.enabled must be true.')
    if config.get('source_enrichment',{}).get('enabled',False):
        raise ValueError('This joint edge-bridge experiment does not support source enrichment.')
    diff=config.get('edge_diffusion',{})
    if diff.get('bridge','centered_logit_brownian')!='centered_logit_brownian':
        raise ValueError('Only centered_logit_brownian edge bridge is implemented.')
    if not diff.get('enabled',True): raise ValueError('edge_diffusion.enabled must be true.')
    if not 0<float(diff.get('smoothing',0.01))<1: raise ValueError('Bad edge smoothing.')
    if not 0<=float(diff.get('endpoint_fraction',0.1))<=1: raise ValueError('Bad endpoint_fraction.')
    for key in ('sigma','spectral_sigma'):
        val=float(diff.get(key,1.0 if key=='sigma' else 0.15))
        if not math.isfinite(val) or val<0: raise ValueError(f'Bad edge_diffusion.{key}.')
    if int(diff.get('sampling_steps',32))<2: raise ValueError('Use at least two soft bridge sampling steps.')
    if not isinstance(diff.get('spectral_enabled',True), bool): raise ValueError('spectral_enabled must be boolean.')
    if int(diff.get('views_per_graph',2))<1: raise ValueError('views_per_graph must be positive.')
    if diff.get('generation_strategy','bridge_then_rewire')!='bridge_then_rewire':
        raise ValueError('Only bridge_then_rewire is implemented; soft state is NOT reset from a rewired graph.')
    weights=config.get('attributed_predictor',{}).get('loss_weights',{})
    for key,val in weights.items():
        if key not in {'edge_ce','edge_logit','typed_consistency','spectrum','clustering_histogram','orbit_summary','induced_graphlet_histogram','induced_graphlet_histogram_ce'}:
            raise ValueError(f'Unknown new-family loss {key}.')
        if not math.isfinite(float(val)) or float(val)<0: raise ValueError(f'Invalid weight {key}.')
    if float(weights.get('edge_logit',0))<=0 or float(weights.get('edge_ce',0))<=0:
        raise ValueError('Positive edge_logit and edge_ce losses are required for this continuous bridge.')
    ss=config.get('structure_summary_prediction',{})
    induced_spec = InducedGraphletSpec.from_config(ss)
    if induced_spec is not None and wants_attributed_histogram(config) and induced_spec.scope != 'all':
        raise ValueError('Attributed induced graphlet histograms currently support induced_graphlet_scope=all only.')
    if ss.get('orbit_summary', True) and int(ss.get('orbit_width',15))!=15:
        raise ValueError('Only the topology ORCA-15 auxiliary orbit summary is supported.')
    if not diff.get('spectral_enabled',True) and float(weights.get('spectrum',0))>0:
        raise ValueError('Spectrum loss requires spectral_enabled=true.')
    if not ss.get('clustering_histogram',True) and float(weights.get('clustering_histogram',0))>0:
        raise ValueError('Histogram loss requires the histogram head.')
    if not ss.get('orbit_summary',True) and float(weights.get('orbit_summary',0))>0:
        raise ValueError('Orbit loss requires the orbit head.')
    if induced_spec is None and any(float(weights.get(key, 0)) > 0 for key in ('induced_graphlet_histogram', 'induced_graphlet_histogram_ce')):
        raise ValueError('Induced graphlet loss requires its prediction head.')
    if ss.get('cycle_graphlet_histogram',False) or config.get('graphlet_diffusion',{}).get('enabled',False):
        raise ValueError('Cycle/attributed graphlet diffusion is not part of this focused edge experiment.')
    jc=config['joint_typed_degree']
    for k in ('loss_weight','kl_loss_weight','learning_rate'):
        v=float(jc.get(k, {'loss_weight':0.01,'kl_loss_weight':0.005,'learning_rate':2e-5}[k]))
        if not math.isfinite(v) or v<0: raise ValueError(f'Invalid joint_typed_degree.{k}.')
    gc=config.get('generation',{})
    if 'degree_source' in gc:
        raise ValueError('This attributed family uses generation.invariant_source, not generation.degree_source.')
    for k in ('max_attempts_per_graph','max_invariant_resample','checkpoint_every'):
        if int(gc.get(k,16))<1: raise ValueError(f'generation.{k} must be positive.')
    if not config.get('constructor',{}).get('ensure_connected',True):
        raise ValueError('Connected typed source construction is required.')
    if config.get('degree_generator',{}).get('checkpoint_path'):
        raise ValueError('Use joint_typed_degree.initialize_degree_checkpoint for training; generation uses the embedded model.')
    if config.get('constructor',{}).get('candidate_ranking','uniform')!='uniform':
        raise ValueError('The joint edge source currently supports uniform typed constructor ranking only.')


def build_model(config,train_graphs,device):
    cat=config['categorical_state']; joint=config['joint_typed_degree']; pc=config['attributed_predictor']
    edge_types=tuple(cat['edge_categories']); atoms=tuple(cat['node_categories'])
    if cat.get('node_attribute','atomic_num')!='atomic_num' or cat.get('edge_attribute','bond_type')!='bond_type':
        raise ValueError('This molecular family uses atomic_num/bond_type (no-bond is separate category 0).')
    path=joint.get('initialize_degree_checkpoint')
    if path:
        path=Path(path)
        if not path.is_file(): raise FileNotFoundError(f'Typed warm-start missing: {path}. Train typed-DH-VAE first, or explicitly set initialization=null and freeze_epochs=0.')
        ckpt=torch.load(path,map_location='cpu',weights_only=True)
        if ckpt.get('format')!=TYPED_CHECKPOINT_FORMAT: raise ValueError('Initializer must be a typed-signature VAE, not ordinary DH-VAE.')
        vectorizer=TypedSignatureVectorizer.from_dict(ckpt['vectorizer'])
        prior=TypedSignatureHistogramVAE(**ckpt['model_config']); prior.load_state_dict(ckpt['model_state_dict'])
        if vectorizer.vocabulary.edge_types!=edge_types: raise ValueError('Typed edge category ordering differs from initializer.')
        if (vectorizer.vocabulary.node_attribute,vectorizer.vocabulary.edge_attribute)!=('atomic_num','bond_type'):
            raise ValueError('Typed initializer attribute convention differs.')
        # Empirical size sampling must reflect CURRENT training data, not an old
        # vectorizer's cached split. Existing signature vocabulary is frozen.
        for g in train_graphs: validate_graph(g,vectorizer,atoms)
        vectorizer.empirical_node_counts=[len(g) for g in train_graphs]
        vectorizer.empirical_invariants=[]  # fallback is disabled; avoid duplicating whole datasets in each checkpoint
    else:
        if int(joint.get('freeze_epochs',0))>0:
            raise ValueError('A randomly initialized typed prior must not be frozen; set freeze_epochs=0.')
        sig=config.get('typed_signature',{})
        vectorizer=TypedSignatureVectorizer.fit(train_graphs,edge_types=edge_types,
          require_connected=True,max_ordinary_degree=sig.get('max_ordinary_degree'),
          max_weighted_valence=sig.get('max_weighted_valence'))
        vectorizer.empirical_invariants=[]
        dims={k:joint[k] for k in ('latent_dim','hidden_dim','size_condition_dim','prior_type','prior_components','num_layers','dropout') if k in joint}
        prior=build_typed_signature_vae(vectorizer,**dict({'latent_dim':64,'hidden_dim':128},**dims))
    ss=config.get('structure_summary_prediction',{})
    induced_spec = InducedGraphletSpec.from_config(ss)
    if induced_spec is not None and wants_attributed_histogram(config) and induced_spec.scope != 'all':
        raise ValueError('Attributed induced graphlet histograms currently support induced_graphlet_scope=all only.')
    induced_basis = fit_training_basis(config, train_graphs)
    model=JointTypedEdgePredictor(typed_model_config=prior.model_config(),vectorizer=vectorizer.to_dict(),
        atom_types=list(atoms), hidden_dim=int(pc.get('hidden_dim',128)),num_layers=int(pc.get('num_layers',4)),
        spectral_layers=int(pc.get('spectral_layers',2)),spectral_heads=int(pc.get('spectral_heads',4)),
        spectral_enabled=bool(config['edge_diffusion'].get('spectral_enabled',True)),
        histogram_bins=int(ss.get('clustering_bins',100)) if ss.get('clustering_histogram',True) else 0,
        orbit_enabled=bool(ss.get('orbit_summary',True)),smoothing=float(config['edge_diffusion'].get('smoothing',0.01)),
        induced_graphlet_basis=induced_basis.to_dict() if induced_basis is not None else None,
        induced_graphlet_k=induced_spec.k if induced_spec is not None and induced_basis is None else None,
        induced_graphlet_scope=induced_spec.scope if induced_spec else 'all')
    model.degree_model.load_state_dict(prior.state_dict()); return model.to(device)


class TypedCheckpointManager:
    """Atomic per-file snapshots plus hash-marked registry, not exact-resume state."""
    def __init__(self,output,config,provenance,model):
        self.output=Path(output); self.config=config; self.provenance=provenance; self.records={}
        self.criteria={'best_joint':'val_joint_loss','best_edges':'val_edge_ce_loss'}
        if model.histogram_bins: self.criteria['best_histogram']='val_clustering_histogram_w1'
        if model.orbit_enabled: self.criteria['best_orbit']='val_orbit_summary_log_rmse'
        if model.induced_graphlet_metadata() is not None: self.criteria['best_graphlet']='val_induced_graphlet_histogram_tv'

    def save(self,kind,model,epoch,metrics,eligible):
        dest=self.output/'checkpoints'/kind; dest.mkdir(parents=True,exist_ok=True)
        degree_hash=state_dict_sha256(model.degree_model.state_dict())
        marker={'kind':kind,'epoch':epoch,'criterion':self.criteria.get(kind),
                'eligible':eligible,'embedded_typed_state_sha256':degree_hash}
        save_checkpoint(dest/'checkpoint.pt',model,self.config,metrics,epoch=epoch,
                        selection=marker,dataset_provenance=self.provenance)
        temporary=dest/'degree_checkpoint.pt.tmp'
        save_typed_signature_checkpoint(temporary,model.degree_model,model.vectorizer,
             config=self.config,metrics={'epoch':epoch,**metrics})
        temporary.replace(dest/'degree_checkpoint.pt')
        marker.update({'checkpoint_sha256':file_sha256(dest/'checkpoint.pt'),
                       'degree_checkpoint_sha256':file_sha256(dest/'degree_checkpoint.pt'),'metrics':metrics})
        atomic_json(marker,dest/'selection.json')
        self.records[kind]={**marker,'path':str(dest.relative_to(self.output))}
        if kind=='best_joint':
            for name in ('checkpoint.pt','degree_checkpoint.pt'):
                tmp=self.output/(name+'.tmp'); shutil.copyfile(dest/name,tmp); tmp.replace(self.output/name)

    def update(self,model,epoch,metrics,eligible):
        for kind,key in self.criteria.items():
            value=float(metrics[key])
            if not math.isfinite(value): raise FloatingPointError(f'Nonfinite checkpoint criterion {key}.')
            if eligible and (kind not in self.records or value<self.records[kind]['metrics'][key]):
                self.save(kind,model,epoch,metrics,eligible)
        self.save('last',model,epoch,metrics,eligible)
        atomic_json({'format':'joint_typed_edge_registry_v1','scope':'inference_snapshot_no_optimizer_or_rng',
                     'last_completed_epoch':epoch,'selections':self.records},self.output/'checkpoint_registry.json')


def run_epoch(model,store,config,*,batch_size,device,optimizer=None,seed=0,beta=None):
    training=optimizer is not None; model.train(training)
    rng=np.random.default_rng(seed); indices=np.arange(len(store))
    if training: rng.shuffle(indices)
    # Fixed validation random stream, independent of both training and generation.
    generator=torch.Generator(device=device).manual_seed(seed)
    rows=[]; total_graphs=0; last_progress=time.perf_counter(); views=int(config['edge_diffusion'].get('views_per_graph',2))
    j=config['joint_typed_degree']; weights=config['attributed_predictor']['loss_weights']
    for start in range(0,len(indices),batch_size):
        items=[store[int(i)] for i in indices[start:start+batch_size]]
        batch=collate(items,model.vectorizer,model.atom_types,device=device,
                      rng=rng if training and config.get('training_sources',{}).get('shared_relabel_augmentation',True) else None)
        count=len(items)
        if training: optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            # One VAE objective per graph minibatch; not repeated per bridge view.
            degree_out,mu,lv=model.degree_model(batch['typed_features'],batch['n'])
            targets={k.removeprefix('degree_'):v for k,v in batch.items() if k.startswith('degree_')}
            degree_loss,degree_metrics=typed_signature_vae_loss(degree_out,targets,mu,lv,
                  beta=float(j.get('kl_loss_weight',0.005) if beta is None else beta),weights=j.get('loss_weights'))
            # Backward each view separately to keep memory O(batch), not O(batch*views).
            accum={}
            for _ in range(views):
                noisy=noisy_batch(batch,model,config,generator=generator)
                pred=model(noisy); loss,metrics=structural_loss(pred,noisy,model,weights)
                if training: (loss/views).backward()
                for k,v in metrics.items(): accum[k]=accum.get(k,0)+v/views
            degree_weight=float(j.get('loss_weight',0.01))
            if training and degree_loss.requires_grad: (degree_weight*degree_loss).backward()
            if training:
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),float(config['attributed_predictor'].get('grad_clip',1.0)))
                if not torch.isfinite(norm): raise FloatingPointError('Nonfinite joint gradient norm.')
                optimizer.step()
        accum['degree_loss']=float(degree_loss.detach().cpu()); accum['joint_loss']=accum['structure_loss']+degree_weight*accum['degree_loss']
        accum.update({'degree_'+k:v for k,v in degree_metrics.items() if k!='loss'})
        if not all(math.isfinite(v) for v in accum.values()): raise FloatingPointError('Nonfinite training metrics.')
        rows.append((count,accum)); total_graphs+=count
        pc=config['attributed_predictor']; interval=int(pc.get('batch_progress_interval',50))
        seconds=float(pc.get('progress_interval_seconds',60))
        if (interval>0 and len(rows)%interval==0) or (seconds>0 and time.perf_counter()-last_progress>=seconds):
            print(f"[JointTypedEdge] phase={'train' if training else 'val'} graphs={total_graphs}/{len(store)} "
                  f"joint_loss={accum['joint_loss']:.6f}",flush=True)
            last_progress=time.perf_counter()
    return {k:(max(r[k] for _,r in rows) if k.endswith('_max_abs') else
               sum(n*r[k] for n,r in rows)/total_graphs) for k in rows[0][1]}


def train_joint_typed_edge(config,args):
    validate_config(config); config=deepcopy(config)
    seed=int(args.seed if args.seed is not None else config.get('seed',42)); config['seed']=seed
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    pc=config['attributed_predictor']; j=config['joint_typed_degree']
    for k in ('batch_progress_interval','progress_interval_seconds'):
        if getattr(args,k,None) is not None: pc[k]=getattr(args,k)
    epochs=int(args.epochs if args.epochs is not None else pc.get('epochs',100))
    batch_size=int(args.batch_size if args.batch_size is not None else pc.get('batch_size',16))
    freeze=int(j.get('freeze_epochs',0)); trainable=bool(j.get('trainable',True))
    if epochs<1 or batch_size<1 or (trainable and freeze>=epochs):
        raise ValueError('Need positive epochs/batch size and at least one post-freeze epoch for joint training.')
    output=Path(args.output_dir or pc['checkpoint_path']).resolve()
    if args.output_dir is None: output=output.parent
    if output.exists() and any(output.iterdir()): raise FileExistsError(f'Use a fresh output directory: {output}')
    output.mkdir(parents=True,exist_ok=True)
    splits,provenance=load_splits(config)
    train_limit=args.max_train_graphs if args.max_train_graphs is not None else config['dataset'].get('max_train_graphs')
    val_limit=args.max_val_graphs if args.max_val_graphs is not None else config['dataset'].get('max_val_graphs')
    train=splits['train'][:int(train_limit)] if train_limit else splits['train']
    val=splits['val'][:int(val_limit)] if val_limit else splits['val']
    device=resolve_torch_device(args.device or pc.get('device','auto'))
    model=build_model(config,train,device)
    # Fail with split/index before training if held-out typed support is unknown.
    for split,graphs in [('train',train),('val',val)]:
        for i,g in enumerate(graphs):
            try: validate_graph(g,model.vectorizer,model.atom_types)
            except ValueError as e: raise ValueError(f'{split}[{i}]: {e}') from e
    config['attributed_predictor'].update(epochs=epochs,batch_size=batch_size)
    config['dataset'].update(max_train_graphs=train_limit,max_val_graphs=val_limit)
    dataset_info={'train_graphs':len(train),'val_graphs':len(val),'provenance':provenance,
                  'typed_initializer_sha256':file_sha256(j['initialize_degree_checkpoint']) if j.get('initialize_degree_checkpoint') else None,
                  'source_alignment':'indexed_typed_signatures_shared_node_permutation',
                  'endpoint_valence_policy':ENDPOINT_VALENCE_POLICY,
                  'validation_vocabulary_policy':'strict_typed_support; graphlet_vocabulary_fitted_train_only_with_overflow',
                  'induced_graphlet_metadata':model.induced_graphlet_metadata()}
    if model.induced_graphlet_basis is not None:
        atomic_json({'basis':model.induced_graphlet_basis.to_dict(),
                     'metadata':model.induced_graphlet_metadata(),
                     'training_graphs':len(train), 'dataset_provenance':provenance},
                    output/'attributed_graphlet_basis.json')
    atomic_json({'config':config,**dataset_info},output/'run_config.json')
    print('[JointTypedEdge] endpoints preserve prepared target bond types; '
          'degree and chemical valence caps apply to generation.', flush=True)
    cache=config.get('training_sources',{}).get('endpoint_cache_path')
    training=EndpointStore(train,model.vectorizer,model.atom_types,config,seed=seed,cache_path=cache,graphlet_basis=model.induced_graphlet_basis)
    validation=EndpointStore(val,model.vectorizer,model.atom_types,config,seed=seed+1,cache_path=cache,graphlet_basis=model.induced_graphlet_basis)
    degree_params=list(model.degree_model.parameters()); ids={id(p) for p in degree_params}
    optimizer=torch.optim.AdamW([
        {'params':[p for p in model.parameters() if id(p) not in ids],'lr':float(pc.get('learning_rate',1e-4))},
        {'params':degree_params,'lr':float(j.get('learning_rate',2e-5))}],weight_decay=float(pc.get('weight_decay',1e-5)))
    manager=TypedCheckpointManager(output,config,provenance,model); history=[]; start=time.perf_counter()
    try:
        for epoch in range(1,epochs+1):
            unfrozen=trainable and epoch>freeze; model.set_degree_trainable(unfrozen)
            metrics=run_epoch(model,training,config,batch_size=batch_size,device=device,
                              optimizer=optimizer,seed=seed+epoch*101)
            cuda_devices=[device.index or 0] if device.type=='cuda' else []
            # Fork/reset also protects stochastic VAE validation from changing optimizer RNG.
            with torch.random.fork_rng(devices=cuda_devices):
                torch.manual_seed(seed+991)
                val_metrics=run_epoch(model,validation,config,batch_size=batch_size,device=device,seed=seed+991)
            row={'epoch':epoch,'degree_trainable':unfrozen,
                 **{'train_'+k:v for k,v in metrics.items()},**{'val_'+k:v for k,v in val_metrics.items()}}
            history.append(row); atomic_json(history,output/'history.json')
            manager.update(model,epoch,row,eligible=(unfrozen or not trainable))
            print(f"[JointTypedEdge] epoch={epoch}/{epochs} degree_trainable={unfrozen} "
                  f"joint={row['val_joint_loss']:.6f} edge_CE={row['val_edge_ce_loss']:.6f} "
                  f"soft_typed_RMSE={row['val_soft_typed_degree_rmse']:.6f}",flush=True)
    finally: training.close(); validation.close()
    atomic_json({'config':config,**dataset_info,'runtime_seconds':time.perf_counter()-start,
                 'checkpoint_registry':manager.records,'best_epoch':manager.records['best_joint']['epoch'],
                 'generation_strategy':'soft_bridge_then_frozen_endpoint_typed_rewiring',
                 'typed_degree_export':'degree_checkpoint.pt'},output/'report.json')
    print(f'Saved joint typed model: {output}/checkpoint.pt',flush=True)

"""Managed joint training and generation for the attributed GDSM extension."""
from __future__ import annotations
import copy
import json
import pickle
import shutil
import tempfile
import time
from collections import defaultdict, Counter
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
import numpy as np
import torch
import yaml

from grapher.models.artifacts import ArtifactLayout
from grapher.models.base import GenerationArtifacts
from grapher.models.errors import ArtifactCollisionError
from grapher.models.gdsm_simple.wrapper import _graphs, _jsonable, _resolve_device, _seed_everything, _sha256, _write_json
from grapher.rewiring_mlp.attributed.data import GraphCategoryVocabulary
from .config import resolve
from .data import TypedGraphlets3, make_record, collate, permute_aligned, decode_graph
from .model import SpectralCategoricalDenoiser, losses, predictions_numpy
from .noise import MarginalNoise, cosine_alpha_bar, draw_graph, spectral_q_sample, spectral_reverse
from .spectral import degree_anchor, eigenpairs
from .refiner import refine
from .multiscale import (TypedGraphletsMulti, fit_basis, pack_training_counts,
                         remap_training_counts, pack_basis_counts)

FORMAT='gdsm_spectral_categorical_checkpoint_v1'


def _basis_for_size(bank, n, rng):
    choices=bank.get(n,[])
    if choices:
        index=int(rng.integers(len(choices)))
        return choices[index],index,False
    # Validation can include unseen sizes. No target eigenvectors are used.
    q,_=np.linalg.qr(rng.normal(size=(n,n)))
    return q.astype(np.float32),-1,True


def prepare_data(train_graphs,val_graphs,max_nodes,cfg,seed):
    """Only training graphs define category marginals, graphlet vocabulary and basis bank."""
    vocab=GraphCategoryVocabulary.from_graphs(train_graphs,cfg['categories'])
    bins=int(cfg['graphlets']['clustering_bins'])
    rng=np.random.default_rng(seed+271)
    reservoir=defaultdict(list); seen_sizes=defaultdict(int); all_keys=set()
    train=[]; val=[]
    multi=cfg['graphlets'].get('sizes') is not None
    all_counts={k:Counter() for k in cfg['graphlets']['sizes']} if multi else None
    codebooks={k:{} for k in all_counts} if multi else None
    node_counts=np.zeros(vocab.num_node_categories,np.int64)
    edge_counts=np.zeros(vocab.num_edge_categories,np.int64)
    limit=int(cfg['initialization']['basis_max_per_size'])
    for i,g in enumerate(train_graphs):
        row,u=make_record(g,vocab,max_nodes,bins,cfg['graphlets']); train.append(row)
        n=len(row['x']); seen_sizes[n]+=1
        if len(reservoir[n])<limit: reservoir[n].append(u)
        else:
            j=int(rng.integers(seen_sizes[n]))
            if j<limit: reservoir[n][j]=u
        if multi:
            for k in all_counts: all_counts[k].update(row['counts'][k])
            row['counts']=pack_training_counts(row['counts'],codebooks)
        else:
            all_keys.update(row['counts'])
        node_counts+=np.bincount(row['x'],minlength=len(node_counts))
        edge_counts+=np.bincount(row['e'][np.triu_indices(n,1)],minlength=len(edge_counts))
        if (i+1)%10000==0: print(f'[gdsm-categorical] prepared {i+1} training graphs',flush=True)
    if multi:
        basis,coverage=fit_basis(all_counts,cfg['graphlets'])
        lookup={k:np.array([basis.index[k].get(key,len(basis.keys_by_order[k])) for key in codebooks[k]],np.int32) for k in basis.orders}
        for row in train: row['counts']=remap_training_counts(row['counts'],lookup)
        del all_counts,codebooks,lookup
    else:
        basis=TypedGraphlets3(sorted(all_keys))
    for g in val_graphs:
        row,_=make_record(g,vocab,max_nodes,bins,cfg['graphlets'])
        if multi: row['counts']=pack_basis_counts(row['counts'],basis)
        val.append(row)
    anchor_cache={}; unseen_validation_sizes=set(); anchor_residuals=[]
    for rows in (train,val):
        for row in rows:
            n=len(row['x']); d=np.sort(row['degrees'])[::-1]
            if cfg['initialization']['mode']=='gaussian':
                row['anchor']=np.zeros(n,np.float32); continue
            u,index,fallback=_basis_for_size(reservoir,n,rng)
            if fallback: unseen_validation_sizes.add(n)
            key=(n,index,tuple(d))
            if fallback or key not in anchor_cache:
                anchor,diag=degree_anchor(u,d,ridge=cfg['initialization']['ridge'],diagonal_weight=cfg['initialization']['diagonal_weight'])
                if not fallback and len(anchor_cache)<50000: anchor_cache[key]=(anchor,diag)
            else: anchor,diag=anchor_cache[key]
            row['anchor']=anchor
            anchor_residuals.append(diag['row_sum_rmse'])
    pseudo=float(cfg['noise']['pseudocount'])
    xm=(node_counts+pseudo)/(node_counts.sum()+pseudo*len(node_counts))
    em=(edge_counts+pseudo)/(edge_counts.sum()+pseudo*len(edge_counts))
    metadata={
        'category_vocabulary':vocab.to_dict(),'graphlet_keys':[] if multi else [list(k) for k in basis.keys],
        'node_marginal':xm.tolist(),'edge_marginal':em.tolist(),
        'node_category_counts':node_counts.tolist(),'edge_category_counts':edge_counts.tolist(),
        'edge_counting':'unordered_pairs_including_no_edge_excluding_padding_and_diagonal',
        'graphlets':'connected_induced_typed_3_plus_overflow_and_connected_triple_mass',
        'unseen_validation_sizes_haar_anchor_fallback':sorted(unseen_validation_sizes),
        'mean_anchor_row_sum_rmse':float(np.mean(anchor_residuals)) if anchor_residuals else 0.,
    }
    if multi:
        metadata.update(basis.schema())
        metadata['training_graphlet_vocabulary_coverage']=coverage
    return train,val,vocab,basis,dict(reservoir),metadata


def _make_model_config(options,cfg,vocab,basis,max_nodes):
    mc=options['model']
    config=dict(node_classes=vocab.num_node_categories,edge_classes=vocab.num_edge_categories,
                graphlet_classes=basis.dimension,max_nodes=max_nodes,
                hidden_dim=int(mc['hidden_dim']),num_layers=int(mc['num_layers']),num_heads=int(mc['num_heads']),
                ff_dim=int(mc['ff_dim']),dropout=float(mc['dropout']),
                clustering_bins=int(cfg['graphlets']['clustering_bins']),spectral_conditioning=cfg['spectral_conditioning'])
    if getattr(basis,'multiscale',False):
        config.update(graphlet_block_sizes=list(basis.block_sizes),graphlet_orders=list(basis.orders),
                      graphlet_size_weights=basis.size_weights.tolist())
    return config


def _loss_batch(model,records,basis,cfg,schedule,node_noise,edge_noise,generator,device,*,permutations):
    batch=collate(records,basis,device=device)
    if permutations: batch=permute_aligned(batch,generator)
    mask=batch['mask']; total=len(schedule)-1
    t=torch.randint(1,total+1,(len(records),),device=device,generator=generator)
    xt,et=draw_graph(node_noise.forward_probs(batch['x'],t),edge_noise.forward_probs(batch['e'],t),mask,generator)
    noise=torch.randn(batch['z'].shape,device=device,generator=generator)*mask
    zt=spectral_q_sample(batch['z'],noise,batch['anchor'],schedule,t,mask)
    pred=model(xt,et,zt,t,batch['anchor'],mask,total)
    return losses(pred,batch,cfg['loss_weights'])


def train(wrapper,request,options):
    cfg=resolve(options)
    layout=request.run.layout; artifacts=wrapper._artifacts(request)
    fingerprint=request.dataset.fingerprint()
    if layout.training_manifest_path.is_file() and not request.overwrite:
        old=json.loads(layout.training_manifest_path.read_text())
        if old.get('dataset',{}).get('fingerprint')==fingerprint and old.get('options')==_jsonable(options) and artifacts.checkpoint_path.is_file():
            return artifacts
        raise ArtifactCollisionError('Existing categorical run differs; use a new run-id or --overwrite')
    ArtifactLayout.require_available(layout.train_dir,overwrite=request.overwrite)
    _seed_everything(request.run.train_seed)
    device=_resolve_device(options['runtime'])
    if device.type=='cuda' and device.index is None: device=torch.device('cuda',torch.cuda.current_device())
    train_graphs=_graphs(request.dataset.split_paths['train']); val_graphs=_graphs(request.dataset.split_paths['val'])
    cap=options['train'].get('max_train_graphs')
    if cap is not None:
        if int(cap)<1: raise ValueError('max_train_graphs must be null or positive')
        train_graphs=train_graphs[:int(cap)]
    max_nodes=int(options['model'].get('max_nodes') or max(len(g) for g in train_graphs))
    started=time.monotonic()
    train_rows,val_rows,vocab,basis,bank,metadata=prepare_data(train_graphs,val_graphs,max_nodes,cfg,request.run.train_seed)
    del train_graphs,val_graphs
    mc=_make_model_config(options,cfg,vocab,basis,max_nodes)
    model=SpectralCategoricalDenoiser(**mc).to(device)
    schedule=cosine_alpha_bar(int(options['diffusion']['steps']),device=device)
    xn=MarginalNoise(metadata['node_marginal'],schedule); en=MarginalNoise(metadata['edge_marginal'],schedule)
    tc=options['train']
    optimizer=torch.optim.AdamW(model.parameters(),lr=float(tc['lr']),weight_decay=float(tc['weight_decay']))
    torch_rng=torch.Generator(device=device).manual_seed(request.run.train_seed+91)
    order_rng=np.random.default_rng(request.run.train_seed+92)
    batch_size=int(tc['batch_size']); history=[]; best=float('inf'); best_epoch=0; best_state=None; optimizer_steps=0
    layout.train_dir.parent.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix='.gdsm_categorical_train_',dir=layout.train_dir.parent))
    try:
        with (staging/'train.log').open('w') as log:
            log.write(json.dumps({'stage':'preparation','train_graphs':len(train_rows),'val_graphs':len(val_rows),
                                  'typed_graphlet_classes':basis.dimension,'duration_seconds':time.monotonic()-started})+'\n')
            for epoch in range(1,int(tc['epochs'])+1):
                model.train(); sums=defaultdict(float); count=0
                order=order_rng.permutation(len(train_rows))
                for start in range(0,len(order),batch_size):
                    rows=[train_rows[i] for i in order[start:start+batch_size]]
                    loss,parts=_loss_batch(model,rows,basis,cfg,schedule,xn,en,torch_rng,device,permutations=True)
                    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite joint training loss')
                    optimizer.zero_grad(set_to_none=True); loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),float(tc.get('grad_norm') or 1.))
                    optimizer.step(); optimizer_steps+=1
                    for key,value in {'loss':loss,**parts}.items(): sums[key]+=float(value.detach())*len(rows)
                    count+=len(rows)
                row={'epoch':epoch,'optimizer_steps':optimizer_steps,**{'train_'+k:v/count for k,v in sums.items()}}
                if epoch==1 or epoch%int(tc['validation_every'])==0 or epoch==int(tc['epochs']):
                    model.eval(); vsums=defaultdict(float); vcount=0
                    vrng=torch.Generator(device=device).manual_seed(request.run.train_seed+100003)
                    with torch.no_grad():
                        for start in range(0,len(val_rows),batch_size):
                            rows=val_rows[start:start+batch_size]
                            loss,parts=_loss_batch(model,rows,basis,cfg,schedule,xn,en,vrng,device,permutations=False)
                            for key,value in {'loss':loss,**parts}.items(): vsums[key]+=float(value)*len(rows)
                            vcount+=len(rows)
                    row.update({'val_'+k:v/vcount for k,v in vsums.items()})
                    if not np.isfinite(row['val_loss']): raise FloatingPointError('Nonfinite validation loss')
                    if row['val_loss']<best:
                        best=row['val_loss']; best_epoch=epoch
                        best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                history.append(row); log.write(json.dumps(row)+'\n'); log.flush()
                if epoch==1 or epoch%int(tc['log_every'])==0 or epoch==int(tc['epochs']):
                    print(f"[gdsm-categorical] epoch={epoch} train={row['train_loss']:.5f} best_val={best:.5f}",flush=True)
        (staging/'checkpoints').mkdir()
        checkpoint=staging/'checkpoints/gdsm_simple.pt'
        state={'format':FORMAT,'model_state':best_state,'model_config':mc,'categorical_config':cfg,
               'diffusion_steps':len(schedule)-1,'schema':metadata,'basis_bank':bank,
               'basis_degree_sequences':[r['degrees'].tolist() for r in train_rows],
               'max_nodes':max_nodes,'optimizer_steps':optimizer_steps,'best_epoch':best_epoch,'best_val_loss':best,'history':history}
        torch.save(state,checkpoint)
        _write_json(staging/'training_metrics.json',{'history':history,'best_epoch':best_epoch,'best_val_loss':best})
        _write_json(staging/'categorical_schema.json',metadata)
        resolved_options=copy.deepcopy(options)
        resolved_options['extensions']['attributed_categorical']=cfg
        (staging/'resolved_config.yaml').write_text(yaml.safe_dump({wrapper.model_id:resolved_options},sort_keys=False))
        manifest={'format':'grapher_gdsm_spectral_categorical_training_v1','model_id':wrapper.model_id,
                  'run_id':request.run.run_id,'train_seed':request.run.train_seed,'created_at':datetime.now(timezone.utc).isoformat(),
                  'duration_seconds':time.monotonic()-started,'optimizer_steps':optimizer_steps,
                  'dataset':{'benchmark_id':request.dataset.benchmark_id,'serialized_id':request.dataset.serialized_id,
                             'fingerprint':fingerprint,'split_sha256':{k:_sha256(v) for k,v in request.dataset.split_paths.items()},
                             'num_train_graphs_used':len(train_rows),'num_val_graphs_used':len(val_rows)},
                  'options':_jsonable(options),'checkpoint':{'path':'checkpoints/gdsm_simple.pt','sha256':_sha256(checkpoint)},
                  'checkpoint_selection':{'kind':'best_validation_joint_loss','epoch':best_epoch,'loss':best},
                  'contract':{'spectral_prediction':'clean_sorted_binary_adjacency_eigenvalues_div_sqrt_n',
                              'categorical_noise':'marginal_nodes_and_edges','categorical_posterior':'clean_endpoint_mixture_exact_skip_posterior',
                              'schedule':'cosine_exact_terminal_0_clean_T_marginal','node_alignment':'same_graph_categorical_corruption',
                              'categorical_adjacency_is_authoritative':True,'basis':'current_categorical_adjacency',
                              'spectral_proposal_degeneracy':'mean_coefficients_within_equal_current_eigenvalue_blocks',
                              'rewiring_during_training':False,'original_degrees_enforced':False},
                  'test_used_for_training':False}
        _write_json(staging/'manifest.json',manifest)
        if layout.train_dir.exists(): shutil.rmtree(layout.train_dir)
        staging.replace(layout.train_dir)
        _write_json(layout.run_manifest_path,{'format':'grapher_baseline_run_v1','model_id':wrapper.model_id,
                                             'dataset_id':request.run.dataset_id,'run_id':request.run.run_id,'train_seed':request.run.train_seed})
        return artifacts
    except BaseException:
        shutil.rmtree(staging,ignore_errors=True); raise


def validate_generation(state,options):
    if state.get('format')!=FORMAT:
        raise ValueError('Categorical diffusion needs a new jointly trained checkpoint; legacy Structure3 cannot be reused')
    cfg=resolve(options); trained=state['categorical_config']
    for key in ('categories','noise','graphlets','spectral_conditioning','loss_weights'):
        if cfg[key]!=trained[key]:
            raise ValueError(f'Generation cannot change trained attributed_categorical.{key}; retrain under a new run-id')
    # The degree generator is generation-only: training anchors use the clean
    # graph degree sequence, never this sampler.  Table-7 ablations may switch
    # learned vs empirical degree sources while reusing the same denoiser.
    trained_init=copy.deepcopy(trained['initialization'])
    requested_init=copy.deepcopy(cfg['initialization'])
    trained_init.pop('degree_generator',None)
    requested_init.pop('degree_generator',None)
    if requested_init!=trained_init:
        raise ValueError('Generation cannot change trained attributed_categorical.initialization except initialization.degree_generator')
    return cfg


@torch.no_grad()
def generate(wrapper,request,state,manifest,options):
    cfg=validate_generation(state,options)
    layout=request.run.layout; target=layout.generation_dir(request.resolved_generation_id)
    ArtifactLayout.require_available(target,overwrite=request.overwrite)
    device=_resolve_device(options['runtime'])
    if device.type=='cuda' and device.index is None: device=torch.device('cuda',torch.cuda.current_device())
    _seed_everything(request.generation_seed)
    model=SpectralCategoricalDenoiser(**state['model_config']).to(device)
    model.load_state_dict(state['model_state']); model.eval()
    total=int(state['diffusion_steps']); abar=cosine_alpha_bar(total,device=device)
    schema=state['schema']; vocab=GraphCategoryVocabulary.from_dict(schema['category_vocabulary'])
    basis=TypedGraphletsMulti.from_schema(schema) if schema.get('graphlet_schema_version')==2 else TypedGraphlets3(schema['graphlet_keys'])
    bins=int(cfg['graphlets']['clustering_bins'])
    xn=MarginalNoise(schema['node_marginal'],abar); en=MarginalNoise(schema['edge_marginal'],abar)
    sample_steps=int(options['sample']['steps'])
    times=np.rint(np.linspace(total,0,sample_steps+1)).astype(int).tolist()
    if any(s>=t for t,s in zip(times,times[1:])): raise ValueError('Sampling schedule must strictly decrease')
    noise_rng=torch.Generator(device=device).manual_seed(request.generation_seed)
    prior_rng=np.random.default_rng(request.generation_seed+1009)
    basis_rng=np.random.default_rng(request.generation_seed+2003)
    refine_rng=np.random.default_rng(request.generation_seed+1000003)
    bank={int(k):v for k,v in state['basis_bank'].items()}
    init=cfg['initialization']
    if init['mode']=='degree_basis':
        from grapher.models.gdsm_simple.structured_pipeline import _load_degree_sampler
        sampler,prior_info=_load_degree_sampler(init,state,manifest,device,request.generation_seed)
    else: sampler,prior_info=None,{'type':'unused_gaussian_initialization','learned':False}
    outputs=[]; initials=[]; pre_final=[]; anchors_saved=[]; degree_samples=[]
    final_values=[]; final_vectors=[]; predictions=[]; diagnostics=[]; trajectories=[]
    started=time.monotonic(); batch_size=int(options['generation_batch_size']); guide=cfg['guidance']
    acceptance=cfg['final_acceptance']
    require_connected=bool(acceptance['require_connected'])
    max_graph_attempts=(int(np.ceil(request.num_graphs*float(acceptance['max_attempt_multiplier'])))
                        if require_connected else request.num_graphs)
    attempted_graphs=0; connected_attempts=0; rejected_disconnected=0; rejection_records=[]; rejected_graphs=[]
    while len(outputs)<request.num_graphs:
        remaining_attempts=max_graph_attempts-attempted_graphs
        if remaining_attempts<=0:
            raise RuntimeError(
                'Final connected-sample acceptance exhausted its graph-attempt budget: '
                f'accepted={len(outputs)}/{request.num_graphs}, attempted={attempted_graphs}, '
                f'rejected_disconnected={rejected_disconnected}, max_attempts={max_graph_attempts}. '
                'Increase attributed_categorical.final_acceptance.max_attempt_multiplier '
                'or diagnose the categorical sampler.'
            )
        b=min(batch_size,request.num_graphs-len(outputs),remaining_attempts); ds=[]; aa=[]; init_rows=[]
        batch_attempt_start=attempted_graphs; attempted_graphs+=b
        for _ in range(b):
            if sampler is None:
                d=np.sort(np.asarray(state['basis_degree_sequences'][int(prior_rng.integers(len(state['basis_degree_sequences'])))],np.int64))[::-1]
                dd={'source':prior_info['type']}
            else:
                sample=sampler.sample(rng=prior_rng)
                d=np.sort(np.asarray(sample['degree_sequence'],np.int64))[::-1]
                dd=sample.get('sampling_diagnostics',{})
            n=len(d)
            if n<1 or n>state['max_nodes'] or not nx.is_graphical(d.tolist()):
                raise ValueError('Degree prior returned an invalid or oversized sequence; no silent fallback')
            u,idx,fallback=_basis_for_size(bank,n,basis_rng)
            if init['mode']=='degree_basis':
                anchor,ad=degree_anchor(u,d,init['ridge'],init['diagonal_weight'])
            else: anchor,ad=np.zeros(n,np.float32),{}
            ds.append(d); aa.append(anchor)
            init_rows.append({'basis_bank_index_within_size':idx,'haar_basis_fallback':fallback,
                              'anchor':ad,'degree_sampling':dd})
        width=max(map(len,ds)); mask=torch.zeros((b,width),dtype=torch.bool,device=device)
        anchor=torch.zeros((b,width),device=device)
        for i,d in enumerate(ds): mask[i,:len(d)]=True; anchor[i,:len(d)]=torch.as_tensor(aa[i],device=device)
        px=xn.marginal.expand(b,width,-1); pe=en.marginal.expand(b,width,width,-1)
        x,e=draw_graph(px,pe,mask,noise_rng)
        z=(anchor+torch.randn(anchor.shape,device=device,generator=noise_rng))*mask
        current_pairs=eigenpairs(e,mask)
        rows=[]; batch_initials=[]; batch_pre_final=[None]*b
        for i,d in enumerate(ds):
            n=len(d); ex=e[i,:n,:n].cpu().numpy(); xx=x[i,:n].cpu().numpy()
            batch_initials.append(decode_graph(xx,ex,vocab))
            rows.append({'sample_index':None,'attempt_index':batch_attempt_start+i,'num_nodes':n,'initialization':init_rows[i],
                         'categorical_degree_change_steps':0,'node_category_change_steps':0,'edge_recolor_steps':0,
                         'basis_updates':1,'guidance_events':[],'trajectory':[]})
        for t,s in zip(times,times[1:]):
            tv=torch.full((b,),t,device=device,dtype=torch.long); sv=torch.full((b,),s,device=device,dtype=torch.long)
            pred=model(x,e,z,tv,anchor,mask,total,current_pairs=current_pairs)
            rp_x=xn.reverse_probs(pred['node_logits'].softmax(-1),x,tv,sv)
            rp_e=en.reverse_probs(pred['edge_logits'].softmax(-1),e,tv,sv)
            new_x,new_e=draw_graph(rp_x,rp_e,mask,noise_rng)
            degree_changed=((e>0).sum(2)!=(new_e>0).sum(2)).any(1).cpu().tolist()
            node_changed=((x!=new_x)&mask).any(1).cpu().tolist()
            recolored=((e>0)&(new_e>0)&(e!=new_e)).any(2).any(1).cpu().tolist()
            event=bool(guide['enabled']) and s/total<=guide['start_fraction'] and (s==0 or s%int(guide['every'])==0)
            # Independently scheduled spectral feedback is late by default.
            feedback_active=s/total<=float(cfg.get('feedback_start_fraction',.2))
            for i,d in enumerate(ds):
                n=len(d)
                rows[i]['categorical_degree_change_steps']+=int(degree_changed[i])
                rows[i]['node_category_change_steps']+=int(node_changed[i])
                rows[i]['edge_recolor_steps']+=int(recolored[i])
                # Do not transfer every graph's tensors to the CPU at every
                # reverse step; only refinement, final output or trace needs them.
                if not (event or s==0 or cfg['save_trajectory']):
                    continue
                xx=new_x[i,:n].cpu().numpy(); ee=new_e[i,:n,:n].cpu().numpy()
                if s==0: batch_pre_final[i]=decode_graph(xx,ee,vocab)
                if event:
                    targets=predictions_numpy(pred,i,n)
                    ee,rd=refine(xx,ee,targets,basis,bins,guide,refine_rng)
                    new_e[i,:n,:n]=torch.as_tensor(ee,device=device,dtype=torch.long)
                    rows[i]['guidance_events'].append({'from_t':t,'to_t':s,**rd})
                if cfg['save_trajectory']:
                    rows[i]['trajectory'].append({'t':s,'node_categories':xx.copy(),'edge_categories':ee.copy()})
            # This recomputation is unconditional: even no accepted swap or only
            # a categorical update must be represented in the next decoder basis.
            current_pairs=eigenpairs(new_e,mask)
            coefficient=float(cfg['spectrum_feedback']) if feedback_active else 0.
            guided=(1-coefficient)*pred['clean_spectrum']+coefficient*current_pairs[0]
            z=spectral_reverse(z,guided,anchor,abar,tv,sv,mask)
            x,e=new_x,new_e
            for row in rows: row['basis_updates']+=1
        for i,d in enumerate(ds):
            n=len(d); xx=x[i,:n].cpu().numpy(); ee=e[i,:n,:n].cpu().numpy()
            graph=decode_graph(xx,ee,vocab); connected=nx.is_connected(graph)
            connected_attempts+=int(connected)
            if require_connected and not connected:
                rejected_disconnected+=1
                components=sorted((len(c) for c in nx.connected_components(graph)),reverse=True)
                rejection_records.append({
                    'attempt_index':rows[i]['attempt_index'],'reason':'disconnected_final_graph',
                    'num_nodes':n,'num_edges':int(graph.number_of_edges()),
                    'num_components':len(components),'component_sizes':components,
                    'accepted_guidance_steps':int(sum(ev['accepted_steps'] for ev in rows[i]['guidance_events'])),
                })
                rejected_graphs.append(graph)
                continue
            accepted_index=len(outputs); rows[i]['sample_index']=accepted_index
            initial=batch_initials[i]
            outputs.append(graph); initials.append(initial); pre_final.append(batch_pre_final[i])
            degree_samples.append(d.tolist()); anchors_saved.append(aa[i])
            values=current_pairs[0][i,:n].cpu().numpy(); vectors=current_pairs[1][i,:n,:n].cpu().numpy()
            final_values.append(values); final_vectors.append(vectors)
            predictions.append(predictions_numpy(pred,i,n))
            actual=(ee>0).sum(1)
            rows[i].update({'final_degrees':actual.tolist(),'prior_degree_preserved':bool(np.array_equal(np.sort(actual),np.sort(d))),
                            'initial_degree_preserved':bool(np.array_equal(actual,np.array([initial.degree(v) for v in range(n)]))),
                            'connected':connected,
                            'final_eigenpair_reconstruction_max_error':float(np.max(np.abs((vectors*(values*np.sqrt(n))[None])@vectors.T-(ee>0)))),
                            'final_typed_graphlet_overflow_mass':(basis.overflow_mean(basis.summary(xx,ee)[0]) if getattr(basis,'multiscale',False) else float(basis.summary(xx,ee)[0][-1]))})
            if getattr(basis,'multiscale',False):
                rows[i]['final_typed_graphlet_overflow_by_order']=basis.overflow_by_order(basis.summary(xx,ee)[0])
            if cfg['save_trajectory']: trajectories.append(rows[i].pop('trajectory'))
            else: rows[i].pop('trajectory')
            diagnostics.append(rows[i])
        print(f'[gdsm-categorical] generated {len(outputs)}/{request.num_graphs} attempts={attempted_graphs} '
              f'rejected_disconnected={rejected_disconnected}',flush=True)
    target.parent.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix='.gdsm_categorical_generation_',dir=target.parent))
    try:
        objects={'base_graphs.pkl':outputs,'initial_graphs.pkl':initials,'final_pre_rewire_graphs.pkl':pre_final,
                 'sampled_degree_sequences.pkl':degree_samples,'spectral_anchors.pkl':anchors_saved,
                 'final_degree_sequences.pkl':[r['final_degrees'] for r in diagnostics],
                 'final_adjacency_eigenvalues.pkl':final_values,'final_eigenvectors.pkl':final_vectors,
                 'predicted_summaries.pkl':predictions}
        if request.run.dataset_id in ('qm9','zinc'): objects['molecular_graphs.pkl']=outputs
        if rejected_graphs: objects['rejected_disconnected_graphs.pkl']=rejected_graphs
        if cfg['save_trajectory']: objects['categorical_trajectories.pkl']=trajectories
        hashes={}
        for name,obj in objects.items():
            with (staging/name).open('wb') as f: pickle.dump(obj,f,protocol=pickle.HIGHEST_PROTOCOL)
            hashes[name]=_sha256(staging/name)
        aggregate={'num_graphs':len(outputs),'requested':request.num_graphs,
                   'generation_attempts':attempted_graphs,
                   'rejected_disconnected_final_graphs':rejected_disconnected,
                   'replacement_attempts':attempted_graphs-len(outputs),
                   'generation_yield':len(outputs)/max(attempted_graphs,1),
                   'raw_final_connectedness_rate':connected_attempts/max(attempted_graphs,1),
                   'prior_degree_preservation_rate':float(np.mean([r['prior_degree_preserved'] for r in diagnostics])),
                   'initial_degree_preservation_rate':float(np.mean([r['initial_degree_preserved'] for r in diagnostics])),
                   'connectedness_rate':float(np.mean([r['connected'] for r in diagnostics])),
                   'categorical_degree_change_steps_mean':float(np.mean([r['categorical_degree_change_steps'] for r in diagnostics])),
                   'mean_accepted_steps':float(np.mean([sum(ev['accepted_steps'] for ev in r['guidance_events']) for r in diagnostics])),
                   'basis_updates_per_graph':sample_steps+1,
                   'max_final_eigenpair_reconstruction_error':max(r['final_eigenpair_reconstruction_max_error'] for r in diagnostics)}
        _write_json(staging/'rewiring_diagnostics.json',{'aggregate':aggregate,'graphs':diagnostics,
                    'final_rejections':rejection_records})
        _write_json(staging/'final_acceptance_diagnostics.json',{
                    'config':acceptance,'num_requested':request.num_graphs,'num_attempted':attempted_graphs,
                    'num_returned':len(outputs),'num_rejected_disconnected':rejected_disconnected,
                    'raw_final_connectedness_rate':aggregate['raw_final_connectedness_rate'],
                    'generation_yield':aggregate['generation_yield'],'records':rejection_records})
        _write_json(staging/'categorical_schema.json',schema)
        generation_manifest={
            'format':'grapher_gdsm_spectral_categorical_generation_v1','model_id':wrapper.model_id,
            'run_id':request.run.run_id,'generation_id':request.resolved_generation_id,'generation_seed':request.generation_seed,
            'num_requested':request.num_graphs,'num_generated':len(outputs),'generation_count':len(outputs),
            'num_attempted':attempted_graphs,'generation_yield':aggregate['generation_yield'],
            'domain':'attributed','duration_seconds':time.monotonic()-started,
            'base_graphs':{'path':'base_graphs.pkl','sha256':hashes['base_graphs.pkl'],'role':'final_categorical_graph_after_local_refinement'},
            'initial_graphs':{'path':'initial_graphs.pkl','role':'terminal_categorical_marginal_samples_not_degree_exact'},
            'final_pre_rewire_graphs':{'path':'final_pre_rewire_graphs.pkl','role':'last_posterior_draw_before_final_local_swaps'},
            'checkpoint':{'path':str(request.checkpoint_path.resolve()),'sha256':_sha256(request.checkpoint_path)},
            'dataset':manifest['dataset'],'degree_prior':prior_info,'categorical_config':cfg,
            'sampling_timesteps':times,'artifact_sha256':hashes,'diagnostics':aggregate,
            'decode':{'categorical_edge_state_is_authoritative':True,'spectral_thresholding':False,
                      'basis':'recomputed_from_current_binary_categorical_adjacency_every_step','fixed_initial_eigenbasis':False,
                      'initial_degree_projection':False,'edge_absorbing_noise':False,'construct_projector':False},
            'prior_note':'Degrees initialize a soft spectral anchor only. The fixed anchor remains the source-centred coordinate offset; categories start independently at the exact marginal endpoint.',
            'sampling_intervention_note':'Local rewiring, spectral feedback, and optional final-sample acceptance modify the learned reverse kernel; no exact likelihood/ConStruct constraint guarantee is claimed.',
            'final_sample_acceptance':{**acceptance,'mode':('reject_and_resample' if require_connected else 'disabled'),
                                       'num_attempted':attempted_graphs,'num_rejected_disconnected':rejected_disconnected,
                                       'generation_yield':aggregate['generation_yield'],
                                       'diagnostics_path':'final_acceptance_diagnostics.json'},
            'posthoc_repair':False,'largest_component_filter':False,'rejected_final_graphs':rejected_disconnected,
        }
        _write_json(staging/'manifest.json',generation_manifest)
        (staging/'generation.log').write_text(json.dumps(aggregate,indent=2)+'\n')
        if target.exists(): shutil.rmtree(target)
        staging.replace(target)
        return GenerationArtifacts(run_dir=layout.run_dir,generation_dir=target,graphs_path=target/'base_graphs.pkl',
                                   manifest_path=target/'manifest.json',num_requested=request.num_graphs,num_generated=len(outputs),
                                   graphs_sha256=hashes['base_graphs.pkl'],log_path=target/'generation.log')
    except BaseException:
        shutil.rmtree(staging,ignore_errors=True); raise

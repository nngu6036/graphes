"""Soft bridge sampling followed by hard, same-bond-type constrained realization."""
from __future__ import annotations

from collections import Counter,defaultdict
from copy import deepcopy
from itertools import combinations
import math
from pathlib import Path
import time

import networkx as nx
import numpy as np
import torch

from grapher.models.dhvae_hh.typed_constructor import construct_typed_graph,TypedConstructionError
from grapher.rewiring_mlp.attributed.soft_edge_bridge import labels_to_logits,advance_edges,spectral_noise,edge_probabilities
from grapher.rewiring_mlp.attributed.joint_typed_edge_model import load_checkpoint
from grapher.rewiring_mlp.attributed.joint_typed_edge_data import (
    collate,inference_item,load_splits,graph_record,record_hash,graph_from_record,
)
from grapher.rewiring_mlp.attributed.joint_typed_edge_training import validate_config
from grapher.rewiring_mlp.core.rewiring import candidate_actions_from_edge_pair,is_valid_action
from grapher.rewiring_mlp.generic.clustering import extract_clustering_histogram,clustering_histogram_wasserstein
from grapher.rewiring_mlp.generic.orbit import extract_orbit_summary,orbit_summary_distance
from grapher.rewiring_mlp.molecular.typed_invariants import (
    TypedInvariant,extract_typed_invariant,typed_invariant_matches_graph,
)
from grapher.rewiring_mlp.molecular.graph_io import is_valid_molecular_graph,require_rdkit
from grapher.rewiring_mlp.generic.joint_checkpointing import atomic_json,file_sha256
from grapher.utils.io import save_pickle
from grapher.rewiring_mlp.generic.induced_graphlets import (
    InducedGraphletSpec, InducedGraphletCounter, extract_histogram as extract_induced_histogram,
    histogram_distance as induced_histogram_distance, validate_histogram as validate_induced_histogram,
)
from grapher.rewiring_mlp.attributed.induced_graphlets import (
    AttributedInducedGraphletCounter,
    histogram_distance as attributed_histogram_distance,
    metadata as attributed_graphlet_metadata,
    validate_histogram as validate_attributed_histogram, validate_model_graphlets,
)


from .adjacency_diffusion import ADJACENCY_MODE, LEGACY_MODE, validate_model_config


@torch.no_grad()
def sample_soft_endpoint(model,source,config,*,seed):
    """Maintain an independent SOFT trajectory; never overwrite it with hard edits.

    The learned endpoint estimate replaces the unknown terminal logits in each
    analytic Gaussian bridge transition. This finite-step plug-in sampler is an
    explicit approximation; the final hard graph is produced by the rewirer.
    """
    model.eval(); device=next(model.parameters()).device
    validate_model_config(model, config)
    legacy_spectral = model.spectral_mode == LEGACY_MODE and model.spectral_enabled
    batch=collate([inference_item(source,model.vectorizer,model.atom_types,
                  include_spectra=(model.spectral_mode==LEGACY_MODE))],
                  model.vectorizer,model.atom_types,device=device)
    mask=batch['mask']; cfg=config['edge_diffusion']; steps=int(cfg.get('sampling_steps',32))
    if steps<2: raise ValueError('At least two bridge sampling steps are required.')
    gen=torch.Generator(device=device).manual_seed(int(seed))
    state=labels_to_logits(batch['source_labels'],model.categories,mask,model.smoothing)
    spectral=(batch['source_spectra']/batch['spectral_scale'][...,None]) if legacy_spectral else None
    trace=[]
    for i in range(steps):
        t=i/steps; s=(i+1)/steps
        inp={**batch,'time':torch.full((1,),t,device=device),'edge_state':state}
        if legacy_spectral: inp['spectral_state']=spectral
        prediction=model(inp)
        state=advance_edges(state,prediction['clean_edge_logits'],t,s,mask,float(cfg.get('sigma',1)),generator=gen)
        if legacy_spectral:
            end=prediction['clean_spectra']/batch['spectral_scale'][...,None]
            a=(s-t)/(1-t); sd=float(cfg.get('spectral_sigma',0.15))*math.sqrt((s-t)*(1-s)/(1-t))
            spectral=spectral+a*(end-spectral)+sd*spectral_noise(spectral,mask,generator=gen)
        if not torch.isfinite(state).all(): raise FloatingPointError('Nonfinite soft bridge state.')
        trace.append({'step':i+1,'time':s,'edge_logit_rms':float(state.square().mean().sqrt().cpu())})
    # Predict summaries on the sampled clean state; pair probabilities come from
    # that sampled endpoint, NOT a new independent argmax/categorical graph draw.
    final_input={**batch,'time':torch.ones(1,device=device),'edge_state':state}
    if legacy_spectral: final_input['spectral_state']=spectral
    final=model(final_input)
    n=len(source)
    targets={'edge_probabilities':edge_probabilities(state,mask)[0,:n,:n].cpu().numpy()}
    if model.histogram_bins: targets['histogram']=final['clean_clustering_histogram'][0].cpu().numpy()
    if model.orbit_enabled: targets['orbit']=final['clean_orbit_summary'][0].cpu().numpy()
    if model.induced_graphlet_basis is not None or model.induced_graphlet_spec is not None:
        targets['induced_histogram'] = final['clean_induced_graphlet_histogram'][0].cpu().numpy()
        targets['induced_graphlet_metadata'] = model.induced_graphlet_metadata()
    if legacy_spectral: targets['spectra']=final['clean_spectra'][0,:,:n].cpu().numpy()
    if model.spectral_mode == ADJACENCY_MODE:
        targets['adjacency_metadata']=model.diffusion_metadata()
        if model.adjacency_output_spectra:
            # Crucial: use the SAMPLED endpoint whose bond probabilities are
            # passed to the rewirer, not an additional t=1 denoising prediction.
            realized=model.adjacency_features(edge_probabilities(state,mask),mask)
            targets['adjacency_spectra']=realized['spectra'][0,:,:n].cpu().numpy()
    return targets,{'prediction_calls':steps+1,'sampling_steps':steps,'trajectory':trace,
                    'diffusion':model.diffusion_metadata()}


def validate_refiner(cfg,model):
    allowed={'steps','proposal_budget','valid_candidate_budget','weights','normalization','epsilon',
             'min_improvement','preserve_connectivity','strict_same_bond','rdkit_candidate_filter'}
    unknown=set(cfg)-allowed
    if unknown: raise ValueError(f'Unsupported joint soft-edge refiner settings: {sorted(unknown)}')
    if not cfg.get('preserve_connectivity',True) or not cfg.get('strict_same_bond',True):
        raise ValueError('Joint typed edge generation requires connected same-bond-type rewiring.')
    weights=dict(cfg.get('weights',{'edge':1.0}))
    if set(weights)-{'edge','clustering','orbit','graphlet'}: raise ValueError('Only edge, clustering, orbit, and graphlet scoring are implemented.')
    if not weights or any(not math.isfinite(float(x)) or float(x)<0 for x in weights.values()) or sum(weights.values())<=0:
        raise ValueError('Need finite nonnegative guidance weights with at least one positive.')
    if weights.get('clustering',0)>0 and not model.histogram_bins: raise ValueError('Clustering guidance needs a histogram head.')
    if weights.get('orbit',0)>0 and not model.orbit_enabled: raise ValueError('Orbit guidance needs an orbit head.')
    if weights.get('graphlet',0)>0 and model.induced_graphlet_basis is None and model.induced_graphlet_spec is None: raise ValueError('Graphlet guidance needs a trained induced graphlet head.')
    for k in ('proposal_budget','valid_candidate_budget'):
        if int(cfg.get(k,256))<=0: raise ValueError(f'{k} must be positive.')
    if int(cfg.get('steps',32))<0: raise ValueError('steps must be nonnegative.')
    if cfg.get('normalization','initial') not in ('initial','none'): raise ValueError('normalization must be initial or none.')
    if float(cfg.get('epsilon',1e-6))<=0 or float(cfg.get('min_improvement',1e-8))<0: raise ValueError('Bad improvement thresholds.')


def candidate_graphs(current,edge_types,cfg,rng,seen,stats):
    groups=defaultdict(list)
    for u,v,d in current.edges(data=True): groups[d['bond_type']].append(tuple(sorted((u,v))))
    pairs=[(r,e1,e2) for r,edges in groups.items() for e1,e2 in combinations(sorted(edges),2)]
    order=rng.permutation(len(pairs)); proposed=0; accepted=0
    for i in order:
        r,e1,e2=pairs[int(i)]
        actions=candidate_actions_from_edge_pair(e1,e2); rng.shuffle(actions)
        for action in actions:
            if proposed>=int(cfg.get('proposal_budget',1024)) or accepted>=int(cfg.get('valid_candidate_budget',256)):
                return
            proposed+=1; stats['proposals']+=1
            if not is_valid_action(current,action,preserve_connectivity=True):
                stats['topology_rejections']+=1; continue
            candidate=current.copy(); removed,added=action
            candidate.remove_edges_from(removed)
            for u,v in added:
                candidate.add_edge(u,v,bond_type=int(r),bond_order=1.5 if r==4 else float(r))
            key=record_hash(graph_record(candidate))
            if key in seen:
                stats['visited_rejections']+=1; continue
            if cfg.get('rdkit_candidate_filter',True) and not is_valid_molecular_graph(candidate):
                stats['rdkit_rejections']+=1; continue
            accepted+=1; stats['valid_candidates_scored']+=1
            yield action,candidate,key


def edge_energy(graph,probabilities,edge_types):
    """All unordered pairs: no-bond log probabilities MUST be included."""
    n=len(graph); labels=np.zeros((n,n),dtype=np.int64)
    for u,v,d in graph.edges(data=True): labels[u,v]=labels[v,u]=edge_types.index(d['bond_type'])+1
    u,v=np.triu_indices(n,1)
    if len(u)==0: return 0.0
    return float(-np.log(np.maximum(probabilities[u,v,labels[u,v]],1e-12)).mean())


def refine_typed_graph(source,targets,model,config,*,seed):
    cfg=config['attributed_refiner']; validate_refiner(cfg,model)
    current=source.copy(); edge_types=model.edge_types; rng=np.random.default_rng(seed)
    invariant=extract_typed_invariant(source,edge_types=edge_types)
    weights={k:float(v) for k,v in cfg.get('weights',{'edge':1.0}).items() if float(v)>0}
    probs=np.asarray(targets['edge_probabilities'],dtype=np.float64)
    if probs.shape!=(len(source),len(source),model.categories) or not np.isfinite(probs).all():
        raise ValueError('Wrong endpoint probability shape or nonfinite values.')
    if np.any(probs<0) or not np.allclose(probs.sum(-1),1,atol=1e-5) or not np.allclose(probs,probs.transpose(1,0,2),atol=1e-6):
        raise ValueError('Endpoint edge probabilities must be symmetric distributions.')

    graphlet_counter = None
    graphlet_basis = model.induced_graphlet_basis
    graphlet_spec = model.induced_graphlet_spec
    if 'graphlet' in weights:
        if targets.get('induced_graphlet_metadata') != model.induced_graphlet_metadata():
            raise ValueError('Soft endpoint induced graphlet catalogue differs from the model.')
        if graphlet_basis is not None:
            validate_attributed_histogram(targets.get('induced_histogram'), graphlet_basis)
            graphlet_counter = AttributedInducedGraphletCounter(current, graphlet_basis)
        else:
            validate_induced_histogram(targets.get('induced_histogram'), graphlet_spec)
            graphlet_counter = InducedGraphletCounter(current, graphlet_spec)
    def discrepancies(g, *, action=None, graphlet_histogram=None):
        out={}
        if 'edge' in weights: out['edge']=edge_energy(g,probs,edge_types)
        if 'clustering' in weights:
            out['clustering']=clustering_histogram_wasserstein(extract_clustering_histogram(g,model.histogram_bins),targets['histogram'])
        if 'orbit' in weights: out['orbit']=orbit_summary_distance(extract_orbit_summary(g),targets['orbit'],distance='log_rmse')
        if graphlet_counter is not None:
            if graphlet_histogram is None:
                if graphlet_basis is not None:
                    graphlet_histogram = graphlet_counter.histogram() if g is current else graphlet_counter.candidate_histogram(g, action)
                else:
                    graphlet_histogram = graphlet_counter.histogram() if g is current else graphlet_counter.candidate_histogram(g)
            if graphlet_basis is not None:
                out['graphlet'] = (attributed_histogram_distance(graphlet_histogram, targets['induced_histogram'], graphlet_basis)
                    if len(g) >= graphlet_counter.k else 0.0)
            else:
                out['graphlet'] = (induced_histogram_distance(graphlet_histogram, targets['induced_histogram'], graphlet_spec)
                    if len(g) >= graphlet_spec.k else 0.0)
        return out
    initial=discrepancies(current); eps=float(cfg.get('epsilon',1e-6))
    scales={k:max(v,eps) if cfg.get('normalization','initial')=='initial' else 1.0 for k,v in initial.items()}
    def energy(d): return sum(weights[k]*v/scales[k] for k,v in d.items())
    before=initial; current_energy=energy(before); trace=[]; seen={record_hash(graph_record(current))}; stop='step_limit'
    search_totals=Counter()
    for step in range(int(cfg.get('steps',32))):
        best=None; valid=0; search_step=Counter()
        for action,candidate,key in candidate_graphs(current,edge_types,cfg,rng,seen,search_step):
            valid+=1
            candidate_histogram = None
            if graphlet_counter is not None and graphlet_basis is not None:
                candidate_histogram = graphlet_counter.candidate_histogram(candidate, action)
            elif graphlet_counter is not None:
                candidate_histogram = graphlet_counter.candidate_histogram(candidate)
            distances=discrepancies(candidate, action=action, graphlet_histogram=candidate_histogram); e=energy(distances)
            if best is None or e<best[0]: best=(e,candidate,key,action,distances)
        search_totals.update(search_step)
        if best is None: stop='no_sampled_valid_same_type_swap'; break
        e,candidate,key,action,distances=best
        gain=current_energy-e
        if gain<=float(cfg.get('min_improvement',1e-8)):
            stop='no_sampled_frozen_target_improvement'; break
        if not typed_invariant_matches_graph(candidate,invariant): raise AssertionError('Same-type swap changed indexed typed degrees.')
        if not nx.is_connected(candidate): raise AssertionError('Swap disconnected graph.')
        trace.append({'step':step+1,'valid_candidates_scored':valid,'energy_before':current_energy,
                      'energy_after':e,'gain':gain,'discrepancies_before':before,'discrepancies_after':distances,
                      'removed':action[0],'added':action[1],'candidate_search':dict(search_step)})
        current=candidate; before=distances; current_energy=e; seen.add(key)
        if graphlet_counter is not None:
            if graphlet_basis is not None:
                graphlet_counter.accept(current, action)
            else:
                graphlet_counter = InducedGraphletCounter(current, model.induced_graphlet_spec)
    return current,{'accepted_steps':len(trace),'stop_reason':stop,'trace':trace,'initial_discrepancies':initial,
                    'final_discrepancies':before,'normalization_scales':scales,'weights':weights,
                    'typed_degree_preserved':typed_invariant_matches_graph(current,invariant),
                    'connected':nx.is_connected(current),'candidate_search_totals':dict(search_totals),'target_scope':'single_frozen_soft_bridge_endpoint'}


def generation_sources(model,train_graphs,config,*,seed,num_generate):
    """One independent source RNG per output. Rewiring cannot consume this stream."""
    gc=config.get('generation',{}); source_mode=gc.get('invariant_source','learned')
    if source_mode not in ('learned','train_empirical'): raise ValueError('invariant_source must be learned or train_empirical (no held-out oracle default).')
    if config.get('degree_generator',{}).get('checkpoint_path'):
        raise ValueError('Joint generation must use the embedded typed-DH-VAE, not an external checkpoint.')
    require_valid=bool(gc.get('require_rdkit_source_validity',True))
    if require_valid or config['attributed_refiner'].get('rdkit_candidate_filter',True): require_rdkit()
    dm=model.degree_model; v=model.vectorizer; dev=next(model.parameters()).device
    mode=gc.get('sample_num_nodes','empirical')
    if mode not in ('empirical','model'): raise ValueError('sample_num_nodes must be empirical or model.')
    counter=Counter(); counter['requested_graphs']=num_generate
    for index in range(num_generate):
        source_started=time.perf_counter()
        rng=np.random.default_rng(np.random.SeedSequence([int(seed),index,1907]))
        result=None
        for attempt in range(int(gc.get('max_attempts_per_graph',128))):
            counter['invariant_proposals']+=1
            if source_mode=='train_empirical':
                graph=train_graphs[int(rng.integers(len(train_graphs)))]
                invariant=extract_typed_invariant(graph,edge_types=model.edge_types)
                samp={'attempts_used':1,'fallback_used':False}
            else:
                nodes=[v.sample_empirical_node_count(rng)] if mode=='empirical' else None
                devices=[dev.index or 0] if dev.type=='cuda' else []
                # Use a reproducible per-proposal Torch seed without modifying the denoising RNG.
                with torch.random.fork_rng(devices=devices),torch.no_grad():
                    torch.manual_seed(int(rng.integers(0,2**31-1)))
                    outputs=dm.sample_outputs(1,node_counts=nodes,device=dev)
                budget=int(gc.get('max_invariant_resample',1000))
                try:
                    summary=v.outputs_to_summaries(outputs,rng=rng,max_resample=budget,
                              fallback='error',include_diagnostics=True)[0]
                    invariant=TypedInvariant.from_dict(summary['typed_invariant']); samp=summary['sampling_diagnostics']
                    counter['histogram_draws']+=samp['attempts_used']
                except RuntimeError:
                    counter['histogram_draws']+=budget; counter['invariant_sampling_failures']+=1; continue
            try:
                constructed,diag=construct_typed_graph(invariant,config.get('constructor'),rng)
                source=graph_from_record(graph_record(constructed))
            except TypedConstructionError:
                counter['constructor_failures']+=1; continue
            counter['constructed_sources']+=1
            if require_valid and not is_valid_molecular_graph(source):
                counter['rdkit_source_rejections']+=1; continue
            observed=extract_typed_invariant(source,edge_types=model.edge_types)
            if Counter(observed.signatures)!=Counter(invariant.signatures): raise AssertionError('Constructor changed typed multiset.')
            result=(source,{'source_index':index,'attempts':attempt+1,'sampling':samp,'constructor':diag,
                            'constructor_target_typed_match':True,'source_seconds':time.perf_counter()-source_started},dict(counter)); break
        if result is None:
            raise RuntimeError(f'Failed to realize source {index} within sampling/constructor/RDKit budget. Counts={dict(counter)}. No empirical fallback was used.')
        yield result


def generate_joint_typed_edge(config,args):
    validate_config(config); config=deepcopy(config)
    seed=int(args.seed if args.seed is not None else config.get('seed',42)); config['seed']=seed
    checkpoint=args.checkpoint or config['attributed_predictor'].get('checkpoint_path')
    if not checkpoint: raise ValueError('Supply the trained joint attributed checkpoint.')
    model,ckpt=load_checkpoint(checkpoint,args.device or 'auto'); validate_refiner(config['attributed_refiner'],model)
    validate_model_config(model,config)
    validate_model_graphlets(model, config)
    cat=config['categorical_state']
    if tuple(cat['edge_categories'])!=model.edge_types or tuple(cat['node_categories'])!=model.atom_types:
        raise ValueError('Checkpoint/config atom or bond categories differ.')
    if not math.isclose(float(config['edge_diffusion'].get('smoothing',0.01)),model.smoothing):
        raise ValueError('Endpoint smoothing is a training/checkpoint semantic and cannot change during sampling.')
    trained_diff=ckpt['config']['edge_diffusion']
    for key in ('sigma','spectral_sigma','spectral_enabled','bridge'):
        default={'sigma':1.0,'spectral_sigma':0.0 if model.spectral_mode==ADJACENCY_MODE else 0.15,
                 'spectral_enabled':True,'bridge':'centered_logit_brownian'}[key]
        if config['edge_diffusion'].get(key,default)!=trained_diff.get(key,default):
            raise ValueError(f'edge_diffusion.{key} differs from training. Use the matching config.')
    splits,provenance=load_splits(config)
    if ckpt.get('dataset_provenance',{}).get('fingerprint')!=provenance['fingerprint']:
        raise ValueError('Processed dataset fingerprint differs from the joint training checkpoint. Refusing stale-dataset generation.')
    num=int(args.num_generate if args.num_generate is not None else config.get('generation',{}).get('num_generate',64))
    if num<1: raise ValueError('num_generate must be positive.')
    output=Path(args.output_dir)
    if output.exists() and any(output.iterdir()): raise FileExistsError(f'Use a fresh generation directory: {output}')
    output.mkdir(parents=True,exist_ok=True)
    sources=[]; finals=[]; records=[]; targets_saved=[]; sampling_counts={}; start=time.perf_counter()
    if model.spectral_mode == ADJACENCY_MODE:
        print('[AdjacencyDiffusion] sampling one soft categorical adjacency; signed spectra derived each step. '
              'No independent Laplacian/eigenvalue process.',flush=True)

    # Empirical conditioning must use the SAME training subset used for fitting.
    effective_limit=ckpt['config']['dataset'].get('max_train_graphs')
    train_graphs=splits['train'][:int(effective_limit)] if effective_limit else splits['train']
    report={'format':'joint_typed_soft_edge_generation_v1','config':config,'seed':seed,
      'checkpoint':str(checkpoint),'checkpoint_sha256':file_sha256(checkpoint),'checkpoint_selection':ckpt.get('selection'),
      'dataset_provenance':provenance,'degree_sampler_source':'joint_checkpoint_embedded',
      'invariant_source':config['generation'].get('invariant_source','learned'),
      'strategy':'bridge_then_rewire','node_diffusion':False,'edge_diffusion':True,
      'diffusion':model.diffusion_metadata(),'checkpoint_format':ckpt['format'],
      'summaries':'topology-only clustering and orbits; induced graphlets use node-type and edge-type labels for attributed checkpoints',
      'induced_graphlet_metadata': model.induced_graphlet_metadata(),
      'target_scope':'fixed_endpoint_per_graph','raw_graphs_no_posthoc_repair':True}
    def flush(complete=False):
        for name,obj in [('coarse_graphs.pkl',sources),('molecular_graphs.pkl',finals),('soft_endpoints.pkl',targets_saved)]:
            destination=output/(name if complete else 'partial_'+name)
            tmp=destination.with_name(destination.name+'.tmp'); save_pickle(obj,tmp); tmp.replace(destination)
        report.update({'num_generated':len(finals),'complete':complete,'records':records,'sampling_counts':sampling_counts,
            'runtime_seconds':time.perf_counter()-start,
            'source_graphs_sha256':record_hash([graph_record(g) for g in sources])})
        atomic_json(report,output/'report.json')
    try:
        for i,(source,src_report,counts) in enumerate(generation_sources(model,train_graphs,config,seed=seed,num_generate=num)):
            t0=time.perf_counter(); sampling_counts=counts
            targets,bridge=sample_soft_endpoint(model,source,config,seed=seed+i*1009+7043)
            final,refinement=refine_typed_graph(source,targets,model,config,seed=seed+i*1009+9049)
            sources.append(source); finals.append(final); targets_saved.append(targets)
            soft_seconds=time.perf_counter()-t0
            records.append({**src_report,'bridge':bridge,**refinement,'soft_sampling_and_rewiring_seconds':soft_seconds,
                            'seconds':src_report['source_seconds']+soft_seconds})
            print(f'[JointTypedEdge] graph={i+1}/{num} n={len(final)} soft_steps={bridge["sampling_steps"]} '
                  f'accepted_swaps={refinement["accepted_steps"]} typed_preserved={refinement["typed_degree_preserved"]}',flush=True)
            if (i+1)%int(config.get('generation',{}).get('checkpoint_every',16))==0: flush()
    except Exception as e:
        report['failure']=str(e); flush(False); raise
    def vals(g):
        return [sum(float(d.get('bond_order', d['bond_type'])) for _,_,d in g.edges(v,data=True)) for v in range(len(g))]
    report['diagnostics']={'node_type_preservation_rate':float(np.mean([
        all(a.nodes[v]['atomic_num']==b.nodes[v]['atomic_num'] for v in a) for a,b in zip(sources,finals)])),
        'ordinary_degree_preservation_rate':float(np.mean([dict(a.degree())==dict(b.degree()) for a,b in zip(sources,finals)])),
        'bond_type_counts_preservation_rate':float(np.mean([
            Counter(d['bond_type'] for _,_,d in a.edges(data=True))==Counter(d['bond_type'] for _,_,d in b.edges(data=True))
            for a,b in zip(sources,finals)])),
        'weighted_valence_preservation_rate':float(np.mean([np.allclose(vals(a),vals(b)) for a,b in zip(sources,finals)])),
        'typed_degree_preservation_rate' :float(np.mean([r['typed_degree_preserved'] for r in records])),
        'connectedness_rate':float(np.mean([r['connected'] for r in records])),
        'mean_accepted_steps':float(np.mean([r['accepted_steps'] for r in records])),
        'mean_bridge_prediction_calls':float(np.mean([r['bridge']['prediction_calls'] for r in records])),
        'scoring_weights':config['attributed_refiner']['weights'],
        'diffusion':model.diffusion_metadata()}
    flush(True); print(f'Saved molecular graphs: {output}/molecular_graphs.pkl',flush=True)

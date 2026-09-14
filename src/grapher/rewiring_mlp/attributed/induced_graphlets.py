from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import time
from math import comb
from typing import Any

import networkx as nx
import numpy as np

from grapher.utils.motifs import _canonicalize_attributed_tokens
from grapher.rewiring_mlp.attributed.data import GraphletBasis
from grapher.rewiring_mlp.generic.induced_graphlets import InducedGraphletSpec, validate_k
from grapher.rewiring_mlp.attributed.graphlet_diffusion import (
    candidate_attributed_graphlet_counts,
    extract_attributed_graphlet_counts,
)

VERSION = "attributed_induced_histogram_multik_v2"


def requested_sizes(config: dict) -> tuple[int, ...]:
    values = config.get("structure_summary_prediction", {}) or {}
    if not values.get("induced_graphlet_histogram", False):
        return ()
    if "induced_graphlet_k_min" in values or "induced_graphlet_k_max" in values:
        k_min = validate_k(int(values.get("induced_graphlet_k_min", values.get("induced_graphlet_k", 3))))
        k_max = validate_k(int(values.get("induced_graphlet_k_max", values.get("induced_graphlet_k", 5))))
        if k_min > k_max:
            raise ValueError("induced_graphlet_k_min must be <= induced_graphlet_k_max.")
        return tuple(range(k_min, k_max + 1))
    return (validate_k(int(values.get("induced_graphlet_k", 5))),)


def validate_supported_basis(basis: GraphletBasis) -> tuple[int, ...]:
    if not basis.attributed:
        raise ValueError("Attributed induced graphlet histograms require an attributed GraphletBasis.")
    if basis.topology_filter != "all":
        raise ValueError("Attributed induced graphlet histograms currently support topology_filter='all' only.")
    if basis.connected_only:
        raise ValueError("Attributed induced graphlet histograms currently support scope='all' only.")
    if not basis.node_attribute or not basis.edge_attribute:
        raise ValueError("Attributed induced graphlet histograms require node_attribute and edge_attribute.")
    if basis.attributed_backend != "python":
        raise ValueError("Attributed histogram extraction and local deltas require the python canonicalizer.")
    sizes = tuple(validate_k(int(k)) for k in basis.sizes)
    if not sizes or tuple(sorted(sizes)) != sizes:
        raise ValueError("Attributed graphlet orders must be a nonempty increasing subset of {3,4,5}.")
    for size in basis.sizes:
        keys = basis.keys_by_k[size]
        if not keys or len(set(keys)) != len(keys):
            raise ValueError("Attributed vocabulary must have unique nonempty bin identifiers per order.")
        if basis.overflow_key is None or basis.overflow_key not in keys:
            raise ValueError("Attributed vocabulary requires an explicit unseen-type overflow bin per order.")
    return sizes


def metadata(basis: GraphletBasis) -> dict[str, Any]:
    sizes = validate_supported_basis(basis)
    payload = {
        "version": VERSION,
        "sizes": list(sizes),
        "k_min": min(sizes),
        "k_max": max(sizes),
        "scope": "all",
        "width": basis.width,
        "block_widths": [stop-start for start,stop in basis.slices],
        "block_slices": [list(x) for x in basis.slices],
        "attributed": True,
        "node_attribute": str(basis.node_attribute),
        "edge_attribute": str(basis.edge_attribute),
        "overflow_key": basis.overflow_key,
        "normalization": {str(k): f"choose(n,{k})" for k in sizes},
        "vocabulary_policy": "observed_training_graphlets_plus_unseen_overflow_per_order",
        "canonical_backend": basis.attributed_backend,
        "bin_ids_by_k": {k:list(basis.keys_by_k[k]) for k in basis.sizes},
    }
    payload["fingerprint"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return payload


def _placeholder_block(width: int, overflow_index: int) -> np.ndarray:
    out=np.zeros(width,dtype=np.float64); out[overflow_index]=1.0; return out


def validate_histogram(values: Any, basis: GraphletBasis) -> np.ndarray:
    validate_supported_basis(basis)
    x=np.asarray(values,dtype=np.float64)
    if x.ndim != 1 or x.shape != (basis.width,):
        raise ValueError("Attributed induced graphlet histogram width mismatch.")
    if not np.isfinite(x).all() or np.any(x < -1e-7) or np.any(x > 1+1e-7):
        raise ValueError("Attributed induced graphlet histogram must be finite and nonnegative.")
    out=x.copy()
    for size,(start,stop) in zip(basis.sizes,basis.slices):
        block=np.clip(out[start:stop],0.0,1.0)
        total=float(block.sum())
        if not np.isclose(total,1.0,atol=1e-6,rtol=0.0):
            raise ValueError(f"Attributed induced graphlet block k={size} must sum to one.")
        out[start:stop]=block/max(total,1e-12)
    return out


def histogram_from_counts(counts_by_size: dict[str,dict[str,int]], num_nodes: int, basis: GraphletBasis) -> np.ndarray:
    validate_supported_basis(basis)
    blocks=[]
    for size in basis.sizes:
        k=int(size); keys=basis.keys_by_k[size]; total=comb(int(num_nodes),k) if int(num_nodes)>=k else 0
        if total<=0:
            overflow=keys.index(basis.overflow_key)
            blocks.append(_placeholder_block(len(keys),overflow)); continue
        counts=counts_by_size.get(size,{}) or {}
        block=np.asarray([float(counts.get(key,0)) for key in keys],dtype=np.float64)/float(total)
        # all k-subsets must map to exactly one observed/overflow attributed class
        if not np.isclose(block.sum(),1.0,atol=1e-8,rtol=0.0):
            raise ValueError(f"Attributed graphlet counts for k={k} do not partition all induced subsets.")
        blocks.append(block)
    return validate_histogram(np.concatenate(blocks),basis)


def extract_histogram(graph: nx.Graph,basis: GraphletBasis)->np.ndarray:
    return histogram_from_counts(extract_attributed_graphlet_counts(graph,graphlet_basis=basis),graph.number_of_nodes(),basis)


def histogram_distance(left:Any,right:Any,basis:GraphletBasis)->float:
    a,b=validate_histogram(left,basis),validate_histogram(right,basis)
    distances=[]
    for start,stop in basis.slices:
        distances.append(0.5*np.abs(a[start:stop]-b[start:stop]).sum())
    return float(np.mean(distances)) if distances else 0.0


@dataclass
class AttributedInducedGraphletCounter:
    graph:nx.Graph
    basis:GraphletBasis
    def __post_init__(self):
        self.sizes=validate_supported_basis(self.basis); self.graph=self.graph.copy()
        self.counts_by_size=extract_attributed_graphlet_counts(self.graph,graphlet_basis=self.basis)
    @property
    def k(self): return max(self.sizes)
    @property
    def min_k(self): return min(self.sizes)
    def histogram(self): return histogram_from_counts(self.counts_by_size,self.graph.number_of_nodes(),self.basis)
    def candidate_counts(self,candidate:nx.Graph,action:Any|None=None):
        if action is None: return extract_attributed_graphlet_counts(candidate,graphlet_basis=self.basis)
        return candidate_attributed_graphlet_counts(self.graph,candidate,action,current_counts=self.counts_by_size,graphlet_basis=self.basis)
    def candidate_histogram(self,candidate:nx.Graph,action:Any|None=None):
        return histogram_from_counts(self.candidate_counts(candidate,action),candidate.number_of_nodes(),self.basis)
    def accept(self,candidate:nx.Graph,action:Any|None=None):
        self.counts_by_size=self.candidate_counts(candidate,action); self.graph=candidate.copy()


def block_softmax(logits,basis:GraphletBasis):
    import torch
    return torch.cat([logits[:,start:stop].softmax(-1) for start,stop in basis.slices],dim=-1)


def prediction_and_loss(logits,target,sizes,basis:GraphletBasis):
    import torch
    import torch.nn.functional as F
    orders=validate_supported_basis(basis)
    if logits.ndim!=2 or logits.shape[-1]!=basis.width or target.shape!=logits.shape:
        raise ValueError("Wrong attributed induced graphlet output/target shape.")
    target=target.to(logits)
    losses=[]; ces=[]; tvs=[]; maes=[]; valid_fractions=[]
    for k,(start,stop) in zip(orders,basis.slices):
        block_target=target[:,start:stop]
        if not torch.isfinite(block_target).all() or (block_target < -1e-6).any() or not torch.allclose(block_target.sum(-1),torch.ones_like(sizes,dtype=target.dtype),atol=1e-5):
            raise ValueError(f"Attributed induced graphlet target block k={k} must be normalized.")
        valid=(sizes>=k).to(logits.dtype); denom=valid.sum().clamp_min(1)
        block_logits=logits[:,start:stop]; p=block_logits.softmax(-1); delta=p-block_target
        losses.append((delta.square().sum(-1)*valid).sum()/denom)
        ces.append((-(block_target*F.log_softmax(block_logits,-1)).sum(-1)*valid).sum()/denom)
        tvs.append((0.5*delta.abs().sum(-1)*valid).sum()/denom)
        totals=torch.as_tensor([comb(int(n),k) if int(n)>=k else 0 for n in sizes.detach().cpu().tolist()],dtype=logits.dtype,device=logits.device)
        maes.append((delta.abs().mean(-1)*totals*valid).sum()/denom)
        valid_fractions.append(valid.mean())
    brier=torch.stack(losses).mean(); ce=torch.stack(ces).mean()
    metrics={
      "induced_graphlet_histogram_loss":brier,
      "induced_graphlet_histogram_ce":ce,
      "induced_graphlet_histogram_tv":torch.stack(tvs).mean(),
      "induced_graphlet_count_mae":torch.stack(maes).mean(),
      "induced_graphlet_valid_fraction":torch.stack(valid_fractions).mean(),
    }
    for k,tv in zip(orders,tvs): metrics[f"induced_graphlet_histogram_tv_k{k}"]=tv
    return brier,ce,metrics


def mask_prediction(probabilities,sizes,basis:GraphletBasis):
    import torch
    validate_supported_basis(basis); blocks=[]
    for size,(start,stop) in zip(basis.sizes,basis.slices):
        k=int(size); block=probabilities[:,start:stop]
        placeholder=torch.zeros_like(block); placeholder[:,basis.keys_by_k[size].index(basis.overflow_key)]=1
        blocks.append(torch.where((sizes>=k).unsqueeze(-1),block,placeholder))
    return torch.cat(blocks,dim=-1)


__all__=["AttributedInducedGraphletCounter","block_softmax","extract_histogram","histogram_distance","histogram_from_counts","mask_prediction","metadata","prediction_and_loss","requested_sizes","validate_histogram","validate_supported_basis"]


def wants_attributed_histogram(config:dict)->bool:
    values=config.get("structure_summary_prediction",{}) or {}; flag=values.get("induced_graphlet_attributed",True)
    if not isinstance(flag,bool): raise ValueError("induced_graphlet_attributed must be boolean.")
    return bool(values.get("induced_graphlet_histogram",False) and flag)


def validate_model_graphlets(model,config:dict)->None:
    basis=model.induced_graphlet_basis
    if basis is not None:
        sizes=validate_supported_basis(basis)
        if not wants_attributed_histogram(config) or requested_sizes(config)!=sizes:
            raise ValueError("Attributed induced graphlet config/catalogue differs from checkpoint.")
        values=config.get("structure_summary_prediction",{}) or {}
        if values.get("induced_graphlet_scope","all")!="all": raise ValueError("Attributed graphlet scope must be all.")
        cat=config.get("categorical_state",{})
        if cat.get("node_attribute","atomic_num")!=basis.node_attribute or cat.get("edge_attribute","bond_type")!=basis.edge_attribute:
            raise ValueError("Attributed graphlet label attributes differ from checkpoint.")
    else:
        requested=InducedGraphletSpec.from_config(config.get("structure_summary_prediction"))
        if requested is not None and wants_attributed_histogram(config):
            raise ValueError("This checkpoint does not have an attributed graphlet vocabulary; train a new checkpoint.")
        if requested!=model.induced_graphlet_spec: raise ValueError("Induced graphlet config/catalogue differs from checkpoint.")


def fit_training_basis(config:dict,train_graphs)->GraphletBasis|None:
    if not wants_attributed_histogram(config): return None
    sizes=requested_sizes(config); values=config.get("structure_summary_prediction",{}) or {}
    if values.get("induced_graphlet_scope","all")!="all": raise ValueError("Attributed induced graphlets currently support scope=all only.")
    cat=config["categorical_state"]
    settings=dict(graphlet_history=True,graphlet_k_min=min(sizes),graphlet_k_max=max(sizes),graphlet_connected_only=False,
      graphlet_topology_filter="all",graphlet_num_samples=None,attributed=True,node_attribute=cat.get("node_attribute","atomic_num"),
      edge_attribute=cat.get("edge_attribute","bond_type"),attributed_backend="python")

    # Vocabulary discovery does not need to rescan the whole molecular training
    # split.  Honour the existing graphlet_prediction.max_basis_graphs option
    # and use a deterministic training-only sample.  Unseen classes remain
    # well-defined because every order already contains an overflow bin.
    gp=config.get("graphlet_prediction",{}) or {}
    requested_limit=gp.get("max_basis_graphs")
    basis_graphs=list(train_graphs)
    if requested_limit is not None and int(requested_limit)>0 and len(basis_graphs)>int(requested_limit):
        limit=int(requested_limit); seed=int(config.get("seed",42))
        rng=np.random.default_rng(seed)
        selected=np.sort(rng.choice(len(basis_graphs),size=limit,replace=False))
        basis_graphs=[basis_graphs[int(index)] for index in selected]
        print(f"[AttributedGraphlets] vocabulary subset: selected={len(basis_graphs)}/{len(train_graphs)} "
              f"strategy=uniform_without_replacement seed={seed} overflow=enabled",flush=True)
    print(f"[AttributedGraphlets] fitting TRAIN-only labeled vocabulary: graphs={len(basis_graphs)} k={min(sizes)}..{max(sizes)} scope=all; labels={settings['node_attribute']}/{settings['edge_attribute']}",flush=True)
    node_counts=[g.number_of_nodes() for g in basis_graphs]
    subset_counts=[sum(comb(n,k) if n>=k else 0 for k in sizes) for n in node_counts]; total_subsets=sum(subset_counts)
    pc=config.get("attributed_predictor",{}); seconds=float(pc.get("progress_interval_seconds",60)); graph_interval=int(pc.get("graphlet_progress_interval",1000))
    print(f"[AttributedGraphlets] CPU preprocessing: induced_subsets={total_subsets:,} across k={list(sizes)}; progress_interval_seconds={seconds:g} graphlet_progress_interval={graph_interval}",flush=True)
    started=last_progress=time.perf_counter(); completed_subsets=last_subsets=last_bins=0; previous_graph_finished=started
    initial_cache=_canonicalize_attributed_tokens.cache_info()
    def progress(k:int,done:int,total:int,bins:int):
      nonlocal last_progress,completed_subsets,last_subsets,last_bins,previous_graph_finished
      # The optimized exact path may report several k values after each graph;
      # sampled/filtered paths retain the historical k-major progress order.
      now=time.perf_counter(); by_graph=graph_interval>0 and done%graph_interval==0; by_time=seconds>0 and now-last_progress>=seconds
      if done not in (0,1,total) and not by_graph and not by_time:return
      print(f"[AttributedGraphlets] vocabulary progress k={k} graphs={done}/{total} bins={bins} elapsed={now-started:.1f}s",flush=True); last_progress=now; last_bins=bins
    basis=GraphletBasis.fit_from_graphs(basis_graphs,settings,attributed=True,seed=int(config.get("seed",42)),progress_callback=progress)
    info=metadata(basis)
    print(f"[AttributedGraphlets] attributed=True sizes={info['sizes']} bins={info['width']} block_widths={info['block_widths']} fingerprint={info['fingerprint']}",flush=True)
    return basis

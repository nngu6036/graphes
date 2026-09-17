"""Generation-only typed structural refinement, not a ConStruct projector.

Same-type double-edge swaps preserve the CURRENT graph's indexed degrees,
node types and per-node typed degrees. The following categorical transition
may change all of them. No permanent blocking table or initial-degree mask.
"""
from __future__ import annotations
from itertools import combinations
import networkx as nx
import numpy as np
from .data import topology_summary
from .multiscale import update_counts


def candidates(e, cfg, rng):
    edges=list(zip(*np.nonzero(np.triu(e,1))))
    budget=int(cfg['proposal_budget']); limit=int(cfg['valid_candidate_budget'])
    if len(edges)<2: return
    if budget<0:
        pairs=combinations(range(len(edges)),2)
    else:
        pairs=(tuple(rng.choice(len(edges),2,replace=False)) for _ in range(budget))
    seen=set(); count=0
    for i,j in pairs:
        u,v=edges[i]; w,z=edges[j]
        if len({u,v,w,z})!=4 or e[u,v]!=e[w,z]: continue
        for new in (((u,w),(v,z)),((u,z),(v,w))):
            if any(e[a,b]!=0 for a,b in new): continue
            key=tuple(sorted((tuple(sorted((u,v))),tuple(sorted((w,z))))))+tuple(sorted(tuple(sorted(p)) for p in new))
            if key in seen: continue
            seen.add(key)
            out=e.copy(); typ=e[u,v]
            out[u,v]=out[v,u]=out[w,z]=out[z,w]=0
            for a,b in new: out[a,b]=out[b,a]=typ
            yield out
            count+=1
            if limit>0 and count>=limit: return


def energy(x,e,target,basis,bins,weights,*,counts=None):
    hist,mass=basis.summary(x,e) if counts is None else basis.encode_counts(counts,len(x))
    if getattr(basis,'multiscale',False):
        graphlet=mass_error=0.
        per_order={}
        active_weight=sum(w for k,w in zip(basis.orders,basis.size_weights) if len(x)>=k)
        for i,(k,w) in enumerate(zip(basis.orders,basis.size_weights)):
            if len(x)<k: continue
            sl=basis.slices[k]
            dh=float(np.abs(hist[sl]-target['histogram'][sl]).sum()/2)
            dm=float((mass[i]-target['mass'][i])**2)
            w=w/max(active_weight,1e-12)
            graphlet+=w*dh;mass_error+=w*dm
            per_order[str(k)]={'graphlet':dh,'mass':dm}
    else:
        graphlet=float(np.abs(hist-target['histogram']).sum()/2)
        mass_error=float((mass-target['mass'])**2)
        per_order=None
    clustering,orbit=topology_summary(e,bins)
    raw={
        'graphlet':graphlet,
        'mass':mass_error,
        'clustering':float(np.abs(clustering.cumsum()-target['clustering'].cumsum()).mean()),
        'orbit':float(np.mean((orbit-target['orbit'])**2)),
    }
    structure=sum(raw[k]*weights[k] for k in raw)
    if weights['spectral']:
        values=np.linalg.eigvalsh((e>0).astype(float))/len(x)**.5
        raw['spectral']=float(np.mean((values-target['spectrum'])**2))
    else: raw['spectral']=0.
    ij=np.triu_indices(len(x),1)
    raw['edge']=float(-np.log(np.maximum(target['edge_probs'][ij[0],ij[1],e[ij]],1e-12)).mean()) if len(ij[0]) else 0.
    result={'total':structure+weights['spectral']*raw['spectral']+weights['edge']*raw['edge'],'structure':structure,**raw}
    if per_order is not None: result['by_order']=per_order
    return result


def refine(x,e,target,basis,bins,cfg,rng):
    e=e.copy(); before=e.copy()
    multi=getattr(basis,'multiscale',False)
    counts=basis.count(x,e) if multi else None
    initial_hist=basis.encode_counts(counts,len(x))[0] if multi else basis.summary(x,e)[0]
    old=energy(x,e,target,basis,bins,cfg['weights'],counts=counts); initial=dict(old)
    seen={e.tobytes()}; accepted=[]; tested=0
    keep_connected=bool(cfg['preserve_connectivity_if_connected']) and nx.is_connected(nx.from_numpy_array(e>0))
    for _ in range(int(cfg['max_steps_per_event'])):
        best,best_energy=None,old
        best_counts=counts
        for candidate in candidates(e,cfg,rng):
            if candidate.tobytes() in seen: continue
            if keep_connected and not nx.is_connected(nx.from_numpy_array(candidate>0)): continue
            tested+=1
            candidate_counts=(update_counts(x,e,candidate,counts,basis.orders,limit=basis.limit) if multi else None)
            score=energy(x,candidate,target,basis,bins,cfg['weights'],counts=candidate_counts)
            if cfg['require_structure_improvement'] and score['structure']>=old['structure']-cfg['min_improvement']: continue
            if score['total']<best_energy['total']-cfg['min_improvement']:
                best,best_energy=candidate,score
                best_counts=candidate_counts
        if best is None: break
        accepted.append({'before':old,'after':best_energy}); e,old=best,best_energy; counts=best_counts; seen.add(e.tobytes())
    degree_equal=np.array_equal((e>0).sum(1),(before>0).sum(1))
    types_equal=all(np.array_equal((e==k).sum(1),(before==k).sum(1)) for k in range(1,int(max(e.max(),before.max()))+1))
    if not degree_equal or not types_equal: raise AssertionError('Local swap invariant violation')
    final_hist=basis.encode_counts(counts,len(x))[0] if multi else basis.summary(x,e)[0]
    diagnostics={'initial':initial,'final':old,'accepted_steps':len(accepted),'tested_candidates':tested,
              'degree_preserved':degree_equal,'typed_degrees_preserved':types_equal,
              'accepted':accepted,'initial_overflow_mass':basis.overflow_mean(initial_hist) if multi else float(initial_hist[-1]),
              'final_overflow_mass':basis.overflow_mean(final_hist) if multi else float(final_hist[-1])}
    if multi:
        diagnostics['initial_overflow_by_order']=basis.overflow_by_order(initial_hist)
        diagnostics['final_overflow_by_order']=basis.overflow_by_order(final_hist)
        diagnostics['graphlet_counting']='exact_connected_induced_local_delta'
    return e,diagnostics

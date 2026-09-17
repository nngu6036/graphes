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


def energy(x,e,target,basis,bins,weights):
    hist,mass=basis.summary(x,e)
    clustering,orbit=topology_summary(e,bins)
    raw={
        'graphlet':float(np.abs(hist-target['histogram']).sum()/2),
        'mass':float((mass-target['mass'])**2),
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
    return {'total':structure+weights['spectral']*raw['spectral']+weights['edge']*raw['edge'],'structure':structure,**raw}


def refine(x,e,target,basis,bins,cfg,rng):
    e=e.copy(); before=e.copy(); old=energy(x,e,target,basis,bins,cfg['weights']); initial=dict(old)
    seen={e.tobytes()}; accepted=[]; tested=0
    keep_connected=bool(cfg['preserve_connectivity_if_connected']) and nx.is_connected(nx.from_numpy_array(e>0))
    for _ in range(int(cfg['max_steps_per_event'])):
        best,best_energy=None,old
        for candidate in candidates(e,cfg,rng):
            if candidate.tobytes() in seen: continue
            if keep_connected and not nx.is_connected(nx.from_numpy_array(candidate>0)): continue
            tested+=1
            score=energy(x,candidate,target,basis,bins,cfg['weights'])
            if cfg['require_structure_improvement'] and score['structure']>=old['structure']-cfg['min_improvement']: continue
            if score['total']<best_energy['total']-cfg['min_improvement']:
                best,best_energy=candidate,score
        if best is None: break
        accepted.append({'before':old,'after':best_energy}); e,old=best,best_energy; seen.add(e.tobytes())
    degree_equal=np.array_equal((e>0).sum(1),(before>0).sum(1))
    types_equal=all(np.array_equal((e==k).sum(1),(before==k).sum(1)) for k in range(1,int(max(e.max(),before.max()))+1))
    if not degree_equal or not types_equal: raise AssertionError('Local swap invariant violation')
    return e,{'initial':initial,'final':old,'accepted_steps':len(accepted),'tested_candidates':tested,
              'degree_preserved':degree_equal,'typed_degrees_preserved':types_equal,
              'accepted':accepted,'initial_overflow_mass':float(basis.summary(x,before)[0][-1]),
              'final_overflow_mass':float(basis.summary(x,e)[0][-1])}

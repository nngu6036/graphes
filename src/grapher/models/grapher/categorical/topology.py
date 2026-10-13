"""Spectral edge existence, independent of categorical bond probabilities.

The constrained decoder is a bounded local score optimizer, NOT a global
b-matching solver. Feasibility (indexed degrees, simplicity, and optional
connectivity) is exact even when no improving switch can be found. A separate
threshold decoder deliberately makes no degree/connectivity guarantee.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import torch

from .noise import pair_mask


def _switch(a, old, new):
    out = a.copy()
    for i, j in old:
        out[i, j] = out[j, i] = False
    for i, j in new:
        out[i, j] = out[j, i] = True
    return out


def _proposals(a, budget, rng):
    edges = np.column_stack(np.nonzero(np.triu(a, 1)))
    if len(edges) < 2:
        return
    # Independent ordered edge-pair proposals; equal/shared endpoints are
    # rejected. Both orientations are considered, without using bond labels.
    pairs = rng.integers(len(edges), size=(budget, 2))
    for p, q in pairs:
        u, v = map(int, edges[p]); w, z = map(int, edges[q])
        if len({u, v, w, z}) != 4:
            continue
        for new in (((u, w), (v, z)), ((u, z), (v, w))):
            if any(a[i, j] for i, j in new):
                continue
            yield ((u, v), (w, z)), new


def _best_switch(a, score, budget, rng, keep_connected):
    """Vectorize proposals; materialize a candidate only when its gain is positive."""
    edges=np.column_stack(np.nonzero(np.triu(a,1)))
    if len(edges)<2:
        return None,0.,0
    pairs=rng.integers(len(edges),size=(budget,2))
    first,second=edges[pairs[:,0]],edges[pairs[:,1]]
    old=np.concatenate((first,second),axis=1)
    distinct=np.all(np.diff(np.sort(old,axis=1),axis=1)!=0,axis=1)
    old=old[distinct]
    if not len(old):
        return None,0.,0
    # The two switch orientations for each old edge pair.
    new=np.stack((old[:,[0,2,1,3]],old[:,[0,3,1,2]]),axis=1).reshape(-1,4)
    old=np.repeat(old,2,axis=0)
    valid=~a[new[:,0],new[:,1]] & ~a[new[:,2],new[:,3]]
    old,new=old[valid],new[valid]
    tested=len(old)
    if not tested:
        return None,0.,0
    gains=(score[new[:,0],new[:,1]]+score[new[:,2],new[:,3]]
           -score[old[:,0],old[:,1]]-score[old[:,2],old[:,3]])
    eligible=np.flatnonzero(gains>1e-8)
    if not len(eligible):
        return None,0.,tested
    # Under connectivity constraints, try positive candidates in descending
    # gain order. Otherwise only the best candidate is materialized.
    ordering=eligible[np.argsort(-gains[eligible],kind='stable')]
    for index in ordering:
        candidate=_switch(a,old[index].reshape(2,2),new[index].reshape(2,2))
        if not keep_connected or _connected(candidate):
            return candidate,float(gains[index]),tested
    return None,0.,tested


def _connect_initial_realization(graph, rng):
    """Merge components using a non-bridge edge, with a strict progress check.

    A connected-realizable positive degree sequence has enough cycle surplus
    for these component merges. Unlike a blind random switch, each merge is
    guaranteed to reduce the number of components by one.
    """
    graph=graph.copy()
    while not nx.is_connected(graph):
        components=list(nx.connected_components(graph))
        bridges={frozenset(e) for e in nx.bridges(graph)}
        cycle_edges=[e for e in graph.edges() if frozenset(e) not in bridges]
        if not cycle_edges:
            raise ValueError('Initial sequence has no connected realization')
        u,v=cycle_edges[int(rng.integers(len(cycle_edges)))]
        outside=next(c for c in components if u not in c)
        other=list(graph.subgraph(outside).edges())
        if not other:
            raise ValueError('An isolated zero-degree node cannot be connected')
        w,z=other[int(rng.integers(len(other)))]
        graph.remove_edge(u,v);graph.remove_edge(w,z)
        graph.add_edge(u,w);graph.add_edge(v,z)
        if nx.number_connected_components(graph)!=len(components)-1:
            raise AssertionError('Initial connectivity realization did not make progress')
    return graph


def _connected(a):
    return nx.is_connected(nx.from_numpy_array(a))


def initial_support(degrees, cfg, rng):
    """An explicit feasible initial state, not an implicit degree repair.

    The same indexed degree vector is retained throughout generation. The
    initial random-switch warm-up is finite; no uniform-prior claim is made.
    """
    from grapher.models.dhvae_hh.havel_hakimi import construct_indexed_havel_hakimi

    d = np.asarray(degrees, dtype=np.int64)
    if d.ndim != 1 or not len(d) or not nx.is_graphical(d.tolist()):
        raise ValueError('A nonempty graphical degree vector is required')
    connected = bool(cfg['preserve_connectivity'])
    if connected and len(d)>1 and (np.any(d==0) or d.sum()<2*(len(d)-1)):
        raise ValueError('The sampled degree sequence has no connected realization')
    graph = construct_indexed_havel_hakimi(d.tolist(), ensure_connected=False, rng=rng)
    if connected and not nx.is_connected(graph):
        graph = _connect_initial_realization(graph, rng)
    a = nx.to_numpy_array(graph, nodelist=range(len(d)), dtype=bool)
    attempts = int(cfg['initial_random_swaps_per_edge'])*int(d.sum()//2)
    accepted = 0
    for _ in range(attempts):
        proposals = list(_proposals(a, 1, rng))
        if not proposals:
            continue
        old, new = proposals[int(rng.integers(len(proposals)))]
        candidate = _switch(a, old, new)
        if connected and not _connected(candidate):
            continue
        a = candidate; accepted += 1
    if not np.array_equal(a.sum(1), d):
        raise AssertionError('Initial spectral support changed indexed degrees')
    return a, {'constructor':'indexed_havel_hakimi_then_random_switches',
               'random_switch_attempts':attempts,'accepted_random_switches':accepted,
               'indexed_degree_exact':True,'connected':_connected(a)}


@torch.no_grad()
def decode_topology(scores, current_edges, mask, target_degrees, cfg, rng):
    """Decode the current spectral pair scores, never categorical edge logits.

    For fixed degree, maximize sum_{i<j} A_ij S_ij by improving 2-switches.
    Every accepted switch preserves A 1=d, symmetry, and a zero diagonal.
    The old topology remains feasible if the proposal budget finds no move.
    """
    active = pair_mask(mask)
    if scores.shape != active.shape or current_edges.shape != active.shape:
        raise ValueError('Topology decoder expects scores and edges [B,N,N]')
    if not bool(torch.isfinite(scores[active]).all()):
        raise FloatingPointError('Nonfinite spectral topology scores')
    scores = .5*(scores + scores.transpose(1, 2))
    before = (current_edges > 0) & active
    if cfg['decoder']=='threshold':
        out = (scores > float(cfg['threshold'])) & active
        diagnostics = [{'decoder':'threshold','degree_guarantee':False} for _ in mask]
        return out, diagnostics
    if cfg['decoder']!='degree_preserving':
        raise ValueError('Unknown spectral topology decoder')
    if target_degrees.shape != mask.shape:
        raise ValueError('target_degrees must have shape [B,N]')
    if not torch.equal(before.sum(-1), target_degrees):
        raise AssertionError('Input graph violates its fixed indexed degree sequence')
    out = before.clone(); diagnostics = []
    arrays = before.cpu().numpy(); values = scores.detach().cpu().numpy()
    for row, count in enumerate(mask.sum(1).tolist()):
        n = int(count); a = arrays[row, :n, :n].copy(); score = values[row, :n, :n]
        if cfg['preserve_connectivity'] and not _connected(a):
            raise AssertionError('Connected spectral decoder received a disconnected graph')
        accepted = tested = 0; gain_sum = 0.
        for _ in range(int(cfg['max_swaps_per_step'])):
            best,best_gain,count = _best_switch(a,score,int(cfg['proposal_budget']),rng,
                                                bool(cfg['preserve_connectivity']))
            tested += count
            if best is None:
                break
            a = best; accepted += 1; gain_sum += best_gain
        out[row, :n, :n] = torch.as_tensor(a, device=out.device)
        diagnostics.append({'decoder':'degree_preserving_spectral_score_switches',
                            'degree_guarantee':True,'accepted_swaps':accepted,
                            'tested_candidates':tested,'spectral_score_gain':gain_sum})
    if not torch.equal(out.sum(-1), target_degrees):
        raise AssertionError('Spectral decoding changed an indexed node degree')
    return out, diagnostics

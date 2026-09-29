"""Hard degree constraints for the opt-in spectral-topology/bond-only variant.

Spectral pair scores rank *valid* double-edge swaps. This is a local constrained
projection, not independent thresholding and not an exact spectral inverse.
Topology corruption during training uses finite random-swap augmentation; no
claim of an exactly solved joint topology/categorical reverse process is made.
"""
from __future__ import annotations

import math
import networkx as nx
import numpy as np
import torch

from .noise import pair_mask
from .refiner import candidates


def check_topology(adjacency, degrees=None):
    a = np.asarray(adjacency)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] < 1:
        raise ValueError('Topology must be a nonempty square matrix')
    if not np.isin(a, [0, 1]).all() or not np.array_equal(a, a.T) or np.diag(a).any():
        raise ValueError('Topology must be binary, symmetric and loop-free')
    if degrees is not None and not np.array_equal(a.sum(1), np.asarray(degrees)):
        raise AssertionError('Indexed degree constraint violated')
    return a.astype(bool, copy=True)


def random_swaps(adjacency, attempts, rng, *, preserve_connected=False):
    """Attempt a fixed number of symmetric edge switches, including rejections.

    A fixed attempt budget, rather than insisting on successful switches, also
    handles rigid sequences, stars, complete graphs and isolates without hangs.
    The resulting distribution is NOT asserted to be uniformly mixed.
    """
    a = check_topology(adjacency)
    degrees = a.sum(1)
    edges = [tuple(uv) for uv in np.argwhere(np.triu(a, 1))]
    keep = bool(preserve_connected) and nx.is_connected(nx.from_numpy_array(a))
    accepted = 0
    if len(edges) < 2 or len(a) < 4:
        return a, accepted
    for _ in range(int(attempts)):
        i, j = rng.choice(len(edges), 2, replace=False)
        u, v = edges[i]; w, z = edges[j]
        if len({u, v, w, z}) != 4:
            continue
        if rng.integers(2):
            w, z = z, w
        if a[u, w] or a[v, z]:
            continue
        a[u, v] = a[v, u] = a[w, z] = a[z, w] = False
        a[u, w] = a[w, u] = a[v, z] = a[z, v] = True
        if keep and not nx.is_connected(nx.from_numpy_array(a)):
            a[u, w] = a[w, u] = a[v, z] = a[z, v] = False
            a[u, v] = a[v, u] = a[w, z] = a[z, w] = True
            continue
        edges[i] = tuple(sorted((u, w))); edges[j] = tuple(sorted((v, z)))
        accepted += 1
    check_topology(a, degrees)
    return a, accepted


def _connect_realization(graph):
    """Merge components with degree-preserving switches, when feasible.

    A non-bridge edge in a cyclic component and an edge in another component
    can always be cross-connected without splitting either resulting component.
    """
    n = len(graph)
    if n == 1:
        return graph
    if min(dict(graph.degree()).values()) == 0 or graph.number_of_edges() < n - 1:
        raise ValueError('Sampled degree sequence admits no connected realization; '
                         'disable topology.require_connected or fix the degree prior')
    while not nx.is_connected(graph):
        components = list(nx.connected_components(graph))
        donor = donor_edge = None
        for comp in components:
            sub = graph.subgraph(comp)
            bridges = {frozenset(uv) for uv in nx.bridges(sub)}
            donor_edge = next((uv for uv in sub.edges() if frozenset(uv) not in bridges), None)
            if donor_edge is not None:
                donor = comp
                break
        if donor is None:
            raise AssertionError('Connected-realization construction lost its edge surplus')
        other = next(comp for comp in components if comp is not donor)
        u, v = donor_edge; w, z = next(iter(graph.subgraph(other).edges()))
        graph.remove_edge(u, v); graph.remove_edge(w, z)
        graph.add_edge(u, w); graph.add_edge(v, z)
    return graph


def initial_topology(degrees, cfg, rng):
    d = np.asarray(degrees)
    if d.ndim != 1 or not len(d) or not np.isfinite(d).all() or not np.equal(d, np.rint(d)).all():
        raise ValueError('Degree sequence must contain finite integers')
    d = d.astype(np.int64)
    if not nx.is_graphical(d.tolist()):
        raise ValueError('Non-graphical degree sequence; no rounding or repair is permitted')
    graph = nx.havel_hakimi_graph(d.tolist())
    if cfg['require_connected']:
        graph = _connect_realization(graph)
    a = nx.to_numpy_array(graph, nodelist=range(len(d)), dtype=np.int64) > 0
    check_topology(a, d)
    attempts = math.ceil(float(cfg['initial_swap_attempts_per_edge']) * int(d.sum() // 2))
    a, accepted = random_swaps(a, attempts, rng,
                              preserve_connected=cfg['preserve_connectivity_if_connected'])
    check_topology(a, d)
    return a, {'construction': 'havel_hakimi_exact_indexed_degrees',
               'random_swap_attempts': attempts, 'random_swaps_accepted': accepted,
               'connected': bool(nx.is_connected(nx.from_numpy_array(a)))}


def training_topology(clean_edges, mask, t, total, cfg, generator):
    """Topology augmentation must not reveal the clean adjacency to the encoder."""
    seed = int(torch.randint(0, 2**31 - 1, (), generator=generator,
                             device=clean_edges.device).item())
    rng = np.random.default_rng(seed)
    out = torch.zeros_like(clean_edges, dtype=torch.bool)
    for i, n in enumerate(mask.sum(1).tolist()):
        a = (clean_edges[i, :n, :n] > 0).detach().cpu().numpy()
        attempts = math.ceil(float(cfg['train_swap_attempts_per_edge']) * int(a.sum() // 2)
                             * float(t[i].item()) / total)
        changed, _ = random_swaps(a, attempts, rng,
                                 preserve_connected=cfg['preserve_connectivity_if_connected'])
        out[i, :n, :n] = torch.as_tensor(changed, device=out.device)
    return out & pair_mask(mask)


def spectral_topology_step(adjacency, scores, cfg, rng):
    """Improve sum_{i<j} A_ij S_ij within Omega(d) using bounded local search.

    Since the edge count is fixed, maximizing this score is equivalent to
    minimizing ||A-S||_F^2. The optimizer need not find the global minimizer;
    every accepted state, including an unchanged fallback, is degree-exact.
    """
    a = check_topology(adjacency)
    d = a.sum(1)
    score = np.asarray(scores, dtype=np.float64)
    if score.shape != a.shape or not np.isfinite(score).all():
        raise ValueError('Spectral pair scores must be finite and match the topology')
    score = (score + score.T) / 2
    np.fill_diagonal(score, 0.)
    value = lambda graph: float(np.sum(np.triu(graph, 1) * score))
    initial = current = value(a)
    keep = cfg['preserve_connectivity_if_connected'] and nx.is_connected(nx.from_numpy_array(a))
    accepted = tested = 0
    for _ in range(int(cfg['max_steps_per_event'])):
        best = None; best_value = current
        for candidate in candidates(a.astype(np.int64), cfg, rng):
            tested += 1
            candidate_value = value(candidate > 0)
            if candidate_value <= best_value + float(cfg['min_improvement']):
                continue
            if keep and not nx.is_connected(nx.from_numpy_array(candidate > 0)):
                continue
            best, best_value = candidate > 0, candidate_value
        if best is None:
            break
        a, current = best, best_value
        accepted += 1
    check_topology(a, d)
    return a, {'accepted_steps': accepted, 'tested_candidates': tested,
               'initial_pair_score': initial, 'final_pair_score': current,
               'degree_preserved': True, 'solver': 'local_degree_constrained_spectral_score_swaps'}

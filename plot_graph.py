import networkx as nx
import numpy as np
import random
import os
import argparse
import toml
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import math
import random
from collections import deque
import networkx as nx

from utils import load_graph_from_directory, constraint_configuration_model_from_multiset, havel_hakimi_construction
from create_dataset import generate_community_graph, generate_grid_graph

def plot_graph_evolution(snapshots):
    """
    Plot a sequence of graph snapshots in one horizontal figure.

    Args:
        snapshots (list[tuple[nx.Graph, int]]): list of (graph, step) pairs, for ONE graph.
        idx (int): 0-based index of this graph (used in filename).
        out_dir (str): output folder.
    """

    if not snapshots:
        return

    # Use a FIXED layout across all panels for comparability.
    # Compute on the first snapshot and reuse positions.
    G0, _ = snapshots[0]
    # Seeded layout for reproducibility; tweak seed if you like.
    pos = nx.spring_layout(G0, seed=42)

    fig, axes = plt.subplots(1, len(snapshots), figsize=(4 * len(snapshots), 4))
    if len(snapshots) == 1:
        axes = [axes]

    for ax, (G, label) in zip(axes, snapshots):
        # Draw with fixed positions; nodes that don't exist will be ignored (same N here).
        nx.draw(G, pos=pos, node_size=40, with_labels=False, ax=ax)
        ax.set_title(label)
        ax.axis("off")

    plt.show()

def hh_graph_from_G(G):
    """
    Build a canonical Havel–Hakimi realization that uses the same node labels as G.
    Ties are broken by (higher degree first, then smaller node id).
    """
    deg_pairs = sorted(((d, u) for u, d in G.degree()), key=lambda x: (-x[0], x[1]))
    seq = [d for d, _ in deg_pairs]
    # Build HH graph on 0..n-1 then relabel back to original nodes in this order
    H_int = nx.havel_hakimi_graph(seq)
    mapping = {i: deg_pairs[i][1] for i in range(len(seq))}
    H = nx.relabel_nodes(H_int, mapping, copy=True)
    return H

def _ek(u, v):
    return (u, v) if u <= v else (v, u)

def _pick_valid_swap(G, max_tries=128, rng=random):
    """
    Propose a valid 2-edge swap (e1, e2) -> (f1, f2).
    Returns (e1, e2, f1, f2) or None if not found.
    """
    E = list(G.edges())
    nE = len(E)
    if nE < 2:
        return None
    for _ in range(max_tries):
        (a, b) = E[rng.randrange(nE)]
        (c, d) = E[rng.randrange(nE)]
        if len({a, b, c, d}) != 4:  # endpoints must be all different
            continue
        # two possible rewires; pick one at random
        if rng.random() < 0.5:
            f1, f2 = (a, c), (b, d)
        else:
            f1, f2 = (a, d), (b, c)
        # no self-loops and no multi-edges
        if f1[0] == f1[1] or f2[0] == f2[1]:
            continue
        if G.has_edge(*f1) or G.has_edge(*f2):
            continue
        # Also avoid creating parallel edges across the pair
        if _ek(*f1) == _ek(*f2):
            continue
        return ( (a,b), (c,d), f1, f2 )
    return None

def _khop_neighborhoods(G, k):
    """
    Precompute closed k-hop neighborhoods (excluding the center itself).
    """
    N = {}
    for u in G.nodes():
        dists = nx.single_source_shortest_path_length(G, u, cutoff=k)
        N[u] = {v for v, dist in dists.items() if 0 < dist <= k}
    return N

def _within_k(u, v, k, neighborhoods, G_current, locality_reference, cache_dynamic):
    """
    Check if dist(u, v) <= k according to chosen reference.
    """
    if k is None:
        return True
    if locality_reference == "initial":
        return v in neighborhoods[u]
    # dynamic: compute on-demand BFS (cached per (anchor, k))
    key = (u, k)
    if key not in cache_dynamic:
        dists = nx.single_source_shortest_path_length(G_current, u, cutoff=k)
        cache_dynamic[key] = {x for x, dist in dists.items() if 0 < dist <= k}
    return v in cache_dynamic[key]

def _propose_swap_with_locality(
    G, rng, k, neighborhoods, locality_reference, max_tries=256
):
    """
    Propose a valid 2-edge swap (e1,e2)->(f1,f2) that respects k-hop locality.
    Returns (e1, e2, f1, f2) or None.
    """
    E = list(G.edges())
    m = len(E)
    if m < 2: return None
    dyn_cache = {}  # for dynamic k-hop lookups

    for _ in range(max_tries):
        (a, b) = E[rng.randrange(m)]
        (c, d) = E[rng.randrange(m)]
        if len({a, b, c, d}) != 4:
            continue

        # Two orientations; try the one sampled first, fall back to the other
        for (f1, f2) in ( ((a, c), (b, d)), ((a, d), (b, c)) if rng.random() < 0.5 else ((a, d), (b, c),), ):
            # simple-edge constraints
            if f1[0] == f1[1] or f2[0] == f2[1]:
                continue
            if G.has_edge(*f1) or G.has_edge(*f2):
                continue
            if _ek(*f1) == _ek(*f2):
                continue
            # k-hop locality constraints
            if not _within_k(f1[0], f1[1], k, neighborhoods, G, locality_reference, dyn_cache):
                continue
            if not _within_k(f2[0], f2[1], k, neighborhoods, G, locality_reference, dyn_cache):
                continue
            return ( (a,b), (c,d), f1, f2 )

    return None

# ---------- main routine ----------

def transform_to_hh_via_stochastic_rewiring(
    G,
    max_steps=10000,
    beta=2.5,           # bias toward HH edges
    T0=1.0,             # initial temperature
    cooling=0.995,      # simulated annealing cooling per accepted step
    ensure_connected=True,
    k_hop=None,         # e.g., 2 or 3 to preserve locality; None disables
    locality_reference="initial",  # "initial" (default) or "current"
    seed=None,
    return_trajectory=False,
):
    """
    Stochastically transform G to its Havel–Hakimi realization using biased 2-edge swaps
    while (a) preserving degree sequence, (b) enforcing k-hop locality for *new* edges,
    and (c) rejecting swaps that break connectivity (if ensure_connected=True).
    """
    assert locality_reference in ("initial", "current")
    rng = random.Random(seed)
    Gc = G.copy()

    # Target HH graph and scoring
    H = hh_graph_from_G(Gc)
    H_set = {_ek(u, v) for u, v in H.edges()}
    def matches_in_H(edges): return sum(1 for e in edges if _ek(*e) in H_set)
    cur_matches = matches_in_H(Gc.edges())

    # Precompute k-hop neighborhoods on the chosen reference graph
    neighborhoods = None
    if k_hop is not None:
        ref_graph = G if locality_reference == "initial" else Gc
        neighborhoods = _khop_neighborhoods(ref_graph, k_hop)

    T = T0
    traj = [Gc.copy()] if return_trajectory else None
    m = Gc.number_of_edges()

    for _ in range(max_steps):
        prop = _propose_swap_with_locality(
            Gc, rng, k_hop, neighborhoods, locality_reference, max_tries=256
        )
        if prop is None:
            # no valid locality-respecting swap found under the budget
            break

        (e1, e2, f1, f2) = prop
        before = int(_ek(*e1) in H_set) + int(_ek(*e2) in H_set)
        after  = int(_ek(*f1) in H_set) + int(_ek(*f2) in H_set)
        dmatches = after - before

        # Metropolis acceptance (symmetric proposals)
        accept = (dmatches >= 0) or (rng.random() < math.exp(beta * dmatches / max(T, 1e-8)))
        if not accept:
            continue

        # Tentatively apply and enforce connectivity
        Gc.remove_edges_from([e1, e2])
        Gc.add_edges_from([f1, f2])

        if ensure_connected and not nx.is_connected(Gc):
            # revert if it breaks connectivity
            Gc.remove_edges_from([f1, f2])
            Gc.add_edges_from([e1, e2])
            continue

        # Commit
        cur_matches += dmatches
        T *= cooling
        if return_trajectory:
            traj.append(Gc.copy())

        if cur_matches == m:  # reached HH exactly
            break

        # If using dynamic locality, refresh neighborhoods occasionally (cheap heuristic)
        if k_hop is not None and locality_reference == "current":
            # Only recompute for touched nodes to keep it light
            for u in {e1[0], e1[1], e2[0], e2[1], f1[0], f1[1], f2[0], f2[1]}:
                dists = nx.single_source_shortest_path_length(Gc, u, cutoff=k_hop)
                neighborhoods[u] = {x for x, dist in dists.items() if 0 < dist <= k_hop}
    print(len(traj))
    return (Gc, H, traj) if return_trajectory else (Gc, H)



# Example Usage:
def main():
    G = generate_community_graph(20,30)
    print("Edge count ",10*G.number_of_edges())
    try:
        while True:
            print("Looping... Press Ctrl+C to exit.")
            G_to_HH, H, _ = transform_to_hh_via_stochastic_rewiring(
                G,
                max_steps=20000,
                beta=3.0,
                T0=1.0,
                cooling=0.997,
                ensure_connected=True,
                k_hop=None,
                return_trajectory = True,
                locality_reference="initial",
                seed=42,
            )
            plot_graph_evolution([(G,"G"),(G_to_HH,"G_to_HH"),(H,"H")])
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt detected. Exiting loop.")
    finally:
        print("Program finished.")

    
if __name__ == "__main__":
    main()

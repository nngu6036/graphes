import torch
import os
import networkx as nx
from torch_geometric.utils.convert import from_networkx
import numpy as np
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix
import random

def load_degree_sequence_from_directory(directory_path):
    max_node = 0 
    max_edge = 0
    seqs = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            G = nx.read_edgelist(file_path, nodetype=int)
            max_node = max(max_node, G.number_of_nodes())
            max_edge = max(max_node, G.number_of_edges())
    print("Max node: ", max_node, " Max edge:", max_edge)
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            G = nx.read_edgelist(file_path, nodetype=int)
            G = nx.convert_node_labels_to_integers(G)
            seq = [deg for _, deg in G.degree()]
            if seq is not None:
                seqs.append(seq)
    return seqs, max_node

def load_graph_from_directory(directory_path):
    max_node = 0 
    graphs = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            graph = nx.read_edgelist(file_path, nodetype=int)
            max_node = max(max_node, graph.number_of_nodes())
    print("Max node: ", max_node)
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            graph = nx.read_edgelist(file_path, nodetype=int)
            graph = nx.convert_node_labels_to_integers(graph)
            graphs.append(graph)
    return graphs, max_node


def _safe_eigvecs(G: nx.Graph, k: int) -> np.ndarray:
    """
    Return up to `k` Laplacian eigenvectors for G, coping with tiny graphs.

    For |V| ≤ 2  (or any failure in ARPACK), we return a constant vector so that
    every node gets identical features.  This keeps dimensions consistent while
    still giving the GNN *something* to work with.
    """
    n = G.number_of_nodes()
    if n == 0:
        return np.empty((0, k))

    # For very small graphs we skip ARPACK entirely.
    if n <= 2:
        return np.ones((n, 1))

    A = csr_matrix(nx.to_scipy_sparse_array(G, dtype=float))
    L = csgraph.laplacian(A, normed=True)

    # eigsh requires k < n.  Use n-2 so that the zero eigenpair is avoided.
    k_eff = min(k, n - 2)
    if k_eff < 1:
        k_eff = 1

    try:
        _, eigvecs = eigsh(L, k=k_eff, which="SM")
        return eigvecs  # shape: (n , k_eff)
    except Exception as e:
        print(f"[utils] Laplacian eigendecomposition failed on |V|={n}: {e}")
        return np.ones((n, 1))


def graph_to_data(G: nx.Graph, k_eigen: int):
    """
    Convert a NetworkX graph to a PyG Data object with Laplacian‐eigen features.

    The returned Data.x has fixed width `k_eigen` for *all* graphs, so the
    downstream GIN layers in GraphER always receive the expected dimension.
    """
    eigvecs = _safe_eigvecs(G, k_eigen)           # (n , ≤k_eigen)
    n, cur_k = eigvecs.shape

    # Pad or truncate to exactly k_eigen columns.
    if cur_k < k_eigen:
        pad = np.zeros((n, k_eigen - cur_k), dtype=eigvecs.dtype)
        eigvecs = np.concatenate([eigvecs, pad], axis=1)
    elif cur_k > k_eigen:
        eigvecs = eigvecs[:, :k_eigen]

    # Attach features to nodes.
    for i, node in enumerate(G.nodes()):
        G.nodes[node]["x"] = eigvecs[i].astype(np.float32)

    return from_networkx(G)


def check_sequence_validity(seq):
    """Checks if a degree sequence is valid after removing all zeros."""
    if len(seq) == 0:
        return False,1
    # Degree sequence sum must be even
    if sum(seq) % 2 != 0:
        return False,2
    # Sort in descending order
    sorted_seq = sorted(seq, reverse=True)
    # Apply Erdős–Gallai theorem
    for k in range(1, len(sorted_seq) + 1):
        lhs = sum(sorted_seq[:k])
        rhs = k * (k - 1) + sum(min(d, k) for d in sorted_seq[k:])
        if lhs > rhs:
            return False,3
    return True, 0


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
        options = [((a, c), (b, d)), ((a, d), (b, c))]
        if rng.random() < 0.5:
            options.reverse()
        for (f1, f2) in options:
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

def _ek(u, v):
    return (u, v) if u <= v else (v, u)

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


def transform_to_hh_via_stochastic_rewiring(
    G,
    H,
    max_steps,
    ensure_connected=True,
    k_hop=2,         # e.g., 2 or 3 to preserve locality; None disables
    locality_reference="initial",  # "initial" (default) or "current"
):
    """
    Stochastically transform G to its Havel–Hakimi realization using biased 2-edge swaps
    while (a) preserving degree sequence, (b) enforcing k-hop locality for *new* edges,
    and (c) rejecting swaps that break connectivity (if ensure_connected=True).
    """
    rng = random.Random()
    Gc = G.copy()
    # Target HH graph and scoring
    H_set = {_ek(u, v) for u, v in H.edges()}
    def matches_in_H(edges): return sum(1 for e in edges if _ek(*e) in H_set)
    cur_matches = matches_in_H(Gc.edges())
    # Precompute k-hop neighborhoods on the chosen reference graph
    neighborhoods = None
    if k_hop is not None:
        ref_graph = G if locality_reference == "initial" else Gc
        neighborhoods = _khop_neighborhoods(ref_graph, k_hop)

    traj = []
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
        # Tentatively apply and enforce connectivity
        Gb = Gc.copy()  
        Gc.remove_edges_from([e1, e2])
        Gc.add_edges_from([f1, f2])
        if ensure_connected and not nx.is_connected(Gc):
            # revert if it breaks connectivity
            Gc.remove_edges_from([f1, f2])
            Gc.add_edges_from([e1, e2])
            continue
        else:
            traj.append((Gb,(e1, e2), (f1, f2)))

        # Commit
        cur_matches += dmatches
        if cur_matches == m:  # reached HH exactly
            break
        # If using dynamic locality, refresh neighborhoods occasionally (cheap heuristic)
        if k_hop is not None and locality_reference == "current":
            # Only recompute for touched nodes to keep it light
            for u in {e1[0], e1[1], e2[0], e2[1], f1[0], f1[1], f2[0], f2[1]}:
                dists = nx.single_source_shortest_path_length(Gc, u, cutoff=k_hop)
                neighborhoods[u] = {x for x, dist in dists.items() if 0 < dist <= k_hop}
    return traj


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

import torch
import os
import networkx as nx
from torch_geometric.utils.convert import from_networkx
import numpy as np
from typing import Dict
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix
import random
import math
from typing import Optional, Iterable
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from networkx.algorithms import community as nx_comm  # near the other nx imports
from torch_geometric.datasets import ZINC, QM9


def load_degree_sequence_from_directory(directory_path):
    max_node = 0
    max_edge = 0
    max_degree = 0
    seqs = []

    # First pass: compute statistics
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            G = nx.read_edgelist(file_path, nodetype=int)
            G = nx.convert_node_labels_to_integers(G)

            max_node = max(max_node, G.number_of_nodes())
            max_edge = max(max_edge, G.number_of_edges())

            if G.number_of_nodes() > 0:
                local_max_deg = max(dict(G.degree()).values())
                max_degree = max(max_degree, local_max_deg)
    print(
        "Max node:", max_node,
        "Max edge:", max_edge,
        "Max degree:", max_degree
    )
    # Second pass: collect degree sequences
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            G = nx.read_edgelist(file_path, nodetype=int)
            G = nx.convert_node_labels_to_integers(G)
            seq = [deg for _, deg in G.degree()]
            if seq:
                seqs.append(seq)
    # return max_degree instead of max_node
    return seqs, max_node, max_degree


def load_pyg_graph_from_directory(pyg_name, directory_path):
    if pyg_name == 'QM9':
        dataset = QM9(root=directory_path)
    elif pyg_name == 'ZINC':
        dataset = ZINC(root=directory_path, subset=True)
    else:
        raise ValueError("Invalid PYG dataset name")

    graphs = []
    max_node = max_edge = max_degree = 0

    for data in dataset:
        G = to_networkx(
            data,
            to_undirected=True,
            node_attrs=['x'],        # keep if you want later
            edge_attrs=['edge_attr'] # keep if you want later
        )
        G = nx.convert_node_labels_to_integers(G, ordering="sorted")
        graphs.append(G)

        max_node = max(max_node, G.number_of_nodes())
        max_edge = max(max_edge, G.number_of_edges())
        if G.number_of_nodes() > 0:
            max_degree = max(max_degree, max(dict(G.degree()).values()))

    print("Max node:", max_node, "Max edge:", max_edge, "Max degree:", max_degree)
    return graphs, max_node, max_degree


def load_graph_from_directory(directory_path):
    max_node = 0
    max_degree = 0
    graphs = []
    # First pass: compute statistics
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            G = nx.read_edgelist(file_path, nodetype=int)
            G = nx.convert_node_labels_to_integers(G)

            max_node = max(max_node, G.number_of_nodes())

            if G.number_of_nodes() > 0:
                local_max_deg = max(dict(G.degree()).values())
                max_degree = max(max_degree, local_max_deg)
    print(
        "Max node:", max_node,
        "Max degree:", max_degree
    )
    # Second pass: load graphs
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            G = nx.read_edgelist(file_path, nodetype=int)
            G = nx.convert_node_labels_to_integers(G)
            graphs.append(G)
    # return max_degree instead of max_node
    return graphs, max_node, max_degree


def load_pyg_graph_from_directory(pyg_name, directory_path):
    max_node = 0
    max_edge = 0
    max_degree = 0
    seqs = []
    if pyg_name =='QM9':
        dataset = QM9(root=directory_path)
    elif pyg_name =='ZINC':
        dataset = ZINC(root=directory_path, subset=True)
    else:
        raise Error("Invalid PYG dataset name")
    graphs = []
    for data in dataset:
        G = to_networkx(
            data,
            to_undirected=True,
            node_attrs=['x'],
            edge_attrs=['edge_attr'],
        )
        graphs.append(G)
    # First pass: compute statistics
    for G in graphs:
        max_node = max(max_node, G.number_of_nodes())
        max_edge = max(max_edge, G.number_of_edges())
        if G.number_of_nodes() > 0:
            local_max_deg = max(dict(G.degree()).values())
            max_degree = max(max_degree, local_max_deg)
    print(
        "Max node:", max_node,
        "Max edge:", max_edge,
        "Max degree:", max_degree
    )
    # Second pass: load graphs
    for G in graphs:
        G = nx.read_edgelist(file_path, nodetype=int)
        G = nx.convert_node_labels_to_integers(G)
        graphs.append(G)
    # return max_degree instead of max_node
    return graphs, max_node, max_degree


def nx_to_undirected_edge_index(G):
    edges = list(G.edges())
    if len(edges) == 0:
        return torch.empty((2, 0), dtype=torch.long)

    ei = torch.tensor(edges, dtype=torch.long).t().contiguous()  # [2, E]
    ei_rev = ei.flip(0)                                         # [2, E]
    ei = torch.cat([ei, ei_rev], dim=1)                         # [2, 2E]
    return ei


def graph_to_data(G, k_gen, x = None):
    """
    Convert nx.Graph into PyG Data for GraphER.
    Includes:
        - node features
        - edge_index
        - valid disjoint edge pairs
        - optional positional encodings
    """
    # --------------------------------------
    # Node features (degree only for now)
    # --------------------------------------
    if x is None:
        deg = np.array([d for _, d in G.degree()], dtype=np.float32)
        x = torch.tensor(deg).view(-1, 1)
    # --------------------------------------
    # Edge index
    # --------------------------------------
    edges = list(G.edges())
    valid_pairs = []
    m = len(edges)
    for i in range(m):
        u, v = edges[i]
        for j in range(i + 1, m):
            x2, y2 = edges[j]
            if len({u, v, x2, y2}) == 4:
                valid_pairs.append([i, j])
    if valid_pairs:
        edge_pairs = torch.tensor(valid_pairs, dtype=torch.long)
    else:
        edge_pairs = torch.empty((0, 2), dtype=torch.long)
    # --------------------------------------
    # Build PyG Data
    # --------------------------------------
    edge_index  = nx_to_undirected_edge_index(G)
    data = Data(
        x=x,                                   # node features
        edge_index=edge_index,                         # edges
        edge_pairs=edge_pairs,                 # candidate swap pairs
        num_nodes=G.number_of_nodes(),
        num_edges=G.number_of_edges(),
    )
    # --------------------------------------
    # Laplacian positional encoding
    # --------------------------------------
    A = nx.to_numpy_array(G, dtype=float)
    deg = A.sum(axis=1)
    deg_sqrt_inv = np.zeros_like(deg)
    deg_sqrt_inv[deg > 0] = 1.0 / np.sqrt(deg[deg > 0])
    # Compute D^{-1/2} * A * D^{-1/2}
    D_inv_sqrt = np.diag(deg_sqrt_inv)
    I = np.eye(G.number_of_nodes())
    L_dense = I - D_inv_sqrt @ A @ D_inv_sqrt
    eigvals, eigvecs = np.linalg.eigh(L_dense)
    k = min(k_gen, eigvecs.shape[1])
    pe = torch.tensor(eigvecs[:, :k], dtype=torch.float)
    data.x = torch.cat([data.x, pe], dim=-1)
    return data


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

def _simple_valid(graph: nx.Graph, a: int, b: int, c: int, d: int) -> bool:
    # All endpoints must be distinct; no loops; no parallel edges.
    if len({a, b, c, d}) < 4:
        return False
    if a == b or c == d:
        return False
    # We will check duplicates after removing old edges in the temp graph.
    return True


def rewire(graph: nx.Graph, e1, e2, orientation: int, ensure_connected):
    """
    orientation=0 -> add (u,x) and (v,y)
    orientation=1 -> add (u,y) and (v,x)
    Returns (new_graph, ((a,b),(c,d))) if valid; otherwise None.
    """
    (u, v), (x, y) = e1, e2
    if len({u, v, x, y}) < 4:
        return None
    if orientation == 0:
        a, b, c, d = u, x, v, y
    else:
        a, b, c, d = u, y, v, x
    if not _simple_valid(graph, a, b, c, d):
        return None
    if not graph.has_edge(u, v) or not graph.has_edge(x, y):
        return None
    G_post = graph.copy()
    G_post.remove_edge(u, v)
    G_post.remove_edge(x, y)
    remove_edges = (u, v),(x, y)
    # Disallow duplicates after removal
    if G_post.has_edge(a, b) or G_post.has_edge(c, d):
        return None
    G_post.add_edge(a, b)
    G_post.add_edge(c, d)
    add_edges = (a, b), (c, d)
    if ensure_connected and not nx.is_connected(G_post):
        return None
    return G_post, add_edges, remove_edges


def _k_hop_ok_cached(e1, e2, dist_u, dist_v):
    """
    Fast k-hop check using cached BFS maps from endpoints of e1 = (u,v).
    dist_u: nodes within k hops of u
    dist_v: nodes within k hops of v
    """
    (x, y) = e2
    return (x in dist_u) or (y in dist_u) or (x in dist_v) or (y in dist_v)


def transform_to_hh_via_guided_rewiring(
    G: nx.Graph,
    H: nx.Graph,
    lambda_dist,
    max_steps: int,
    ensure_connected: bool = True,
    k_hop: Optional[int] = 2,
    max_e2_candidates: Optional[int] = None,
    energy_fn=None,
    energy_weight: float = 0.0,
):
    """
    Guided edge-rewiring trajectory from G toward H, implemented as a generator.

    This function no longer materializes the whole trajectory in memory.
    Instead, it yields each step as soon as it is found:

        yields: (G_t, added_edges, removed_edges, score_t)

    where:
        - G_t          : graph AFTER performing the rewiring at this step
        - added_edges  : ((a,b), (c,d)) edges added in this step
        - removed_edges: ((u,v), (x,y)) edges removed in this step
        - score_t      : combined_score(G_t) = lambda_dist(G_t) + energy term

    Example usage:
        for G_t, added, removed, score in transform_to_hh_via_guided_rewiring(
                G, H, lambda_dist, max_steps=1000,
                ensure_connected=True, k_hop=2,
                max_e2_candidates=128,
                energy_fn=community_like_energy,
                energy_weight=0.1,
        ):
            # do something with this step, or break early if desired
            pass

    Notes:
        - `H` is not used directly here, but is assumed to have been used to
          build `lambda_dist` (e.g. via `make_lambda_dist`).
        - The original behavior of stopping early when distance ~= 0 is kept.
        - This is a generator: calling the function returns an iterator object.
    """
    rng = random.Random()
    G_curr = G.copy()

    def combined_score(graph: nx.Graph) -> float:
        base = lambda_dist(graph)
        if energy_fn is not None and energy_weight != 0.0:
            return base + energy_weight * energy_fn(graph)
        return base

    best_global = combined_score(G_curr)

    for step in range(max_steps):
        edges = list(G_curr.edges())
        if len(edges) < 2:
            # not enough edges to rewire
            break

        # 1) Sample first edge uniformly
        e1 = edges[rng.randrange(len(edges))]

        # 2) Build candidate pool for second edge, respecting k-hop locality
        u, v = e1
        dist_u = nx.single_source_shortest_path_length(G_curr, u, cutoff=k_hop)
        dist_v = nx.single_source_shortest_path_length(G_curr, v, cutoff=k_hop)

        e2_pool = [
            e for e in edges
            if e != e1
            and len(set(e1 + e)) == 4
            and _k_hop_ok_cached(e1, e, dist_u, dist_v)
        ]

        # If no candidates under k-hop constraint, relax locality for this step
        if not e2_pool:
            e2_pool = [
                e
                for e in edges
                if e != e1 and len(set(e1 + e)) == 4
            ]
            if not e2_pool:
                # Truly stuck
                break

        # Optional subsample for speed
        if (max_e2_candidates is not None) and (len(e2_pool) > max_e2_candidates):
            rng.shuffle(e2_pool)
            e2_pool = e2_pool[:max_e2_candidates]

        best_step_score = math.inf
        best_step_graph = None
        best_step_added = None
        best_step_removed = None

        # 3) Greedy search over candidate second edges and orientations
        for e2 in e2_pool:
            for orient in (0, 1):
                out = rewire(G_curr, e1, e2, orient, ensure_connected)
                if out is None:
                    continue
                G_cand, added_pair, removed_pair = out
                score = combined_score(G_cand)
                if score < best_step_score:
                    best_step_score = score
                    best_step_graph = G_cand
                    best_step_added = added_pair
                    best_step_removed = removed_pair

        # 4) Decide move: greedy non-worsening if possible, else random valid move
        if best_step_graph is None:
            continue
        if best_step_score <= best_global:
            # Greedy move (non-worsening)
            G_curr = best_step_graph
            best_global = best_step_score
            # Yield this step
            yield G_curr, best_step_added, best_step_removed, best_global
        else:
            G_curr = best_step_graph
            best_global = best_step_score
            yield G_curr, best_step_added, best_step_removed, best_global


def havel_hakimi_construction(G: nx.Graph) -> nx.Graph:
    """
    Build a canonical Havel–Hakimi realization that uses the same node labels as G.
    Ties are broken deterministically by (higher degree first, then smaller node id).
    """
    # (degree, node) pairs
    deg_pairs = sorted(
        ((d, u) for u, d in G.degree()),
        key=lambda x: (-x[0], x[1])  # sort by degree desc, node id asc
    )
    seq = [d for d, _ in deg_pairs]

    # HH on integer-labeled nodes 0..n-1
    H_int = nx.havel_hakimi_graph(seq)

    # Map back to original node labels according to deg_pairs order
    mapping = {i: deg_pairs[i][1] for i in range(len(seq))}
    H = nx.relabel_nodes(H_int, mapping, copy=True)
    return H

def _merge_two_components_deterministically(H: nx.Graph, comp1_nodes, comp2_nodes):
    """
    Deterministically merge two connected components via degree-preserving rewiring,
    but ONLY accept a rewiring that truly merges the two components.

    Key idea:
      - Enumerate candidate internal edges from each component (deterministic order).
      - Prefer non-bridge edges (so removing them doesn't split the component).
      - For each edge pair and orientation, simulate the swap and check that
        comp1 and comp2 become connected (i.e., component count decreases).
    """
    comp1_nodes = sorted(comp1_nodes)
    comp2_nodes = sorted(comp2_nodes)

    # induced subgraphs
    H1 = H.subgraph(comp1_nodes).copy()
    H2 = H.subgraph(comp2_nodes).copy()

    edges1_all = sorted((tuple(sorted(e)) for e in H1.edges()))
    edges2_all = sorted((tuple(sorted(e)) for e in H2.edges()))

    if not edges1_all or not edges2_all:
        raise RuntimeError(
            "Cannot merge a component with no internal edges. "
            "If your degree sequence contains 0s, a connected realization may be impossible."
        )

    # Prefer non-bridge edges (removing a bridge splits the component)
    bridges1 = set(tuple(sorted(e)) for e in nx.bridges(H1)) if H1.number_of_edges() else set()
    bridges2 = set(tuple(sorted(e)) for e in nx.bridges(H2)) if H2.number_of_edges() else set()

    edges1_pref = [e for e in edges1_all if e not in bridges1] + [e for e in edges1_all if e in bridges1]
    edges2_pref = [e for e in edges2_all if e not in bridges2] + [e for e in edges2_all if e in bridges2]

    # Representative nodes to test "merged-ness"
    r1 = comp1_nodes[0]
    r2 = comp2_nodes[0]

    base_cc = nx.number_connected_components(H)

    def try_swap(a, b, c, d, orient):
        # orient 0: (a,c),(b,d) ; orient 1: (a,d),(b,c)
        if orient == 0:
            u1, v1 = a, c
            u2, v2 = b, d
        else:
            u1, v1 = a, d
            u2, v2 = b, c

        e_new1 = (min(u1, v1), max(u1, v1))
        e_new2 = (min(u2, v2), max(u2, v2))

        # no self-loops
        if e_new1[0] == e_new1[1] or e_new2[0] == e_new2[1]:
            return None

        # simulate on a copy
        G = H.copy()
        if not G.has_edge(a, b) or not G.has_edge(c, d):
            return None

        G.remove_edge(a, b)
        G.remove_edge(c, d)

        # disallow multi-edges after removal
        if G.has_edge(*e_new1) or G.has_edge(*e_new2):
            return None

        G.add_edge(*e_new1)
        G.add_edge(*e_new2)

        # must actually merge comp1 and comp2 (strict progress)
        if nx.number_connected_components(G) >= base_cc:
            return None
        # (optional extra safety)
        if not nx.has_path(G, r1, r2):
            return None

        return (G, (a, b), (c, d), e_new1, e_new2)

    # Deterministic search: edges1, edges2, orientation
    best = None
    for (a, b) in edges1_pref:
        for (c, d) in edges2_pref:
            # small determinism tweak: try orientation 0 then 1
            out = try_swap(a, b, c, d, orient=0)
            if out is None:
                out = try_swap(a, b, c, d, orient=1)
            if out is not None:
                best = out
                break
        if best is not None:
            break

    if best is None:
        # This can happen if the degree sequence *cannot* admit a connected realization
        # or if these components are "all degree-1 matchings" and there's no higher-degree
        # component to absorb them via a non-bridge edge.
        raise RuntimeError(
            "Could not find a degree-preserving swap that reduces components. "
            "Try merging tiny components into a large cyclic component first, "
            "or verify the degree sequence admits a connected realization."
        )

    G_new, (a, b), (c, d), e_new1, e_new2 = best

    # Apply in-place
    H.remove_edge(a, b)
    H.remove_edge(c, d)
    H.add_edge(*e_new1)
    H.add_edge(*e_new2)



def constraint_configuration_model_from_multiset(degree_sequence, max_retries=None, max_failures=1000):
    N = len(degree_sequence)
    if max_retries is None:
        max_retries = N
    for _ in range(max_retries):
        stubs = []
        for node, deg in enumerate(degree_sequence):
            stubs.extend([node] * deg)
        random.shuffle(stubs)
        G = nx.Graph()
        G.add_nodes_from(range(N))
        failures = 0
        while len(stubs) >= 2 and failures < max_failures:
            u = stubs.pop()
            v = stubs.pop()
            if u == v or G.has_edge(u, v):
                # Invalid pair: put them back and count as failure
                stubs.extend([u, v])
                random.shuffle(stubs)
                failures += 1
                continue
            G.add_edge(u, v)
            failures = 0  # Reset on success
        if sorted([d for _, d in G.degree()]) == sorted(degree_sequence):
            return G
    return None  # Failed to generate a valid graph

def configuration_model_from_multiset(degrees):
    G = nx.configuration_model(degrees)
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


def deterministic_connected_havel_hakimi(G = None, seq = None) -> nx.Graph:
    """
    Deterministic Connected Havel–Hakimi (DCHH) realization for a connected graph G.

    Steps:
      1. Build the standard HH realization H with the same node labels as G
         (using havel_hakimi_construction).
      2. If H is connected, return it.
      3. Otherwise, merge connected components one-by-one using deterministic,
         degree-preserving rewiring until the graph is connected.

    This yields a unique, fully deterministic connected realization for each degree
    sequence (given the same labeled input graph G).
    """

    # Step 1: canonical HH realization
    if G is not None:
        H = havel_hakimi_construction(G)
    if seq is not None:
        H = nx.havel_hakimi_graph(seq)
    # Quick exit if already connected
    if nx.is_connected(H):
        return H

    # Step 2: deterministically merge components via rewiring
    while not nx.is_connected(H):
        # Connected components as sorted lists of nodes
        components = [sorted(c) for c in nx.connected_components(H)]
        # Sort components by (size, smallest node id) to make selection deterministic
        components.sort(key=lambda comp: (len(comp), comp[0]))

        # Choose the two smallest components to merge
        comp_small = components[0]
        comp_big   = components[-1]

        # Merge them deterministically (in-place)
        _merge_two_components_deterministically(H, comp_small, comp_big)

        # Loop continues until the whole graph is connected
    return H


def _align_nodelist(G: nx.Graph, H: nx.Graph, nodelist: Optional[Iterable]=None):
    if nodelist is not None:
        return list(nodelist)
    # Default: union, sorted for determinism (works for int/str labels)
    return sorted(set(G.nodes()) | set(H.nodes()))

def _adjacency_matrix(G: nx.Graph, nodelist) -> np.ndarray:
    idx = {u:i for i,u in enumerate(nodelist)}
    n = len(nodelist)
    A = np.zeros((n, n), dtype=float)
    # weight=1.0 if not provided; undirected simple graphs
    for u, v, d in G.edges(data=True):
        if u in idx and v in idx and u != v:
            i, j = idx[u], idx[v]
            w = float(d.get("weight", 1.0))
            A[i, j] += w
            A[j, i] += w
    return A

def _laplacian(A: np.ndarray) -> np.ndarray:
    d = A.sum(axis=1)
    return np.diag(d) - A

def _normalized_laplacian(A: np.ndarray) -> np.ndarray:
    d = A.sum(axis=1)
    with np.errstate(divide='ignore'):
        inv_sqrt_d = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    Dhalf = np.diag(inv_sqrt_d)
    I = np.eye(A.shape[0])
    return I - Dhalf @ A @ Dhalf

def _eigvals_sym(M: np.ndarray) -> np.ndarray:
    # Ascending eigenvalues for symmetric matrices
    return np.linalg.eigvalsh(M)

def _sqrt_psd(M: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh(M)
    vals = np.clip(vals, 0.0, None)
    return (vecs * np.sqrt(vals)) @ vecs.T

def _pinv_laplacian(L: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    # Moore-Penrose pseudoinverse via EVD; zero out tiny eigenvalues
    w, U = np.linalg.eigh(L)
    invw = np.zeros_like(w)
    mask = w > tol
    invw[mask] = 1.0 / w[mask]
    return (U * invw) @ U.T

def _heat_kernel_from_L(L: np.ndarray, t: float) -> np.ndarray:
    w, U = np.linalg.eigh(L)
    exps = np.exp(-t * np.clip(w, 0.0, None))
    return (U * exps) @ U.T

def _degree_array(G: nx.Graph, nodelist) -> np.ndarray:
    idx = {u:i for i,u in enumerate(nodelist)}
    d = np.zeros(len(nodelist), dtype=int)
    for u, deg in G.degree():
        if u in idx:
            d[idx[u]] = deg
    return d

def _w1_degree_sorted(d1: np.ndarray, d2: np.ndarray) -> float:
    # 1D Wasserstein-1 via L1 of CDFs on integer support
    a = np.sort(d1)
    b = np.sort(d2)
    lo = int(min(a.min() if a.size else 0, b.min() if b.size else 0))
    hi = int(max(a.max() if a.size else 0, b.max() if b.size else 0))
    if hi < lo:
        return 0.0
    xs = np.arange(lo, hi + 1)
    Ca = np.cumsum(np.histogram(a, bins=xs, range=(lo, hi))[0])
    Cb = np.cumsum(np.histogram(b, bins=xs, range=(lo, hi))[0])
    return float(np.abs(Ca - Cb).sum())

# ----------------- Individual distance builders -----------------

def _make_symmetric_edit(H: nx.Graph, nodelist):
    H_edges = {tuple(sorted(e)) for e in H.subgraph(nodelist).edges()}
    def dist(G: nx.Graph) -> float:
        G_edges = {tuple(sorted(e)) for e in G.subgraph(nodelist).edges()}
        return float(len(G_edges ^ H_edges))
    return dist

def _make_spectral(H: nx.Graph, nodelist, *, normalized=True, k: Optional[int]=None, squared=False):
    AH = _adjacency_matrix(H, nodelist)
    LH = _normalized_laplacian(AH) if normalized else _laplacian(AH)
    evH = _eigvals_sym(LH)
    if k is not None:
        evH = evH[:min(k, len(evH))]
    def dist(G: nx.Graph) -> float:
        AG = _adjacency_matrix(G, nodelist)
        LG = _normalized_laplacian(AG) if normalized else _laplacian(AG)
        evG = _eigvals_sym(LG)
        if k is not None:
            evG = evG[:min(k, len(evG))]
        diff = evG - evH
        val = float(np.dot(diff, diff))
        return val if squared else math.sqrt(val)
    return dist

def _make_bures_wasserstein(H: nx.Graph, nodelist, *, t: float=0.5, use_normalized_L: bool=True, squared=False):
    if t <= 0:
        raise ValueError("t must be > 0 for heat-kernel BW.")
    AH = _adjacency_matrix(H, nodelist)
    LH = _normalized_laplacian(AH) if use_normalized_L else _laplacian(AH)
    KH = _heat_kernel_from_L(LH, t)
    KH_sqrt = _sqrt_psd(KH)
    trKH = float(np.trace(KH))
    def dist(G: nx.Graph) -> float:
        AG = _adjacency_matrix(G, nodelist)
        LG = _normalized_laplacian(AG) if use_normalized_L else _laplacian(AG)
        KG = _heat_kernel_from_L(LG, t)
        mid = KH_sqrt @ KG @ KH_sqrt
        mid_sqrt = _sqrt_psd(mid)
        d2 = float(np.trace(KG) + trKH - 2.0 * np.trace(mid_sqrt))
        d2 = max(d2, 0.0)
        return d2 if squared else math.sqrt(d2)
    return dist

def _make_effective_resistance_fro(H: nx.Graph, nodelist, *, tol: float=1e-12):
    AH = _adjacency_matrix(H, nodelist)
    LH = _laplacian(AH)  # use combinatorial Laplacian
    LHp = _pinv_laplacian(LH, tol=tol)
    diagH = np.diag(LHp)
    RH = diagH[:,None] + diagH[None,:] - 2*LHp
    def dist(G: nx.Graph) -> float:
        AG = _adjacency_matrix(G, nodelist)
        LG = _laplacian(AG)
        LGp = _pinv_laplacian(LG, tol=tol)
        diagG = np.diag(LGp)
        RG = diagG[:,None] + diagG[None,:] - 2*LGp
        return float(np.linalg.norm(RG - RH, 'fro'))
    return dist

def _make_deltacon(H: nx.Graph, nodelist, *, eps: float=1e-3):
    # S = (I + eps^2 D - eps A)^{-1}
    AH = _adjacency_matrix(H, nodelist)
    dH = AH.sum(1)
    SH = np.linalg.inv(np.eye(len(nodelist)) + (eps**2)*np.diag(dH) - eps*AH)
    SH_sqrt = np.sqrt(np.maximum(SH, 0.0))
    def dist(G: nx.Graph) -> float:
        AG = _adjacency_matrix(G, nodelist)
        dG = AG.sum(1)
        SG = np.linalg.inv(np.eye(len(nodelist)) + (eps**2)*np.diag(dG) - eps*AG)
        SG_sqrt = np.sqrt(np.maximum(SG, 0.0))
        return float(np.linalg.norm(SG_sqrt - SH_sqrt, 'fro'))
    return dist

def _make_netlsd(H: nx.Graph, nodelist, *, T_min=1e-2, T_max=1e2, num=50, normalized=True):
    AH = _adjacency_matrix(H, nodelist)
    LH = _normalized_laplacian(AH) if normalized else _laplacian(AH)
    wH = _eigvals_sym(LH)
    ts = np.logspace(np.log10(T_min), np.log10(T_max), num)
    hH = np.array([(np.exp(-t*np.clip(wH,0,None))).sum() for t in ts])
    def dist(G: nx.Graph) -> float:
        AG = _adjacency_matrix(G, nodelist)
        LG = _normalized_laplacian(AG) if normalized else _laplacian(AG)
        wG = _eigvals_sym(LG)
        hG = np.array([(np.exp(-t*np.clip(wG,0,None))).sum() for t in ts])
        return float(np.linalg.norm(hG - hH))
    return dist

def _make_degree_emd(H: nx.Graph, nodelist):
    dH = _degree_array(H, nodelist)
    def dist(G: nx.Graph) -> float:
        dG = _degree_array(G, nodelist)
        return _w1_degree_sorted(dG, dH)
    return dist

# ----------------- Factory -----------------

def make_lambda_dist(
    name: str,
    H: nx.Graph,
    *,
    nodelist: Optional[Iterable]=None,
    **kwargs
):
    """
    Build a distance function lambda_dist(G) that compares G to target H.

    Parameters
    ----------
    name : str
        One of:
          - 'symmetric_edit'
          - 'spectral' (kwargs: normalized=True, k=None, squared=False)
          - 'bures_wasserstein' (kwargs: t=0.5, use_normalized_L=True, squared=False)
          - 'effective_resistance_fro' (kwargs: tol=1e-12)
          - 'deltacon' (kwargs: eps=1e-3)
          - 'netlsd' (kwargs: T_min=1e-2, T_max=1e2, num=50, normalized=True)
    H : nx.Graph
        Target graph.
    nodelist : Iterable, optional
        Fixed node order used for all computations. If None, uses union of G and H.

    Returns
    -------
    Callable[[nx.Graph], float]
        A function taking a graph G and returning the distance d(G, H).
    """
    # Pre-fix the nodelist for consistency and speed
    nodelist = _align_nodelist(H, H, nodelist=nodelist)

    name = name.lower()
    if name == 'symmetric_edit':
        return _make_symmetric_edit(H, nodelist)
    if name == 'spectral':
        return _make_spectral(H, nodelist, **kwargs)
    if name == 'bures_wasserstein':
        return _make_bures_wasserstein(H, nodelist, **kwargs)
    if name == 'effective_resistance_fro':
        return _make_effective_resistance_fro(H, nodelist, **kwargs)
    if name == 'deltacon':
        return _make_deltacon(H, nodelist, **kwargs)
    if name == 'netlsd':
        return _make_netlsd(H, nodelist, **kwargs)

    raise ValueError(f"Unknown distance '{name}'.")

def draw_graphs_grid(
    graphs,
    n_cols: int = 10,
    layout: str = "spring",
    figsize_per_graph=(1.5, 1.5),
    node_size=3,
    with_labels=True,
    titles=None,
    seed: int = 42,
    savepath: str = None,
    show: bool = True,
    dpi: int = 300,
    **draw_kwargs,
):
    """
    Draw a list of NetworkX graphs on one figure in a grid.

    Parameters
    ----------
    graphs : List[nx.Graph]
        Graphs to draw.
    n_cols : int
        Number of graphs per row.
    layout : str
        One of {'spring','kamada_kawai','circular','spectral','shell','random'}.
    figsize_per_graph : tuple
        (width, height) in inches per subplot.
    node_size : int
        Node size passed to nx.draw.
    with_labels : bool
        Whether to show node labels.
    titles : list[str] or None
        Optional titles per graph.
    seed : int
        Random seed for layouts that accept it.
    savepath : str or None
        If provided, save the figure to this path.
    show : bool
        Whether to display the figure with plt.show().
    dpi : int
        DPI for saving.
    **draw_kwargs :
        Forwarded to nx.draw (edge_color, node_color, width, alpha, etc.)

    Returns
    -------
    fig, axes
        Matplotlib figure and axes.
    """
    if not graphs:
        raise ValueError("`graphs` is empty.")

    layout_fn_map = {
        "spring": lambda G: nx.spring_layout(G, seed=seed),
        "kamada_kawai": lambda G: nx.kamada_kawai_layout(G),
        "circular": lambda G: nx.circular_layout(G),
        "spectral": lambda G: nx.spectral_layout(G),
        "shell": lambda G: nx.shell_layout(G),
        "random": lambda G: nx.random_layout(G, seed=seed),
    }
    if layout not in layout_fn_map:
        raise ValueError(f"Unknown layout '{layout}'.")

    n = len(graphs)
    n_rows = math.ceil(n / n_cols)
    fig_w = figsize_per_graph[0] * n_cols
    fig_h = figsize_per_graph[1] * n_rows

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))

    # Normalize axes to 2D list
    if isinstance(axes, plt.Axes):
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]
    elif n_cols == 1:
        axes = [[ax] for ax in axes]

    for idx, G in enumerate(graphs):
        r, c = divmod(idx, n_cols)
        ax = axes[r][c]
        pos = layout_fn_map[layout](G)

        title = titles[idx] if titles and idx < len(titles) else f"G{idx}"
        ax.set_title(title, fontsize=9)

        nx.draw(
            G,
            pos=pos,
            ax=ax,
            node_size=node_size,
            with_labels=with_labels,
            **draw_kwargs,
        )
        ax.set_axis_off()

    # Hide unused subplots
    for idx in range(n, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r][c].set_visible(False)
    plt.tight_layout()
    if savepath is not None:
        os.makedirs(os.path.dirname(savepath), exist_ok=True) if os.path.dirname(savepath) else None
        fig.savefig(savepath, bbox_inches="tight", dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, axes

def build_candidates(
    G,
    anchor_edge,
    ensure_connected: bool = True,
    k_hop: int = 2,
):
    """
    Optimized version:
    - Computes BFS distance maps once from each endpoint of anchor_edge (cutoff=k_hop)
    - Candidate locality check becomes O(1) per edge instead of repeated shortest_path_length calls.

    Constraints preserved:
      - disjoint endpoints
      - within k_hop of anchor endpoints (if k_hop is not None)
      - at least one orientation yields a valid rewiring (simple + optionally connected)
    """
    u, v = anchor_edge
    edges = list(G.edges())
    candidates = []

    # --- Precompute distances from u and v up to cutoff k_hop ---
    dist_u = dist_v = None
    if k_hop is not None:
        # single_source_shortest_path_length returns {node: dist} for nodes within cutoff
        dist_u = nx.single_source_shortest_path_length(G, u, cutoff=k_hop)
        dist_v = nx.single_source_shortest_path_length(G, v, cutoff=k_hop)

        def close_enough(x, y):
            # True if x or y is within k hops of u or v
            return (
                (x in dist_u) or (y in dist_u) or
                (x in dist_v) or (y in dist_v)
            )
    else:
        def close_enough(x, y):
            return True

    for e2 in edges:
        if e2 == anchor_edge:
            continue

        x, y = e2

        # Need 4 distinct endpoints
        if len({u, v, x, y}) < 4:
            continue

        # k-hop locality in O(1)
        if not close_enough(x, y):
            continue

        # Check if at least one orientation yields valid rewiring
        valid = False
        for orient in (0, 1):
            out = rewire(G, anchor_edge, e2, orient, ensure_connected)
            if out is not None:
                valid = True
                break

        if valid:
            candidates.append(e2)

    return candidates


def compute_struct_features(G: nx.Graph):
    """
    Compute simple structural features for auxiliary prediction:
    - modularity (via greedy modularity communities, if possible)
    - average clustering coefficient

    Returns: torch.tensor of shape [2] = [modularity, avg_clustering]
    """
    # Modularity: if community detection fails or trivial, use 0.0
    try:
        comms = list(nx_comm.greedy_modularity_communities(G))
        if len(comms) > 1:
            Q = nx_comm.modularity(G, comms)
        else:
            Q = 0.0
    except Exception:
        Q = 0.0

    # Average clustering: 0.0 if no edges
    if G.number_of_edges() > 0:
        C = nx.average_clustering(G)
    else:
        C = 0.0

    return torch.tensor([Q, C], dtype=torch.float32)


def community_like_energy(G: nx.Graph) -> float:
    """
    A scalar 'energy' lower for 'good' community graphs.

    We simply use negative modularity + clustering as an example:
      E(G) = -(Q(G) + C(G))

    You can tweak this if you want different behaviour.
    """
    feats = compute_struct_features(G)
    Q, C = float(feats[0]), float(feats[1])
    # Lower energy for higher modularity & clustering
    return -(Q + C)

def ego_like_energy(G: nx.Graph) -> float:
    """
    A scalar 'energy' lower for ego-like graphs.

    Heuristic:
    - Ego/star-like graphs have one dominant hub connected to most nodes.
    - Most edges are incident to that hub.
    - There should ideally be only one such hub.

    We measure:
        - center_edge_fraction = max_deg / |E|
        - hubs = number of nodes with degree >= 0.5 * max_deg

    Then define:
        score = center_edge_fraction - 0.1 * max(0, hubs - 1)
        E(G) = -score  (lower energy for more ego-like)
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()
    if n == 0 or m == 0:
        # Neutral energy if graph is empty or edgeless
        return 0.0

    degs = np.array([d for _, d in G.degree()], dtype=float)
    max_deg = float(degs.max())
    num_edges = float(m)

    # Fraction of edges incident to the main hub:
    # for a perfect star on n nodes: max_deg = n-1, |E| = n-1 => fraction = 1.0
    center_edge_fraction = max_deg / max(1.0, num_edges)

    # Count "hubs" that are at least half as connected as the main hub.
    # Ego/star-like graphs ideally have exactly one such hub.
    hubs = int((degs >= 0.5 * max_deg).sum())
    hub_penalty = max(0, hubs - 1)

    score = center_edge_fraction - 0.1 * hub_penalty
    return -float(score)  # lower energy for higher score (more ego-like)


def grid_like_energy(G: nx.Graph) -> float:
    """
    A scalar 'energy' lower for grid-like graphs.

    Heuristic properties of a 2D grid:
    - Degrees mostly in {2, 3, 4}.
    - Degree variance is small.
    - Clustering coefficient is very low (exactly 0 for an ideal grid).

    We combine:
        p_234    = fraction of nodes whose degree is in {2,3,4}
        var_term = 1 / (1 + Var(deg))
        cl_term  = 1 - C(G)    (larger when clustering is small)

    Then:
        score = p_234 + var_term + cl_term
        E(G)  = -score  (lower energy for more grid-like)
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()
    if n == 0:
        return 0.0

    degs = np.array([d for _, d in G.degree()], dtype=float)
    if len(degs) == 0:
        return 0.0

    # Fraction of degrees in {2,3,4}
    p_234 = float(np.isin(degs, [2, 3, 4]).mean())

    # Degree variance term: small variance -> term close to 1
    var_deg = float(degs.var()) if len(degs) > 1 else 0.0
    var_term = 1.0 / (1.0 + var_deg)  # in (0,1]

    # Clustering: for an ideal grid, this is 0
    if m > 0:
        C = float(nx.average_clustering(G))
    else:
        C = 0.0
    cl_term = 1.0 - max(0.0, min(1.0, C))  # keep in [0,1]

    score = p_234 + var_term + cl_term
    return -float(score)  # lower energy for more grid-like
import torch
import os
import networkx as nx
from torch_geometric.utils import from_networkx
import torch
import numpy as np
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix
import scipy.sparse as sp

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

def precompute_khop(G: nx.Graph, k: int):
    """node -> set of nodes within <=k hops (including itself)."""
    khop = {}
    for u in G.nodes():
        seen = {u}
        frontier = [u]
        for _ in range(k):
            nxt = set()
            for x in frontier:
                nxt.update(G.neighbors(x))
            nxt -= seen
            if not nxt:
                break
            seen |= nxt
            frontier = list(nxt)
        khop[u] = seen
    return khop

def are_four_endpoints_k_local(e1, e2, khop):
    """Require all 4 endpoints to be pairwise within k hops."""
    u, v = e1; x, y = e2
    S = (u, v, x, y)
    for i in range(4):
        A = khop[S[i]]
        for j in range(4):
            if S[j] not in A:
                return False
    return True

def components_increase_if_remove_two(G: nx.Graph, e1, e2):
    """Return True if removing both edges increases #components."""
    c0 = nx.number_connected_components(G)
    removed = []
    for e in (e1, e2):
        if G.has_edge(*e):
            G.remove_edge(*e); removed.append(e)
    c1 = nx.number_connected_components(G)
    for e in removed:
        G.add_edge(*e)
    return c1 > c0

def edge_is_bridge(G: nx.Graph, e):
    """Bridge test using nx.bridges (O(m+n))."""
    if not G.has_edge(*e):  # if already removed virtually, not a bridge now
        return False
    return tuple(e) in set(nx.bridges(G)) or (e[1], e[0]) in set(nx.bridges(G))

def swap_is_connectivity_safe(G: nx.Graph, e1, e2, enforce_biconnected=False):
    """Reject if removing both edges splits components; optionally forbid removing bridges."""
    if enforce_biconnected and (edge_is_bridge(G, e1) or edge_is_bridge(G, e2)):
        return False
    if components_increase_if_remove_two(G, e1, e2):
        return False
    return True

def triangles_delta_for_orientation(G: nx.Graph, remove_edges, add_edges):
    """Δtriangles = (triangles gained by add) - (lost by remove) using |N(a)∩N(b)|."""
    touched = set()
    for a,b in remove_edges+add_edges:
        touched.add(a); touched.add(b)
    N = {u: set(G.neighbors(u)) for u in touched}
    def tri_cnt(a,b):
        return len(N[a] & N[b])
    lost   = sum(tri_cnt(a,b) for (a,b) in remove_edges if G.has_edge(a,b))
    gained = sum(tri_cnt(a,b) for (a,b) in add_edges if not G.has_edge(a,b))
    return gained - lost

def triangle_aware_score(G: nx.Graph, e1, e2, add_edges, clustering=None, alpha=0.5):
    """Score orientation by Δtriangles, optionally up-weighting high-clustering endpoints."""
    rem = [tuple(sorted(e)) for e in (e1, e2)]
    add = [tuple(sorted(e)) for e in add_edges]
    base = triangles_delta_for_orientation(G, rem, add)
    if clustering is None:
        return base
    W = sum(clustering[n] for n in {e1[0], e1[1], e2[0], e2[1]})
    return base * (1.0 + alpha * W)

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


def havel_hakimi_construction(G):
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

def _laplacian_eigs_no_pad(G, k=8):
    """
    Compute up to k smallest nontrivial Laplacian eigenpairs (drop all zero modes).
    Works with scipy.sparse csr_array/csr_matrix or falls back to dense.
    """
    import numpy as np
    import networkx as nx

    n = G.number_of_nodes()
    if n == 0:
        return np.array([], dtype=float), np.zeros((0, 0), dtype=float)

    # Build Laplacian robustly
    L_raw = nx.laplacian_matrix(G)  # can be csr_array, csr_matrix, or ndarray depending on versions

    # Try sparse path first
    try:
        import scipy.sparse as sp
        from scipy.sparse.linalg import eigsh as _eigsh

        if sp.issparse(L_raw):
            # Ensure float dtype and CSR **matrix** (not array)
            L = L_raw.astype(np.float64)
            if not isinstance(L, sp.csr_matrix):
                L = sp.csr_matrix(L)
        else:
            # Dense path
            L = np.asarray(L_raw, dtype=np.float64)

        # Choose solver path
        use_dense = (not sp.issparse(L)) or (n <= 2)
        if use_dense:
            Ld = L.toarray() if hasattr(L, "toarray") else np.asarray(L)
            evals, evecs = np.linalg.eigh(Ld)
        else:
            # eigsh requires 1 <= k < n
            kk = min(max(1, k + 1), n - 1)
            evals, evecs = _eigsh(L, k=kk, which="SM")
    except Exception:
        # No SciPy available or something went wrong: fall back to dense
        Ld = L_raw.toarray() if hasattr(L_raw, "toarray") else np.asarray(L_raw)
        Ld = Ld.astype(np.float64, copy=False)
        evals, evecs = np.linalg.eigh(Ld)

    # Sort ascending
    idx = np.argsort(evals)
    evals = evals[idx]
    evecs = evecs[:, idx]

    # Drop all ~0 eigenvalues (multiplicity = #components)
    zero_mask = np.abs(evals) < 1e-12
    evals = evals[~zero_mask]
    evecs = evecs[:, ~zero_mask]

    # Keep up to k
    k_eff = min(k, evals.shape[0])
    return evals[:k_eff], evecs[:, :k_eff]


def _laplacian_eigs_fixed(G, k=8):
    """
    Same as _laplacian_eigs_no_pad but pads to exactly k columns with zeros
    (and pads evals with large values so heat kernels go ~0 on padded cols).
    """
    import numpy as np

    evals, evecs = _laplacian_eigs_no_pad(G, k=k)
    n = G.number_of_nodes()

    if evecs.shape[1] < k:
        pad_cols = k - evecs.shape[1]
        if n == 0:
            evecs = np.zeros((0, k), dtype=float)
            evals = np.full((k,), 1e6, dtype=float)
        else:
            evecs = np.concatenate([evecs, np.zeros((n, pad_cols), dtype=float)], axis=1)
            if evals.size == 0:
                evals = np.full((k,), 1e6, dtype=float)
            else:
                evals = np.concatenate([evals, np.full((pad_cols,), 1e6, dtype=float)], axis=0)

    # Ensure exact shapes
    evals = evals[:k]
    evecs = evecs[:, :k]
    return evals, evecs



def build_node_features(
    G: nx.Graph,
    k_lap: int = 8,
    heat_times = (0.1, 1.0),
    use_clustering: bool = False,
    use_avg_nbr_deg: bool = False,
    use_core: bool = False
):
    """
    Returns np.ndarray of shape [n_nodes, feat_dim] with:
      - degree, degree_normalized
      - Laplacian PE: k_lap eigenvectors (columns)
      - Heat-scaled LapPE: concat_{t in heat_times} (evecs * exp(-t*evals))  [columnwise scaling]
      - Local: clustering coefficient, avg-neighbor-degree, core number
    """
    n = G.number_of_nodes()
    nodes = list(range(n))  # assume 0..n-1 labels; if not, relabel upstream

    # degree features
    deg = np.array([G.degree(i) for i in nodes], dtype=float).reshape(n, 1)
    deg_norm = deg / max(1, n - 1)

    # Laplacian global features
    evals, evecs = _laplacian_eigs_fixed(G, k=k_lap)  # evecs: [n, k_lap_eff]
    if evecs.size == 0:
        lap_pe = np.zeros((n, 0), dtype=float)
        heat_blocks = []
    else:
        lap_pe = evecs  # [n, k]
        # Heat-scaled copies: multiply each eigenvector column j by exp(-t * evals[j])
        heat_blocks = []
        for t in heat_times:
            scales = np.exp(-t * evals)  # [k]
            heat_blocks.append(lap_pe * scales.reshape(1, -1))
    heat_feats = np.concatenate(heat_blocks, axis=1) if len(heat_blocks) else np.zeros((n, 0), dtype=float)

    # Local properties
    locals_list = []

    if use_clustering:
        # returns dict {node: coeff}
        cc = nx.clustering(G)
        locals_list.append(np.array([cc[i] for i in nodes], dtype=float).reshape(n, 1))

    if use_avg_nbr_deg:
        andeg = nx.average_neighbor_degree(G)
        locals_list.append(np.array([andeg[i] for i in nodes], dtype=float).reshape(n, 1))

    if use_core:
        core = nx.core_number(G)
        # normalize by max core to keep scale stable
        core_vals = np.array([core[i] for i in nodes], dtype=float)
        core_norm = (core_vals / max(1.0, core_vals.max())).reshape(n, 1)
        locals_list.append(core_norm)

    local_feats = np.concatenate(locals_list, axis=1) if len(locals_list) else np.zeros((n, 0), dtype=float)

    X = np.concatenate([deg, deg_norm, lap_pe, heat_feats, local_feats], axis=1).astype(np.float32)
    return X


def to_pyg_with_features(G: nx.Graph, **feature_kwargs):
    """Build a PyG Data with x from build_node_features."""
    data = from_networkx(G)
    X = build_node_features(G, **feature_kwargs)  # np array
    data.x = torch.from_numpy(X)
    return data
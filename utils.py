import torch
import os
import networkx as nx
from torch_geometric.utils.convert import from_networkx
import numpy as np
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix
from scipy.sparse import csgraph, csr_matrix

def load_degree_sequence_from_directory(directory_path):
    max_node = 0 
    min_node = -1
    seqs = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            G = nx.read_edgelist(file_path, nodetype=int)
            max_node = max(max_node, G.number_of_nodes())
            min_node = min(min_node, G.number_of_edges()) if min_node > 0 else G.number_of_edges()
    print("Max node: ", max_node, " Min node:", min_node)
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            G = nx.read_edgelist(file_path, nodetype=int)
            G = nx.convert_node_labels_to_integers(G)
            seq = [deg for _, deg in G.degree()]
            if seq is not None:
                seqs.append(seq)
    return seqs, max_node, min_node

def load_graph_from_directory(directory_path):
    max_node = 0 
    min_node = -1
    graphs = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            G = nx.read_edgelist(file_path, nodetype=int)
            max_node = max(max_node, G.number_of_nodes())
            min_node = min(min_node, G.number_of_edges()) if min_node > 0 else G.number_of_edges()
    print("Max node: ", max_node, " Min node:", min_node)
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            G = nx.read_edgelist(file_path, nodetype=int)
            graph = nx.convert_node_labels_to_integers(G)
            graphs.append(graph)
    return graphs, max_node, min_node


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


def laplacian_eigs(G: nx.Graph, k: int, normed: bool = True):
    """
    Return (vals, vecs) for the K smallest *non-zero* eigenpairs of the (normalized) Laplacian.
    Falls back gracefully on tiny graphs.
    """
    n = G.number_of_nodes()
    if n == 0:
        return np.empty((0,), dtype=np.float32), np.empty((0, 0), dtype=np.float32)
    if n <= 2:
        # trivial spectrum: normalized Laplacian of K2 has {0, 2}
        vals = np.array([1.0] * min(k, max(0, n - 1)), dtype=np.float32)
        vecs = np.ones((n, min(k, max(0, n - 1))), dtype=np.float32) / np.sqrt(n or 1)
        return vals, vecs

    A = csr_matrix(nx.to_scipy_sparse_array(G, dtype=float))
    L = csgraph.laplacian(A, normed=normed)

    # ask for k+1 to include the zero eigenvalue, then drop it
    want = min(max(1, k + 1), n - 1)
    try:
        vals, vecs = eigsh(L, k=want, which="SM")
    except Exception:
        # robust fallback
        return np.ones((k,), dtype=np.float32), np.ones((n, k), dtype=np.float32) / np.sqrt(n)

    idx = np.argsort(vals)
    vals, vecs = vals[idx], vecs[:, idx]
    # drop the (near-)zero eigenvalue
    if vals[0] < 1e-8:
        vals, vecs = vals[1:], vecs[:, 1:]
    # cap/pad to exactly k
    vals = vals[:k].astype(np.float32)
    vecs = vecs[:, :k].astype(np.float32)
    if vals.shape[0] < k:
        pad = k - vals.shape[0]
        vals = np.pad(vals, (0, pad))
        vecs = np.pad(vecs, ((0, 0), (0, pad)))
    return vals, vecs


def normalized_laplacian_dense(G: nx.Graph) -> np.ndarray:
    """Dense normalized Laplacian as float64 ndarray (for small graphs this is fine)."""
    return csgraph.laplacian(
        csr_matrix(nx.to_scipy_sparse_array(G, dtype=float)), normed=True
    ).toarray()


# ---- inner-products needed for fast Frobenius scoring of a double-edge swap ----
def _B_inner(M: np.ndarray, a: int, b: int) -> float:
    # <M, (e_a - e_b)(e_a - e_b)^T> = M_aa + M_bb - 2 M_ab
    return float(M[a, a] + M[b, b] - 2.0 * M[a, b])

def _pair_inner(a: int, b: int, c: int, d: int) -> float:
    # <B_ab, B_cd> = ( (e_a - e_b)^T (e_c - e_d) )^2
    z = (a == c) - (a == d) - (b == c) + (b == d)
    return float(z * z)


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

def havel_hakimi_construction(degree_sequence):
    if not nx.is_valid_degree_sequence_havel_hakimi(degree_sequence):
        print("The degree sequence is not graphical.")
        return None
    # Make a copy to avoid modifying the original
    deg_seq = list(degree_sequence)
    nodes = list(range(len(deg_seq)))
    G = nx.Graph()
    G.add_nodes_from(nodes)
    while any(deg_seq):
        # Sort nodes by remaining degree (descending)
        node_deg_pairs = sorted(zip(nodes, deg_seq), key=lambda x: -x[1])
        u, du = node_deg_pairs[0]
        nodes = [x for x, _ in node_deg_pairs]
        deg_seq = [d for _, d in node_deg_pairs]
        # Take the top node and connect to next 'du' nodes
        for i in range(1, du + 1):
            v = nodes[i]
            G.add_edge(u, v)
            deg_seq[i] -= 1
        deg_seq[0] = 0  # All of u's degree is used
        # Remove nodes with 0 degree for next round
        nodes = [n for n, d in zip(nodes, deg_seq) if d > 0]
        deg_seq = [d for d in deg_seq if d > 0]
    return G
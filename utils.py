import torch
import os
import networkx as nx
from torch_geometric.utils.convert import from_networkx
import numpy as np
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix

def load_degree_sequence_from_directory(directory_path):
    max_node = 0 
    max_edge = 0
    seqs = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            graph = nx.read_edgelist(file_path, nodetype=int)
            max_node = max(max_node, graph.number_of_nodes())
            max_edge = max(max_node, graph.number_of_edges())
    print("Max node: ", max_node, " Max edge:", max_edge)
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            graph = nx.read_edgelist(file_path, nodetype=int)
            graph = nx.convert_node_labels_to_integers(graph)
            seq = [deg for _, deg in graph.degree()]
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


def compute_laplacian_eigenvectors(G, k):
    """
    Compute the first k eigenvectors of the normalized Laplacian matrix of G.
    """
    A = csr_matrix(nx.to_scipy_sparse_array(G, dtype=float))  # ✅ fix
    L = csgraph.laplacian(A, normed=True)
    n = G.number_of_nodes()
    k = min(k, n - 2) if n > 2 else 1  # avoid crash on tiny graphs
    try:
        eigvals, eigvecs = eigsh(L, k=k, which='SM')
        return eigvecs  # shape: (n, k)
    except Exception as e:
        print(f"Eigen decomposition failed: {e}")
        return np.ones((n, 1))  # fallback



def graph_to_data(G, k_eigen):
    eigvecs = compute_laplacian_eigenvectors(G, k_eigen)
    for i, node in enumerate(G.nodes()):
        G.nodes[node]['x'] = eigvecs[i].tolist()
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


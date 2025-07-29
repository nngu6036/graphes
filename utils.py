import torch
import os
import networkx as nx
from torch_geometric.utils.convert import from_networkx
import numpy as np
from scipy.sparse.linalg import eigs
import math

def second_largest_eigenvalue(G, lazy=True):
    # Get adjacency matrix (sparse format)
    A = nx.adjacency_matrix(G).astype(float)
    # Degree vector and diagonal matrix
    degrees = np.array(A.sum(axis=1)).flatten()
    D_inv = np.diag(1.0 / degrees)
    # Transition matrix for simple random walk
    P = D_inv @ A.toarray()
    if lazy:
        # Lazy random walk: P_lazy = 0.5 * (I + P)
        P = 0.5 * (np.eye(G.number_of_nodes()) + P)
    # Compute the 2 largest-magnitude eigenvalues
    eigvals = eigs(P, k=2, which='LM', return_eigenvectors=False)
    # Sort by absolute value
    eigvals_sorted = sorted(np.abs(eigvals), reverse=True)
    return eigvals_sorted[1].real  # Second-largest eigenvalue (magnitude)


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
    max_lambda_2 = 0
    graphs = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            graph = nx.read_edgelist(file_path, nodetype=int)
            max_node = max(max_node, graph.number_of_nodes())
            lambda_2 = second_largest_eigenvalue(graph)
            max_lambda_2 = max(lambda_2,max_lambda_2)
    print("Max node: ", max_node, "Max lambda 2:", max_lambda_2)
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            graph = nx.read_edgelist(file_path, nodetype=int)
            graph = nx.convert_node_labels_to_integers(graph)
            graphs.append(graph)
    return graphs, max_node, max_lambda_2

def graph_to_data(G):
    for node in G.nodes:
        G.nodes[node]['x'] = [1.0]
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
import torch
import os
import networkx as nx
from torch_geometric.utils.convert import from_networkx
import numpy as np
from scipy.sparse.linalg import eigs
import math

def estimate_mixing_time(G, epsilon=0.0001):
    A = nx.adjacency_matrix(G).astype(float)
    D = np.array(A.sum(axis=1)).flatten()
    D_inv = np.diag(1.0 / D)
    P = D_inv @ A.toarray()

    # Lazy walk: P_lazy = 0.5 * (I + P)
    P_lazy = 0.5 * (np.eye(G.number_of_nodes()) + P)

    # Compute second largest eigenvalue
    eigenvalues = eigs(P_lazy, k=2, which='LM', return_eigenvectors=False)
    lambda_2 = sorted(np.abs(eigenvalues))[-2]
    gap = 1 - lambda_2

    pi_min = np.min(D / np.sum(D))
    t_mix = (1 / gap) * np.log(1 / (epsilon * pi_min))

    return t_mix


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
    max_mix_time = 0
    graphs = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            graph = nx.read_edgelist(file_path, nodetype=int)
            max_node = max(max_node, graph.number_of_nodes())
            mix_time = estimate_mixing_time(graph)
            max_mix_time = max(mix_time,max_mix_time)
    print("Max node: ", max_node, "Max mix time:", max_mix_time)
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            graph = nx.read_edgelist(file_path, nodetype=int)
            graph = nx.convert_node_labels_to_integers(graph)
            graphs.append(graph)
    return graphs, max_node, max_mix_time

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
import torch
import os
import networkx as nx
from torch_geometric.utils.convert import from_networkx
import numpy as np


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

# Compute statistical features of the graph
def graph_features(G):
    clustering = np.mean(list(nx.clustering(G).values()))
    degree_seq = np.array(sorted([d for _, d in G.degree()]))
    return clustering, degree_seq

# Measure distance between feature distributions
def features_distance(f1, f2):
    clustering_diff = np.abs(f1[0] - f2[0])
    degree_diff = np.linalg.norm(f1[1] - f2[1])
    return clustering_diff + degree_diff

# Function for a single edge rewiring step
def edge_rewire(G):
    edges = list(G.edges())
    np.random.shuffle(edges)
    for (u, v) in edges:
        for (x, y) in edges:
            # Check conditions to avoid self-loop and multiple edges
            if len({u, v, x, y}) == 4 and not G.has_edge(u, x) and not G.has_edge(v, y):
                G.remove_edges_from([(u, v), (x, y)])
                G.add_edges_from([(u, x), (v, y)])
                return
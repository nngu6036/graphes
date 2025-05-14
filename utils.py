import torch
import torch.nn.functional as F
import os
import argparse
import toml
from pathlib import Path
import networkx as nx
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data

def encode_degree_sequence(degree_sequence , max_class):
    sorted_degree_sequence = sorted(degree_sequence, reverse=True)
    one_hot_tensor = torch.zeros(max_class, dtype=torch.float32)
    for deg in sorted_degree_sequence:
        if 1 <= deg <= max_class:  # Only consider degrees within range
            one_hot_tensor[deg - 1] += 1  # (i-th index represents degree i+1)
        else:
            raise Error(f'Invalid degree sequqnce  {degree_sequence}')
    return one_hot_tensor

def decode_degree_sequence(one_hot_tensor):
    def decode_row(row):
        """Helper function to decode a single row."""
        degree_sequence = []
        for i, count in enumerate(row):
            degree = i + 1  # Degree is index + 1
            count = int(count.item())  # Convert float to int
            degree_sequence.extend([degree] * count)  # Append 'count' times
        return degree_sequence

    # Check if input is 1D or 2D
    if one_hot_tensor.dim() == 1:  # Single degree sequence
        return decode_row(one_hot_tensor)
    elif one_hot_tensor.dim() == 2:  # Batch processing for multiple degree sequences
        return [decode_row(row) for row in one_hot_tensor]
    else:
        raise ValueError("Input tensor must be 1D or 2D.")

def load_graph_sequence_from_file(file_path, max_nodes):
    """
    Load a graph from a single file and apply one-hot encoding.
    The file format should be compatible with NetworkX's read functions.
    """
    try:
        graph = nx.read_edgelist(file_path, nodetype=int)
        graph = nx.convert_node_labels_to_integers(graph)
        x = encode_degree_sequence([deg for _, deg in graph.degree()],max_nodes)
        batch = torch.zeros(max_nodes, dtype=torch.long)
        return Data(x=x, batch=batch)
    except Exception as e:
        print(f"Error loading graph from {file_path}: {e}")
        return None

def create_graph_sequence_from_directory(directory_path):
    graphs = []
    max_nodes = 0
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            graph = nx.read_edgelist(file_path, nodetype=int)
            max_nodes = max(max_nodes, graph.number_of_nodes())
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            graph = load_graph_sequence_from_file(file_path, max_nodes)
            if graph is not None:
                graphs.append(graph)
    return graphs, max_nodes

def check_sequence_validity(degree_sequence):
    """Checks if a degree sequence is valid after removing all zeros."""
    if len(degree_sequence) == 0:
        return False
    # Degree sequence sum must be even
    if sum(degree_sequence) % 2 != 0:
        return False
    # Sort in descending order
    sorted_degrees = sorted(degree_sequence, reverse=True)
    # Apply Erdős–Gallai theorem
    for k in range(1, len(sorted_degrees) + 1):
        lhs = sum(sorted_degrees[:k])
        rhs = k * (k - 1) + sum(min(d, k) for d in sorted_degrees[k:])
        if lhs > rhs:
            return False
    return True


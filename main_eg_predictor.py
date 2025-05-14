import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import os
import argparse
import random
from pathlib import Path
import networkx as nx
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
import numpy as np
#from utils import create_graph_sequence_from_directory, split_graph_data

num_epochs = 400
learning_rate = 0.001

class EGSumLosEstimator(torch.nn.Module):
    def __init__(self, input_dim):
        super(EGSumLosEstimator, self).__init__()
        self.linear = torch.nn.Linear(input_dim, 1, bias=False)  # Linear sum

    def forward(self, one_hot_tensor):
        N = one_hot_tensor.shape[1]
        indices = torch.arange(N, dtype=torch.float32)
        x = one_hot_tensor * (indices+1)
        sum_pred = self.linear(x) # Predict sum of elements
        return (1 - torch.cos(np.pi * sum_pred)) / 2  # Cosine approximation of mod 2


class EGSequenceLosEstimator(torch.nn.Module):
    def __init__(self, input_dim):
        super(EGSequenceLosEstimator, self).__init__()
        self.hidden = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),  # Increased neurons
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),  # Increased second layer neurons
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )

    def forward(self, one_hot_tensor):
        n = one_hot_tensor.shape[0]
        indices = torch.arange(n, dtype=torch.float32)  # Ensure floating point
        x = one_hot_tensor * (indices+1)
        k_values = torch.cumsum(one_hot_tensor)
        lhs = torch.cumsum(x)
        rhs = k_values * (k_values - 1) + 0.5 *(torch.sum(x) - torch.cumsum(x) )
        return torch.sum(torch,sigmoid(lhs-rhs))


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

def split_graph_data(graphs, test_ratio=0.2):
    """Splits graph dataset into training and test sets."""
    train_graphs, test_graphs = train_test_split(graphs, test_size=test_ratio, random_state=42)
    return train_graphs, test_graphs

def generate_invalid_degree_vector(valid_seq,max_nodes):
    """
    Generate an invalid degree sequence by modifying a valid one.
    """
    invalid_seq = valid_seq.clone()
    invalid_seq[random.randint(0, max_nodes-1)] += random.choice([ -3, -1, 1, 3])  # Break the even
    return invalid_seq.float()

def erdos_gallai_loss(degree_vec):
    """Differentiable penalty based on Erdős–Gallai theorem violations."""
    degree_sequence = degree_sequence.flatten()
    # Compute the sum of i * degree_sequence[i]
    n = degree_sequence.shape[1]
    indices = torch.arange(n, dtype=torch.float32)  # Ensure floating point
    result = torch.sum(indices * degree_sequence[0])

    # Differentiable modulo approximation using cosine
    sum_penalty = 0.5 * (1 - torch.cos(torch.pi * result))

    sorted_degrees = degree_sequence.sort(descending=True)[0] + 1e-6 * torch.arange(len(degree_sequence), device=degree_sequence.device)
    # Soft penalty for even sum constraint
    sum_penalty = torch.sigmoid(10 * (torch.sin(torch.pi * torch.sum(sorted_degrees))))
    # Soft constraint for Erdős–Gallai conditions
    k_values = torch.arange(1, len(sorted_degrees) + 1, device=degree_sequence.device)
    lhs = torch.cumsum(sorted_degrees, dim=0)
    rhs = k_values * (k_values - 1) + torch.cumsum(torch.minimum(sorted_degrees, k_values), dim=0)
    # Soft differentiable constraint penalty
    constraint_penalty = torch.sum(F.softplus(lhs - rhs))
    return constraint_penalty + sum_penalty


def train_sum_predictor(model, x_train , num_epochs, learning_rate):
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()
    model.train()
    y_train = torch.sum(x_train, axis=1, keepdims=True)    # Compute sum for each vector
    y_train = (y_train % 2).to(torch.float32)  # or .to(torch.long) if needed
    for epoch in range(num_epochs):
        print(f"Training Epoch {epoch + 1}/{num_epochs}")
        total_loss = 0
        optimizer.zero_grad()
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train) 
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(x_train)}")


def train_eg_predictor(model, true_data, false_data , num_epochs, learning_rate):
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.BCELoss()  # Use BCE Loss for classification
    model.train()
    y_train_true = torch.tensor([1.0] * len(true_data)).float()
    y_train_false = torch.tensor([0.0] * len(false_data)).float()
    for epoch in range(num_epochs):
        print(f"Training Epoch {epoch + 1}/{num_epochs}")
        total_loss = 0
        for data, label in zip(true_data + false_data, y_train_true.tolist() + y_train_false.tolist()):
            optimizer.zero_grad()
            predictions = model(data.flatten()).squeeze()  # Ensure input is flattened
            loss = criterion(predictions, torch.tensor(label).float())  # Use correct label
            loss += erdos_gallai_loss(data) * 0.1  # Add Erdős–Gallai constraint penalty
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / (len(true_data) + len(false_data)):.4f}")

def evaluate_seq_classifier(model, test_data):
    """Evaluates the model on test dataset."""
    model.eval()
    with torch.no_grad():
        degree_validities = [model(data.flatten()).item() for data in test_data]  # Ensure flattening
        validity_percentage = (sum(1.0 if score > 0.5 else 0.0 for score in degree_validities) / len(degree_validities)) * 100
        return {"Degree Validity (%)": validity_percentage}


def evaluate_sum_loss_estimator(model, x_test):
    """Evaluates the model on test dataset."""
    model.eval()
    y_test = torch.sum(x_test, axis=1, keepdims=True)    # Compute sum for each vector
    y_test = (y_test % 2).to(torch.float32)  # or .to(torch.long) if needed
    with torch.no_grad():
        predictions = model(x_test)
        print("Predicted label:", torch.round(predictions).squeeze().tolist())
        print("Test label:", y_test.squeeze().tolist())  # Ground truth
        print("MSE Error:", torch.sum((predictions).squeeze() - y_test.squeeze())**2)  

def main(args):
    dataset_dir = Path("datasets")
    model_dir = Path("models")
    dataset_dir = dataset_dir / args.dataset_dir
    graphs, max_node = create_graph_sequence_from_directory(dataset_dir)
    train_graphs, test_graphs = split_graph_data(graphs)
    train_data = torch.stack([graph.x for graph in train_graphs])
    test_data = torch.stack([graph.x for graph in test_graphs])
    model = EGSumLosEstimator(input_dim=max_node)
    train_sum_predictor(model, train_data, num_epochs, learning_rate)
    #if args.output_model:
    #    model.save_model(model_dir / args.output_model)
    #    print(f"Model saved to {args.output_model}")


    #train_true_data = train_data
    #train_false_data = [generate_invalid_degree_vector(data, max_node) for data in train_true_data ]
    #model = EGSeqLosEstimator(input_dim=max_node)
    #train_seq_loss_estimator(model, train_data, num_epochs, learning_rate)
    #if args.output_model:
    #    model.save_model(model_dir / args.output_model)
    #    print(f"Model saved to {args.output_model}")

    print(f"Evaluate sum loss")
    evaluate_sum_loss_estimator(model, test_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MS-VAE for Graph Generation')
    parser.add_argument('--dataset-dir', type=str, help='Path to the directory containing graph files')
    parser.add_argument('--output-model', type=str, help='Path to save the trained model')
    args = parser.parse_args()
    main(args)

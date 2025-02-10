import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import os
import argparse
import toml
from pathlib import Path
from torch_geometric.utils import from_networkx
import networkx as nx
from scipy.optimize import linear_sum_assignment
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class MSVAEEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(MSVAEEncoder, self).__init__()
        self.hidden_layer = torch.nn.Linear(input_dim, hidden_dim)
        self.mean_layer = torch.nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = torch.nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, batch):
        h = F.relu(self.hidden_layer(x))
        mean, logvar = self.mean_layer(h), self.logvar_layer(h)
        return mean, logvar

class MSVAEDecoder(torch.nn.Module):
    def __init__(self, latent_dim, hidden_dim, max_output_dim):
        super(MSVAEDecoder, self).__init__()
        self.hidden_layer = torch.nn.Linear(latent_dim, hidden_dim)
        self.degree_layer = torch.nn.Linear(hidden_dim, max_output_dim)
        self.multiplicity_layer = torch.nn.Linear(hidden_dim, max_output_dim)

    def forward(self, z, output_dim):
        h = F.relu(self.hidden_layer(z))
        degrees = F.softmax(self.degree_layer(h), dim=-1)
        multiplicities = F.softplus(self.multiplicity_layer(h))
        return degrees, multiplicities

class MSVAE(torch.nn.Module):
    def __init__(self, max_input_dim, hidden_dim, latent_dim):
        super(MSVAE, self).__init__()
        self.encoder = MSVAEEncoder(max_input_dim, hidden_dim, latent_dim)
        self.decoder = MSVAEDecoder(latent_dim, hidden_dim, max_output_dim=max_input_dim)
        self.latent_dim = latent_dim

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, data):
        x, batch = data.x, data.batch
        mean, logvar = self.encoder(x, batch)
        z = self.reparameterize(mean, logvar)
        degrees, frequencies = self.decoder(z, x.size(0))
        return degrees, frequencies, mean, logvar, x.size(0)

    def generate(self, num_samples):
        self.eval()
        with torch.no_grad():
            z = torch.randn((num_samples, self.latent_dim))
            degrees, frequencies = self.decoder(z, self.decoder.degree_layer.out_features)
            # Use reconstruct_multiset to get the degree multiset
            degree_vec = reconstruct_degree_vector(degrees, frequencies,train_mode = False)
            return decode_degree_sequence(degree_vec)

    def save_model(self, file_path):
        torch.save(self.state_dict(), file_path)

    def load_model(self, file_path):
        self.load_state_dict(torch.load(file_path))
        self.eval()


def relaxed_round(x):
    return (x - x.detach()) + x.round()

def encode_degree_sequence(degree_sequence , max_class):
    sorted_degree_sequence = sorted(degree_sequence, reverse=True)
    one_hot_tensor = torch.zeros(max_class, dtype=torch.float32)
    for deg in sorted_degree_sequence:
        if 1 <= deg <= max_class:  # Only consider degrees within range
            one_hot_tensor[deg - 1] += 1  # (i-th index represents degree i+1)
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

def compute_chamfer_distance(set1, set2):
    if len(set1) == 0 or len(set2) == 0:
        return float('inf')  # If one of the sets is empty, the distance is undefined.
    set1 = torch.tensor(set1, dtype=torch.float).unsqueeze(1)
    set2 = torch.tensor(set2, dtype=torch.float).unsqueeze(1)
    dists_1_to_2 = torch.min(torch.cdist(set1, set2, p=2), dim=1).values
    dists_2_to_1 = torch.min(torch.cdist(set2, set1, p=2), dim=1).values
    chamfer_distance = torch.sum(dists_1_to_2) + torch.sum(dists_2_to_1)
    return chamfer_distance.item()

def compute_earth_movers_distance(set1, set2):
    if len(set1) == 0 or len(set2) == 0:
        return float('inf')  # Undefined if one of the sets is empty.
    set1 = torch.tensor(set1, dtype=torch.float).unsqueeze(1)
    set2 = torch.tensor(set2, dtype=torch.float).unsqueeze(1)
    cost_matrix = torch.cdist(set1, set2, p=2).cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    emd = cost_matrix[row_ind, col_ind].sum()
    return emd

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

def reconstruct_degree_vector(degrees, frequencies, train_mode = False):
    # Compute the weighted degrees
    degree_vec = degrees * frequencies
    # Round to ensure discrete values (use relaxed_round during training if needed)
    if train_mode:
        degree_vec = relaxed_round(degree_vec)
    else:
        degree_vec = degree_vec.round()
    return degree_vec


def train_vae_decoder_for_degree_sequence(model, graphs, num_epochs, learning_rate, weights):
    optimizer = Adam(model.parameters(), lr=learning_rate)
    model.train()
    max_node = max([graph.num_nodes for graph in graphs])
    for epoch in range(num_epochs):
        print("Traininig iteration ", epoch)
        total_loss = 0
        for graph in graphs:
            optimizer.zero_grad()
            degrees, frequencies, mean, logvar, set_size = model(graph)
            # Reconstruct the multiset
            recon_degree_vec = reconstruct_degree_vector(degrees, frequencies, train_mode = True)
            # Compute the loss
            loss = loss_function(recon_degree_vec, graph.x, mean, logvar, weights)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(graphs):.4f}")

def loss_function(recon_degree_vec, target_degree_vec, mean, logvar, weights):
    recon_weight, kl_weight = weights.get('reconstruction', 1.0), weights.get('kl_divergence', 1.0)
    erdos_gallai_weight = weights.get('erdos_gallai', 1.0)
    max_size = max(recon_degree_vec.size(0), target_degree_vec.size(0))
    recon_degree_vec = F.pad(recon_degree_vec, (0, max_size - recon_degree_vec.size(0)))
    target_degree_vec = F.pad(target_degree_vec, (0, max_size - target_degree_vec.size(0)))
    recon_loss = torch.sum((recon_degree_vec - target_degree_vec) ** 2)
    kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    degree_sequence = decode_degree_sequence(recon_degree_vec)
    erdos_gallai_loss = sum(degree_sequence) if degree_sequence else max_size
    return recon_weight * recon_loss + kl_weight * kl_loss + erdos_gallai_weight * erdos_gallai_loss

def evaluate_generated_multisets(model, graphs, num_samples):
    model.eval()
    with torch.no_grad():
        generated_sets = model.generate(num_samples)
        reference_degrees = [graph.x.squeeze(0) for graph in graphs]
        reference_sets = [decode_degree_sequence(deg) for deg in reference_degrees]
        chamfer_distances = [compute_chamfer_distance(ref, gen) for ref in reference_sets for gen in generated_sets]
        avg_chamfer_distance = sum(chamfer_distances) / len(chamfer_distances) if chamfer_distances else float('inf')
        emd_distances = [compute_earth_movers_distance(ref, gen) for ref in reference_sets for gen in generated_sets]
        avg_emd_distance = sum(emd_distances) / len(emd_distances) if emd_distances else float('inf')
        degree_validities = [check_sequence_validity(degree_sequence) for degree_sequence in generated_sets]
        validity_percentage = (sum(degree_validities) / len(degree_validities)) * 100 if len(degree_validities) > 0 else 0
        return {
            "Chamfer Distance": avg_chamfer_distance,
            "Earth Mover's Distance": avg_emd_distance,
            "Degree Validity (%)": validity_percentage
        }

def split_graph_data(graphs, test_ratio=0.2):
    """Splits graph dataset into training and test sets."""
    train_graphs, test_graphs = train_test_split(graphs, test_size=test_ratio, random_state=42)
    return train_graphs, test_graphs

def evaluate_test_multisets(model, test_graphs):
    """Evaluates the model on test dataset."""
    model.eval()
    with torch.no_grad():
        test_degrees = [graph.x.squeeze(0) for graph in test_graphs]
        test_sets = [decode_degree_sequence(deg) for deg in test_degrees]
        chamfer_distances = [compute_chamfer_distance(ref, gen) for ref in test_sets for gen in test_sets]
        avg_chamfer_distance = sum(chamfer_distances) / len(chamfer_distances) if chamfer_distances else float('inf')
        emd_distances = [compute_earth_movers_distance(ref, gen) for ref in test_sets for gen in test_sets]
        avg_emd_distance = sum(emd_distances) / len(emd_distances) if emd_distances else float('inf')
        degree_validities = [check_sequence_validity(degree_sequence) for degree_sequence in test_sets]
        validity_percentage = (sum(degree_validities) / len(degree_validities)) * 100 if len(degree_validities) > 0 else 0
        return {
            "Chamfer Distance": avg_chamfer_distance,
            "Earth Mover's Distance": avg_emd_distance,
            "Degree Validity (%)": validity_percentage
        }

def main(args):
    config_dir = Path("configs")
    dataset_dir = Path("datasets")
    model_dir = Path("models")
    config_file = config_dir / args.config_file
    config = toml.load(config_file)
    dataset_dir = dataset_dir / args.dataset_dir
    graphs, max_node = create_graph_sequence_from_directory(dataset_dir)
    if len(graphs) == 0:
        print("No valid graph files found in the directory.")
        return
    train_graphs, test_graphs = split_graph_data(graphs)
    hidden_dim = config['training']['hidden_dim']
    latent_dim = config['training']['latent_dim']
    input_dim = max_node  # input dimension matches one-hot encoded degrees
    model = MSVAE(max_input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
    if args.input_model:
        model.load_model(model_dir / args.input_model)
        print(f"Model loaded from {args.input_model}")
    else:
        num_epochs = config['training']['num_epochs']
        learning_rate = config['training']['learning_rate']
        weights = config['training']['weights']
        train_vae_decoder_for_degree_sequence(model, train_graphs, num_epochs, learning_rate, weights)
    if args.output_model:
        model.save_model(model_dir / args.output_model)
        print(f"Model saved to {args.output_model}")

    if args.evaluate:
        print(f"Evaluate generated multiset")
        generated_degrees = model.generate(config['inference']['generate_samples'])
        evaluation_metrics = evaluate_generated_multisets(model, graphs, config['inference']['generate_samples'])
        for metric, value in evaluation_metrics.items():
            print(f"{metric}: {value:.4f}")

        print(f"Evaluate test multiset")
        evaluation_metrics = evaluate_test_multisets(model, test_graphs)
        for metric, value in evaluation_metrics.items():
            print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MS-VAE for Graph Generation')
    parser.add_argument('--dataset-dir', type=str, help='Path to the directory containing graph files')
    parser.add_argument('--config-file', type=str, required=True, help='Path to the configuration file in TOML format')
    parser.add_argument('--output-model', type=str, help='Path to save the trained model')
    parser.add_argument('--input-model', type=str, help='Path to load a pre-trained model')
    parser.add_argument('--evaluate', action='store_true', help='Whether we evaluate the model')
    args = parser.parse_args()
    main(args)

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
            degree_multiset = reconstruct_multiset(degrees, frequencies,train_mode = False)
            return degree_multiset

    def save_model(self, file_path):
        torch.save(self.state_dict(), file_path)

    def load_model(self, file_path):
        self.load_state_dict(torch.load(file_path))
        self.eval()

def relaxed_round(x):
    return (x - x.detach()) + x.round()


def reconstruct_multiset(degrees, frequencies, train_mode = False):
    # Compute the weighted degrees
    degree_multiset = degrees * frequencies
    # Round to ensure discrete values (use relaxed_round during training if needed)
    if train_mode:
        degree_multiset = relaxed_round(degree_multiset)
    else:
        degree_multiset = degree_multiset.round()
    return degree_multiset


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
            recon_multiset = reconstruct_multiset(degrees, frequencies, train_mode = True)
            # Compute the loss
            loss = loss_function(recon_multiset, graph.x, mean, logvar, weights)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(graphs):.4f}")

def loss_function(recon_multiset, target_multiset, mean, logvar, weights):
    recon_weight, kl_weight = weights.get('reconstruction', 1.0), weights.get('kl_divergence', 1.0)
    erdos_gallai_weight = weights.get('erdos_gallai', 1.0)

    max_size = max(recon_multiset.size(0), target_multiset.size(0))
    recon_multiset = F.pad(recon_multiset, (0, max_size - recon_multiset.size(0)))
    target_multiset = F.pad(target_multiset, (0, max_size - target_multiset.size(0)))

    recon_loss = torch.sum((recon_multiset - target_multiset) ** 2)
    kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

    erdos_gallai_loss = 0.0
    degree_sequence = recon_multiset.sum(dim=0).sort(descending=True).values
    for k in range(1, len(degree_sequence) + 1):
        lhs = torch.sum(degree_sequence[:k])
        rhs = k * (k - 1) + torch.sum(torch.min(degree_sequence[k:], torch.tensor(k, dtype=torch.float)))
        if lhs > rhs:
            erdos_gallai_loss += (lhs - rhs) ** 2

    return recon_weight * recon_loss + kl_weight * kl_loss + erdos_gallai_weight * erdos_gallai_loss

def compute_chamfer_distance(set1, set2):
    """
    Compute Chamfer Distance between two sets, ensuring it works for sets of different sizes.
    """
    if set1.size(0) == 0 or set2.size(0) == 0:
        return float('inf')  # If one of the sets is empty, the distance is undefined.

    dists_1_to_2 = torch.min(torch.cdist(set1, set2, p=2), dim=1).values
    dists_2_to_1 = torch.min(torch.cdist(set2, set1, p=2), dim=1).values

    chamfer_distance = torch.sum(dists_1_to_2) + torch.sum(dists_2_to_1)
    return chamfer_distance

def compute_earth_movers_distance(set1, set2):
    """
    Compute Earth Mover's Distance (EMD) between two sets, ensuring compatibility for different sizes.
    """
    if set1.size(0) == 0 or set2.size(0) == 0:
        return float('inf')  # Undefined if one of the sets is empty.
    cost_matrix = torch.cdist(set1, set2, p=2).cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    emd = cost_matrix[row_ind, col_ind].sum()
    return emd

def compute_coverage(reference_sets, generated_sets):
    """
    Compute coverage percentage of generated sets matching the reference sets.
    """
    coverage_count = 0
    for ref_set in reference_sets:
        if ref_set.size(0) == 0:
            continue
        distances = [compute_chamfer_distance(ref_set, gen_set) for gen_set in generated_sets if gen_set.size(0) > 0]
        if distances and min(distances) < 1e-4:  # Threshold for coverage match
            coverage_count += 1
    return (coverage_count / len(reference_sets)) * 100 if len(reference_sets) > 0 else 0

def compute_1nn_accuracy(reference_sets, generated_sets):
    """
    Compute 1-NN accuracy for distinguishing reference sets from generated sets.
    """
    if not reference_sets or not generated_sets:
        return 0.0

    distances = []
    labels = []

    for ref_set in reference_sets:
        if ref_set.size(0) == 0:
            continue
        for gen_set in generated_sets:
            if gen_set.size(0) > 0:
                distances.append(compute_chamfer_distance(ref_set, gen_set))
                labels.append(0)  # Reference label

    for i, gen_set1 in enumerate(generated_sets):
        if gen_set1.size(0) == 0:
            continue
        for j, gen_set2 in enumerate(generated_sets):
            if i != j and gen_set2.size(0) > 0:
                distances.append(compute_chamfer_distance(gen_set1, gen_set2))
                labels.append(1)  # Generated label

    if not distances:
        return 0.0

    distances = torch.tensor(distances)
    labels = torch.tensor(labels)
    nearest_indices = torch.argmin(distances.view(len(reference_sets), -1), dim=1)
    correct_predictions = torch.sum(labels[nearest_indices] == 0)
    return (correct_predictions / len(reference_sets)) * 100 if len(reference_sets) > 0 else 0

def evaluate_generated_multisets(model, graphs, num_samples):
    model.eval()
    with torch.no_grad():
        generated_degrees = model.generate(num_samples)
        generated_sets = [gen.unsqueeze(0) for gen in generated_degrees if gen.size(0) > 0]
        reference_degrees = [graph.x.sum(dim=0, keepdim=True) for graph in graphs]
        reference_sets = [ref.unsqueeze(0) for ref in reference_degrees if ref.size(0) > 0]
        
        chamfer_distances = [compute_chamfer_distance(ref, gen) for ref in reference_sets for gen in generated_sets]
        avg_chamfer_distance = sum(chamfer_distances) / len(chamfer_distances) if chamfer_distances else float('inf')

        #emd_distances = [compute_earth_movers_distance(ref, gen) for ref in reference_sets for gen in generated_sets]
        #avg_emd_distance = sum(emd_distances) / len(emd_distances) if emd_distances else float('inf')

        coverage = compute_coverage(reference_sets, generated_sets)
        one_nn_accuracy = compute_1nn_accuracy(reference_sets, generated_sets)

        degree_validities = []
        for degree_sequence in generated_degrees:
            if degree_sequence.size(0) == 0:
                continue
            sorted_degrees = degree_sequence.sort(descending=True).values
            is_valid = True
            for k in range(1, len(sorted_degrees) + 1):
                lhs = torch.sum(sorted_degrees[:k])
                rhs = k * (k - 1) + torch.sum(torch.min(sorted_degrees[k:], torch.tensor(k, dtype=torch.float)))
                if lhs > rhs:
                    is_valid = False
                    break
            degree_validities.append(is_valid)
        validity_percentage = (sum(degree_validities) / len(degree_validities)) * 100 if len(degree_validities) > 0 else 0

        return {
            "Chamfer Distance": avg_chamfer_distance,
            #"Earth Mover's Distance": avg_emd_distance,
            "Coverage (%)": coverage,
            "1-NN Accuracy (%)": one_nn_accuracy,
            "Degree Validity (%)": validity_percentage
        }

def load_graph_from_file(file_path, max_nodes):
    """
    Load a graph from a single file and apply one-hot encoding.
    The file format should be compatible with NetworkX's read functions.
    """
    try:
        graph = nx.read_edgelist(file_path, nodetype=int)
        graph = nx.convert_node_labels_to_integers(graph)
        x = F.one_hot(torch.tensor([graph.degree[n] for n in range(graph.number_of_nodes())]), 
                      num_classes=max_nodes).float()
        x = x.sum(dim=0, keepdim=True)
        batch = torch.zeros(max_nodes, dtype=torch.long)
        return Data(x=x, batch=batch)
    except Exception as e:
        print(f"Error loading graph from {file_path}: {e}")
        return None

def create_graph_data_from_directory(directory_path):
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
            graph = load_graph_from_file(file_path, max_nodes)
            if graph is not None:
                graphs.append(graph)
    return graphs, max_nodes

def main():
    parser = argparse.ArgumentParser(description='MS-VAE for Graph Generation')
    parser.add_argument('--dataset-dir', type=str, required=True, help='Path to the directory containing graph files')
    parser.add_argument('--config-file', type=str, required=True, help='Path to the configuration file in TOML format')
    parser.add_argument('--output-model', type=str, help='Path to save the trained model')
    parser.add_argument('--input-model', type=str, help='Path to load a pre-trained model')
    args = parser.parse_args()

    config_dir = Path("configs")
    dataset_dir = Path("datasets")
    model_dir = Path("models")

    dataset_dir = dataset_dir / args.dataset_dir
    config_file = config_dir / args.config_file

    config = toml.load(config_file)
    graphs, max_node = create_graph_data_from_directory(dataset_dir)
    if len(graphs) == 0:
        print("No valid graph files found in the directory.")
        return
        
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

        train_vae_decoder_for_degree_sequence(model, graphs, num_epochs, learning_rate, weights)

    if args.output_model:
        model.save_model(model_dir / args.output_model)
        print(f"Model saved to {args.output_model}")

    if config['inference']['generate_samples'] > 0:
        generated_degrees = model.generate(config['inference']['generate_samples'])

    if config['inference']['evaluate']:
        evaluation_metrics = evaluate_generated_multisets(model, graphs, config['inference']['generate_samples'])
        for metric, value in evaluation_metrics.items():
            print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()

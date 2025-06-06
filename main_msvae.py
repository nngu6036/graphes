import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import os
import argparse
import toml
import math
from pathlib import Path
from torch_geometric.utils import from_networkx
import networkx as nx
from scipy.optimize import linear_sum_assignment
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

N = 0

class MSVAEEncoder(torch.nn.Module):
    def __init__(self,  input_dim, hidden_dim, latent_dim):
        super(MSVAEEncoder, self).__init__()
        self.hidden_layer = torch.nn.Linear(input_dim, hidden_dim)
        self.mean_layer = torch.nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = torch.nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = F.relu(self.hidden_layer(x))
        mean, logvar = self.mean_layer(h), self.logvar_layer(h)
        return mean, logvar

class MSVAEDecoder(torch.nn.Module):
    def __init__(self, latent_dim, hidden_dim, max_output_dim):
        super(MSVAEDecoder, self).__init__()
        self.hidden_layer = torch.nn.Linear(latent_dim, hidden_dim)
        self.frequency_layer = torch.nn.Linear(hidden_dim, max_output_dim)

    def forward(self, z, batch):
        h = F.relu(self.hidden_layer(z))
        h = h.unsqueeze(1) 
        multiplicities = F.softplus(self.frequency_layer(h))  # Ensure positive values
        return multiplicities

class MSVAE(torch.nn.Module):
    def __init__(self, max_input_dim, hidden_dim, latent_dim):
        super(MSVAE, self).__init__()
        self.encoder = MSVAEEncoder( max_input_dim, hidden_dim, latent_dim)
        self.decoder = MSVAEDecoder(latent_dim, hidden_dim,max_input_dim)
        self.latent_dim = latent_dim

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, batch):
        mean, logvar = self.encoder(batch)
        z = self.reparameterize(mean, logvar)
        frequencies = self.decoder(z,batch)
        return frequencies, mean, logvar

    def fix_degree_sum_even(self, freq: torch.Tensor) -> torch.Tensor:
        """
        Ensures the sum of degrees in `freq` is even.
        Subtracts 1 from the highest non-zero entry if sum is odd.
        """
        indices = torch.arange(1, N+1).float().to(freq.device)
        total = torch.sum(freq * indices).item()
        if total % 2 != 0:
            even_indices = torch.arange(0, freq.size(0), 2)  # get even indices: 0, 2, 4, ...
            even_values = freq[even_indices]                # values at even indices
            relative_max_idx = torch.argmax(even_values)    # index within the even subset
            max_idx = even_indices[relative_max_idx]        # map back to original index
            if freq[max_idx] > 0:
                freq[max_idx] += 1
        return freq

    def generate(self, num_samples):
        self.eval()
        with torch.no_grad():
            z = torch.randn((num_samples, self.latent_dim))
            dummy_batch = torch.zeros((num_samples, self.decoder.frequency_layer.out_features))
            frequencies = self.decoder(z, dummy_batch)
            return torch.stack(frequencies)
            fixed_sequences = []
            for freq in frequencies:
                freq_rounded = freq.squeeze().round()
                freq_fixed = self.fix_degree_sum_even(freq_rounded)
                fixed_sequences.append(freq_fixed)
            return torch.stack(fixed_sequences)

    def save_model(self, file_path):
        torch.save(self.state_dict(), file_path)

    def load_model(self, file_path):
        self.load_state_dict(torch.load(file_path))
        self.eval()

def encode_degree_sequence(degree_sequence , max_class):
    sorted_degree_sequence = sorted(degree_sequence, reverse=True)
    one_hot_tensor = torch.zeros(max_class, dtype=torch.float32)
    for deg in sorted_degree_sequence:
        if 1 <= deg <= max_class:
            one_hot_tensor[deg - 1] += 1
        else:
            raise ValueError(f'Invalid degree sequence {degree_sequence}')
    return one_hot_tensor

def decode_degree_sequence(one_hot_tensor):
    degree_sequence = []
    for i, count in enumerate(one_hot_tensor.squeeze()):
        degree = i + 1  # Degree is index + 1
        count = int(count.item())  # Convert float to int
        degree_sequence.extend([degree] * count)  # Append 'count' times
    return degree_sequence


def load_degree_sequence_from_directory(directory_path):
    global N
    seqs = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            graph = nx.read_edgelist(file_path, nodetype=int)
            N = max(N, graph.number_of_nodes())
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            graph = nx.read_edgelist(file_path, nodetype=int)
            graph = nx.convert_node_labels_to_integers(graph)
            seq = encode_degree_sequence([deg for _, deg in graph.degree()],N)
            if seq is not None:
                seqs.append(seq)
    return seqs

def compute_chamfer_distance(set1, set2):
    if len(set1) == 0 or len(set2) == 0:
        return float('inf')  # If one of the sets is empty, the distance is undefined.
    set1 = torch.tensor(set1, dtype=torch.float).unsqueeze(1)
    set2 = torch.tensor(set2, dtype=torch.float).unsqueeze(1)
    dists_1_to_2 = torch.min(torch.cdist(set1, set2, p=2), dim=1).values
    dists_2_to_1 = torch.min(torch.cdist(set2, set1, p=2), dim=1).values
    chamfer_distance = torch.sum(dists_1_to_2) + torch.sum(dists_2_to_1)
    return chamfer_distance.item()

def compute_degree_histograms(sequences, max_degree):
    histograms = []
    for seq in sequences:
        hist = torch.zeros(max_degree + 1)
        for deg in seq:
            if 0 <= deg <= max_degree:
                hist[deg] += 1
        hist /= hist.sum() + 1e-8  # normalize + stability
        histograms.append(hist)
    return torch.stack(histograms)

def compute_kl_divergence(p_hist, q_hist, eps=1e-8):
    p_ = p_hist + eps
    q_ = q_hist + eps
    p_ = p_ / p_.sum()
    q_ = q_ / q_.sum()
    return torch.sum(p_ * torch.log(p_ / q_)).item()


def compute_mmd(X, Y, gamma=1.0):
    X_np = X.detach().cpu().numpy()
    Y_np = Y.detach().cpu().numpy()
    K_xx = rbf_kernel(X_np, X_np, gamma=gamma)
    K_yy = rbf_kernel(Y_np, Y_np, gamma=gamma)
    K_xy = rbf_kernel(X_np, Y_np, gamma=gamma)

    return float(K_xx.mean() + K_yy.mean() - 2 * K_xy.mean())


def compute_earth_movers_distance(set1, set2):
    if len(set1) == 0 or len(set2) == 0:
        return float('inf')  # Undefined if one of the sets is empty.
    set1 = torch.tensor(set1, dtype=torch.float).unsqueeze(1)
    set2 = torch.tensor(set2, dtype=torch.float).unsqueeze(1)
    cost_matrix = torch.cdist(set1, set2, p=2).cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    emd = cost_matrix[row_ind, col_ind].sum()
    return emd

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

def train_msvae(model, dataloader, num_epochs, learning_rate, weights):
    optimizer = Adam(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(num_epochs):
        print("Traininig Multiset-VAE iteration ", epoch)
        total_loss = 0
        total_eg_loss = 0
        for batch  in dataloader:
            X = batch[0]
            optimizer.zero_grad()
            frequencies, mean, logvar = model(X)
            # Compute the loss
            loss = loss_function(X, frequencies, mean, logvar, weights, epoch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch Multiset-VAE [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(dataloader):.4f}")

def eg_loss(frequencies):
    indices = torch.arange(1, N+1).float().to(frequencies.device)

    deg_sum = torch.sum(frequencies * indices, dim=2)
    sum_modulo_loss = (1 - torch.cos(np.pi * deg_sum)) / 2

    eg_inequality_loss = 0
    for k in range(1, N+1):
        lhs = torch.sum(frequencies[:, :, :k] * indices[:k], dim=2)
        rhs = k * (k-1) + torch.sum(torch.minimum(indices[k:], torch.tensor(k, device=frequencies.device)) * frequencies[:, :, k:], dim=2)
        eg_inequality_loss += F.relu(lhs - rhs)

    return torch.sum(sum_modulo_loss+ eg_inequality_loss)

def loss_function(target_freq_vec, frequencies,mean, logvar, weights, epoch):
    recon_weight, kl_weight, erdos_gallai_weight = weights.get('reconstruction', 1.0), weights.get('kl_divergence', 1.0),weights.get('erdos_gallai', 1.0)
    recon_weight = max(0.1, recon_weight * (0.95 ** epoch))
    kl_weight = min(kl_weight, epoch / 10)  
    recon_loss = torch.sum( (target_freq_vec - frequencies)**2)
    kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    erdos_gallai_loss = eg_loss(frequencies)
    total_loss = recon_weight * recon_loss + kl_weight * kl_loss + erdos_gallai_weight * erdos_gallai_loss
    return total_loss

def evaluate_multisets_distance(source_tensor, target_tensor):
    source_seqs = [decode_degree_sequence(tensor) for tensor in source_tensor]
    target_seqs = [decode_degree_sequence(tensor) for tensor in target_tensor]
    chamfer_distances = [compute_chamfer_distance(s, t) for s in source_seqs for t in target_seqs]
    avg_chamfer_distance = sum(chamfer_distances) / len(chamfer_distances) if chamfer_distances else float('inf')
    emd_distances = [compute_earth_movers_distance(s, t) for s in source_seqs for t in target_seqs]
    avg_emd_distance = sum(emd_distances) / len(emd_distances) if emd_distances else float('inf')
    source_hist = compute_degree_histograms(source_seqs,N)
    target_hist = compute_degree_histograms(target_seqs,N)
    kl = [compute_kl_divergence(s, t) for s in source_hist for t in target_hist]
    avg_kl = sum(kl) / len(kl) if kl else float('inf')
    mmd = compute_mmd(source_hist, target_hist)
    return {
        "Chamfer Distance": avg_chamfer_distance,
        "Earth Mover's Distance": avg_emd_distance,
        "KL Distance": avg_kl,
        "MMD": mmd
    }


def evaluate_generated_multisets(generated_tensor):
    generated_seqs = [decode_degree_sequence(tensor) for tensor in generated_tensor]
    validity_checks = [check_sequence_validity(gen_seq) for gen_seq in generated_seqs]
    degree_validities = [result for result, code in validity_checks if result]
    error_codes = [code for result, code in validity_checks if not result]
    error_count = Counter(error_codes)
    empty_degree = [1] * error_count.get(1, 0)
    odd_sum_degree = [1] * error_count.get(2, 0)
    invalid_degree = [1] * error_count.get(3, 0)
    validity_percentage = (sum(degree_validities) / len(validity_checks)) * 100 if len(validity_checks) > 0 else 0
    empty_percentage = (sum(empty_degree) / len(validity_checks)) * 100 if len(validity_checks) > 0 else 0
    odd_percentage = (sum(odd_sum_degree) / len(validity_checks)) * 100 if len(validity_checks) > 0 else 0
    invalidity_percentage = (sum(invalid_degree) / len(validity_checks)) * 100 if len(validity_checks) > 0 else 0
    return {
        "Degree Validity (%)": validity_percentage,
        "Degree Empty (%)": empty_percentage,
        "Degree Odd (%)": odd_percentage,
        "Degree Invalidity (%)": invalidity_percentage
    }

def main(args):
    config_dir = Path("configs")
    dataset_dir = Path("datasets")
    model_dir = Path("models")
    config_file = config_dir / args.config_file
    config = toml.load(config_file)
    dataset_dir = dataset_dir / args.dataset_dir
    batch_size = config['training']['batch_size']
    tensor = load_degree_sequence_from_directory(dataset_dir)
    train_tensor, test_tensor = train_test_split(tensor, test_size=0.2, random_state=42)
    train_dataset = TensorDataset(torch.stack(train_tensor))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    hidden_dim = config['training']['hidden_dim']
    latent_dim = config['training']['latent_dim']
    input_dim = N  # input dimension matches one-hot encoded degrees
    model = MSVAE(max_input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
    if args.input_model:
        model.load_model(model_dir / args.input_model)
        print(f"Model loaded from {args.input_model}")
    else:
        num_epochs = config['training']['num_epochs']
        learning_rate = config['training']['learning_rate']
        weights = config['training']['weights']
        train_msvae(model, train_dataloader, num_epochs, learning_rate, weights)
    if args.output_model:
        model.save_model(model_dir / args.output_model)
        print(f"Model saved to {args.output_model}")
    if args.evaluate:
        model.eval()
        generated_tensor = model.generate(config['inference']['generate_samples'])
        print(f"Evaluate generated multiset")
        metrics =  evaluate_generated_multisets(generated_tensor)
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        print(f"Evaluate generated multiset to training data")
        evaluation_metrics = evaluate_multisets_distance(train_tensor,generated_tensor)
        print(evaluation_metrics)
        for metric, value in evaluation_metrics.items():
            print(f"{metric}: {value:.4f}")
        print(f"Evaluate generated multiset to test data")
        evaluation_metrics = evaluate_multisets_distance(test_tensor,generated_tensor)
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

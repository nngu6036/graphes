import torch
import torch.nn as nn
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

from model_stdvae import StdVAE
from utils import compute_chamfer_distance, compute_degree_histograms, compute_kl_divergence, compute_mmd, compute_earth_movers_distance


def encode_degree_sequence(degree_sequence , max_class):
    degree_sequence += [0] * (max_class - len(degree_sequence))
    return torch.tensor(degree_sequence).float()

def decode_degree_sequence(tensor):
    tensor = tensor.long()
    non_zero_tensor = tensor[tensor != 0]
    return non_zero_tensor.tolist()

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
            seq = encode_degree_sequence([deg for _, deg in graph.degree()],max_node)
            if seq is not None:
                seqs.append(seq)
    return torch.stack(seqs), max_node

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

def train_stdvae(model, dataloader, num_epochs, learning_rate, weights, warmup_epochs,max_node):
    optimizer = Adam(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(num_epochs):
        print("Traininig Std-VAE iteration ", epoch)
        total_loss = 0
        for batch  in dataloader:
            X = batch[0]
            optimizer.zero_grad()
            logits,mean, logvar = model(X)
            # Compute the loss
            loss = loss_function(X, logits, mean, logvar, weights,warmup_epochs, epoch, max_node)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch Std-VAE [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(dataloader):.4f}")

def loss_function(target_freq, logits,mean, logvar, weights,warmup_epochs, epoch,max_node):
    logits_flat = logits.view(-1, logits.size(-1))        # shape (B×D, N)
    targets_flat = target_freq.long().view(-1)                    # shape (B×D,)
    recon_loss = F.cross_entropy(logits_flat, targets_flat, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    total_loss = (weights['reconstruction'] * recon_loss +
                  weights['kl_divergence'] * kl_loss )
    return total_loss

def evaluate_multisets_distance(source_tensor, target_tensor,max_node):
    source_seqs = [decode_degree_sequence(tensor) for tensor in source_tensor]
    target_seqs = [decode_degree_sequence(tensor) for tensor in target_tensor]
    chamfer_distances = [compute_chamfer_distance(s, t) for s in source_seqs for t in target_seqs]
    avg_chamfer_distance = sum(chamfer_distances) / len(chamfer_distances) if chamfer_distances else float('inf')
    emd_distances = [compute_earth_movers_distance(s, t) for s in source_seqs for t in target_seqs]
    avg_emd_distance = sum(emd_distances) / len(emd_distances) if emd_distances else float('inf')
    source_hist = compute_degree_histograms(source_seqs,max_node)
    target_hist = compute_degree_histograms(target_seqs,max_node)
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
    dataset_dir = Path("datasets") / args.dataset_dir
    model_dir = Path("models")
    config_file = config_dir / args.config_file
    config = toml.load(config_file)
    batch_size = config['training']['batch_size']
    tensor, max_node = load_degree_sequence_from_directory(dataset_dir)
    train_tensor, test_tensor = train_test_split(tensor, test_size=0.2, random_state=42)
    train_dataset = TensorDataset(train_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    hidden_dim = config['training']['hidden_dim']
    latent_dim = config['training']['latent_dim']
    model = StdVAE(max_input_dim=max_node, hidden_dim=hidden_dim, latent_dim=latent_dim)
    if args.input_model:
        model.load_model(model_dir / args.input_model)
        print(f"Model loaded from {args.input_model}")
    else:
        num_epochs = config['training']['num_epochs']
        learning_rate = config['training']['learning_rate']
        warmup_epochs = config['training']['warmup_epochs']
        weights = config['training']['weights']
        train_stdvae(model, train_dataloader, num_epochs, learning_rate, weights,warmup_epochs, max_node)
    if args.output_model:
        model.save_model(model_dir / args.output_model)
        print(f"Model saved to {args.output_model}")
    if args.evaluate:
        model.eval()
        generated_tensor = model.generate(config['inference']['generate_samples'])
        print(f"Evaluate generated degree sequence")
        metrics =  evaluate_generated_multisets(generated_tensor)
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        print(f"Evaluate baseline: train <-> test")
        evaluation_metrics = evaluate_multisets_distance(train_tensor,test_tensor,max_node)
        for metric, value in evaluation_metrics.items():
            print(f"{metric}: {value:.4f}")
        print(f"Evaluate fit: train <-> generated")
        evaluation_metrics = evaluate_multisets_distance(train_tensor,generated_tensor,max_node)
        for metric, value in evaluation_metrics.items():
            print(f"{metric}: {value:.4f}")
        print(f"Evaluate fit: test <-> generated")
        evaluation_metrics = evaluate_multisets_distance(test_tensor,generated_tensor,max_node)
        for metric, value in evaluation_metrics.items():
            print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Standard-VAE for Graph Generation')
    parser.add_argument('--dataset-dir', type=str, help='Path to the directory containing graph files')
    parser.add_argument('--config-file', type=str, required=True, help='Path to the configuration file in TOML format')
    parser.add_argument('--output-model', type=str, help='Path to save the trained model')
    parser.add_argument('--input-model', type=str, help='Path to load a pre-trained model')
    parser.add_argument('--evaluate', action='store_true', help='Whether we evaluate the model')
    args = parser.parse_args()
    main(args)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
import os
import argparse
import toml
import math
import random
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

from model_msvae import MSVAE
from model_grapher import GraphER
from utils import graph_to_data,compute_mmd_degree_emd,compute_mmd_cluster, compute_mmd_orbit

def graph_structure_loss(G_orig, G_corrupted):
    A_orig = nx.to_numpy_array(G_orig)
    A_corrupt = nx.to_numpy_array(G_corrupted)
    # Resize if node sets differ (padding)
    max_size = max(A_orig.shape[0], A_corrupt.shape[0])
    A_pad = lambda A: np.pad(A, ((0, max_size - A.shape[0]), (0, max_size - A.shape[1])), constant_values=0)
    A_orig, A_corrupt = A_pad(A_orig), A_pad(A_corrupt)
    # Binary matrix difference
    diff = torch.tensor(np.abs(A_orig - A_corrupt), dtype=torch.float32)
    return diff.mean()

def rewire_edges(G, num_rewirings):
    edges = list(G.edges())
    last_rewired_pair = None
    for _ in range(num_rewirings):
        e1, e2 = random.sample(edges, 2)
        u, v = e1
        x, y = e2
        if len({u, v, x, y}) == 4:
            if not G.has_edge(u, x) and not G.has_edge(v, y):
                G.remove_edges_from([e1, e2])
                G.add_edges_from([(u, x), (v, y)])
                last_rewired_pair = (e1, e2)
            elif not G.has_edge(u, y) and not G.has_edge(v, x):
                G.remove_edges_from([e1, e2])
                G.add_edges_from([(u, y), (v, x)])
                last_rewired_pair = (e1, e2)
        edges = list(G.edges())
    return G, last_rewired_pair

def load_graph_from_directory(directory_path):
    max_node = 0 
    graphs = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            graph = nx.read_edgelist(file_path, nodetype=int)
            max_node = max(max_node, graph.number_of_nodes())
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            graph = nx.read_edgelist(file_path, nodetype=int)
            graph = nx.convert_node_labels_to_integers(graph)
            graphs.append(graph)
    return graphs, max_node


def train_grapher(model, graphs, num_epochs, learning_rate, max_node, T, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for G in graphs:
            batch_loss = 0.0
            num_rewirings = random.randint(1, T)
            G_corrupted, last_rewired_pair = rewire_edges(G.copy(), num_rewirings)
            (u,v), (x,y) = last_rewired_pair
            data = graph_to_data(G_corrupted).to(device)
            candidate_edges = [e for e in G_corrupted.edges()]
            positive_edges = {frozenset((u, x)), frozenset((u, y))}
            labels = torch.tensor([1.0 if frozenset((s, t)) in positive_edges else 0.0 for (s, t) in candidate_edges])
            scores = model(data.x, data.edge_index, last_rewired_pair, candidate_edges, t=num_rewirings)
            loss = criterion(scores.squeeze(), labels)
            batch_loss += loss
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            epoch_loss += batch_loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")


def loss_function(target_freq_vec, frequencies,mean, logvar, weights, epoch,max_node):
    recon_weight, kl_weight, erdos_gallai_weight = weights.get('reconstruction', 1.0), weights.get('kl_divergence', 1.0),weights.get('erdos_gallai', 1.0)
    recon_weight = max(0.1, recon_weight * (0.95 ** epoch))
    kl_weight = min(kl_weight, epoch / 10)  
    recon_loss = torch.sum( (target_freq_vec - frequencies)**2)
    kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    erdos_gallai_loss = eg_loss(frequencies, max_node)
    total_loss = recon_weight * recon_loss + kl_weight * kl_loss + erdos_gallai_weight * erdos_gallai_loss
    return total_loss


def evaluate_generated_graphs(generated_graphs, test_graphs):
    mmd_degree = compute_mmd_degree(generated_graphs, test_graphs)
    mmd_cluster = compute_mmd_cluster(generated_graphs, test_graphs)
    mmd_orbit = compute_mmd_orbit(generated_graphs, test_graphs)
    return {
        "MMD Degree": mmd_degree,
        "MMD Cluster": mmd_cluster,
        "MMD Orbit": mmd_orbit,
    }

def load_msvae_from_file(max_node,config, model_path):
    hidden_dim = config['training']['hidden_dim']
    latent_dim = config['training']['latent_dim']
    model = MSVAE(max_input_dim=max_node, hidden_dim=hidden_dim, latent_dim=latent_dim, max_frequency = max_node)
    model.load_model(model_path)
    return model

def main(args):
    config_dir = Path("configs")
    dataset_dir = Path("datasets") / args.dataset_dir
    model_dir = Path("models")
    config_file = config_dir / args.config_file
    msvae_config_file = config_dir / args.msvae_config_file
    msvae_model_file = model_dir / args.msvae_model
    config = toml.load(config_file)
    msvae_config = toml.load(msvae_config_file)
    graphs, max_node = load_graph_from_directory(dataset_dir)
    train_graphs, test_graphs = train_test_split(graphs, test_size=0.2, random_state=42)
    msvae_model  = load_msvae_from_file(max_node, config,msvae_model_file)
    hidden_dim = config['training']['hidden_dim']
    latent_dim = config['training']['latent_dim']
    num_layer = config['training']['num_layer']
    model = GraphER(in_channels=1, hidden_dim=hidden_dim,num_layer=num_layer)
    if args.input_model:
        model.load_model(model_dir / args.input_model)
        print(f"Model Graph-ER loaded from {args.input_model}")
    else:
        num_epochs = config['training']['num_epochs']
        learning_rate = config['training']['learning_rate']
        T = config['training']['T']
        train_grapher(model, train_graphs, num_epochs, learning_rate, max_node,T, 'cpu')
    if args.output_model:
        model.save_model(model_dir / args.output_model)
        print(f"Model saved to {args.output_model}")
    if args.evaluate:
        model.eval()
        if args.ablation:
            generated_graphs = model.generate(config['inference']['generate_samples'],T,msvae_model = None)
        else:
            generated_graphs = model.generate(config['inference']['generate_samples'],T,msvae_model = msvae_model)
        print(f"Evaluate generated graphs")
        metrics =  evaluate_generated_graphs(generated_graphs, test_graphs)
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GRAPH-ER Model')
    parser.add_argument('--dataset-dir', type=str, required=True,help='Path to the directory containing graph files')
    parser.add_argument('--config-file', type=str, required=True, help='Path to the configuration file in TOML format of Graph-ER')
    parser.add_argument('--msvae-config-file', type=str, required=True, help='Path to the configuration file in TOML format of MS-VAE')
    parser.add_argument('--msvae-model', type=str, required=True,help='Path to load a pre-trained MS-VAE model')
    parser.add_argument('--output-model', type=str, help='Path to save the trained model')
    parser.add_argument('--input-model', type=str, help='Path to load a pre-trained model')
    parser.add_argument('--evaluate', action='store_true', help='Whether we evaluate the model')
    parser.add_argument('--ablation', action='store_true', help='Whether to run ablation study')
    args = parser.parse_args()
    main(args)

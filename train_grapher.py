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
from eval import DegreeSequenceEvaluator, GraphsEvaluator
from utils import *

def count_edge_triangles(G, u, v):
    """Return number of triangles that edge (u,v) participates in."""
    neighbors_u = set(G.neighbors(u))
    neighbors_v = set(G.neighbors(v))
    return len(neighbors_u & neighbors_v)

def rewire_edges(G):
    edges = list(G.edges())
    removed_pair = None
    added_pair = None
    e1, e2 = random.sample(edges, 2)
    u, v = e1
    x, y = e2
    if len({u, v, x, y}) != 4:
        return G, None, None
    triangle_removed = count_edge_triangles(G, u, v) + count_edge_triangles(G, x, y)
    # Option 1: (u,x), (v,y)
    if not G.has_edge(u, x) and not G.has_edge(v, y):
        tri_added = count_common_neighbors(G, u, x) + count_common_neighbors(G, v, y)
        if tri_added >= triangle_removed:
            removed_pair = ((u,v), (x,y))
            added_pair = ((u, x), (v, y))
            G.remove_edges_from(removed_pair)
            G.add_edges_from(added_pair)
            return G, removed_pair, added_pair
    # Option 2: (u,y), (v,x)
    if not G.has_edge(u, y) and not G.has_edge(v, x):
        tri_added = count_common_neighbors(G, u, y) + count_common_neighbors(G, v, x)
        if tri_added >= triangle_removed:
            removed_pair = ((u,v), (x,y))
            added_pair = [(u, y), (v, x)]
            G.remove_edges_from(removed_pair)
            G.add_edges_from(added_pair)
            return G, removed_pair, added_pair
    return G, None, None

def count_common_neighbors(G, a, b):
    """Return number of common neighbors of nodes a and b."""
    return len(set(G.neighbors(a)) & set(G.neighbors(b)))

def train_grapher(model, graphs, num_epochs, learning_rate, T, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for G in graphs:
            # --- Corrupt graph with t edge rewirings ---
            G_prev = G
            max_rewiring = random.randint(1, T)
            timestep = 0
            for _ in range(max_rewiring):
                G_next, removed_pair, added_pair = rewire_edges(G_prev.copy())
                if not removed_pair or not added_pair:
                    continue
                timestep +=1
                # --- Define anchor and target edge ---
                first_edge_added, second_edge_added = added_pair  # predict second_edge_added given first_edge_added
                # --- Graph to PyG format ---
                data = graph_to_data(G_next).to(device)
                # --- Edge candidates from corrupted graph ---
                u, v = first_edge_added
                candidate_edges = [
                    e for e in G_next.edges()
                    if len(set(e + (u, v))) == 4  # disjoint
                ]
                # --- Construct binary labels ---
                labels = torch.tensor(
                    [1.0 if frozenset(edge) == frozenset(second_edge_added) else 0.0 for edge in candidate_edges],
                    dtype=torch.float32,
                    device=device
                )
                # --- Forward pass ---
                scores = model(data.x, data.edge_index, first_edge_added, candidate_edges, t=timestep)
                loss = criterion(scores.squeeze(), labels)
                # --- Backpropagation ---
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                G_prev = G_next
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")


def load_msvae_from_file(max_node,config, model_path):
    hidden_dim = config['training']['hidden_dim']
    latent_dim = config['training']['latent_dim']
    model = MSVAE(max_input_dim=max_node, hidden_dim=hidden_dim, latent_dim=latent_dim, max_frequency = max_node)
    print(f"MS-VAE Model loaded from {model_path}")
    model.load_model(model_path)
    return model

def main(args):
    config_dir = Path("configs")
    dataset_dir = Path("datasets") / args.dataset_dir
    model_dir = Path("models")
    config = toml.load(config_dir / args.config)
    msvae_config = toml.load(config_dir / args.msvae_config)
    graphs, max_node = load_graph_from_directory(dataset_dir)
    print(f"Loading graphs dataset {len(graphs)}")
    train_graphs, test_graphs = train_test_split(graphs, test_size=0.2, random_state=42)
    msvae_model  = load_msvae_from_file(max_node, msvae_config, model_dir /args.msvae_model)
    hidden_dim = config['training']['hidden_dim']
    num_layer = config['training']['num_layer']
    T = config['training']['T']
    model = GraphER(1, hidden_dim,num_layer,T)
    if args.input_model:
        model.load_model(model_dir / args.input_model)
        print(f"Model Graph-ER loaded from {args.input_model}")
    else:
        num_epochs = config['training']['num_epochs']
        learning_rate = config['training']['learning_rate']
        train_grapher(model, train_graphs,num_epochs, learning_rate,T, 'cpu')
    if args.output_model:
        model.save_model(model_dir / args.output_model)
        print(f"Model saved to {args.output_model}")
    if args.evaluate:
        graph_eval = GraphsEvaluator()
        if args.ablation:
            sample_graphs = random.sample(train_graphs,min(len(train_graphs),config['inference']['generate_samples']))
            degree_sequences = [[deg for _, deg in graph.degree()] for graph in sample_graphs]
            generated_graphs = model.generate_without_msvae(T,degree_sequences)
            print(f"Evaluate generated graphs sampled from training")
            print(f"MMD Degree: {graph_eval.compute_mmd_degree_emd(test_graphs,generated_graphs,max_node)}")
            print(f"MMD Clustering Coefficient: {graph_eval.compute_mmd_cluster(test_graphs,generated_graphs)}")
            print(f"MMD Orbit count: {graph_eval.compute_mmd_orbit(test_graphs,generated_graphs)}")
        else:
            generated_graphs, generated_seqs = model.generate(config['inference']['generate_samples'],T, msvae_model = msvae_model)
            print(f"Evaluate generated graphs using Configuraiton Model")
            print(f"MMD Degree: {graph_eval.compute_mmd_degree_emd(test_graphs,generated_graphs,max_node)}")
            print(f"MMD Clustering Coefficient: {graph_eval.compute_mmd_cluster(test_graphs,generated_graphs)}")
            print(f"MMD Orbit count: {graph_eval.compute_mmd_orbit(test_graphs,generated_graphs)}")
            deg_eval = DegreeSequenceEvaluator()
            test_seqs = [[deg for _, deg in graph.degree()] for graph in test_graphs ]
            print(f"KL Distance: {deg_eval.evaluate_multisets_kl_distance(test_seqs,generated_seqs,max_node)}")
            print(f"MMD Distance: {deg_eval.evaluate_multisets_mmd_distance(test_seqs,generated_seqs,max_node)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GRAPH-ER Model')
    parser.add_argument('--dataset-dir', type=str, required=True,help='Path to the directory containing graph files')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file in TOML format of Graph-ER')
    parser.add_argument('--msvae-config', type=str, required=True, help='Path to the configuration file in TOML format of MS-VAE')
    parser.add_argument('--msvae-model', type=str, required=True,help='Path to load a pre-trained MS-VAE model')
    parser.add_argument('--output-model', type=str, help='Path to save the trained model')
    parser.add_argument('--input-model', type=str, help='Path to load a pre-trained model')
    parser.add_argument('--evaluate', action='store_true', help='Whether we evaluate the model')
    parser.add_argument('--ablation', action='store_true', help='Whether to run ablation study')
    args = parser.parse_args()
    main(args)

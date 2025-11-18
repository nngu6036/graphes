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


def train_grapher(
    model,
    graphs,
    num_epochs,
    learning_rate,
    num_rewirings,
    dist_name,
    k_eigen,
    lambda_rec: float = 0.5  # weight for embedding reconstruction loss
):
    """
    Training with edge-pair BCE loss + embedding reconstruction loss (MSE).

    Reconstruction target = HH(G) embedding.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    model.train()

    graphs = [(G, hh_graph_from_G(G)) for G in graphs]

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for G, G_hh in graphs:
            H_data = graph_to_data(G_hh, k_eigen)
            with torch.no_grad():
                H_repr = model.encode_graph(H_data.x, H_data.edge_index).detach()  # [1, hidden_dim]

            # Forward diffusion trajectory (stochastic rewiring toward HH)
            lambda_dist = make_lambda_dist(dist_name, G_hh)

            # Build trajectory (list of (G_after_rewire, added_pair))
            traj = transform_to_hh_via_guided_rewiring(G, G_hh, lambda_dist, G.number_of_edges())
            for step_idx, (G_post, added_pair) in enumerate(traj, start=1):
                (a, b), (c, d) = added_pair
                anchor      = (a, b)
                pos_partner = (c, d)

                # Current graph features
                data = graph_to_data(G_post, k_eigen)

                # Candidate edges: disjoint from anchor, from CURRENT graph
                candidate_edges = build_candidates(G_post,anchor, ensure_connected=True,k_hop=2)

                if not candidate_edges:
                    continue

                labels = torch.tensor(
                    [1.0 if frozenset(edge) == frozenset(pos_partner) else 0.0 for edge in candidate_edges],
                    dtype=torch.float32
                )

                # Edge-pair prediction loss
                scores = model(data.x, data.edge_index, anchor, candidate_edges, t=step_idx).squeeze()
                try:
                    loss_edge = bce(scores, labels)
                except:
                    import pdb
                    pdb.set_trace()
                # Embedding reconstruction loss (current → HH)
                G_repr = model.encode_graph(data.x, data.edge_index)  # [1, hidden_dim]
                loss_rec = mse(G_repr, H_repr)

                loss = loss_edge + lambda_rec * loss_rec

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}  Loss: {epoch_loss:.4f}")


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
    k_eigen = config['training']['k_eigen']
    dist_name = config['training']['dist_name']
    model = GraphER(k_eigen, hidden_dim,num_layer,T)
    if args.input_model:
        model.load_model(model_dir / args.input_model)
        print(f"Model Graph-ER loaded from {args.input_model}")
    else:
        num_epochs = config['training']['num_epochs']
        learning_rate = config['training']['learning_rate']
        train_grapher(model, train_graphs,num_epochs, learning_rate,T, dist_name,k_eigen)
    if args.output_model:
        model.save_model(model_dir / args.output_model)
        print(f"Model saved to {args.output_model}")
    if args.evaluate:
        graph_eval = GraphsEvaluator()
        deg_eval = DegreeSequenceEvaluator()
        sample_graphs = random.choices(train_graphs,k=config['inference']['generate_samples'])
        test_seqs = [[deg for _, deg in graph.degree()] for graph in test_graphs ]
        degree_sequences = [[deg for _, deg in graph.degree()] for graph in sample_graphs]
    
        generated_graphs, generated_seqs = model.generate(config['inference']['generate_samples'],T, msvae_model,k_eigen,method = 'havei_hakimi')
        print(f"Evaluate generated graphs using Havei Hamimi Model and MS-VAE")
        print(f"MMD Degree: {graph_eval.compute_mmd_degree_emd(test_graphs,generated_graphs,max_node)}")
        print(f"MMD Clustering Coefficient: {graph_eval.compute_mmd_cluster(test_graphs,generated_graphs)}")
        print(f"MMD Orbit count: {graph_eval.compute_mmd_orbit(test_graphs,generated_graphs)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GRAPH-ER Model')
    parser.add_argument('--dataset-dir', type=str, required=True,help='Path to the directory containing graph files')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file in TOML format of Graph-ER')
    parser.add_argument('--msvae-config', type=str, required=True, help='Path to the configuration file in TOML format of MS-VAE')
    parser.add_argument('--msvae-model', type=str, required=True,help='Path to load a pre-trained MS-VAE model')
    parser.add_argument('--output-model', type=str, help='Path to save the trained model')
    parser.add_argument('--input-model', type=str, help='Path to load a pre-trained model')
    parser.add_argument('--evaluate', action='store_true', help='Whether we evaluate the model')
    args = parser.parse_args()
    main(args)

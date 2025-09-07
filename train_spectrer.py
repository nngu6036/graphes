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
from model_setvae import SetVAE
from model_spectraler import SpectralER

from eval import DegreeSequenceEvaluator, GraphsEvaluator
from utils import *


def train_spectral(model, graphs, num_epochs, learning_rate, T, k_eigen,device):
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for G in graphs:
            G_hh = hh_graph_from_G(G)
            # --- Corrupt graph with t edge rewirings ---
            traj = transform_to_hh_via_stochastic_rewiring(G, G_hh, G.number_of_edges())
            step = 0
            for G_t, added_pair, removed_pair in traj:
                step += 1
                # Undo the last swap to get G_{t-1}
                G_prev = G_t.copy()
                (u, v), (x, y) = removed_pair
                (a, b), (c, d) = added_pair
                if G_prev.has_edge(a, b): G_prev.remove_edge(a, b)
                if G_prev.has_edge(c, d): G_prev.remove_edge(c, d)
                G_prev.add_edge(u, v); G_prev.add_edge(x, y)

                lam_t, _    = laplacian_eigs(G_t,   k_eigen, normed=True)
                lam_prev, _ = laplacian_eigs(G_prev, k_eigen, normed=True)
                lam_t   = torch.from_numpy(lam_t).to(device)
                lam_t_1 = torch.from_numpy(lam_prev).to(device)

                # size features
                n = G_t.number_of_nodes()
                m_edges = G_t.number_of_edges()
                avg_deg = (2.0 * m_edges) / max(1, n)
                density = (2.0 * m_edges) / max(1, n * (n - 1))
                extra_feat = torch.tensor([math.log(max(n, 2)), avg_deg, density],
                                          device=device, dtype=lam_t.dtype)

                mu, logvar = model(lam_t, step, extra_feat)

                # Masked diagonal-Gaussian NLL over valid eigenvalues
                m_valid = min(k_eigen, max(0, n - 1))
                mask = torch.zeros_like(lam_t_1)
                if m_valid > 0:
                    mask[:m_valid] = 1.0
                diff = lam_t_1 - mu
                nll_vec = 0.5 * (diff.pow(2) * torch.exp(-logvar) + logvar)
                nll = (nll_vec * mask).sum() / mask.sum().clamp_min(1.0)

                opt.zero_grad()
                nll.backward()
                opt.step()
                epoch_loss += float(nll.item())
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")


def load_msvae_from_file(max_node,config, model_path):
    hidden_dim = config['training']['hidden_dim']
    latent_dim = config['training']['latent_dim']
    model = MSVAE(max_input_dim=max_node, hidden_dim=hidden_dim, latent_dim=latent_dim, max_frequency = max_node-1)
    print(f"MS-VAE Model loaded from {model_path}")
    model.load_model(model_path)
    return model


def load_setvae_from_file(max_node,config, model_path):
    hidden_dim = config['training']['hidden_dim']
    latent_dim = config['training']['latent_dim']
    model = SetVAE(hidden_dim=hidden_dim, latent_dim=latent_dim, max_degree = max_node-1)
    print(f"Set-VAE Model loaded from {model_path}")
    model.load_model(model_path)
    return model

def main(args):
    msvae_model, setvae_model = None, None
    config_dir = Path("configs")
    dataset_dir = Path("datasets") / args.dataset_dir
    model_dir = Path("models")
    config = toml.load(config_dir / args.config)

    graphs, max_node, min_node = load_graph_from_directory(dataset_dir)
    print(f"Loading graphs dataset {len(graphs)}")

    train_graphs, test_graphs = train_test_split(graphs, test_size=0.2, random_state=42)

    if args.msvae_model:
        msvae_config = toml.load(config_dir / args.msvae_config)
        msvae_model  = load_msvae_from_file(max_node, msvae_config, model_dir / args.msvae_model)

    if args.setvae_model:
        setvae_config = toml.load(config_dir / args.setvae_config)
        setvae_model  = load_setvae_from_file(max_node, setvae_config, model_dir / args.setvae_model)

    hidden_dim = config['training']['hidden_dim']
    T         = config['training']['T']
    k_eigen   = config['data']['k_eigen']

    # FIX: correct ctor args (k, hidden, T)
    model = SpectralER(k_eigen, hidden_dim, T)

    if args.input_model:
        model.load_model(model_dir / args.input_model)
        print(f"SpectralER model loaded from {args.input_model}")
    else:
        num_epochs    = config['training']['num_epochs']
        learning_rate = config['training']['learning_rate']
        train_spectral(model, train_graphs,num_epochs, learning_rate,T, k_eigen,'cpu')

    if args.output_model:
        model.save_model(model_dir / args.output_model)
        print(f"Model saved to {args.output_model}")

    if args.evaluate:
        graph_eval = GraphsEvaluator()
        deg_eval = DegreeSequenceEvaluator()
        test_seqs = [[deg for _, deg in g.degree()] for g in test_graphs]

        sample_graphs = random.choices(train_graphs,k=config['inference']['generate_samples'])
        degree_sequences = [[deg for _, deg in graph.degree()] for graph in sample_graphs]

        if msvae_model:
            # how many samples to generate
            num_gen = config['inference']['generate_samples']
            generated_graphs, generated_seqs = model.generate_with_msvae(
                num_gen, T, msvae_model, k_eigen, method='havel_hakimi'
            )
            print(f"Evaluate generated graphs (MS-VAE + Havel–Hakimi init)")
            print(f"MMD Degree: {graph_eval.compute_mmd_degree_emd(test_graphs, generated_graphs, max_node)}")
            print(f"MMD Clustering Coefficient: {graph_eval.compute_mmd_cluster(test_graphs, generated_graphs)}")
            print(f"MMD Orbit count: {graph_eval.compute_mmd_orbit(test_graphs, generated_graphs)}")
            print(f"KL Distance: {deg_eval.evaluate_multisets_kl_distance(test_seqs, generated_seqs, max_node)}")
            print(f"MMD Distance: {deg_eval.evaluate_multisets_mmd_distance(test_seqs, generated_seqs, max_node)}")

        if setvae_model:
            N_nodes = [G.number_of_nodes() for G in random.choices(train_graphs, k=config['inference']['generate_samples'])]
            generated_graphs, generated_seqs = model.generate_with_setvae(N_nodes, T, setvae_model, k_eigen, method='havel_hakimi')
            print(f"Evaluate generated graphs using Havei Hamimi Model and Set-VAE")
            print(f"MMD Degree: {graph_eval.compute_mmd_degree_emd(test_graphs,generated_graphs,max_node)}")
            print(f"MMD Clustering Coefficient: {graph_eval.compute_mmd_cluster(test_graphs,generated_graphs)}")
            print(f"MMD Orbit count: {graph_eval.compute_mmd_orbit(test_graphs,generated_graphs)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GRAPH-ER Model')
    parser.add_argument('--dataset-dir', type=str, required=True,help='Path to the directory containing graph files')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file in TOML format of Graph-ER')
    parser.add_argument('--msvae-config', type=str, help='Path to the configuration file in TOML format of MS-VAE')
    parser.add_argument('--msvae-model', type=str,help='Path to load a pre-trained MS-VAE model')
    parser.add_argument('--setvae-config', type=str,help='Path to the configuration file in TOML format of Set-VAE')
    parser.add_argument('--setvae-model', type=str,help='Path to load a pre-trained Set-VAE model')
    parser.add_argument('--output-model', type=str, help='Path to save the trained model')
    parser.add_argument('--input-model', type=str, help='Path to load a pre-trained model')
    parser.add_argument('--evaluate', action='store_true', help='Whether we evaluate the model')
    args = parser.parse_args()
    main(args)

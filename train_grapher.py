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
from model_grapher import GraphER
from eval import DegreeSequenceEvaluator, GraphsEvaluator
from utils import *

def _orientation_label(anchor, partner, removed_pair):
    (a,b), (c,d) = anchor, partner
    rem = {frozenset(removed_pair[0]), frozenset(removed_pair[1])}
    if {frozenset((a,c)), frozenset((b,d))} == rem: return 0
    if {frozenset((a,d)), frozenset((b,c))} == rem: return 1
    raise AssertionError(f"Bad orientation mapping: {anchor=} {partner=} {removed_pair=}")

def train_grapher(model, graphs, num_epochs, learning_rate, T, k_eigen, device,
                  lambda_kernel: float = 0.1, t_heat: float = 0.5):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for G in graphs:
            G_hh = hh_graph_from_G(G)
            # ---- Compute trajector from G to G_hh (Havel-Hakimi graph)
            traj = transform_to_hh_via_stochastic_rewiring(G, G_hh, G.number_of_edges())
            for step_idx, (G_pre, removed_pair, added_pair) in enumerate(traj, start=1):
                (a,b), (c,d) = removed_pair
                anchor      = (a, b)
                pos_partner = (c, d)

                # Build LOCAL candidate partners within k hops of the anchor.
                k_local = getattr(model, "partner_k_hop", 2)
                cand_edges = local_partner_candidates(G_pre, anchor, k_local)
                # Ensure the positive partner is present even if it lies outside k-hop
                if frozenset(pos_partner) not in {frozenset(e) for e in cand_edges}:
                    cand_edges.append(pos_partner)

                # partner label index
                pos_idx = None
                for i, e in enumerate(cand_edges):
                    if frozenset(e) == frozenset(pos_partner):
                        pos_idx = i; break
                if pos_idx is None:
                    continue

                orient_y = _orientation_label(anchor, pos_partner, added_pair)

                # forward
                data = graph_to_data(G_pre, k_eigen)
                x, edge_index = data.x.to(device), data.edge_index.to(device)
                partner_logits, orient_logits = model(
                    x=x, edge_index=edge_index,
                    first_edge=anchor, candidate_edges=cand_edges, t=step_idx
                )   # shapes: (C,), (C,2)

                # losses
                target_idx = torch.tensor([pos_idx], dtype=torch.long, device=device)
                loss_partner = F.cross_entropy(partner_logits.unsqueeze(0), target_idx)   # (1,C) vs (1)
                if model.use_orientation:
                    loss_orient = F.cross_entropy(
                        orient_logits[pos_idx].unsqueeze(0),
                        torch.tensor([orient_y], dtype=torch.long, device=device)
                    )
                    loss = loss_partner + 0.5 * loss_orient
                else:
                    loss = loss_partner

                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += float(loss.item())

        print(f"Epoch {epoch+1}/{num_epochs}  Loss: {total_loss:.4f}")


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
    msvae_model = None
    setvae_model = None
    config_dir = Path("configs")
    dataset_dir = Path("datasets") / args.dataset_dir
    model_dir = Path("models")
    config = toml.load(config_dir / args.config)
    graphs, max_node, min_node = load_graph_from_directory(dataset_dir)
    print(f"Loading graphs dataset {len(graphs)}")
    train_graphs, test_graphs = train_test_split(graphs, test_size=0.2, random_state=42)
    if args.msvae_model:
        msvae_config = toml.load(config_dir / args.msvae_config)
        msvae_model  = load_msvae_from_file(max_node, msvae_config, model_dir /args.msvae_model)
    if args.setvae_model:
        setvae_config = toml.load(config_dir / args.setvae_config)
        setvae_model  = load_setvae_from_file(max_node, setvae_config, model_dir /args.setvae_model)
    hidden_dim = config['training']['hidden_dim']
    num_layer = config['training']['num_layer']
    T = config['training']['T']
    k_eigen = config['data']['k_eigen']
    partner_k_hop = int(config['training'].get('partner_k_hop', 2))
    cfg_use_orient = bool(config['training'].get('predict_orientation', True))
    model = GraphER(k_eigen, hidden_dim, num_layer, T,
                    use_orientation=cfg_use_orient,
                    partner_k_hop=partner_k_hop)
    if args.input_model:
        model.load_model(model_dir / args.input_model)
        print(f"Model Graph-ER loaded from {args.input_model}")
    else:
        num_epochs = config['training']['num_epochs']
        learning_rate = config['training']['learning_rate']
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        train_grapher(model, train_graphs, num_epochs, learning_rate, T, k_eigen, device)
    if args.output_model:
        model.save_model(model_dir / args.output_model)
        print(f"Model saved to {args.output_model}")
    if args.evaluate:
        graph_eval = GraphsEvaluator()
        deg_eval = DegreeSequenceEvaluator()
        sample_graphs = random.choices(train_graphs,k=config['inference']['generate_samples'])
        test_seqs = [[deg for _, deg in graph.degree()] for graph in test_graphs ]
        degree_sequences = [[deg for _, deg in graph.degree()] for graph in sample_graphs]
  
        if msvae_model:
            generated_graphs, generated_seqs = model.generate_with_msvae(config['inference']['generate_samples'],T, msvae_model,k_eigen)
            print(f"Evaluate generated graphs using Havei Hamimi Model and MS-VAE")
            print(f"MMD Degree: {graph_eval.compute_mmd_degree_emd(test_graphs,generated_graphs,max_node)}")
            print(f"MMD Clustering Coefficient: {graph_eval.compute_mmd_cluster(test_graphs,generated_graphs)}")
            print(f"MMD Orbit count: {graph_eval.compute_mmd_orbit(test_graphs,generated_graphs)}")

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
    parser.add_argument('--setvae-config', type=str, help='Path to the configuration file in TOML format of Set-VAE')
    parser.add_argument('--setvae-model', type=str,help='Path to load a pre-trained Set-VAE model')
    parser.add_argument('--output-model', type=str, help='Path to save the trained model')
    parser.add_argument('--input-model', type=str, help='Path to load a pre-trained model')
    parser.add_argument('--evaluate', action='store_true', help='Whether we evaluate the model')
    args = parser.parse_args()
    main(args)

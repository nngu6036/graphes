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


from model_msvae import MSVAE
from model_setvae import SetVAE
from model_grapher import GraphER
from eval import DegreeSequenceEvaluator, GraphsEvaluator
from utils import *

# --- helper to infer orientation label (0 or 1) for reverse step ---
def _orientation_label(anchor, partner, removed_pair):
    """
    anchor:   (a,b)  in G_t
    partner:  (c,d)  in G_t
    removed_pair: ((u,v), (x,y))  that were present in G_{t-1}
    Return:
      0 if { (a,c), (b,d) } equals removed edges (unordered)
      1 if { (a,d), (b,c) } equals removed edges
    Raises if neither matches (shouldn't happen if trajectory is consistent).
    """
    (a,b), (c,d) = anchor, partner
    rem = {frozenset(removed_pair[0]), frozenset(removed_pair[1])}
    opt0 = {frozenset((a, c)), frozenset((b, d))}
    if opt0 == rem: return 0
    opt1 = {frozenset((a, d)), frozenset((b, c))}
    if opt1 == rem: return 1
    # Fallback: pick the closer one (rare). Or raise.
    raise ValueError("Orientation does not match removed edges; check trajectory construction.")

def train_grapher(model, graphs, num_epochs, learning_rate, T, k_eigen,device):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for G in graphs:
            G_hh = hh_graph_from_G(G)
            # --- Corrupt graph with t edge rewirings ---
            traj = transform_to_hh_via_stochastic_rewiring(G, G_hh, G.number_of_edges())
            for step_idx, (G_t, added_pair, removed_pair) in enumerate(traj, start=1):
                # Teacher-forced anchor = one of added edges
                (a,b), (c,d) = added_pair
                anchor = (a, b)
                pos_partner = (c, d)

                # Build candidate set = all edges disjoint with anchor (a,b)
                all_edges = list(G_t.edges())
                cand_edges = []
                for (x,y) in all_edges:
                    if frozenset((x,y)) == frozenset(anchor):
                        continue
                    if len({a,b,x,y}) == 4:   # disjoint
                        cand_edges.append((x,y))

                if not cand_edges:
                    continue  # nothing to learn this step

                # Build labels
                # Partner label: index of the positive in cand_edges
                pos_idx = None
                for i, e in enumerate(cand_edges):
                    if frozenset(e) == frozenset(pos_partner):
                        pos_idx = i; break

                orient_y = _orientation_label(anchor, pos_partner, removed_pair)
                # ---- Model forward on G_t ----
                data = graph_to_data(G_t, k_eigen) 
                x, edge_index = data.x.to(device), data.edge_index.to(device)

                partner_logits, orient_logits = model(
                    x=x, edge_index=edge_index,
                    first_edge=anchor, candidate_edges=cand_edges, t=step_idx
                )
                # ---- Losses ----
                # (1) Candidate partner classification (softmax CE over candidates)
                target_idx = torch.tensor([pos_idx], dtype=torch.long, device=device)
                loss_partner = F.cross_entropy(partner_logits.unsqueeze(0), target_idx)

                # (2) Orientation classification *for the positive candidate only*
                orient_logits_pos = orient_logits[pos_idx].unsqueeze(0)   # (1,2)
                orient_target = torch.tensor([orient_y], dtype=torch.long, device=device)
                loss_orient = F.cross_entropy(orient_logits_pos, orient_target)

                loss = loss_partner + 0.5 * loss_orient  # λ=0.5 is a good start

                # ---- Opt step ----
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += float(loss.item())
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss:.4f}")


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
    model = GraphER(k_eigen, hidden_dim,num_layer,T)
    if args.input_model:
        model.load_model(model_dir / args.input_model)
        print(f"Model Graph-ER loaded from {args.input_model}")
    else:
        num_epochs = config['training']['num_epochs']
        learning_rate = config['training']['learning_rate']
        train_grapher(model, train_graphs,num_epochs, learning_rate,T, k_eigen,'cpu')
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

        generated_graphs, generated_seqs = model.generate_from_sequences(test_seqs, k_eigen, method='havel_hakimi')
        print(f"Evaluate generated graphs using Havei Hamimi Model and test sequences")
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

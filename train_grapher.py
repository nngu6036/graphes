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
from model_energy import EnergyMLP
from utils import (
    build_candidates,
    rewire,
    graph_to_data,
    load_graph_from_directory,
    load_pyg_degree_sequence_from_directory,
    load_pyg_graph_from_directory,
    transform_to_hh_via_guided_rewiring,
    deterministic_connected_havel_hakimi,
    make_lambda_dist,
    compute_struct_features,
)

def energy_softmax_nce(
    E_all: torch.Tensor,
    pos_index: torch.Tensor,
    tau: float = 0.2,
) -> torch.Tensor:
    """
    Softmax / NCE style:
      -log softmax(-E/tau)[pos_index]

    Shapes:
      E_all:     [B, K] energies for candidate set per state (includes the positive)
      pos_index: [B] index of the positive within each row
    """
    # logits = -E / tau  (lower energy => higher logit)
    logits = (-E_all) / max(tau, 1e-8)
    logp = F.log_softmax(logits, dim=-1)  # [B,K]
    loss = -logp.gather(dim=-1, index=pos_index.view(-1, 1)).mean()
    return loss

def make_energy_fn(grapher, energy_mlp, k_eigen, device):
    grapher.eval()
    energy_mlp.eval()
    def energy_fn(G_nx):
        with torch.no_grad():
            data = graph_to_data(G_nx, k_eigen).to(device)
            z = grapher.encode_graph(data.x, data.edge_index).squeeze(0)
            return float(energy_mlp(z.unsqueeze(0)).item())
    return energy_fn

def train_energy(
    model: EnergyMLP,
    grapher: GraphER,
    graphs,
    steps_per_graph,
    dist_name,
    learning_rate,
    l2_reg,
    grad_clip,
    tau,
    k_eigen
):
    print("Training EnergyMLP")
    device = next(grapher.parameters()).device
    model = model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Freeze GraphER encoder
    grapher.eval()
    for p in grapher.parameters():
        p.requires_grad_(False)
    rng = random.Random()
    graphs_tuple = []
    for G in graphs:
        G_hh = deterministic_connected_havel_hakimi(G=G)
        graphs_tuple.append((G, G_hh))
    try:
        for idx, (G0, G_hh) in enumerate(graphs_tuple):
            # Work on a copy so we can optionally "walk" along positives
            G = G0.copy()
            total_loss = 0
            # Build once per graph
            lambda_dist = make_lambda_dist(dist_name, G_hh)
            for _ in range(steps_per_graph):
                edges = list(G.edges())
                if len(edges) < 2:
                    break
                anchor = edges[rng.randrange(len(edges))]
                candidate_edges = build_candidates(
                    G,
                    anchor,
                    ensure_connected=True,
                    k_hop=2,
                )
                if not candidate_edges:
                    continue
                z_candidates = []
                cand_dists = []
                cand_graphs = []   # <-- FIX: track graphs
                # One candidate per partner edge: choose best orientation
                for edge in candidate_edges:
                    best = None
                    for orient in (0, 1):
                        out = rewire(G, anchor, edge, orient, ensure_connected=True)
                        if out is None:
                            continue
                        Gc, _, _ = out
                        d = lambda_dist(Gc)
                        if best is None or d < best[0]:
                            best = (d, Gc)

                    if best is None:
                        continue
                    best_d, best_graph = best
                    data = graph_to_data(best_graph, k_eigen).to(device)
                    with torch.no_grad():
                        z_cand = grapher.encode_graph(data.x, data.edge_index).squeeze(0)  # [h]
                    z_candidates.append(z_cand)
                    cand_dists.append(best_d)
                    cand_graphs.append(best_graph)
                if not z_candidates:
                    continue
                pos_idx = int(np.argmin(cand_dists))
                Z = torch.stack(z_candidates, dim=0)   # [K, h]
                E = model(Z)                           # [K]
                E_all = E.view(1, -1)                  # [1, K]
                pos_index = torch.tensor([pos_idx], device=device, dtype=torch.long)
                loss = energy_softmax_nce(E_all, pos_index, tau=tau)
                if l2_reg > 0:
                    loss = loss + l2_reg * (E.pow(2).mean())
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip and grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                G = cand_graphs[pos_idx]
                total_loss += loss.item()
            print(f"{idx+1}/{len(graphs)} graph, Average Loss: {total_loss/steps_per_graph:.4f}")
    finally:
        for p in grapher.parameters():
            p.requires_grad_(True)
        grapher.train()
            

def train_grapher(
    model,
    graphs,
    learning_rate,
    T,
    dist_name,
    k_eigen,
    energy_mlp = None,
    energy_weight = 0.1
):
    """
    Training with edge-pair BCE loss + optional structural energy prediction loss.

    Reconstruction target = HH(G) embedding (currently commented out).
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    device = next(model.parameters()).device
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()
    model.train()
    energy_fn = None
    if energy_mlp is not None:
        energy_fn = make_energy_fn(model, energy_mlp, k_eigen, device=device)
    # Precompute HH graphs for each G
    graphs_tuple = []
    for idx,(G) in enumerate(graphs):
        G_hh =  deterministic_connected_havel_hakimi(G = G)
        # Forward diffusion trajectory (stochastic rewiring toward HH)
        lambda_dist = make_lambda_dist(dist_name, G_hh)
        step_idx = 0
        traj_loss = 0
        trajectory = transform_to_hh_via_guided_rewiring(
            G, G_hh, lambda_dist, T,
            ensure_connected = True, k_hop = 2, energy_fn = energy_fn, energy_weight = energy_weight
        )
        for (G_post, added_pair, removed_pair,_) in trajectory:
            (a, b), (c, d) = added_pair
            anchor      = (a, b)
            pos_partner = (c, d)
            # Candidate edges: use build_candidates under same constraints as inference
            candidate_edges = build_candidates(
                G_post,
                anchor,
                ensure_connected=True,
                k_hop=2,
            )
            if not candidate_edges:
                continue
            labels = torch.tensor(
                [1.0 if frozenset(edge) == frozenset(pos_partner) else 0.0
                 for edge in candidate_edges],
                dtype=torch.float32,
                device=device
            )
            # Edge-pair prediction loss
            data = graph_to_data(G_post, k_eigen).to(device)
            scores = model(data.x, data.edge_index, anchor, candidate_edges, t=step_idx)
            loss_edge = bce(scores, labels)
            # NEW: structural energy prediction loss (multi-task)
            if hasattr(model, "energy_head") and model.energy_head is not None and energy_weight > 0.0:
                # Graph embedding
                G_repr = model.encode_graph(data.x, data.edge_index)  # [1, hidden_dim]
                energy_pred = model.energy_head(G_repr).squeeze(0)    # [num_energy_targets]
                # True structural features (modularity, clustering)
                struct_targets = compute_struct_features(G_post).to(energy_pred.device)
                energy_loss = F.mse_loss(energy_pred, struct_targets)
                loss = loss_edge + energy_weight * energy_loss
            else:
                loss = loss_edge
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            traj_loss += float(loss.item())
            step_idx += 1
        print(f"{idx+1}/{len(graphs)} graph, Average Loss: {traj_loss/T:.4f}")

def load_msvae_from_file(max_degree,max_node, config, model_path):
    hidden_dim = config['training']['hidden_dim']
    latent_dim = config['training']['latent_dim']
    model = MSVAE(max_input_dim=max_degree, hidden_dim=hidden_dim, latent_dim=latent_dim, max_frequency = max_node)
    print(f"MS-VAE Model loaded from {model_path}")
    model.load_model(model_path)
    return model

def evaluate(train_graphs, test_graphs, model, msvae_model, T, k_eigen, num_samples):
    max_node_local = max(g.number_of_nodes() for g in test_graphs)
    graph_eval = GraphsEvaluator()
    deg_eval = DegreeSequenceEvaluator()
    sample_graphs = random.choices(train_graphs,k=num_samples)
    test_seqs = [[deg for _, deg in graph.degree()] for graph in test_graphs ]
    degree_sequences = [[deg for _, deg in graph.degree()] for graph in sample_graphs]
    generated_graphs, generated_seqs = model.generate(num_samples,T, msvae_model,k_eigen,method = 'havei_hakimi')
    print(f"Evaluate generated graphs using Havei Hamimi Model and MS-VAE")
    print(f"MMD Degree: {graph_eval.compute_mmd_degree_emd(test_graphs,generated_graphs,max_node_local)}")
    print(f"MMD Clustering Coefficient: {graph_eval.compute_mmd_cluster(test_graphs,generated_graphs)}")
    print(f"MMD Orbit count: {graph_eval.compute_mmd_orbit(test_graphs,generated_graphs)}")

def main(args):
    grapher_config = toml.load(args.config)
    msvae_config = toml.load(args.msvae_config)
    graphs, max_node, max_degree = load_graph_from_directory(args.dataset_dir)
    print(f"Loading graphs dataset {len(graphs)}")
    train_graphs, test_graphs = train_test_split(graphs, test_size=0.2, random_state=42)
    msvae_model  = load_msvae_from_file(max_degree, max_node, msvae_config, args.msvae_model)
    grapher_hidden_dim = grapher_config['training']['hidden_dim']
    num_layer = grapher_config['training']['num_layer']
    T = grapher_config['training']['T']
    k_eigen = grapher_config['training']['k_eigen']
    dist_name = grapher_config['guided_rewiring']['dist_name']
    energy_weight = grapher_config['guided_rewiring']['energy_weight']
    emb_dim = grapher_hidden_dim
    energy_hidden_dim = grapher_config['energy']['hidden_dim']
    num_layers = grapher_config['energy']['num_layers']
    dropout = grapher_config['energy']['dropout']
    steps_per_graph = grapher_config['energy']['steps_per_graph']
    learning_rate = grapher_config['training']['learning_rate']
    l2_reg = grapher_config['energy']['l2_reg']
    grad_clip = grapher_config['energy']['grad_clip']
    tau = grapher_config['energy']['tau']
    num_samples = grapher_config['inference']['generate_samples']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GraphER(k_eigen + 1, grapher_hidden_dim,num_layer,T).to(device)
    energy_model = EnergyMLP(emb_dim=emb_dim, hidden_dim=energy_hidden_dim, num_layers=num_layers, dropout=dropout).to(device)
    if args.input_model:
        model.load_model(args.input_model)
        print(f"Model Graph-ER loaded from {args.input_model}")
    else:
        print("Train GraphER without energy")
        train_grapher(model, train_graphs, learning_rate,T, dist_name,k_eigen,energy_mlp = None, energy_weight = energy_weight)
        print(f"Model saved to {args.output_model}")
        model.save_model( args.output_model)
        evaluate(train_graphs, test_graphs, model, msvae_model,T,k_eigen ,num_samples)
        train_energy(energy_model, model,train_graphs,steps_per_graph,dist_name,learning_rate,l2_reg,grad_clip, tau, k_eigen)
        print(f"Energy model saved to energy")
        energy_model.save_model( 'energy')
        print("Train GraphER with energy")
        train_grapher(model, train_graphs, learning_rate,T, dist_name,k_eigen,energy_mlp = energy_model, energy_weight= energy_weight)
        model.save_model( args.output_model)
        print(f"Model saved to {args.output_model}")
    if args.evaluate:
        evaluate(train_graphs, test_graphs, model, msvae_model,T,k_eigen ,num_samples)


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

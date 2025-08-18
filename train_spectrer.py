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
from model_spectrer import SpectralER

from eval import DegreeSequenceEvaluator, GraphsEvaluator
from utils import (
    graph_to_data,
    check_sequence_validity,
    load_graph_from_directory,
    laplacian_eigs,
    normalized_laplacian_dense,
    _B_inner,
    _pair_inner,
)

# --- Add these helpers near the top of train_grapher.py (or move to utils.py if you prefer) ---

def _count_simple_paths_exact_k(G: nx.Graph, s: int, t: int, k: int, forbidden_edge=None, cap=100000):
    """
    Count simple paths of *exact* length k (in edges) from s to t, without revisiting nodes.
    If forbidden_edge is given as a frozenset({u,v}), that edge is not allowed in the path.
    'cap' is a safety limit to avoid pathological blow-ups; counting stops after reaching cap.
    """
    if k < 0:
        return 0
    if k == 0:
        return int(s == t)
    # Quick pruning: can't reach in k steps if distance lower bound is too big.
    # (Optional: comment out if you don't have precomputed distances)
    count = 0
    target = t
    forb = forbidden_edge
    def dfs(u, depth, visited):
        nonlocal count
        if count >= cap:
            return
        if depth == k:
            if u == target:
                count += 1
            return
        for w in G.neighbors(u):
            if w in visited:
                continue
            if forb and frozenset((u, w)) == forb:
                continue
            # prevent early arrival: we need exact length k
            if w == target and depth + 1 < k:
                continue
            visited.add(w)
            dfs(w, depth + 1, visited)
            visited.remove(w)
    dfs(s, 0, {s})
    return count

def count_edge_cycles_n(G: nx.Graph, u: int, v: int, n: int) -> int:
    """
    Number of simple cycles of length n that contain edge (u,v).
    For n=3, use fast common-neighbors count. For n>3, count simple paths
    of length n-1 between u and v in G with edge (u,v) forbidden.
    """
    if n < 3:
        return 0
    if n == 3:
        # triangles through (u,v)
        return len(set(G.neighbors(u)) & set(G.neighbors(v)))
    forb = frozenset((u, v))
    return _count_simple_paths_exact_k(G, u, v, n - 1, forbidden_edge=forb)

def count_cycles_if_add_edge_n(G: nx.Graph, a: int, b: int, n: int) -> int:
    """
    Number of simple cycles of length n that would be created by adding (a,b).
    Equals the number of simple paths of length n-1 between a and b in the current G.
    """
    if n < 3:
        return 0
    if n == 3:
        # New triangles formed by adding (a,b) are common neighbors of a and b
        return len(set(G.neighbors(a)) & set(G.neighbors(b)))
    return _count_simple_paths_exact_k(G, a, b, n - 1, forbidden_edge=None)

# --- Replace your rewire_edges with this generic cycle-preserving version ---

def rewire_edges(G: nx.Graph, num_rewirings: int, cycle_len: int = 3, cap_per_count: int = 100000):
    """
    Perform up to num_rewirings double-edge swaps that preserve (or increase) the count of
    cycles of length 'cycle_len'. Uses '>=' criterion like your triangle-preserving version.

    cap_per_count: safety cap passed to the path counter to guard against blow-ups on dense graphs.
    """
    step = 0
    removed_pair = None
    added_pair = None
    for _ in range(num_rewirings):
        edges = list(G.edges())
        if len(edges) < 2:
            break
        e1, e2 = random.sample(edges, 2)
        u, v = e1
        x, y = e2
        if len({u, v, x, y}) != 4:
            continue
        # cycles removed by deleting both edges
        # Use capped counters to avoid pathological runtimes
        c_removed = (
            count_edge_cycles_n(G, u, v, cycle_len) +
            count_edge_cycles_n(G, x, y, cycle_len)
        )
        # Option 1: (u,x) & (v,y)
        can1 = (not G.has_edge(u, x)) and (not G.has_edge(v, y))
        if can1:
            c_add1 = (
                count_cycles_if_add_edge_n(G, u, x, cycle_len) +
                count_cycles_if_add_edge_n(G, v, y, cycle_len)
            )
        else:
            c_add1 = -1  # invalid
        # Option 2: (u,y) & (v,x)
        can2 = (not G.has_edge(u, y)) and (not G.has_edge(v, x))
        if can2:
            c_add2 = (
                count_cycles_if_add_edge_n(G, u, y, cycle_len) +
                count_cycles_if_add_edge_n(G, v, x, cycle_len)
            )
        else:
            c_add2 = -1  # invalid
        # Choose the best valid option that preserves or increases cycle count
        best_add = max(c_add1, c_add2)
        if best_add >= c_removed and best_add >= 0:
            if c_add1 >= c_add2 and can1:
                G.remove_edges_from([(u, v), (x, y)])
                G.add_edges_from([(u, x), (v, y)])
                removed_pair = ((u, v), (x, y))
                added_pair = ((u, x), (v, y))
                step += 1
            elif can2:
                G.remove_edges_from([(u, v), (x, y)])
                G.add_edges_from([(u, y), (v, x)])
                removed_pair = ((u, v), (x, y))
                added_pair = ((u, y), (v, x))
                step += 1
    return G, removed_pair, added_pair, step


def count_common_neighbors(G, a, b):
    """Return number of common neighbors of nodes a and b."""
    return len(set(G.neighbors(a)) & set(G.neighbors(b)))


def train_spectral(model, graphs, num_epochs, learning_rate, T, k_eigen, cycle, device):
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device).train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for G in graphs:
            num_rewirings = random.randint(1, T)
            G_t, removed_pair, added_pair, step = rewire_edges(G.copy(), num_rewirings, cycle_len=cycle)
            if not removed_pair or not added_pair:
                continue

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
        print(f"[spectral] Epoch {epoch+1}/{num_epochs}  NLL: {epoch_loss:.4f}")


def load_msvae_from_file(max_node,config, model_path):
    hidden_dim = config['training']['hidden_dim']
    latent_dim = config['training']['latent_dim']
    model = MSVAE(max_input_dim=max_node, hidden_dim=hidden_dim, latent_dim=latent_dim, max_frequency = max_node)
    print(f"MS-VAE Model loaded from {model_path}")
    model.load_model(model_path)
    return model

def load_setvae_from_file(max_node,config, model_path):
    hidden_dim = config['training']['hidden_dim']
    latent_dim = config['training']['latent_dim']
    model = SetVAE(hidden_dim=hidden_dim, latent_dim=latent_dim, max_degree = max_node)
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
    cycle     = config['training']['cycle']

    # FIX: correct ctor args (k, hidden, T)
    model = SpectralER(k_eigen, hidden_dim, T)

    if args.input_model:
        model.load_model(model_dir / args.input_model)
        print(f"SpectralER model loaded from {args.input_model}")
    else:
        num_epochs    = config['training']['num_epochs']
        learning_rate = config['training']['learning_rate']
        train_spectral(model, train_graphs, num_epochs, learning_rate, T, k_eigen, cycle, 'cpu')

    if args.output_model:
        model.save_model(model_dir / args.output_model)
        print(f"Model saved to {args.output_model}")

    if args.evaluate:
        graph_eval = GraphsEvaluator()
        deg_eval = DegreeSequenceEvaluator()
        test_seqs = [[deg for _, deg in g.degree()] for g in test_graphs]

        sample_graphs = random.choices(train_graphs,k=config['inference']['generate_samples'])
        degree_sequences = [[deg for _, deg in graph.degree()] for graph in sample_graphs]

        """
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
        """
        generated_graphs, generated_seqs = model.generate_from_sequences(T,degree_sequences,k_eigen,method = 'constraint_configuration_model')
        print(f"Evaluate generated graphs sampled from training using constraint configuration model")
        print(f"MMD Degree: {graph_eval.compute_mmd_degree_emd(test_graphs,generated_graphs,max_node)}")
        print(f"MMD Clustering Coefficient: {graph_eval.compute_mmd_cluster(test_graphs,generated_graphs)}")
        print(f"MMD Orbit count: {graph_eval.compute_mmd_orbit(test_graphs,generated_graphs)}")

        generated_graphs, generated_seqs = model.generate_from_sequences(T,degree_sequences,k_eigen,method = 'havei_hakimi')
        print(f"Evaluate generated graphs sampled from training using constraint configuration model")
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

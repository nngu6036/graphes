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
from model_spectrer import SpectralER

from eval import DegreeSequenceEvaluator, GraphsEvaluator
from utils import *

def gaussian_nll(y, mu, logvar):
    return 0.5 * ((y - mu) ** 2 * torch.exp(-logvar) + logvar)

def kl_to_std_normal(mu, logvar, prior_var=1.0):
    return 0.5 * ((torch.exp(logvar) + mu**2) / prior_var - 1.0 + math.log(prior_var) - logvar)

def affected_nodes_from_swap(G, removed_pair, added_pair):
    """
    Return a small set of nodes affected by a (ground-truth) 2-swap:
    endpoints and their 1-hop neighbors in G.
    """
    (u, v), (x, y) = removed_pair, added_pair
    core = {u, v, x, y}
    A = set(core)
    for a in core:
        A.update(G.neighbors(a))
    return sorted(A)

def laplacian_patch_mse(L_true: torch.Tensor, L_pred: torch.Tensor, idx: list[int]) -> torch.Tensor:
    """
    Frobenius MSE on the |idx| x |idx| submatrix (rows/cols = idx).
    """
    if not idx:
        return torch.zeros((), device=L_true.device)
    I = torch.as_tensor(idx, device=L_true.device, dtype=torch.long)
    Lt = L_true.index_select(0, I).index_select(1, I)
    Lp = L_pred.index_select(0, I).index_select(1, I)
    return (Lt - Lp).pow(2).mean()

def train_spectral(model, graphs, num_epochs, learning_rate, T, k_eigen, device,
                   beta_kl=0.0, gamma_lap=0.0, eps_y=1e-6, cycle=0, gamma_local=0.0):

    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    sched = None
    if cycle and cycle > 0:
        T0 = max(1, num_epochs // cycle)
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=T0)

    model.to(device)
    model.train()

    # Schedules
    KL_DELAY, KL_WARMUP = 10, 20
    LAP_WARMUP = 50
    LOC_WARMUP = 30

    def kl_weight(epoch):
        if beta_kl <= 0.0: return 0.0
        if epoch < KL_DELAY: return 0.0
        ramp = min(1.0, (epoch - KL_DELAY + 1) / max(1, KL_WARMUP))
        return beta_kl * ramp

    def lap_weight(epoch):
        if gamma_lap <= 0.0: return 0.0
        ramp = min(1.0, (epoch + 1) / max(1, LAP_WARMUP))
        return gamma_lap * ramp

    def loc_weight(epoch):
        if gamma_local <= 0.0: return 0.0
        ramp = min(1.0, (epoch + 1) / max(1, LOC_WARMUP))
        return gamma_local * ramp

    LOGVAR_MIN, LOGVAR_MAX = -5.0, 2.0

    for epoch in range(num_epochs):
        epoch_loss, n_steps = 0.0, 0

        for G in graphs:
            G_hh = hh_graph_from_G(G)
            traj = transform_to_hh_via_stochastic_rewiring(G, G_hh, G.number_of_edges())

            for step, (G_t, added_pair, removed_pair) in enumerate(traj, start=1):
                # Recover G_{t-1}
                G_t_prev = G_t.copy()
                (u, v), (x, y) = removed_pair
                (a, b), (c, d) = added_pair
                if G_t_prev.has_edge(a, b): G_t_prev.remove_edge(a, b)
                if G_t_prev.has_edge(c, d): G_t_prev.remove_edge(c, d)
                G_t_prev.add_edge(u, v); G_t_prev.add_edge(x, y)

                # Spectra
                lam_t_np, U_t_np = laplacian_eigs(G_t,      k_eigen, normed=True)
                lam_tm1_np, _    = laplacian_eigs(G_t_prev, k_eigen, normed=True)
                lam_t   = torch.from_numpy(lam_t_np).to(device=device, dtype=torch.float32)
                lam_tm1 = torch.from_numpy(lam_tm1_np).to(device=device, dtype=torch.float32)

                V = G_t.number_of_nodes()
                E = G_t.number_of_edges()
                K = min(k_eigen, max(0, V - 1))
                if K <= 0:  # (You said tiny graphs won't appear, but keep guard.)
                    continue

                # Extras
                avg_deg = (2.0 * E) / max(1, V)
                density = (2.0 * E) / max(1, V * (V - 1))
                extra = torch.tensor([math.log(max(V, 2)), avg_deg, density],
                                     device=device, dtype=torch.float32)

                # Predict Gaussian over y_{t-1} | lam_t
                mu_y_full, logvar_y_full = model(lam_t, step, extra)
                mu_y, logvar_y = mu_y_full[:K], logvar_y_full[:K]
                logvar_y = logvar_y.clamp(LOGVAR_MIN, LOGVAR_MAX)

                # Targets in y-space
                y_target = lam_to_y(lam_tm1[:K])  # ε=1e-6 default

                # (1) Reconstruction NLL
                recon_nll = gaussian_nll(y_target, mu_y, logvar_y).mean()

                # Scheduled weights
                bw_kl  = kl_weight(epoch)
                bw_lap = lap_weight(epoch)
                bw_loc = loc_weight(epoch)

                # (2) KL regularizer (tiny)
                kl_loss = kl_to_std_normal(mu_y, logvar_y).mean() if bw_kl > 0.0 else torch.zeros((), device=device)

                # Build predicted L̂_{t-1} once (deterministic mean in y-space)
                lam_hat_tm1 = y_to_lam(mu_y.detach())  # (K,)
                L_t_np   = normalized_laplacian_dense(G_t)
                L_tm1_np = normalized_laplacian_dense(G_t_prev)
                L_t   = torch.from_numpy(L_t_np).to(device=device, dtype=torch.float32)
                L_tm1 = torch.from_numpy(L_tm1_np).to(device=device, dtype=torch.float32)
                U_t   = torch.from_numpy(U_t_np[:, :K]).to(device=device, dtype=torch.float32)
                L_hat = L_t - U_t @ torch.diag(lam_t[:K]) @ U_t.T \
                           + U_t @ torch.diag(lam_hat_tm1) @ U_t.T

                # (3) Global Laplacian reconstruction aux
                lap_recon = (L_tm1 - L_hat).pow(2).mean() if bw_lap > 0.0 else torch.zeros((), device=device)

                # (4) Local Laplacian patch aux on affected neighborhood only
                if bw_loc > 0.0:
                    # nodes impacted by the ground-truth swap (endpoints + their neighbors)
                    A = affected_nodes_from_swap(G_t, removed_pair, added_pair)
                    # The dense Laplacians above use G_t's node ordering. Because G_t_prev is a
                    # direct edge edit of G_t, the order matches; we can index directly.
                    # Map node labels to positions in the Laplacian:
                    node_index = {n: i for i, n in enumerate(G_t.nodes())}
                    idx = [node_index[n] for n in A if n in node_index]
                    lap_local = laplacian_patch_mse(L_tm1, L_hat, idx)
                else:
                    lap_local = torch.zeros((), device=device)

                loss = recon_nll + bw_kl * kl_loss + bw_lap * lap_recon + bw_loc * lap_local

                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()

                epoch_loss += float(loss.item())
                n_steps += 1

        if sched is not None:
            sched.step(epoch)

        mean_loss = epoch_loss / max(1, n_steps)
        print(f"Epoch {epoch+1}/{num_epochs} | mean_loss={mean_loss:.4f} "
              f"(steps={n_steps}, β_KL_sched={kl_weight(epoch):.5f}, "
              f"γ_Lap_sched={lap_weight(epoch):.5f}, γ_Local_sched={loc_weight(epoch):.5f})")


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
        num_epochs    = config["training"].get("num_epochs", 400)
        learning_rate = config["training"].get("learning_rate", 1e-3)
        T             = config["training"].get("T", 100)
        beta_kl       = config["training"].get("beta_kl", 0.0)
        gamma_lap     = config["training"].get("gamma_lap", 0.0)
        gamma_local = config["training"].get("gamma_local", 0.0)

        k_eigen       = config["data"].get("k_eigen", 15)

        device = "cpu"  # or your chosen device
        train_spectral(
            model, train_graphs, num_epochs, learning_rate, T, k_eigen,
            device=device,
            beta_kl=beta_kl,
            gamma_lap=gamma_lap,
            eps_y=1e-6,
            cycle=0,
            gamma_local=gamma_local,
        )

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

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

def edge_set(G: nx.Graph):
    """Return edges as a set, robust for undirected vs directed vs multigraph."""
    if G.is_multigraph():
        # Count parallel edges separately by using (u,v,key)
        if G.is_directed():
            return {(u, v, k) for u, v, k in G.edges(keys=True)}
        else:
            # Use frozenset for undirected endpoints + key to distinguish parallels
            return {(frozenset((u, v)), k) for u, v, k in G.edges(keys=True)}
    else:
        if G.is_directed():
            return {(u, v) for u, v in G.edges()}
        else:
            # undirected simple graph
            return {frozenset((u, v)) for u, v in G.edges()}

def symmetric_difference_edit_distance(G: nx.Graph, H: nx.Graph,
                                       normalize: bool = False,
                                       swap_units: bool = False) -> float:
    """
    dΔ(G,H) = |E(G) Δ E(H)|.
    - normalize=True -> divide by max(|E(G)|, |E(H)|) to get [0,1].
    - swap_units=True -> return dΔ/2 (useful when progress counted in 2-edge swaps).
    """
    # (Optional) sanity: same node labels
    if set(G.nodes()) != set(H.nodes()):
        raise ValueError("Graphs must have the same node set (align/permute first).")

    EG, EH = edge_set(G), edge_set(H)
    d_delta = len((EG - EH) | (EH - EG))  # symmetric difference size

    val = d_delta / 2.0 if swap_units else float(d_delta)
    if normalize:
        denom = max(len(EG), len(EH), 1)
        val /= denom
    return val

def rewire_edges(
    G: nx.Graph,
    G_target,
    num_rewirings: int,
    max_trials_per_step: int = 64,
    preserve_components: bool = True,
):
    """
    Degree-preserving rewiring that *also* preserves the number of connected components.

    - If the input graph is connected, the result after each accepted swap remains connected.
    - When preserve_components is True, we require that the number of CCs stays the same.
    - We skip edges that are bridges (for connected graphs) to avoid obvious disconnects.
    """
    step = 0
    removed_pair = None
    added_pair = None

    base_cc = nx.number_connected_components(G) if preserve_components else None

    for _ in range(num_rewirings):
        success = False

        # Optional: avoid picking bridges when the graph is connected.
        bridges = set()
        if preserve_components and base_cc == 1:
            # nx.bridges returns an iterator of edges that are bridges.
            bridges = {frozenset(e) for e in nx.bridges(G)}

        # Candidate edge list (optionally excluding bridges).
        all_edges = [e for e in G.edges() if frozenset(e) not in bridges] if bridges else list(G.edges())
        if len(all_edges) < 2:
            break
        base_dist =  symmetric_difference_edit_distance(G,G_target)
        for _ in range(max_trials_per_step):
            e1, e2 = random.sample(all_edges, 2)
            u, v = e1
            x, y = e2
            # Disjoint endpoints for a valid 2-edge swap
            if len({u, v, x, y}) != 4:
                continue

            # Two possible rewiring options
            for new_e1, new_e2 in (((u, x), (v, y)), ((u, y), (v, x))):
                # Keep it a simple graph (no loops, no duplicate edges)
                if new_e1[0] == new_e1[1] or new_e2[0] == new_e2[1]:
                    continue
                if G.has_edge(*new_e1) or G.has_edge(*new_e2):
                    continue

                G_try = G.copy()
                G_try.remove_edges_from([e1, e2])
                G_try.add_edges_from([new_e1, new_e2])

                # Preserve connected components (connectedness if base_cc==1)
                if preserve_components:
                    if nx.number_connected_components(G_try) != base_cc:
                        continue
                new_dist =  symmetric_difference_edit_distance(G_try,G_target)
                if new_dist > base_dist:
                    continue
                # Accept
                G = G_try
                removed_pair = (e1, e2)
                added_pair = (new_e1, new_e2)
                step += 1
                success = True
                break  # stop checking options

            if success:
                break  # proceed to next rewiring step

        # If we couldn't find any valid swap this step, just move on.
        if not success:
            continue

    return G, removed_pair, added_pair, step



def count_common_neighbors(G, a, b):
    """Return number of common neighbors of nodes a and b."""
    return len(set(G.neighbors(a)) & set(G.neighbors(b)))

def train_grapher(model, graphs, num_epochs, learning_rate, T, k_eigen,device):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for G in graphs:
            # Construc the Havel Hakimi graph
            G_hh = havel_hakimi_construction([deg for _, deg in G.degree()])
            # --- Corrupt graph with t edge rewirings ---
            num_rewirings = random.randint(1,T)
            G_corrupt, removed_pair, added_pair,step = rewire_edges(G.copy(),G_hh,num_rewirings)
            if not removed_pair or not added_pair:
                continue
            # --- Define anchor and target edge ---
            first_edge_added, second_edge_added = added_pair  # predict second_edge_added given first_edge_added
            # --- Graph to PyG format ---
            data = graph_to_data(G_corrupt,k_eigen).to(device)
            # --- Edge candidates from corrupted graph ---
            u, v = first_edge_added
            uv = frozenset(first_edge_added)
            candidate_edges = [e for e in G_corrupt.edges() if frozenset(e) != uv and len(set(e + first_edge_added)) == 4]
            # --- Construct binary labels ---
            labels = torch.tensor(
                [1.0 if frozenset(edge) == frozenset(second_edge_added) else 0.0 for edge in candidate_edges],
                dtype=torch.float32,
                device=device
            )
            # --- Forward pass ---
            scores = model(data.x, data.edge_index, first_edge_added, candidate_edges, t=step)
            loss = criterion(scores.squeeze(), labels)
            # --- Backpropagation ---
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")


def load_msvae_from_file(max_node,config, model_path):
    hidden_dim = config['training']['hidden_dim']
    latent_dim = config['training']['latent_dim']
    model = MSVAE(max_input_dim=max_node, hidden_dim=hidden_dim, latent_dim=latent_dim, max_frequency = max_node)
    print(f"MS-VAE Model loaded from {model_path}")
    model.load_model(model_path)
    return model


def main(args):
    msvae_model = None
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
        """
        generated_graphs, generated_seqs = model.generate_from_sequences(T,degree_sequences,k_eigen,method = 'havei_hakimi')
        print(f"Evaluate generated graphs sampled from training using havei-hakimi model")
        print(f"MMD Degree: {graph_eval.compute_mmd_degree_emd(test_graphs,generated_graphs,max_node)}")
        print(f"MMD Clustering Coefficient: {graph_eval.compute_mmd_cluster(test_graphs,generated_graphs)}")
        print(f"MMD Orbit count: {graph_eval.compute_mmd_orbit(test_graphs,generated_graphs)}")
        
        generated_graphs, generated_seqs = model.generate(config['inference']['generate_samples'],T, msvae_model,k_eigen,method = 'constraint_configuration_model')
        print(f"Evaluate generated graphs using constraint Configuraiton Model")
        print(f"MMD Degree: {graph_eval.compute_mmd_degree_emd(test_graphs,generated_graphs,max_node)}")
        print(f"MMD Clustering Coefficient: {graph_eval.compute_mmd_cluster(test_graphs,generated_graphs)}")
        print(f"MMD Orbit count: {graph_eval.compute_mmd_orbit(test_graphs,generated_graphs)}")
        deg_eval = DegreeSequenceEvaluator()
        test_seqs = [[deg for _, deg in graph.degree()] for graph in test_graphs ]
        print(f"KL Distance: {deg_eval.evaluate_multisets_kl_distance(test_seqs,generated_seqs,max_node)}")
        print(f"MMD Distance: {deg_eval.evaluate_multisets_mmd_distance(test_seqs,generated_seqs,max_node)}")

        generated_graphs, generated_seqs = model.generate(config['inference']['generate_samples'],T, msvae_model,k_eigen,method = 'configuration_model')
        print(f"Evaluate generated graphs using  Configuraiton Model")
        print(f"MMD Degree: {graph_eval.compute_mmd_degree_emd(test_graphs,generated_graphs,max_node)}")
        print(f"MMD Clustering Coefficient: {graph_eval.compute_mmd_cluster(test_graphs,generated_graphs)}")
        print(f"MMD Orbit count: {graph_eval.compute_mmd_orbit(test_graphs,generated_graphs)}")
        deg_eval = DegreeSequenceEvaluator()
        test_seqs = [[deg for _, deg in graph.degree()] for graph in test_graphs ]
        print(f"KL Distance: {deg_eval.evaluate_multisets_kl_distance(test_seqs,generated_seqs,max_node)}")
        print(f"MMD Distance: {deg_eval.evaluate_multisets_mmd_distance(test_seqs,generated_seqs,max_node)}")
        """
        if msvae_model:
            generated_graphs, generated_seqs = model.generate_with_msvae(config['inference']['generate_samples'],T, msvae_model,k_eigen)
            print(f"Evaluate generated graphs using Havei Hamimi Model and MS-VAE")
            print(f"MMD Degree: {graph_eval.compute_mmd_degree_emd(test_graphs,generated_graphs,max_node)}")
            print(f"MMD Clustering Coefficient: {graph_eval.compute_mmd_cluster(test_graphs,generated_graphs)}")
            print(f"MMD Orbit count: {graph_eval.compute_mmd_orbit(test_graphs,generated_graphs)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GRAPH-ER Model')
    parser.add_argument('--dataset-dir', type=str, required=True,help='Path to the directory containing graph files')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file in TOML format of Graph-ER')
    parser.add_argument('--msvae-config', type=str, help='Path to the configuration file in TOML format of MS-VAE')
    parser.add_argument('--msvae-model', type=str,help='Path to load a pre-trained MS-VAE model')
    parser.add_argument('--output-model', type=str, help='Path to save the trained model')
    parser.add_argument('--input-model', type=str, help='Path to load a pre-trained model')
    parser.add_argument('--evaluate', action='store_true', help='Whether we evaluate the model')
    args = parser.parse_args()
    main(args)

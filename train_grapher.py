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
from collections import deque

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

def _within_k_hops(G: nx.Graph, s, t, k: int) -> bool:
    """Return True iff dist_G(s,t) <= k (BFS early-stop)."""
    if s == t:
        return True
    if k <= 0:
        return False
    seen = {s}
    q = deque([(s, 0)])
    while q:
        v, d = q.popleft()
        if d == k:
            continue
        for w in G.neighbors(v):
            if w == t:
                return True
            if w not in seen:
                seen.add(w)
                q.append((w, d + 1))
    return False

def _edges_within_k_hops(G: nx.Graph, e1, e2, k: int) -> bool:
    """Require the two SOURCE edges to be 'k-local' to each other."""
    (u, v), (x, y) = e1, e2
    return (
        _within_k_hops(G, u, x, k) or
        _within_k_hops(G, u, y, k) or
        _within_k_hops(G, v, x, k) or
        _within_k_hops(G, v, y, k)
    )

def _assort_score(a, b, deg, mode="product"):
    """
    Larger is better for high-high pairs.
      mode='product'      -> deg[a]*deg[b]   (strongly favors high-high)
      mode='similarity'   -> - (deg[a]-deg[b])**2   (favors similar degrees)
    """
    if mode == "similarity":
        da, db = deg[a], deg[b]
        return - (da - db) ** 2
    # default: product
    return deg[a] * deg[b]

def _weighted_two_edges(edges, weights):
    """Pick two distinct edges with probability proportional to weights."""
    # First edge
    idx1 = random.choices(range(len(edges)), weights=weights, k=1)[0]
    # Second edge (resample until different)
    while True:
        idx2 = random.choices(range(len(edges)), weights=weights, k=1)[0]
        if idx2 != idx1:
            break
    return edges[idx1], edges[idx2]

# ---------- main ----------
def rewire_edges_k_local_assortative(
    G: nx.Graph,
    num_rewirings,
    k: int = 2,
    keep_connected: bool = True,
    forbid_bridges: bool = True,
    assortative_bias: float = 7.0,
    assortative_mode: str = "product",   # 'product' or 'similarity'
    weight_edge_sampling: bool = True,   # bias which source edges we pick
):
    """
    One degree-preserving 2-edge swap that:
      • keeps the graph connected (reverts if not),
      • only considers source edges within k hops (locality),
      • BIASES the new edges to connect high-degree nodes together.

    Bias controls:
      - weight_edge_sampling=True biases which *source* edges are chosen;
        edges with large deg[u]*deg[v] are sampled more often.
      - assortative_bias>0 applies a softmax over the two candidate pairings,
        preferring the one with larger sum of scores for (a,b) and (c,d).
      - assortative_mode='product' (default) strongly prefers high-high pairs;
        use 'similarity' to favor degree-similar pairs (assortative but gentler).

    Returns:
      G (modified in-place; unchanged if no acceptable swap is found).
    """
    add,remove = None, None
    step = 0
    if G.number_of_edges() < 2:
        return G, add, remove

    deg = dict(G.degree())

    # Precompute optional bridge set (helps avoid disconnections)
    bridge_set = set()
    if forbid_bridges:
        bridge_set = {tuple(sorted(e)) for e in nx.bridges(G)}

    edges = list(G.edges())

    # Optional: bias *which* edges we propose to rewire
    if weight_edge_sampling:
        base_w = [deg[u] * deg[v] for (u, v) in edges]  # product works well
        # Avoid zero weights
        min_pos = 1 if all(w == 0 for w in base_w) else 0
        weights = [w + min_pos for w in base_w]
    else:
        weights = [1.0] * len(edges)

    for _ in range(num_rewirings):
        # Pick two source edges (possibly weighted toward high-degree endpoints)
        e1, e2 = _weighted_two_edges(edges, weights)
        u, v = e1
        x, y = e2

        # Distinct endpoints for a valid 2-switch
        if len({u, v, x, y}) != 4:
            continue

        # k-hop locality between the *source* edges
        if not _edges_within_k_hops(G, e1, e2, k):
            continue

        # (Optional) avoid deleting bridges
        if forbid_bridges and (tuple(sorted(e1)) in bridge_set or tuple(sorted(e2)) in bridge_set):
            continue

        # Two candidate pairings
        candidates = [ (u, x, v, y), (u, y, v, x) ]
        random.shuffle(candidates)  # avoid deterministic tie behavior

        # Score each valid pairing by how assortative the *new* edges are
        scored = []
        for a, b, c, d in candidates:
            if a == b or c == d:
                continue
            if G.has_edge(a, b) or G.has_edge(c, d):
                continue
            s = _assort_score(a, b, deg, assortative_mode) + _assort_score(c, d, deg, assortative_mode)
            scored.append(((a, b, c, d), s))

        if not scored:
            continue

        # Softmax over scores -> prefer higher assortativity, but keep some randomness
        max_s = max(s for _, s in scored)
        probs = [math.exp(assortative_bias * (s - max_s)) for _, s in scored]
        total = sum(probs)
        probs = [p / total for p in probs]
        choice_idx = random.choices(range(len(scored)), weights=probs, k=1)[0]
        a, b, c, d = scored[choice_idx][0]

        # Execute 
        add = (a, b), (c, d)
        remove = (u, v), (x, y)
        step += 1
        G.remove_edges_from([(u, v), (x, y)])
        G.add_edges_from([(a, b), (c, d)])

        # Enforce connectivity
        if keep_connected and not nx.is_connected(G):
            # revert and try again
            add, remove = None
            step -= 1
            G.remove_edges_from([(a, b), (c, d)])
            G.add_edges_from([(u, v), (x, y)])
            continue

    # No acceptable swap found
    return G, add, remove, step


def train_grapher(model, graphs, num_epochs, learning_rate, T, k_eigen,device):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for G in graphs:
            # --- Corrupt graph with t edge rewirings ---
            num_rewirings = random.randint(1,T)
            G_corrupt, added_pair, removed_pair,step = rewire_edges_k_local_assortative(G.copy(),num_rewirings)
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

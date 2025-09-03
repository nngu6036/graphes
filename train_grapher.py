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
from utils import rewire_edges_k_local_assortative, load_graph_from_directory, graph_to_data


def hh_graph_from_G(G):
    """
    Build a canonical Havel–Hakimi realization that uses the same node labels as G.
    Ties are broken by (higher degree first, then smaller node id).
    """
    deg_pairs = sorted(((d, u) for u, d in G.degree()), key=lambda x: (-x[0], x[1]))
    seq = [d for d, _ in deg_pairs]
    # Build HH graph on 0..n-1 then relabel back to original nodes in this order
    H_int = nx.havel_hakimi_graph(seq)
    mapping = {i: deg_pairs[i][1] for i in range(len(seq))}
    H = nx.relabel_nodes(H_int, mapping, copy=True)
    return H

def _ek(u, v):
    return (u, v) if u <= v else (v, u)

def _pick_valid_swap(G, max_tries=128, rng=random):
    """
    Propose a valid 2-edge swap (e1, e2) -> (f1, f2).
    Returns (e1, e2, f1, f2) or None if not found.
    """
    E = list(G.edges())
    nE = len(E)
    if nE < 2:
        return None
    for _ in range(max_tries):
        (a, b) = E[rng.randrange(nE)]
        (c, d) = E[rng.randrange(nE)]
        if len({a, b, c, d}) != 4:  # endpoints must be all different
            continue
        # two possible rewires; pick one at random
        if rng.random() < 0.5:
            f1, f2 = (a, c), (b, d)
        else:
            f1, f2 = (a, d), (b, c)
        # no self-loops and no multi-edges
        if f1[0] == f1[1] or f2[0] == f2[1]:
            continue
        if G.has_edge(*f1) or G.has_edge(*f2):
            continue
        # Also avoid creating parallel edges across the pair
        if _ek(*f1) == _ek(*f2):
            continue
        return ( (a,b), (c,d), f1, f2 )
    return None

def _khop_neighborhoods(G, k):
    """
    Precompute closed k-hop neighborhoods (excluding the center itself).
    """
    N = {}
    for u in G.nodes():
        dists = nx.single_source_shortest_path_length(G, u, cutoff=k)
        N[u] = {v for v, dist in dists.items() if 0 < dist <= k}
    return N

def _within_k(u, v, k, neighborhoods, G_current, locality_reference, cache_dynamic):
    """
    Check if dist(u, v) <= k according to chosen reference.
    """
    if k is None:
        return True
    if locality_reference == "initial":
        return v in neighborhoods[u]
    # dynamic: compute on-demand BFS (cached per (anchor, k))
    key = (u, k)
    if key not in cache_dynamic:
        dists = nx.single_source_shortest_path_length(G_current, u, cutoff=k)
        cache_dynamic[key] = {x for x, dist in dists.items() if 0 < dist <= k}
    return v in cache_dynamic[key]

def _propose_swap_with_locality(
    G, rng, k, neighborhoods, locality_reference, max_tries=256
):
    """
    Propose a valid 2-edge swap (e1,e2)->(f1,f2) that respects k-hop locality.
    Returns (e1, e2, f1, f2) or None.
    """
    E = list(G.edges())
    m = len(E)
    if m < 2: return None
    dyn_cache = {}  # for dynamic k-hop lookups

    for _ in range(max_tries):
        (a, b) = E[rng.randrange(m)]
        (c, d) = E[rng.randrange(m)]
        if len({a, b, c, d}) != 4:
            continue

        # Two orientations; try the one sampled first, fall back to the other
        for (f1, f2) in ( ((a, c), (b, d)), ((a, d), (b, c)) if rng.random() < 0.5 else ((a, d), (b, c),), ):
            # simple-edge constraints
            if f1[0] == f1[1] or f2[0] == f2[1]:
                continue
            if G.has_edge(*f1) or G.has_edge(*f2):
                continue
            if _ek(*f1) == _ek(*f2):
                continue
            # k-hop locality constraints
            if not _within_k(f1[0], f1[1], k, neighborhoods, G, locality_reference, dyn_cache):
                continue
            if not _within_k(f2[0], f2[1], k, neighborhoods, G, locality_reference, dyn_cache):
                continue
            return ( (a,b), (c,d), f1, f2 )

    return None

# ---------- main routine ----------

def transform_to_hh_via_stochastic_rewiring(
    G,
    max_steps=10000,
    beta=3.0,           # bias toward HH edges
    T0=1.0,             # initial temperature
    cooling=0.995,      # simulated annealing cooling per accepted step
    ensure_connected=True,
    k_hop=None,         # e.g., 2 or 3 to preserve locality; None disables
    locality_reference="initial",  # "initial" (default) or "current"
    seed=None,
):
    """
    Stochastically transform G to its Havel–Hakimi realization using biased 2-edge swaps
    while (a) preserving degree sequence, (b) enforcing k-hop locality for *new* edges,
    and (c) rejecting swaps that break connectivity (if ensure_connected=True).
    """
    rng = random.Random(seed)
    Gc = G.copy()

    # Target HH graph and scoring
    H = hh_graph_from_G(Gc)
    H_set = {_ek(u, v) for u, v in H.edges()}
    def matches_in_H(edges): return sum(1 for e in edges if _ek(*e) in H_set)
    cur_matches = matches_in_H(Gc.edges())

    # Precompute k-hop neighborhoods on the chosen reference graph
    neighborhoods = None
    if k_hop is not None:
        ref_graph = G if locality_reference == "initial" else Gc
        neighborhoods = _khop_neighborhoods(ref_graph, k_hop)

    T = T0
    traj = []
    m = Gc.number_of_edges()

    for _ in range(max_steps):
        prop = _propose_swap_with_locality(
            Gc, rng, k_hop, neighborhoods, locality_reference, max_tries=256
        )
        if prop is None:
            # no valid locality-respecting swap found under the budget
            break

        (e1, e2, f1, f2) = prop
        before = int(_ek(*e1) in H_set) + int(_ek(*e2) in H_set)
        after  = int(_ek(*f1) in H_set) + int(_ek(*f2) in H_set)
        dmatches = after - before

        # Metropolis acceptance (symmetric proposals)
        accept = (dmatches >= 0) or (rng.random() < math.exp(beta * dmatches / max(T, 1e-8)))
        if not accept:
            continue

        # Tentatively apply and enforce connectivity
        Gc.remove_edges_from([e1, e2])
        Gc.add_edges_from([f1, f2])

        if ensure_connected and not nx.is_connected(Gc):
            # revert if it breaks connectivity
            Gc.remove_edges_from([f1, f2])
            Gc.add_edges_from([e1, e2])
            continue
        else:
            traj.append((Gc.copy(),(f1, f2),(e1, e2)))

        # Commit
        cur_matches += dmatches
        T *= cooling

        if cur_matches == m:  # reached HH exactly
            break

        # If using dynamic locality, refresh neighborhoods occasionally (cheap heuristic)
        if k_hop is not None and locality_reference == "current":
            # Only recompute for touched nodes to keep it light
            for u in {e1[0], e1[1], e2[0], e2[1], f1[0], f1[1], f2[0], f2[1]}:
                dists = nx.single_source_shortest_path_length(Gc, u, cutoff=k_hop)
                neighborhoods[u] = {x for x, dist in dists.items() if 0 < dist <= k_hop}
    return traj


def train_grapher(model, graphs, num_epochs, learning_rate, T, k_eigen,device):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for G in graphs:
            # --- Corrupt graph with t edge rewirings ---
            traj = transform_to_hh_via_stochastic_rewiring(G, G.number_of_edges())
            step = 1
            for G_corrupt, added_pair, removed_pair in traj:
                step += 1
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

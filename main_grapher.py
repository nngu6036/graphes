import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import os
import argparse
import toml
import math
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
from sklearn.metrics.pairwise import rbf_kernel
from torch_geometric.nn import GINConv, global_mean_pool
from scipy.stats import wasserstein_distance
from networkx.algorithms.graphlet import graphlet_degree_vectors
from .main_msvae import MSVAE


class GINPredictor(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_layer):
        super().__init__()
        self.gin_layers = nn.ModuleList([
            GINConv(nn.Sequential(nn.Linear(in_channels if i == 0 else hidden_dim, hidden_dim),
                                  nn.ReLU(),
                                  nn.Linear(hidden_dim, hidden_dim)))
            for i in range(num_layer)
        ])
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, edge_index, edge_pairs, candidate_edges):
        for gin in self.gin_layers:
            x = gin(x, edge_index)
        
        u, v = edge_pairs[:, 0], edge_pairs[:, 1]
        x_u, x_v = x[u], x[v]
        target_edge = torch.cat([x_u, x_v], dim=-1)

        scores = []
        for edge in candidate_edges:
            i, j = edge[0], edge[1]
            xi, xj = x[i], x[j]
            concat = torch.cat([xi, xj], dim=-1)
            score = self.edge_predictor(torch.cat([target_edge, concat], dim=-1))
            scores.append(score)
        logits = torch.cat(scores, dim=1)  # shape: (batch_size, num_candidates)
        return logits

    def save_model(self, file_path):
        torch.save(self.state_dict(), file_path)

    def load_model(self, file_path):
        self.load_state_dict(torch.load(file_path))
        self.eval()

    def generate(self, num_samples, msvae_model, num_steps=10):
        self.eval()
        degree_seqs = msvae_model.generate(num_samples)
        generated_graphs = []

        for seq in degree_seqs:
            degrees = decode_degree_sequence(seq)
            G = configuration_model_from_multiset(degrees)

            for _ in range(num_steps):
                edges = list(G.edges())
                if len(edges) < 2:
                    break

                e1 = random.choice(edges)
                u, v = e1
                edge_candidates = [e for e in edges if e != e1 and len(set(e + e1)) == 4]

                if not edge_candidates:
                    continue

                candidate_tensor = torch.tensor(edge_candidates, dtype=torch.long)
                edge_pair = torch.tensor([[u, v]], dtype=torch.long)

                data = graph_to_data(G)
                edge_index = data.edge_index

                scores = self(data.x, edge_index, edge_pair, candidate_tensor)
                top_idx = torch.argmax(scores, dim=1).item()
                x, y = candidate_tensor[top_idx].tolist()

                if not G.has_edge(u, x) and not G.has_edge(v, y):
                    G.remove_edges_from([e1, (x, y)])
                    G.add_edges_from([(u, x), (v, y)])
                elif not G.has_edge(u, y) and not G.has_edge(v, x):
                    G.remove_edges_from([e1, (x, y)])
                    G.add_edges_from([(u, y), (v, x)])

            generated_graphs.append(G)

        return generated_graphs

def rewire_edges(G, num_rewirings=10):
    edges = list(G.edges())
    for _ in range(num_rewirings):
        if len(edges) < 2:
            break
        e1, e2 = random.sample(edges, 2)
        u, v = e1
        x, y = e2
        if len({u, v, x, y}) == 4:
            if not G.has_edge(u, x) and not G.has_edge(v, y):
                G.remove_edges_from([e1, e2])
                G.add_edges_from([(u, x), (v, y)])
            elif not G.has_edge(u, y) and not G.has_edge(v, x):
                G.remove_edges_from([e1, e2])
                G.add_edges_from([(u, y), (v, x)])
        edges = list(G.edges())
    return G

def sample_edge_pairs(G, num_samples=16):
    edges = list(G.edges())
    edge_pairs = []
    for _ in range(num_samples):
        if len(edges) < 2:
            break
        e1, e2 = random.sample(edges, 2)
        u, v = e1
        x, y = e2
        edge_pairs.append((random.choice([u, v]), random.choice([x, y])))
    return torch.tensor(edge_pairs, dtype=torch.long)

def graph_to_data(G):
    for node in G.nodes:
        G.nodes[node]['x'] = [1.0]
    return from_networkx(G)

def decode_degree_sequence(seq):
    degrees = []
    for degree, count in enumerate(seq):
        degrees.extend([degree] * int(count))
    return degrees

def configuration_model_from_multiset(degrees):
    G = nx.configuration_model(degrees)
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    return G

def gaussian_emd_kernel(X, Y, sigma=1.0):
    K = np.zeros((len(X), len(Y)))
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            emd = wasserstein_distance(np.arange(len(x)), np.arange(len(y)), x, y)
            K[i, j] = np.exp(-emd**2 / (2 * sigma**2))
    return K

def compute_mmd_degree_emd(graphs_1, graphs_2, max_degree=None, sigma=1.0):
    def degree_histogram(graphs, max_degree):
        histograms = []
        for G in graphs:
            degree_sequence = [deg for _, deg in G.degree()]
            hist = np.zeros(max_degree + 1)
            for deg in degree_sequence:
                if deg <= max_degree:
                    hist[deg] += 1
            if hist.sum() == 0:
                hist[0] = 1.0
            hist /= hist.sum()
            histograms.append(hist)
        return np.array(histograms)

    if max_degree is None:
        max_d1 = max((max(dict(G.degree()).values()) if len(G) > 0 else 0) for G in graphs_1)
        max_d2 = max((max(dict(G.degree()).values()) if len(G) > 0 else 0) for G in graphs_2)
        max_degree = max(max_d1, max_d2)

    H1 = degree_histogram(graphs_1, max_degree)
    H2 = degree_histogram(graphs_2, max_degree)

    K_xx = gaussian_emd_kernel(H1, H1, sigma)
    K_yy = gaussian_emd_kernel(H2, H2, sigma)
    K_xy = gaussian_emd_kernel(H1, H2, sigma)

    mmd = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
    return float(mmd)

def compute_mmd_cluster(graphs_1, graphs_2, bins=10, sigma=1.0):
    def clustering_histogram(graphs, bins=10):
        histograms = []
        for G in graphs:
            clustering = list(nx.clustering(G).values())
            hist, _ = np.histogram(clustering, bins=bins, range=(0, 1), density=True)
            if hist.sum() == 0:
                hist[0] = 1.0
            hist /= hist.sum()
            histograms.append(hist)
        return np.array(histograms)

    H1 = clustering_histogram(graphs_1, bins)
    H2 = clustering_histogram(graphs_2, bins)

    K_xx = gaussian_emd_kernel(H1, H1, sigma)
    K_yy = gaussian_emd_kernel(H2, H2, sigma)
    K_xy = gaussian_emd_kernel(H1, H2, sigma)

    mmd = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
    return float(mmd)


def compute_mmd_orbit(graphs_1, graphs_2, sigma=1.0):
    def orbit_histogram(graphs):
        histograms = []
        for G in graphs:
            try:
                gdv = graphlet_degree_vectors(G, 4)
                counts = np.array([v for vec in gdv.values() for v in vec])
                if len(counts) == 0:
                    counts = np.zeros(1)
            except Exception:
                counts = np.zeros(1)
            counts = counts.astype(float)
            counts /= counts.sum() + 1e-8  # normalize to make it a histogram
            histograms.append(counts)

        max_len = max(len(h) for h in histograms)
        padded = [np.pad(h, (0, max_len - len(h))) for h in histograms]
        return np.array(padded)

    H1 = orbit_histogram(graphs_1)
    H2 = orbit_histogram(graphs_2)

    K_xx = gaussian_emd_kernel(H1, H1, sigma)
    K_yy = gaussian_emd_kernel(H2, H2, sigma)
    K_xy = gaussian_emd_kernel(H1, H2, sigma)

    mmd = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
    return float(mmd)
    
def load_graph_from_directory(directory_path):
    max_node = 0 
    graphs = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            graph = nx.read_edgelist(file_path, nodetype=int)
            max_node = max(max_node, graph.number_of_nodes())
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            graph = nx.read_edgelist(file_path, nodetype=int)
            graph = nx.convert_node_labels_to_integers(graph)
            graphs.append(graph)
    return torch.stack(graphs), max_node


def train_grapher(model, graphs, num_epochs, learning_rate, max_node):
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for G in graphs:
            # Simulate graph and diffusion corruption
            G_corrupted = rewire_edges(G.copy())

            data = graph_to_data(G_corrupted)
            edge_pairs = sample_edge_pairs(G_corrupted, num_samples=16)

            # Simulate ground-truth labels (dummy 0/1 for now)
            labels = torch.randint(0, 2, (edge_pairs.size(0),), dtype=torch.float)

            pred_scores = model(data.x, data.edge_index, edge_pairs)
            loss = criterion(pred_scores, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")

def loss_function(target_freq_vec, frequencies,mean, logvar, weights, epoch,max_node):
    recon_weight, kl_weight, erdos_gallai_weight = weights.get('reconstruction', 1.0), weights.get('kl_divergence', 1.0),weights.get('erdos_gallai', 1.0)
    recon_weight = max(0.1, recon_weight * (0.95 ** epoch))
    kl_weight = min(kl_weight, epoch / 10)  
    recon_loss = torch.sum( (target_freq_vec - frequencies)**2)
    kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    erdos_gallai_loss = eg_loss(frequencies, max_node)
    total_loss = recon_weight * recon_loss + kl_weight * kl_loss + erdos_gallai_weight * erdos_gallai_loss
    return total_loss


def evaluate_generated_graphs(generated_graphs, test_graphs):
    mmd_degree = compute_mmd_degree(generated_graphs, test_graphs)
    mmd_cluster = compute_mmd_cluster(generated_graphs, test_graphs)
    mmd_orbit = compute_mmd_orbit(generated_graphs, test_graphs)
    return {
        "MMD Degree": mmd_degree,
        "MMS Cluster": mmd_cluster,
        "MMD Orbit": mmd_orbit,
    }

def load_msvae_from_file(max_node,config, model_path):
    hidden_dim = config['training']['hidden_dim']
    latent_dim = config['training']['latent_dim']
    model = MSVAE(max_input_dim=max_node, hidden_dim=hidden_dim, latent_dim=latent_dim)
    model.load_model(model_path)
    return model

def main(args):
    config_dir = Path("configs")
    dataset_dir = Path("datasets") / args.dataset_dir
    model_dir = Path("models")
    config_file = config_dir / args.config_file
    msvae_config_file = config_dir / args.msvae_config_file
    msvae_model_file = config_dir / args.msvae_model
    config = toml.load(config_file)
    msvae_config = toml.load(msvae_config_file)
    graphs, max_node = load_graph_from_directory(dataset_dir)
    train_graphs, test_graphs = train_test_split(graphs, test_size=0.2, random_state=42)
    msvae_model  = load_msvae_from_file(max_node, msvae_config,msvae_model_file)
    hidden_dim = config['training']['hidden_dim']
    latent_dim = config['training']['latent_dim']
    num_layer = config['training']['num_layer']
    model = GINPredictor(in_channels=max_node, hidden_dim=hidden_dim,num_layer=num_layer)
    
    else:
        num_epochs = config['training']['num_epochs']
        learning_rate = config['training']['learning_rate']
        train_grapher(model, train_graphs, num_epochs, learning_rate, max_node)
    if args.output_model:
        model.save_model(model_dir / args.output_model)
        print(f"Model saved to {args.output_model}")
    if args.evaluate:
        model.eval()
        generated_graphs = model.generate(config['inference']['generate_samples'])
        print(f"Evaluate generated graphs")
        metrics =  evaluate_generated_graphs(generated_graphs, test_graphs)
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GRAPH-ER Model')
    parser.add_argument('--dataset-dir', type=str, required=True,help='Path to the directory containing graph files')
    parser.add_argument('--config-file', type=str, required=True, help='Path to the configuration file in TOML format of Graph-ER')
    parser.add_argument('--msvae-config-file', type=str, required=True, help='Path to the configuration file in TOML format of MS-VAE')
    parser.add_argument('--msvae-model', type=str, required=True,help='Path to load a pre-trained MS-VAE model')
    parser.add_argument('--output-model', type=str, help='Path to save the trained model')
    parser.add_argument('--input-model', type=str, help='Path to load a pre-trained model')
    args = parser.parse_args()
    main(args)

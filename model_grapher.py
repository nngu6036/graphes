import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool
import networkx as nx
from collections import Counter
import random

from utils import graph_to_data, check_sequence_validity

def get_edge_representation(x, u, v, method="sum_absdiff"):
    x_u, x_v = x[u], x[v]
    if method == "mean":
        return (x_u + x_v) / 2
    elif method == "sum":
        return x_u + x_v
    elif method == "max":
        return torch.max(x_u, x_v)
    elif method == "sum_absdiff":
        return torch.cat([x_u + x_v, torch.abs(x_u - x_v)], dim=-1)
    else:
        return torch.cat([x_u, x_v], dim=-1)

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

def havel_hakimi_construction(degree_sequence):
    """
    Constructs a simple undirected graph from a given degree sequence
    using the Havel-Hakimi algorithm.

    Args:
        degree_sequence (list): A list of non-negative integers sorted in non-increasing order.

    Returns:
        G (nx.Graph): A simple undirected graph with the given degree sequence,
                      or None if the sequence is not graphical.
    """
    # Check if the degree sequence is graphical
    if not nx.is_valid_degree_sequence_havel_hakimi(degree_sequence):
        print("The degree sequence is not graphical.")
        return None

    # Make a copy to avoid modifying the original
    deg_seq = list(degree_sequence)
    nodes = list(range(len(deg_seq)))
    G = nx.Graph()
    G.add_nodes_from(nodes)

    while any(deg_seq):
        # Sort nodes by remaining degree (descending)
        node_deg_pairs = sorted(zip(nodes, deg_seq), key=lambda x: -x[1])
        u, du = node_deg_pairs[0]
        nodes = [x for x, _ in node_deg_pairs]
        deg_seq = [d for _, d in node_deg_pairs]

        # Take the top node and connect to next 'du' nodes
        for i in range(1, du + 1):
            v = nodes[i]
            G.add_edge(u, v)
            deg_seq[i] -= 1

        deg_seq[0] = 0  # All of u's degree is used
        # Remove nodes with 0 degree for next round
        nodes = [n for n, d in zip(nodes, deg_seq) if d > 0]
        deg_seq = [d for d in deg_seq if d > 0]

    return G

class GraphER(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_layer):
        super().__init__()
        self.gin_layers = nn.ModuleList([
            GINConv(nn.Sequential(nn.Linear(in_channels if i == 0 else hidden_dim, hidden_dim),
                                  nn.ReLU(),
                                  nn.Linear(hidden_dim, hidden_dim)))
            for i in range(num_layer)
        ])
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 4 + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x, edge_index ,first_edge ,candidate_edges, t):
        device = x.device
        for gin in self.gin_layers:
            x = gin(x, edge_index)
        first_edge_feat = get_edge_representation(x, first_edge[0], first_edge[1])
        t_tensor = torch.tensor([[t]], dtype=torch.float32, device=device)
        t_embed = self.time_embedding(t_tensor).squeeze(0) 
        scores = []
        for edge in candidate_edges:
            edge_feat = get_edge_representation(x, edge[0], edge[1])
            feat = torch.cat([first_edge_feat, edge_feat, t_embed], dim=-1)
            score = self.edge_predictor(feat)
            scores.append(score)
        logits = torch.cat(scores)
        return logits

    def save_model(self, file_path):
        torch.save(self.state_dict(), file_path)

    def load_model(self, file_path):
        self.load_state_dict(torch.load(file_path))
        self.eval()

    def generate(self, num_samples, num_steps, degree_sequences = None, msvae_model = None):
        self.eval()
        device = next(self.parameters()).device
        generated_graphs = []
        generated_seqs = degree_sequences if degree_sequences else msvae_model.generate(num_samples)
        for idx,seq in enumerate(generated_seqs):
            valid, _ = check_sequence_validity(seq)
            if not valid:
                print("Invalid degree")
                continue
            G = havel_hakimi_construction(seq)
            if not G:
                continue
            print(f"Generating graph {idx+1}")
            pre_seq = [deg for _, deg in G.degree()]
            for t in reversed(range(num_steps +1)):
                edges = list(G.edges())
                u,v = random.choice(edges)
                edge_candidates = [e for e in edges if e != (u,v) and len(set(e + (u,v))) == 4]
                if not edge_candidates:
                    continue
                candidate_tensor = torch.tensor(edge_candidates, dtype=torch.long, device=device)
                data = graph_to_data(G).to(device)
                edge_index = data.edge_index
                scores = self(data.x, edge_index, (u,v), candidate_tensor, t)
                top_idx = torch.argmax(scores).item()
                x, y = candidate_tensor[top_idx].tolist()
                if not G.has_edge(u, x) and not G.has_edge(v, y):
                    G.remove_edges_from([(u,v), (x, y)])
                    G.add_edges_from([(u, x), (v, y)])
                elif not G.has_edge(u, y) and not G.has_edge(v, x):
                    G.remove_edges_from([(u,v), (x, y)])
                    G.add_edges_from([(u, y), (v, x)])
            post_seq = [deg for _, deg in G.degree()]
            if set(pre_seq) != set(post_seq):
                import pdb
                pdb.set_trace()
            generated_graphs.append(G)
        return generated_graphs
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool
import networkx as nx
from collections import Counter
import random
import math
import numpy as np

from utils import *

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


def initialize_graphs(method, seq):
    G = None
    if method == 'havei_hakimi':
        G = havel_hakimi_construction(seq)
    if method == 'configuration_model':
        G = configuration_model_from_multiset(seq)
    if method == 'constraint_configuration_model':
        G = constraint_configuration_model_from_multiset(seq)
    if not nx.is_connected(G):
        G = None
    return G

    
class GraphER(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_layer,T):
        super().__init__()
        self.hidden_dim = hidden_dim
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
        self.t_embed = nn.Embedding(T + 1, hidden_dim)
        nn.init.xavier_uniform_(self.t_embed.weight)

    def forward(self, x, edge_index, first_edge, candidate_edges, t):
        device = x.device
        for gin in self.gin_layers:
            x = gin(x, edge_index)
        first_edge_feat = get_edge_representation(x, first_edge[0], first_edge[1])
        t = int(t)
        T_max = self.t_embed.num_embeddings - 1
        t = max(0, min(t, T_max))
        t_embed = self.t_embed(torch.tensor([t], dtype=torch.long, device=device)).squeeze(0)
        scores = [self.edge_predictor(torch.cat([first_edge_feat,get_edge_representation(x, e[0], e[1]), t_embed], dim=-1)).squeeze(-1)                   for e in candidate_edges]
        return torch.stack(scores, dim=0)  # [num_candidates]

    def save_model(self, file_path):
        torch.save(self.state_dict(), file_path)

    def load_model(self, file_path):
        self.load_state_dict(torch.load(file_path))
        self.eval()

    def generate_from_sequences(self, num_steps, degree_sequences,k_eigen, method = 'constraint_configuration_model'):
        self.eval()
        device = next(self.parameters()).device
        generated_graphs = []
        generated_seqs = []
        initial_graphs = []
        for seq in degree_sequences:
            G = initialize_graphs(method, seq)
            if G:
                initial_graphs.append(G)
                generated_seqs.append(seq)
        for idx, G in enumerate(initial_graphs):
            print(f"Generating graph {idx + 1}")
            snapshots = []
            step_size = max(1, num_steps // 8)   # ~8 panels
            plot_index = num_steps
            for t in reversed(range(num_steps + 1)):
                edges = list(G.edges())
                if len(edges) < 2:
                    continue
                # Select a random anchor edge
                u, v = random.choice(edges)
                # Generate swappable candidates (disjoint with (u,v))
                uv = frozenset((u, v))
                all_candidates = [e for e in edges if frozenset(e) != uv and len(set(e + (u, v))) == 4]
                if not all_candidates:
                    continue
                data = graph_to_data(G,k_eigen).to(device)
                scores = self(data.x, data.edge_index, (u, v), all_candidates, t)
                top_idx = int(torch.argmax(scores).item())
                x_, y_ = all_candidates[top_idx]
                # Rewire using valid option that matches triangle analysis
                if not G.has_edge(u, x_) and not G.has_edge(v, y_):
                    G_try = G.copy()
                    G_try.remove_edges_from([(u, v), (x_, y_)])
                    G_try.add_edges_from([(u, x_), (v, y_)])
                    if nx.is_connected(G_try):
                        G = G_try
                    continue
                if not G.has_edge(u, y_) and not G.has_edge(v, x_):
                    G_try = G.copy()
                    G_try.remove_edges_from([(u, v), (x_, y_)])
                    G_try.add_edges_from([(u, y_), (v, x_)])
                    if nx.is_connected(G_try):
                        G = G_try
            generated_graphs.append(G)
            if not snapshots or snapshots[-1][1] != 0:
                snapshots.append((G.copy(), 0))

            # Save the evolution strip for this graph
            save_graph_evolution(snapshots, idx, out_dir="evolutions_seq")
        return generated_graphs, generated_seqs


    def generate_with_msvae(self, num_samples, num_steps, msvae_model,k_eigen,method = 'havei_hakimi'):
        self.eval()
        device = next(self.parameters()).device
        generated_graphs = []
        generated_seqs = []
        initial_graphs = []
        while len(generated_graphs) < num_samples:
            print(len(generated_graphs))
            degree_sequences = msvae_model.generate(num_samples)
            for idx, seq in enumerate(degree_sequences):
                valid, _ = check_sequence_validity(seq)
                if not valid:
                    continue
                G = initialize_graphs(method, seq) 
                if G:
                    initial_graphs.append(G)
                    generated_seqs.append(seq)
                    if len(initial_graphs) >= num_samples:
                        break
        for idx, G in enumerate(initial_graphs): 
            snapshots = []
            step_size = max(1, num_steps // 8)   # ~8 panels
            plot_index = num_steps
            print(f"Generating graph {idx + 1}")
            for t in reversed(range(num_steps + 1)):
                edges = list(G.edges())
                if len(edges) < 2:
                    continue
                if t == plot_index:
                    snapshots.append((G.copy(), t))  # store a copy of the graph and the step
                    plot_index -= step_size
                # Select a random anchor edge
                u, v = random.choice(edges)
                # Generate swappable candidates (disjoint with (u,v))
                uv = frozenset((u, v))
                all_candidates = [e for e in edges if frozenset(e) != uv and len(set(e + (u, v))) == 4]
                if not all_candidates:
                    continue
                data = graph_to_data(G,k_eigen).to(device)
                scores = self(data.x,data.edge_index,(u,v), all_candidates,t).squeeze(-1) 
                top_idx = torch.argmax(scores).item()
                x_, y_ = all_candidates[top_idx]
                # Rewire only if no duplicates
                if not G.has_edge(u, x_) and not G.has_edge(v, y_):
                    G_try = G.copy()
                    G_try.remove_edges_from([(u, v), (x_, y_)])
                    G_try.add_edges_from([(u, x_), (v, y_)])
                    if nx.is_connected(G_try):
                        G = G_try
                    continue
                if not G.has_edge(u, y_) and not G.has_edge(v, x_):
                    G_try = G.copy()
                    G_try.remove_edges_from([(u, v), (x_, y_)])
                    G_try.add_edges_from([(u, y_), (v, x_)])
                    if nx.is_connected(G_try):
                        G = G_try
            generated_graphs.append(G)
            if not snapshots or snapshots[-1][1] != 0:
                snapshots.append((G.copy(), 0))

            # Save the evolution strip for this graph
            save_graph_evolution(snapshots, idx, out_dir=f"evolutions_{method}")
        return generated_graphs, generated_seqs

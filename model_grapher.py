import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool
import networkx as nx
from collections import Counter
import random
import math
from numpy import np

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

def constraint_configuration_model_from_multiset(degree_sequence, max_retries=None, max_failures=1000):
    N = len(degree_sequence)
    if max_retries is None:
        max_retries = N
    for _ in range(max_retries):
        stubs = []
        for node, deg in enumerate(degree_sequence):
            stubs.extend([node] * deg)
        random.shuffle(stubs)
        G = nx.Graph()
        G.add_nodes_from(range(N))
        failures = 0
        while len(stubs) >= 2 and failures < max_failures:
            u = stubs.pop()
            v = stubs.pop()
            if u == v or G.has_edge(u, v):
                # Invalid pair: put them back and count as failure
                stubs.extend([u, v])
                random.shuffle(stubs)
                failures += 1
                continue
            G.add_edge(u, v)
            failures = 0  # Reset on success
        if sorted([d for _, d in G.degree()]) == sorted(degree_sequence):
            return G
    return None  # Failed to generate a valid graph

def configuration_model_from_multiset(degrees):
    G = nx.configuration_model(degrees)
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    return G

def get_sinusoidal_embedding(t, dim, max_period=10000):
    device = t.device
    half_dim = dim // 2
    freqs = torch.exp(
        -torch.arange(0, half_dim, dtype=torch.float32, device=device) * (math.log(max_period) / half_dim)
    )
    t = t.float().unsqueeze(-1)  # shape [1, 1]
    args = t * freqs  # shape [1, half_dim]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # shape [1, dim]
    return emb.squeeze(0)  # shape [dim]

def havel_hakimi_construction(degree_sequence):
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

def initialize_graphs(method, seq):
    if method == 'havei_hakimi':
        G = havel_hakimi_construction(seq)
    if method == 'configuration_model':
        G = configuration_model_from_multiset(seq)
    if method == 'constraint_configuration_model':
        G = constraint_configuration_model_from_multiset(seq)
    if G:
        edges = list(G.edges())
        np.random.shuffle(edges)
        for _ in range(10*len(edges)):
            e1, e2 = random.sample(edges, 2)
            u, v = e1
            x, y = e2
            if len({u, v, x, y}) != 4:
                continue
            # Option 1: (u,x), (v,y)
            if not G.has_edge(u, x) and not G.has_edge(v, y):
                G.remove_edges_from( ((u,v), (x,y)))
                G.add_edges_from(((u, x), (v, y)))
            # Option 2: (u,y), (v,x)
            if not G.has_edge(u, y) and not G.has_edge(v, x):
                G.remove_edges_from(((u,v), (x,y)))
                G.add_edges_from(((u, y), (v, x)))
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
        #t_tensor = torch.tensor([t], dtype=torch.float32, device=device)
        #t_embed = get_sinusoidal_embedding(t_tensor, dim=self.hidden_dim)  # [hidden_dim]
        t_tensor = torch.tensor([t], dtype=torch.long, device=device)  # or pass t directly if it's already a tensor
        t_embed = self.t_embed(t_tensor).squeeze(0)  # shape: [hidden_dim]
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

    def generate_without_msvae(self, num_steps, degree_sequences,k_eigen, method = 'constraint_configuration_model'):
        self.eval()
        device = next(self.parameters()).device
        generated_graphs = []
        generated_seqs = []
        initial_graphs = [initialize_graphs(method, seq) for seq in degree_sequences]
        initial_graphs = [G for G in initial_graphs if G]
        for idx, G in enumerate(initial_graphs):
            print(f"Generating graph {idx + 1}")
            swap = 0
            for t in reversed(range(num_steps + 1)):
                edges = list(G.edges())
                if len(edges) < 2:
                    continue
                # Select a random anchor edge
                u, v = random.choice(edges)
                # Generate swappable candidates (disjoint with (u,v))
                all_candidates = [
                    e for e in edges if e != (u, v) and len(set(e + (u, v))) == 4
                ]
                if not all_candidates:
                    continue
                data = graph_to_data(G,k_eigen).to(device)
                scores = self(data.x,data.edge_index,(u,v), all_candidates,t).squeeze(-1) 
                top_idx = torch.argmax(scores).item()
                x_, y_ = all_candidates[top_idx]
                # Rewire using valid option that matches triangle analysis
                if not G.has_edge(u, x_) and not G.has_edge(v, y_):
                    G.remove_edges_from([(u, v), (x_, y_)])
                    G.add_edges_from([(u, x_), (v, y_)])
                    swap +=1
                elif not G.has_edge(u, y_) and not G.has_edge(v, x_):
                    G.remove_edges_from([(u, v), (x_, y_)])
                    G.add_edges_from([(u, y_), (v, x_)])
                    swap +=1
            print('Swap',swap)
            generated_seqs.append([deg for _, deg in G.degree()])
            generated_graphs.append(G)
        return generated_graphs, generated_seqs

    def generate(self, num_samples, num_steps, msvae_model,k_eigen,method = 'constraint_configuration_model',threshold = 0.01):
        self.eval()
        device = next(self.parameters()).device
        generated_graphs = []
        generated_seqs = []
        initial_graphs = []
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
            print(f"Generating graph {idx + 1}")
            for t in reversed(range(num_steps + 1)):
                edges = list(G.edges())
                if len(edges) < 2:
                    continue
                # Select a random anchor edge
                u, v = random.choice(edges)
                # Generate swappable candidates (disjoint with (u,v))
                all_candidates = [
                    e for e in edges if e != (u, v) and len(set(e + (u, v))) == 4
                ]
                if not all_candidates:
                    continue
                data = graph_to_data(G,k_eigen).to(device)
                scores = self(data.x,data.edge_index,(u,v), all_candidates,t).squeeze(-1) 
                top_idx = torch.argmax(scores).item()
                x_, y_ = all_candidates[top_idx]
                # Rewire only if no duplicates
                if not G.has_edge(u, x_) and not G.has_edge(v, y_):
                    G.remove_edges_from([(u, v), (x_, y_)])
                    G.add_edges_from([(u, x_), (v, y_)])
                elif not G.has_edge(u, y_) and not G.has_edge(v, x_):
                    G.remove_edges_from([(u, v), (x_, y_)])
                    G.add_edges_from([(u, y_), (v, x_)])
            generated_graphs.append(G)
        return generated_graphs, generated_seqs

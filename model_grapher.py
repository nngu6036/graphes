import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool
import networkx as nx

from utils import graph_to_data

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
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, edge_index, edge_pairs, candidate_edges):
        for gin in self.gin_layers:
            x = gin(x, edge_index)
        u, v = edge_pairs[:, 0], edge_pairs[:, 1]
        x_u, x_v = x[u], x[v]
        target_edge = get_edge_representation(x, u, v)
        scores = []
        for edge in candidate_edges:
            i, j = edge[0], edge[1]
            edge_feat = get_edge_representation(x, torch.tensor([i]), torch.tensor([j]))
            score = self.edge_predictor(torch.cat([target_edge, edge_feat], dim=-1))
            scores.append(score)
        logits = torch.cat(scores, dim=1)  # shape: (batch_size, num_candidates)
        return logits

    def save_model(self, file_path):
        torch.save(self.state_dict(), file_path)

    def load_model(self, file_path):
        self.load_state_dict(torch.load(file_path))
        self.eval()

    def generate(self, num_samples, num_steps, msvae_model = None):
        self.eval()
        degree_seqs = msvae_model.generate(num_samples)
        generated_graphs = []

        for seq in degree_seqs:
            degrees = decode_degree_sequence(seq)
            G = configuration_model_from_multiset(degrees)
            for _ in range(num_steps):
                edges = list(G.edges())
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
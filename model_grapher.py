import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool
import networkx as nx
from collections import Counter
import random
import math
import numpy as np

from utils import (
    build_candidates,
    _rewire,
    graph_to_data,
    check_sequence_validity,
    deterministic_connected_havel_hakimi,
)

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


def get_sinusoidal_embedding(t, dim, max_period=10000):
    half_dim = dim // 2
    freqs = torch.exp(
        -torch.arange(0, half_dim, dtype=torch.float32) * (math.log(max_period) / half_dim)
    )
    t = t.float().unsqueeze(-1)  # shape [1, 1]
    args = t * freqs  # shape [1, half_dim]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # shape [1, dim]
    return emb.squeeze(0)  # shape [dim]

def initialize_graphs(method, seq):
    if method == 'havei_hakimi':
        G = deterministic_connected_havel_hakimi(seq = seq)
    if method == 'configuration_model':
        G = configuration_model_from_multiset(seq)
    return G
    
class GraphER(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_layer, T,num_energy_targets: int = 2,energy_hidden_dim: int = 64):
        """
        in_channels: we treat this as `k_eigen` (number of Laplacian PE dims).
        Actual node feature dim from graph_to_data is:
            1 (degree) + in_channels (k_eigen) = in_channels + 1
        """
        super().__init__()
        self.hidden_dim = hidden_dim

        # Actual input feature dimension = degree (1) + Laplacian PE (in_channels)
        node_in_dim = in_channels + 1

        self.gin_layers = nn.ModuleList([
            GINConv(
                nn.Sequential(
                    nn.Linear(node_in_dim if i == 0 else hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
            )
            for i in range(num_layer)
        ])

        # get_edge_representation("sum_absdiff") gives 2*hidden_dim per edge,
        # so [first_edge_feat (2h), edge_feat (2h), t_embed (h)] -> 5h total.
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 4 + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.t_embed = nn.Embedding(T + 1, hidden_dim)
        nn.init.xavier_uniform_(self.t_embed.weight)

        if num_energy_targets > 0:
            self.energy_head = nn.Sequential(
                nn.Linear(hidden_dim, energy_hidden_dim),
                nn.ReLU(),
                nn.Linear(energy_hidden_dim, num_energy_targets),
            )
        else:
            self.energy_head = None

    def forward(self, x, edge_index, first_edge, candidate_edges, t):
        # x: [num_nodes, 1 + k_eigen]
        for gin in self.gin_layers:
            x = gin(x, edge_index)

        # first edge feature
        first_edge_feat = get_edge_representation(x, first_edge[0], first_edge[1])
        # time embedding
        t_tensor = torch.tensor([t], dtype=torch.long, device=x.device)
        t_embed = self.t_embed(t_tensor).squeeze(0)  # [hidden_dim]

        scores = []
        for edge in candidate_edges:
            edge_feat = get_edge_representation(x, edge[0], edge[1])
            feat = torch.cat([first_edge_feat, edge_feat, t_embed], dim=-1)  # [5 * hidden_dim]
            score = self.edge_predictor(feat)
            scores.append(score)

        logits = torch.cat(scores, dim=0).squeeze(-1) 
        return logits

    # NEW: just run GIN and return node embeddings
    def encode_nodes(self, x, edge_index):
        for gin in self.gin_layers:
            x = gin(x, edge_index)
        return x  # [num_nodes, hidden_dim]

    # NEW: pooled graph embedding (single-graph batch)
    def encode_graph(self, x, edge_index):
        x = self.encode_nodes(x, edge_index)
        # single graph => batch of zeros
        batch = x.new_zeros(x.size(0), dtype=torch.long)
        g = global_mean_pool(x, batch)  # [1, hidden_dim]
        return g

    def save_model(self, file_path):
        torch.save(self.state_dict(), file_path)

    def load_model(self, file_path):
        self.load_state_dict(torch.load(file_path))
        self.eval()

    def predict_energy(self, x, edge_index):
        """
        x, edge_index: same as encode_graph
        Returns tensor of shape [num_energy_targets]
        """
        if self.energy_head is None:
            raise RuntimeError("energy_head is not defined (num_energy_targets=0).")
        g = self.encode_graph(x, edge_index)  # [1, hidden_dim]
        energy_pred = self.energy_head(g)      # [1, num_energy_targets]
        return energy_pred.squeeze(0)


    def generate(
        self,
        num_samples,
        num_steps,
        msvae_model,
        k_eigen,
        method: str = 'havei_hakimi',
        ensure_connected: bool = False,
        k_hop: int = 2,
    ):
        """
        Sample degree sequences from msvae_model, build initial graphs
        with `initialize_graphs`, then run a learned edge-rewiring process.

        Uses build_candidates(...) so that at each step the model only
        scores feasible partner edges (at least one valid orientation),
        and applies swaps via _rewire to preserve constraints.
        """
        self.eval()
        generated_graphs = []
        generated_seqs = []
        initial_graphs = []

        # 1) Sample degree sequences and build initial graphs
        degree_sequences = msvae_model.generate(num_samples)
        print("Initialize sequqnces")
        for idx,seq in  enumerate(degree_sequences):
            valid, _ = check_sequence_validity(seq)
            if not valid:
                continue

            G0 = initialize_graphs(method, seq)
            initial_graphs.append(G0)
            generated_seqs.append(seq)
            print("Initialize graph", idx)
            if len(initial_graphs) >= num_samples:
                break
        print("Initialize graphes")
        # 2) Reverse-time rewiring for each initial graph
        for idx, G0 in enumerate(initial_graphs):
            print(f"Generating graph {idx + 1}")
            G = G0.copy()

            for t in reversed(range(num_steps + 1)):
                edges = list(G.edges())
                if len(edges) < 2:
                    continue

                # Random anchor edge
                anchor = random.choice(edges)
                u, v = anchor

                # Feasible candidates under same constraints as training
                candidate_edges = build_candidates(
                    G,
                    anchor,
                    ensure_connected=ensure_connected,
                    k_hop=k_hop,
                )
                if not candidate_edges:
                    continue
                device = next(self.parameters()).device
                data = graph_to_data(G, k_eigen).to(device)
                scores = self(data.x, data.edge_index, anchor, candidate_edges, t=t)

                top_idx = torch.argmax(scores).item()
                e2 = candidate_edges[top_idx]

                # Apply the swap via _rewire to keep constraints consistent
                applied = False
                for orient in (0, 1):
                    out = _rewire(G, anchor, e2, orient, ensure_connected=ensure_connected)
                    if out is None:
                        continue
                    G_post, added_edges, removed_edges = out
                    G = G_post
                    applied = True
                    break

                if not applied:
                    # Should be rare; candidate was feasible in build_candidates but
                    # may fail here if the graph changed in the meantime
                    continue
            generated_graphs.append(G)

        return generated_graphs, generated_seqs

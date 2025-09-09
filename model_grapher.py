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

def decode_degree_sequence(seq):
    degrees = []
    for degree, count in enumerate(seq):
        degrees.extend([degree] * int(count))
    return degrees


def initialize_graphs(method, seq):
    G = None
    if method == 'havel_hakimi':
        G = nx.havel_hakimi_graph(seq)
    if method == 'configuration_model':
        G = configuration_model_from_multiset(seq)
    if method == 'constraint_configuration_model':
        G = constraint_configuration_model_from_multiset(seq)
    return G

class ResGINLayer(nn.Module):
    def __init__(self, in_dim, out_dim, eps_train=True, p_drop=0.1):
        super().__init__()
        nn1 = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(),
                            nn.Linear(out_dim, out_dim))
        self.gin = GINConv(nn1, train_eps=eps_train)
        self.bn = nn.BatchNorm1d(out_dim)
        self.drop = nn.Dropout(p_drop)
        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
    def forward(self, x, edge_index):
        h = self.gin(x, edge_index)
        h = self.bn(h)
        h = self.drop(h)
        return h + self.proj(x)

class TimeEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.W = nn.Linear(1, dim // 2, bias=False)
        self.P = nn.Parameter(torch.zeros(dim // 2))
        self.aff = nn.Linear(1, dim - dim // 2)
    def forward(self, t_scalar: int):
        t = torch.tensor([[float(t_scalar)]], device=self.P.device)
        sincos = torch.cat([torch.sin(self.W(t)+self.P), torch.cos(self.W(t)+self.P)], dim=-1)
        return torch.cat([sincos, self.aff(t)], dim=-1).squeeze(0)

class GraphER(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_layer,T):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gin_layers = nn.ModuleList([
            ResGINLayer(in_channels if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layer)
        ])
        self.time_embed_dim = hidden_dim          # was: time_embed_dim or hidden_dim  (undefined)
        self.t_embed = TimeEmbed(hidden_dim)
        # Input features = [h_u, h_v, h_x, h_y, t_embed]  ->  4*H + H_t
        in_feat = hidden_dim * 4 + self.time_embed_dim
        self.edge_predictor = nn.Sequential(
            nn.Linear(in_feat, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # partner *logit*
        )
        # Same input as partner head; outputs two logits:
        # orientation 0 => (u,x)&(v,y), orientation 1 => (u,y)&(v,x)
        self.orient_head = nn.Sequential(
            nn.Linear(in_feat, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # orientation logits
        )

    def _time_embed(self, t: int, device):
        return self.t_embed(int(t))  # just pass scalar; TimeEmbed builds the tensor inside

    @staticmethod
    def _edge_rep(h, a, b):
        """
        Make an edge representation from node embeddings h (n,H).
        Order-invariant variant (sum + abs diff) is a good default.
        """
        ha, hb = h[a], h[b]  # (H,), (H,)
        return torch.cat([ha + hb, torch.abs(ha - hb)], dim=-1)

    def _cheap_pair_feats(G, u, v):
        deg_u, deg_v = G.degree(u), G.degree(v)
        # neighbors
        Nu, Nv = set(G.neighbors(u)), set(G.neighbors(v))
        cn = len(Nu & Nv)                # common neighbors
        jacc = cn / max(1, len(Nu | Nv)) # Jaccard
        # local clustering proxies
        cu = nx.clustering(G, u); cv = nx.clustering(G, v)
        return torch.tensor([deg_u, deg_v, deg_u+deg_v, abs(deg_u-deg_v), cn, jacc, cu, cv], dtype=torch.float32)

    def forward(self, x, edge_index, first_edge, candidate_edges, t):
        """
        x: node features input to the GIN
        edge_index: PyG COO edges
        first_edge: tuple(int,int) anchor edge (u,v) (exists in G_t)
        candidate_edges: list[tuple(int,int)] disjoint with (u,v)
        t: int timestep
        Returns:
          partner_logits: FloatTensor [C]
          orient_logits:  FloatTensor [C,2]
        """
        device = x.device
        for gin in self.gin_layers:
            x = gin(x, edge_index)
        h = x                                        # (n, H)
        u, v = first_edge
        uv_repr = self._edge_rep(h, u, v)            # (2H)
        t_emb = self._time_embed(t, device)          # (H_t,)
        feats = []
        for (x1, y1) in candidate_edges:
            xy_repr = self._edge_rep(h, x1, y1)      # (2H)
            f = torch.cat([uv_repr, xy_repr, t_emb], dim=-1)  # (4H + H_t)
            feats.append(f)
        if len(feats) == 0:
            # no valid candidates: return empty tensors
            return (torch.empty(0, device=device), torch.empty(0, 2, device=device))
        Fmat = torch.stack(feats, dim=0)             # (C, 4H + H_t)

        partner_logits = self.edge_predictor(Fmat).squeeze(-1)  # (C,)
        orient_logits = self.orient_head(Fmat)                  # (C, 2)
        return partner_logits, orient_logits

    def save_model(self, file_path):
        torch.save(self.state_dict(), file_path)

    def load_model(self, file_path):
        self.load_state_dict(torch.load(file_path))
        self.eval()

    @torch.no_grad()
    def generate_from_sequences(self, degree_sequences,k_eigen, method = 'havel_hakimi'):
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
            num_steps = G.number_of_edges() 
            for t in reversed(range(num_steps + 1)):
                edges = list(G.edges())
            if len(edges) < 2:
                continue

            # 1) pick an anchor (tuple, not frozenset)
            u, v = random.choice(edges)
            anchor = (u, v)

            # 2) candidate partners: disjoint edges
            cand_edges = [e for e in edges if frozenset(e) != frozenset(anchor)
                          and len({e[0], e[1], u, v}) == 4]
            if not cand_edges:
                continue

            # 3) PyG data -> send to device
            data = graph_to_data(G, k_eigen)
            x, edge_index = data.x.to(device), data.edge_index.to(device)

            # 4) run model (use self, not "model")
            partner_logits, orient_logits = self(
                x=x, edge_index=edge_index,
                first_edge=anchor, candidate_edges=cand_edges, t=t
            )
            if partner_logits.numel() == 0:
                continue

            # 5) pick best partner then orientation
            order = torch.argsort(partner_logits, descending=True).tolist()
            committed = False
            for idx_best in order:
                partner = cand_edges[idx_best]
                oi = int(torch.argmax(orient_logits[idx_best]).item())
                if try_apply_swap_with_orientation(G, anchor, partner, oi, ensure_connected=True, k_hop=2):
                    committed = True
                    break
                if try_apply_swap_with_orientation(G, anchor, partner, 1-oi, ensure_connected=True, k_hop=2):
                    committed = True
                    break
            generated_graphs.append(G)
            # Save the graphs
            save_graph_evolution(generated_graphs, filename="seqs")
        return generated_graphs, generated_seqs

    @torch.no_grad()
    def generate_with_msvae(self, num_samples, num_steps, msvae_model,k_eigen,method = 'havel_hakimi'):
        self.eval()
        device = next(self.parameters()).device
        generated_graphs = []
        generated_seqs = []
        initial_graphs = []
        while len(initial_graphs) < num_samples:
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

                # 1) pick an anchor (tuple, not frozenset)
                u, v = random.choice(edges)
                anchor = (u, v)

                # 2) candidate partners: disjoint edges
                cand_edges = [e for e in edges if frozenset(e) != frozenset(anchor)
                              and len({e[0], e[1], u, v}) == 4]
                if not cand_edges:
                    continue

                # 3) PyG data -> send to device
                data = graph_to_data(G, k_eigen)
                x, edge_index = data.x.to(device), data.edge_index.to(device)

                # 4) run model (use self, not "model")
                partner_logits, orient_logits = self(
                    x=x, edge_index=edge_index,
                    first_edge=anchor, candidate_edges=cand_edges, t=t
                )
                if partner_logits.numel() == 0:
                    continue

                # 5) pick best partner then orientation
                order = torch.argsort(partner_logits, descending=True).tolist()
                committed = False
                for idx_best in order:
                    partner = cand_edges[idx_best]
                    oi = int(torch.argmax(orient_logits[idx_best]).item())
                    if try_apply_swap_with_orientation(G, anchor, partner, oi, ensure_connected=True, k_hop=2):
                        committed = True
                        break
                    if try_apply_swap_with_orientation(G, anchor, partner, 1-oi, ensure_connected=True, k_hop=2):
                        committed = True
                        break
            generated_graphs.append(G)

            # Save the evolution strip for this graph
            save_graph_evolution(generated_graphs, filename=f"msvae")
        return generated_graphs, generated_seqs

    @torch.no_grad()
    def generate_with_setvae(self, N_nodes, num_steps, setvae_model, k_eigen, method='havel_hakimi'):
        self.eval()
        device = next(self.parameters()).device
        generated_graphs = []
        generated_seqs = []
        initial_graphs = []
        degree_sequences = setvae_model.generate(N_nodes)
        for seq in degree_sequences:
            G = initialize_graphs(method, seq)
            if G:
                initial_graphs.append(G)
                generated_seqs.append(seq)
        for idx, G in enumerate(initial_graphs): 
            print(f"Generating graph {idx + 1}")
            for t in reversed(range(num_steps + 1)):
                edges = list(G.edges())
                if len(edges) < 2:
                    continue

                # 1) pick an anchor (tuple, not frozenset)
                u, v = random.choice(edges)
                anchor = (u, v)

                # 2) candidate partners: disjoint edges
                cand_edges = [e for e in edges if frozenset(e) != frozenset(anchor)
                              and len({e[0], e[1], u, v}) == 4]
                if not cand_edges:
                    continue

                # 3) PyG data -> send to device
                data = graph_to_data(G, k_eigen)
                x, edge_index = data.x.to(device), data.edge_index.to(device)

                # 4) run model (use self, not "model")
                partner_logits, orient_logits = self(
                    x=x, edge_index=edge_index,
                    first_edge=anchor, candidate_edges=cand_edges, t=t
                )
                if partner_logits.numel() == 0:
                    continue

                # 5) pick best partner then orientation
                order = torch.argsort(partner_logits, descending=True).tolist()
                committed = False
                for idx_best in order:
                    partner = cand_edges[idx_best]
                    oi = int(torch.argmax(orient_logits[idx_best]).item())
                    if try_apply_swap_with_orientation(G, anchor, partner, oi, ensure_connected=True, k_hop=2):
                        committed = True
                        break
                    if try_apply_swap_with_orientation(G, anchor, partner, 1-oi, ensure_connected=True, k_hop=2):
                        committed = True
                        break
            generated_graphs.append(G)
            # Save the evolution strip for this graph
            save_graph_evolution(generated_graphs, filename=f"setvae")
        return generated_graphs, generated_seqs


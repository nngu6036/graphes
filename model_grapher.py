# model_grapher.py  — Flow Matching variant of GraphER
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool
import networkx as nx
import random
import math
import torch_geometric
from torch_geometric.utils import from_networkx
import copy
from utils import *

# ---------- Small utilities ----------
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

def initialize_graphs(method, seq):
    if method == 'havel_hakimi':
        return nx.havel_hakimi_graph(seq)
    elif method == 'configuration_model':
        # Simple configuration model (no self-loops/multiedges)
        G = nx.configuration_model(seq)
        G = nx.Graph(G)
        G.remove_edges_from(nx.selfloop_edges(G))
        return G
    else:
        # default fallback to HH
        return nx.havel_hakimi_graph(seq)

# ---------- GraphERFlow ----------
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=128, depth=2, dropout=0.0):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth - 1):
            layers += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class GraphERFlow(nn.Module):
    """
    Flow-Matching GraphER:
      - Encoder (GIN) -> graph embedding h and node embeddings z
      - velocity_head(h, t_emb) -> latent velocity for FM
      - action_head(z, h, t_emb, e1, cand_e2) -> logits over legal partner edges
    """
    def __init__(self, in_channels, hidden_dim=128, num_layers=4, time_emb_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.time_emb_dim = time_emb_dim

        # GIN encoder (node -> z; pooled -> h)
        self.gin_layers = nn.ModuleList([
            GINConv(nn.Sequential(
                nn.Linear(in_channels if i == 0 else hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )) for i in range(num_layers)
        ])

        # time embedding (small learned table + sin/cos mix)
        self.t_embed = nn.Embedding(512, time_emb_dim)  # enough bins; we pass ceil(T*τ)
        nn.init.xavier_uniform_(self.t_embed.weight)

        # velocity head: input [h || t_emb] -> vel in R^{hidden_dim}
        self.velocity_head = MLP(hidden_dim + time_emb_dim, hidden_dim, hidden=hidden_dim, depth=2, dropout=0.0)

        self.edge_feat_mode = "sum_absdiff"  # matches get_edge_representation
        if self.edge_feat_mode == "sum_absdiff":
            self.edge_ctx_dim = 2 * hidden_dim
        else:
            # if you ever switch to 'sum' or 'mean', that's 1 * hidden_dim
            self.edge_ctx_dim = hidden_dim

        # feats = e2_ctx (2h) + e1_ctx (2h) + h (h) + t_emb (t)
        action_in = (2 * self.edge_ctx_dim) + hidden_dim + time_emb_dim  # = 5h + t if sum_absdiff
        self.action_head = MLP(action_in, 1, hidden=hidden_dim, depth=2, dropout=0.0)

        # optional progress head (remain steps)
        self.progress_head = MLP(hidden_dim, 1, hidden=hidden_dim, depth=2, dropout=0.0)

    # ---------- Encoder ----------
    def encode(self, x, edge_index, batch=None):
        z = x
        for gin in self.gin_layers:
            z = gin(z, edge_index)
        if batch is None:
            # single-graph case
            h = z.mean(dim=0, keepdim=True)
        else:
            h = global_mean_pool(z, batch)
        return z, h  # node embeddings, graph embeddings

    # ---------- Velocity ----------
    def velocity(self, h, t_idx):
        """
        h: [B, hidden_dim]
        t_idx: [B] int indices for time embedding (e.g., ceil(τ*T))
        """
        t_emb = self.t_embed(t_idx)  # [B, time_emb_dim]
        v = self.velocity_head(torch.cat([h, t_emb], dim=-1))  # [B, hidden_dim]
        return v, t_emb

    # ---------- Action scoring ----------
    def action_logits(self, z, h, t_emb, e1_uv, cand_e2_pairs):
        """
        z: [N, hidden_dim] node embeddings
        h: [B, hidden_dim] graph embeddings (assume B=1 if single graph)
        t_emb: [B, time_emb_dim]
        e1_uv: tuple (u,v)
        cand_e2_pairs: list[(x,y)]
        returns: logits tensor [K] aligned to cand_e2_pairs
        """
        if len(cand_e2_pairs) == 0:
            return torch.empty(0, device=z.device)

        u, v = e1_uv
        e1_ctx = get_edge_representation(z, u, v)  # [2*hidden] (sum_absdiff)
        Hv = torch.cat([h.squeeze(0), t_emb.squeeze(0)], dim=-1)  # [hidden + time]

        feats = []
        for (x, y) in cand_e2_pairs:
            e2_ctx = get_edge_representation(z, x, y)             # [2*hidden]
            feat = torch.cat([e2_ctx, e1_ctx, Hv], dim=-1)         # [2h + 2h + (h+t)] -> but e1_ctx is 2h, h is used only once
            feats.append(feat)
        feats = torch.stack(feats, dim=0)                          # [K, D]
        logits = self.action_head(feats).squeeze(-1)               # [K]
        return logits

    # ---------- Convenience ----------
    def save_model(self, file_path):
        torch.save(self.state_dict(), file_path)

    def load_model(self, file_path):
        self.load_state_dict(torch.load(file_path, map_location="cpu"))
        self.eval()

    # ---------- Inference (single graph) ----------
    @torch.no_grad()
    def fm_generate_from_degree(self, degree_sequence, num_steps=32, graph_init='havel_hakimi',
                                mask_builder=None, first_edge_sampler='uniform'):
        """
        Start from HH(deg) and perform 'num_steps' swaps; each step:
         - encode -> velocity (not used to move yet, but could gate policy)
         - pick first edge e1 (uniform), score partner edges with masks, apply best legal swap
        """
        device = next(self.parameters()).device

        generated_graphs, generated_seqs = [], []
        initial_graphs = []
        for seq in degree_sequences:
            G = initialize_graphs(method, seq)
            if G is not None:
                generated_seqs.append(seq)
        
            for step in range(num_steps, 0, -1):
                if len(G.edges()) < 2:
                    break
                edges = list(G.edges())
                # sample anchor edge
                e1 = random.choice(edges) if first_edge_sampler == 'uniform' else edges[0]
                # candidate partners (disjoint endpoints)
                cand = [(x, y) for (x, y) in edges if len({e1[0], e1[1], x, y}) == 4]
                if mask_builder is not None:
                    cand = mask_builder(G, e1, cand)  # filter illegal

                if not cand:
                    continue
                # build a trivial node feature (deg) for now; adapt if you have k_eigen features
                pyg = to_pyg_with_features(G, k_lap=8, heat_times=(0.1, 1.0))
                z, h = self.encode(pyg.x, pyg.edge_index)
                t_idx = torch.tensor([step], dtype=torch.long, device=z.device)
                v, t_emb = self.velocity(h, t_idx)  # not directly used here; could be used for scheduling

                logits = self.action_logits(z, h, t_emb, e1, cand)
                top = torch.argmax(logits).item()
                x, y = cand[top]

                # apply legal swap (prefer (u,x)+(v,y), else (u,y)+(v,x))
                u, v = e1
                if (u != x) and (v != y) and (not G.has_edge(u, x)) and (not G.has_edge(v, y)):
                    G.remove_edges_from([(u, v), (x, y)])
                    G.add_edges_from([(u, x), (v, y)])
                elif (u != y) and (v != x) and (not G.has_edge(u, y)) and (not G.has_edge(v, x)):
                    G.remove_edges_from([(u, v), (x, y)])
                    G.add_edges_from([(u, y), (v, x)])
            generated_graphs.append(G)
        return generated_graphs, generated_seqs 

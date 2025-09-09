# model_grapher.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv
import networkx as nx
import random
import numpy as np

from utils import graph_to_data, try_apply_swap_with_orientation, configuration_model_from_multiset, constraint_configuration_model_from_multiset, save_graphs

def decode_degree_sequence(seq):
    degrees = []
    for degree, count in enumerate(seq):
        degrees.extend([degree] * int(count))
    return degrees

def initialize_graphs(method, seq):
    if method == 'havel_hakimi':
        return nx.havel_hakimi_graph(seq)
    if method == 'configuration_model':
        return configuration_model_from_multiset(seq)
    if method == 'constraint_configuration_model':
        return constraint_configuration_model_from_multiset(seq)
    return None

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
        h = self.gin(x, edge_index)            # (n, out_dim)
        h = self.bn(h)                         # (n, out_dim)
        h = self.drop(h)                       # (n, out_dim)
        return h + self.proj(x)                # (n, out_dim)

class TimeEmbed(nn.Module):
    """Returns a vector of size dim for any scalar t."""
    def __init__(self, dim):
        super().__init__()
        self.W = nn.Linear(1, dim // 2, bias=False)       # produces dim//2 channels
        self.P = nn.Parameter(torch.zeros(dim // 2))      # phase

    def forward(self, t_scalar: int):
        t = torch.tensor([[float(t_scalar)]], device=self.P.device)    # (1,1)
        z = self.W(t) + self.P                                         # (1, dim//2)
        return torch.cat([torch.sin(z), torch.cos(z)], dim=-1).squeeze(0)  # (dim,)

class GraphER(nn.Module):
    """
    Dimensions:
      - Node feature in_channels = k_eigen (from utils.graph_to_data)
      - Hidden dim = H = hidden_dim
      - Edge rep = [h_u+h_v, |h_u-h_v|] ∈ R^{2H}
      - Time embed t_emb ∈ R^{H}
      - MLP input = [uv(2H), xy(2H), t_emb(H)] = R^{5H}
      - partner_logits: (C,), orient_logits: (C,2)
    """
    def __init__(self, in_channels, hidden_dim, num_layer, T):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Encoder: (n, in_channels) -> (n, H)
        self.gin_layers = nn.ModuleList([
            ResGINLayer(in_channels if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layer)
        ])

        # Time embedding
        self.time_embed_dim = hidden_dim
        self.t_embed = TimeEmbed(self.time_embed_dim)

        # Heads
        in_feat = hidden_dim * 4 + self.time_embed_dim  # 4H + H = 5H
        self.edge_predictor = nn.Sequential(
            nn.Linear(in_feat, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.orient_head = nn.Sequential(
            nn.Linear(in_feat, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

    def _time_embed(self, t: int, device):
        # TimeEmbed handles device internally; just pass scalar.
        return self.t_embed(int(t))                     # (H,)

    @staticmethod
    def _edge_rep(h, a, b):
        """Return 2H-dimensional edge representation from node embeddings h ∈ R^{n×H}."""
        ha, hb = h[a], h[b]                             # (H,), (H,)
        return torch.cat([ha + hb, torch.abs(ha - hb)], dim=-1)  # (2H,)

    def forward(self, x, edge_index, first_edge, candidate_edges, t):
        """
        x:           (n, in_channels)
        edge_index:  (2, E)
        first_edge:  (u,v)
        candidate_edges: list[(x,y)] disjoint with (u,v)
        t: int
        Returns:
          partner_logits: (C,)
          orient_logits:  (C, 2)
        """
        device = x.device
        # Encoder: (n,in) -> (n,H)
        for gin in self.gin_layers:
            x = gin(x, edge_index)
        h = x                                          # (n, H)

        u, v = first_edge
        uv_repr = self._edge_rep(h, u, v)              # (2H,)
        t_emb  = self._time_embed(t, device)           # (H,)

        feats = []
        for (x1, y1) in candidate_edges:
            xy_repr = self._edge_rep(h, x1, y1)        # (2H,)
            f = torch.cat([uv_repr, xy_repr, t_emb], dim=-1)  # (4H + H,)
            feats.append(f)

        if len(feats) == 0:
            return (torch.empty(0, device=device), torch.empty(0, 2, device=device))

        Fmat = torch.stack(feats, dim=0)               # (C, 5H)
        partner_logits = self.edge_predictor(Fmat).squeeze(-1)  # (C,)
        orient_logits  = self.orient_head(Fmat)                 # (C, 2)
        return partner_logits, orient_logits

    def save_model(self, file_path):
        torch.save(self.state_dict(), file_path)

    def load_model(self, file_path):
        self.load_state_dict(torch.load(file_path, map_location="cpu"))
        self.eval()

    @torch.no_grad()
    def generate_from_sequences(self, degree_sequences, k_eigen, method='havel_hakimi'):
        self.eval()
        device = next(self.parameters()).device
        generated_graphs, generated_seqs = [], []

        initial_graphs = []
        for seq in degree_sequences:
            G = initialize_graphs(method, seq)
            if G is not None:
                initial_graphs.append(G)
                generated_seqs.append(seq)

        for gi, G in enumerate(initial_graphs):
            num_steps = G.number_of_edges()
            for t in reversed(range(num_steps + 1)):   # iterate steps
                edges = list(G.edges())
                if len(edges) < 2:
                    continue

                # anchor (u,v) and disjoint candidates
                u, v = random.choice(edges)
                anchor = (u, v)
                cand_edges = [e for e in edges if frozenset(e) != frozenset(anchor)
                              and len({e[0], e[1], u, v}) == 4]
                if not cand_edges:
                    continue

                data = graph_to_data(G, k_eigen)
                x, edge_index = data.x.to(device), data.edge_index.to(device)

                partner_logits, orient_logits = self(
                    x=x, edge_index=edge_index,
                    first_edge=anchor, candidate_edges=cand_edges, t=t
                )
                if partner_logits.numel() == 0:
                    continue

                order = torch.argsort(partner_logits, descending=True).tolist()
                committed = False
                for idx_best in order:
                    partner = cand_edges[idx_best]
                    oi = int(torch.argmax(orient_logits[idx_best]).item())
                    if try_apply_swap_with_orientation(G, anchor, partner, oi, ensure_connected=True, k_hop=2):
                        committed = True
                        break
                    if try_apply_swap_with_orientation(G, anchor, partner, 1 - oi, ensure_connected=True, k_hop=2):
                        committed = True
                        break

            generated_graphs.append(G)

        # Save a panel of final graphs (optional)
        #if len(generated_graphs):
        #    save_graphs(generated_graphs, filename="seqs")
        return generated_graphs, generated_seqs

    @torch.no_grad()
    def generate_with_msvae(self, num_samples, num_steps, msvae_model, k_eigen, method='havel_hakimi'):
        self.eval()
        device = next(self.parameters()).device
        generated_graphs, generated_seqs = [], []

        # init graphs from MSVAE sequences
        initial_graphs = []
        while len(initial_graphs) < num_samples:
            degree_sequences = msvae_model.generate(num_samples)
            for seq in degree_sequences:
                valid, _ = check_sequence_validity(seq)
                if not valid:
                    continue
                G = initialize_graphs(method, seq)
                if G is not None:
                    initial_graphs.append(G)
                    generated_seqs.append(seq)
                    if len(initial_graphs) >= num_samples: break

        for gi, G in enumerate(initial_graphs):
            for t in reversed(range(num_steps + 1)):
                edges = list(G.edges())
                if len(edges) < 2:
                    continue
                u, v = random.choice(edges)
                anchor = (u, v)
                cand_edges = [e for e in edges if frozenset(e) != frozenset(anchor)
                              and len({e[0], e[1], u, v}) == 4]
                if not cand_edges:
                    continue

                data = graph_to_data(G, k_eigen)
                x, edge_index = data.x.to(device), data.edge_index.to(device)
                partner_logits, orient_logits = self(
                    x=x, edge_index=edge_index,
                    first_edge=anchor, candidate_edges=cand_edges, t=t
                )
                if partner_logits.numel() == 0:
                    continue

                order = torch.argsort(partner_logits, descending=True).tolist()
                committed = False
                for idx_best in order:
                    partner = cand_edges[idx_best]
                    oi = int(torch.argmax(orient_logits[idx_best]).item())
                    if try_apply_swap_with_orientation(G, anchor, partner, oi, ensure_connected=True, k_hop=2):
                        committed = True
                        break
                    if try_apply_swap_with_orientation(G, anchor, partner, 1 - oi, ensure_connected=True, k_hop=2):
                        committed = True
                        break
            generated_graphs.append(G)

        if len(generated_graphs):
            save_graphs(generated_graphs, filename="msvae")
        return generated_graphs, generated_seqs

    @torch.no_grad()
    def generate_with_setvae(self, N_nodes, num_steps, setvae_model, k_eigen, method='havel_hakimi'):
        self.eval()
        device = next(self.parameters()).device
        generated_graphs, generated_seqs = [], []

        degree_sequences = setvae_model.generate(N_nodes)
        initial_graphs = []
        for seq in degree_sequences:
            G = initialize_graphs(method, seq)
            if G is not None:
                initial_graphs.append(G); generated_seqs.append(seq)

        for gi, G in enumerate(initial_graphs):
            for t in reversed(range(num_steps + 1)):
                edges = list(G.edges())
                if len(edges) < 2:
                    continue
                u, v = random.choice(edges)
                anchor = (u, v)
                cand_edges = [e for e in edges if frozenset(e) != frozenset(anchor)
                              and len({e[0], e[1], u, v}) == 4]
                if not cand_edges:
                    continue

                data = graph_to_data(G, k_eigen)
                x, edge_index = data.x.to(device), data.edge_index.to(device)
                partner_logits, orient_logits = self(
                    x=x, edge_index=edge_index,
                    first_edge=anchor, candidate_edges=cand_edges, t=t
                )
                if partner_logits.numel() == 0:
                    continue

                order = torch.argsort(partner_logits, descending=True).tolist()
                committed = False
                for idx_best in order:
                    partner = cand_edges[idx_best]
                    oi = int(torch.argmax(orient_logits[idx_best]).item())
                    if try_apply_swap_with_orientation(G, anchor, partner, oi, ensure_connected=True, k_hop=2):
                        committed = True
                        break
                    if try_apply_swap_with_orientation(G, anchor, partner, 1 - oi, ensure_connected=True, k_hop=2):
                        committed = True
                        break
            generated_graphs.append(G)

        if len(generated_graphs):
            save_graphs(generated_graphs, filename="setvae")
        return generated_graphs, generated_seqs

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool
import networkx as nx
from collections import Counter
import random
import math
import numpy as np

from utils import graph_to_data, check_sequence_validity


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
    # Accept both names
    if method in ('havei_hakimi', 'havel_hakimi'):
        G = havel_hakimi_construction(seq)
    elif method == 'configuration_model':
        G = configuration_model_from_multiset(seq)
    elif method == 'constraint_configuration_model':
        G = constraint_configuration_model_from_multiset(seq)
    else:
        raise ValueError(f"Unknown init method: {method}")

    if G:
        # quick mixing
        for _ in range(10 * G.number_of_edges()):
            edges = list(G.edges())
            if len(edges) < 2:
                break
            e1, e2 = random.sample(edges, 2)
            u, v = e1
            x, y = e2
            if len({u, v, x, y}) != 4:
                continue
            if (not G.has_edge(u, x)) and (not G.has_edge(v, y)):
                G.remove_edges_from([(u, v), (x, y)])
                G.add_edges_from([(u, x), (v, y)])
            elif (not G.has_edge(u, y)) and (not G.has_edge(v, x)):
                G.remove_edges_from([(u, v), (x, y)])
                G.add_edges_from([(u, y), (v, x)])
    return G


def _score_swap_error(M: np.ndarray, terms):
    # unchanged
    base = float((M * M).sum())
    inner = 0.0
    delta_norm = 0.0
    cross = 0.0
    for s, (a, b) in terms:
        inner += s * _B_inner(M, a, b)
        delta_norm += (s * s) * 4.0
    for i in range(len(terms)):
        s1, (a1, b1) = terms[i]
        for j in range(i + 1, len(terms)):
            s2, (a2, b2) = terms[j]
            cross += 2.0 * s1 * s2 * _pair_inner(a1, b1, a2, b2)
    return base + 2.0 * inner + delta_norm + cross


def pick_second_edge_by_spectral(G, u, v, L_hat: np.ndarray):
    edges = list(G.edges())
    cands = [e for e in edges if len({u, v, e[0], e[1]}) == 4]
    if not cands:
        return None, None
    L = normalized_laplacian_dense(G)
    M = L - L_hat
    best = best_opt = None
    best_val = None
    for (x, y) in cands:
        termsA = [(-1, (u, v)), (-1, (x, y)), (+1, (u, x)), (+1, (v, y))]
        valA = _score_swap_error(M, termsA)
        termsB = [(-1, (u, v)), (-1, (x, y)), (+1, (u, y)), (+1, (v, x))]
        valB = _score_swap_error(M, termsB)
        if (best_val is None) or (valA < best_val) or (valB < best_val):
            if valA <= valB:
                best, best_opt, best_val = (x, y), (u, x, v, y), valA
            else:
                best, best_opt, best_val = (x, y), (u, y, v, x), valB
    return best, best_opt


class SpectralER(nn.Module):
    """
    Predict q(lambda_{t-1} | lambda_t, t, extras) as a diagonal Gaussian.
    """
    def __init__(self, k, hidden, T, extra_dim=3):
        super().__init__()
        self.k = k
        self.t_embed = nn.Embedding(T + 1, hidden)
        nn.init.xavier_uniform_(self.t_embed.weight)
        self.net = nn.Sequential(
            nn.Linear(k + hidden + extra_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * k),
        )

    def forward(self, lam_t: torch.Tensor, t: int, extra_feat: torch.Tensor):
        t_idx = min(int(t), self.t_embed.num_embeddings - 1)
        te = self.t_embed(torch.tensor([t_idx], device=lam_t.device)).squeeze(0)
        h = torch.cat([lam_t, te, extra_feat], dim=-1)
        out = self.net(h)
        mu, logvar = out[: self.k], out[self.k :]
        logvar = torch.clamp(logvar, min=-10.0, max=5.0)
        return mu, logvar

    @torch.no_grad()
    def sample(self, lam_t: torch.Tensor, t: int, extra_feat: torch.Tensor):
        mu, logvar = self(lam_t, t, extra_feat)
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        lam_tm1 = mu + eps * std
        return lam_tm1, mu, logvar

    def save_model(self, file_path):
        torch.save(self.state_dict(), file_path)

    def load_model(self, file_path):
        self.load_state_dict(torch.load(file_path, map_location="cpu"))
        self.eval()

    @torch.no_grad()
    def generate_with_msvae(self, num_samples, num_steps, msvae_model, k_eigen, method='constraint_configuration_model'):
        """
        Sample degree sequences from MS-VAE, initialize graphs, and run spectral reverse steps.
        """
        self.eval()
        device = next(self.parameters()).device

        generated_graphs, generated_seqs, initial_graphs = [], [], []
        degree_sequences = msvae_model.generate(num_samples)

        for seq in degree_sequences:
            valid, _ = check_sequence_validity(seq)
            if not valid:
                continue
            G = initialize_graphs(method, seq)
            if G is not None:
                initial_graphs.append(G)
                generated_seqs.append(seq)
                if len(initial_graphs) >= num_samples:
                    break

        for idx, G in enumerate(initial_graphs):
            print(f"[spectral] Generating graph {idx + 1}")
            for t in reversed(range(num_steps + 1)):
                edges = list(G.edges())
                if len(edges) < 2:
                    continue
                u, v = random.choice(edges)

                # (1) current spectrum
                lam_t_np, U_t = laplacian_eigs(G, k_eigen, normed=True)
                lam_t = torch.from_numpy(lam_t_np).to(device)

                # (extras) size features
                n = G.number_of_nodes()
                m_edges = G.number_of_edges()
                avg_deg = (2.0 * m_edges) / max(1, n)
                density = (2.0 * m_edges) / max(1, n * (n - 1))
                extra_feat = torch.tensor(
                    [math.log(max(n, 2)), avg_deg, density],
                    device=device, dtype=lam_t.dtype
                )

                # (2) predict / sample lambda_{t-1}
                lam_pred, mu, logvar = self.sample(lam_t, t, extra_feat)
                lam_pred_np = lam_pred.clamp_min(0.0).clamp_max(2.0).cpu().numpy()

                # (3) residual-preserving target L_{t-1}^hat
                L_t = normalized_laplacian_dense(G)
                # replace low-frequency block only:
                # L_hat = L_t - U diag(lam_t) U^T + U diag(lam_pred) U^T
                L_hat = (L_t
                         - (U_t @ np.diag(lam_t_np) @ U_t.T)
                         + (U_t @ np.diag(lam_pred_np) @ U_t.T)).astype(np.float64)

                # (4) pick best second edge
                choice, orient = pick_second_edge_by_spectral(G, u, v, L_hat)
                if choice is None:
                    continue
                x, y = choice
                p, q, r, s = orient
                if (not G.has_edge(p, q)) and (not G.has_edge(r, s)):
                    G.remove_edges_from([(u, v), (x, y)])
                    G.add_edges_from([(p, q), (r, s)])

            generated_graphs.append(G)

        return generated_graphs, generated_seqs
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool
import networkx as nx
import random, math, numpy as np
import matplotlib.pyplot as plt

from utils import save_graph_evolution,graph_to_data, check_sequence_validity, laplacian_eigs, normalized_laplacian_dense, _B_inner, _pair_inner


def constraint_configuration_model_from_multiset(degree_sequence, max_retries=None, max_failures=1000):
    N = len(degree_sequence)
    if max_retries is None:
        max_retries = N
    for _ in range(max_retries):
        stubs = []
        for node, deg in enumerate(degree_sequence):
            stubs.extend([node] * deg)
        random.shuffle(stubs)
        G = nx.Graph(); G.add_nodes_from(range(N))
        failures = 0
        while len(stubs) >= 2 and failures < max_failures:
            u, v = stubs.pop(), stubs.pop()
            if u == v or G.has_edge(u, v):
                stubs.extend([u, v])
                random.shuffle(stubs)
                failures += 1
                continue
            G.add_edge(u, v)
            failures = 0
        if sorted([d for _, d in G.degree()]) == sorted(degree_sequence):
            return G
    return None

def configuration_model_from_multiset(degrees):
    G = nx.configuration_model(degrees)
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    return G

def havel_hakimi_construction(degree_sequence):
    try:
        seq_sorted = sorted([int(d) for d in degree_sequence], reverse=True)
        if not nx.is_valid_degree_sequence_havel_hakimi(seq_sorted):
            return None
        return nx.generators.degree_seq.havel_hakimi_graph(seq_sorted, create_using=nx.Graph())
    except Exception:
        return None

def initialize_graphs(method, seq):
    if method in ('havei_hakimi', 'havel_hakimi'):
        G = havel_hakimi_construction(seq)
    elif method == 'configuration_model':
        G = configuration_model_from_multiset(seq)
    elif method == 'constraint_configuration_model':
        G = constraint_configuration_model_from_multiset(seq)
    else:
        raise ValueError(f"Unknown init method: {method}")
    if G is not None:
        for _ in range(10 * G.number_of_edges()):
            edges = list(G.edges())
            if len(edges) < 2: break
            e1, e2 = random.sample(edges, 2)
            u, v = e1; x, y = e2
            if len({u, v, x, y}) != 4: continue
            if (not G.has_edge(u, x)) and (not G.has_edge(v, y)):
                G.remove_edges_from([(u, v), (x, y)]); G.add_edges_from([(u, x), (v, y)])
            elif (not G.has_edge(u, y)) and (not G.has_edge(v, x)):
                G.remove_edges_from([(u, v), (x, y)]); G.add_edges_from([(u, y), (v, x)])
    return G

def _apply_swap_if_valid(G, u, v, x, y, p, q, r, s, preserve_connectivity=False):
    if G.has_edge(p, q) or G.has_edge(r, s):
        return False
    if preserve_connectivity:
        was_connected = nx.is_connected(G)
    G.remove_edges_from([(u, v), (x, y)])
    G.add_edges_from([(p, q), (r, s)])
    if preserve_connectivity and not nx.is_connected(G):
        G.remove_edges_from([(p, q), (r, s)])
        G.add_edges_from([(u, v), (x, y)])
        return False
    return True

def _score_swap_error(M: np.ndarray, terms):
    base = float((M * M).sum())
    inner = delta_norm = cross = 0.0
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
    if not cands: return None, None
    L = normalized_laplacian_dense(G)
    M = L - L_hat
    best = best_opt = None; best_val = None
    for (x, y) in cands:
        termsA = [(-1,(u,v)),(-1,(x,y)),(+1,(u,x)),(+1,(v,y))]
        termsB = [(-1,(u,v)),(-1,(x,y)),(+1,(u,y)),(+1,(v,x))]
        valA = _score_swap_error(M, termsA)
        valB = _score_swap_error(M, termsB)
        if best_val is None or valA < best_val or valB < best_val:
            if valA <= valB:
                best, best_opt, best_val = (x,y),(u,x,v,y),valA
            else:
                best, best_opt, best_val = (x,y),(u,y,v,x),valB
    return best, best_opt


def _pad_or_trim_tensor_1d(x: torch.Tensor, target_len: int):
    """
    Make x length==target_len by zero-padding or truncation (returns contiguous tensor).
    """
    cur = x.numel()
    if cur == target_len:
        return x.contiguous()
    if cur > target_len:
        return x[:target_len].contiguous()
    # pad with zeros
    pad = torch.zeros(target_len - cur, dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=0).contiguous()

class SpectralER(nn.Module):
    def __init__(self, k, hidden, T, extra_dim=3):
        super().__init__()
        self.k = k
        self.t_embed = nn.Embedding(T+1, hidden)
        nn.init.xavier_uniform_(self.t_embed.weight)
        self.net = nn.Sequential(
            nn.Linear(k+hidden+extra_dim, hidden),
            nn.SiLU(), nn.Linear(hidden, hidden),
            nn.SiLU(), nn.Linear(hidden, 2*k)
        )
    def forward(self, lam_t, t, extra_feat):
        t_idx = min(int(t), self.t_embed.num_embeddings-1)
        te = self.t_embed(torch.tensor([t_idx], device=lam_t.device)).squeeze(0)
        h = torch.cat([lam_t, te, extra_feat], dim=-1)
        out = self.net(h)
        mu, logvar = out[:self.k], out[self.k:]
        logvar = torch.clamp(logvar, min=-10.0, max=5.0)
        return mu, logvar

    @torch.no_grad()
    def sample(self, lam_t, t, extra_feat):
        mu, logvar = self(lam_t,t,extra_feat)
        std = (0.5*logvar).exp(); eps = torch.randn_like(std)
        return mu+eps*std, mu, logvar

    def save_model(self, file_path):
        torch.save(self.state_dict(), file_path)

    def load_model(self, file_path):
        self.load_state_dict(torch.load(file_path))
        self.eval()

    def generate_from_sequences(self, num_steps, degree_sequences, k_eigen, method='constraint_configuration_model'):
        self.eval()
        device = next(self.parameters()).device
        generated_graphs, generated_seqs, initial_graphs = [], [], []

        # Build initial graphs
        for seq in degree_sequences:
            G = initialize_graphs(method, seq)
            if G is not None:
                initial_graphs.append(G)
                generated_seqs.append(seq)

        for idx, G in enumerate(initial_graphs):
            print(f"[spectral] Generating graph {idx + 1}")

            # Reset snapshot collectors PER GRAPH
            snapshots = []
            step_size = max(1, num_steps // 8)   # ~8 panels
            plot_index = num_steps

            for t in reversed(range(num_steps + 1)):
                # snapshot before doing the step evaluation (so we capture t)
                if t == plot_index:
                    snapshots.append((G.copy(), t))  # store a copy of the graph and the step
                    plot_index -= step_size

                edges = list(G.edges())
                if len(edges) < 2:
                    continue
                u, v = random.choice(edges)

                # spectrum (clamp k to graph size)
                n = G.number_of_nodes()
                k_eff = min(k_eigen, max(1, n - 1))
                lam_t_np, U_t = laplacian_eigs(G, k_eff, normed=True)
                lam_t = torch.from_numpy(lam_t_np).to(device)

                lam_t_in = _pad_or_trim_tensor_1d(lam_t, self.k)


                # extras
                m_edges = G.number_of_edges()
                avg_deg = (2.0 * m_edges) / max(1, n)
                density = (2.0 * m_edges) / max(1, n * (n - 1))
                extra_feat = torch.tensor([math.log(max(n, 2)), avg_deg, density],
                                          device=device, dtype=lam_t_in.dtype)

                # predict lambda_{t-1}
                lam_pred, _, _ = self.sample(lam_t_in, t, extra_feat)
                lam_pred_np = lam_pred[:k_eff].clamp_min(0.0).clamp_max(2.0).cpu().numpy()

                # build L_hat
                L_t = normalized_laplacian_dense(G)
                L_hat = (L_t
                         - (U_t @ np.diag(lam_t_np) @ U_t.T)
                         + (U_t @ np.diag(lam_pred_np) @ U_t.T)).astype(np.float64)

                # best second edge and apply swap (connectivity-safe)
                choice, orient = pick_second_edge_by_spectral(G, u, v, L_hat)
                if choice is None:
                    continue
                x, y = choice
                p, q, r, s = orient
                _apply_swap_if_valid(G, u, v, x, y, p, q, r, s, preserve_connectivity=True)

            # Ensure we include the final state if not already captured
            if not snapshots or snapshots[-1][1] != 0:
                snapshots.append((G.copy(), 0))

            # Save the evolution strip for this graph
            save_graph_evolution(snapshots, idx, out_dir="evolutions")

            generated_graphs.append(G)

        return generated_graphs, generated_seqs

    @torch.no_grad()
    def generate_with_msvae(self, num_samples, num_steps, msvae_model, k_eigen, method='constraint_configuration_model'):
        """
        Sample degree sequences from MS-VAE/SetVAE, initialize graphs, and run spectral reverse steps.
        Saves a horizontal evolution strip per graph.
        """
        self.eval()
        device = next(self.parameters()).device

        generated_graphs, generated_seqs, initial_graphs = [], [], []

        # Try to be robust to either API: generate(num_samples) or generate(batch_size, N_nodes)
        try:
            degree_sequences = msvae_model.generate(num_samples)
        except TypeError:
            # Fallback: sample Ns; adjust as you like
            Ns = torch.randint(low=5, high=20, size=(num_samples,), dtype=torch.long, device=device)
            degree_sequences = msvae_model.generate(batch_size=num_samples, N_nodes=Ns)

        # Build initial graphs
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

            # Reset snapshot collectors PER GRAPH
            snapshots = []
            step_size = max(1, num_steps // 8)   # ~8 panels
            plot_index = num_steps

            for t in reversed(range(num_steps + 1)):
                if t == plot_index:
                    snapshots.append((G.copy(), t))
                    plot_index -= step_size

                edges = list(G.edges())
                if len(edges) < 2:
                    continue
                u, v = random.choice(edges)

                # spectrum (clamp k to graph size)
                n = G.number_of_nodes()
                k_eff = min(k_eigen, max(1, n - 1))
                lam_t_np, U_t = laplacian_eigs(G, k_eff, normed=True)
                lam_t = torch.from_numpy(lam_t_np).to(device)

                lam_t_in = _pad_or_trim_tensor_1d(lam_t, self.k)


                # extras
                m_edges = G.number_of_edges()
                avg_deg = (2.0 * m_edges) / max(1, n)
                density = (2.0 * m_edges) / max(1, n * (n - 1))
                extra_feat = torch.tensor([math.log(max(n, 2)), avg_deg, density],
                                          device=device, dtype=lam_t_in.dtype)

                # predict lambda_{t-1}
                lam_pred, _, _ = self.sample(lam_t_in, t, extra_feat)
                lam_pred_np = lam_pred[:k_eff].clamp_min(0.0).clamp_max(2.0).cpu().numpy()

                # build L_hat
                L_t = normalized_laplacian_dense(G)
                L_hat = (L_t
                         - (U_t @ np.diag(lam_t_np) @ U_t.T)
                         + (U_t @ np.diag(lam_pred_np) @ U_t.T)).astype(np.float64)

                # best second edge and apply swap (connectivity-safe)
                choice, orient = pick_second_edge_by_spectral(G, u, v, L_hat)
                if choice is None:
                    continue
                x, y = choice
                p, q, r, s = orient
                _apply_swap_if_valid(G, u, v, x, y, p, q, r, s, preserve_connectivity=True)

            # Ensure we include the final state if not already captured
            if not snapshots or snapshots[-1][1] != 0:
                snapshots.append((G.copy(), 0))

            # Save the evolution strip for this graph
            save_graph_evolution(snapshots, idx, out_dir="evolutions")

            generated_graphs.append(G)

        return generated_graphs, generated_seqs

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool
import networkx as nx
import random, math, numpy as np
import matplotlib.pyplot as plt

from utils import *

# ---- inner-products needed for fast Frobenius scoring of a double-edge swap ----
def _B_inner(M: np.ndarray, a: int, b: int) -> float:
    # <M, (e_a - e_b)(e_a - e_b)^T> = M_aa + M_bb - 2 M_ab
    return float(M[a, a] + M[b, b] - 2.0 * M[a, b])

def _pair_inner(a: int, b: int, c: int, d: int) -> float:
    # <B_ab, B_cd> = ( (e_a - e_b)^T (e_c - e_d) )^2
    z = (a == c) - (a == d) - (b == c) + (b == d)
    return float(z * z)


def _apply_swap_if_valid(G, u, v, x, y, p, q, r, s):
    if G.has_edge(p, q) or G.has_edge(r, s):
        return False
    G.remove_edges_from([(u, v), (x, y)])
    G.add_edges_from([(p, q), (r, s)])
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

def initialize_graphs(method, seq):
    G = None
    if method == 'havel_hakimi':
        G = nx.havel_hakimi_graph(seq)
    if method == 'configuration_model':
        G = configuration_model_from_multiset(seq)
    if method == 'constraint_configuration_model':
        G = constraint_configuration_model_from_multiset(seq)
    return G

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

    @torch.no_grad()
    def predict_prev_lambda_from_y(
        self,
        G_t: nx.Graph,
        step: int,
        k_eigen: int,
        device: torch.device,
        deterministic: bool = True,
        return_L_hat: bool = True,
        eps: float = 1e-6,
    ):
        lam_t_np, U_np = laplacian_eigs(G_t, k_eigen, normed=True)  # (K,), (n,K)
        V = G_t.number_of_nodes()
        E = G_t.number_of_edges()
        K = min(k_eigen, max(0, V - 1))
        assert K > 0, "Expect K>0 (no tiny graphs)."

        lam_t = torch.from_numpy(lam_t_np[:K]).to(device=device, dtype=torch.float32)
        lam_t_in = _pad_or_trim_tensor_1d(lam_t, self.k)  # <-- ensure correct input dim

        avg_deg = (2.0 * E) / max(1, V)
        density = (2.0 * E) / max(1, V * (V - 1))
        extra_feat = torch.tensor(
            [math.log(max(V, 2)), avg_deg, density],
            device=device, dtype=torch.float32,
        )

        mu_y, logvar_y = self(lam_t_in, step, extra_feat)  # shapes: (self.k,), (self.k,)
        mu_y      = mu_y[:K]       # <-- slice back to K valid dims
        logvar_y  = logvar_y[:K]

        if deterministic:
            y_hat = mu_y
        else:
            std = torch.exp(0.5 * logvar_y)
            y_hat = mu_y + std * torch.randn_like(std)

        lam_hat_prev = y_to_lam(y_hat, eps=eps)                 # (K,)
        lam_hat_prev_np = lam_hat_prev.detach().cpu().numpy()

        if return_L_hat:
            A = nx.to_scipy_sparse_array(G_t, dtype=float)
            L_t_np = csgraph.laplacian(A, normed=True).toarray()

            U_t = torch.from_numpy(U_np[:, :K]).to(device=device, dtype=torch.float32)
            L_t = torch.from_numpy(L_t_np).to(device=device, dtype=torch.float32)
            lam_t_th = lam_t
            lam_hat_prev_th = torch.from_numpy(lam_hat_prev_np).to(device=device, dtype=torch.float32)

            L_hat = L_t - U_t @ torch.diag(lam_t_th) @ U_t.T + U_t @ torch.diag(lam_hat_prev_th) @ U_t.T
            L_hat_np = L_hat.detach().cpu().numpy()
        else:
            L_hat_np = None

        return lam_hat_prev_np, U_np[:, :K], L_hat_np, lam_t_np[:K]

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
            for t in reversed(range(num_steps + 1)):
                edges = list(G.edges())
                if len(edges) < 2:
                    continue
                u, v = random.choice(edges)

                # --- new: predict y -> λ̂_{t-1} and build L̂ in one call
                _, _, L_hat, _ = self.predict_prev_lambda_from_y(
                    G_t=G, step=t, k_eigen=k_eigen, device=device,
                    deterministic=True, return_L_hat=True, eps=1e-6
                )
                if L_hat is None:
                    continue

                # best second edge and apply swap (connectivity-safe)
                choice, orient = pick_second_edge_by_spectral(G, u, v, L_hat)
                if choice is None:
                    continue
                x, y = choice
                p, q, r, s = orient
                orient_idx = 0 if (p, q, r, s) == (u, x, v, y) else 1
                try_apply_swap_with_orientation(
                    G, (u, v), (x, y), orient_idx,
                    ensure_connected=True, k_hop=None
                )
            generated_graphs.append(G)
        save_graphs(generated_graphs)
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
            for t in reversed(range(num_steps + 1)):
                edges = list(G.edges())
                if len(edges) < 2:
                    continue
                u, v = random.choice(edges)

                # --- new: predict y -> λ̂_{t-1} and build L̂ in one call
                _, _, L_hat, _ = self.predict_prev_lambda_from_y(
                    G_t=G, step=t, k_eigen=k_eigen, device=device,
                    deterministic=True, return_L_hat=True, eps=1e-6
                )
                if L_hat is None:
                    continue

                # best second edge and apply swap (connectivity-safe)
                choice, orient = pick_second_edge_by_spectral(G, u, v, L_hat)
                if choice is None:
                    continue
                x, y = choice
                p, q, r, s = orient
                orient_idx = 0 if (p, q, r, s) == (u, x, v, y) else 1
                try_apply_swap_with_orientation(
                    G, (u, v), (x, y), orient_idx,
                    ensure_connected=True, k_hop=None
                )
            generated_graphs.append(G)
        save_graphs(generated_graphs)
        return generated_graphs, generated_seqs


    @torch.no_grad()
    def generate_with_setvae(self, N_nodes, num_steps, setvae_model, k_eigen, method='constraint_configuration_model'):
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
            print(f"[spectral] Generating graph {idx + 1}")
            for t in reversed(range(num_steps + 1)):
                edges = list(G.edges())
                if len(edges) < 2:
                    continue
                u, v = random.choice(edges)

                # --- new: predict y -> λ̂_{t-1} and build L̂ in one call
                _, _, L_hat, _ = self.predict_prev_lambda_from_y(
                    G_t=G, step=t, k_eigen=k_eigen, device=device,
                    deterministic=True, return_L_hat=True, eps=1e-6
                )
                if L_hat is None:
                    continue

                # best second edge and apply swap (connectivity-safe)
                choice, orient = pick_second_edge_by_spectral(G, u, v, L_hat)
                if choice is None:
                    continue
                x, y = choice
                p, q, r, s = orient
                orient_idx = 0 if (p, q, r, s) == (u, x, v, y) else 1
                try_apply_swap_with_orientation(
                    G, (u, v), (x, y), orient_idx,
                    ensure_connected=True, k_hop=None
                )
            generated_graphs.append(G)
        save_graphs(generated_graphs)
        return generated_graphs, generated_seqs
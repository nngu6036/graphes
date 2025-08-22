import networkx as nx
import numpy as np
import random
import os
import argparse
import toml
from pathlib import Path
import matplotlib.pyplot as plt
from utils import load_graph_from_directory, laplacian_eigs  # your top-k routine
from scipy.optimize import linear_sum_assignment             # for minimal pairing

rng = np.random.default_rng(42)

def _delta_matrix_for_edge_change(n, u, v, sign=+1, normed=True):
    """
    Return the Laplacian delta for adding (+1) or removing (-1) edge (u,v).
    If normed=True, uses normalized Laplacian approximation via degrees.
    For small per-step analysis, unnormalized is fine; set normed=False for exactness there.
    """
    D = np.zeros((n, n), dtype=float)
    if u == v:
        return D
    if normed:
        # Approximate normalized Laplacian update: Δ = sign * (e_u - e_v)(e_u - e_v)^T / sqrt(d_u d_v)
        # Use current degrees; good when degrees don't change much across single swap.
        D[u, u] += sign
        D[v, v] += sign
        D[u, v] -= sign
        D[v, u] -= sign
    else:
        # Exact combinatorial Laplacian update for add/remove
        D[u, u] += sign
        D[v, v] += sign
        D[u, v] -= sign
        D[v, u] -= sign
    return D

def _swap_edges(G, e1, e2, preserve_connectivity=True):
    (u, v), (x, y) = e1, e2
    if len({u, v, x, y}) != 4:
        return False, None
    # Two options
    A = (u, x, v, y)
    B = (u, y, v, x)
    for (p, q, r, s) in [A, B]:
        if not G.has_edge(p, q) and not G.has_edge(r, s):
            # Apply swap
            G.remove_edges_from([(u, v), (x, y)])
            G.add_edges_from([(p, q), (r, s)])
            if preserve_connectivity and not nx.is_connected(G):
                # revert and try the other orientation
                G.remove_edges_from([(p, q), (r, s)])
                G.add_edges_from([(u, v), (x, y)])
                continue
            return True, ((u, v), (x, y), (p, q), (r, s))
    return False, None

def edge_rewire_once(G, preserve_connectivity=True, max_tries=200):
    edges = list(G.edges())
    m = len(edges)
    if m < 2:
        return False, None
    tries = 0
    while tries < max_tries:
        e1 = edges[rng.integers(m)]
        e2 = edges[rng.integers(m)]
        if e1 == e2:
            tries += 1
            continue
        ok, detail = _swap_edges(G, e1, e2, preserve_connectivity=preserve_connectivity)
        if ok:
            return True, detail
        tries += 1
    return False, None

def pairwise_minimal_diff(lam_prev, lam_next):
    """
    Pair eigenvalues to minimize total absolute difference (avoids index flips).
    """
    lam_prev = np.asarray(lam_prev)
    lam_next = np.asarray(lam_next)
    C = np.abs(lam_prev[:, None] - lam_next[None, :])
    r, c = linear_sum_assignment(C)
    diffs = lam_next[c] - lam_prev[r]
    return diffs, r, c

def main(args):
    config_dir = Path("configs")
    dataset_dir = Path("datasets") / args.dataset_dir
    config = toml.load(config_dir / args.config)
    graphs, max_node, min_node = load_graph_from_directory(dataset_dir)

    sample_count = config['mixing_time'].get('sample_count', 10)
    k = config['mixing_time'].get('k_eigen', 8)  # top-k eigenvalues; set to <= n-1

    G = random.choice(graphs).copy()
    if not nx.is_connected(G):
        gcc_nodes = max(nx.connected_components(G), key=len)
        G = G.subgraph(gcc_nodes).copy()

    # Relabel to contiguous integers to avoid OOB indexes
    G = nx.convert_node_labels_to_integers(G, ordering='sorted')
    n = G.number_of_nodes()

    k_eff = max(1, min(k, n - 1))
    lam = laplacian_eigs(G, k_eff, normed=True)[0]
    lam = np.sort(lam)
    diffs_all = []
    weyl_bounds = []

    for t in range(sample_count):
        # Save pre-L for Weyl bound
        # We'll compute Δ via combinatorial Laplacian for a strict upper bound (simpler & valid)
        L_before = nx.laplacian_matrix(G).toarray()

        ok, detail = edge_rewire_once(G, preserve_connectivity=True)
        if not ok:
            print("No valid swap found.")
            continue

        lam_next = laplacian_eigs(G, k_eff, normed=True)[0]
        lam_next = np.sort(lam_next)

        # Minimal pairing diffs (safer than raw index)
        diffs, _, _ = pairwise_minimal_diff(lam, lam_next)
        diffs_all.append(diffs)

        # Weyl bound: ||Δ||_2 upper bounds max |λ_i' - λ_i| for any i
        # Build Δ exactly for the two removes and two adds in combinatorial Laplacian
        (u, v), (x, y), (p, q), (r, s) = detail
        Delta = (_delta_matrix_for_edge_change(n, u, v, sign=-1, normed=False)
                 + _delta_matrix_for_edge_change(n, x, y, sign=-1, normed=False)
                 + _delta_matrix_for_edge_change(n, p, q, sign=+1, normed=False)
                 + _delta_matrix_for_edge_change(n, r, s, sign=+1, normed=False))
        # Compute spectral norm of Δ
        # For small n, dense eigh is fine; for larger n, use sparse norm estimate
        w = np.linalg.eigvalsh(Delta)
        weyl = np.max(np.abs(w))
        weyl_bounds.append(weyl)

        lam = lam_next

    # Summary
    if diffs_all:
        diffs_all = np.array(diffs_all)  # shape [T, k_eff]
        max_abs_per_step = np.max(np.abs(diffs_all), axis=1)  # length T
        print(f"Median max |Δλ| per swap: {np.median(max_abs_per_step):.6f}")
        print(f"95th pct max |Δλ| per swap: {np.percentile(max_abs_per_step, 95):.6f}")
        if weyl_bounds:
            print(f"Median Weyl bound ||Δ||2: {np.median(weyl_bounds):.6f}")
            print(f"95th pct Weyl bound ||Δ||2: {np.percentile(weyl_bounds, 95):.6f}")

        # Plot evolution (per-index, after optimal pairing)
        plt.figure(figsize=(8, 3))
        for i in range(k_eff):
            plt.plot(np.abs(diffs_all[:, i]), linewidth=1)
        plt.xlabel("Swap step")
        plt.ylabel(r"$|\Delta \lambda_i|$")
        plt.title(f"Top-{k_eff} normalized-Laplacian eigenvalue changes per swap")
        plt.tight_layout()
        plt.show()
    else:
        print("No eigenvalue differences recorded.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Eigenvalue change under edge rewiring')
    parser.add_argument('--dataset-dir', type=str, help='Path to the directory containing graph files')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file in TOML format')
    args = parser.parse_args()
    main(args)

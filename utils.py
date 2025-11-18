import torch
import os
import networkx as nx
from torch_geometric.utils.convert import from_networkx
import numpy as np
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix
import random
import math
from typing import Optional, Iterable
import matplotlib.pyplot as plt
from torch_geometric.data import Data

def load_degree_sequence_from_directory(directory_path):
    max_node = 0 
    max_edge = 0
    seqs = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            G = nx.read_edgelist(file_path, nodetype=int)
            max_node = max(max_node, G.number_of_nodes())
            max_edge = max(max_node, G.number_of_edges())
    print("Max node: ", max_node, " Max edge:", max_edge)
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            G = nx.read_edgelist(file_path, nodetype=int)
            G = nx.convert_node_labels_to_integers(G)
            seq = [deg for _, deg in G.degree()]
            if seq is not None:
                seqs.append(seq)
    return seqs, max_node

def load_graph_from_directory(directory_path):
    max_node = 0 
    graphs = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            graph = nx.read_edgelist(file_path, nodetype=int)
            max_node = max(max_node, graph.number_of_nodes())
    print("Max node: ", max_node)
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            graph = nx.read_edgelist(file_path, nodetype=int)
            graph = nx.convert_node_labels_to_integers(graph)
            graphs.append(graph)
    return graphs, max_node

def graph_to_data(G, k_gen):
    """
    Convert nx.Graph into PyG Data for GraphER.
    Includes:
        - node features
        - edge_index
        - valid disjoint edge pairs
        - optional positional encodings
    """

    # --------------------------------------
    # Node features (degree only for now)
    # --------------------------------------
    deg = np.array([d for _, d in G.degree()], dtype=np.float32)
    x = torch.tensor(deg).view(-1, 1)

    # --------------------------------------
    # Edge index
    # --------------------------------------
    edges = list(G.edges())
    ei = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # --------------------------------------
    # Build disjoint edge pairs
    # --------------------------------------
    valid_pairs = []
    m = len(edges)
    for i in range(m):
        u, v = edges[i]
        for j in range(i + 1, m):
            x2, y2 = edges[j]
            if len({u, v, x2, y2}) == 4:
                valid_pairs.append([i, j])

    if valid_pairs:
        edge_pairs = torch.tensor(valid_pairs, dtype=torch.long)
    else:
        edge_pairs = torch.empty((0, 2), dtype=torch.long)

    # --------------------------------------
    # Build PyG Data
    # --------------------------------------
    data = Data(
        x=x,                                   # node features
        edge_index=ei,                         # edges
        edge_pairs=edge_pairs,                 # candidate swap pairs
        num_nodes=G.number_of_nodes(),
        num_edges=G.number_of_edges(),
    )

    # --------------------------------------
    # Laplacian positional encoding
    # --------------------------------------
    A = nx.to_numpy_array(G, dtype=float)
    deg = A.sum(axis=1)
    deg_sqrt_inv = np.zeros_like(deg)
    deg_sqrt_inv[deg > 0] = 1.0 / np.sqrt(deg[deg > 0])

    # Compute D^{-1/2} * A * D^{-1/2}
    D_inv_sqrt = np.diag(deg_sqrt_inv)
    I = np.eye(G.number_of_nodes())

    L_dense = I - D_inv_sqrt @ A @ D_inv_sqrt
    eigvals, eigvecs = np.linalg.eigh(L_dense)

    k = min(k_gen, eigvecs.shape[1])
    pe = torch.tensor(eigvecs[:, :k], dtype=torch.float)
    data.x = torch.cat([data.x, pe], dim=-1)

    return data


def check_sequence_validity(seq):
    """Checks if a degree sequence is valid after removing all zeros."""
    if len(seq) == 0:
        return False,1
    # Degree sequence sum must be even
    if sum(seq) % 2 != 0:
        return False,2
    # Sort in descending order
    sorted_seq = sorted(seq, reverse=True)
    # Apply Erdős–Gallai theorem
    for k in range(1, len(sorted_seq) + 1):
        lhs = sum(sorted_seq[:k])
        rhs = k * (k - 1) + sum(min(d, k) for d in sorted_seq[k:])
        if lhs > rhs:
            return False,3
    return True, 0

def _simple_valid(graph: nx.Graph, a: int, b: int, c: int, d: int) -> bool:
    # All endpoints must be distinct; no loops; no parallel edges.
    if len({a, b, c, d}) < 4:
        return False
    if a == b or c == d:
        return False
    # We will check duplicates after removing old edges in the temp graph.
    return True

def _rewire(graph: nx.Graph, e1, e2, orientation: int, ensure_connected):
    """
    orientation=0 -> add (u,x) and (v,y)
    orientation=1 -> add (u,y) and (v,x)
    Returns (new_graph, ((a,b),(c,d))) if valid; otherwise None.
    """
    (u, v), (x, y) = e1, e2
    if len({u, v, x, y}) < 4:
        return None
    if orientation == 0:
        a, b, c, d = u, x, v, y
    else:
        a, b, c, d = u, y, v, x
    if not _simple_valid(graph, a, b, c, d):
        return None
    if not graph.has_edge(u, v) or not graph.has_edge(x, y):
        return None

    G_post = graph.copy()
    G_post.remove_edge(u, v)
    G_post.remove_edge(x, y)
    remove_edges = (u, v),(x, y)
    # Disallow duplicates after removal
    if G_post.has_edge(a, b) or G_post.has_edge(c, d):
        return None
    G_post.add_edge(a, b)
    G_post.add_edge(c, d)
    add_edges = (a, b), (c, d)
    if ensure_connected and not nx.is_connected(G_post):
        return None

    return G_post, add_edges, remove_edges

def _k_hop_ok(G, e, k_hop) -> bool:
    if k_hop is None:
        return True
    # Option: prefer edges whose endpoints are not too far apart to preserve locality.
    (s, t) = e
    try:
        # Use a quick bound; exact APSP would be overkill here.
        return nx.shortest_path_length(G, s, t) <= k_hop
    except nx.NetworkXNoPath:
        return False


def transform_to_hh_via_guided_rewiring(
    G,
    H,
    lambda_dist,
    max_steps,
    ensure_connected=True,
    k_hop=2,         # e.g., 2 or 3 to preserve locality; None disables
    max_e2_candidates: Optional[int] = None,  # subsample second-edge candidates for speed
):
    rng = random.Random()
    G = G.copy()
    traj = []
    best_global = lambda_dist(G)
    no_improve = 0

    for step in range(max_steps):
        edges = list(G.edges())

        # Draw first edge uniformly
        e1 = edges[rng.randrange(len(edges))]

        # Candidate pool for e2
        e2_pool = [e for e in edges if e != e1 and len(set(e1 + e)) == 4 and _k_hop_ok(G, e, k_hop)]
        if not e2_pool:
            # fallback: ignore k_hop restriction for this step
            e2_pool = [e for e in edges if e != e1 and len(set(e1 + e)) == 4]
            if not e2_pool:
                print("No edge candidate found")
                continue
        # Optional subsample for speed on dense graphs
        if (max_e2_candidates is not None) and (len(e2_pool) > max_e2_candidates):
            rng.shuffle(e2_pool)
            e2_pool = e2_pool[:max_e2_candidates]

        best_step_dist = math.inf
        best_step_graph = None
        best_step_added = None
        best_step_removed = None
        for e2 in e2_pool:
            for orient in (0, 1):
                out = _rewire(G, e1, e2, orient, ensure_connected)
                if out is None:
                    continue
                G_cand, added_pair, removed_pair = out
                dist = lambda_dist(G_cand)
                if dist < best_step_dist:
                    best_step_dist = dist
                    best_step_graph = G_cand
                    best_step_added = added_pair
                    best_step_removed = removed_pair
        # Decide whether to take a greedy move, a random valid move, or skip
        if best_step_graph is not None and best_step_dist <= best_global:
            # Greedy non-worsening move
            G = best_step_graph
            traj.append((G, best_step_added, best_step_removed))  # graph AFTER rewiring
            best_global = best_step_dist
        else:
            rng.shuffle(edges)
            moved = False
            for e1_try in edges:
                for e2_try in edges:
                    if e2_try == e1_try or len(set(e1_try + e2_try)) < 4:
                        continue
                    for orient in (0, 1):
                        out = _rewire(G, e1_try, e2_try, orient, ensure_connected)
                        if out is None:
                            continue
                        G_cand, added_pair, removed_pair = out
                        G = G_cand.copy()
                        traj.append((G, added_pair, removed_pair))
                        best_global = lambda_dist(G)
                        moved = True
                        break
                    if moved:
                        break
                if moved:
                    break

            # If we still couldn’t move, we’re truly stuck; exit early.
            if not moved:
                print("Cannot move")
                break

        # Early exit if we’ve matched H exactly (or near-exactly if lambda is continuous)
        if best_global == 0.0 or best_global < 1e-12:
            break
    return traj


def hh_graph_from_G(G):
    """
    Build a canonical Havel–Hakimi realization that uses the same node labels as G.
    Ties are broken by (higher degree first, then smaller node id).
    """
    deg_pairs = sorted(((d, u) for u, d in G.degree()), key=lambda x: (-x[0], x[1]))
    seq = [d for d, _ in deg_pairs]
    # Build HH graph on 0..n-1 then relabel back to original nodes in this order
    H_int = nx.havel_hakimi_graph(seq)
    mapping = {i: deg_pairs[i][1] for i in range(len(seq))}
    H = nx.relabel_nodes(H_int, mapping, copy=True)
    return H


def _align_nodelist(G: nx.Graph, H: nx.Graph, nodelist: Optional[Iterable]=None):
    if nodelist is not None:
        return list(nodelist)
    # Default: union, sorted for determinism (works for int/str labels)
    return sorted(set(G.nodes()) | set(H.nodes()))

def _adjacency_matrix(G: nx.Graph, nodelist) -> np.ndarray:
    idx = {u:i for i,u in enumerate(nodelist)}
    n = len(nodelist)
    A = np.zeros((n, n), dtype=float)
    # weight=1.0 if not provided; undirected simple graphs
    for u, v, d in G.edges(data=True):
        if u in idx and v in idx and u != v:
            i, j = idx[u], idx[v]
            w = float(d.get("weight", 1.0))
            A[i, j] += w
            A[j, i] += w
    return A

def _laplacian(A: np.ndarray) -> np.ndarray:
    d = A.sum(axis=1)
    return np.diag(d) - A

def _normalized_laplacian(A: np.ndarray) -> np.ndarray:
    d = A.sum(axis=1)
    with np.errstate(divide='ignore'):
        inv_sqrt_d = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    Dhalf = np.diag(inv_sqrt_d)
    I = np.eye(A.shape[0])
    return I - Dhalf @ A @ Dhalf

def _eigvals_sym(M: np.ndarray) -> np.ndarray:
    # Ascending eigenvalues for symmetric matrices
    return np.linalg.eigvalsh(M)

def _sqrt_psd(M: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh(M)
    vals = np.clip(vals, 0.0, None)
    return (vecs * np.sqrt(vals)) @ vecs.T

def _pinv_laplacian(L: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    # Moore-Penrose pseudoinverse via EVD; zero out tiny eigenvalues
    w, U = np.linalg.eigh(L)
    invw = np.zeros_like(w)
    mask = w > tol
    invw[mask] = 1.0 / w[mask]
    return (U * invw) @ U.T

def _heat_kernel_from_L(L: np.ndarray, t: float) -> np.ndarray:
    w, U = np.linalg.eigh(L)
    exps = np.exp(-t * np.clip(w, 0.0, None))
    return (U * exps) @ U.T

def _degree_array(G: nx.Graph, nodelist) -> np.ndarray:
    idx = {u:i for i,u in enumerate(nodelist)}
    d = np.zeros(len(nodelist), dtype=int)
    for u, deg in G.degree():
        if u in idx:
            d[idx[u]] = deg
    return d

def _w1_degree_sorted(d1: np.ndarray, d2: np.ndarray) -> float:
    # 1D Wasserstein-1 via L1 of CDFs on integer support
    a = np.sort(d1)
    b = np.sort(d2)
    lo = int(min(a.min() if a.size else 0, b.min() if b.size else 0))
    hi = int(max(a.max() if a.size else 0, b.max() if b.size else 0))
    if hi < lo:
        return 0.0
    xs = np.arange(lo, hi + 1)
    Ca = np.cumsum(np.histogram(a, bins=xs, range=(lo, hi))[0])
    Cb = np.cumsum(np.histogram(b, bins=xs, range=(lo, hi))[0])
    return float(np.abs(Ca - Cb).sum())

# ----------------- Individual distance builders -----------------

def _make_symmetric_edit(H: nx.Graph, nodelist):
    H_edges = {tuple(sorted(e)) for e in H.subgraph(nodelist).edges()}
    def dist(G: nx.Graph) -> float:
        G_edges = {tuple(sorted(e)) for e in G.subgraph(nodelist).edges()}
        return float(len(G_edges ^ H_edges))
    return dist

def _make_spectral(H: nx.Graph, nodelist, *, normalized=True, k: Optional[int]=None, squared=False):
    AH = _adjacency_matrix(H, nodelist)
    LH = _normalized_laplacian(AH) if normalized else _laplacian(AH)
    evH = _eigvals_sym(LH)
    if k is not None:
        evH = evH[:min(k, len(evH))]
    def dist(G: nx.Graph) -> float:
        AG = _adjacency_matrix(G, nodelist)
        LG = _normalized_laplacian(AG) if normalized else _laplacian(AG)
        evG = _eigvals_sym(LG)
        if k is not None:
            evG = evG[:min(k, len(evG))]
        diff = evG - evH
        val = float(np.dot(diff, diff))
        return val if squared else math.sqrt(val)
    return dist

def _make_bures_wasserstein(H: nx.Graph, nodelist, *, t: float=0.5, use_normalized_L: bool=True, squared=False):
    if t <= 0:
        raise ValueError("t must be > 0 for heat-kernel BW.")
    AH = _adjacency_matrix(H, nodelist)
    LH = _normalized_laplacian(AH) if use_normalized_L else _laplacian(AH)
    KH = _heat_kernel_from_L(LH, t)
    KH_sqrt = _sqrt_psd(KH)
    trKH = float(np.trace(KH))
    def dist(G: nx.Graph) -> float:
        AG = _adjacency_matrix(G, nodelist)
        LG = _normalized_laplacian(AG) if use_normalized_L else _laplacian(AG)
        KG = _heat_kernel_from_L(LG, t)
        mid = KH_sqrt @ KG @ KH_sqrt
        mid_sqrt = _sqrt_psd(mid)
        d2 = float(np.trace(KG) + trKH - 2.0 * np.trace(mid_sqrt))
        d2 = max(d2, 0.0)
        return d2 if squared else math.sqrt(d2)
    return dist

def _make_effective_resistance_fro(H: nx.Graph, nodelist, *, tol: float=1e-12):
    AH = _adjacency_matrix(H, nodelist)
    LH = _laplacian(AH)  # use combinatorial Laplacian
    LHp = _pinv_laplacian(LH, tol=tol)
    diagH = np.diag(LHp)
    RH = diagH[:,None] + diagH[None,:] - 2*LHp
    def dist(G: nx.Graph) -> float:
        AG = _adjacency_matrix(G, nodelist)
        LG = _laplacian(AG)
        LGp = _pinv_laplacian(LG, tol=tol)
        diagG = np.diag(LGp)
        RG = diagG[:,None] + diagG[None,:] - 2*LGp
        return float(np.linalg.norm(RG - RH, 'fro'))
    return dist

def _make_deltacon(H: nx.Graph, nodelist, *, eps: float=1e-3):
    # S = (I + eps^2 D - eps A)^{-1}
    AH = _adjacency_matrix(H, nodelist)
    dH = AH.sum(1)
    SH = np.linalg.inv(np.eye(len(nodelist)) + (eps**2)*np.diag(dH) - eps*AH)
    SH_sqrt = np.sqrt(np.maximum(SH, 0.0))
    def dist(G: nx.Graph) -> float:
        AG = _adjacency_matrix(G, nodelist)
        dG = AG.sum(1)
        SG = np.linalg.inv(np.eye(len(nodelist)) + (eps**2)*np.diag(dG) - eps*AG)
        SG_sqrt = np.sqrt(np.maximum(SG, 0.0))
        return float(np.linalg.norm(SG_sqrt - SH_sqrt, 'fro'))
    return dist

def _make_netlsd(H: nx.Graph, nodelist, *, T_min=1e-2, T_max=1e2, num=50, normalized=True):
    AH = _adjacency_matrix(H, nodelist)
    LH = _normalized_laplacian(AH) if normalized else _laplacian(AH)
    wH = _eigvals_sym(LH)
    ts = np.logspace(np.log10(T_min), np.log10(T_max), num)
    hH = np.array([(np.exp(-t*np.clip(wH,0,None))).sum() for t in ts])
    def dist(G: nx.Graph) -> float:
        AG = _adjacency_matrix(G, nodelist)
        LG = _normalized_laplacian(AG) if normalized else _laplacian(AG)
        wG = _eigvals_sym(LG)
        hG = np.array([(np.exp(-t*np.clip(wG,0,None))).sum() for t in ts])
        return float(np.linalg.norm(hG - hH))
    return dist

def _make_degree_emd(H: nx.Graph, nodelist):
    dH = _degree_array(H, nodelist)
    def dist(G: nx.Graph) -> float:
        dG = _degree_array(G, nodelist)
        return _w1_degree_sorted(dG, dH)
    return dist

# ----------------- Factory -----------------

def make_lambda_dist(
    name: str,
    H: nx.Graph,
    *,
    nodelist: Optional[Iterable]=None,
    **kwargs
):
    """
    Build a distance function lambda_dist(G) that compares G to target H.

    Parameters
    ----------
    name : str
        One of:
          - 'symmetric_edit'
          - 'spectral' (kwargs: normalized=True, k=None, squared=False)
          - 'bures_wasserstein' (kwargs: t=0.5, use_normalized_L=True, squared=False)
          - 'effective_resistance_fro' (kwargs: tol=1e-12)
          - 'deltacon' (kwargs: eps=1e-3)
          - 'netlsd' (kwargs: T_min=1e-2, T_max=1e2, num=50, normalized=True)
    H : nx.Graph
        Target graph.
    nodelist : Iterable, optional
        Fixed node order used for all computations. If None, uses union of G and H.

    Returns
    -------
    Callable[[nx.Graph], float]
        A function taking a graph G and returning the distance d(G, H).
    """
    # Pre-fix the nodelist for consistency and speed
    nodelist = _align_nodelist(H, H, nodelist=nodelist)

    name = name.lower()
    if name == 'symmetric_edit':
        return _make_symmetric_edit(H, nodelist)
    if name == 'spectral':
        return _make_spectral(H, nodelist, **kwargs)
    if name == 'bures_wasserstein':
        return _make_bures_wasserstein(H, nodelist, **kwargs)
    if name == 'effective_resistance_fro':
        return _make_effective_resistance_fro(H, nodelist, **kwargs)
    if name == 'deltacon':
        return _make_deltacon(H, nodelist, **kwargs)
    if name == 'netlsd':
        return _make_netlsd(H, nodelist, **kwargs)

    raise ValueError(f"Unknown distance '{name}'.")


def draw_graphs_grid(
    graphs,
    n_cols: int = 8,
    layout: str = "spring",
    figsize_per_graph=(3.5, 3.5),
    node_size=300,
    with_labels=True,
    titles=None,
    seed: int = 42,
    savepath: str = None,
    **draw_kwargs,
):
    """
    Draw a list of NetworkX graphs on one figure in a grid (n_cols per row).

    Args:
        graphs (List[nx.Graph]): graphs to draw.
        n_cols (int): number of graphs per row.
        layout (str): one of {'spring','kamada_kawai','circular','spectral','shell','random'}.
        figsize_per_graph (tuple): (width,height) inches per subplot.
        node_size (int): node size passed to nx.draw.
        with_labels (bool): show node labels.
        titles (List[str] or None): optional titles per graph.
        seed (int): random seed for layouts that accept it.
        savepath (str or None): if given, save the figure here.
        **draw_kwargs: forwarded to nx.draw (e.g., node_color, edge_color, width, alpha).
    Returns:
        (fig, axes): Matplotlib figure and axes array.
    """
    if not graphs:
        raise ValueError("`graphs` is empty.")

    # Pick a layout function
    layout_fn_map = {
        "spring": lambda G: nx.spring_layout(G, seed=seed),
        "kamada_kawai": lambda G: nx.kamada_kawai_layout(G),
        "circular": lambda G: nx.circular_layout(G),
        "spectral": lambda G: nx.spectral_layout(G),
        "shell": lambda G: nx.shell_layout(G),
        "random": lambda G: nx.random_layout(G, seed=seed),
    }
    if layout not in layout_fn_map:
        raise ValueError(f"Unknown layout '{layout}'. Choose from {list(layout_fn_map)}.")

    n = len(graphs)
    n_rows = math.ceil(n / n_cols)
    fig_w = figsize_per_graph[0] * n_cols
    fig_h = figsize_per_graph[1] * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))
    if isinstance(axes, plt.Axes):
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]
    elif n_cols == 1:
        axes = [[ax] for ax in axes]

    # Draw each graph
    for idx, G in enumerate(graphs):
        r, c = divmod(idx, n_cols)
        ax = axes[r][c]
        pos = layout_fn_map[layout](G)
        ax.set_title(titles[idx] if titles and idx < len(titles) else f"G{idx}", fontsize=10)
        nx.draw(
            G,
            pos=pos,
            ax=ax,
            node_size=node_size,
            with_labels=with_labels,
            **draw_kwargs,
        )
        ax.set_axis_off()

    # Hide any leftover empty subplots
    for idx in range(n, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r][c].set_visible(False)

    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, bbox_inches="tight", dpi=300)
        plt.close(fig)
    else:
        plt.show()
    return fig, axes


def build_candidates(
    G,
    amchor_edge,
    ensure_connected: bool = True,
    k_hop: int = 2,
):
    """
    Return a list of candidate edges given the anchor edge, under the following constraints:
    - Candidate edge is disjoint from the anchor edge (4 distinct endpoints).
    - At least one orientation of rewiring (u,v) with candidate edge is valid:
        * no self-loops or parallel edges
        * (optionally) resulting graph remains connected if `ensure_connected` is True
    - Locality: the candidate edge is within `k_hop` hops of the anchor edge, i.e.
      at least one endpoint of the candidate is within distance <= k_hop from at
      least one endpoint of the anchor edge. If k_hop is None, locality is ignored.
    """
    u, v = amchor_edge
    edges = list(G.edges())
    candidates = []

    for e2 in edges:
        # skip same edge
        if e2 == amchor_edge:
            continue

        x, y = e2

        # require 4 distinct endpoints to allow a proper swap
        if len({u, v, x, y}) < 4:
            continue

        # --- k-hop locality between anchor endpoints and candidate endpoints ---
        if k_hop is not None:
            close_enough = False
            for a in (u, v):
                for b in (x, y):
                    try:
                        if nx.shortest_path_length(G, a, b) <= k_hop:
                            close_enough = True
                            break
                    except nx.NetworkXNoPath:
                        # if there's no path between a and b, this pair can't be "local"
                        continue
                if close_enough:
                    break

            if not close_enough:
                continue

        # --- check if at least one orientation yields a valid rewiring ---
        valid = False
        for orient in (0, 1):
            out = _rewire(G, amchor_edge, e2, orient, ensure_connected)
            if out is not None:
                valid = True
                break

        if valid:
            candidates.append(e2)

    return candidates
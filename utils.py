import torch
import os
import random
import networkx as nx
from torch_geometric.utils.convert import from_networkx
import numpy as np
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix
from scipy.sparse import csgraph, csr_matrix
from collections import Counter
import numpy as np
from collections import deque
import math
import matplotlib.pyplot as plt

def load_degree_sequence_from_directory(directory_path):
    max_node = 0 
    min_node = -1
    seqs = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            G = nx.read_edgelist(file_path, nodetype=int)
            max_node = max(max_node, G.number_of_nodes())
            min_node = min(min_node, G.number_of_nodes()) if min_node > 0 else G.number_of_nodes()
    print("Max node: ", max_node, " Min node:", min_node)
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            G = nx.read_edgelist(file_path, nodetype=int)
            G = nx.convert_node_labels_to_integers(G)
            seq = [deg for _, deg in G.degree()]
            if seq is not None:
                seqs.append(seq)
    return seqs, max_node, min_node

def load_graph_from_directory(directory_path):
    max_node = 0 
    min_node = -1
    max_edge = 0 
    min_edge = -1
    graphs = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            G = nx.read_edgelist(file_path, nodetype=int)
            max_node = max(max_node, G.number_of_nodes())
            min_node = min(min_node, G.number_of_nodes()) if min_node > 0 else G.number_of_nodes()
            max_edge = max(max_edge, G.number_of_edges())
            min_edge = min(min_edge, G.number_of_edges()) if min_edge > 0 else G.number_of_edges()
    print("Max node: ", max_node, " Min node:", min_node)
    print("Max edge: ", max_edge, " Min edge:", min_edge)
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            G = nx.read_edgelist(file_path, nodetype=int)
            graph = nx.convert_node_labels_to_integers(G)
            graphs.append(graph)
    return graphs, max_node, min_node


def _safe_eigvecs(G: nx.Graph, k: int) -> np.ndarray:
    """
    Return up to `k` Laplacian eigenvectors for G, coping with tiny graphs.

    For |V| ≤ 2  (or any failure in ARPACK), we return a constant vector so that
    every node gets identical features.  This keeps dimensions consistent while
    still giving the GNN *something* to work with.
    """
    n = G.number_of_nodes()
    if n == 0:
        return np.empty((0, k))

    # For very small graphs we skip ARPACK entirely.
    if n <= 2:
        return np.ones((n, 1))

    A = csr_matrix(nx.to_scipy_sparse_array(G, dtype=float))
    L = csgraph.laplacian(A, normed=True)

    # eigsh requires k < n.  Use n-2 so that the zero eigenpair is avoided.
    k_eff = min(k, n - 2)
    if k_eff < 1:
        k_eff = 1

    try:
        _, eigvecs = eigsh(L, k=k_eff, which="SM")
        return eigvecs  # shape: (n , k_eff)
    except Exception as e:
        print(f"[utils] Laplacian eigendecomposition failed on |V|={n}: {e}")
        return np.ones((n, 1))


def graph_to_data(G: nx.Graph, k_eigen: int):
    """
    Convert a NetworkX graph to a PyG Data object with Laplacian‐eigen features.

    The returned Data.x has fixed width `k_eigen` for *all* graphs, so the
    downstream GIN layers in GraphER always receive the expected dimension.
    """
    eigvecs = _safe_eigvecs(G, k_eigen)           # (n , ≤k_eigen)
    n, cur_k = eigvecs.shape

    # Pad or truncate to exactly k_eigen columns.
    if cur_k < k_eigen:
        pad = np.zeros((n, k_eigen - cur_k), dtype=eigvecs.dtype)
        eigvecs = np.concatenate([eigvecs, pad], axis=1)
    elif cur_k > k_eigen:
        eigvecs = eigvecs[:, :k_eigen]

    # Attach features to nodes.
    for i, node in enumerate(G.nodes()):
        G.nodes[node]["x"] = eigvecs[i].astype(np.float32)

    return from_networkx(G)


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


def _ek(u, v):
    return (u, v) if u <= v else (v, u)

def _pick_valid_swap(G, max_tries=128, rng=random):
    """
    Propose a valid 2-edge swap (e1, e2) -> (f1, f2).
    Returns (e1, e2, f1, f2) or None if not found.
    """
    E = list(G.edges())
    nE = len(E)
    if nE < 2:
        return None
    for _ in range(max_tries):
        (a, b) = E[rng.randrange(nE)]
        (c, d) = E[rng.randrange(nE)]
        if len({a, b, c, d}) != 4:  # endpoints must be all different
            continue
        # two possible rewires; pick one at random
        if rng.random() < 0.5:
            f1, f2 = (a, c), (b, d)
        else:
            f1, f2 = (a, d), (b, c)
        # no self-loops and no multi-edges
        if f1[0] == f1[1] or f2[0] == f2[1]:
            continue
        if G.has_edge(*f1) or G.has_edge(*f2):
            continue
        # Also avoid creating parallel edges across the pair
        if _ek(*f1) == _ek(*f2):
            continue
        return ( (a,b), (c,d), f1, f2 )
    return None

def _khop_neighborhoods(G, k):
    """
    Precompute closed k-hop neighborhoods (excluding the center itself).
    """
    N = {}
    for u in G.nodes():
        dists = nx.single_source_shortest_path_length(G, u, cutoff=k)
        N[u] = {v for v, dist in dists.items() if 0 < dist <= k}
    return N

def _within_k(u, v, k, neighborhoods, G_current, locality_reference, cache_dynamic):
    """
    Check if dist(u, v) <= k according to chosen reference.
    """
    if k is None:
        return True
    if locality_reference == "initial":
        return v in neighborhoods[u]
    # dynamic: compute on-demand BFS (cached per (anchor, k))
    key = (u, k)
    if key not in cache_dynamic:
        dists = nx.single_source_shortest_path_length(G_current, u, cutoff=k)
        cache_dynamic[key] = {x for x, dist in dists.items() if 0 < dist <= k}
    return v in cache_dynamic[key]

def _propose_swap_with_locality(
    G, rng, k, neighborhoods, locality_reference, max_tries=256
):
    """
    Propose a valid 2-edge swap (e1,e2)->(f1,f2) that respects k-hop locality.
    Returns (e1, e2, f1, f2) or None.
    """
    E = list(G.edges())
    m = len(E)
    if m < 2: return None
    dyn_cache = {}  # for dynamic k-hop lookups

    for _ in range(max_tries):
        (a, b) = E[rng.randrange(m)]
        (c, d) = E[rng.randrange(m)]
        if len({a, b, c, d}) != 4:
            continue

        # Two orientations; try the one sampled first, fall back to the other
        for (f1, f2) in ( ((a, c), (b, d)), ((a, d), (b, c)) if rng.random() < 0.5 else ((a, d), (b, c),), ):
            # simple-edge constraints
            if f1[0] == f1[1] or f2[0] == f2[1]:
                continue
            if G.has_edge(*f1) or G.has_edge(*f2):
                continue
            if _ek(*f1) == _ek(*f2):
                continue
            # k-hop locality constraints
            if not _within_k(f1[0], f1[1], k, neighborhoods, G, locality_reference, dyn_cache):
                continue
            if not _within_k(f2[0], f2[1], k, neighborhoods, G, locality_reference, dyn_cache):
                continue
            return ( (a,b), (c,d), f1, f2 )

    return None

# ---------- main routine ----------
def transform_to_hh_via_stochastic_rewiring(
    G,
    H,
    max_steps=10000,
    beta=3.0,           # bias toward HH edges
    T0=1.0,             # initial temperature
    cooling=0.995,      # simulated annealing cooling per accepted step
    ensure_connected=True,
    k_hop=2,         # e.g., 2 or 3 to preserve locality; None disables
    locality_reference="initial",  # "initial" (default) or "current"
    seed=None,
):
    """
    Stochastically transform G to its Havel–Hakimi realization using biased 2-edge swaps
    while (a) preserving degree sequence, (b) enforcing k-hop locality for *new* edges,
    and (c) rejecting swaps that break connectivity (if ensure_connected=True).
    """
    rng = random.Random(seed)
    Gc = G.copy()

    # Target HH graph and scoring
    
    H_set = {_ek(u, v) for u, v in H.edges()}
    def matches_in_H(edges): return sum(1 for e in edges if _ek(*e) in H_set)
    cur_matches = matches_in_H(Gc.edges())

    # Precompute k-hop neighborhoods on the chosen reference graph
    neighborhoods = None
    if k_hop is not None:
        ref_graph = G if locality_reference == "initial" else Gc
        neighborhoods = _khop_neighborhoods(ref_graph, k_hop)

    T = T0
    traj = []
    m = Gc.number_of_edges()

    for _ in range(max_steps):
        prop = _propose_swap_with_locality(
            Gc, rng, k_hop, neighborhoods, locality_reference, max_tries=256
        )
        if prop is None:
            # no valid locality-respecting swap found under the budget
            break

        (e1, e2, f1, f2) = prop
        before = int(_ek(*e1) in H_set) + int(_ek(*e2) in H_set)
        after  = int(_ek(*f1) in H_set) + int(_ek(*f2) in H_set)
        dmatches = after - before

        # Metropolis acceptance (symmetric proposals)
        accept = (dmatches >= 0) or (rng.random() < math.exp(beta * dmatches / max(T, 1e-8)))
        if not accept:
            continue

        # Tentatively apply and enforce connectivity
        Gc.remove_edges_from([e1, e2])
        Gc.add_edges_from([f1, f2])

        if ensure_connected and not nx.is_connected(Gc):
            # revert if it breaks connectivity
            Gc.remove_edges_from([f1, f2])
            Gc.add_edges_from([e1, e2])
            continue
        else:
            traj.append((Gc.copy(),(f1, f2),(e1, e2)))

        # Commit
        cur_matches += dmatches
        T *= cooling

        if cur_matches == m:  # reached HH exactly
            break

        # If using dynamic locality, refresh neighborhoods occasionally (cheap heuristic)
        if k_hop is not None and locality_reference == "current":
            # Only recompute for touched nodes to keep it light
            for u in {e1[0], e1[1], e2[0], e2[1], f1[0], f1[1], f2[0], f2[1]}:
                dists = nx.single_source_shortest_path_length(Gc, u, cutoff=k_hop)
                neighborhoods[u] = {x for x, dist in dists.items() if 0 < dist <= k_hop}
    #plot_graph_evolution([(G,"G"),(Gc,"G_to_HH"),(H,"H")])
    return traj


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

def save_graphs(snapshots, save_to_local=True, filename="graphs", out_dir="logs"):
    """
    Draw a panel (8 per row). Accepts a list of nx.Graph objects.
    Uses a union-graph layout so every node across all snapshots has a position.
    """
    os.makedirs(out_dir, exist_ok=True)
    if not snapshots:
        return

    # 1) Build a union graph over all snapshots (nodes+edges)
    G_union = nx.Graph()
    for G in snapshots:
        G_union.add_nodes_from(G.nodes())
        G_union.add_edges_from(G.edges())

    # 2) Compute positions once for the union (covers *all* nodes)
    pos = nx.spring_layout(G_union, seed=42)

    # 3) Set up grid
    N = len(snapshots)
    ncols = 8
    nrows = math.ceil(N / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.6 * nrows))

    # Flatten axes
    if nrows * ncols == 1:
        axes = [axes]
    elif nrows == 1:
        axes = list(axes)
    else:
        axes = [ax for row in axes for ax in row]

    # 4) Draw each snapshot using the union positions
    for G, ax in zip(snapshots, axes):
        # (Optional safety) fill any unexpected missing nodes
        missing = [n for n in G.nodes() if n not in pos]
        if missing:
            # Initialize missing nodes near origin; or re-run spring_layout seeded by existing pos:
            # pos.update(nx.spring_layout(G, pos=pos, fixed=[n for n in pos if n in G], seed=42))
            for n in missing:
                pos[n] = np.zeros(2, dtype=float)

        nx.draw(G, pos=pos, with_labels=False, ax=ax)
        ax.axis("off")

    # Hide any unused axes
    for j in range(N, nrows * ncols):
        axes[j].axis("off")

    plt.tight_layout()
    if save_to_local:
        path = os.path.join(out_dir, f"{filename}.png")
        plt.savefig(path, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()



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

def hh_graph_from_seq(seq):
    H_int = nx.havel_hakimi_graph(seq)
    return H_int  # nodes are 0..n-1 already


def laplacian_eigs(G: nx.Graph, k: int, normed: bool = True):
    """
    Return (vals, vecs) for the K smallest *non-zero* eigenpairs of the (normalized) Laplacian.
    Falls back gracefully on tiny graphs.
    """
    n = G.number_of_nodes()
    A = csr_matrix(nx.to_scipy_sparse_array(G, dtype=float))
    L = csgraph.laplacian(A, normed=normed)

    # ask for k+1 to include the zero eigenvalue, then drop it
    want = min(max(1, k + 1), n - 1)
    vals, vecs = eigsh(L, k=want, which="SM")
    idx = np.argsort(vals)
    vals, vecs = vals[idx], vecs[:, idx]
    # drop the (near-)zero eigenvalue
    if vals[0] < 1e-8:
        vals, vecs = vals[1:], vecs[:, 1:]
    # cap/pad to exactly k
    vals = vals[:k].astype(np.float32)
    vecs = vecs[:, :k].astype(np.float32)
    if vals.shape[0] < k:
        pad = k - vals.shape[0]
        vals = np.pad(vals, (0, pad))
        vecs = np.pad(vecs, ((0, 0), (0, pad)))
    return vals, vecs


def normalized_laplacian_dense(G: nx.Graph) -> np.ndarray:
    """Dense normalized Laplacian as float64 ndarray (for small graphs this is fine)."""
    return csgraph.laplacian(
        csr_matrix(nx.to_scipy_sparse_array(G, dtype=float)), normed=True
    ).toarray()


def try_apply_swap_with_orientation(G, anchor, partner, orient_idx,
                                    ensure_connected=True, k_hop=None):
    """
    anchor:  (u,v)
    partner: (x,y)
    orient_idx: 0 -> (u,x)&(v,y); 1 -> (u,y)&(v,x)
    Returns True if applied; else False (graph is left unchanged).
    """
    (u,v), (x,y) = anchor, partner
    if orient_idx == 0:
        f1, f2 = (u, x), (v, y)
    else:
        f1, f2 = (u, y), (v, x)
    # basic validity
    if len({*f1}) < 2 or len({*f2}) < 2:  # self-loops
        return False
    if f1 == f2:
        return False
    if G.has_edge(*f1) or G.has_edge(*f2):
        return False
    # tentative commit
    G.remove_edges_from([anchor, partner])
    G.add_edges_from([f1, f2])
    ok = True
    if ensure_connected and not nx.is_connected(G):
        ok = False
    if ok and k_hop is not None:
        for (p,q) in (f1, f2):
            try:
                d = nx.shortest_path_length(G, p, q)
            except nx.NetworkXNoPath:
                d = 10**9
            if d > k_hop:
                ok = False
                break
    if not ok:
        # revert
        G.remove_edges_from([f1, f2])
        G.add_edges_from([anchor, partner])
        return False
    return True
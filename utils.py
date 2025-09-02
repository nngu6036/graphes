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

def load_degree_sequence_from_directory(directory_path):
    max_node = 0 
    min_node = -1
    seqs = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            G = nx.read_edgelist(file_path, nodetype=int)
            max_node = max(max_node, G.number_of_nodes())
            min_node = min(min_node, G.number_of_edges()) if min_node > 0 else G.number_of_edges()
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
    graphs = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            G = nx.read_edgelist(file_path, nodetype=int)
            max_node = max(max_node, G.number_of_nodes())
            min_node = min(min_node, G.number_of_edges()) if min_node > 0 else G.number_of_edges()
    print("Max node: ", max_node, " Min node:", min_node)
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


def _within_k_hops(G: nx.Graph, s, t, k: int) -> bool:
    """Return True iff dist_G(s,t) <= k (BFS early-stop)."""
    if s == t:
        return True
    if k <= 0:
        return False
    seen = {s}
    q = deque([(s, 0)])
    while q:
        v, d = q.popleft()
        if d == k:
            continue
        for w in G.neighbors(v):
            if w == t:
                return True
            if w not in seen:
                seen.add(w)
                q.append((w, d + 1))
    return False

def _edges_within_k_hops(G: nx.Graph, e1, e2, k: int) -> bool:
    """Require the two SOURCE edges to be 'k-local' to each other."""
    (u, v), (x, y) = e1, e2
    return (
        _within_k_hops(G, u, x, k) or
        _within_k_hops(G, u, y, k) or
        _within_k_hops(G, v, x, k) or
        _within_k_hops(G, v, y, k)
    )

def _assort_score(a, b, deg, mode="product"):
    """
    Larger is better for high-high pairs.
      mode='product'      -> deg[a]*deg[b]   (strongly favors high-high)
      mode='similarity'   -> - (deg[a]-deg[b])**2   (favors similar degrees)
    """
    if mode == "similarity":
        da, db = deg[a], deg[b]
        return - (da - db) ** 2
    # default: product
    return deg[a] * deg[b]

def _weighted_two_edges(edges, weights):
    """Pick two distinct edges with probability proportional to weights."""
    # First edge
    idx1 = random.choices(range(len(edges)), weights=weights, k=1)[0]
    # Second edge (resample until different)
    while True:
        idx2 = random.choices(range(len(edges)), weights=weights, k=1)[0]
        if idx2 != idx1:
            break
    return edges[idx1], edges[idx2]

# ---------- main ----------
def rewire_edges_k_local_assortative(
    G: nx.Graph,
    k: int = 2,
    max_retries = 10,
    keep_connected: bool = True,
    forbid_bridges: bool = True,
    assortative_bias: float = 7.0,
    assortative_mode: str = "product",   # 'product' or 'similarity'
    weight_edge_sampling: bool = True,   # bias which source edges we pick
):
    """
    One degree-preserving 2-edge swap that:
      • keeps the graph connected (reverts if not),
      • only considers source edges within k hops (locality),
      • BIASES the new edges to connect high-degree nodes together.

    Bias controls:
      - weight_edge_sampling=True biases which *source* edges are chosen;
        edges with large deg[u]*deg[v] are sampled more often.
      - assortative_bias>0 applies a softmax over the two candidate pairings,
        preferring the one with larger sum of scores for (a,b) and (c,d).
      - assortative_mode='product' (default) strongly prefers high-high pairs;
        use 'similarity' to favor degree-similar pairs (assortative but gentler).

    Returns:
      G (modified in-place; unchanged if no acceptable swap is found).
    """
    add_pair,remove_pair = None, None
    step = 0
    if G.number_of_edges() < 2:
        return G, None, None

    deg = dict(G.degree())

    # Precompute optional bridge set (helps avoid disconnections)
    bridge_set = set()
    if forbid_bridges:
        bridge_set = {tuple(sorted(e)) for e in nx.bridges(G)}

    edges = list(G.edges())

    # Optional: bias *which* edges we propose to rewire
    if weight_edge_sampling:
        base_w = [deg[u] * deg[v] for (u, v) in edges]  # product works well
        # Avoid zero weights
        min_pos = 1 if all(w == 0 for w in base_w) else 0
        weights = [w + min_pos for w in base_w]
    else:
        weights = [1.0] * len(edges)

    for _ in range(max_retries):
        # Pick two source edges (possibly weighted toward high-degree endpoints)
        e1, e2 = _weighted_two_edges(edges, weights)
        u, v = e1
        x, y = e2

        # Distinct endpoints for a valid 2-switch
        if len({u, v, x, y}) != 4:
            continue

        # k-hop locality between the *source* edges
        if not _edges_within_k_hops(G, e1, e2, k):
            continue

        # (Optional) avoid deleting bridges
        if forbid_bridges and (tuple(sorted(e1)) in bridge_set or tuple(sorted(e2)) in bridge_set):
            continue

        # Two candidate pairings
        candidates = [ (u, x, v, y), (u, y, v, x) ]
        random.shuffle(candidates)  # avoid deterministic tie behavior

        # Score each valid pairing by how assortative the *new* edges are
        scored = []
        for a, b, c, d in candidates:
            if a == b or c == d:
                continue
            if G.has_edge(a, b) or G.has_edge(c, d):
                continue
            s = _assort_score(a, b, deg, assortative_mode) + _assort_score(c, d, deg, assortative_mode)
            scored.append(((a, b, c, d), s))

        if not scored:
            continue

        # Softmax over scores -> prefer higher assortativity, but keep some randomness
        max_s = max(s for _, s in scored)
        probs = [math.exp(assortative_bias * (s - max_s)) for _, s in scored]
        total = sum(probs)
        probs = [p / total for p in probs]
        choice_idx = random.choices(range(len(scored)), weights=probs, k=1)[0]
        a, b, c, d = scored[choice_idx][0]

        # Execute 
        add = (a, b), (c, d)
        remove = (u, v), (x, y)
        G.remove_edges_from([(u, v), (x, y)])
        G.add_edges_from([(a, b), (c, d)])

        # Enforce connectivity
        if keep_connected and not nx.is_connected(G):
            # revert and try again
            add, remove = None, None
            G.remove_edges_from([(a, b), (c, d)])
            G.add_edges_from([(u, v), (x, y)])
            continue
        else:
            return G, add, remove

    return G, None, None


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

def save_graph_evolution(snapshots, idx, out_dir="evolutions"):
    """
    Save a sequence of graph snapshots in one horizontal figure.

    Args:
        snapshots (list[tuple[nx.Graph, int]]): list of (graph, step) pairs, for ONE graph.
        idx (int): 0-based index of this graph (used in filename).
        out_dir (str): output folder.
    """
    import matplotlib.pyplot as plt
    import os

    if not snapshots:
        return

    os.makedirs(out_dir, exist_ok=True)

    # Use a FIXED layout across all panels for comparability.
    # Compute on the first snapshot and reuse positions.
    G0, _ = snapshots[0]
    # Seeded layout for reproducibility; tweak seed if you like.
    pos = nx.spring_layout(G0, seed=42)

    fig, axes = plt.subplots(1, len(snapshots), figsize=(4 * len(snapshots), 4))
    if len(snapshots) == 1:
        axes = [axes]

    for ax, (G, t) in zip(axes, snapshots):
        # Draw with fixed positions; nodes that don't exist will be ignored (same N here).
        nx.draw(G, pos=pos, node_size=40, with_labels=False, ax=ax)
        ax.set_title(f"Step {t}")
        ax.axis("off")

    filename = os.path.join(out_dir, f"graph_{idx+1}_evolution.png")
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()

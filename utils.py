import torch
from torch_geometric.utils import from_networkx
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from scipy.stats import wasserstein_distance


def compute_chamfer_distance(set1, set2):
    if len(set1) == 0 or len(set2) == 0:
        return float('inf')  # If one of the sets is empty, the distance is undefined.
    set1 = torch.tensor(set1, dtype=torch.float).unsqueeze(1)
    set2 = torch.tensor(set2, dtype=torch.float).unsqueeze(1)
    dists_1_to_2 = torch.min(torch.cdist(set1, set2, p=2), dim=1).values
    dists_2_to_1 = torch.min(torch.cdist(set2, set1, p=2), dim=1).values
    chamfer_distance = torch.sum(dists_1_to_2) + torch.sum(dists_2_to_1)
    return chamfer_distance.item()

def compute_degree_histograms(sequences, max_degree):
    histograms = []
    for seq in sequences:
        hist = torch.zeros(max_degree + 1)
        for deg in seq:
            if 0 <= deg <= max_degree:
                hist[deg] += 1
        hist /= hist.sum() + 1e-8  # normalize + stability
        histograms.append(hist)
    return torch.stack(histograms)

def compute_kl_divergence(p_hist, q_hist, eps=1e-8):
    p_ = p_hist + eps
    q_ = q_hist + eps
    p_ = p_ / p_.sum()
    q_ = q_ / q_.sum()
    return torch.sum(p_ * torch.log(p_ / q_)).item()

def compute_mmd(X, Y, gamma=1.0):
    X_np = X.detach().cpu().numpy()
    Y_np = Y.detach().cpu().numpy()
    K_xx = rbf_kernel(X_np, X_np, gamma=gamma)
    K_yy = rbf_kernel(Y_np, Y_np, gamma=gamma)
    K_xy = rbf_kernel(X_np, Y_np, gamma=gamma)
    return float(K_xx.mean() + K_yy.mean() - 2 * K_xy.mean())

def compute_earth_movers_distance(set1, set2):
    if len(set1) == 0 or len(set2) == 0:
        return float('inf')  # Undefined if one of the sets is empty.
    set1 = torch.tensor(set1, dtype=torch.float).unsqueeze(1)
    set2 = torch.tensor(set2, dtype=torch.float).unsqueeze(1)
    cost_matrix = torch.cdist(set1, set2, p=2).cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    emd = cost_matrix[row_ind, col_ind].sum()
    return emd


def graph_to_data(G):
    for node in G.nodes:
        G.nodes[node]['x'] = [1.0]
    return from_networkx(G)


def gaussian_emd_kernel(X, Y, sigma=1.0):
    K = np.zeros((len(X), len(Y)))
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            emd = wasserstein_distance(np.arange(len(x)), np.arange(len(y)), x, y)
            K[i, j] = np.exp(-emd**2 / (2 * sigma**2))
    return K


def compute_mmd_degree_emd(graphs_1, graphs_2, max_degree=None, sigma=1.0):
    def degree_histogram(graphs, max_degree):
        histograms = []
        for G in graphs:
            degree_sequence = [deg for _, deg in G.degree()]
            hist = np.zeros(max_degree + 1)
            for deg in degree_sequence:
                if deg <= max_degree:
                    hist[deg] += 1
            if hist.sum() == 0:
                hist[0] = 1.0
            hist /= hist.sum()
            histograms.append(hist)
        return np.array(histograms)

    if max_degree is None:
        max_d1 = max((max(dict(G.degree()).values()) if len(G) > 0 else 0) for G in graphs_1)
        max_d2 = max((max(dict(G.degree()).values()) if len(G) > 0 else 0) for G in graphs_2)
        max_degree = max(max_d1, max_d2)
    H1 = degree_histogram(graphs_1, max_degree)
    H2 = degree_histogram(graphs_2, max_degree)
    K_xx = gaussian_emd_kernel(H1, H1, sigma)
    K_yy = gaussian_emd_kernel(H2, H2, sigma)
    K_xy = gaussian_emd_kernel(H1, H2, sigma)
    mmd = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
    return float(mmd)


def compute_mmd_cluster(graphs_1, graphs_2, bins=10, sigma=1.0):
    def clustering_histogram(graphs, bins=10):
        histograms = []
        for G in graphs:
            clustering = list(nx.clustering(G).values())
            hist, _ = np.histogram(clustering, bins=bins, range=(0, 1), density=True)
            if hist.sum() == 0:
                hist[0] = 1.0
            hist /= hist.sum()
            histograms.append(hist)
        return np.array(histograms)
    H1 = clustering_histogram(graphs_1, bins)
    H2 = clustering_histogram(graphs_2, bins)
    K_xx = gaussian_emd_kernel(H1, H1, sigma)
    K_yy = gaussian_emd_kernel(H2, H2, sigma)
    K_xy = gaussian_emd_kernel(H1, H2, sigma)
    mmd = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
    return float(mmd)


def compute_mmd_orbit(graphs_1, graphs_2, sigma=1.0):
    from networkx.algorithms import graphlet_degree_vectors
    import numpy as np
    def orbit_histogram(graphs):
        histograms = []
        for G in graphs:
            try:
                gdv = graphlet_degree_vectors(G, 4)
                counts = np.array([v for vec in gdv.values() for v in vec])
                if len(counts) == 0:
                    counts = np.zeros(1)
            except Exception:
                counts = np.zeros(1)
            counts = counts.astype(float)
            counts /= counts.sum() + 1e-8  # normalize to make it a histogram
            histograms.append(counts)

        max_len = max(len(h) for h in histograms)
        padded = [np.pad(h, (0, max_len - len(h))) for h in histograms]
        return np.array(padded)
    H1 = orbit_histogram(graphs_1)
    H2 = orbit_histogram(graphs_2)
    K_xx = gaussian_emd_kernel(H1, H1, sigma)
    K_yy = gaussian_emd_kernel(H2, H2, sigma)
    K_xy = gaussian_emd_kernel(H1, H2, sigma)
    mmd = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
    return float(mmd)

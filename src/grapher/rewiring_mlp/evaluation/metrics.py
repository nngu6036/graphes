from __future__ import annotations

from math import comb
from typing import Callable, Sequence

import networkx as nx
import numpy as np

try:
    from scipy.spatial.distance import cdist as _scipy_cdist
except ImportError:  # Keep the core metric usable without the optional SciPy stack.
    _scipy_cdist = None

from grapher.properties.summary import (
    SummaryConfig,
    clustering_histogram,
    degree_histogram,
    graphlet_statistics_summary,
    motif_proxy_vector,
    orbit_count_vector,
    spectral_histogram,
)
from grapher.utils.motifs import (
    attributed_graphlet_count_dict,
    flatten_graphlet_history,
    normalize_count_dict,
    topology_graphlet_keys_by_size,
)

_EMD_BLOCK_SIZE = 512
_EMD_MEDIAN_MAX_PAIRS = 100_000


def descriptor_matrix(
    graphs: Sequence[nx.Graph], fn: Callable[[nx.Graph], np.ndarray]
) -> np.ndarray:
    rows = [np.asarray(fn(g), dtype=np.float64).reshape(-1) for g in graphs]
    if not rows:
        return np.zeros((0, 1), dtype=np.float64)
    width = max(row.size for row in rows)
    out = np.zeros((len(rows), width), dtype=np.float64)
    for i, row in enumerate(rows):
        out[i, : row.size] = row
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _median_sigma(x: np.ndarray, y: np.ndarray) -> float:
    z = np.vstack([x, y])
    if z.shape[0] < 2:
        return 1.0
    d = np.sum((z[:, None, :] - z[None, :, :]) ** 2, axis=-1)
    vals = d[np.triu_indices(z.shape[0], k=1)]
    vals = vals[vals > 0]
    if vals.size == 0:
        return 1.0
    return float(np.sqrt(0.5 * np.median(vals))) or 1.0


def mmd_rbf(x: np.ndarray, y: np.ndarray, sigma: float | None = None) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size == 0 or y.size == 0:
        return float("nan")
    sigma = _median_sigma(x, y) if sigma is None else float(sigma)
    gamma = 1.0 / (2.0 * sigma * sigma)
    kxx = np.exp(-gamma * np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=-1))
    kyy = np.exp(-gamma * np.sum((y[:, None, :] - y[None, :, :]) ** 2, axis=-1))
    kxy = np.exp(-gamma * np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1))
    return float(np.mean(kxx) + np.mean(kyy) - 2.0 * np.mean(kxy))


def gaussian_emd_kernel(
    x: np.ndarray,
    y: np.ndarray,
    sigma: float = 1.0,
    *,
    distance_scaling: float = 1.0,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if y.ndim == 1:
        y = y.reshape(1, -1)
    width = max(x.shape[1], y.shape[1])
    xp = np.zeros((x.shape[0], width), dtype=np.float64)
    yp = np.zeros((y.shape[0], width), dtype=np.float64)
    xp[:, : x.shape[1]] = x
    yp[:, : y.shape[1]] = y
    emd = np.sum(
        np.abs(np.cumsum(xp[:, None, :] - yp[None, :, :], axis=-1)),
        axis=-1,
    ) / max(float(distance_scaling), 1.0e-12)
    return np.exp(-(emd * emd) / (2.0 * float(sigma) * float(sigma)))


def _padded_cdfs(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    width = max(x.shape[1], y.shape[1])
    xp = np.zeros((x.shape[0], width), dtype=np.float64)
    yp = np.zeros((y.shape[0], width), dtype=np.float64)
    xp[:, : x.shape[1]] = x
    yp[:, : y.shape[1]] = y
    return np.cumsum(xp, axis=1), np.cumsum(yp, axis=1)


def _paired_emd_distances(
    cdfs: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    *,
    distance_scaling: float,
    chunk_size: int = 4096,
) -> np.ndarray:
    values = np.empty(left.size, dtype=np.float64)
    scaling = max(float(distance_scaling), 1.0e-12)
    for start in range(0, left.size, int(chunk_size)):
        stop = min(start + int(chunk_size), left.size)
        values[start:stop] = np.sum(
            np.abs(cdfs[left[start:stop]] - cdfs[right[start:stop]]),
            axis=1,
        ) / scaling
    return values


def _median_emd_bandwidth(
    cdfs: np.ndarray,
    *,
    distance_scaling: float,
    max_pairs: int = _EMD_MEDIAN_MAX_PAIRS,
) -> float:
    """Median positive EMD, using deterministic pair sampling at large scale."""

    count = int(cdfs.shape[0])
    if count < 2:
        return 1.0
    pair_count = count * (count - 1) // 2
    if pair_count <= int(max_pairs):
        left, right = np.triu_indices(count, k=1)
    else:
        rng = np.random.default_rng(0)
        left = rng.integers(0, count, size=int(max_pairs), dtype=np.int64)
        right = rng.integers(0, count, size=int(max_pairs), dtype=np.int64)
        equal = left == right
        while np.any(equal):
            right[equal] = rng.integers(
                0, count, size=int(np.count_nonzero(equal)), dtype=np.int64
            )
            equal = left == right
    values = _paired_emd_distances(
        cdfs,
        left,
        right,
        distance_scaling=distance_scaling,
    )
    values = values[values > 0.0]
    return float(np.median(values)) if values.size else 1.0


def _gaussian_emd_kernel_mean_from_cdfs(
    x_cdf: np.ndarray,
    y_cdf: np.ndarray,
    *,
    sigma: float,
    distance_scaling: float,
    symmetric: bool,
    block_size: int = _EMD_BLOCK_SIZE,
) -> float:
    """Compute a kernel mean without materializing an N x M x D tensor."""

    if _scipy_cdist is None:
        # The NumPy fallback creates a B x B x D difference workspace.
        block_size = min(int(block_size), 128)
    gamma = 1.0 / (2.0 * float(sigma) * float(sigma))
    scaling = max(float(distance_scaling), 1.0e-12)
    total = 0.0
    for i_start in range(0, x_cdf.shape[0], int(block_size)):
        x_block = x_cdf[i_start : i_start + int(block_size)]
        j_first = i_start if symmetric else 0
        for j_start in range(j_first, y_cdf.shape[0], int(block_size)):
            y_block = y_cdf[j_start : j_start + int(block_size)]
            if _scipy_cdist is not None:
                distances = _scipy_cdist(x_block, y_block, metric="cityblock") / scaling
            else:
                difference = x_block[:, None, :] - y_block[None, :, :]
                np.abs(difference, out=difference)
                distances = np.sum(difference, axis=-1) / scaling
            block_sum = float(np.exp(-gamma * distances * distances).sum())
            total += block_sum if not symmetric or i_start == j_start else 2.0 * block_sum
    return total / float(x_cdf.shape[0] * y_cdf.shape[0])


def mmd_gaussian_emd(
    x: np.ndarray,
    y: np.ndarray,
    sigma: float | None = None,
    *,
    distance_scaling: float = 1.0,
) -> float:
    """Biased Gaussian-EMD MMD with bounded-memory exact kernel means.

    The median bandwidth is exact for at most ``_EMD_MEDIAN_MAX_PAIRS`` pairs
    and uses a fixed-seed pair sample above that threshold.
    """

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size == 0 or y.size == 0:
        return float("nan")
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if y.ndim == 1:
        y = y.reshape(1, -1)
    x_cdf, y_cdf = _padded_cdfs(x, y)
    if sigma is None:
        sigma = _median_emd_bandwidth(
            np.vstack([x_cdf, y_cdf]),
            distance_scaling=distance_scaling,
        )
    sigma = max(float(sigma), 1.0e-12)
    mean_xx = _gaussian_emd_kernel_mean_from_cdfs(
        x_cdf,
        x_cdf,
        sigma=sigma,
        distance_scaling=distance_scaling,
        symmetric=True,
    )
    mean_yy = _gaussian_emd_kernel_mean_from_cdfs(
        y_cdf,
        y_cdf,
        sigma=sigma,
        distance_scaling=distance_scaling,
        symmetric=True,
    )
    mean_xy = _gaussian_emd_kernel_mean_from_cdfs(
        x_cdf,
        y_cdf,
        sigma=sigma,
        distance_scaling=distance_scaling,
        symmetric=False,
    )
    return float(mean_xx + mean_yy - 2.0 * mean_xy)


def mmd_orbit_graphrnn(
    reference: Sequence[nx.Graph],
    generated: Sequence[nx.Graph],
    sigma: float = 30.0,
) -> float:
    """GraphRNN/SPECTRE four-node orbit MMD.

    ``orbit_count_vector`` already returns the mean per-node 15-orbit vector,
    matching GraphRNN's ORCA ``sum(axis=0) / num_nodes`` descriptor.  The
    historical benchmark applies a Gaussian L2 kernel with sigma=30 and does
    *not* renormalize this vector into a probability histogram.
    """

    ref = descriptor_matrix(reference, orbit_count_vector)
    gen = descriptor_matrix(generated, orbit_count_vector)
    return mmd_rbf(ref, gen, sigma=float(sigma))


def orbit_histogram_matrix(graphs: Sequence[nx.Graph]) -> np.ndarray:
    def histogram(g: nx.Graph) -> np.ndarray:
        counts = np.asarray(orbit_count_vector(g), dtype=np.float64).reshape(-1)
        total = float(np.sum(counts))
        return counts / (total + 1e-8)

    return descriptor_matrix(graphs, histogram)


def mmd_orbit(
    reference: Sequence[nx.Graph], generated: Sequence[nx.Graph], sigma: float = 1.0
) -> float:
    h_ref = orbit_histogram_matrix(reference)
    h_gen = orbit_histogram_matrix(generated)
    if h_ref.size == 0 or h_gen.size == 0:
        return float("nan")
    return mmd_gaussian_emd(h_ref, h_gen, sigma=float(sigma))


def mmd_graphlet_statistics(
    reference: Sequence[nx.Graph],
    generated: Sequence[nx.Graph],
    *,
    k_min: int = 3,
    k_max: int = 5,
    connected_only: bool = True,
    topology_filter: str = "all",
    num_samples: int | None = None,
    backend: str = "sampled",
    node_label_attr: str | None = None,
    edge_label_attr: str | None = None,
    attributed_backend: str = "auto",
    return_by_order: bool = False,
) -> (
    tuple[float, float]
    | tuple[float, float, dict[str, float], dict[str, float]]
):
    """MMD for selected graphlet composition and induced-subset mass."""

    cfg = SummaryConfig(
        graphlet_history=True,
        graphlet_k_min=int(k_min),
        graphlet_k_max=int(k_max),
        graphlet_connected_only=bool(connected_only),
        graphlet_topology_filter=str(topology_filter),
        graphlet_num_samples=num_samples,
        graphlet_backend=str(backend),
    )
    # Build a shared basis from both sets. This is important because generated
    # graphs may contain graphlets unseen in the reference or vice versa.
    combined = list(reference) + list(generated)
    attributed = node_label_attr is not None or edge_label_attr is not None
    if attributed and (not node_label_attr or not edge_label_attr):
        raise ValueError(
            "Attributed graphlet evaluation requires both node_label_attr and "
            "edge_label_attr."
        )
    if attributed:
        histories: list[dict[str, dict[str, float]]] = []
        masses: list[dict[str, float]] = []
        keys_by_k: dict[str, list[str]] = {
            str(k): [] for k in range(int(k_min), int(k_max) + 1)
        }
        rng = np.random.default_rng(0)
        for graph in combined:
            history: dict[str, dict[str, float]] = {}
            mass: dict[str, float] = {}
            for k in range(int(k_min), int(k_max) + 1):
                counts = attributed_graphlet_count_dict(
                    graph,
                    k,
                    node_label_attr=str(node_label_attr),
                    edge_label_attr=str(edge_label_attr),
                    connected_only=bool(connected_only),
                    topology_filter=cfg.graphlet_topology_filter,
                    num_samples=num_samples,
                    rng=rng,
                    backend=str(attributed_backend),
                )
                history[str(k)] = normalize_count_dict(counts)
                keys_by_k[str(k)].extend(str(key) for key in counts)
                total = (
                    comb(graph.number_of_nodes(), k)
                    if graph.number_of_nodes() >= k
                    else 0
                )
                sampled = (
                    min(total, int(num_samples))
                    if num_samples is not None and int(num_samples) > 0
                    else total
                )
                mass[str(k)] = (
                    float(sum(counts.values()) / sampled)
                    if sampled > 0
                    and (connected_only or cfg.graphlet_topology_filter != "all")
                    else float(sampled > 0)
                )
            histories.append(history)
            masses.append(mass)
        keys_by_k = {key: sorted(set(values)) for key, values in keys_by_k.items()}
    else:
        statistics = [graphlet_statistics_summary(graph, cfg) for graph in combined]
        histories = [item[0] for item in statistics]
        masses = [item[1] for item in statistics]
        keys_by_k = topology_graphlet_keys_by_size(
            int(k_min),
            int(k_max),
            connected_only=bool(connected_only),
            topology_filter=cfg.graphlet_topology_filter,
        )
    ref_rows = [
        flatten_graphlet_history(h, keys_by_k) for h in histories[: len(reference)]
    ]
    gen_rows = [
        flatten_graphlet_history(h, keys_by_k) for h in histories[len(reference) :]
    ]
    if not ref_rows or not gen_rows:
        if return_by_order:
            empty = {
                str(k): float("nan")
                for k in range(int(k_min), int(k_max) + 1)
            }
            return float("nan"), float("nan"), dict(empty), dict(empty)
        return float("nan"), float("nan")
    mass_keys = [str(k) for k in range(int(k_min), int(k_max) + 1)]
    ref_mass = np.asarray(
        [
            [item.get(key, 0.0) for key in mass_keys]
            for item in masses[: len(reference)]
        ],
        dtype=np.float64,
    )
    gen_mass = np.asarray(
        [
            [item.get(key, 0.0) for key in mass_keys]
            for item in masses[len(reference) :]
        ],
        dtype=np.float64,
    )
    histogram_width = sum(len(values) for values in keys_by_k.values())
    histogram_mmd = (
        0.0
        if histogram_width == 0
        else mmd_gaussian_emd(
            np.asarray(ref_rows, dtype=np.float64),
            np.asarray(gen_rows, dtype=np.float64),
        )
    )
    selected_mass_mmd = mmd_gaussian_emd(ref_mass, gen_mass)
    if not return_by_order:
        return histogram_mmd, selected_mass_mmd

    histogram_by_order: dict[str, float] = {}
    mass_by_order: dict[str, float] = {}
    for key in mass_keys:
        order_keys = {key: keys_by_k.get(key, [])}
        order_ref_rows = [
            flatten_graphlet_history({key: history.get(key, {})}, order_keys)
            for history in histories[: len(reference)]
        ]
        order_gen_rows = [
            flatten_graphlet_history({key: history.get(key, {})}, order_keys)
            for history in histories[len(reference) :]
        ]
        order_width = len(order_keys[key])
        histogram_by_order[key] = (
            0.0
            if order_width == 0
            else mmd_gaussian_emd(
                np.asarray(order_ref_rows, dtype=np.float64),
                np.asarray(order_gen_rows, dtype=np.float64),
            )
        )
        order_ref_mass = np.asarray(
            [[item.get(key, 0.0)] for item in masses[: len(reference)]],
            dtype=np.float64,
        )
        order_gen_mass = np.asarray(
            [[item.get(key, 0.0)] for item in masses[len(reference) :]],
            dtype=np.float64,
        )
        mass_by_order[key] = mmd_gaussian_emd(order_ref_mass, order_gen_mass)
    return histogram_mmd, selected_mass_mmd, histogram_by_order, mass_by_order


def connectedness_rate(graphs: Sequence[nx.Graph]) -> float:
    if not graphs:
        return 0.0
    return float(
        np.mean(
            [nx.is_connected(g) if g.number_of_nodes() > 0 else False for g in graphs]
        )
    )


def simple_graph_validity_rate(graphs: Sequence[nx.Graph]) -> float:
    if not graphs:
        return 0.0
    vals = []
    for g in graphs:
        vals.append(
            isinstance(g, nx.Graph)
            and not g.is_directed()
            and not g.is_multigraph()
            and nx.number_of_selfloops(g) == 0
        )
    return float(np.mean(vals))


def wl_uniqueness_rate(graphs: Sequence[nx.Graph]) -> float:
    if not graphs:
        return 0.0
    hashes = [nx.weisfeiler_lehman_graph_hash(g) for g in graphs]
    return float(len(set(hashes)) / len(hashes))


def wl_novelty_rate(generated: Sequence[nx.Graph], train: Sequence[nx.Graph]) -> float:
    if not generated:
        return 0.0
    train_hashes = {nx.weisfeiler_lehman_graph_hash(g) for g in train}
    gen_hashes = [nx.weisfeiler_lehman_graph_hash(g) for g in generated]
    return float(np.mean([h not in train_hashes for h in gen_hashes]))


def degree_preservation_rate(
    before: Sequence[nx.Graph], after: Sequence[nx.Graph]
) -> float:
    if not before:
        return 0.0
    if len(before) != len(after):
        raise ValueError(
            "Degree-preservation inputs must contain the same number of graphs."
        )
    vals = []
    for g0, g1 in zip(before, after):
        if set(g0.nodes()) != set(g1.nodes()):
            vals.append(False)
            continue
        vals.append(
            all(int(g0.degree(node)) == int(g1.degree(node)) for node in g0.nodes())
        )
    return float(np.mean(vals))


def degree_target_match_rate(
    graphs: Sequence[nx.Graph],
    target_degree_sequences: Sequence[Sequence[int]],
) -> float:
    """Return the fraction of graphs matching target degree multisets.

    Degree sequences are canonicalized before comparison, so the result is
    invariant to node ordering and constructor relabeling.
    """

    if not graphs:
        return 0.0
    if len(graphs) != len(target_degree_sequences):
        raise ValueError(
            "Graphs and target degree sequences must have the same length."
        )
    matches = []
    for graph, target in zip(graphs, target_degree_sequences):
        actual = tuple(sorted((int(d) for _, d in graph.degree()), reverse=True))
        expected = tuple(sorted((int(d) for d in target), reverse=True))
        matches.append(actual == expected)
    return float(np.mean(matches))


def evaluate_graph_sets(
    reference: Sequence[nx.Graph],
    generated: Sequence[nx.Graph],
    train: Sequence[nx.Graph] | None = None,
    *,
    compute_orbit: bool = True,
    compute_graphlet_history: bool = False,
    graphlet_k_min: int = 3,
    graphlet_k_max: int = 5,
    graphlet_connected_only: bool = True,
    graphlet_topology_filter: str = "all",
    graphlet_num_samples: int | None = None,
    graphlet_backend: str = "sampled",
    graphlet_node_label_attr: str | None = None,
    graphlet_edge_label_attr: str | None = None,
    attributed_graphlet_backend: str = "auto",
) -> dict[str, float]:
    max_degree = 0
    for g in list(reference) + list(generated):
        if g.number_of_nodes():
            max_degree = max(max_degree, max(dict(g.degree()).values()))
    deg_ref = descriptor_matrix(reference, lambda g: degree_histogram(g, max_degree))
    deg_gen = descriptor_matrix(generated, lambda g: degree_histogram(g, max_degree))
    clus_ref = descriptor_matrix(reference, lambda g: clustering_histogram(g, 20))
    clus_gen = descriptor_matrix(generated, lambda g: clustering_histogram(g, 20))
    spec_ref = descriptor_matrix(reference, lambda g: spectral_histogram(g, 20))
    spec_gen = descriptor_matrix(generated, lambda g: spectral_histogram(g, 20))
    motif_ref = descriptor_matrix(reference, motif_proxy_vector)
    motif_gen = descriptor_matrix(generated, motif_proxy_vector)
    graphlet_mmd, connected_mass_mmd = (
        mmd_graphlet_statistics(
            reference,
            generated,
            k_min=graphlet_k_min,
            k_max=graphlet_k_max,
            connected_only=graphlet_connected_only,
            topology_filter=graphlet_topology_filter,
            num_samples=graphlet_num_samples,
            backend=graphlet_backend,
            node_label_attr=graphlet_node_label_attr,
            edge_label_attr=graphlet_edge_label_attr,
            attributed_backend=attributed_graphlet_backend,
        )
        if compute_graphlet_history
        else (float("nan"), float("nan"))
    )
    metrics = {
        "num_graphs": float(len(generated)),
        "degree_mmd": mmd_gaussian_emd(deg_ref, deg_gen),
        "clustering_mmd": mmd_gaussian_emd(clus_ref, clus_gen),
        "spectral_mmd": mmd_gaussian_emd(spec_ref, spec_gen),
        "motif_proxy_mmd": mmd_gaussian_emd(motif_ref, motif_gen),
        "orbit_mmd": mmd_orbit(reference, generated) if compute_orbit else float("nan"),
        "graphlet_history_mmd": graphlet_mmd,
        "graphlet_connected_mass_mmd": connected_mass_mmd,
        "connectedness_rate": connectedness_rate(generated),
        "validity_rate": simple_graph_validity_rate(generated),
        "uniqueness_rate": wl_uniqueness_rate(generated),
    }
    if train is not None:
        metrics["novelty_rate"] = wl_novelty_rate(generated, train)
    return metrics

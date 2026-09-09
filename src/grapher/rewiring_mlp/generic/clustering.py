from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np

from grapher.properties.summary import clustering_histogram


def clustering_histogram_bins(config: dict[str, Any] | None) -> int | None:
    """Return enabled target width, or None for legacy scalar/spectral models."""
    values = dict(config or {})
    if not bool(values.get("clustering_histogram", False)):
        return None
    bins = values.get("clustering_bins", 100)
    if isinstance(bins, bool) or int(bins) != bins or int(bins) < 2:
        raise ValueError("structure_summary_prediction.clustering_bins must be an integer >= 2.")
    return int(bins)


def validate_clustering_histogram(values: np.ndarray, *, bins: int | None = None) -> np.ndarray:
    """Validate a probability histogram without silently repairing model outputs."""
    hist = np.asarray(values, dtype=np.float64)
    if hist.ndim != 1 or hist.size < 2 or (bins is not None and hist.size != bins):
        raise ValueError("Clustering histogram must be a vector with the configured number of bins.")
    if not np.all(np.isfinite(hist)) or np.any(hist < 0.0):
        raise ValueError("Clustering histogram must contain finite nonnegative probabilities.")
    if not np.isclose(hist.sum(), 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError("Clustering histogram probabilities must sum to one.")
    return hist / hist.sum()  # remove only floating-point summation error


def extract_clustering_histogram(graph: nx.Graph, bins: int = 100) -> np.ndarray:
    """Fraction of nodes in each local-clustering bin on [0, 1].

    Reuses the external evaluator's numpy.histogram convention: bins are
    left-closed/right-open, except that the final bin includes coefficient 1.
    Nodes with degree < 2 contribute to bin 0; the empty-graph convention also
    places all mass in bin 0. No node counts or targets from other graphs enter.
    """
    if isinstance(bins, bool) or int(bins) != bins or int(bins) < 2:
        raise ValueError("Clustering histogram bins must be an integer >= 2.")
    return validate_clustering_histogram(clustering_histogram(graph, int(bins)), bins=int(bins))


def clustering_histogram_wasserstein(a: np.ndarray, b: np.ndarray) -> float:
    """1-D W1 for equal-width clustering bins (in coefficient units, not bin units)."""
    x = validate_clustering_histogram(a)
    y = validate_clustering_histogram(b, bins=x.size)
    return float(np.abs(np.cumsum(x - y)[:-1]).sum() / x.size)

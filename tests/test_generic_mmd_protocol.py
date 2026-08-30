from __future__ import annotations

import networkx as nx
import numpy as np

from grapher.rewiring_mlp.evaluation.metrics import (
    descriptor_matrix,
    mmd_gaussian_emd,
    mmd_orbit_graphrnn,
)
from grapher.properties.summary import clustering_histogram, degree_histogram
from scripts.evaluate_graph_generation_report import _paper_mmd


def test_graphrnn_protocol_uses_fixed_historical_kernels() -> None:
    reference = [nx.path_graph(5), nx.cycle_graph(5)]
    candidate = [nx.star_graph(4), nx.complete_graph(5)]
    max_degree = max(
        max(dict(g.degree()).values()) for g in reference + candidate
    )
    degree_ref = descriptor_matrix(reference, lambda g: degree_histogram(g, max_degree))
    degree_gen = descriptor_matrix(candidate, lambda g: degree_histogram(g, max_degree))
    cluster_ref = descriptor_matrix(reference, lambda g: clustering_histogram(g, 100))
    cluster_gen = descriptor_matrix(candidate, lambda g: clustering_histogram(g, 100))

    observed = _paper_mmd(reference, candidate, metric_protocol="graphrnn")

    assert observed["degree_mmd"] == mmd_gaussian_emd(
        degree_ref, degree_gen, sigma=1.0
    )
    assert observed["clustering_mmd"] == mmd_gaussian_emd(
        cluster_ref, cluster_gen, sigma=0.1, distance_scaling=100.0
    )
    assert observed["orbit_mmd"] == mmd_orbit_graphrnn(
        reference, candidate, sigma=30.0
    )


def test_distance_scaling_matches_histogram_bin_coordinate_scaling() -> None:
    left = np.asarray([[1.0, 0.0, 0.0]])
    right = np.asarray([[0.0, 0.0, 1.0]])
    unscaled = mmd_gaussian_emd(left, right, sigma=1.0, distance_scaling=1.0)
    scaled = mmd_gaussian_emd(left, right, sigma=1.0, distance_scaling=10.0)
    assert scaled < unscaled

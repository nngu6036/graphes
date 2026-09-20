import networkx as nx
import numpy as np
import pytest

from grapher.rewiring_mlp.properties.summary import spectral_histogram


@pytest.mark.parametrize(
    "graph, expected_bins",
    [
        (nx.empty_graph(3), {0: 1.0}),
        (nx.disjoint_union(nx.path_graph(2), nx.empty_graph(1)), {0: 2 / 3, 19: 1 / 3}),
        (nx.path_graph(4), {0: 1 / 4, 5: 1 / 4, 15: 1 / 4, 19: 1 / 4}),
        (nx.cycle_graph(6), {0: 1 / 6, 5: 2 / 6, 15: 2 / 6, 19: 1 / 6}),
    ],
)
def test_known_normalized_laplacian_spectra(graph, expected_bins):
    expected = np.zeros(20)
    for index, mass in expected_bins.items():
        expected[index] = mass
    np.testing.assert_allclose(spectral_histogram(graph), expected, atol=1e-14)


@pytest.mark.parametrize("bins", [10, 20, 200])
def test_spectral_histogram_is_invariant_to_node_order(bins):
    graphs = [
        nx.cycle_graph(6),
        nx.path_graph(8),
        nx.stochastic_block_model([10, 10], [[0.7, 0.05], [0.05, 0.7]], seed=42),
    ]
    rng = np.random.default_rng(1)
    for graph in graphs:
        expected = spectral_histogram(graph, bins)
        for _ in range(20):
            order = rng.permutation(len(graph)).tolist()
            permuted = nx.from_numpy_array(nx.to_numpy_array(graph, nodelist=order))
            np.testing.assert_array_equal(spectral_histogram(permuted, bins), expected)


def test_failed_eigendecomposition_does_not_fabricate_a_spectrum(monkeypatch):
    def fail(_):
        raise np.linalg.LinAlgError("did not converge")

    monkeypatch.setattr(np.linalg, "eigvalsh", fail)
    with pytest.raises(np.linalg.LinAlgError, match="did not converge"):
        spectral_histogram(nx.path_graph(4))


def test_empty_spectral_histogram():
    np.testing.assert_array_equal(spectral_histogram(nx.empty_graph()), np.zeros(20))

import networkx as nx
import numpy as np
import pytest

from grapher.rewiring_mlp.evaluation import metrics


@pytest.mark.parametrize("attributed", [False, True])
@pytest.mark.parametrize("return_by_order", [False, True])
def test_graphlet_progress_preserves_results_and_covers_all_kernels(
    attributed, return_by_order, monkeypatch, capsys,
):
    reference = [nx.path_graph(5), nx.cycle_graph(5)]
    generated = [nx.complete_graph(5)]
    options = {
        "k_min": 3,
        "k_max": 4,
        "return_by_order": return_by_order,
    }
    if attributed:
        for graph in reference + generated:
            nx.set_node_attributes(graph, {n: n % 2 for n in graph}, "atom")
            nx.set_edge_attributes(graph, 1, "bond")
        options.update(
            node_label_attr="atom", edge_label_attr="bond",
            attributed_backend="python",
        )
    expected = metrics.mmd_graphlet_statistics(reference, generated, **options)
    assert capsys.readouterr().out == ""

    events = []
    completed_kernels = []
    kernel = metrics.mmd_gaussian_emd

    def measured_kernel(*args, **kwargs):
        assert events[-1] == ("graphlet MMD", 0, 1)
        result = kernel(*args, **kwargs)
        completed_kernels.append(result)
        return result

    def progress(phase, completed, total):
        events.append((phase, completed, total))
        if phase == "graphlet MMD" and completed == total:
            assert len(completed_kernels) == (6 if return_by_order else 2)

    monkeypatch.setattr(metrics, "mmd_gaussian_emd", measured_kernel)
    actual = metrics.mmd_graphlet_statistics(
        reference, generated, progress=progress, **options,
    )

    assert actual == expected
    assert events == [
        ("graphlet descriptors", completed, 3) for completed in range(4)
    ] + [("graphlet MMD", 0, 1), ("graphlet MMD", 1, 1)]


def test_graphlet_progress_finishes_for_empty_inputs():
    events = []
    result = metrics.mmd_graphlet_statistics(
        [], [], progress=lambda *event: events.append(event),
    )
    assert all(np.isnan(value) for value in result)
    assert events == [
        ("graphlet descriptors", 0, 0),
        ("graphlet MMD", 0, 1),
        ("graphlet MMD", 1, 1),
    ]


@pytest.mark.parametrize("graphrnn", [False, True])
def test_orbit_progress_preserves_descriptor_normalization(
    graphrnn, monkeypatch, capsys,
):
    reference = [nx.path_graph(4), nx.cycle_graph(4)]
    generated = [nx.complete_graph(4)]

    def orbit_vector(graph):
        return np.array([graph.number_of_nodes(), graph.number_of_edges(), 0.0])

    monkeypatch.setattr(metrics, "orbit_count_vector", orbit_vector)
    metric = metrics.mmd_orbit_graphrnn if graphrnn else metrics.mmd_orbit
    ref = np.stack([orbit_vector(graph) for graph in reference])
    gen = np.stack([orbit_vector(graph) for graph in generated])
    if graphrnn:
        expected = metrics.mmd_rbf(ref, gen, sigma=30.0)
    else:
        ref = ref / (ref.sum(axis=1, keepdims=True) + 1e-8)
        gen = gen / (gen.sum(axis=1, keepdims=True) + 1e-8)
        expected = metrics.mmd_gaussian_emd(ref, gen, sigma=1.0)
    assert metric(reference, generated) == expected
    assert capsys.readouterr().out == ""

    events = []
    actual = metric(
        reference, generated, progress=lambda *event: events.append(event),
    )
    assert actual == expected
    assert events == [
        ("orbit descriptors", completed, 3) for completed in range(4)
    ] + [("orbit MMD", 0, 1), ("orbit MMD", 1, 1)]

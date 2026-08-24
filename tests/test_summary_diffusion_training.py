from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from grapher.rewiring_mlp.generic.basis import TopologyGraphletBasis
from grapher.rewiring_mlp.generic.spectral import laplacian_eigenvalues
from grapher.rewiring_mlp.generic.spectral_data import (
    TopologySpectralDiffusionIterableDataset,
    build_spectral_diffusion_examples,
)


def _graph() -> nx.Graph:
    graph = nx.Graph()
    graph.add_edges_from(
        [
            (0, 1), (0, 2), (0, 3),
            (1, 2), (1, 4),
            (2, 5), (3, 4), (3, 6),
            (4, 7), (5, 6), (5, 7), (6, 7),
        ]
    )
    assert nx.is_connected(graph)
    return graph


def _basis() -> TopologyGraphletBasis:
    return TopologyGraphletBasis.from_config(
        {
            "graphlet_history": True,
            "graphlet_k_min": 3,
            "graphlet_k_max": 4,
            "graphlet_connected_only": True,
        }
    )


def test_diffusion_training_states_are_continuous_not_rewired_graph_states() -> None:
    examples, report = build_spectral_diffusion_examples(
        [_graph()],
        diffusion_config={
            "schedule": "linear",
            "spectral_sigma": 0.5,
            "graphlet_sigma": 0.5,
            "samples_per_graph": 8,
            "paths_per_graph": 1,
            "time_sampling": "stratified",
            "preserve_spectral_trace": True,
            "fix_spectral_lambda1": True,
        },
        source_config={
            "ensure_connected_source": True,
            "random_relabel_source": False,
            "source_randomization_steps": 0,
        },
        spectral_config={"require_same_degree_sequence": True},
        graphlet_basis=_basis(),
        seed=7,
    )
    assert report["rewiring_used_for_training_states"] is False
    assert report["training_state_source"] == "continuous_summary_diffusion"
    assert len(examples) == 8

    # The graph object is the fixed source context, while the current spectrum
    # is a stochastic continuous bridge sample.  At least one interior state
    # must therefore differ from the source graph's exact spectrum.
    assert any(
        not np.allclose(
            example.current_spectrum,
            laplacian_eigenvalues(example.current_graph),
            atol=1.0e-8,
        )
        for example in examples
        if 0.05 < example.time < 0.95
    )

    # Same-degree endpoints share trace, and projected diffusion noise preserves it.
    for example in examples:
        assert float(np.sum(example.current_spectrum)) == pytest.approx(
            float(np.sum(example.source_spectrum)), abs=2.0e-5
        )
        assert float(example.current_spectrum[0]) == pytest.approx(0.0, abs=1.0e-8)

        assert example.current_graphlet_logits is not None
        assert example.graphlet_coordinate_mask is not None
        basis = _basis()
        for start, stop in basis.simplex_slices:
            if np.any(example.graphlet_coordinate_mask[start:stop]):
                assert float(example.current_graphlet_logits[start:stop].sum()) == pytest.approx(
                    0.0, abs=2.0e-5
                )


def test_streaming_diffusion_resamples_noise_between_epochs() -> None:
    dataset = TopologySpectralDiffusionIterableDataset(
        [_graph()],
        diffusion_config={
            "spectral_sigma": 0.4,
            "samples_per_graph": 3,
            "paths_per_graph": 1,
            "time_sampling": "stratified",
        },
        source_config={
            "ensure_connected_source": True,
            "random_relabel_source": False,
        },
        spectral_config={"require_same_degree_sequence": True},
        seed=11,
        shuffle_graphs=False,
    )
    dataset.set_epoch(0)
    first = list(dataset)
    dataset.set_epoch(1)
    second = list(dataset)
    assert len(first) == len(second) == 3
    assert any(
        not np.allclose(a.current_spectrum, b.current_spectrum)
        for a, b in zip(first, second)
    )

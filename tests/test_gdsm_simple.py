from __future__ import annotations

import numpy as np
import torch

from grapher.models.gdsm_simple.model import (
    EigenvalueDenoiser,
    make_schedule,
    q_sample,
    reconstruct_soft_adjacency,
)
from grapher.models.registry import create_baseline, normalize_baseline_id


def test_gdsm_simple_registered():
    assert normalize_baseline_id("gdsm-simple") == "gdsm_simple"
    wrapper = create_baseline("gdsm_simple")
    assert wrapper.model_id == "gdsm_simple"


def test_spectral_reconstruction_recovers_symmetric_matrix():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(6, 6))
    a = (a + a.T) / 2
    values, vectors = np.linalg.eigh(a)
    u = torch.tensor(vectors, dtype=torch.float64).unsqueeze(0)
    lam = torch.tensor(values, dtype=torch.float64).unsqueeze(0)
    reconstructed = reconstruct_soft_adjacency(u, lam)[0].numpy()
    assert np.allclose(reconstructed, a, atol=1e-10)


def test_q_sample_end_shapes_and_masked_transformer():
    model = EigenvalueDenoiser(max_nodes=8, hidden_dim=32, num_layers=1, num_heads=4, ff_dim=64)
    schedule = make_schedule(steps=20, beta_start=1e-4, beta_end=0.02)
    clean = torch.randn(3, 8)
    mask = torch.tensor([[1]*6+[0]*2, [1]*7+[0], [1]*8], dtype=torch.bool)
    n = mask.sum(-1)
    t = torch.tensor([0, 5, 19])
    noise = torch.randn_like(clean) * mask
    noisy = q_sample(clean, t, noise, schedule) * mask
    pred = model(noisy, t, mask, n, 20)
    assert pred.shape == clean.shape
    assert torch.all(pred[~mask] == 0)
    assert torch.isfinite(pred).all()


def test_degree_preserving_spectral_rewiring_moves_toward_target():
    import networkx as nx

    from grapher.models.gdsm_simple.refiner import (
        SpectralRewireConfig,
        adjacency_spectral_rmse,
        normalized_adjacency_eigenvalues,
        refine_graph_toward_adjacency_spectrum,
    )

    source = nx.Graph()
    source.add_nodes_from(range(8))
    source.add_edges_from(
        [
            (0, 4),
            (1, 2),
            (1, 4),
            (1, 5),
            (1, 7),
            (2, 4),
            (3, 5),
            (4, 7),
            (5, 6),
            (6, 7),
        ]
    )
    target = source.copy()
    target.remove_edges_from([(0, 4), (5, 6)])
    target.add_edges_from([(0, 5), (4, 6)])
    target_spectrum = normalized_adjacency_eigenvalues(target)

    initial = adjacency_spectral_rmse(source, target_spectrum)
    refined, diagnostics = refine_graph_toward_adjacency_spectrum(
        source,
        target_spectrum,
        rng=np.random.default_rng(123),
        config=SpectralRewireConfig(
            max_steps=1,
            proposal_budget=-1,
            valid_candidate_budget=-1,
            min_relative_improvement=0.0,
            preserve_connectivity_if_source_connected=True,
        ),
    )
    final = adjacency_spectral_rmse(refined, target_spectrum)

    assert final < initial
    assert final < 1.0e-10
    assert [source.degree(i) for i in range(8)] == [refined.degree(i) for i in range(8)]
    assert nx.is_connected(refined)
    assert diagnostics["accepted_steps"] == 1
    assert diagnostics["degree_preserved"] is True
    assert diagnostics["all_accepted_steps_improve"] is True

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


def test_source_preserving_projector_is_sign_invariant():
    from grapher.models.gdsm_simple.refiner import (
        projector_chordal_distance,
        target_eigenspace_projector,
    )

    rng = np.random.default_rng(7)
    q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    lam = np.linspace(-1.0, 1.0, 8)
    p1, indices = target_eigenspace_projector(q, lam, rank=4)
    q2 = q.copy()
    q2[:, indices] *= -1.0
    p2, _ = target_eigenspace_projector(q2, lam, rank=4)
    assert projector_chordal_distance(p1, p2) < 1.0e-12


def test_source_preserving_rewiring_improves_lambda_without_losing_target_eigenspace():
    import networkx as nx

    from grapher.models.gdsm_simple.refiner import (
        SpectralRewireConfig,
        adjacency_spectral_rmse,
        graph_eigenspace_projector,
        normalized_adjacency_eigenvalues,
        projector_chordal_distance,
        refine_graph_toward_adjacency_spectrum,
        target_eigenspace_projector,
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

    target_adj = nx.to_numpy_array(target, nodelist=range(8), dtype=np.float64)
    target_values, target_vectors = np.linalg.eigh(target_adj)
    target_spectrum = target_values / np.sqrt(8.0)
    target_projector, indices = target_eigenspace_projector(
        target_vectors, target_spectrum, rank=4
    )
    source_projector = graph_eigenspace_projector(source, indices)
    initial_projector_error = projector_chordal_distance(
        source_projector, target_projector
    )
    initial_lambda_error = adjacency_spectral_rmse(source, target_spectrum)

    refined, diagnostics = refine_graph_toward_adjacency_spectrum(
        source,
        target_spectrum,
        target_eigenvectors=target_vectors,
        rng=np.random.default_rng(123),
        config=SpectralRewireConfig(
            mode="source_preserving",
            max_steps=1,
            proposal_budget=-1,
            valid_candidate_budget=-1,
            min_relative_improvement=0.0,
            lambda_weight=1.0,
            projector_weight=1.0,
            source_weight=0.0,
            projector_rank=4,
            projector_relative_worsening_tolerance=10.0,
            require_lambda_improvement=True,
            preserve_connectivity_if_source_connected=True,
        ),
    )

    final_lambda_error = adjacency_spectral_rmse(refined, target_spectrum)
    final_projector = graph_eigenspace_projector(refined, indices)
    final_projector_error = projector_chordal_distance(
        final_projector, target_projector
    )
    assert final_lambda_error < initial_lambda_error
    assert final_projector_error <= initial_projector_error + 1.0e-12
    assert [source.degree(i) for i in range(8)] == [refined.degree(i) for i in range(8)]
    assert diagnostics["mode"] == "source_preserving"
    assert diagnostics["accepted_steps"] == 1
    assert diagnostics["all_accepted_steps_improve"] is True


def test_initial_lambda_gate_only_selects_high_residual_samples():
    from grapher.models.gdsm_simple.refiner import initial_lambda_gate

    mask, threshold = initial_lambda_gate(
        [0.1, 0.2, 0.3, 0.4], enabled=True, quantile=0.75
    )
    assert threshold is not None
    assert mask.tolist() == [False, False, False, True]

    all_mask, no_threshold = initial_lambda_gate(
        [0.1, 0.2], enabled=False, quantile=0.75
    )
    assert all_mask.tolist() == [True, True]
    assert no_threshold is None

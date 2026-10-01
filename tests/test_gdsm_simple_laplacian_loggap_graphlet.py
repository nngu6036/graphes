from __future__ import annotations

import torch

from grapher.models.gdsm_simple.vanilla_gsdm import (
    GSDMSpectrumGraphletScore,
    _fit_laplacian_log_gap_stats,
    _laplacian_eigenvalues_to_log_gap_state,
    _log_gap_state_to_laplacian_eigenvalues,
    _graphlet_auxiliary_loss,
)


def test_laplacian_log_gap_roundtrip_preserves_ordered_spectrum() -> None:
    eig = torch.tensor(
        [
            [0.0, 0.4, 1.0, 2.2, 3.0, 0.0],
            [0.0, 0.2, 0.2, 1.7, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    flags = torch.tensor(
        [[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 0, 0]], dtype=torch.float32
    )
    stats = _fit_laplacian_log_gap_stats(eig, flags, epsilon=1e-6, min_std=1e-3)
    state = _laplacian_eigenvalues_to_log_gap_state(
        eig, flags, mean=stats["mean"], std=stats["std"], epsilon=1e-6
    )
    recovered = _log_gap_state_to_laplacian_eigenvalues(
        state,
        flags,
        mean=stats["mean"],
        std=stats["std"],
        epsilon=1e-6,
        exp_clip=20.0,
    )
    assert torch.allclose(recovered[0, :5], eig[0, :5], atol=2e-5)
    assert torch.allclose(recovered[1, :4], eig[1, :4], atol=2e-5)
    assert torch.equal(recovered[:, 0], torch.zeros(2))


def test_arbitrary_log_gap_state_decodes_nonnegative_nondecreasing() -> None:
    state = torch.tensor(
        [[0.0, -4.0, 3.0, -2.0, 7.0], [0.0, 2.0, -8.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    flags = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]], dtype=torch.float32)
    mean = torch.zeros(5)
    std = torch.ones(5)
    eig = _log_gap_state_to_laplacian_eigenvalues(
        state, flags, mean=mean, std=std, epsilon=1e-6, exp_clip=20.0
    )
    for row, n in enumerate([5, 4]):
        active = eig[row, :n]
        assert active[0].item() == 0.0
        assert torch.all(active >= 0.0)
        assert torch.all(active[1:] >= active[:-1])
    assert eig[1, 4].item() == 0.0


def test_joint_graphlet_head_outputs_valid_histograms_and_loss() -> None:
    model = GSDMSpectrumGraphletScore(
        max_feat_num=6,
        max_nodes=6,
        hidden_dim=8,
        depth=2,
        graphlet_slices=((0, 2), (2, 8), (8, 29)),
    )
    b, n = 2, 6
    x = torch.randn(b, n, 6)
    adj = torch.randn(b, n, n)
    adj = 0.5 * (adj + adj.transpose(-1, -2))
    flags = torch.ones(b, n)
    u = torch.eye(n).repeat(b, 1, 1)
    state = torch.randn(b, n)
    outputs = model.forward_all(x, adj, flags, u, state)
    hist, mass = model.graphlet_means_from_outputs(outputs)
    assert hist.shape == (b, 29)
    assert mass.shape == (b, 3)
    for a, z in model.graphlet_slices:
        assert torch.allclose(hist[:, a:z].sum(dim=-1), torch.ones(b), atol=1e-6)
    assert torch.all((mass > 0.0) & (mass < 1.0))

    target = hist.detach().clone()
    mass_target = mass.detach().clone()
    loss, metrics = _graphlet_auxiliary_loss(
        model,
        outputs,
        target,
        mass_target,
        histogram_weight=1.0,
        mass_weight=0.25,
    )
    assert torch.isfinite(loss)
    assert metrics["graphlet_histogram_loss"] >= 0.0
    assert metrics["graphlet_mass_loss"] >= 0.0


def test_loggap_sampling_smoke_preserves_spectral_constraints() -> None:
    from grapher.models.gdsm_simple.vanilla_gsdm import (
        GSDMNodeScore,
        VPSDE,
        _laplacian_eigh_padded,
        sample_batch,
    )

    donor = torch.zeros((1, 5, 5), dtype=torch.float32)
    # Five-node cycle: connected and nontrivial repeated Laplacian eigenvalues.
    for i in range(5):
        donor[0, i, (i + 1) % 5] = 1.0
        donor[0, (i + 1) % 5, i] = 1.0
    sizes = torch.tensor([5], dtype=torch.long)
    eig, _ = _laplacian_eigh_padded(donor, sizes)
    flags = torch.ones((1, 5), dtype=torch.float32)
    stats = _fit_laplacian_log_gap_stats(eig, flags, epsilon=1e-6, min_std=1e-3)
    transform = {
        "kind": "laplacian_log_gap",
        "mean": stats["mean"],
        "std": stats["std"],
        "epsilon": 1e-6,
        "exp_clip": 20.0,
    }
    mx = GSDMNodeScore(max_feat_num=5, hidden_dim=8, depth=1)
    ml = GSDMSpectrumGraphletScore(
        max_feat_num=5,
        max_nodes=5,
        hidden_dim=8,
        depth=1,
        graphlet_slices=((0, 2), (2, 8), (8, 29)),
    )
    sde_x = VPSDE(beta_min=0.1, beta_max=0.2, num_scales=2, device=torch.device("cpu"))
    sde_l = VPSDE(beta_min=0.1, beta_max=0.2, num_scales=2, device=torch.device("cpu"))
    _, soft, sampled_eig, _ = sample_batch(
        mx,
        ml,
        donor_adjacencies=donor,
        donor_sizes=sizes,
        sde_x=sde_x,
        sde_lam=sde_l,
        sample_cfg={
            "predictor": "euler",
            "corrector": "none",
            "n_steps": 0,
            "noise_removal": True,
            "eps": 1e-2,
        },
        eigen_mask_mode="laplacian_nonzero_prefix",
        spectral_operator="combinatorial_laplacian",
        spectral_transform=transform,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(7),
    )
    assert torch.isfinite(soft).all()
    active = sampled_eig[0, :5]
    assert active[0].item() == 0.0
    assert torch.all(active >= 0.0)
    assert torch.all(active[1:] >= active[:-1])

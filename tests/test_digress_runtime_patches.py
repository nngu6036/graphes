from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from grapher.models.digress.workers import common


torch = pytest.importorskip("torch")


def _upstream_cpu_reference(probX, probE, node_mask):
    """Reproduce the attached DiGress sampler on its safe CPU code path."""

    batch_size, num_nodes, _ = probX.shape
    probX[~node_mask] = 1 / probX.shape[-1]
    sampled_nodes = probX.reshape(batch_size * num_nodes, -1).multinomial(1)
    sampled_nodes = sampled_nodes.reshape(batch_size, num_nodes)

    inverse_edge_mask = ~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))
    diagonal = torch.eye(num_nodes).unsqueeze(0).expand(batch_size, -1, -1)
    probE[inverse_edge_mask] = 1 / probE.shape[-1]
    probE[diagonal.bool()] = 1 / probE.shape[-1]
    sampled_edges = probE.reshape(batch_size * num_nodes * num_nodes, -1)
    sampled_edges = sampled_edges.multinomial(1).reshape(
        batch_size,
        num_nodes,
        num_nodes,
    )
    sampled_edges = torch.triu(sampled_edges, diagonal=1)
    sampled_edges = sampled_edges + sampled_edges.transpose(1, 2)
    return SimpleNamespace(X=sampled_nodes, E=sampled_edges)


def _fake_upstream_modules(monkeypatch):
    source_package = ModuleType("src")
    source_package.__path__ = []

    canonical_utils = ModuleType("src.utils")
    canonical_utils.encode_no_edge = lambda edges: edges
    legacy_utils = ModuleType("utils")
    legacy_utils.encode_no_edge = lambda edges: edges

    diffusion_package = ModuleType("src.diffusion")
    diffusion_package.__path__ = []
    canonical_diffusion = ModuleType("src.diffusion.diffusion_utils")
    canonical_diffusion.PlaceHolder = lambda **values: SimpleNamespace(**values)
    canonical_diffusion.sample_discrete_features = lambda *args, **kwargs: None
    legacy_diffusion = ModuleType("diffusion.diffusion_utils")
    legacy_diffusion.sample_discrete_features = lambda *args, **kwargs: None

    source_package.utils = canonical_utils
    source_package.diffusion = diffusion_package
    diffusion_package.diffusion_utils = canonical_diffusion

    monkeypatch.setitem(sys.modules, "src", source_package)
    monkeypatch.setitem(sys.modules, "src.utils", canonical_utils)
    monkeypatch.setitem(sys.modules, "utils", legacy_utils)
    monkeypatch.setitem(sys.modules, "src.diffusion", diffusion_package)
    monkeypatch.setitem(
        sys.modules,
        "src.diffusion.diffusion_utils",
        canonical_diffusion,
    )
    monkeypatch.setitem(sys.modules, "diffusion.diffusion_utils", legacy_diffusion)
    return canonical_utils, legacy_utils, canonical_diffusion, legacy_diffusion


def test_runtime_patch_replaces_cuda_sensitive_boolean_assignments(monkeypatch) -> None:
    (
        canonical_utils,
        legacy_utils,
        canonical_diffusion,
        legacy_diffusion,
    ) = _fake_upstream_modules(monkeypatch)

    common.install_upstream_runtime_patches()

    assert canonical_utils.encode_no_edge is legacy_utils.encode_no_edge
    assert (
        canonical_diffusion.sample_discrete_features
        is legacy_diffusion.sample_discrete_features
    )

    edges = torch.zeros((1, 3, 3, 2), dtype=torch.float32)
    encoded = canonical_utils.encode_no_edge(edges)
    assert torch.all(torch.diagonal(encoded, dim1=1, dim2=2) == 0)
    off_diagonal = ~torch.eye(3, dtype=torch.bool).unsqueeze(0)
    assert torch.all(encoded[..., 0][off_diagonal] == 1)

    node_mask = torch.tensor(
        [[True, True, False], [True, False, False]],
        dtype=torch.bool,
    )
    node_probabilities = torch.zeros((2, 3, 2), dtype=torch.float32)
    node_probabilities[..., 0] = 1
    edge_probabilities = torch.zeros((2, 3, 3, 2), dtype=torch.float32)
    edge_probabilities[..., 1] = 1

    torch.manual_seed(7)
    expected = _upstream_cpu_reference(
        node_probabilities.clone(),
        edge_probabilities.clone(),
        node_mask,
    )
    torch.manual_seed(7)
    sampled = canonical_diffusion.sample_discrete_features(
        node_probabilities,
        edge_probabilities,
        node_mask,
    )

    assert torch.all(node_probabilities[~node_mask] == 0.5)
    valid_edges = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
    diagonal = torch.eye(3, dtype=torch.bool).unsqueeze(0)
    invalid_edges = (~valid_edges) | diagonal
    assert torch.all(edge_probabilities[invalid_edges] == 0.5)
    assert sampled.X.shape == (2, 3)
    assert sampled.E.shape == (2, 3, 3)
    assert sampled.y.shape == (2, 0)
    assert torch.equal(sampled.X, expected.X)
    assert torch.equal(sampled.E, expected.E)
    assert torch.equal(sampled.E, sampled.E.transpose(1, 2))
    assert torch.all(torch.diagonal(sampled.E, dim1=1, dim2=2) == 0)

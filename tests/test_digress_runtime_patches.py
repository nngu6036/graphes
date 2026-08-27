from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from grapher.models.digress.workers import common


torch = pytest.importorskip("torch")


class _FakePlaceHolder(SimpleNamespace):
    def mask(self, node_mask):
        del node_mask
        return self


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


def _upstream_mask_distributions_reference(
    true_X,
    true_E,
    pred_X,
    pred_E,
    node_mask,
):
    row_X = torch.zeros(true_X.size(-1), dtype=torch.float, device=true_X.device)
    row_X[0] = 1
    row_E = torch.zeros(true_E.size(-1), dtype=torch.float, device=true_E.device)
    row_E[0] = 1
    off_diagonal = ~torch.eye(
        node_mask.size(1),
        device=node_mask.device,
        dtype=torch.bool,
    ).unsqueeze(0)
    true_X[~node_mask] = row_X
    pred_X[~node_mask] = row_X
    invalid_edges = ~(
        node_mask.unsqueeze(1) * node_mask.unsqueeze(2) * off_diagonal
    )
    true_E[invalid_edges, :] = row_E
    pred_E[invalid_edges, :] = row_E

    true_X = true_X + 1e-7
    pred_X = pred_X + 1e-7
    true_E = true_E + 1e-7
    pred_E = pred_E + 1e-7
    true_X = true_X / torch.sum(true_X, dim=-1, keepdim=True)
    pred_X = pred_X / torch.sum(pred_X, dim=-1, keepdim=True)
    true_E = true_E / torch.sum(true_E, dim=-1, keepdim=True)
    pred_E = pred_E / torch.sum(pred_E, dim=-1, keepdim=True)
    return true_X, true_E, pred_X, pred_E


def _upstream_reconstruction_mask_reference(
    node_probabilities,
    edge_probabilities,
    node_mask,
):
    node_probabilities[~node_mask] = torch.ones(
        node_probabilities.shape[-1]
    ).type_as(node_probabilities)
    invalid_edges = ~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))
    edge_probabilities[invalid_edges] = torch.ones(
        edge_probabilities.shape[-1]
    ).type_as(edge_probabilities)
    diagonal = torch.eye(edge_probabilities.shape[1]).type_as(
        edge_probabilities
    ).bool()
    diagonal = diagonal.unsqueeze(0).expand(edge_probabilities.shape[0], -1, -1)
    edge_probabilities[diagonal] = torch.ones(
        edge_probabilities.shape[-1]
    ).type_as(edge_probabilities)
    return node_probabilities, edge_probabilities


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
    canonical_diffusion.PlaceHolder = _FakePlaceHolder
    canonical_diffusion.sample_discrete_features = lambda *args, **kwargs: None
    canonical_diffusion.mask_distributions = lambda *args, **kwargs: None
    legacy_diffusion = ModuleType("diffusion.diffusion_utils")
    legacy_diffusion.sample_discrete_features = lambda *args, **kwargs: None
    legacy_diffusion.mask_distributions = lambda *args, **kwargs: None

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
    assert canonical_diffusion.mask_distributions is legacy_diffusion.mask_distributions

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


@pytest.mark.parametrize(
    ("node_classes", "edge_classes"),
    [(1, 2), (3, 4)],
)
def test_distribution_mask_patch_matches_upstream_cpu_semantics(
    monkeypatch,
    node_classes: int,
    edge_classes: int,
) -> None:
    _, _, canonical_diffusion, legacy_diffusion = _fake_upstream_modules(monkeypatch)
    common.install_upstream_runtime_patches()

    assert canonical_diffusion.mask_distributions is legacy_diffusion.mask_distributions
    node_mask = torch.tensor(
        [[True, True, False], [True, False, False]],
        dtype=torch.bool,
    )
    torch.manual_seed(11)
    source_tensors = (
        torch.rand((2, 3, node_classes)),
        torch.rand((2, 3, 3, edge_classes)),
        torch.rand((2, 3, node_classes)),
        torch.rand((2, 3, 3, edge_classes)),
    )
    reference_inputs = tuple(tensor.clone() for tensor in source_tensors)
    patched_inputs = tuple(tensor.clone() for tensor in source_tensors)

    expected = _upstream_mask_distributions_reference(
        *reference_inputs,
        node_mask,
    )
    actual = canonical_diffusion.mask_distributions(
        *patched_inputs,
        node_mask,
    )

    for actual_tensor, expected_tensor in zip(actual, expected):
        assert torch.allclose(actual_tensor, expected_tensor)
    for actual_input, expected_input in zip(patched_inputs, reference_inputs):
        assert torch.equal(actual_input, expected_input)


def test_reconstruction_mask_matches_upstream_cpu_semantics() -> None:
    node_mask = torch.tensor(
        [[True, True, False], [True, False, False]],
        dtype=torch.bool,
    )
    torch.manual_seed(13)
    node_probabilities = torch.rand((2, 3, 1))
    edge_probabilities = torch.rand((2, 3, 3, 2))
    expected = _upstream_reconstruction_mask_reference(
        node_probabilities.clone(),
        edge_probabilities.clone(),
        node_mask,
    )
    actual = common._mask_reconstruction_distributions(
        node_probabilities,
        edge_probabilities,
        node_mask,
    )

    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])
    assert actual[0] is node_probabilities
    assert actual[1] is edge_probabilities


def test_discrete_model_patch_is_installed_idempotently() -> None:
    class FakeModel:
        def reconstruction_logp(self, *args, **kwargs):
            del args, kwargs
            return None

    original = FakeModel.reconstruction_logp
    common.install_discrete_model_runtime_patches(FakeModel)
    replacement = FakeModel.reconstruction_logp
    common.install_discrete_model_runtime_patches(FakeModel)

    assert replacement is not original
    assert FakeModel.reconstruction_logp is replacement
    assert FakeModel._grapher_cuda_indexing_patch is True

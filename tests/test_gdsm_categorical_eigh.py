"""Numerical-regression tests; injected failures do not require a CUDA device."""
from __future__ import annotations

import copy
import warnings

import networkx as nx
import numpy as np
import pytest
import torch

from grapher.models.gdsm_simple.categorical import spectral as sp


@pytest.fixture(autouse=True)
def reset_solver_state(monkeypatch):
    monkeypatch.delenv('GDSM_EIGH_BACKEND', raising=False)
    old_groups, old_counts = sp._EIGH_CPU_GROUPS.copy(), sp._EIGH_COUNTS.copy()
    sp._EIGH_CPU_GROUPS.clear()
    sp._EIGH_COUNTS.clear()
    yield
    sp._EIGH_CPU_GROUPS.clear()
    sp._EIGH_CPU_GROUPS.update(old_groups)
    sp._EIGH_COUNTS.clear()
    sp._EIGH_COUNTS.update(old_counts)


def pack(graphs, device='cpu'):
    n = max(map(len, graphs))
    edges = torch.zeros(len(graphs), n, n, dtype=torch.long, device=device)
    mask = torch.zeros(len(graphs), n, dtype=torch.bool, device=device)
    for i, graph in enumerate(graphs):
        size = len(graph)
        edges[i, :size, :size] = torch.as_tensor(nx.to_numpy_array(graph), dtype=torch.long, device=device)
        mask[i, :size] = True
    return edges, mask


def check_pairs(edges, mask, pairs):
    values, vectors = pairs
    assert values.dtype == vectors.dtype == torch.float32
    assert values.device == vectors.device == edges.device
    assert not values.requires_grad and not vectors.requires_grad
    assert not values[~mask].any()
    for i, n in enumerate(mask.sum(1).tolist()):
        v, u = values[i, :n].double(), vectors[i, :n, :n].double()
        a = (edges[i, :n, :n] > 0).double()
        torch.testing.assert_close((u * (v * n**.5)[None]) @ u.T, a, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(u.T @ u, torch.eye(n, device=u.device, dtype=u.dtype), atol=2e-6, rtol=2e-6)
        assert bool((v[1:] >= v[:-1]).all())
        assert not vectors[i, n:, :].any() and not vectors[i, :, n:].any()
        assert abs(float(v.sum())) < 1e-5


def fail_eigh(*args, **kwargs):
    raise RuntimeError('linalg.eigh: (Batch element 0): The algorithm failed to converge (error code: 1)')


@pytest.mark.parametrize('backend', ['auto', 'cpu'])
@pytest.mark.parametrize('n', [1, 2, 3, 6, 9, 38])
def test_degenerate_graphs_reconstruct_without_jitter(backend, n, monkeypatch):
    monkeypatch.setenv('GDSM_EIGH_BACKEND', backend)
    graphs = [nx.empty_graph(n), nx.complete_graph(n), nx.path_graph(n), nx.star_graph(n-1)]
    if n > 2:
        graphs.append(nx.cycle_graph(n))
    e, mask = pack(graphs)
    original = e.clone()
    rng_state = torch.random.get_rng_state().clone()
    pairs = sp.eigenpairs(e, mask)
    check_pairs(e, mask, pairs)
    assert torch.equal(e, original) and torch.equal(torch.random.get_rng_state(), rng_state)
    assert torch.equal(pairs[0][0], torch.zeros(n))
    assert torch.equal(pairs[1][0], torch.eye(n))


def test_all_atlas_graphs_with_padding_reconstruct():
    graphs = [g for g in nx.graph_atlas_g() if len(g) > 0]
    e, mask = pack(graphs)
    check_pairs(e, mask, sp.eigenpairs(e, mask))


def test_original_device_solver_gets_float64_nonzero_blocks(monkeypatch):
    original = torch.linalg.eigh
    seen = []
    def checked(a):
        assert a.dtype == torch.float64 and a.is_contiguous()
        assert bool(a.bool().any(-1).any(-1).all())
        seen.append(a.shape)
        return original(a)
    monkeypatch.setattr(torch.linalg, 'eigh', checked)
    e, mask = pack([nx.empty_graph(9), nx.complete_graph(9), nx.path_graph(4)])
    check_pairs(e, mask, sp.eigenpairs(e, mask))
    assert seen == [torch.Size([1, 4, 4]), torch.Size([1, 9, 9])]


def test_all_edgeless_and_singletons_need_no_solver(monkeypatch):
    monkeypatch.setattr(torch.linalg, 'eigh', fail_eigh)
    monkeypatch.setattr(np.linalg, 'eigh', lambda *args: pytest.fail('Edgeless graph should not invoke LAPACK'))
    e, mask = pack([nx.empty_graph(9), nx.empty_graph(1), nx.empty_graph(4)])
    check_pairs(e, mask, sp.eigenpairs(e, mask))
    assert sp.eigh_diagnostics()['counts']['edgeless_graphs'] == 3
    assert 'torch_float64_calls' not in sp.eigh_diagnostics()['counts']


def test_convergence_failure_falls_back_and_caches_group(monkeypatch):
    e, mask = pack([nx.cycle_graph(6), nx.complete_graph(6)])
    calls = []
    def broken(a):
        calls.append(a.clone())
        fail_eigh(a)
    monkeypatch.setattr(torch.linalg, 'eigh', broken)
    with pytest.warns(RuntimeWarning, match='Recovered with CPU float64'):
        pairs = sp.eigenpairs(e, mask)
    check_pairs(e, mask, pairs)
    with warnings.catch_warnings(record=True) as caught:
        check_pairs(e, mask, sp.eigenpairs(e, mask))
    assert len(calls) == 1 and not caught
    assert sp.eigh_diagnostics()['counts']['fallback_batches'] == 1
    assert sp.eigh_diagnostics()['cached_cpu_groups'] == [{'device': 'cpu', 'nodes': 6}]
    np.testing.assert_array_equal(calls[0].numpy(), e.numpy())


@pytest.mark.parametrize('kind', ['nan', 'wrong_values', 'nonorthogonal', 'unsorted'])
def test_invalid_successful_solver_output_is_not_used(kind, monkeypatch):
    original = torch.linalg.eigh
    def corrupt(a):
        v, u = original(a)
        if kind == 'nan':
            v[:] = float('nan')
        elif kind == 'wrong_values':
            v[:] = 0
        elif kind == 'nonorthogonal':
            u[:] = 0
        else:
            v, u = v.flip(-1), u.flip(-1)
        return v, u
    monkeypatch.setattr(torch.linalg, 'eigh', corrupt)
    e, mask = pack([nx.cycle_graph(6)])
    with pytest.warns(RuntimeWarning, match='Recovered with CPU float64'):
        pairs = sp.eigenpairs(e, mask)
    check_pairs(e, mask, pairs)


def test_forced_cpu_bypasses_torch_solver(monkeypatch):
    monkeypatch.setenv('GDSM_EIGH_BACKEND', 'cpu')
    monkeypatch.setattr(torch.linalg, 'eigh', lambda *args: pytest.fail('CPU policy used torch eigh'))
    e, mask = pack([nx.path_graph(9)])
    check_pairs(e, mask, sp.eigenpairs(e, mask))
    assert sp.eigh_diagnostics()['backend_policy'] == 'cpu'


def test_numpy_batched_failure_retries_individual_matrices(monkeypatch):
    monkeypatch.setenv('GDSM_EIGH_BACKEND', 'cpu')
    original = np.linalg.eigh
    seen = []
    def broken_batch(a):
        seen.append(a.copy())
        assert a.dtype == np.float64
        if a.ndim == 3:
            raise np.linalg.LinAlgError('Injected batch failure')
        return original(a)
    monkeypatch.setattr(np.linalg, 'eigh', broken_batch)
    e, mask = pack([nx.path_graph(9), nx.complete_graph(9)])
    check_pairs(e, mask, sp.eigenpairs(e, mask))
    assert [a.ndim for a in seen] == [3, 2, 2]
    assert sp.eigh_diagnostics()['counts']['cpu_individual_calls'] == 2


def test_numpy_partial_nonfinite_result_recovers_only_bad_matrix(monkeypatch):
    monkeypatch.setenv('GDSM_EIGH_BACKEND', 'cpu')
    original = np.linalg.eigh
    def corrupt_batch(a):
        v, u = original(a)
        if a.ndim == 3:
            v[1] = np.nan
        return v, u
    monkeypatch.setattr(np.linalg, 'eigh', corrupt_batch)
    e, mask = pack([nx.path_graph(9), nx.cycle_graph(9)])
    check_pairs(e, mask, sp.eigenpairs(e, mask))
    assert sp.eigh_diagnostics()['counts']['cpu_individual_calls'] == 1


def test_both_torch_and_numpy_fail_use_qr_on_unchanged_graph(monkeypatch):
    monkeypatch.setattr(torch.linalg, 'eigh', fail_eigh)
    def fail_numpy(*args):
        raise np.linalg.LinAlgError('Injected NumPy failure')
    monkeypatch.setattr(np.linalg, 'eigh', fail_numpy)
    import scipy.linalg
    original = scipy.linalg.eigh
    seen = []
    def qr(a, **kwargs):
        assert kwargs['driver'] == 'ev'
        seen.append(a.copy())
        return original(a, **kwargs)
    monkeypatch.setattr(scipy.linalg, 'eigh', qr)
    e, mask = pack([nx.cycle_graph(6)])
    with pytest.warns(RuntimeWarning, match='Recovered with CPU float64'):
        pairs = sp.eigenpairs(e, mask)
    check_pairs(e, mask, pairs)
    np.testing.assert_array_equal(seen[0], e[0].numpy())
    assert sp.eigh_diagnostics()['counts']['cpu_qr_calls'] == 1


def test_total_solver_failure_raises_no_fake_eigenpairs(monkeypatch):
    monkeypatch.setattr(torch.linalg, 'eigh', fail_eigh)
    def fail_numpy(*args, **kwargs):
        raise np.linalg.LinAlgError('Injected CPU failure')
    import scipy.linalg
    monkeypatch.setattr(np.linalg, 'eigh', fail_numpy)
    monkeypatch.setattr(scipy.linalg, 'eigh', fail_numpy)
    e, mask = pack([nx.cycle_graph(6)])
    with pytest.raises(RuntimeError, match='No graph was changed or skipped'):
        sp.eigenpairs(e, mask)
    assert not sp.eigh_diagnostics()['cached_cpu_groups']


@pytest.mark.parametrize('message', ['CUDA out of memory', 'CUDA illegal memory access',
                                     'device-side assert triggered', 'Programming bug elsewhere'])
def test_non_numerical_runtime_errors_are_not_hidden(message, monkeypatch):
    def broken(*args):
        raise RuntimeError(message)
    monkeypatch.setattr(torch.linalg, 'eigh', broken)
    e, mask = pack([nx.cycle_graph(6)])
    with pytest.raises(RuntimeError, match=message):
        sp.eigenpairs(e, mask)
    assert not sp.eigh_diagnostics()['cached_cpu_groups']


@pytest.mark.parametrize('bad', ['nan', 'inf', 'fractional', 'negative', 'asymmetric', 'self_loop',
                                'mask_hole', 'empty_graph', 'wrong_shape', 'mask_float'])
def test_invalid_input_is_rejected_before_solver(bad, monkeypatch):
    e, mask = pack([nx.path_graph(6)])
    if bad in {'nan', 'inf', 'fractional', 'negative'}:
        e = e.float()
        e[0, 0, 1] = e[0, 1, 0] = {'nan': float('nan'), 'inf': float('inf'), 'fractional': .5, 'negative': -1}[bad]
    elif bad == 'asymmetric': e[0, 0, 1] = 0
    elif bad == 'self_loop': e[0, 0, 0] = 1
    elif bad == 'mask_hole': mask[0, 1] = False
    elif bad == 'empty_graph': mask[:] = False
    elif bad == 'wrong_shape': e = e[:, :3, :3]
    elif bad == 'mask_float': mask = mask.float()
    monkeypatch.setattr(torch.linalg, 'eigh', lambda *args: pytest.fail('Invalid data reached solver'))
    with pytest.raises(ValueError):
        sp.eigenpairs(e, mask)


def test_invalid_backend_rejected(monkeypatch):
    monkeypatch.setenv('GDSM_EIGH_BACKEND', 'jitter')
    e, mask = pack([nx.path_graph(6)])
    with pytest.raises(ValueError, match='GDSM_EIGH_BACKEND'):
        sp.eigenpairs(e, mask)


def test_fallback_full_model_backward_and_permutation_equivariance(monkeypatch):
    from test_gdsm_categorical import prepared_model, config
    from grapher.models.gdsm_simple.categorical.model import losses
    model, batch, _ = prepared_model()
    model.eval()
    t = torch.tensor([5, 5, 5])
    def call(b):
        return model(b['x'], b['e'], b['z'], t, b['anchor'], b['mask'], 10)
    expected = call(batch)
    monkeypatch.setattr(torch.linalg, 'eigh', fail_eigh)
    with pytest.warns(RuntimeWarning, match='Recovered with CPU float64'):
        actual = call(batch)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], atol=1e-5, rtol=1e-5)
    loss, parts = losses(actual, batch, config()['loss_weights'])
    loss.backward()
    assert torch.isfinite(loss)
    for head in (model.node_head, model.edge_head, model.spec_out, model.graphlet_head,
                 model.mass_head, model.clustering_head, model.orbit_head):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
    altered = copy.deepcopy(batch)
    for i, n in enumerate(batch['mask'].sum(1).tolist()):
        p = torch.arange(n-1, -1, -1)
        altered['x'][i, :n] = batch['x'][i, :n][p]
        altered['e'][i, :n, :n] = batch['e'][i, :n, :n][p][:, p]
    permuted = call(altered)
    for key in ('clean_spectrum', 'graphlet_logits', 'graphlet_mass', 'clustering_logits', 'orbit_log_mean'):
        torch.testing.assert_close(actual[key], permuted[key], atol=2e-5, rtol=2e-5)
    for i, n in enumerate(batch['mask'].sum(1).tolist()):
        torch.testing.assert_close(actual['node_logits'][i, :n].flip(0), permuted['node_logits'][i, :n], atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(actual['edge_logits'][i, :n, :n].flip((0, 1)), permuted['edge_logits'][i, :n, :n], atol=2e-5, rtol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Real CUDA device unavailable')
@pytest.mark.parametrize('backend', ['auto', 'cpu'])
@pytest.mark.parametrize('n', [9, 38])
def test_real_cuda_degenerate_graphs_and_output_device(backend, n, monkeypatch):
    monkeypatch.setenv('GDSM_EIGH_BACKEND', backend)
    e, mask = pack([nx.empty_graph(n), nx.complete_graph(n), nx.path_graph(n), nx.cycle_graph(n)], 'cuda:0')
    check_pairs(e, mask, sp.eigenpairs(e, mask))


def test_training_generation_roundtrip_with_injected_solver_failure(tmp_path, monkeypatch):
    import json
    import pickle
    from test_gdsm_categorical import labelled, options
    from grapher.models.base import DatasetReference, RunSpec, TrainRequest, GenerateRequest
    from grapher.models.gdsm_simple.wrapper import GDSMSimpleWrapper
    from grapher.models.gdsm_simple.categorical.evaluation import audit
    folder = tmp_path / 'data' / 'toy'
    folder.mkdir(parents=True)
    graphs = [labelled(nx.cycle_graph(6)), labelled(nx.path_graph(6)), labelled(nx.complete_graph(4))]
    for split in ('train', 'val', 'test'):
        with (folder / (split + '.pkl')).open('wb') as stream:
            pickle.dump(graphs, stream)
    run = RunSpec('gdsm_simple', 'community_small', 'fallback', 42, tmp_path / 'runs')
    request = TrainRequest(run, DatasetReference('community_small', tmp_path / 'data', 'toy'), options=options())
    monkeypatch.setattr(torch.linalg, 'eigh', fail_eigh)
    with pytest.warns(RuntimeWarning, match='Recovered with CPU float64'):
        artifacts = GDSMSimpleWrapper().train(request)
    generated = GDSMSimpleWrapper().generate(GenerateRequest(run, artifacts.checkpoint_path, 6, 42, generation_id='fallback'))
    result = audit(generated.generation_dir)
    assert result['status'] == 'passed' and result['max_eigenpair_reconstruction_error'] < 1e-5
    diagnostics = json.loads((generated.generation_dir / 'rewiring_diagnostics.json').read_text())
    assert any(g['categorical_degree_change_steps'] > 0 for g in diagnostics['graphs'])
    assert sp.eigh_diagnostics()['counts']['cpu_routed_batches'] > 0

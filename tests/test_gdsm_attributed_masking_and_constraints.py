from __future__ import annotations

import copy
import importlib.util
import json
import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import torch

from grapher.models.base import GenerateRequest, RunSpec, TrainRequest
from grapher.models.gdsm_simple import attributed_loggap as al
from grapher.models.gdsm_simple.attribute_decoding import (
    NEUTRAL_CAPACITIES, sample_atoms, sample_bonds, decoding_diagnostics,
    sample_categorical_logits, validate_decode_config,
)
from grapher.models.gdsm_simple.wrapper import GDSMSimpleWrapper
from test_gdsm_simple_attributed_loggap import _options, _write_dataset


def _small_model():
    return al.AttributedSpectrumPPGNScore(
        max_feat_num=5, max_nodes=5, node_classes=4, edge_classes=3,
        graphlet_slices=((0, 2), (2, 4), (4, 6)), structural_features={'enabled': False},
        clustering_bins=0, orbit_width=0, ppgn_hidden_dim=8, ppgn_depth=1,
    )


def _corruption_inputs(batch=16):
    nl = torch.zeros((batch, 5), dtype=torch.long)
    el = torch.full((batch, 5, 5), -1, dtype=torch.long)
    for i, j, label in [(0, 1, 0), (1, 2, 1), (2, 3, 2)]:
        el[:, i, j] = el[:, j, i] = label
    flags = torch.ones((batch, 5)); flags[:, 4] = 0
    nl[:, 4] = -1
    return nl, el, flags, (el >= 0).float()


@pytest.mark.parametrize('p', [0.0, 0.15, 0.5, 1.0])
def test_undirected_corruption_never_leaks_a_supervised_label(p):
    nl, el, flags, current = _corruption_inputs()
    # One missing true edge and one spurious current edge.
    current[:, 0, 1] = current[:, 1, 0] = 0
    current[:, 0, 3] = current[:, 3, 0] = 1
    ni, ei, nm, em = al._mask_categorical_inputs(
        nl, el, flags, current, torch.full((len(nl),), 0.5), _small_model(),
        {'corruption': {'mask_probability_min': p, 'mask_probability_max': p, 'full_mask_probability': 0}},
        torch.Generator().manual_seed(31),
    )
    assert torch.equal(ei, ei.transpose(1, 2))
    assert torch.equal(em, em.transpose(1, 2))
    assert set(ei.unique().tolist()).issubset({0.0, 1.0})
    assert torch.all(ei[em][:, :-1] == 0)
    assert torch.all(ei[em][:, -1] == 1)
    assert torch.all(ei[current == 0] == 0)
    assert torch.all(ni[:, 4] == 0)
    supervised = (el >= 0) & torch.triu(torch.ones_like(el, dtype=torch.bool), diagonal=1) & (em | ~current.bool())
    assert torch.all(ei[..., :-1][supervised] == 0)
    assert torch.all(ni[nm][:, :-1] == 0)


def test_full_mask_probability_masks_all_active_categories():
    nl, el, flags, current = _corruption_inputs(2)
    ni, ei, nm, em = al._mask_categorical_inputs(
        nl, el, flags, current, torch.zeros(2), _small_model(),
        {'corruption': {'mask_probability_min': 0, 'mask_probability_max': 0, 'full_mask_probability': 1}},
        torch.Generator().manual_seed(8),
    )
    assert torch.equal(nm, flags.bool())
    assert torch.equal(em, current.bool())
    assert ni[..., :-1].sum() == ei[..., :-1].sum() == 0


def test_empty_mask_has_differentiable_zero_losses_not_visible_label_fallback():
    nl, el, flags, current = _corruption_inputs(1)
    out = {'node_logits': torch.randn((1, 5, 4), requires_grad=True),
           'edge_logits': torch.randn((1, 5, 5, 3), requires_grad=True)}
    nc, ec, metrics = al._categorical_denoising_loss(
        out, nl, el, flags, current, torch.zeros_like(nl, dtype=torch.bool), torch.zeros_like(el, dtype=torch.bool),
    )
    assert nc.item() == ec.item() == 0
    assert metrics['node_supervised_count'] == metrics['edge_supervised_count'] == 0
    (nc + ec).backward()
    assert torch.count_nonzero(out['node_logits'].grad) == 0
    assert torch.count_nonzero(out['edge_logits'].grad) == 0


def test_missing_clean_bond_remains_a_legitimate_unknown_target():
    nl, el, flags, current = _corruption_inputs(1)
    current[:, 0, 1] = current[:, 1, 0] = 0
    out = {'node_logits': torch.zeros((1, 5, 4), requires_grad=True),
           'edge_logits': torch.zeros((1, 5, 5, 3), requires_grad=True)}
    nc, ec, metrics = al._categorical_denoising_loss(
        out, nl, el, flags, current, torch.zeros_like(nl, dtype=torch.bool), torch.zeros_like(el, dtype=torch.bool),
    )
    assert metrics['edge_supervised_count'] == 1
    assert ec.item() == pytest.approx(np.log(3))
    assert nc.item() == 0


def _decode_graph(graph, *, mode='atom_bond_valence', node_values=(6, 7, 8, 9), edge_values=(1, 2, 3),
                  node_logits=None, edge_logits=None, edge_order='random', policy='retain', seed=42):
    n = graph.number_of_nodes()
    discrete = torch.tensor(nx.to_numpy_array(graph, weight=None), dtype=torch.bool).unsqueeze(0)
    original = discrete.clone()
    flags = torch.ones((1, n))
    if node_logits is None:
        node_logits = torch.zeros((1, n, len(node_values)))
        node_logits[..., -1] = 10  # F preferred, deliberately incompatible with large degree.
    if edge_logits is None:
        edge_logits = torch.zeros((1, n, n, len(edge_values)))
        edge_logits += torch.arange(len(edge_values)).view(1, 1, 1, -1) * 5
    cfg = {'constraint_mode': mode, 'node_mode': 'argmax', 'edge_mode': 'argmax',
           'edge_order': edge_order, 'infeasible_policy': policy}
    validate_decode_config(cfg, node_values, edge_values)
    gen = torch.Generator().manual_seed(seed)
    nodes, masks = sample_atoms(node_logits, discrete, flags, node_values, cfg, gen)
    edges, rows = sample_bonds(edge_logits, discrete, flags, nodes, node_values, edge_values, cfg, gen)
    rows = decoding_diagnostics(discrete, flags, nodes, edges, node_values, edge_values, masks, rows)
    assert torch.equal(discrete, original)
    assert torch.equal(edges, edges.transpose(1, 2))
    return nodes, edges, rows[0]


def test_degree_aware_atom_sampling_masks_insufficient_capacity():
    nodes, edges, row = _decode_graph(nx.star_graph(4), mode='atom_degree')
    assert nodes[0, 0] == 0  # carbon only at degree four
    assert torch.all(nodes[0, 1:] == 3)  # terminal fluorine is allowed
    assert row['atom_constraint_activations'] == 1
    assert not row['atom_degree_incompatible']
    assert row['bond_valence_exceeded']  # atom-only intentionally does not constrain bonds


@pytest.mark.parametrize('edge_order', ['random', 'lexicographic'])
def test_bond_budget_reserves_single_valence_for_later_edges(edge_order):
    _, edges, row = _decode_graph(nx.star_graph(3), node_values=(6,), edge_order=edge_order)
    # Degree-three carbon has only one extra unit despite the triple-bond preference.
    assert (edges[0, 0, 1:] + 1).sum() == 4
    assert (edges[0, 0, 1:] == 1).sum() == 1  # exactly one double bond
    assert row['capacity_constraints_satisfied']
    assert row['bond_constraint_activations'] == 3
    assert len(row['bond_allocation_order']) == 3


def test_reordered_edge_vocabulary_uses_bond_order_not_class_index():
    _, edges, row = _decode_graph(nx.cycle_graph(3), node_values=(8,), edge_values=(3, 1, 2))
    active = torch.tensor(nx.to_numpy_array(nx.cycle_graph(3)), dtype=torch.bool)
    assert torch.all(edges[0][active] == 1)  # class one represents SINGLE here
    assert row['capacity_constraints_satisfied']


def test_infeasible_topology_is_retained_and_flagged_without_edge_deletion():
    nodes, edges, row = _decode_graph(nx.star_graph(5))
    assert nodes.shape == (1, 6)
    assert row['topology_infeasible']
    assert row['topology_infeasible_node_count'] == 1
    assert not row['capacity_constraints_satisfied']
    assert row['topology_preserved']
    with pytest.raises(ValueError, match='Topology'):
        _decode_graph(nx.star_graph(5), policy='error')


def test_singleton_and_no_edges_are_valid_decoder_cases():
    _, edges, row = _decode_graph(nx.empty_graph(1))
    assert edges.shape == (1, 1, 1)
    assert edges.sum() == 0
    assert row['bond_allocation_order'] == []
    assert row['capacity_constraints_satisfied']


@pytest.mark.parametrize('nodes,edges', [([16], [1, 2, 3]), ([6, 7], [1, 1.5]), ([6], [2, 3])])
def test_constraints_refuse_unsupported_chemistry(nodes, edges):
    with pytest.raises(ValueError):
        validate_decode_config({'constraint_mode': 'atom_bond_valence'}, nodes, edges)


def test_masked_sampler_rejects_impossible_rows_and_nonfinite_logits():
    for logits, error in [(torch.tensor([[-torch.inf, -torch.inf]]), ValueError),
                          (torch.tensor([[0., float('nan')]]), FloatingPointError)]:
        with pytest.raises(error):
            sample_categorical_logits(logits, mode='sample', temperature=1., generator=torch.Generator())
    out = sample_categorical_logits(torch.tensor([[-torch.inf, 0.]]), mode='sample', temperature=.6, generator=torch.Generator())
    assert out.item() == 1


def test_rng_streams_isolate_arbitrary_decoder_random_consumption():
    _, top1, attr1, _ = al._generation_rngs(42, torch.device('cpu'))
    _, top2, attr2, _ = al._generation_rngs(42, torch.device('cpu'))
    for _ in range(3):
        assert torch.equal(torch.randn(13, generator=top1), torch.randn(13, generator=top2))
        torch.rand(100, generator=attr1)
        torch.rand(3, generator=attr2)
    _, legacy_top, legacy_attr, _ = al._generation_rngs(42, torch.device('cpu'), False)
    assert legacy_top is legacy_attr


def test_corrected_training_roundtrip_and_three_way_paired_generation(tmp_path):
    torch.set_num_threads(1)
    dataset = _write_dataset(tmp_path)
    wrapper = GDSMSimpleWrapper()
    run = RunSpec('gdsm_simple', 'qm9', 'corrected-three-way', 42, tmp_path/'runs')
    options = _options()
    trained = wrapper.train(TrainRequest(run, dataset, options=options))
    ckpt = torch.load(trained.checkpoint_path, map_location='cpu', weights_only=False)
    assert ckpt['categorical_training_contract'] == al.CATEGORICAL_TRAINING_CONTRACT
    manifests = []; hashes = []; paths = []
    for mode in ['none', 'atom_degree', 'atom_bond_valence']:
        gen = wrapper.generate(GenerateRequest(
            run, trained.checkpoint_path, 12, 101, generation_id=mode,
            options={'attributed': {'decode': {'constraint_mode': mode, 'node_mode': 'sample',
                      'edge_mode': 'sample', 'edge_temperature': .6, 'edge_order': 'random'}}},
        ))
        paths.append(gen.generation_dir)
        manifest = json.loads(gen.manifest_path.read_text()); manifests.append(manifest)
        hashes.append(json.loads((gen.generation_dir/'topology_pairing.json').read_text())['per_graph_sha256'])
        assert manifest['num_requested'] == manifest['num_generated'] == 12
        assert manifest['num_filtered'] == 0
        rows = pickle.loads(gen.graphs_path.read_bytes())
        assert len(rows) == 12
        if mode == 'atom_bond_valence':
            for graph in rows:
                assert graph.graph['attribute_decoding']['capacity_constraints_satisfied']
                for node, data in graph.nodes(data=True):
                    valence = sum(d['bond_type'] for _, _, d in graph.edges(node, data=True))
                    assert valence <= NEUTRAL_CAPACITIES[data['atomic_num']]
    assert hashes[0] == hashes[1] == hashes[2]
    path = Path(__file__).resolve().parents[1] / 'scripts/verify_gdsm_decoder_pairing.py'
    spec = importlib.util.spec_from_file_location('_verify_pairing', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    assert module.verify_pairing(paths)['paired']
    # Serialized graph mutation is detected even with unchanged hash records.
    corrupt_path = paths[-1]/'base_graphs.pkl'
    rows = pickle.loads(corrupt_path.read_bytes()); rows[0].add_node(999, atomic_num=6)
    corrupt_path.write_bytes(pickle.dumps(rows))
    assert not module.verify_pairing(paths)['paired']


@pytest.mark.parametrize('seed', [42, 43, 44])
@pytest.mark.parametrize('label,mode', [('baseline', 'none'), ('atom_degree', 'atom_degree'),
                                       ('atom_bond_valence', 'atom_bond_valence')])
def test_new_explicit_configs_resolve_and_only_change_generation_controls(tmp_path, seed, label, mode):
    root = Path(__file__).resolve().parents[1]
    config = root/f'configs/experiments/gdsm_laplacian_loggap_attributed_valence_explicit/qm9_seed_{seed}_{label}.yaml'
    from grapher.models.base import DatasetReference
    request = TrainRequest(RunSpec('gdsm_simple', 'qm9', 'cfg', seed, tmp_path/'runs'),
                           DatasetReference('qm9', tmp_path/'datasets', 'unused'), config_path=config)
    options = GDSMSimpleWrapper()._options(request)
    assert options['attributed']['decode']['constraint_mode'] == mode
    assert options['sample']['separate_attribute_rng'] is True
    assert options['structure_summary']['graphlet']['orders'] == [3, 4, 5]
    assert options['attributed']['decode']['edge_temperature'] == .6
    assert options['train']['epochs'] == 200
    assert options['graphlet_refinement']['enabled'] is False


def test_legacy_checkpoint_is_labeled_and_not_silently_reused_for_training(tmp_path):
    from grapher.models.errors import ArtifactCollisionError
    dataset = _write_dataset(tmp_path)
    wrapper = GDSMSimpleWrapper()
    run = RunSpec('gdsm_simple', 'qm9', 'legacy-provenance', 42, tmp_path/'runs')
    opts = _options(); req = TrainRequest(run, dataset, options=opts)
    trained = wrapper.train(req)
    ckpt = torch.load(trained.checkpoint_path, map_location='cpu', weights_only=False)
    ckpt.pop('categorical_training_contract')
    torch.save(ckpt, trained.checkpoint_path)
    manifest = json.loads(trained.manifest_path.read_text())
    manifest.pop('categorical_training_contract')
    manifest['checkpoint']['sha256'] = al._sha256(trained.checkpoint_path)
    trained.manifest_path.write_text(json.dumps(manifest))
    with pytest.warns(RuntimeWarning, match='predates'):
        generated = wrapper.generate(GenerateRequest(run, trained.checkpoint_path, 2, 7, generation_id='legacy'))
    result = json.loads(generated.manifest_path.read_text())
    assert result['categorical_training_contract'] == 'legacy_unverified'
    with pytest.raises(ArtifactCollisionError):
        wrapper.train(req)

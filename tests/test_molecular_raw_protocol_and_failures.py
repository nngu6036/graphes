from __future__ import annotations

import importlib.util
import pickle
from pathlib import Path

import networkx as nx
import pytest

from grapher.rewiring_mlp.evaluation.molecular_failures import molecular_failure_diagnostics


@pytest.fixture(scope='module')
def evaluator():
    path = Path(__file__).resolve().parents[1]/'scripts/evaluate_generated_molecules.py'
    spec = importlib.util.spec_from_file_location('_raw_evaluator', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_raw_protocol_default_and_explicit_legacy(evaluator):
    parser = evaluator.build_parser()
    assert evaluator._resolve_metric_protocol(parser.parse_args([])) == ('raw_valid', False, False)
    assert evaluator._resolve_metric_protocol(parser.parse_args(['--hogdiff-compatible-metrics'])) == ('corrected_valid', True, True)
    assert evaluator._resolve_metric_protocol(parser.parse_args(['--strict-raw-metrics', '--metric-molecule-source', 'raw_valid'])) == ('raw_valid', False, False)


@pytest.mark.parametrize('flags', [
    ['--strict-raw-metrics', '--hogdiff-compatible-metrics'],
    ['--strict-raw-metrics', '--fcd-use-corrected'],
    ['--strict-raw-metrics', '--metric-molecule-source', 'corrected_valid'],
    ['--hogdiff-compatible-metrics', '--metric-molecule-source', 'raw_valid'],
])
def test_conflicting_populations_fail_before_reading_files(evaluator, flags):
    args = evaluator.build_parser().parse_args(flags)
    with pytest.raises(ValueError):
        evaluator.evaluate(args)


def test_explicit_graphlet_scope_is_retained(evaluator):
    args = evaluator.build_parser().parse_args(['--graphlet-mmd', '--graphlet-topology-filter', 'all',
                                               '--graphlet-k-min', '3', '--graphlet-k-max', '5'])
    assert args.graphlet_topology_filter == 'all'
    assert (args.graphlet_k_min, args.graphlet_k_max) == (3, 5)
    # Preserve old default behavior; all new commands explicitly set the scope.
    args = evaluator.build_parser().parse_args([])
    assert args.graphlet_topology_filter == 'simple_cycle'


def _molecule(g, atom=6, bond=1):
    nx.set_node_attributes(g, atom, 'atomic_num')
    nx.set_edge_attributes(g, bond, 'bond_type')
    return g


def test_failure_categories_and_marginals_include_all_draws():
    graphs = [_molecule(nx.path_graph(2)), _molecule(nx.star_graph(5)),
              _molecule(nx.star_graph(3), atom=8), _molecule(nx.path_graph(3), bond=3),
              _molecule(nx.path_graph(2), atom=16), _molecule(nx.empty_graph(2)),
              _molecule(nx.empty_graph(0))]
    before = [g.copy() for g in graphs]
    report = molecular_failure_diagnostics(graphs, ['CC', None, None, None, None, 'C.C', None])
    assert report['num_graphs'] == 7
    assert report['primary_counts'] == {'raw_valid': 2, 'topology_infeasible': 1,
        'atom_degree_incompatible': 1, 'bond_valence_exceeded': 1,
        'outside_neutral_cnof_diagnostic_scope': 1, 'empty_topology': 1}
    assert report['overlapping_flags']['disconnected']['count'] == 1
    assert sum(report['atom_counts'].values()) == sum(g.number_of_nodes() for g in graphs)
    assert sum(report['bond_counts'].values()) == sum(g.number_of_edges() for g in graphs)
    assert all(nx.utils.graphs_equal(a, b) for a, b in zip(before, graphs))


def test_charged_atoms_are_not_misdiagnosed_as_neutral():
    graph = _molecule(nx.star_graph(4), atom=7)
    graph.nodes[0]['formal_charge'] = 1
    report = molecular_failure_diagnostics([graph], [None])
    assert not report['per_graph'][0]['neutral_cnof_diagnostics_applicable']
    assert report['per_graph'][0]['bond_valence_exceeded'] is None


def test_strict_raw_evaluator_smoke_with_invalid_graph_retained(evaluator, tmp_path):
    pytest.importorskip('rdkit')
    data = tmp_path/'datasets'/'toy'; data.mkdir(parents=True)
    good = _molecule(nx.path_graph(2)); bad = _molecule(nx.star_graph(5))
    for split in ['train', 'val', 'test']:
        (data/f'{split}.pkl').write_bytes(pickle.dumps([good, _molecule(nx.path_graph(3))]))
    generated = tmp_path/'generated.pkl'; generated.write_bytes(pickle.dumps([good, bad]))
    args = evaluator.build_parser().parse_args([
        '--generated-graphs', str(generated), '--dataset-root', str(data.parent), '--dataset', 'toy',
        '--reference-split', 'val', '--strict-raw-metrics', '--metric-molecule-source', 'raw_valid',
        '--nspdk-backend', 'proxy', '--skip-fcd', '--graphlet-mmd',
        '--graphlet-topology-filter', 'all', '--graphlet-k-min', '3', '--graphlet-k-max', '5',
    ])
    report = evaluator.evaluate(args)
    assert report['metrics']['validity_without_correction'] == .5
    assert report['metrics']['num_generated_graphs'] == 2
    assert report['protocol']['metric_population_count'] == 1
    assert report['protocol']['strict_raw_metrics'] is True
    assert report['protocol']['evaluation_correction_diagnostics_only'] is True
    assert report['protocol']['graphlet']['topology_filter'] == 'all'
    assert report['failure_diagnostics']['num_graphs'] == 2
    assert report['metrics']['fcd_generated_smiles_source'] == 'raw_valid'

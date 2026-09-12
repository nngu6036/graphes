"""Regression tests: the typed CLI must build and reuse a genuinely labeled vocabulary."""
from copy import deepcopy
from itertools import permutations
from math import comb

import networkx as nx
import numpy as np
import pytest
import torch

from grapher.rewiring_mlp.attributed.data import GraphletBasis
from grapher.rewiring_mlp.attributed.induced_graphlets import (
    extract_histogram, histogram_distance, metadata, fit_training_basis,
    AttributedInducedGraphletCounter, validate_model_graphlets,
)
from grapher.rewiring_mlp.generic.induced_graphlets import InducedGraphletSpec, extract_histogram as topo_histogram
from grapher.rewiring_mlp.attributed.joint_typed_edge_training import build_model
from grapher.rewiring_mlp.attributed.joint_typed_edge_data import EndpointStore


def graph(k):
    g=nx.path_graph(k)
    nx.set_node_attributes(g,6,'atomic_num')
    nx.set_edge_attributes(g,1,'bond_type')
    return g


def config(k):
    from test_joint_typed_edge import tiny_config
    cfg=tiny_config()
    cfg['structure_summary_prediction'].update(induced_graphlet_histogram=True,
        induced_graphlet_attributed=True, induced_graphlet_k=k, induced_graphlet_scope='all')
    cfg['attributed_predictor']['loss_weights']['induced_graphlet_histogram']=1.0
    return cfg


@pytest.mark.parametrize('k',[3,4,5])
def test_node_and_edge_labels_are_distinct_bins_but_node_ids_are_not(k):
    a=graph(k); b=a.copy(); b.nodes[0]['atomic_num']=8
    c=a.copy(); c.edges[0,1]['bond_type']=2
    d=nx.empty_graph(k); nx.set_node_attributes(d,6,'atomic_num')
    e=d.copy(); e.nodes[0]['atomic_num']=8
    basis=fit_training_basis(config(k),[a,b,c,d,e])
    assert basis.width==6  # five observed classes plus unseen-type bin
    assert metadata(basis)['attributed'] is True
    histograms=[extract_histogram(g,basis) for g in [a,b,c,d,e]]
    assert len({tuple(h) for h in histograms})==5
    assert np.array_equal(topo_histogram(a,InducedGraphletSpec(k)),topo_histogram(b,InducedGraphletSpec(k)))
    assert np.array_equal(topo_histogram(a,InducedGraphletSpec(k)),topo_histogram(c,InducedGraphletSpec(k)))
    for perm in list(permutations(range(k)))[:24]:
        x=nx.relabel_nodes(b,dict(enumerate(perm)))
        np.testing.assert_array_equal(extract_histogram(b,basis),extract_histogram(x,basis))
    unknown=b.copy(); unknown.nodes[1]['atomic_num']=8
    assert extract_histogram(unknown,basis)[-1]==1.0


def test_shared_atom_multiset_and_bond_multiset_do_not_erase_positions():
    a=graph(5); a.nodes[1]['atomic_num']=8
    b=graph(5); b.nodes[2]['atomic_num']=8
    c=graph(5); c.edges[0,1]['bond_type']=2
    d=graph(5); d.edges[1,2]['bond_type']=2
    basis=fit_training_basis(config(5),[a,b,c,d])
    assert basis.width==5
    assert histogram_distance(extract_histogram(a,basis),extract_histogram(b,basis),basis)==1
    assert histogram_distance(extract_histogram(c,basis),extract_histogram(d,basis),basis)==1


@pytest.mark.parametrize('k',[3,4,5])
def test_incremental_labeled_counts_match_recount(k):
    a=nx.cycle_graph(7); nx.set_node_attributes(a,6,'atomic_num'); a.nodes[0]['atomic_num']=8
    nx.set_edge_attributes(a,1,'bond_type'); a.edges[2,3]['bond_type']=2
    action=(((0,1),(3,4)),((0,3),(1,4)))
    b=a.copy(); b.remove_edges_from(action[0]); b.add_edges_from(action[1],bond_type=1)
    basis=fit_training_basis(config(k),[a,b])
    counter=AttributedInducedGraphletCounter(a,basis)
    np.testing.assert_array_equal(counter.candidate_histogram(b,action),extract_histogram(b,basis))
    counter.accept(b,action)
    np.testing.assert_array_equal(counter.histogram(),extract_histogram(b,basis))
    assert sum(counter.counts_by_size[str(k)].values())==comb(7,k)


def test_training_model_requires_fixed_basis_on_endpoints(tmp_path):
    cfg=config(3); a=graph(5); b=graph(5); b.nodes[2]['atomic_num']=8
    model=build_model(cfg,[a,b],torch.device('cpu'))
    assert model.induced_graphlet_basis is not None
    assert model.induced_graphlet_spec is None
    validate_model_graphlets(model,cfg)
    with pytest.raises(ValueError,match='basis'):
        EndpointStore([a],model.vectorizer,model.atom_types,cfg,seed=42)
    store=EndpointStore([a],model.vectorizer,model.atom_types,cfg,seed=42,
                        graphlet_basis=model.induced_graphlet_basis,cache_path=tmp_path/'cache.sqlite')
    np.testing.assert_array_equal(store[0]['induced_histogram'],extract_histogram(a,model.induced_graphlet_basis))
    store.close()
    wrong=deepcopy(cfg); wrong['structure_summary_prediction']['induced_graphlet_k']=4
    with pytest.raises(ValueError,match='catalogue'):
        validate_model_graphlets(model,wrong)
    legacy=deepcopy(cfg); legacy['structure_summary_prediction']['induced_graphlet_attributed']=False
    old=build_model(legacy,[a,b],torch.device('cpu'))
    assert old.induced_graphlet_basis is None
    with pytest.raises(ValueError,match='topology-only'):
        validate_model_graphlets(old,cfg)


@pytest.mark.parametrize('interval,expected_middle', [(10, True), (60, False), (0, False)])
def test_vocabulary_reports_progress_without_changing_bins(monkeypatch, capsys, interval, expected_middle):
    from grapher.rewiring_mlp.attributed import induced_graphlets

    cfg = config(3)
    graphs = [graph(5), graph(4), graph(2)]
    baseline = fit_training_basis(cfg, graphs)
    capsys.readouterr()
    cfg['attributed_predictor']['progress_interval_seconds'] = interval
    ticks = iter([0.0, 1.0, 12.0, 13.0])
    monkeypatch.setattr(induced_graphlets.time, 'perf_counter', lambda: next(ticks))
    basis = fit_training_basis(cfg, graphs)
    output = capsys.readouterr().out

    assert 'CPU preprocessing: induced_subsets=14' in output
    assert 'graphs=1/3 subsets=10/14' in output
    assert ('graphs=2/3 subsets=14/14' in output) == expected_middle
    assert 'graphs=3/3 subsets=14/14' in output
    assert 'elapsed=13.0s' in output and 'eta=0.0s' in output
    assert metadata(basis) == metadata(baseline)
    assert basis.to_dict() == baseline.to_dict()


def test_local_counts_share_exact_label_tokens_with_vocabulary():
    a = graph(7)
    a.nodes[0]['atomic_num'] = True
    a.nodes[1]['atomic_num'] = 1
    a.nodes[2]['atomic_num'] = 1.0
    a.nodes[3]['atomic_num'] = [6]
    a.edges[0, 1]['bond_type'] = None
    a.edges[3, 4]['bond_type'] = None
    action = (((0, 1), (3, 4)), ((0, 3), (1, 4)))
    b = a.copy()
    b.remove_edges_from(action[0])
    b.add_edges_from(action[1], bond_type=None)
    basis = fit_training_basis(config(3), [a, b])
    counter = AttributedInducedGraphletCounter(a, basis)
    np.testing.assert_array_equal(counter.candidate_histogram(b, action), extract_histogram(b, basis))
    counter.accept(b, action)
    np.testing.assert_array_equal(counter.histogram(), extract_histogram(b, basis))
    assert sum(counter.counts_by_size['3'].values()) == comb(7, 3)

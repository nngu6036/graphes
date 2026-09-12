from collections import Counter
from dataclasses import replace
import json

import networkx as nx
import numpy as np
import pytest

from grapher.models.dhvae_hh.degree_perturbation import METHODS, DegreePerturbationError, sum_preserving_round
from grapher.models.dhvae_hh.typed_degree_perturbation import (
    PerturbedEmpiricalTypedDegreeSampler, canonical_invariant, typed_key, typed_totals,
    typed_moments, typed_distance, align_typed_rows, typed_fingerprint,
)
from grapher.models.dhvae_hh.typed_constructor import construct_typed_graph
from grapher.rewiring_mlp.molecular.typed_invariants import (
    TypedInvariant, TypedDegreeSignature, typed_invariant_matches_graph, extract_typed_invariant,
)
from grapher.rewiring_mlp.attributed.typed_prior import build_typed_empirical_sampler, validate_strict_typed_refinement
from grapher.rewiring_mlp.attributed.spectral_graphlet_refiner import AttributedSpectralGraphletRefinerConfig


def inv(degrees, atoms=None, types=(1,2,3)):
    return TypedInvariant(tuple(TypedDegreeSignature(atom,tuple(row)) for atom,row in
        zip(atoms or [6]*len(degrees),degrees)),types)


def parents(mixed=True):
    ds=([4,4,2,2,2,2],[3,3,3,3,2,2],[3,3,3,2,2,1],[4,2,2,2,2,2])
    return [inv([(d,1 if mixed else 0,0) for d in degrees]) for degrees in ds]


def sampler(method='unit_transfer',bank=None,**settings):
    return PerturbedEmpiricalTypedDegreeSampler(bank or parents(),
        dict(method=method,probability=1.,failure_policy='keep_original',**settings),
        constructor_config={'max_restarts':1,'max_backtracks':1000},seed=42)


@pytest.mark.parametrize('method',METHODS)
def test_all_four_multichannel_feasible_changes(method):
    s=sampler(method)
    outputs=[s.perturb_parent(i) for i in range(4)]
    assert sum(row['sampling_diagnostics']['changed'] for row in outputs)>0
    for i,row in enumerate(outputs):
        target=TypedInvariant.from_dict(row['typed_invariant']);parent=s.invariants[i]
        assert typed_totals(target)==typed_totals(parent)
        assert [x.node_type for x in target.signatures]==[x.node_type for x in parent.signatures]
        assert target.num_nodes==parent.num_nodes
        assert all(v%2==0 for v in typed_totals(target))
        graph,_=construct_typed_graph(target,{'randomize_assignment':False},np.random.default_rng(43))
        assert typed_invariant_matches_graph(graph,target) and nx.is_connected(graph)
        assert target.edge_counts[2]>0 # Real multichannel example, not just a 1-type scalar wrapper.
        if method=='moment_preserving':assert typed_moments(parent)==typed_moments(target)
    report=s.report();json.dumps(report)
    assert report['all_preserve_node_categories'] and report['all_preserve_edge_type_counts']
    assert report['all_joint_realizations_verified']


@pytest.mark.parametrize('method',METHODS)
def test_heterogeneous_node_types_preserved(method):
    # Two O records and six C records; moment blocks can operate inside C.
    bank=[inv([(d,0,0) for d in ds]+[(1,0,0),(1,0,0)],atoms=[6]*6+[8,8])
          for ds in ([4,4,2,2,2,2],[3,3,3,3,2,2],[3,3,3,2,2,1],[4,2,2,2,2,2])]
    s=sampler(method,bank)
    for i in range(4):
        row=s.perturb_parent(i);d=row['sampling_diagnostics'];target=TypedInvariant.from_dict(row['typed_invariant'])
        assert [x.node_type for x in target.signatures]==[x.node_type for x in s.invariants[i].signatures]
        assert Counter(x.node_type for x in target.signatures)==Counter({6:6,8:2})
        assert d['preserved_edge_type_counts']


def test_joint_collision_not_accepted_by_independent_graphicality():
    bad=inv(((2,1),(1,3),(1,0),(0,2),(0,2)),types=(1,2))
    assert nx.is_graphical(bad.degree_sequence)
    for r in range(2):assert nx.is_graphical([s.edge_degrees[r] for s in bad.signatures])
    s=PerturbedEmpiricalTypedDegreeSampler([bad],{'probability':0},constructor_config={'max_restarts':0,'max_backtracks':20000})
    with pytest.raises(DegreePerturbationError,match='no_typed_realization'):s.sample()
    assert not s.report()['all_joint_realizations_verified']


@pytest.mark.parametrize('value',[1.5,True,np.float64(1.0)])
def test_no_silent_integer_coercion(value):
    with pytest.raises(ValueError,match='integers'):
        sampler(bank=[inv([(value,0,0),(1,0,0)])])


def test_zero_columns_allowed_but_zero_aggregate_not_connected():
    g=inv([(1,0,0),(1,0,0)])
    s=sampler(bank=[g]);assert s.sample()['typed_invariant']['signatures']
    with pytest.raises(ValueError,match='No training'):
        sampler(bank=[inv([(0,0,0),(0,0,0)])])


def test_moments_allow_zero_entries():
    # Full-block scalar replacement (0,2,2,2)<->(1,1,1,3) in active column,
    # with one incidence of a second type on every node.
    bank=[inv([(d,1,0) for d in [0,2,2,2,1,1]])]
    s=sampler('moment_preserving',bank)
    row=s.sample();assert row['sampling_diagnostics']['changed']
    assert typed_moments(TypedInvariant.from_dict(row['typed_invariant']))==typed_moments(s.invariants[0])


def test_label_aware_matching_does_not_sort_columns_independently():
    a=inv([(2,0,0),(0,2,0),(1,1,0),(1,1,0)])
    b=inv([(2,2,0),(0,0,0),(1,1,0),(1,1,0)])
    # Same marginal columns, but DIFFERENT joint signatures.
    assert all(sorted(x.edge_degrees[r] for x in a.signatures)==sorted(x.edge_degrees[r] for x in b.signatures) for r in range(3))
    assert typed_key(a)!=typed_key(b) and typed_distance(a,b)>0
    c=replace(a,signatures=tuple(reversed(a.signatures)))
    assert typed_distance(a,c)==0 and typed_fingerprint([a])==typed_fingerprint([c])
    hetero=inv([(1,0,0),(0,1,0)],atoms=[6,8])
    switched=inv([(0,1,0),(1,0,0)],atoms=[6,8])
    assert typed_distance(hetero,switched)==2


def test_rounding_preserves_indices_and_per_type_totals():
    rng=np.random.default_rng(11)
    assert sum_preserving_round([1,3,2],6,rng,sort_result=False)==(1,3,2)
    assert sum_preserving_round([1,3,2],6,rng)==(3,2,1)
    x=np.array([[0.5,1.5],[1.5,0.5],[1.,1.]])
    for _ in range(100):
        y=np.array([sum_preserving_round(x[:,r],3,rng,sort_result=False) for r in range(2)]).T
        assert (y.sum(0)==3).all()
        assert (y>=np.floor(x)).all() and (y<=np.ceil(x)).all()


@pytest.mark.parametrize('method',METHODS)
def test_checkpoint_vocabulary_mask_and_caps(method):
    bank=parents(False)
    allowed=set(s for p in bank for s in p.signatures)
    s=PerturbedEmpiricalTypedDegreeSampler(bank,{'method':method,'probability':1.,'failure_policy':'keep_original'},
       allowed_signatures=allowed,constructor_config={'max_ordinary_degree':4,'max_weighted_valence':{6:4},'max_restarts':1,'max_backtracks':1000})
    for i in range(4):
        target=TypedInvariant.from_dict(s.perturb_parent(i)['typed_invariant'])
        assert all(x in allowed and x.degree<=4 and x.weighted_degree(target.edge_types)<=4 for x in target.signatures)
    assert s.report()['checkpoint_signature_mask_enabled']


def test_endpoint_compatibility_on_all_realizations():
    # K_2,3 degrees; perturbing must remain bipartite w.r.t. node types.
    bank=[inv([(3,0,0)]*2+[(2,0,0)]*3,atoms=[6]*2+[7]*3)]
    s=PerturbedEmpiricalTypedDegreeSampler(bank,{'method':'edge_relocation','probability':1.,'failure_policy':'keep_original'},
       endpoint_compatible=lambda a,b,r:a!=b)
    for _ in range(6):
        target=TypedInvariant.from_dict(s.sample()['typed_invariant'])
        g,_=construct_typed_graph(target,{'randomize_assignment':False},endpoint_compatible=lambda a,b,r:a!=b)
        assert all(g.nodes[u]['atomic_num']!=g.nodes[v]['atomic_num'] for u,v in g.edges)


def test_rdkit_validation():
    from grapher.rewiring_mlp.molecular.graph_io import is_valid_molecular_graph
    from grapher.rewiring_mlp.attributed.joint_typed_edge_data import graph_record,graph_from_record
    calls=[]
    def validator(g):
        result=is_valid_molecular_graph(graph_from_record(graph_record(g)));calls.append(result);return result
    s=PerturbedEmpiricalTypedDegreeSampler(parents(False),{'method':'unit_transfer','probability':1.,'failure_policy':'keep_original'},
        constructor_config={'max_ordinary_degree':4,'max_weighted_valence':{6:4}},graph_validator=validator)
    for i in range(4):s.perturb_parent(i)
    assert calls and s.report()['domain_validator_enabled']
    assert s.report()['all_joint_realizations_verified']


def test_domain_failure_no_silent_parent_redraw():
    s=PerturbedEmpiricalTypedDegreeSampler(parents(False),{'method':'unit_transfer','probability':1.,'failure_policy':'keep_original','max_attempts':3},
                                         graph_validator=lambda g:False)
    with pytest.raises(DegreePerturbationError,match='domain_validator'):s.perturb_parent(0)
    assert len(s.records)==1 and s.records[0]['parent_train_index']==0


@pytest.mark.parametrize('method',METHODS)
def test_determinism_independent_parents_and_mixture(method):
    a=sampler(method,max_attempts=1);b=sampler(method,max_attempts=48)
    for _ in range(8):a.sample();b.sample()
    assert a.report()['parent_typed_fingerprint']==b.report()['parent_typed_fingerprint']
    c=sampler(method,max_attempts=48)
    for _ in range(8):c.sample()
    assert b.report()==c.report()


def test_transactional_failure_and_identity_reporting():
    s=sampler('unit_transfer',steps=2,max_distance=.1)
    out=s.sample();row=out['sampling_diagnostics']
    assert not row['changed'] and row['fallback_used'] and row['accepted_steps']==0
    assert row['typed_invariant']==row['parent_typed_invariant']
    e=PerturbedEmpiricalTypedDegreeSampler(parents(),{'method':'unit_transfer','probability':1.,'max_distance':.1,'failure_policy':'error'})
    with pytest.raises(DegreePerturbationError):e.sample()
    assert len(e.records)==1 and not e.records[0]['fallback_used']


def test_parent_exclusions_recorded_and_shared_across_methods():
    bank=parents(False)+[inv([(5,0,0),(1,0,0),(1,0,0),(1,0,0),(1,0,0),(1,0,0)])]
    for method in METHODS:
        s=PerturbedEmpiricalTypedDegreeSampler(bank,dict(method=method),constructor_config={'max_ordinary_degree':4})
        assert len(s.invariants)==4
        assert s.report()['parent_exclusions'][0]['train_index']==4


def test_inconsistent_schema_rejected():
    with pytest.raises(ValueError,match='vocabulary'):
        sampler(bank=[parents()[0],replace(parents()[1],edge_types=(3,2,1))])


def test_no_partner_reports_clear_identity():
    s=sampler('interpolation',[parents()[0]])
    row=s.sample()['sampling_diagnostics']
    assert not row['changed'] and row['failure_reason'].startswith('no_distinct_training_partner')


def test_factory_strict_legacy_refiner_guard():
    cfg=AttributedSpectralGraphletRefinerConfig.from_dict({})
    with pytest.raises(ValueError,match='Cross-type'):validate_strict_typed_refinement(cfg)
    cfg=replace(cfg,require_same_edge_type_pair=True,preserve_typed_degree=True,preserve_weighted_valence=True)
    validate_strict_typed_refinement(cfg)


def test_invariants_only_no_adjacency_retained_and_input_unchanged():
    bank=parents(False);graphs=[construct_typed_graph(p)[0] for p in bank]
    before=[g.copy() for g in graphs]
    s=PerturbedEmpiricalTypedDegreeSampler.fit(graphs,dict(method='edge_relocation',probability=1.,failure_policy='keep_original'),edge_types=(1,2,3))
    for _ in range(5):s.sample()
    assert not any(isinstance(v,nx.Graph) for v in vars(s).values())
    assert all(nx.utils.graphs_equal(g,h) for g,h in zip(graphs,before))


@pytest.mark.parametrize('method',METHODS)
def test_all_three_nonempty_type_channels(method):
    bank=[inv([(d,1,1) for d in ds]+[(1,1,1),(1,1,1)])
          for ds in ([4,4,2,2,2,2],[3,3,3,3,2,2],[3,3,3,2,2,1],[4,2,2,2,2,2])]
    s=sampler(method,bank,max_attempts=64)
    for i in range(4):
        result=TypedInvariant.from_dict(s.perturb_parent(i)['typed_invariant'])
        assert typed_totals(result)==typed_totals(s.invariants[i])
        assert all(value>0 for value in typed_totals(result))
        graph,_=construct_typed_graph(result,{'randomize_assignment':False})
        assert set(nx.get_edge_attributes(graph,'bond_type').values())=={1,2,3}
    assert s.report()['num_changed']>0


def test_invalid_joint_candidate_is_rejected_before_acceptance(monkeypatch):
    graph=nx.cycle_graph(5);graph.add_edge(0,2);nx.set_node_attributes(graph,6,'atomic_num')
    for i,(u,v) in enumerate(graph.edges):graph[u][v]['bond_type']=1 if i<2 else 2
    good=extract_typed_invariant(graph,edge_types=(1,2))
    bad=inv(((2,1),(1,3),(1,0),(0,2),(0,2)),types=(1,2))
    assert typed_totals(good)==typed_totals(bad)
    s=sampler('unit_transfer',[good],max_distance=None)
    monkeypatch.setattr(s,'_proposals',lambda current,rng:iter([(bad,None,{})]))
    output=s.sample()['sampling_diagnostics']
    assert output['fallback_used'] and not output['changed']
    assert output['proposal_rejections'].get('joint_constructor_no_typed_realization',0)>0


def test_wrong_source_does_not_silently_ignore_perturbation_config():
    from grapher.rewiring_mlp.attributed.typed_prior import uses_typed_empirical_kernel
    with pytest.raises(ValueError,match='silently ignored'):
        uses_typed_empirical_kernel({'generation':{'invariant_source':'learned','degree_perturbation':{'method':'unit_transfer'}}})

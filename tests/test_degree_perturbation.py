from __future__ import annotations

from collections import defaultdict
from itertools import combinations_with_replacement
import json

import networkx as nx
import numpy as np
import pytest

from grapher.models.dhvae_hh.degree_perturbation import (
    METHODS, DegreePerturbationConfig, DegreePerturbationError,
    PerturbedEmpiricalDegreeSampler, _moment_blocks, canonical_degrees,
    connected_feasible, degree_distance, sequence_fingerprint, sum_preserving_round,
)
from grapher.models.dhvae_hh.degree_sampler import EmpiricalDegreeSampler
from grapher.models.dhvae_hh.havel_hakimi import construct_coarse_graph, assert_constructor_validity

PARENTS = [
    [4, 3, 3, 2, 2, 2], [3, 3, 3, 3, 2, 2], [4, 4, 2, 2, 2, 2],
    [3, 3, 3, 2, 2, 1], [4, 2, 2, 2, 2, 2],
]


@pytest.fixture(autouse=True)
def preserve_global_numpy_rng():
    state = np.random.get_state()
    yield
    np.random.set_state(state)


def sampler(method, parents=PARENTS, **kwargs):
    return PerturbedEmpiricalDegreeSampler(parents,
        dict(method=method, probability=1.0, failure_policy="keep_original", **kwargs), seed=42)


@pytest.mark.parametrize("method", METHODS)
def test_seeded_invariants_construction_and_novelty_accounting(method):
    s = sampler(method)
    for i in range(30):
        row = s.sample()
        d = row["degree_sequence"]
        rec = row["sampling_diagnostics"]
        parent = rec["parent_degree_sequence"]
        assert len(d) == len(parent) and sum(d) == sum(parent)
        assert connected_feasible(d)
        assert rec["changed"] == (d != parent)
        assert rec["novel_vs_training"] == (tuple(d) not in s.training_set)
        assert rec["distance_half_l1"] == degree_distance(d, parent)
        if method == "moment_preserving":
            assert sum(x*x for x in d) == sum(x*x for x in parent)
        graph = construct_coarse_graph(row, {"ensure_connected": True}, np.random.default_rng(i))
        assert_constructor_validity(graph, row)
    r = s.report()
    assert r["all_preserve_n_m"] and r["all_connected_feasible"]
    assert r["num_changed"] > 0
    assert r["num_samples"] == 30
    json.dumps(r, allow_nan=False)


@pytest.mark.parametrize("method", METHODS)
def test_reproducible_streams(method):
    a, b = sampler(method), sampler(method)
    for _ in range(10):
        np.random.seed(988)  # ambient RNG must not affect the sampler
        ar = a.sample()["sampling_diagnostics"]
        np.random.seed(55)
        assert ar == b.sample()["sampling_diagnostics"]
    assert a.report() == b.report()


def test_parents_and_mixture_flags_identical_despite_different_rejections():
    reports = []
    for method in METHODS:
        s = PerturbedEmpiricalDegreeSampler(PARENTS,
            dict(method=method, probability=.4, failure_policy="keep_original"), seed=43)
        rng = np.random.default_rng(np.random.SeedSequence(43, spawn_key=(3,)))
        for _ in range(40):
            s.sample(rng)
        reports.append(s.report())
    assert len({r["parent_degree_fingerprint"] for r in reports}) == 1
    assert len({tuple(x["requested"] for x in r["records"]) for r in reports}) == 1


@pytest.mark.parametrize("method", METHODS)
def test_zero_probability_exactly_matches_empirical(method):
    s = PerturbedEmpiricalDegreeSampler(PARENTS, dict(method=method, probability=0), seed=123)
    empirical = EmpiricalDegreeSampler(PARENTS)
    a, b = np.random.default_rng(9), np.random.default_rng(9)
    for _ in range(20):
        row = s.sample(a)
        assert row["degree_sequence"] == empirical.sample(b)["degree_sequence"]
        assert not row["sampling_diagnostics"]["requested"]
        assert not row["sampling_diagnostics"]["fallback_used"]
    assert a.random() == b.random()  # exactly one parent integer consumed each time


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("parent", [[0], [1, 1], [3, 3, 3, 3]])
def test_impossible_requests_explicit_identity_or_strict_error(method, parent):
    s = sampler(method, [parent])
    row = s.sample()
    assert row["degree_sequence"] == parent
    assert row["sampling_diagnostics"]["requested"]
    assert row["sampling_diagnostics"]["fallback_used"]
    assert row["sampling_diagnostics"]["failure_reason"]
    assert s.report()["num_identity_fallbacks"] == 1
    strict = PerturbedEmpiricalDegreeSampler([parent], dict(method=method, probability=1, failure_policy="error"))
    with pytest.raises(DegreePerturbationError, match="No parent redraw"):
        strict.sample()
    assert len(strict.records) == 1


def test_permutations_are_not_degree_novelty():
    assert canonical_degrees([2, 3, 2, 3]) == canonical_degrees([3, 2, 3, 2])
    assert degree_distance([2, 3, 2, 3], [3, 2, 3, 2]) == 0
    s = sampler("unit_transfer", [[2, 2, 1, 1]])
    row = s.sample()
    assert row["degree_sequence"] == [3, 1, 1, 1]


def test_unit_transfer_can_balance_and_concentrate():
    s = sampler("unit_transfer", [[4, 3, 3, 2, 2, 2]])
    q = sum(d*d for d in s.degree_sequences[0])
    moments = [sum(d*d for d in s.sample()["degree_sequence"]) for _ in range(100)]
    assert min(moments) < q < max(moments)


def test_moment_blocks_match_bruteforce():
    groups = defaultdict(list)
    for block in combinations_with_replacement(range(1, 6), 4):
        groups[(sum(block), sum(d*d for d in block))].append(block)
    for (total, squares), expected in groups.items():
        assert list(_moment_blocks(4, total, squares, 5, 20000)) == expected
    assert (2, 2, 2, 4) in _moment_blocks(4, 10, 28, 5, 20000)


def test_second_moment_impossible_for_minimum_variance_sequence():
    s = sampler("moment_preserving", [[3, 3, 3, 3, 2, 2]])
    row = s.sample()["sampling_diagnostics"]
    assert not row["changed"]
    assert row["failure_reason"] == "no_valid_moment_preserving_block"


def test_block_transaction_rolls_back_incomplete_requested_steps():
    s = sampler("moment_preserving", [[3, 3, 3, 2, 2, 1]], steps=2)
    rec = s.sample()["sampling_diagnostics"]
    assert rec["rolled_back_steps"] == 1
    assert rec["accepted_steps"] == 0
    assert rec["degree_sequence"] == rec["parent_degree_sequence"]


def test_relocation_does_not_store_or_reuse_original_adjacency():
    a = nx.cycle_graph(6)
    b = nx.Graph([(0, 1), (1, 3), (3, 2), (2, 4), (4, 5), (5, 0)])
    cfg = dict(method="edge_relocation", probability=1, failure_policy="keep_original")
    sa = PerturbedEmpiricalDegreeSampler.fit_from_graphs([a], cfg, seed=5)
    sb = PerturbedEmpiricalDegreeSampler.fit_from_graphs([b], cfg, seed=5)
    assert sa.__dict__["degree_sequences"] == sb.__dict__["degree_sequences"]
    a.clear()  # mutating the training graph after fit must not affect sampling
    assert sa.sample()["sampling_diagnostics"] == sb.sample()["sampling_diagnostics"]
    rec = sa.records[0]
    assert rec["changed"] and rec["operations"][0]["witness_connected"]


def test_interpolation_uses_distinct_same_n_m_training_partners():
    parents = [[4, 4, 2, 2, 2, 2], [3, 3, 3, 3, 2, 2], [3, 3, 3, 2, 2, 1], [2]*5]
    s = sampler("interpolation", parents)
    rec = s.perturb_parent(0)["sampling_diagnostics"]
    assert rec["changed"]
    assert rec["degree_sequence"] == [4, 3, 3, 2, 2, 2]
    assert rec["novel_vs_training"]
    assert rec["operations"][0]["partner_degree_sequence"] == parents[1]


def test_interpolation_reports_absent_partner_without_redraw():
    s = sampler("interpolation", [[2]*6, [1, 1], [3, 3, 2, 2, 2]])
    rec = s.perturb_parent(0)["sampling_diagnostics"]
    assert rec["failure_reason"] == "no_distinct_training_partner_same_n_m"
    assert rec["parent_train_index"] == 0 and len(s.records) == 1


def test_interpolation_excludes_copying_other_parent():
    s = sampler("interpolation", [[3, 3, 2, 2, 2, 2], [3, 3, 3, 2, 2, 1]])
    rec = s.perturb_parent(0)["sampling_diagnostics"]
    assert rec["failure_reason"] == "no_valid_nonparent_interpolation_within_budget"
    assert not rec["changed"]


def test_dependent_rounding_has_exact_sum_and_only_floor_ceil_values():
    x = [3.3, 3.3, 2.7, 2.7, 2, 2]
    rng = np.random.default_rng(42)
    for _ in range(200):
        d = sum_preserving_round(x, 16, rng)
        assert sum(d) == 16 and len(d) == 6
        assert all(v in {2, 3, 4} for v in d)
    with pytest.raises(ValueError, match="sum"):
        sum_preserving_round(x, 17, rng)


@pytest.mark.parametrize("method", METHODS)
def test_support_limit_and_distance_cap(method):
    s = PerturbedEmpiricalDegreeSampler([[4, 3, 3, 2, 2, 2], [3, 3, 3, 3, 2, 2]],
        dict(method=method, probability=1, failure_policy="keep_original", max_distance=1), support_max_degree=4)
    for _ in range(12):
        row = s.sample()
        assert max(row["degree_sequence"]) <= 4
        assert row["sampling_diagnostics"]["distance_half_l1"] <= 1
    with pytest.raises(ValueError, match="never clipped"):
        PerturbedEmpiricalDegreeSampler([[4, 3, 3, 2, 2, 2]], support_max_degree=3)


@pytest.mark.parametrize("bad", [dict(method="typo"), dict(probability=1.1), dict(probability=float('nan')),
    dict(steps=0), dict(steps=1.5), dict(max_attempts=0), dict(max_attempts=True), dict(block_size=2),
    dict(block_size=4.5), dict(failure_policy="empirical_nearest"), dict(max_degree=0),
    dict(max_distance=0), dict(interpolation_neighbors=0), dict(interpolation_alpha=0),
    dict(interpolation_max_parent_distance=-2), dict(require_novel="false")])
def test_invalid_config_rejected(bad):
    with pytest.raises((ValueError, TypeError)):
        DegreePerturbationConfig.from_dict(bad)


def test_unknown_config_not_ignored():
    with pytest.raises(ValueError, match="Unknown"):
        DegreePerturbationConfig.from_dict({"probabilty": 0.25})


@pytest.mark.parametrize("parent", [[], [0, 0], [3, 1, 1], [2.0, 2, 2], [True], [-1, 1]])
def test_invalid_parent_not_repaired(parent):
    with pytest.raises(ValueError):
        PerturbedEmpiricalDegreeSampler([parent])


def test_graphical_but_not_connected_feasible_is_rejected():
    assert nx.is_graphical([1, 1, 1, 1])
    assert not connected_feasible([1, 1, 1, 1])
    assert connected_feasible([0])
    assert connected_feasible([1, 1])


def test_training_fingerprint_retains_empirical_multiplicities():
    assert sequence_fingerprint([[1, 1], [2, 2, 2]]) != sequence_fingerprint([[1, 1], [1, 1], [2, 2, 2]])


@pytest.mark.parametrize("method", METHODS)
def test_exhaustive_small_graph_fixture_sanity(method):
    # Atlas supplies diverse connected simple graphs, including impossible cases.
    graphs = [g for g in nx.graph_atlas_g() if 1 <= len(g) <= 6 and nx.is_connected(g)][::3]
    s = PerturbedEmpiricalDegreeSampler.fit_from_graphs(graphs,
        dict(method=method, probability=1, failure_policy="keep_original", max_attempts=32))
    for i in range(len(graphs)):
        row = s.perturb_parent(i)
        assert connected_feasible(row["degree_sequence"])
        rec = row["sampling_diagnostics"]
        assert rec["preserved_n_m"]
        if method == "moment_preserving":
            assert rec["preserved_second_moment"]

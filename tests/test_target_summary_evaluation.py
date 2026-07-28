from __future__ import annotations

import numpy as np

from grapher.evaluation.target_summaries import (
    active_component_names,
    conditional_sample_metrics,
    degree_condition_match_rate,
    evaluate_summary_sets,
    fit_mmd_bandwidths,
    paired_summary_errors,
)


class _FakeVectorizer:
    clustering_bins = 2
    spectral_bins = 2
    motif_dim = 0
    orbit_dim = 0
    graphlet_dim = 2
    motif_scale = []
    orbit_scale = []
    scalar_scale = [1.0, 1.0]

    @staticmethod
    def graphlet_to_vector(summary):
        return np.asarray(summary["graphlet"], dtype=np.float64)

    @staticmethod
    def graphlet_slices():
        return {"3": slice(0, 2)}


def _summary(value: float, degree=(2, 2, 1, 1)):
    return {
        "num_nodes": 4,
        "num_edges": 3,
        "degree_sequence": list(degree),
        "degree_hist": [0.0, 0.5, 0.5],
        "clustering_hist": [1.0 - value, value],
        "spectral_hist": [0.5, 0.5],
        "motif_proxy": [],
        "orbit_count": [],
        "graphlet": [1.0 - value, value],
        "triangle_count_norm": value,
    }


def test_identical_summary_sets_have_zero_error_and_mmd():
    vectorizer = _FakeVectorizer()
    reference = [_summary(0.1), _summary(0.8)]
    components = active_component_names(
        vectorizer,
        {
            "clustering": 1.0,
            "spectral": 0.0,
            "graphlet": 2.0,
            "scalar": 0.25,
        },
    )
    bandwidths = fit_mmd_bandwidths(
        reference,
        reference,
        vectorizer,
        component_names=components,
    )
    metrics = evaluate_summary_sets(
        reference,
        reference,
        vectorizer,
        component_names=components,
        bandwidths=bandwidths,
    )
    paired = paired_summary_errors(
        reference,
        reference,
        vectorizer,
        component_names=components,
    )
    assert abs(metrics["structural_mmd"]) < 1.0e-12
    assert paired["structural_rmse"] == 0.0
    assert paired["structural_mae"] == 0.0


def test_conditional_energy_rewards_accuracy_and_reports_diversity():
    vectorizer = _FakeVectorizer()
    target = [_summary(0.5)]
    components = ["clustering", "graphlet", "triangle"]
    exact = conditional_sample_metrics(
        target,
        [[_summary(0.5), _summary(0.5)]],
        vectorizer,
        component_names=components,
    )
    spread = conditional_sample_metrics(
        target,
        [[_summary(0.2), _summary(0.8)]],
        vectorizer,
        component_names=components,
    )
    assert exact["conditional_energy_score"] == 0.0
    assert exact["within_condition_diversity"] == 0.0
    assert spread["within_condition_diversity"] > 0.0


def test_degree_condition_match_checks_all_hard_invariants():
    target = [_summary(0.5)]
    assert degree_condition_match_rate(target, [_summary(0.8)]) == 1.0
    assert (
        degree_condition_match_rate(
            target,
            [_summary(0.8, degree=(3, 1, 1, 1))],
        )
        == 0.0
    )

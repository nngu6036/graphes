from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from grapher.evaluation.studies import (
    EXACT_REACHABILITY_MAX_NODES,
    aggregate_pipeline_diagnostics,
    aggregate_three_seed_results,
    assess_stability_and_synthesizability,
    constrained_rewiring_reachability,
    constructor_bias_comparison,
    generation_error_decomposition,
    hierarchical_graph_summary,
    local_summary_collision_diagnostics,
    molecular_validity_limitation_audit,
    paired_ablation_comparison,
    prediction_consistency_residuals,
    project_predictions_to_feasible_target,
    quality_cost_pareto_summary,
)


def test_generation_error_decomposition_telescopes_and_respects_direction() -> None:
    result = generation_error_decomposition(
        {
            "oracle": {"mmd": 0.1, "validity": 1.0},
            "constructor": {"mmd": 0.3, "validity": 0.9},
            "refined": {"mmd": 0.2, "validity": 0.95},
        },
        metric_higher_is_better={"validity": True},
    )

    assert result["total_delta"]["mmd"] == pytest.approx(0.1)
    assert result["total_error_change"]["validity"] == pytest.approx(0.05)
    assert result["transitions"][1]["improvement"]["mmd"] == pytest.approx(0.1)
    assert max(result["telescoping_error"].values()) < 1.0e-12


def test_constructor_bias_includes_pairwise_and_rejection_diagnostics() -> None:
    result = constructor_bias_comparison(
        {
            "canonical": [nx.path_graph(4), nx.path_graph(5)],
            "randomized": [nx.cycle_graph(4), nx.cycle_graph(5)],
        },
        reference=[nx.path_graph(4)],
        diagnostics={
            "canonical": {"attempts": 2, "accepted": 2, "restarts": 0},
            "randomized": {
                "attempts": 4,
                "accepted": 2,
                "rejected": 2,
                "restarts": 1,
                "rejection_reasons": {"disconnected": 2},
            },
        },
    )

    assert len(result["pairwise"]) == 1
    randomized = result["constructors"]["randomized"]["construction_diagnostics"]
    assert randomized["acceptance_rate"] == 0.5
    assert randomized["restarts_per_accepted"] == 0.5
    assert randomized["rejection_reasons"] == {"disconnected": 2.0}


def test_exact_rewiring_reachability_and_declared_large_graph_boundary() -> None:
    initial = nx.cycle_graph(4)
    target = nx.Graph([(0, 2), (2, 1), (1, 3), (3, 0)])
    unconstrained = constrained_rewiring_reachability(initial, target=target)
    local_only = constrained_rewiring_reachability(
        initial,
        target=target,
        locality_radius=1,
    )

    assert unconstrained["reachable_count"] == 3
    assert unconstrained["coverage"] == 1.0
    assert unconstrained["target_reachable"] is True
    assert local_only["reachable_count"] == 1
    assert local_only["coverage"] == pytest.approx(1 / 3)
    assert local_only["target_reachable"] is False

    with pytest.raises(NotImplementedError, match="empirical multi-start"):
        constrained_rewiring_reachability(
            nx.path_graph(EXACT_REACHABILITY_MAX_NODES + 1)
        )


def test_prediction_consistency_checks_fixed_invariant_and_graphlet_edges() -> None:
    pair = np.zeros((3, 3), dtype=np.float64)
    pair[0, 1] = pair[1, 0] = 1.0
    pair[1, 2] = pair[2, 1] = 1.0
    result = prediction_consistency_residuals(
        pair,
        [1, 2, 1],
        graphlet_histograms={3: [1.0, 0.0]},
        graphlet_edge_counts={3: [2.0, 3.0]},
    )

    assert result["invariant_l1"] == 0.0
    assert result["symmetry_l1"] == 0.0
    assert result["graphlets"]["3"]["pair_graphlet_edge_residual"] == pytest.approx(0.0)


def test_local_summary_collisions_quantify_unresolved_targets() -> None:
    result = local_summary_collision_diagnostics(
        [nx.path_graph(4), nx.star_graph(3), nx.cycle_graph(4)],
        [[1.0], [1.0], [2.0]],
        targets=[0.0, 1.0, 2.0],
    )

    assert result["collision_group_count"] == 1
    assert result["nonisomorphic_collision_pair_count"] == 1
    assert result["target_sufficiency"]["unresolved_within_summary_sse"] == 0.5
    assert result["target_sufficiency"]["explained_variation_fraction"] < 1.0


def test_molecular_limit_audit_never_claims_unsupported_oracles() -> None:
    graph = nx.Graph()
    graph.add_node(0, atom_type="C")
    graph.add_node(1, atom_type="O")
    graph.add_edge(0, 1, bond_type="SINGLE")
    result = molecular_validity_limitation_audit(
        [graph], allowed_weighted_valence={"C": 4.0, "O": 2.0}
    )

    assert result["simple_rate"] == 1.0
    assert result["weighted_valence_valid_rate"] == 1.0
    assert result["formal_charge_complete_rate"] == 0.0
    assert result["posthoc_repair_applied"] is False
    assert "chemical stability" in result["not_guaranteed"]


@pytest.mark.parametrize(
    "placeholder",
    [
        project_predictions_to_feasible_target,
        hierarchical_graph_summary,
        assess_stability_and_synthesizability,
    ],
)
def test_undefined_scientific_methods_are_explicit_placeholders(placeholder) -> None:
    with pytest.raises(NotImplementedError):
        placeholder()


def test_fixed_three_seed_aggregation_validates_protocol() -> None:
    result = aggregate_three_seed_results(
        {
            42: {"quality": 1.0, "nested": {"yield": 0.8}},
            43: {"quality": 2.0, "nested": {"yield": 0.9}},
            44: {"quality": 3.0, "nested": {"yield": 1.0}},
        }
    )

    assert result["aggregate"]["quality"]["mean"] == 2.0
    assert result["aggregate"]["quality"]["std"] == 1.0
    assert result["aggregate"]["nested.yield"]["mean"] == pytest.approx(0.9)
    with pytest.raises(ValueError, match="missing"):
        aggregate_three_seed_results({42: {"x": 1.0}, 43: {"x": 2.0}})


def _pipeline_record(offset: float) -> dict[str, object]:
    return {
        "predictor_nll": 1.0 + offset,
        "predictor_macro_f1": 0.8,
        "graphlet_error": 0.2,
        "consistency_residual": 0.1,
        "invariant_feasible": True,
        "constructor_success": True,
        "candidate_proposals": 10,
        "candidate_passes": 5,
        "accepted_swaps": 2,
        "stopped": 1,
        "stop_opportunities": 1,
        "rejection_reasons": {"locality": 5},
        "runtime_seconds": 3.0,
        "generation_attempts": 1,
        "generation_successes": 1,
        "fallback_used": False,
    }


def test_pipeline_diagnostics_aggregate_counts_and_refuse_fallback() -> None:
    result = aggregate_pipeline_diagnostics([_pipeline_record(0), _pipeline_record(1)])

    assert result["metrics"]["candidate_pass_rate"] == 0.5
    assert result["metrics"]["proposals_per_accepted_swap"] == 5.0
    assert result["metrics"]["end_to_end_yield"] == 1.0
    assert result["rejection_reasons"] == {"locality": 10.0}

    bad = _pipeline_record(0)
    bad["fallback_used"] = True
    with pytest.raises(ValueError, match="fallback"):
        aggregate_pipeline_diagnostics([bad])


def test_topology_pipeline_diagnostics_do_not_require_pair_metrics() -> None:
    record = {
        "pipeline_mode": "topology",
        "graphlet_error": 0.2,
        "invariant_feasible": True,
        "constructor_success": True,
        "candidate_proposals": 10,
        "candidate_passes": 5,
        "candidate_pass_rate": 0.5,
        "accepted_swaps": 2,
        "proposals_per_accepted_swap": 5.0,
        "stopped": 1,
        "stop_opportunities": 1,
        "stop_rate": 1.0,
        "rejection_reasons": {"connectivity": 5},
        "runtime_seconds": 3.0,
        "generation_attempts": 1,
        "generation_successes": 1,
        "end_to_end_yield": 1.0,
        "fallback_used": False,
    }

    result = aggregate_pipeline_diagnostics([record])

    assert result["pipeline_mode"] == "topology"
    assert "predictor_nll" not in result["metrics"]
    assert "predictor_macro_f1" not in result["metrics"]
    assert result["metrics"]["graphlet_error"]["mean"] == pytest.approx(0.2)


def test_zero_accepted_swaps_leave_proposal_ratio_undefined() -> None:
    record = {
        "pipeline_mode": "topology",
        "graphlet_error": 0.2,
        "invariant_feasible": True,
        "constructor_success": True,
        "candidate_proposals": 10,
        "candidate_passes": 5,
        "accepted_swaps": 0,
        "stopped": 1,
        "stop_opportunities": 1,
        "rejection_reasons": {"no_gain": 5},
        "runtime_seconds": 3.0,
        "generation_attempts": 2,
        "generation_successes": 1,
        "fallback_used": False,
    }

    result = aggregate_pipeline_diagnostics([record])

    assert result["metrics"]["proposals_per_accepted_swap"] is None
    assert result["metrics"]["end_to_end_yield"] == pytest.approx(0.5)


def test_paired_ablation_validates_sources_and_reports_improvement() -> None:
    control = [
        {
            "sample_id": "a",
            "seed": 42,
            "invariant_id": "i1",
            "initial_graph_id": "g1",
            "mmd": 0.4,
        },
        {
            "sample_id": "b",
            "seed": 42,
            "invariant_id": "i2",
            "initial_graph_id": "g2",
            "mmd": 0.5,
        },
    ]
    treatment = [
        {**control[0], "mmd": 0.2},
        {**control[1], "mmd": 0.4},
    ]
    result = paired_ablation_comparison(control, treatment, metrics=["mmd"])

    assert result["num_pairs"] == 2
    assert result["metrics"]["mmd"]["mean_improvement"] == pytest.approx(0.15)
    assert result["metrics"]["mmd"]["wins"] == 2

    treatment[0] = {**treatment[0], "initial_graph_id": "different"}
    with pytest.raises(ValueError, match="does not share"):
        paired_ablation_comparison(control, treatment, metrics=["mmd"])


def test_quality_cost_pareto_frontier_filters_dominated_settings() -> None:
    result = quality_cost_pareto_summary(
        [
            {"name": "cheap", "mmd": 0.5, "runtime": 1.0},
            {"name": "balanced", "mmd": 0.3, "runtime": 2.0},
            {"name": "dominated", "mmd": 0.6, "runtime": 3.0},
            {"name": "quality", "mmd": 0.2, "runtime": 5.0},
        ],
        quality_keys="mmd",
        cost_keys="runtime",
    )

    assert result["frontier_ids"] == ["cheap", "balanced", "quality"]
    dominated = next(item for item in result["records"] if item["id"] == "dominated")
    assert dominated["is_pareto"] is False
    assert dominated["dominated_by"]

from __future__ import annotations

import networkx as nx
import pytest
import torch

from grapher.hybrid.selector import (
    CANDIDATE_FEATURE_NAMES,
    GRAPH_CONTEXT_FEATURE_NAMES,
    CandidateDiagnostics,
    LearnedCandidateSelector,
    build_selector_features,
    build_teacher_distribution,
    collate_selector_features,
    combine_selector_scores,
    load_selector_checkpoint,
    save_selector_checkpoint,
    select_action,
    select_with_selector,
    selector_distribution_loss,
)
from grapher.refinement.rewiring import enumerate_valid_double_edge_swaps


def _graph_and_actions():
    graph = nx.cubical_graph()
    graph = nx.convert_node_labels_to_integers(graph, ordering="sorted")
    actions = enumerate_valid_double_edge_swaps(
        graph,
        preserve_connectivity=True,
    )
    assert len(actions) >= 3
    return graph, tuple(actions[:3])


def _diagnostics(count: int):
    return [
        {
            "hybrid_score": 0.5 - 0.2 * index,
            "categorical_gain": 0.1 * index,
            "probability_gain": 0.2 * index,
            "graphlet_gain": -0.05 * index,
            "validity_slack": 1.0,
        }
        for index in range(count)
    ]


def _model() -> LearnedCandidateSelector:
    return LearnedCandidateSelector(
        candidate_feature_dim=len(CANDIDATE_FEATURE_NAMES),
        graph_context_dim=len(GRAPH_CONTEXT_FEATURE_NAMES),
        hidden_dim=16,
        dropout=0.0,
    )


def test_feature_builder_uses_action_energy_pair_graphlet_and_context():
    graph, actions = _graph_and_actions()
    diagnostics = _diagnostics(len(actions))
    features = build_selector_features(
        graph,
        actions,
        diagnostics,
        graph_diagnostics={
            "time": 0.25,
            "remaining_step_fraction": 0.75,
            "current_energy": 2.5,
        },
    )

    assert features.actions == actions
    assert features.candidate_features.shape == (
        len(actions),
        len(CANDIDATE_FEATURE_NAMES),
    )
    assert features.graph_context.shape == (len(GRAPH_CONTEXT_FEATURE_NAMES),)
    assert torch.isfinite(features.candidate_features).all()
    assert torch.isfinite(features.graph_context).all()
    assert features.candidate_features[0, 0].item() == pytest.approx(0.5)
    assert features.candidate_features[1, 1].item() == pytest.approx(0.1)
    assert features.candidate_features[1, 2].item() == pytest.approx(0.2)
    assert features.candidate_features[1, 3].item() == pytest.approx(-0.05)
    assert features.graph_context[-3:].tolist() == pytest.approx([0.25, 0.75, 2.5])

    permutation = [2, 0, 1]
    permuted = build_selector_features(
        graph,
        [actions[index] for index in permutation],
        [diagnostics[index] for index in permutation],
    )
    baseline = build_selector_features(graph, actions, diagnostics)
    assert torch.allclose(
        permuted.candidate_features,
        baseline.candidate_features[permutation],
    )


def test_shared_scorer_is_candidate_order_equivariant_and_context_conditioned():
    torch.manual_seed(4)
    model = _model().eval()
    candidate_features = torch.randn(5, len(CANDIDATE_FEATURE_NAMES))
    graph_context = torch.randn(len(GRAPH_CONTEXT_FEATURE_NAMES))
    permutation = torch.tensor([3, 0, 4, 1, 2])

    logits = model(candidate_features, graph_context)
    permuted_logits = model(candidate_features[permutation], graph_context)

    assert logits.shape == (6,)
    assert torch.allclose(permuted_logits[:-1], logits[:-1][permutation])
    assert torch.allclose(permuted_logits[-1], logits[-1])
    changed_context_logits = model(candidate_features, graph_context + 0.5)
    assert not torch.allclose(changed_context_logits, logits)


def test_stop_is_always_available_for_empty_and_fully_masked_candidate_sets():
    model = _model().eval()
    context = torch.zeros(len(GRAPH_CONTEXT_FEATURE_NAMES))
    empty_logits = model(
        torch.empty(0, len(CANDIDATE_FEATURE_NAMES)),
        context,
    )
    assert empty_logits.shape == (1,)
    assert torch.isfinite(empty_logits[0])
    assert selector_distribution_loss(
        empty_logits,
        torch.ones(1),
    ).item() == pytest.approx(0.0)
    decision = select_action([], empty_logits, mode="policy")
    assert decision.stopped
    assert decision.action is None
    assert decision.probabilities.tolist() == pytest.approx([1.0])

    candidates = torch.randn(2, 3, len(CANDIDATE_FEATURE_NAMES))
    contexts = torch.randn(2, len(GRAPH_CONTEXT_FEATURE_NAMES))
    mask = torch.tensor([[True, False, False], [False, False, False]])
    logits = model(candidates, contexts, mask)
    assert torch.isneginf(logits[0, 1:3]).all()
    assert torch.isneginf(logits[1, :3]).all()
    assert torch.isfinite(logits[:, -1]).all()
    teachers = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    loss = selector_distribution_loss(logits, teachers, action_mask=mask)
    assert torch.isfinite(loss)


def test_teacher_distribution_hard_soft_ties_and_stop():
    stop = build_teacher_distribution([-1.0, 0.0])
    assert stop.tolist() == pytest.approx([0.0, 0.0, 1.0])

    hard_tie = build_teacher_distribution([0.5, 0.1, 0.5], temperature=0.0)
    assert hard_tie.tolist() == pytest.approx([0.5, 0.0, 0.5, 0.0])

    soft = build_teacher_distribution([0.5, -1.0, 0.25], temperature=0.2)
    assert soft.sum().item() == pytest.approx(1.0)
    assert soft[0] > soft[2] > 0.0
    assert soft[1].item() == 0.0
    assert soft[-1].item() == 0.0


def test_teacher_cross_entropy_and_kl_are_finite_and_differ_by_entropy():
    logits = torch.tensor([0.3, -0.4, 0.1], requires_grad=True)
    teacher = torch.tensor([0.7, 0.0, 0.3])
    cross_entropy = selector_distribution_loss(
        logits,
        teacher,
        objective="cross_entropy",
    )
    kl = selector_distribution_loss(logits, teacher, objective="kl")
    entropy = -(teacher[teacher > 0] * teacher[teacher > 0].log()).sum()

    assert torch.isfinite(cross_entropy)
    assert torch.isfinite(kl)
    assert torch.allclose(cross_entropy - kl, entropy)
    cross_entropy.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_policy_and_hybrid_dispatch_have_distinct_energy_gating():
    _, actions = _graph_and_actions()
    actions = actions[:2]
    # Policy strongly favors candidate zero even though its energy is negative.
    policy_logits = torch.tensor([5.0, 0.0, -2.0])
    improvements = torch.tensor([-1.0, 0.25])

    policy_scores = combine_selector_scores(
        mode="policy",
        policy_logits=policy_logits,
    )
    policy_decision = select_action(actions, policy_scores, mode="policy")
    assert policy_decision.action == actions[0]

    hybrid_scores = combine_selector_scores(
        mode="hybrid",
        policy_logits=policy_logits,
        energy_improvements=improvements,
        policy_weight=0.0,
        policy_shortlist_size=2,
    )
    assert torch.isneginf(hybrid_scores[0])
    hybrid_decision = select_action(actions, hybrid_scores, mode="hybrid")
    assert hybrid_decision.action == actions[1]

    energy_scores = combine_selector_scores(
        mode="energy",
        energy_improvements=torch.tensor([-1.0, 0.0]),
    )
    assert select_action(actions, energy_scores, mode="energy").stopped


def test_hybrid_shortlist_is_order_equivariant_and_retains_cutoff_ties():
    policy = torch.tensor([2.0, 2.0, 0.5, -1.0])
    energy = torch.tensor([0.1, 0.2, 10.0])
    scores = combine_selector_scores(
        mode="hybrid",
        policy_logits=policy,
        energy_improvements=energy,
        policy_weight=0.0,
        policy_shortlist_size=1,
    )
    # Both tied top-policy candidates survive; the third is excluded despite
    # its large energy improvement.
    assert torch.isfinite(scores[:2]).all()
    assert torch.isneginf(scores[2])

    permutation = torch.tensor([2, 0, 1])
    permuted_scores = combine_selector_scores(
        mode="hybrid",
        policy_logits=torch.cat([policy[:-1][permutation], policy[-1:]]),
        energy_improvements=energy[permutation],
        policy_weight=0.0,
        policy_shortlist_size=1,
    )
    assert torch.equal(
        torch.isfinite(permuted_scores[:-1]), torch.isfinite(scores[:-1])[permutation]
    )
    assert torch.allclose(
        permuted_scores[:-1][torch.isfinite(permuted_scores[:-1])],
        scores[:-1][permutation][torch.isfinite(scores[:-1][permutation])],
    )
    assert permuted_scores[-1] == scores[-1]


def test_variable_candidate_collation_and_high_level_selection():
    graph, actions = _graph_and_actions()
    first = build_selector_features(graph, actions[:2], _diagnostics(2))
    second = build_selector_features(graph, [], [])
    batch = collate_selector_features([first, second])

    assert batch.candidate_features.shape == (
        2,
        2,
        len(CANDIDATE_FEATURE_NAMES),
    )
    assert batch.candidate_mask.tolist() == [[True, True], [False, False]]
    model = _model().eval()
    batch_logits = model(
        batch.candidate_features,
        batch.graph_context,
        batch.candidate_mask,
    )
    assert batch_logits.shape == (2, 3)
    assert torch.isfinite(batch_logits[:, -1]).all()

    decision = select_with_selector(
        model,
        first,
        mode="hybrid",
        policy_weight=0.0,
        policy_shortlist_size=2,
    )
    assert decision.action in first.actions or decision.stopped
    assert decision.probabilities.sum().item() == pytest.approx(1.0)


def test_selector_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(9)
    model = _model().eval()
    candidate_features = torch.randn(4, len(CANDIDATE_FEATURE_NAMES))
    context = torch.randn(len(GRAPH_CONTEXT_FEATURE_NAMES))
    expected = model(candidate_features, context)
    checkpoint_path = tmp_path / "selector.pt"

    save_selector_checkpoint(
        model,
        checkpoint_path,
        config={"mode": "hybrid"},
        report={"validation_loss": 0.25},
    )
    loaded, checkpoint = load_selector_checkpoint(checkpoint_path, device="cpu")
    actual = loaded(candidate_features, context)

    assert checkpoint["format"] == "learned_candidate_selector_v1"
    assert checkpoint["config"]["mode"] == "hybrid"
    assert checkpoint["report"]["validation_loss"] == pytest.approx(0.25)
    assert torch.allclose(actual, expected)


def test_validation_rejects_invalid_actions_distributions_and_modes():
    graph, actions = _graph_and_actions()
    with pytest.raises(ValueError, match="deduplicated"):
        build_selector_features(
            graph,
            [actions[0], actions[0]],
            _diagnostics(2),
        )
    with pytest.raises(KeyError, match="diagnostic"):
        CandidateDiagnostics.from_mapping({"graphlet_gain": 1.0})
    with pytest.raises(ValueError, match="sum to one"):
        selector_distribution_loss(
            torch.zeros(2),
            torch.tensor([0.2, 0.2]),
        )
    with pytest.raises(ValueError, match="requires policy_logits"):
        combine_selector_scores(
            mode="policy",
            energy_improvements=torch.tensor([1.0]),
        )
    with pytest.raises(ValueError, match="mode"):
        combine_selector_scores(
            mode="unknown",
            policy_logits=torch.zeros(2),
        )

from __future__ import annotations

from copy import deepcopy

import networkx as nx
import numpy as np
import pytest

from grapher.data.sampling import restore_training_graphs, sample_training_graphs


def _graphs(count: int) -> list[nx.Graph]:
    graphs = []
    for index in range(count):
        graph = nx.path_graph(2 + index % 4)
        graph.graph["record_id"] = index
        graphs.append(graph)
    return graphs


def _record_ids(graphs) -> list[int]:
    return [graph.graph["record_id"] for graph in graphs]


def test_sampling_is_reproducible_and_seed_changes_the_subset() -> None:
    pool = _graphs(50)
    first, metadata = sample_training_graphs(pool, 12, seed=42)
    repeated, repeated_metadata = sample_training_graphs(pool, 12, seed=42)
    different, different_metadata = sample_training_graphs(pool, 12, seed=43)

    assert _record_ids(first) == _record_ids(repeated)
    assert metadata == repeated_metadata
    assert _record_ids(first) != _record_ids(different)
    assert different_metadata["seed"] == 43
    assert metadata["strategy"] == "random_without_replacement"
    assert metadata["seed"] == 42
    assert metadata["available_graphs"] == 50
    assert metadata["requested_graphs"] == metadata["selected_graphs"] == 12


def test_sampling_keeps_original_order_without_duplicates_or_mutation() -> None:
    pool = _graphs(40)
    original_references = list(pool)
    original_content = deepcopy(pool)
    selected, metadata = sample_training_graphs(pool, 9, seed=12)

    indices = metadata["indices"]
    assert len(indices) == len(set(indices)) == len(selected) == 9
    assert indices == sorted(indices)
    assert all(0 <= index < len(pool) for index in indices)
    assert all(graph is pool[index] for graph, index in zip(selected, indices))
    assert all(actual is original for actual, original in zip(pool, original_references))
    assert all(nx.utils.graphs_equal(actual, original) for actual, original in zip(pool, original_content))


def test_sampling_does_not_consume_or_depend_on_global_numpy_rng() -> None:
    pool = _graphs(30)
    original_state = np.random.get_state()
    try:
        np.random.seed(101)
        expected_next = np.random.random(8)
        np.random.seed(101)
        first, metadata = sample_training_graphs(pool, 7, seed=42)
        actual_next = np.random.random(8)
        np.testing.assert_array_equal(actual_next, expected_next)

        np.random.seed(202)
        second, second_metadata = sample_training_graphs(pool, 7, seed=42)
        assert _record_ids(first) == _record_ids(second)
        assert metadata == second_metadata
    finally:
        np.random.set_state(original_state)


@pytest.mark.parametrize("limit", [None, 0, 8, 30])
def test_unlimited_or_capped_sampling_returns_the_whole_pool(limit) -> None:
    pool = _graphs(8)
    selected, metadata = sample_training_graphs(pool, limit, seed=9)

    assert _record_ids(selected) == list(range(8))
    assert metadata["strategy"] == "all"
    assert metadata["available_graphs"] == metadata["selected_graphs"] == 8
    assert metadata["requested_graphs"] == limit
    assert metadata["indices"] is None
    assert metadata["seed"] == 9


@pytest.mark.parametrize("limit", [None, 0, 4])
def test_empty_pool_is_supported(limit) -> None:
    selected, metadata = sample_training_graphs([], limit, seed=42)
    assert selected == []
    assert metadata["strategy"] == "all"
    assert metadata["available_graphs"] == metadata["selected_graphs"] == 0
    assert metadata["indices"] is None


@pytest.mark.parametrize("limit", [-1, -100])
def test_negative_sample_limit_is_rejected(limit) -> None:
    with pytest.raises(ValueError):
        sample_training_graphs(_graphs(8), limit, seed=42)


def test_restore_uses_saved_indices_despite_later_seed_and_limit_changes() -> None:
    pool = _graphs(30)
    selected, metadata = sample_training_graphs(pool, 8, seed=42)
    saved_metadata = deepcopy(metadata)

    restored = restore_training_graphs(pool, {
        "training_subset": metadata,
        "max_train_graphs": 2,
        "seed": 9999,
    })
    again = restore_training_graphs(pool, {
        "training_subset": metadata,
        "max_train_graphs": 25,
        "seed": 1,
    })

    assert _record_ids(restored) == _record_ids(again) == _record_ids(selected)
    assert all(graph is pool[index] for graph, index in zip(restored, metadata["indices"]))
    assert metadata == saved_metadata


def test_restore_all_uses_saved_selection_instead_of_legacy_prefix() -> None:
    pool = _graphs(8)
    _selected, metadata = sample_training_graphs(pool, None, seed=42)
    restored = restore_training_graphs(pool, {
        "training_subset": metadata,
        "max_train_graphs": 2,
    })
    assert _record_ids(restored) == list(range(8))


@pytest.mark.parametrize("limit, expected", [
    (None, list(range(8))),
    (0, list(range(8))),
    (3, [0, 1, 2]),
    (20, list(range(8))),
])
def test_legacy_checkpoint_without_subset_metadata_keeps_prefix_selection(limit, expected) -> None:
    restored = restore_training_graphs(_graphs(8), {"max_train_graphs": limit, "seed": 42})
    assert _record_ids(restored) == expected


def test_legacy_checkpoint_without_training_limit_restores_all_graphs() -> None:
    assert _record_ids(restore_training_graphs(_graphs(8), {})) == list(range(8))


def test_restore_rejects_a_different_prepared_split_size() -> None:
    _selected, metadata = sample_training_graphs(_graphs(8), 3, seed=42)
    with pytest.raises(ValueError, match="split size"):
        restore_training_graphs(_graphs(9), {"training_subset": metadata})


@pytest.mark.parametrize("indices, selected_count", [
    ([0, 1], 3),
    ([1, 1], 2),
    ([-1, 1], 2),
    ([0, 8], 2),
    ([0, 1.5], 2),
    ([0, "1"], 2),
])
def test_restore_rejects_invalid_saved_indices(indices, selected_count) -> None:
    pool = _graphs(8)
    _selected, metadata = sample_training_graphs(pool, 3, seed=42)
    metadata["indices"] = indices
    metadata["selected_graphs"] = selected_count
    with pytest.raises(ValueError, match="indices"):
        restore_training_graphs(pool, {"training_subset": metadata})


@pytest.mark.parametrize("updates", [
    {"strategy": "unknown"},
    {"strategy": "random_without_replacement", "indices": None},
    {"strategy": "all", "indices": [0, 1, 2]},
    {"strategy": "all", "indices": None, "selected_graphs": 2},
])
def test_restore_rejects_inconsistent_selection_metadata(updates) -> None:
    pool = _graphs(8)
    _selected, metadata = sample_training_graphs(pool, 3, seed=42)
    metadata.update(updates)
    with pytest.raises(ValueError):
        restore_training_graphs(pool, {"training_subset": metadata})

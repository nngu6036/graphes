from __future__ import annotations

import numpy as np
import pytest

from grapher.models.graphrnn.workers.common import (
    GraphSequenceDataset,
    decode_adj,
    encode_adj,
    normalize_config,
)


def test_full_width_adjacency_sequence_round_trip() -> None:
    adjacency = np.asarray(
        [
            [0, 1, 0, 0, 1],
            [1, 0, 1, 0, 0],
            [0, 1, 0, 1, 1],
            [0, 0, 1, 0, 1],
            [1, 0, 1, 1, 0],
        ],
        dtype=np.uint8,
    )
    encoded = encode_adj(adjacency, max_prev_node=4)
    decoded = decode_adj(encoded)
    assert encoded.shape == (4, 4)
    assert np.array_equal(decoded, adjacency)


def test_sequence_dataset_emits_start_token_and_terminal_zero_row() -> None:
    adjacency = np.zeros((1, 5, 5), dtype=np.uint8)
    for left, right in ((0, 1), (1, 2), (2, 3)):
        adjacency[0, left, right] = adjacency[0, right, left] = 1
    dataset = GraphSequenceDataset(
        adjacency,
        np.asarray([4]),
        max_num_node=5,
        max_prev_node=4,
    )
    sample = dataset[0]
    assert sample["x"].shape == (5, 4)
    assert sample["y"].shape == (5, 4)
    assert np.all(sample["x"][0] == 1)
    assert np.all(sample["y"][3:] == 0)
    assert sample["len"] == 4


def test_normalize_config_supports_variant_aliases_and_defaults() -> None:
    config = normalize_config(
        {"variant": "rnn", "max_num_node": 6, "max_prev_node": 5},
        dataset_max_num_node=6,
    )
    assert config["variant"] == "GraphRNN_RNN"
    assert config["epochs"] == 3000
    assert config["batch_ratio"] == 32
    assert config["scheduler_step_unit"] == "batch"


def test_normalize_config_rejects_dataset_width_mismatch() -> None:
    with pytest.raises(ValueError, match="disagree"):
        normalize_config(
            {"max_num_node": 7, "max_prev_node": 5},
            dataset_max_num_node=6,
        )


def test_normalize_config_rejects_lookback_wider_than_strict_lower_triangle() -> None:
    with pytest.raises(ValueError, match="max_num_node - 1"):
        normalize_config(
            {"max_num_node": 6, "max_prev_node": 6},
            dataset_max_num_node=6,
        )

from __future__ import annotations

import pytest

from grapher.utils.io import apply_config_overrides


def test_apply_config_overrides_supports_nested_yaml_values() -> None:
    config = {
        "topology_refiner": {"steps": 24},
        "generation": {"num_generate": 1024},
    }

    apply_config_overrides(
        config,
        [
            "topology_refiner.steps=40",
            "topology_refiner.prediction_horizon.mode=annealed",
            "topology_refiner.prediction_horizon.initial_k=8",
            "topology_refiner.prediction_horizon.final_k=1",
            "topology_refiner.prediction_horizon.refresh_on_plateau=true",
            "evaluation.orca_exec=null",
            "protocol.evaluation_seeds=[42, 43, 44]",
        ],
    )

    assert config["topology_refiner"]["steps"] == 40
    horizon = config["topology_refiner"]["prediction_horizon"]
    assert horizon == {
        "mode": "annealed",
        "initial_k": 8,
        "final_k": 1,
        "refresh_on_plateau": True,
    }
    assert config["evaluation"]["orca_exec"] is None
    assert config["protocol"]["evaluation_seeds"] == [42, 43, 44]


def test_apply_config_overrides_rejects_invalid_paths() -> None:
    with pytest.raises(ValueError, match="KEY=VALUE"):
        apply_config_overrides({}, ["topology_refiner.steps"])

    with pytest.raises(ValueError, match="not a mapping"):
        apply_config_overrides(
            {"topology_refiner": 3},
            ["topology_refiner.steps=24"],
        )

from __future__ import annotations

import re
from pathlib import Path

import yaml

from scripts.prepare_generic_dataset import load_requested_dataset_config

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_every_readme_config_path_exists() -> None:
    readme = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
    referenced = sorted(set(re.findall(r"configs/[A-Za-z0-9_./-]+\.yaml", readme)))

    missing = [path for path in referenced if not (REPOSITORY_ROOT / path).is_file()]
    assert referenced
    assert missing == []


def test_every_experiment_dataset_config_path_exists() -> None:
    missing: list[str] = []
    for experiment_path in sorted(
        (REPOSITORY_ROOT / "configs" / "experiments").rglob("*.yaml")
    ):
        config = yaml.safe_load(experiment_path.read_text(encoding="utf-8")) or {}
        config_path = (config.get("dataset") or {}).get("config_path")
        if config_path and not (REPOSITORY_ROOT / str(config_path)).is_file():
            missing.append(
                f"{experiment_path.relative_to(REPOSITORY_ROOT)} -> {config_path}"
            )

    assert missing == []


def test_community_config_preserves_historical_dataset_alias(monkeypatch) -> None:
    monkeypatch.chdir(REPOSITORY_ROOT)
    config, config_path, dataset_name = load_requested_dataset_config("community_small")

    assert config_path == Path("configs/datasets/community_small.yaml")
    assert config["name"] == "sbm"
    assert dataset_name == "sbm"


def test_community_config_matches_the_declared_gdss_protocol() -> None:
    config = yaml.safe_load(
        (REPOSITORY_ROOT / "configs/datasets/community_small.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert config["num_graphs"] == 100
    assert config["split"] == {"train": 0.7, "val": 0.1, "test": 0.2}
    assert config["communities"]["min_total_nodes"] == 12
    assert config["communities"]["max_total_nodes"] == 20
    assert config["communities"]["equal_block_sizes"] is True
    assert config["edge_probs"]["p_inter"] == 0.05


def test_generic_grapher_configs_exist_for_every_readme_dataset() -> None:
    config_dir = REPOSITORY_ROOT / "configs" / "experiments" / "grapher"
    expected = {
        "ego_small_topology_graphlet.yaml",
        "community_small_topology_graphlet.yaml",
        "grid_topology_graphlet.yaml",
    }
    assert expected <= {path.name for path in config_dir.glob("*.yaml")}


def test_generic_grapher_configs_are_decoupled_topology_configs() -> None:
    config_dir = REPOSITORY_ROOT / "configs" / "experiments" / "grapher"
    for path in sorted(config_dir.glob("*_topology_graphlet.yaml")):
        config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

        assert config["pipeline"]["stage"] == "topology"
        assert "categorical_state" not in config
        assert "endpoint_predictor" not in config
        assert "endpoint_trajectory" not in config
        assert "hybrid_refiner" not in config
        assert set(config["topology_predictor"]["loss_weights"]) <= {
            "graphlet_mean",
            "graphlet_distribution",
            "graphlet_mass",
        }
        assert config["topology_refiner"]["mode"] == "energy"
        assert config["topology_refiner"]["refresh_prediction_every"] == 1
        assert config["topology_refiner"]["preserve_connectivity"] is True
        assert config["topology_trajectory"]["storage"] == "eager"
        assert config["topology_trajectory"]["preserve_connectivity"] is True
        assert config["topology_trajectory"]["ensure_connected_source"] is True
        assert config["graphlet_prediction"]["graphlet_num_samples"] is None
        assert (
            config["graphlet_prediction"]["estimator"]
            == "exact_connected_local_delta"
        )
        assert config["generation"]["degree_source"] == "learned"
        assert config["generation"]["max_attempts_per_graph"] > 0
        assert config["generation"]["write_legacy_hybrid_alias"] is False
        assert config["degree_generator"]["type"] == "degree_histogram_vae"
        assert config["degree_generator"]["postprocess_policy"] == "reject_only"
        assert config["degree_generator"]["fallback"] == "error"
        assert config["constructor"]["ensure_connected"] is True

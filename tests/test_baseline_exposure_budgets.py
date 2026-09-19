from importlib import import_module
from pathlib import Path

import pytest
import yaml

from grapher.models import DatasetReference, RunSpec, TrainRequest, create_baseline
from grapher.models.comparison import (
    comparison_request_options,
    resolve_common_config,
    resolve_sampled_exposure_budget,
    resolve_wrapper_config,
)


ROOT = Path(__file__).resolve().parents[1]
TARGETS = dict(community_small=10000, ego_small=10000, grid=10000, qm9=200, zinc=200)
MODELS = ("catflow", "digress", "defog", "gdsm", "gdsm_simple", "gdss", "edge", "spectre")
CASES = [
    (model, dataset) for model in MODELS for dataset in TARGETS
    if not (model in {"gdsm", "edge"} and dataset in {"qm9", "zinc"})
    and not (model == "defog" and dataset == "grid")
    and not (model == "spectre" and dataset == "zinc")
]


@pytest.mark.parametrize("model,dataset", CASES)
def test_effective_default_epoch_budgets(model, dataset):
    config = resolve_wrapper_config(model, dataset, None)
    common = resolve_common_config(dataset, None)
    request = TrainRequest(
        run=RunSpec.for_seed(model_id=model, dataset_id=dataset, seed=42),
        dataset=DatasetReference(dataset),
        config_path=config,
        options=comparison_request_options(model, common, model_config=config),
    )
    if model in {"digress", "defog", "gdss"}:
        options = import_module(f"grapher.models.{model}.wrapper")._load_options(request)
    else:
        options = create_baseline(model)._options(request)
    if model in {"digress", "defog"}:
        epochs = options["n_epochs"]
    else:
        epochs = options["train"]["num_epochs" if model == "gdss" else "epochs"]
    assert epochs == TARGETS[dataset]


@pytest.mark.parametrize("dataset,count", [("community_small", 64), ("ego_small", 128), ("grid", 70)])
def test_graphrnn_exposure_uses_actual_split_and_batch_overrides(dataset, count):
    config = resolve_wrapper_config("graphrnn", dataset, None)
    for actual_count in (count, count + 17):
        for batch in (16, 32):
            options = resolve_sampled_exposure_budget(
                "graphrnn", config, {"batch_size": batch}, num_train_graphs=actual_count,
            )
            record = options["comparison_reference"]["resolved_exposure_budget"]
            assert abs(record["estimated_graph_exposures"] - TARGETS[dataset] * actual_count) <= batch * 32 / 2
            assert not record["explicit_horizon_override"]
            assert options["milestones"]


@pytest.mark.parametrize("dataset,count", [("community_small", 64), ("ego_small", 128), ("qm9", 104665), ("zinc", 200123)])
def test_hog_diff_combines_stage_exposure(dataset, count):
    config = resolve_wrapper_config("hog_diff", dataset, None)
    options = resolve_sampled_exposure_budget(
        "hog_diff", config, {"ou": {"batch_size": 17}}, num_train_graphs=count,
    )
    record = options["comparison_reference"]["resolved_exposure_budget"]
    rounding_bound = sum(stage["batch_size"] / 2 for stage in record["resolved"].values())
    assert abs(record["estimated_graph_exposures"] - TARGETS[dataset] * count) <= rounding_bound
    assert not record["explicit_horizon_override"]
    assert options["ou"]["batch_size"] == 17


@pytest.mark.parametrize("model", ["hog_diff", "graphrnn"])
def test_sampled_wrapper_resolves_from_prepared_training_file(model, tmp_path):
    import pickle

    dataset = DatasetReference("community_small", root=tmp_path)
    dataset.dataset_dir.mkdir()
    with dataset.split_paths["train"].open("wb") as handle:
        pickle.dump([None] * 64, handle)
    config = resolve_wrapper_config(model, "community_small", None)
    request = TrainRequest(
        run=RunSpec.for_seed(model_id=model, dataset_id="community_small", seed=42),
        dataset=dataset, config_path=config,
    )
    options = import_module(f"grapher.models.{model}.wrapper")._load_options(request)
    record = options["comparison_reference"]["resolved_exposure_budget"]
    assert record["num_train_graphs"] == 64
    assert record["target_graph_exposures"] == TARGETS["community_small"] * 64


def test_explicit_horizons_remain_authoritative():
    for model, overrides in [
        ("graphrnn", {"epochs": 7}),
        ("hog_diff", {"higher_order": {"n_iters": 12}, "ou": {"n_iters": 34}}),
    ]:
        config = resolve_wrapper_config(model, "community_small", None)
        options = resolve_sampled_exposure_budget(model, config, overrides, num_train_graphs=64)
        for key, value in overrides.items():
            assert options[key] == value
        assert options["comparison_reference"]["resolved_exposure_budget"]["explicit_horizon_override"]


def test_custom_config_does_not_require_prepared_split(tmp_path):
    config = tmp_path / "custom.yaml"
    config.write_text("graphrnn:\n  epochs: 123\n")
    assert resolve_sampled_exposure_budget("graphrnn", config, {"epochs": 99}) == {"epochs": 99}


def test_gdsm_decay_preserves_original_terminal_rate():
    config = yaml.safe_load((ROOT / "configs/baselines/gdsm_community_small.yaml").read_text())
    train = config["gdsm"]["train"]
    assert train["lr_decay"] ** train["epochs"] == pytest.approx(0.999 ** 200)


def test_dhvae_default_horizons():
    for dataset, target in TARGETS.items():
        suffix = "_typed" if dataset in {"qm9", "zinc"} else ""
        config = yaml.safe_load((ROOT / f"configs/experiments/dhvae/{dataset}{suffix}.yaml").read_text())
        assert config["degree_generator"]["epochs"] == target

from __future__ import annotations

import json
from pathlib import Path

import pytest

from grapher.models.base import RunSpec
from grapher.models.external_cli import (
    _MODEL_DATASETS,
    _generation_options,
    _managed_checkpoint,
    _training_options,
    build_parser,
)
from grapher.models.comparison import resolve_common_config, resolve_wrapper_config


MODELS = (
    "catflow",
    "defog",
    "dhvae_hh",
    "digress",
    "edge",
    "gdsm",
    "gdss",
    "graphrnn",
    "hog_diff",
    "spectre",
)


def _resolved(model: str, argv: list[str]):
    args = build_parser(model).parse_args(argv)
    if model == "dhvae_hh":
        common = None
        config = Path(args.wrapper_config).resolve() if args.wrapper_config else None
    else:
        common = resolve_common_config(
            args.dataset, args.common_config, disabled=args.no_common_config
        )
        config = resolve_wrapper_config(model, args.dataset, args.wrapper_config)
    options = _training_options(
        model,
        args,
        common=common,
        wrapper_config=config,
        profile=_MODEL_DATASETS[model][args.dataset],
    )
    return args, options


def test_all_model_runners_are_thin_shared_cli_shims() -> None:
    repo = Path(__file__).resolve().parents[1]
    for model in MODELS:
        path = repo / "scripts" / f"run_{model}_baseline.py"
        text = path.read_text(encoding="utf-8")
        assert "from grapher.models.external_cli import main" in text
        assert f'main("{model}")' in text
        assert len(text.splitlines()) <= 8
    alias = (repo / "scripts" / "run_gsdm_baseline.py").read_text(encoding="utf-8")
    assert "from grapher.models.external_cli import main" in alias
    assert 'main("gsdm")' in alias
    assert len(alias.splitlines()) <= 8


def test_shared_stage_and_common_arguments_exist_for_every_external_model() -> None:
    for model in MODELS:
        parser = build_parser(model)
        args = parser.parse_args(["--dataset", next(iter(_MODEL_DATASETS[model]))])
        assert args.stage == "all"
        assert args.seed_id == 42
        assert args.num_samples == 1024
        assert args.output_root == Path("outputs/baselines")


def test_legacy_source_and_python_aliases_are_preserved() -> None:
    cases = {
        "defog": ("--defog-root", "--defog-python"),
        "digress": ("--digress-root", "--digress-python"),
        "gdss": ("--gdss-root", "--gdss-python"),
        "graphrnn": ("--graphrnn-root", "--graphrnn-python"),
        "hog_diff": ("--hogdiff-root", "--hogdiff-python"),
        "gdsm": ("--gsdm-root", "--gsdm-python"),
    }
    for model, (root_flag, python_flag) in cases.items():
        dataset = next(iter(_MODEL_DATASETS[model]))
        args = build_parser(model).parse_args(
            [
                "--dataset",
                dataset,
                root_flag,
                "/tmp/source",
                python_flag,
                "/tmp/python",
            ]
        )
        assert args.source_root == Path("/tmp/source")
        assert args.python == Path("/tmp/python")


def test_defog_gpu_and_epoch_compatibility_translation() -> None:
    _args, options = _resolved(
        "defog",
        [
            "--dataset",
            "community_small",
            "--n-epochs",
            "123",
            "--device",
            "gpu",
            "--gpu-id",
            "2",
        ],
    )
    assert options["n_epochs"] == 123
    assert options["runtime"]["gpus"] == 1
    assert options["runtime"]["device"] == "cuda"
    assert options["runtime"]["cuda_visible_devices"] == "2"
    assert options["runtime"]["require_cuda"] is True


def test_digress_legacy_training_controls_translate_to_wrapper_options() -> None:
    _args, options = _resolved(
        "digress",
        [
            "--dataset",
            "community_small",
            "--n-epochs",
            "123",
            "--batch-size",
            "17",
            "--num-workers",
            "3",
            "--check-val-every-n-epochs",
            "9",
            "--save-every-n-epochs",
            "11",
        ],
    )
    assert options["experiment"] == "comm20"
    assert options["n_epochs"] == 123
    assert options["batch_size"] == 17
    assert options["num_workers"] == 3
    assert options["check_val_every_n_epochs"] == 9
    assert options["save_every_n_epochs"] == 11


def test_gdss_epoch_override_stays_nested_in_train_section() -> None:
    _args, options = _resolved(
        "gdss", ["--dataset", "community_small", "--num-epochs", "123"]
    )
    assert options["train"] == {"num_epochs": 123}


def test_graphrnn_controls_and_generation_sample_time_are_preserved() -> None:
    args, options = _resolved(
        "graphrnn",
        [
            "--dataset",
            "community_small",
            "--epochs",
            "123",
            "--batch-ratio",
            "7",
            "--max-prev-node",
            "19",
            "--sample-time",
            "2",
        ],
    )
    assert options["epochs"] == 123
    assert options["batch_ratio"] == 7
    assert options["max_prev_node"] == 19
    assert options["sample_time"] == 2
    assert _generation_options("graphrnn", args)["sample_time"] == 2


def test_hog_diff_keeps_separate_stage_budgets() -> None:
    _args, options = _resolved(
        "hog_diff",
        [
            "--dataset",
            "community_small",
            "--ho-iters",
            "12",
            "--ou-iters",
            "34",
            "--ho-batch-size",
            "5",
            "--ou-batch-size",
            "6",
        ],
    )
    assert options["higher_order"] == {"n_iters": 12, "batch_size": 5}
    assert options["ou"] == {"n_iters": 34, "batch_size": 6}


def test_hog_diff_rejects_ambiguous_single_epoch_override() -> None:
    parser = build_parser("hog_diff")
    args = parser.parse_args(
        ["--dataset", "community_small", "--epochs", "100"]
    )
    common = resolve_common_config("community_small", None, disabled=False)
    config = resolve_wrapper_config("hog_diff", "community_small", None)
    with pytest.raises(ValueError, match="two native training horizons"):
        _training_options(
            "hog_diff",
            args,
            common=common,
            wrapper_config=config,
            profile=_MODEL_DATASETS["hog_diff"]["community_small"],
        )


def test_dhvae_experiment_config_alias_and_estimate_count() -> None:
    _args, options = _resolved(
        "dhvae_hh",
        [
            "--dataset",
            "community_small",
            "--experiment-config",
            "configs/experiments/dhvae/community_small.yaml",
            "--training-estimate-count",
            "8",
        ],
    )
    assert options["training_estimates"] == {"enabled": True, "num_graphs": 8}


def test_source_backed_cli_overrides_model_train_section_not_common_budget() -> None:
    _args, options = _resolved(
        "catflow",
        ["--dataset", "community_small", "--n-epochs", "123", "--batch-size", "17"],
    )
    assert options["train"] == {"epochs": 123, "batch_size": 17}
    assert options["comparison_defaults"]["train"]["epochs"] == 10_000


def test_managed_checkpoint_is_resolved_relative_to_train_dir(tmp_path: Path) -> None:
    run = RunSpec.for_seed(
        model_id="gdss",
        dataset_id="community_small",
        seed=42,
        output_root=tmp_path,
    )
    run.layout.train_dir.mkdir(parents=True)
    run.layout.checkpoints_dir.mkdir(parents=True)
    checkpoint = run.layout.checkpoints_dir / "gdss.pth"
    checkpoint.write_bytes(b"checkpoint")
    run.layout.training_manifest_path.write_text(
        json.dumps({"checkpoint": {"path": "checkpoints/gdss.pth"}}),
        encoding="utf-8",
    )
    assert _managed_checkpoint(run) == checkpoint

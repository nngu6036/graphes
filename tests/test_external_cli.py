from pathlib import Path

from grapher.models.registry import create_baseline, normalize_baseline_id
from grapher.models.grapher.wrapper import GraphERWrapper


def test_grapher_runner_is_shared_cli_shim():
    path = Path("scripts/run_grapher_baseline.py")
    text = path.read_text(encoding="utf-8")
    assert 'main("grapher")' in text
    assert "external_cli" in text


def test_grapher_registry_identity():
    assert normalize_baseline_id("grapher") == "grapher"
    assert normalize_baseline_id("graph-er") == "grapher"
    wrapper = create_baseline("grapher")
    assert isinstance(wrapper, GraphERWrapper)
    assert wrapper.model_id == "grapher"


def test_final_configs_and_commands_cover_five_seeds():
    for dataset in ("community_small", "ego_small", "qm9", "zinc"):
        for seed in range(41, 46):
            assert Path(f"configs/experiments/grapher_final/{dataset}_seed_{seed}.yaml").is_file()
        command = Path(f"commands/{dataset}.sh").read_text(encoding="utf-8")
        assert "SEEDS=(41 42 43 44 45)" in command
        assert "run_grapher_baseline.py" in command

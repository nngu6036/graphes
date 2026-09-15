from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import yaml


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_guidance_factorial_ablation.py"
spec = spec_from_file_location("guidance_factorial", SCRIPT)
module = module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def test_factorial_profile_is_clean_two_component_ablation():
    root = Path(__file__).resolve().parents[1]
    profile = yaml.safe_load(
        (root / "configs/experiments/grapher/ablations/community_small_guidance_factorial.yaml").read_text()
    )
    modes = profile["modes"]
    assert set(modes) == {"spectral", "graphlet", "spectral_graphlet"}
    assert modes["spectral"]["weights"] == {
        "edge": 0.0, "spectral": 1.0, "clustering": 0.0, "orbit": 0.0, "graphlet": 0.0
    }
    assert modes["graphlet"]["weights"] == {
        "edge": 0.0, "spectral": 0.0, "clustering": 0.0, "orbit": 0.0, "graphlet": 1.0
    }
    assert modes["spectral_graphlet"]["weights"] == {
        "edge": 0.0, "spectral": 1.0, "clustering": 0.0, "orbit": 0.0, "graphlet": 1.0
    }


def test_mode_overrides_disable_nonfactorial_energies():
    profile = {
        "protocol": {
            "steps": 32,
            "proposal_budget": 1024,
            "valid_candidate_budget": 256,
            "component_normalization": "initial",
        }
    }
    mode = {
        "weights": {
            "edge": 0.0,
            "spectral": 1.0,
            "clustering": 0.0,
            "orbit": 0.0,
            "graphlet": 1.0,
        }
    }
    overrides = module._mode_overrides(profile, mode)
    assert "topology_refiner.weights.edge=0.0" in overrides
    assert "topology_refiner.weights.clustering=0.0" in overrides
    assert "topology_refiner.weights.orbit=0.0" in overrides
    assert "topology_refiner.weights.spectral=1.0" in overrides
    assert "topology_refiner.weights.graphlet=1.0" in overrides
